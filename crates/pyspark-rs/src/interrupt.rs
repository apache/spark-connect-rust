//! Waits on the server that a signal of the process interrupts, as PySpark's waits are, and the
//! checkpoints of the conversions that run in native code ([`ConversionBudget`]).
//!
//! The wait runs on the calling thread, as PySpark's does, so the progress handlers and the
//! other callbacks of the execution run there too. The wait polls the signals of the process,
//! which only the main thread of Python runs the handlers of. A handler that raises a
//! KeyboardInterrupt, as the default handler of SIGINT does, interrupts the execution in flight of
//! the calling thread on the server; the wait then ends with the server's cancellation, and the
//! exception of the handler is raised. The exception of another handler ends the wait at once, as
//! PySpark's execution interrupts the operation of a KeyboardInterrupt only, unless the part of
//! the call that waits catches it, as PySpark's lookup of the plan compression settings catches an
//! `Exception` (see `spark_connect_core::runtime::catching`).

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use pyo3::types::PyDict;
use spark_connect::session::{KeptThreadRequestContext, SparkSession};
use spark_connect_core::client::SparkConnectClient;
use spark_connect_core::error::SparkError;
use spark_connect_core::runtime::{with_wait_watch, WaitWatch, Watch};

use crate::errors::ResultExt;

/// The steps of a conversion from one of its checkpoints to the next.
const CONVERSION_STEPS_PER_CHECKPOINT: usize = 1024;

/// A Python function that does nothing, whose call is the checkpoint of a conversion.
static CONVERSION_CHECKPOINT: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// The checkpoints of a conversion between Python values and native ones that runs in native
/// code, with the GIL and without a Python frame of its own, where PySpark's conversions run
/// Python code: the Rows of a result, and the local data of `createDataFrame`. A step is a value,
/// or a row, of the conversion. Every 1,024 steps, the first included, the call of a Python
/// function lets the interpreter run its pending work, such as handing the GIL to another thread
/// that waits for it, and the signals of the process are checked, so that the exception of a
/// handler, as the KeyboardInterrupt of Ctrl-C, ends the conversion. The budget is of steps, not
/// of time: the work of a step that is long, as a callback or the allocation of a large value,
/// still holds the GIL.
///
/// The count is atomic only so that what holds a budget is `Sync`, as a pyclass must be; it is
/// counted under the GIL, by one thread at a time.
#[derive(Default)]
pub(crate) struct ConversionBudget {
    steps_until_checkpoint: AtomicUsize,
}

impl ConversionBudget {
    /// A budget whose first checkpoint is after a full budget of steps, as for an operation that
    /// is most often short, of which the many calls of a Python loop would each take one: the
    /// operations of the messages of `pyspark.sql.connect.proto`, such as the parse, the copy and
    /// the merge of a message, a step for each message.
    pub(crate) fn full() -> Self {
        ConversionBudget {
            steps_until_checkpoint: AtomicUsize::new(CONVERSION_STEPS_PER_CHECKPOINT),
        }
    }

    /// One step of the conversion, which is a checkpoint when the budget is spent.
    #[inline]
    pub(crate) fn step(&self, py: Python<'_>) -> PyResult<()> {
        let steps = self.steps_until_checkpoint.load(Ordering::Relaxed);
        if steps > 0 {
            self.steps_until_checkpoint
                .store(steps - 1, Ordering::Relaxed);
            return Ok(());
        }
        self.checkpoint(py)
    }

    #[cold]
    fn checkpoint(&self, py: Python<'_>) -> PyResult<()> {
        let checkpoint = CONVERSION_CHECKPOINT.get_or_try_init(py, || -> PyResult<_> {
            let globals = PyDict::new(py);
            Ok(py.eval(c"lambda: None", Some(&globals), None)?.unbind())
        })?;
        checkpoint.bind(py).call0()?;
        crate::deliver::check_signals(py)?;
        self.steps_until_checkpoint
            .store(CONVERSION_STEPS_PER_CHECKPOINT - 1, Ordering::Relaxed);
        Ok(())
    }

    /// Make the next step a checkpoint, as the first step of a new conversion is.
    pub(crate) fn reset(&self) {
        self.steps_until_checkpoint.store(0, Ordering::Relaxed);
    }

    /// Release `values`, which hold Python references, a step for each: their release holds the
    /// GIL too, as long as their conversion took to take them. A signal that ends the release
    /// releases the rest of them at once.
    pub(crate) fn release<T>(&self, py: Python<'_>, values: Vec<T>) -> PyResult<()> {
        for value in values {
            self.step(py)?;
            drop(value);
        }
        Ok(())
    }
}

/// The poll of the signals of one wait.
struct SignalWatch {
    session: SparkSession,
    /// Whether the execution that the wait is for has been interrupted for the signal: after
    /// that, the wait only waits.
    interrupted: AtomicBool,
    /// The exception of the first handler that raised, which the wait raises once it has ended.
    error: Mutex<Option<PyErr>>,
    /// The error of the interrupt of the execution, if the server fails it. The wait raises it
    /// once it has ended, in place of the KeyboardInterrupt of the signal, as PySpark's
    /// `interrupt_operation` raises out of the `except KeyboardInterrupt` of an execution.
    interrupt_failure: Arc<Mutex<Option<SparkError>>>,
    /// The error of the build of the request of the interrupt, of a value of the attributes of
    /// the client that protobuf refuses or of a read of them that raised, which the wait raises in
    /// place of the KeyboardInterrupt, as the error of `interrupt_failure`, and with which no
    /// interrupt is sent.
    interrupt_unbuilt: Mutex<Option<PyErr>>,
    /// The failure of the call whose conversion ran when the first signal came (see
    /// `spark_connect_core::client::converting_failure`), whose grpcio's error the exception of
    /// the signal has as its context, of the operation ID of its execution, and which interrupts
    /// nothing.
    converting: Mutex<Option<spark_connect_core::client::Conversion>>,
}

impl SignalWatch {
    /// Whether the server has failed the interrupt of the execution.
    fn interrupt_failure_is_set(&self) -> bool {
        self.interrupt_failure
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
    }

    fn poll(&self) -> Watch {
        // The wait whose interrupt could not be built ends at once, as PySpark raises the error of
        // its build.
        if self
            .interrupt_unbuilt
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .is_some()
        {
            return Watch::Abort;
        }
        {
            let mut error = self.error.lock().unwrap_or_else(|e| e.into_inner());
            if error.is_none() {
                // Takes the GIL for the check only, as the wait runs without it. The exception of a
                // handler that is an `Exception` ends the part of the call that PySpark's `except
                // Exception` covers, if one runs, such as the lookup of the plan compression
                // settings, and the call goes on, as PySpark's does (see `runtime::catching`).
                // An exception that a read of the attributes of the client of the call that waits
                // raised meanwhile, that is not an `Exception`, as the `KeyboardInterrupt` of a
                // signal handler that ran while a finalizer read them, acts as the exception of a
                // signal handler that this poll caught (see `keep_for_nearest_call`). A wait that
                // the interpreter cannot take the GIL for any more, as of a daemon thread at the
                // exit, ends.
                let Some(raised) = Python::try_attach(|py| {
                    let raised = match take_raised_of_waiting_call(&self.session) {
                        Some(kept) => Err(kept),
                        None => crate::deliver::check_signals(py),
                    };
                    match raised {
                        Ok(()) => None,
                        Err(signal)
                            if signal.is_instance_of::<pyo3::exceptions::PyException>(py)
                                && spark_connect_core::runtime::catch_signal() =>
                        {
                            None
                        }
                        Err(signal) => {
                            let keyboard_interrupt =
                                signal.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py);
                            Some((signal, keyboard_interrupt))
                        }
                    }
                }) else {
                    return Watch::Abort;
                };
                match raised {
                    None => return Watch::Continue,
                    Some((signal, keyboard_interrupt)) => {
                        *error = Some(signal);
                        // As PySpark's `_handle_error`, which converts the failure of a call in
                        // its `except Exception` clause: a signal while the failure is converted,
                        // as during the request of the details of an error of the server or the
                        // read of the configs of its stack trace, ends the wait at once, of no
                        // interrupt, and has the failure as its context.
                        if let Some(conversion) = spark_connect_core::client::converting_failure() {
                            *self.converting.lock().unwrap_or_else(|e| e.into_inner()) =
                                Some(conversion);
                            return Watch::Abort;
                        }
                        // As PySpark's execution, whose `except KeyboardInterrupt` interrupts the
                        // operation on the server: the exception of another handler ends the
                        // wait at once, as it goes out of the execution as it is.
                        if !keyboard_interrupt {
                            return Watch::Abort;
                        }
                    }
                }
            } else if self
                .converting
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .is_some()
            {
                // The signal came while the failure of the call was converted: the wait goes on
                // to its end, and interrupts nothing.
                return Watch::Abort;
            } else if self.interrupted.load(Ordering::SeqCst) {
                // A failed interrupt ends the wait at once, as PySpark raises the error of the
                // interrupt without waiting for the server to end the execution.
                if self.interrupt_failure_is_set() {
                    return Watch::Abort;
                }
                // After the interrupt, a second signal ends the wait, as PySpark's wait does, and
                // so does an interpreter that the GIL cannot be taken of any more.
                return match Python::try_attach(crate::deliver::check_signals) {
                    Some(Ok(())) => Watch::Continue,
                    Some(Err(_)) | None => Watch::Abort,
                };
            }
        }
        match self.session.running_operation_id() {
            Some(operation_id) => {
                // The request of the interrupt is of the attributes of the client of the call as
                // they are now, as PySpark's `interrupt_operation` builds it once the
                // KeyboardInterrupt is raised: an error of its build replaces the
                // KeyboardInterrupt, and no interrupt is sent (see `reread_nearest_call`).
                let Some(unbuilt) = Python::try_attach(|py| {
                    reread_nearest_call_of(py, &self.session, ReadOf::Requests).err()
                }) else {
                    return Watch::Abort;
                };
                if let Some(unbuilt) = unbuilt {
                    *self
                        .interrupt_unbuilt
                        .lock()
                        .unwrap_or_else(|e| e.into_inner()) = Some(unbuilt);
                    return Watch::Abort;
                }
                // As PySpark's `except KeyboardInterrupt`, which logs the interrupt before it
                // interrupts the operation: the exception of the record replaces the
                // KeyboardInterrupt, which becomes its context, and nothing is interrupted.
                if let Err(raised) = crate::client_log::interrupted(&operation_id) {
                    // Of an interpreter that the GIL cannot be taken of any more, as of a daemon
                    // thread at the exit, the wait ends too.
                    let _ = Python::try_attach(|py| {
                        let raised = crate::errors::spark_error_to_py_exception(raised);
                        let mut error = self.error.lock().unwrap_or_else(|e| e.into_inner());
                        *error = Some(match error.take() {
                            Some(signal) => crate::errors::with_context(py, raised, signal),
                            None => raised,
                        });
                    });
                    return Watch::Abort;
                }
                self.interrupted.store(true, Ordering::SeqCst);
                let session = self.session.clone();
                let failure = Arc::clone(&self.interrupt_failure);
                Watch::Interrupt(Box::pin(async move {
                    if let Err(error) = session.interrupt_operation_async(&operation_id).await {
                        *failure.lock().unwrap_or_else(|e| e.into_inner()) = Some(error);
                    }
                }))
            }
            // No execution to interrupt: a wait that can end before its future, such as the
            // release of `stop()`, ends now; another keeps waiting.
            None => Watch::Abort,
        }
    }
}

/// Run `f` without the GIL, on the calling thread, as the signals of the process are polled while
/// it waits for the server. When a handler raises a KeyboardInterrupt, the execution that the
/// calling thread waits for is interrupted, and the exception of the handler is raised once `f`
/// has ended; the exception of another handler ends the waits of `f` at once, and is raised.
/// Otherwise the result of `f` is returned.
///
/// The execution is interrupted only when a wait of the calling thread is for one. The waits
/// that a signal ends at once, as PySpark's do (a retry sleep, or a call that the server does not
/// answer): the retries and the RPCs of `f` that are not answered when the first signal arrives
/// during an execution, and all of them when no execution is in flight, or after a second
/// signal. A streaming execution is still read after the first one, until the server cancels it.
/// A wait of `block_on_abortable` ends at once as well: the release of `stop()`, the wait of a
/// `stop()` for that of another thread, and the waits of the listener bus for the deliveries and
/// the lifecycle operations of other threads, as a signal interrupts the acquire of PySpark's
/// locks.
///
/// For Python's garbage collector, the call is counted as one that may clone and drop any part of
/// plans and expressions (see [`crate::gc::detach`]), unless its caller set how it is counted
/// first, as [`interruptible_of`], [`interruptible_of_values`] and
/// [`interruptible_without_parts`] do, and `crate::gc::in_use_of_values` before it.
pub(crate) fn interruptible<T: Send, E: Send + Into<Box<SparkError>>>(
    py: Python<'_>,
    session: &SparkSession,
    f: impl FnOnce() -> Result<T, E> + Send,
) -> PyResult<T> {
    of_the_request(session, interruptible_outcome(py, session, f)?)
}

/// As [`interruptible`], for a call that clones no part of plans and expressions, nor drops one
/// that a holder reports, without the GIL: it is not counted for the garbage collector, which
/// sees the parts while it runs (see [`crate::gc::detach_without_parts`]). Of the calls of
/// streaming queries and their manager, the listener bus, the configuration, the catalog and
/// the session, whose requests are of identifiers, protobufs and values only.
pub(crate) fn interruptible_without_parts<T: Send, E: Send + Into<Box<SparkError>>>(
    py: Python<'_>,
    session: &SparkSession,
    f: impl FnOnce() -> Result<T, E> + Send,
) -> PyResult<T> {
    crate::gc::without_parts(|| interruptible(py, session, f))
}

/// As [`interruptible`], for a call that clones and drops only the parts that `holders` hold,
/// directly or below, without the GIL, as a request of a DataFrame builds its plan: those parts
/// are busy while it runs, which their [`crate::gc::SharedPart`]s report nothing of, and the call
/// is not counted as one that may clone any part (see [`crate::gc::of_parts`]).
pub(crate) fn interruptible_of<T: Send, E: Send + Into<Box<SparkError>>>(
    py: Python<'_>,
    session: &SparkSession,
    holders: &[&crate::gc::Parts],
    f: impl FnOnce() -> Result<T, E> + Send,
) -> PyResult<T> {
    crate::gc::of_parts(py, holders, || interruptible(py, session, f))
}

/// As [`interruptible_of`], for a call that clones and drops only the parts that `values` hold,
/// directly or below: values of the core that the caller holds for the time of the call, such as
/// the clone of the DataFrame of an ML fit or a writer of a copy of a state (see
/// [`crate::gc::of_values`]).
pub(crate) fn interruptible_of_values<'a, T: Send, E: Send + Into<Box<SparkError>>>(
    py: Python<'_>,
    session: &SparkSession,
    values: impl IntoIterator<Item = spark_connect::gc::PartRef<'a>>,
    f: impl FnOnce() -> Result<T, E> + Send,
) -> PyResult<T> {
    crate::gc::of_values(py, values, || interruptible(py, session, f))
}

/// The outcome of a request of `session`, of its error as an exception converted of `session`
/// as the session of the request (see [`crate::errors::of_request_session`]).
fn of_the_request<T>(session: &SparkSession, outcome: Result<T, Box<SparkError>>) -> PyResult<T> {
    match outcome {
        Ok(value) => Ok(value),
        Err(error) => crate::errors::of_request_session(session, || Err(error).to_pyerr()),
    }
}

/// Run `f`, a wait that no session stands behind, without the GIL, on the calling thread: a
/// wait of `block_on_abortable` that a signal of the process ends, as the acquire of PySpark's
/// lock is interrupted by one. `f` returns None then, and the exception of the signal is raised.
/// Without a signal, `f` returns its output. It waits for a lock of the core and clones no part,
/// so it is not counted for the garbage collector (see [`crate::gc::detach_without_parts`]).
pub(crate) fn abortable_without_session<T: Send>(
    py: Python<'_>,
    f: impl FnOnce() -> Option<T> + Send,
) -> PyResult<T> {
    let signal: Arc<Mutex<Option<PyErr>>> = Arc::default();
    let poll: Arc<WaitWatch> = {
        let signal = Arc::clone(&signal);
        Arc::new(move || {
            let mut error = signal.lock().unwrap_or_else(|e| e.into_inner());
            if error.is_some() {
                return Watch::Abort;
            }
            // Takes the GIL for the check only, as the wait runs without it; of an interpreter
            // that the GIL cannot be taken of any more, as of a daemon thread at the exit, the
            // wait ends.
            match Python::try_attach(crate::deliver::check_signals) {
                Some(Ok(())) => Watch::Continue,
                Some(Err(raised)) => {
                    *error = Some(raised);
                    Watch::Abort
                }
                None => Watch::Abort,
            }
        })
    };
    match crate::gc::detach_without_parts(py, || with_wait_watch(poll, f)) {
        Some(output) => Ok(output),
        None => Err(signal
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take()
            .unwrap_or_else(|| pyo3::exceptions::PyKeyboardInterrupt::new_err(()))),
    }
}

/// As [`interruptible`], but a failure of `f` is the inner result, which the caller handles, as
/// a warning. The outer error is the exception of a signal handler only, which is never handled
/// as a failure of the call: as PySpark's `except Exception` does not catch it.
pub(crate) fn interruptible_outcome<T: Send, E: Send + Into<Box<SparkError>>>(
    py: Python<'_>,
    session: &SparkSession,
    f: impl FnOnce() -> Result<T, E> + Send,
) -> PyResult<Result<T, Box<SparkError>>> {
    run_interruptible(py, session, f, false, RequestContext::Full)
}

/// As [`interruptible_outcome`], for a call that PySpark makes in a `try` whose `except Exception`
/// warns, as its listener bus does: the error of an interrupt that the server failed, which
/// replaces the KeyboardInterrupt of the signal, is caught there, so it is the inner result too.
pub(crate) fn interruptible_caught<T: Send, E: Send + Into<Box<SparkError>>>(
    py: Python<'_>,
    session: &SparkSession,
    f: impl FnOnce() -> Result<T, E> + Send,
) -> PyResult<Result<T, Box<SparkError>>> {
    run_interruptible(py, session, f, true, RequestContext::Full)
}

/// As [`interruptible`], for a call whose requests carry no user context extensions, as PySpark's
/// requests of `CloneSession`, `ReleaseSession` and `GetStatus` carry none, but the session ID and
/// the user of the client, which are brought up to date before it (see [`RequestContext`]): so an
/// extension that is not an `Any`, which fails each call whose requests carry them, does not fail
/// it, as in PySpark.
pub(crate) fn interruptible_without_user_context<T: Send, E: Send + Into<Box<SparkError>>>(
    py: Python<'_>,
    session: &SparkSession,
    f: impl FnOnce() -> Result<T, E> + Send,
) -> PyResult<T> {
    // Of the session alone, which clones no part (see `interruptible_without_parts`).
    let outcome = crate::gc::without_parts(|| {
        run_interruptible(py, session, f, false, RequestContext::Identity)
    });
    of_the_request(session, outcome?)
}

/// As [`interruptible_outcome`], for a call that clones and drops only the parts that `values`
/// hold (see [`interruptible_of_values`]).
pub(crate) fn interruptible_outcome_of_values<'a, T: Send, E: Send + Into<Box<SparkError>>>(
    py: Python<'_>,
    session: &SparkSession,
    values: impl IntoIterator<Item = spark_connect::gc::PartRef<'a>>,
    f: impl FnOnce() -> Result<T, E> + Send,
) -> PyResult<Result<T, Box<SparkError>>> {
    crate::gc::of_values(py, values, || interruptible_outcome(py, session, f))
}

/// As [`interruptible_outcome`], for a call that clones no part (see
/// [`interruptible_without_parts`]).
pub(crate) fn interruptible_outcome_without_parts<T: Send, E: Send + Into<Box<SparkError>>>(
    py: Python<'_>,
    session: &SparkSession,
    f: impl FnOnce() -> Result<T, E> + Send,
) -> PyResult<Result<T, Box<SparkError>>> {
    crate::gc::without_parts(|| interruptible_outcome(py, session, f))
}

/// As [`interruptible_caught`], for a call that clones no part (see
/// [`interruptible_without_parts`]).
pub(crate) fn interruptible_caught_without_parts<T: Send, E: Send + Into<Box<SparkError>>>(
    py: Python<'_>,
    session: &SparkSession,
    f: impl FnOnce() -> Result<T, E> + Send,
) -> PyResult<Result<T, Box<SparkError>>> {
    crate::gc::without_parts(|| interruptible_caught(py, session, f))
}

/// As [`interruptible_outcome`], for a call whose requests carry no user context extensions (see
/// [`interruptible_without_user_context`]).
pub(crate) fn interruptible_outcome_without_user_context<
    T: Send,
    E: Send + Into<Box<SparkError>>,
>(
    py: Python<'_>,
    session: &SparkSession,
    f: impl FnOnce() -> Result<T, E> + Send,
) -> PyResult<Result<T, Box<SparkError>>> {
    // Of the session alone, which clones no part (see `interruptible_without_parts`).
    crate::gc::without_parts(|| run_interruptible(py, session, f, false, RequestContext::Identity))
}

/// As [`interruptible_without_user_context`], for the `CloneSession` of `cloneSession`, whose
/// failure PySpark's `clone` does not handle (see [`RequestContext::Clone`]).
pub(crate) fn interruptible_of_clone<T: Send, E: Send + Into<Box<SparkError>>>(
    py: Python<'_>,
    session: &SparkSession,
    f: impl FnOnce() -> Result<T, E> + Send,
) -> PyResult<T> {
    // Of the session alone, which clones no part (see `interruptible_without_parts`).
    crate::gc::without_parts(|| run_interruptible(py, session, f, false, RequestContext::Clone))?
        .to_pyerr()
}

/// As [`interruptible`], for a call of the artifacts, whose requests PySpark's ArtifactManager
/// builds of the session ID and the user that the client was made with, and of no extensions:
/// nothing of the client is read before it. Its failure is handled as that of a request of the
/// client when `handled`, as PySpark's `add_artifacts` handles it, and its `copy_from_local_to_fs`
/// and `cache_artifacts` do not (see [`RequestContext::handles_errors`]).
pub(crate) fn interruptible_of_artifacts<T: Send, E: Send + Into<Box<SparkError>>>(
    py: Python<'_>,
    session: &SparkSession,
    handled: bool,
    f: impl FnOnce() -> Result<T, E> + Send,
) -> PyResult<T> {
    let context = if handled {
        RequestContext::Artifacts
    } else {
        RequestContext::UnhandledArtifacts
    };
    // Of files and bytes, which clone no part (see `interruptible_without_parts`).
    crate::gc::without_parts(|| run_interruptible(py, session, f, false, context))?.to_pyerr()
}

/// As [`interruptible`], for a call that continues an execution that an earlier call started,
/// the `next()` of its responses or of its rows, which sends only the reattaches and the releases
/// that copy the request of the execution, as PySpark's: nothing of the client is read before it,
/// and its failure is handled as that of the execution.
pub(crate) fn interruptible_continuation<T: Send, E: Send + Into<Box<SparkError>>>(
    py: Python<'_>,
    session: &SparkSession,
    f: impl FnOnce() -> Result<T, E> + Send,
) -> PyResult<T> {
    run_interruptible(py, session, f, false, RequestContext::Continuation)?.to_pyerr()
}

/// As [`interruptible_continuation`], for a call that clones no part (see
/// [`interruptible_without_parts`]): the next event of the listener bus.
pub(crate) fn interruptible_continuation_without_parts<T: Send, E: Send + Into<Box<SparkError>>>(
    py: Python<'_>,
    session: &SparkSession,
    f: impl FnOnce() -> Result<T, E> + Send,
) -> PyResult<T> {
    crate::gc::without_parts(|| interruptible_continuation(py, session, f))
}

/// What a call reads of the client of its session before it sends its requests, as PySpark's
/// requests read the attributes of its client when they are built (see
/// [`refresh_request_context`]), and whether PySpark's client handles its failure.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum RequestContext {
    /// The session ID, the user, the user context extensions and the tags, of their errors: the
    /// calls whose requests carry the extensions, of executions, analyses, configurations and
    /// interrupts.
    Full,
    /// The session ID and the user, of their errors, and the extensions and the tags, which the
    /// request of the details of a failure carries, of none: the calls of `ReleaseSession` and
    /// `GetStatus`.
    Identity,
    /// As `Identity`, of a call whose failure PySpark's client does not handle: `CloneSession`.
    Clone,
    /// All of them, of none of their errors, for the request of the details of a failure: the
    /// uploads of the artifacts, whose own requests are of the session ID and the user that the
    /// client was made with.
    Artifacts,
    /// As `Artifacts`, of a call whose failure PySpark's client does not handle: the copies of
    /// local files to the file system of the server and the cached artifacts.
    UnhandledArtifacts,
    /// Nothing: the calls that continue an execution (see [`interruptible_continuation`]).
    Continuation,
    /// All of them, before the request of a finalizer, which is not sent when one fails (see
    /// [`before_finalizer_request`]).
    Finalizer,
}

impl RequestContext {
    /// Whether PySpark's client handles the failure of the call in its `_handle_error`, which
    /// sets `inside_error_handling` in its `thread_local` while it does, and leaves it False.
    fn handles_errors(self) -> bool {
        !matches!(
            self,
            RequestContext::Clone | RequestContext::UnhandledArtifacts | RequestContext::Finalizer
        )
    }

    /// The mode of the refresher of the client (see
    /// `pyspark.sql.connect.client.core._RequestContext`): 2 of all of the context, raising its
    /// errors, 3 as 2 of the client of the session rather than of a copy whose method runs, 1 of
    /// the session ID and the user, raising theirs, and of the rest of none, 0 of all of it, of
    /// none, or no refresh.
    fn mode(self) -> Option<u8> {
        match self {
            RequestContext::Full => Some(2),
            RequestContext::Finalizer => Some(3),
            RequestContext::Identity | RequestContext::Clone => Some(1),
            RequestContext::Artifacts | RequestContext::UnhandledArtifacts => Some(0),
            RequestContext::Continuation => None,
        }
    }
}

/// What a refresh of the request context leaves to do once the call ended (see
/// [`after_call`]).
#[derive(Default)]
pub(crate) struct Refreshed {
    /// The thread had no tags: an execution of the call gives it an empty set of them.
    without_tags: bool,
    /// The token of the contextvar of the copy of a client whose method makes the call, which is
    /// reset once the call ended, so that the calls that the call makes meanwhile, as of a
    /// progress handler or a finalizer, are of the state of the session.
    copy_token: Option<Py<PyAny>>,
    /// What the session had of the settings of the client as the call started, which the call
    /// writes back into the attributes of the client when it changed them (see [`CallSnapshot`]).
    snapshot: Option<CallSnapshot>,
}

/// The closed flag, the server-side session ID and the plan compression of a session as a call
/// started, once the session was given the attributes of its client (see [`apply_call_settings`]):
/// what PySpark's client changes of its own attributes as its requests go, its `_closed` when the
/// server says that the session changed, its `_server_session_id` of its first response, and its
/// `_plan_compression_threshold` and `_plan_compression_algorithm` of their lookup, which the
/// call writes back into the attributes when it changed them (see [`after_call`]).
#[derive(Clone, PartialEq)]
pub(crate) struct CallSnapshot {
    closed: bool,
    server_session_id: Option<String>,
    compression: Option<spark_connect_core::client::PlanCompressionSettings>,
}

impl CallSnapshot {
    fn of(session: &SparkSession) -> Self {
        CallSnapshot {
            closed: session.is_stopped(),
            server_session_id: session.server_session_id(),
            compression: session.plan_compression_settings(),
        }
    }
}

/// The settings of a client of `_call_settings` of `pyspark.sql.connect.client.core`: of the
/// closed flag, the server-side session ID, the retries and the plan compression, the value and
/// whether the session has another, and the deadlines, the result chunking, the reattachability,
/// the session hooks and the number of the progress handlers, which the call takes as they are.
type CallSettings<'py> = (
    (bool, bool),
    (Option<String>, bool),
    Bound<'py, PyDict>,
    (Bound<'py, PyAny>, Bound<'py, PyAny>),
    bool,
    (
        (
            Option<Bound<'py, PyDict>>,
            Option<Bound<'py, PyAny>>,
            Option<f64>,
        ),
        bool,
    ),
    ((Bound<'py, PyAny>, Bound<'py, PyAny>), bool),
    Bound<'py, PyAny>,
    usize,
);

/// Give `session` the settings of its client of `settings` (see [`CallSettings`]), as PySpark's
/// requests read the attributes of their client when they are sent: the closed flag, the
/// server-side session ID, the retries and the plan compression where the session has others, the
/// deadlines of the RPCs and the result chunking of the executions of the calling thread, the
/// reattachability of the executions, the session hooks of the executions of the call and the
/// number of the progress handlers. What the session then has of what its calls change is the
/// snapshot that the call writes back from (see [`CallSnapshot`]).
fn apply_call_settings(
    py: Python<'_>,
    session: &SparkSession,
    settings: &Bound<'_, PyAny>,
    of_the_session: bool,
) -> PyResult<CallSnapshot> {
    let (
        (closed, closed_changed),
        (server_session_id, server_session_id_changed),
        deadlines,
        (allow_chunking, preferred_chunk_size),
        reattachable,
        ((native, decider, elapsed), retry_changed),
        ((threshold, algorithm), compression_changed),
        hooks,
        handlers,
    ): CallSettings<'_> = settings.extract()?;
    // As the fields of PySpark's `ResultChunkingOptions`, a bool and an optional int.
    let chunking = crate::client::result_chunking(
        crate::client::proto_bool(&allow_chunking)?,
        if preferred_chunk_size.is_none() {
            None
        } else {
            Some(preferred_chunk_size.extract::<i64>()?)
        },
    );
    let deadlines = crate::client::rpc_deadlines(&deadlines)?;
    if closed_changed {
        if closed {
            session.mark_closed();
        } else {
            session.reopen();
        }
    }
    if server_session_id_changed {
        session.set_server_session_id(server_session_id);
    }
    session.set_thread_call_settings(Some(deadlines), Some(chunking));
    session.set_reattachable_execute(reattachable);
    if retry_changed {
        crate::client::apply_retries(py, session, native.as_ref(), decider, elapsed)?;
    }
    if compression_changed {
        session.set_plan_compression_settings(crate::client::plan_compression_settings(
            &threshold, &algorithm,
        )?);
    }
    crate::session::set_call_hooks(session, &hooks, of_the_session);
    crate::progress::set_handler_count(py, session, handlers);
    Ok(CallSnapshot::of(session))
}

/// Bring the request context of `session` up to date before a call sends its requests, on the
/// calling thread, with the GIL, of the attributes of `client` that PySpark's requests read when
/// they are built: the function of `client` that the session keeps (see
/// `PySparkSession::_set_user_context_refresh`) gives the session ID and the user of
/// `client._session_id` and `_user_id`, the `google.protobuf.Any` messages of its lists of user
/// context extensions, serialized again so that a change of a message in place is sent, and the
/// tags of the `thread_local` of the calling thread, of the client that the thread calls (a copy
/// of a client has its own). They are the requests' of the calling thread from then on, also of
/// the requests that the call sends of no request of its own, as the details of a failure, and
/// those of the session when they are of the state of the session. Their errors are raised as
/// [`RequestContext`] says, as PySpark raises those of the requests that read them from the call,
/// such as upb's `TypeError` of a session ID that is not a str or of an extension that is not an
/// `Any`; a tag that the field of the tags does not take fails only the executions, which read
/// the tags (see `spark_connect::session::Tag`).
pub(crate) fn refresh_request_context(
    py: Python<'_>,
    session: &SparkSession,
    context: RequestContext,
) -> PyResult<Refreshed> {
    let Some(refresher) = session.user_context_refresher() else {
        return Ok(Refreshed::default());
    };
    let Some(refresh) = crate::state::refresh_of(refresher.as_ref()) else {
        return Ok(Refreshed::default());
    };
    let Some(mode) = context.mode() else {
        // A call that reads nothing of the client holds the contextvar of the copy of a client
        // whose method makes it, as a call that reads it does.
        let copy_token = refresh.bind(py).call_method0("_hold")?;
        return Ok(Refreshed {
            without_tags: false,
            copy_token: (!copy_token.is_none()).then(|| copy_token.unbind()),
            snapshot: None,
        });
    };
    let refreshed = refresh.bind(py).call1((mode,))?;
    let (session_id, user_id, global, own, tags, copy_token, of_the_session, settings): RefreshedContext<'_> =
        refreshed.extract()?;
    let snapshot = if settings.is_none() {
        None
    } else {
        Some(apply_call_settings(py, session, &settings, of_the_session)?)
    };
    let unchanged = |value: &Bound<'_, PyAny>| value.is(py.Ellipsis().bind(py));
    let session_id = if unchanged(&session_id) {
        session.client_session_id()
    } else {
        session_id.extract()?
    };
    let user_id = if unchanged(&user_id) {
        session.request_user_id()
    } else {
        user_id.extract()?
    };
    let global = if unchanged(&global) {
        session.global_user_context_extensions()
    } else {
        crate::session::decoded_extensions(global.extract()?)?
    };
    if of_the_session {
        session.set_client_session_id(session_id.clone());
        session.set_request_user_id(user_id.clone());
        session.set_global_user_context_extensions(global.clone());
    }
    session.set_thread_request_context(session_id, user_id, global);
    if !unchanged(&own) {
        match own.extract::<Option<Vec<Vec<u8>>>>()? {
            Some(own) => session.share_threadlocal_user_context_extensions(
                &spark_connect_core::client::SharedUserContextExtensions::new(
                    crate::session::decoded_extensions(own)?,
                ),
            ),
            None => session.clear_threadlocal_user_context_extensions(),
        }
    }
    let copy_token = (!copy_token.is_none()).then(|| copy_token.unbind());
    if unchanged(&tags) {
        return Ok(Refreshed {
            without_tags: false,
            copy_token,
            snapshot,
        });
    }
    let without_tags = tags.is_none();
    if without_tags {
        session.clear_tags();
    } else {
        session.share_thread_tags(&spark_connect::session::SharedTags::new(
            crate::session::core_tags_of(&tags)?,
        ));
    }
    Ok(Refreshed {
        without_tags: without_tags && context == RequestContext::Full
            || without_tags && context == RequestContext::Finalizer,
        copy_token,
        snapshot,
    })
}

/// What the refresher of a client gives (see [`refresh_request_context`]): the session ID and the
/// user of the requests, the bytes of each `google.protobuf.Any` of the global extensions and of
/// those of the thread, if it has a list of them, and the tags of the thread, if it has a set of
/// them, or the `TypeError` of the `list()` of a tags value that is not iterable, each `...` when
/// a refresh of none of the errors failed to read it; the token of the contextvar of a copy of a
/// client whose method calls, or None; and whether they are of the state of the session.
type RefreshedContext<'py> = (
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    bool,
    Bound<'py, PyAny>,
);

/// What PySpark's client leaves in the `thread_local` of the calling thread once a call ended:
/// the empty set of tags that the `get_tags` of the request of an execution gives a thread of
/// none (`read_tags`), and the `inside_error_handling` that its `_handle_error` leaves False once
/// it handled a failure (`handled_error`); and the reset of the contextvar of a copy of a client.
fn after_call(
    py: Python<'_>,
    session: &SparkSession,
    read_tags: bool,
    handled_error: bool,
    copy_token: Option<Py<PyAny>>,
    snapshot: Option<CallSnapshot>,
) -> PyResult<()> {
    // What the call changed of the settings of the client that the session keeps (see
    // `CallSnapshot`), of each `...` when it changed none.
    let written = snapshot
        .map(|before| (CallSnapshot::of(session), before))
        .filter(|(after, before)| after != before)
        .map(|(after, before)| -> PyResult<Py<PyAny>> {
            let ellipsis = || py.Ellipsis().into_bound(py);
            let closed = if after.closed != before.closed {
                pyo3::types::PyBool::new(py, after.closed)
                    .to_owned()
                    .into_any()
            } else {
                ellipsis()
            };
            let server_session_id = if after.server_session_id != before.server_session_id {
                after.server_session_id.into_pyobject(py)?.into_any()
            } else {
                ellipsis()
            };
            let compression = if after.compression != before.compression {
                match after.compression {
                    Some(settings) => (settings.threshold, settings.algorithm)
                        .into_pyobject(py)?
                        .into_any(),
                    None => py.None().into_bound(py),
                }
            } else {
                ellipsis()
            };
            Ok(
                pyo3::types::PyTuple::new(py, [closed, server_session_id, compression])?
                    .into_any()
                    .unbind(),
            )
        })
        .transpose()?;
    if !read_tags && !handled_error && copy_token.is_none() && written.is_none() {
        return Ok(());
    }
    if let Some(refresher) = session.user_context_refresher() {
        if let Some(refresh) = crate::state::refresh_of(refresher.as_ref()) {
            refresh.bind(py).call_method1(
                "_after_call",
                (read_tags, handled_error, copy_token, written),
            )?;
        }
    }
    Ok(())
}

/// The hook of the requests of the finalizers of a session before a request is built (see
/// `spark_connect::session::FinalizerRequestHook`): the context of the requests of the
/// current thread brought up to date of the state of its client, as PySpark's finalizers build
/// their requests when they run, on a thread of Python; whether to send the request, which is not
/// sent when one of the values that it reads is one that protobuf refuses, as PySpark's finalizers
/// catch the error of the request and send none. A thread that is not one of Python, or an
/// interpreter that shuts down, has none to read: the request is of the context of the last call
/// that the thread, or the session, made. The session gives the thread its context of before the
/// hook once the request is built, so that a call that the finalizer ran during, as on a
/// collection of the garbage collector, keeps its own (see [`after_finalizer_request`]). A
/// finalizer that runs while an exception propagates, as the drop of the last owner of a fitted
/// model of a frame that CPython unwinds, keeps that exception, as CPython keeps it around a
/// `__del__` (see [`crate::deliver::preserving_pending_exception`]).
pub(crate) fn before_finalizer_request(session: &SparkSession) -> bool {
    // SAFETY: a read of the thread state of the current thread, which needs no lock.
    if unsafe { pyo3::ffi::PyGILState_GetThisThreadState() }.is_null() {
        return true;
    }
    Python::try_attach(|py| {
        crate::deliver::preserving_pending_exception(py, |py| {
            match read_for_finalizer_request(py, session) {
                Ok(()) => true,
                // As PySpark's finalizers, whose `except Exception` takes the error of the build
                // of the request, out of whose `__del__` another exception goes, which Python
                // reports as an unraisable one: the read after the request is not made then,
                // which would raise it again (see `after_finalizer_request`).
                Err(error) => {
                    if !error.is_instance_of::<pyo3::exceptions::PyException>(py) {
                        error.write_unraisable(py, None);
                        FINALIZER_REPORTED.with(|reported| reported.set(true));
                    }
                    false
                }
            }
        })
    })
    .unwrap_or(true)
}

/// Read what the request of a finalizer of `session` reads of its client as it is built, as
/// [`before_finalizer_request`] does: the error of the build of the request.
fn read_for_finalizer_request(py: Python<'_>, session: &SparkSession) -> PyResult<()> {
    let refreshed = refresh_request_context(py, session, RequestContext::Finalizer)?;
    // The levels of its logs, and the call stack of its request, of the finalizer as it runs, as
    // PySpark's builds its request then, of its `__del__`, whose build the error of the read of
    // the call stack fails. Of a finalizer that runs during a call of the thread, the call stack
    // of that call, as PySpark's ends at the first frame of the pyspark package.
    let started = match enclosing_call_stack(false) {
        Some(call_stack) => {
            crate::client_log::start_nested_call(py, session, call_stack);
            Ok(())
        }
        None => crate::client_log::start_call(py, session, crate::client_log::Called::Finalizer)
            .map(drop),
    };
    // An `Exception` of the reset of its state is not the request's; another one, as the
    // `KeyboardInterrupt` of a signal handler that ran meanwhile, goes out of the finalizer, as
    // out of PySpark's `__del__`.
    match after_call(
        py,
        session,
        refreshed.without_tags,
        false,
        refreshed.copy_token,
        refreshed.snapshot,
    ) {
        Err(error) if !error.is_instance_of::<pyo3::exceptions::PyException>(py) => Err(error),
        _ => started,
    }
}

thread_local! {
    /// Whether the read of the request of the finalizer that runs on this thread raised an
    /// exception that is not an `Exception`, which was reported (see [`before_finalizer_request`]).
    static FINALIZER_REPORTED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

thread_local! {
    /// The calls of [`run_interruptible`] that run on this thread, each during the one before it,
    /// as the call of a progress handler during the call that it reports on.
    static CALLS: std::cell::RefCell<Vec<CallOfThread>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// A call of [`run_interruptible`] that runs on this thread: the refresher of its session, which
/// tells the clients of the session apart from those of another, what it reads of the client of
/// its session, the token of the contextvar of the copy of a client whose method made it, of its
/// refresh, the exception that is not an `Exception` that a read of the attributes of its client
/// for it raised, which its wait or its end raises (see [`keep_for_nearest_call`]), and the errors
/// of the reads for the requests of its failures, which the failures carry (see
/// [`before_failure_requests`]).
struct CallOfThread {
    refresher: Option<Arc<dyn std::any::Any + Send + Sync>>,
    context: RequestContext,
    copy_token: Option<Py<PyAny>>,
    raised: Option<PyErr>,
    failure_errors: Vec<SparkError>,
    /// The call stack of the user's code that the requests of the call carry, once it started,
    /// which the calls that run during it carry too.
    call_stack: Option<prost_types::Any>,
}

/// The call stack of the call during which a call or a finalizer that starts on this thread runs,
/// if one runs: of the call before the last of the calls of the thread, of a call that `entered`
/// already, else of the last. A call stack of none is of a call whose requests carry none.
fn enclosing_call_stack(entered: bool) -> Option<Option<prost_types::Any>> {
    CALLS.with(|calls| {
        let calls = calls.try_borrow().ok()?;
        let at = calls.len().checked_sub(if entered { 2 } else { 1 })?;
        Some(calls[at].call_stack.clone())
    })
}

/// Keep `call_stack` as the call stack of the last call of the thread.
fn keep_call_stack(call_stack: Option<prost_types::Any>) {
    CALLS.with(|calls| {
        if let Ok(mut calls) = calls.try_borrow_mut() {
            if let Some(call) = calls.last_mut() {
                call.call_stack = call_stack;
            }
        }
    });
}

/// A call of [`run_interruptible`] on this thread, as long as it lives. A call that runs during
/// another call of the thread gives the thread the context of its own requests, and the thread has
/// the context of before it again once it ends, however it ends, read again of the client of the
/// nearest call of its session that runs (see [`reread_nearest_call_of`]).
struct Call<'a> {
    session: &'a SparkSession,
    /// The context of the requests of the thread before the call, of a call that runs during
    /// another.
    kept: Option<KeptThreadRequestContext<'a>>,
    /// Whether [`Call::end`] ended it.
    ended: bool,
}

impl<'a> Call<'a> {
    /// Enter a call of `session` of the request context `context`.
    fn enter(session: &'a SparkSession, context: RequestContext) -> Self {
        let call = CallOfThread {
            refresher: session.user_context_refresher(),
            context,
            copy_token: None,
            raised: None,
            failure_errors: Vec::new(),
            call_stack: None,
        };
        let nested = CALLS.with(|calls| {
            let mut calls = calls.borrow_mut();
            calls.push(call);
            calls.len() > 1
        });
        Call {
            session,
            kept: nested.then(|| session.keep_thread_request_context()),
            ended: false,
        }
    }

    /// Keep the token of the contextvar of the copy of a client whose method makes the call, of
    /// its refresh, if any.
    fn of_copy(&self, py: Python<'_>, token: Option<&Py<PyAny>>) {
        CALLS.with(|calls| {
            if let Some(call) = calls.borrow_mut().last_mut() {
                call.copy_token = token.map(|token| token.clone_ref(py));
            }
        });
    }

    /// End the call, after which the thread has the context of the requests of the call during
    /// which it ran, if it ran during another, read again of its client: the exception of that
    /// read that is not an `Exception`, as the `KeyboardInterrupt` of a signal handler that ran
    /// then, which the call raises in place of its outcome, as PySpark raises it where the signal
    /// arrives, in the call; what the call keeps of the reads for it (see [`Ended`]).
    fn end(mut self, py: Python<'_>) -> Ended {
        self.ended = true;
        let (nested, ended) = self.leave();
        let reread = if nested {
            reread_after_call(py, self.session).err()
        } else {
            None
        };
        Ended { reread, ..ended }
    }

    /// End a call whose refresh failed of `error`, as [`Call::end`]: the exception of the read
    /// that is not an `Exception`, or one that a read for the call kept, is raised in place of
    /// `error`, which becomes its context.
    fn end_unrefreshed(mut self, py: Python<'_>, error: PyErr) -> PyErr {
        self.ended = true;
        let (nested, mut ended) = self.leave();
        ended.release_failure_errors(None);
        // An exception that a read for the call kept replaces the error of the refresh, as at the
        // end of a call that waited, and the exception of the read at its end replaces both.
        let error = match ended.raised.take() {
            Some(raised) => crate::errors::with_context(py, raised, error),
            None => error,
        };
        match nested
            .then(|| reread_after_call(py, self.session).err())
            .flatten()
        {
            Some(reread) => crate::errors::with_context(py, reread, error),
            None => error,
        }
    }

    /// Take the call off the stack of the thread, and give the thread its context of before the
    /// call again, of a call that ran during another: whether it did, and what the call keeps of
    /// the reads for it.
    fn leave(&mut self) -> (bool, Ended) {
        // The token and the exceptions are taken once the stack is no longer borrowed.
        let call = CALLS.with(|calls| calls.borrow_mut().pop());
        let ended = match call {
            Some(call) => Ended {
                reread: None,
                raised: call.raised,
                failure_errors: call.failure_errors,
            },
            None => Ended {
                reread: None,
                raised: None,
                failure_errors: Vec::new(),
            },
        };
        match self.kept.take() {
            Some(kept) => {
                drop(kept);
                (true, ended)
            }
            None => (false, ended),
        }
    }
}

impl Drop for Call<'_> {
    fn drop(&mut self) {
        if self.ended {
            return;
        }
        // A call that ends of a panic: the thread has its context of before it again, read again
        // as at the end of any other call, and nothing raises what the call keeps, nor takes an
        // exception that is set meanwhile.
        let (nested, ended) = self.leave();
        let session = self.session;
        let _ = Python::try_attach(|py| {
            crate::deliver::preserving_pending_exception(py, |py| {
                if nested {
                    if let Err(reread) = reread_after_call(py, session) {
                        reread.write_unraisable(py, None);
                    }
                }
                ended.drop_unraised(py);
            })
        });
    }
}

/// What a call keeps of the reads of the attributes of its client for it, once it ended (see
/// [`Call::end`]).
struct Ended {
    /// The exception, that is not an `Exception`, of the read at its end, of a call that ran
    /// during another.
    reread: Option<PyErr>,
    /// The exception, that is not an `Exception`, that a read for it raised during it, which
    /// its wait has not raised (see [`keep_for_nearest_call`]).
    raised: Option<PyErr>,
    /// The errors of the reads for the requests of its failures, which the failures carry (see
    /// [`before_failure_requests`]).
    failure_errors: Vec<SparkError>,
}

impl Ended {
    /// Release the exceptions that the errors of the failures of the call carry, but that of
    /// `failure`, the failure that the call returns, if it is one of them: a failure that the
    /// call handled itself, as the failed lookup of the plan compression settings, drops the
    /// exception of the read for it with it, as PySpark's `except Exception` takes it.
    fn release_failure_errors(&mut self, failure: Option<&SparkError>) {
        let kept = failure.and_then(crate::errors::callback_error_id);
        crate::errors::release_callback_errors(
            self.failure_errors
                .drain(..)
                .filter(|error| crate::errors::callback_error_id(error) != kept),
        );
    }

    /// Report the exceptions that the call keeps, which nothing raises, as unraisable ones are.
    fn drop_unraised(mut self, py: Python<'_>) {
        self.release_failure_errors(None);
        for raised in [self.reread.take(), self.raised.take()]
            .into_iter()
            .flatten()
        {
            raised.write_unraisable(py, None);
        }
    }
}

/// Give the thread the context of the call of `session` that runs on it again, if any, read again
/// of its client of none of their errors, once a call or a finalizer that ran during it ended (see
/// [`reread_nearest_call`]): the exception of the read that is not an `Exception`, as the
/// `KeyboardInterrupt` or the `SystemExit` of a signal handler that ran meanwhile, is returned.
fn reread_after_call(py: Python<'_>, session: &SparkSession) -> PyResult<()> {
    let Some(refresher) = session.user_context_refresher() else {
        return Ok(());
    };
    reread_nearest_call(py, &refresher, session, ReadOf::Rest)
}

/// Keep `raised`, an exception that is not an `Exception` of a read of the attributes of a client
/// for the nearest call of the session whose refresher is `refresher` that runs on the calling
/// thread, for that call: its wait raises it at its next poll, as the exception of a signal
/// handler, which interrupts the execution that it waits for when it is a `KeyboardInterrupt`
/// (see [`SignalWatch::poll`]), or the call does when it returns, in place of its outcome. A call
/// that already keeps one keeps the first, and the other is reported as an unraisable one is, as
/// it is when no call takes it.
fn keep_for_nearest_call(
    py: Python<'_>,
    refresher: &Arc<dyn std::any::Any + Send + Sync>,
    raised: PyErr,
) {
    let unkept = CALLS.with(|calls| {
        let Ok(mut calls) = calls.try_borrow_mut() else {
            return Some(raised);
        };
        match calls.iter_mut().rev().find(|call| is_of(call, refresher)) {
            Some(call) if call.raised.is_none() => {
                call.raised = Some(raised);
                None
            }
            _ => Some(raised),
        }
    });
    if let Some(unkept) = unkept {
        unkept.write_unraisable(py, None);
    }
}

/// The exception that the call of `session` whose wait polls, the innermost call of the thread,
/// keeps for it, which is taken (see [`keep_for_nearest_call`]).
fn take_raised_of_waiting_call(session: &SparkSession) -> Option<PyErr> {
    let refresher = session.user_context_refresher()?;
    CALLS.with(|calls| {
        let mut calls = calls.try_borrow_mut().ok()?;
        let call = calls.last_mut()?;
        if !is_of(call, &refresher) {
            return None;
        }
        call.raised.take()
    })
}

/// Whether `call` is a call of the session whose refresher is `refresher`.
/// The attributes of the copy of a client whose method makes the nearest call of `session` that
/// runs on the calling thread, the `__dict__` of the copy (see `_COPY_CALL` of
/// `pyspark.sql.connect.client.core`), else None, of a call of the client of the session: as the
/// execution of PySpark's copy reads the `_progress_handlers` and the `_profiler_collector` of the
/// copy. The callbacks of an execution, its progress and its Python accumulators, run on the
/// thread of its call.
pub(crate) fn calling_copy_state<'py>(
    py: Python<'py>,
    session: &SparkSession,
) -> Option<Bound<'py, PyDict>> {
    let refresher = session.user_context_refresher()?;
    let token = CALLS.with(|calls| {
        let calls = calls.try_borrow().ok()?;
        let call = calls.iter().rev().find(|call| is_of(call, &refresher))?;
        call.copy_token.as_ref().map(|token| token.clone_ref(py))
    })?;
    // The value of the contextvar before the call held it: the refresher and the attributes of
    // the copy.
    let copy = token
        .bind(py)
        .getattr(pyo3::intern!(py, "old_value"))
        .ok()?
        .get_item(1)
        .ok()?;
    copy.cast_into::<PyDict>().ok()
}

fn is_of(call: &CallOfThread, refresher: &Arc<dyn std::any::Any + Send + Sync>) -> bool {
    call.refresher
        .as_ref()
        .is_some_and(|other| std::ptr::addr_eq(Arc::as_ptr(other), Arc::as_ptr(refresher)))
}

/// What a re-read gives the context of the requests of the calling thread of (see
/// [`reread_nearest_call`]): a session, all of it, or its core client, all of it but the tags,
/// which only the executions carry.
trait ThreadRequestTarget {
    fn current_session_id(&self) -> String;
    fn current_user_id(&self) -> Option<String>;
    fn current_global(&self) -> Vec<(String, prost_types::Any)>;
    fn give_request_context(
        &self,
        session_id: String,
        user_id: Option<String>,
        global: Vec<(String, prost_types::Any)>,
    );
    fn give_own(&self, own: Option<Vec<(String, prost_types::Any)>>);
    fn give_tags(&self, tags: Option<Vec<spark_connect::session::Tag>>);
}

impl ThreadRequestTarget for SparkSession {
    fn current_session_id(&self) -> String {
        self.client_session_id()
    }

    fn current_user_id(&self) -> Option<String> {
        self.request_user_id()
    }

    fn current_global(&self) -> Vec<(String, prost_types::Any)> {
        self.global_user_context_extensions()
    }

    fn give_request_context(
        &self,
        session_id: String,
        user_id: Option<String>,
        global: Vec<(String, prost_types::Any)>,
    ) {
        self.set_thread_request_context(session_id, user_id, global);
    }

    fn give_own(&self, own: Option<Vec<(String, prost_types::Any)>>) {
        match own {
            Some(own) => self.share_threadlocal_user_context_extensions(
                &spark_connect_core::client::SharedUserContextExtensions::new(own),
            ),
            None => self.clear_threadlocal_user_context_extensions(),
        }
    }

    fn give_tags(&self, tags: Option<Vec<spark_connect::session::Tag>>) {
        match tags {
            Some(tags) => self.share_thread_tags(&spark_connect::session::SharedTags::new(tags)),
            None => self.clear_tags(),
        }
    }
}

impl ThreadRequestTarget for SparkConnectClient {
    fn current_session_id(&self) -> String {
        self.client_session_id()
    }

    fn current_user_id(&self) -> Option<String> {
        self.request_user_id()
    }

    fn current_global(&self) -> Vec<(String, prost_types::Any)> {
        self.global_user_context_extensions()
    }

    fn give_request_context(
        &self,
        session_id: String,
        user_id: Option<String>,
        global: Vec<(String, prost_types::Any)>,
    ) {
        self.set_thread_request_context(session_id, user_id, global);
    }

    fn give_own(&self, own: Option<Vec<(String, prost_types::Any)>>) {
        match own {
            Some(own) => self.share_threadlocal_user_context_extensions(
                &spark_connect_core::client::SharedUserContextExtensions::new(own),
            ),
            None => self.clear_threadlocal_user_context_extensions(),
        }
    }

    fn give_tags(&self, _tags: Option<Vec<spark_connect::session::Tag>>) {}
}

/// How strictly a re-read reads the attributes of a client (see [`reread_nearest_call`]).
#[derive(Clone, Copy, PartialEq, Eq)]
enum ReadOf {
    /// For the rest of the requests of a call, once a call or a finalizer that ran during it
    /// ended, of none of the errors of the values: as those of the artifacts are read.
    Rest,
    /// For the requests that a call builds of its own, when it builds them, of the errors of the
    /// session ID, the user and the extensions, as those of a call of the mode 2 are read, which
    /// replace the error of the call: of the details of its failure and of the configs of its
    /// stack trace, as PySpark's `_handle_error` builds them, and of the interrupt of a signal,
    /// as PySpark's `interrupt_operation` builds it in its `except KeyboardInterrupt`. The tags,
    /// which none of them carries, are read of none of their errors.
    Requests,
}

/// Give the requests of the calling thread the context of the nearest call of `session` that runs
/// on it, if any, read again of the attributes of its client as they are now (see
/// [`reread_nearest_call`]).
fn reread_nearest_call_of(py: Python<'_>, session: &SparkSession, read: ReadOf) -> PyResult<()> {
    match session.user_context_refresher() {
        Some(refresher) => reread_nearest_call(py, &refresher, session, read),
        None => Ok(()),
    }
}

/// Give the requests of the calling thread the context of the nearest call that runs on it of the
/// session whose refresher is `refresher`, if any, read again of the attributes of its client as
/// they are now, as PySpark's requests of the rest of that call read them when they are built,
/// given to `target`. It runs once a call that ran during that call ended, as one of a progress
/// handler, of that session or of another, and once a finalizer that ran during it built its
/// request, when the thread has the context of before them again (see
/// `SparkSession::keep_thread_request_context`), and before the requests of the details of a
/// failure, and of the interrupt of a signal, are built (see [`before_failure_requests`] and
/// [`SignalWatch::poll`]): the attributes may have changed meanwhile, as when a progress handler,
/// or another thread, adds a user context extension. The client is the copy whose method made the
/// call, of the token of its refresh, else the session's. Of [`ReadOf::Rest`], a value that
/// cannot be read stays as it was, and so does the whole context when the values cannot be taken,
/// and an exception of the read that is not an `Exception` is returned; of [`ReadOf::Requests`],
/// of the call of a mode that handles its errors, the error of a value is returned, and the
/// context stays as it was. A call of the mode of the artifacts that PySpark's client does not
/// handle the errors of is read as of [`ReadOf::Rest`].
fn reread_nearest_call(
    py: Python<'_>,
    refresher: &Arc<dyn std::any::Any + Send + Sync>,
    target: &dyn ThreadRequestTarget,
    read: ReadOf,
) -> PyResult<()> {
    let call = CALLS.with(|calls| {
        // The stack is only borrowed for a moment, by no code that reads the client.
        let calls = calls.try_borrow().ok()?;
        let call = calls.iter().rev().find(|call| is_of(call, refresher))?;
        Some((
            call.copy_token.as_ref().map(|token| token.clone_ref(py)),
            call.context,
        ))
    });
    let Some((token, context)) = call else {
        return Ok(());
    };
    let strict = read == ReadOf::Requests && context.handles_errors();
    let Some(refresh) = crate::state::refresh_of(refresher.as_ref()) else {
        return Ok(());
    };
    let reread = || -> PyResult<()> {
        let mode = if strict { 2 } else { 0 };
        let values = refresh
            .bind(py)
            .call_method1("_values_of_call", (token, mode))?;
        let (session_id, user_id, global, own, tags): CallValues<'_> = values.extract()?;
        let read = |value: &Bound<'_, PyAny>| !value.is(py.Ellipsis().bind(py));
        // All of them are taken before any is given to the thread.
        let session_id: Option<String> = read(&session_id)
            .then(|| session_id.extract())
            .transpose()?;
        let user_id: Option<Option<String>> =
            read(&user_id).then(|| user_id.extract()).transpose()?;
        let global = read(&global)
            .then(|| crate::session::decoded_extensions(global.extract()?))
            .transpose()?;
        let own = read(&own)
            .then(|| -> PyResult<_> {
                own.extract::<Option<Vec<Vec<u8>>>>()?
                    .map(crate::session::decoded_extensions)
                    .transpose()
            })
            .transpose()?;
        let tags = read(&tags)
            .then(|| -> PyResult<_> {
                if tags.is_none() {
                    return Ok(None);
                }
                crate::session::core_tags_of(&tags).map(Some)
            })
            .transpose();
        target.give_request_context(
            session_id.unwrap_or_else(|| target.current_session_id()),
            user_id.unwrap_or_else(|| target.current_user_id()),
            global.unwrap_or_else(|| target.current_global()),
        );
        if let Some(own) = own {
            target.give_own(own);
        }
        // The tags, which only the executions read, of none of their errors.
        if let Ok(Some(tags)) = tags {
            target.give_tags(tags);
        }
        Ok(())
    };
    match reread() {
        Err(error) if strict || !error.is_instance_of::<pyo3::exceptions::PyException>(py) => {
            Err(error)
        }
        _ => Ok(()),
    }
}

/// What the refresher of a client gives of a call that runs (see [`reread_nearest_call`]): the
/// session ID and the user of the requests, the bytes of each `google.protobuf.Any` of the global
/// extensions and of those of the thread, if it has a list of them, and the tags of the thread, as
/// [`RefreshedContext`] gives them, each `...` when it was not read.
type CallValues<'py> = (
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
    Bound<'py, PyAny>,
);

/// The hook of the requests of the finalizers of a session once a request is built, or not sent
/// (see `spark_connect::session::FinalizerRequestHook`): the thread has the context of the nearest
/// call of the session that runs on it again, if any, of the attributes of its client as they are
/// now (see [`reread_after_call`]); an exception of the read that is not an `Exception` is kept
/// for that call, which no call of the finalizer raises, whose wait raises it at its next poll, or
/// the call when it returns (see [`keep_for_nearest_call`]). As [`before_finalizer_request`], a
/// thread that is not one of Python, or an interpreter that shuts down, has none to read, and an
/// exception that propagates is kept.
pub(crate) fn after_finalizer_request(session: &SparkSession) {
    // A finalizer whose read raised and was reported reads nothing more, as PySpark's `__del__`
    // ends with the exception.
    if FINALIZER_REPORTED.with(|reported| reported.replace(false)) {
        return;
    }
    // SAFETY: a read of the thread state of the current thread, which needs no lock.
    if unsafe { pyo3::ffi::PyGILState_GetThisThreadState() }.is_null() {
        return;
    }
    let _ = Python::try_attach(|py| {
        crate::deliver::preserving_pending_exception(py, |py| {
            if let Err(raised) = reread_after_call(py, session) {
                // No call of the finalizer raises it, but the call during which it ran, at the
                // next poll of its wait, as the exception of a signal handler.
                if let Some(refresher) = session.user_context_refresher() {
                    keep_for_nearest_call(py, &refresher, raised);
                }
            }
        })
    });
}

/// The hook of the requests of the failure of a call of `client`, before the client builds them
/// (see `spark_connect_core::client::set_failure_requests_hook`): the requests of the details of
/// the error and of the configs of its stack trace are of the context of the nearest call of the
/// session of `client` that runs on the calling thread, read again of the attributes of its client
/// as they are now, of their errors (see [`ReadOf::Requests`]), as PySpark's `_handle_error`
/// builds them of them; whether to send them. No request of the failure is sent when the read
/// fails, as PySpark raises the error of the build of the request of the details before it sends
/// anything: an `Exception` of it is the failure, in place of `failure`, the error of the server,
/// which becomes its context, which the client raises, or drops when it handles the failure
/// itself, as PySpark's `except Exception` takes it; another exception, as the
/// `KeyboardInterrupt` of a signal handler, is kept for the call (see [`keep_for_nearest_call`]). Only a thread of Python that runs such a call reads them,
/// which a thread of the runtime is not: the failure of the interrupt of a signal, and of a
/// background read of the responses of an execution, keep the context that they have.
pub(crate) fn before_failure_requests(
    client: &SparkConnectClient,
    failure: &SparkError,
) -> Result<bool, SparkError> {
    let Some(refresher) = client.user_context_refresher() else {
        return Ok(true);
    };
    let of_a_call = CALLS.with(|calls| {
        calls
            .try_borrow()
            .is_ok_and(|calls| calls.iter().any(|call| is_of(call, &refresher)))
    });
    if !of_a_call {
        return Ok(true);
    }
    // SAFETY: a read of the thread state of the current thread, which needs no lock.
    if unsafe { pyo3::ffi::PyGILState_GetThisThreadState() }.is_null() {
        return Ok(true);
    }
    Python::try_attach(
        |py| match reread_nearest_call(py, &refresher, client, ReadOf::Requests) {
            Ok(()) => Ok(true),
            Err(raised) if raised.is_instance_of::<pyo3::exceptions::PyException>(py) => {
                // Of the error of the server as its context, as PySpark raises the error of the
                // build while it handles the failure.
                let context = crate::errors::spark_error_to_py_exception(failure.clone());
                let raised = crate::errors::with_context(py, raised, context);
                let unbuilt = crate::errors::callback_error(raised);
                keep_failure_error(&refresher, &unbuilt);
                Err(unbuilt)
            }
            Err(raised) => {
                keep_for_nearest_call(py, &refresher, raised);
                Ok(false)
            }
        },
    )
    .unwrap_or(Ok(true))
}

/// Keep `failure`, the error of a read for the requests of a failure of the nearest call of the
/// session whose refresher is `refresher` that runs on the calling thread, which the failure
/// carries, for that call: the call releases its exception once it ended, unless it raises it
/// (see [`Ended::release_failure_errors`]).
fn keep_failure_error(refresher: &Arc<dyn std::any::Any + Send + Sync>, failure: &SparkError) {
    CALLS.with(|calls| {
        if let Ok(mut calls) = calls.try_borrow_mut() {
            if let Some(call) = calls.iter_mut().rev().find(|call| is_of(call, refresher)) {
                call.failure_errors.push(failure.clone());
            }
        }
    });
}

/// `error`, the `KeyboardInterrupt` that a handler of the progress of the execution of the
/// operation `operation_id`, which the nearest call of `session` that runs on the calling thread
/// makes, raised, once the operation is interrupted on the server: as the `except
/// KeyboardInterrupt` of PySpark's execution calls `interrupt_operation` of the client of the
/// execution, the copy of a client whose method made it or the session's. The interrupt is a call
/// of that client, whose request reads its attributes as they are now: an error of the call, as
/// of a value of them that protobuf refuses, which sends no interrupt, replaces `error`, which
/// becomes its context.
pub(crate) fn interrupt_for_handler(
    py: Python<'_>,
    session: &SparkSession,
    operation_id: &str,
    error: PyErr,
) -> PyErr {
    // As PySpark's `except KeyboardInterrupt`, which logs the interrupt before it interrupts the
    // operation: the exception of the record replaces the KeyboardInterrupt, which becomes its
    // context, and nothing is interrupted.
    if let Err(raised) = crate::client_log::interrupt_requested(operation_id) {
        let raised = crate::errors::spark_error_to_py_exception(raised);
        return crate::errors::with_context(py, raised, error);
    }
    let refresher = session.user_context_refresher();
    let refresh = refresher
        .as_ref()
        .and_then(|refresher| crate::state::refresh_of(refresher.as_ref()));
    let token = refresher.as_ref().and_then(|refresher| {
        CALLS.with(|calls| {
            let calls = calls.try_borrow().ok()?;
            let call = calls.iter().rev().find(|call| is_of(call, refresher))?;
            call.copy_token.as_ref().map(|token| token.clone_ref(py))
        })
    });
    // The calls of the thread are of the copy whose method made the execution, while the interrupt
    // is made, as the calls of a method of the copy.
    let entered = match (refresh, token) {
        (Some(refresh), Some(token)) => refresh
            .bind(py)
            .call_method1("_enter_call_of", (token,))
            .ok(),
        _ => None,
    };
    // Of the operation's identifier.
    let interrupted =
        interruptible_without_parts(py, session, || session.interrupt_operation(operation_id));
    if let (Some(refresh), Some(entered)) = (refresh, entered) {
        if let Err(left) = refresh.bind(py).call_method1("_leave_call_of", (entered,)) {
            left.write_unraisable(py, None);
        }
    }
    match interrupted {
        Ok(_) => error,
        Err(failure) => crate::errors::with_context(py, failure, error),
    }
}

/// The wait of [`interruptible_outcome`], of whose results a failed interrupt is the inner one
/// when `catch_interrupt_failure`, after the request context of the calls of `context` is brought
/// up to date (see [`refresh_request_context`]). As PySpark's client, an execution of a thread of
/// no tags gives it an empty set of them, as its `get_tags` does when it builds the request, and
/// a failure of `f` that the client handles leaves `inside_error_handling` False in the
/// `thread_local` of the client, as its `_handle_error` does. A call that runs during another
/// call of the thread, as one of a progress handler or of a hook of the executions, gives the
/// thread the context of its own requests until it ends, when the thread has the context of the
/// other call again, of the attributes of its client as they are then (see [`Call`]), which the
/// requests that the other call sends after it carry, as PySpark's requests read the attributes
/// of their own client.
fn run_interruptible<T: Send, E: Send + Into<Box<SparkError>>>(
    py: Python<'_>,
    session: &SparkSession,
    f: impl FnOnce() -> Result<T, E> + Send,
    catch_interrupt_failure: bool,
    context: RequestContext,
) -> PyResult<Result<T, Box<SparkError>>> {
    // How its wait without the GIL is counted for the garbage collector, which a caller of the
    // binding sets for it alone, not for the Python code that the call runs (see `crate::gc`).
    let counting = crate::gc::take_counting();
    let call = Call::enter(session, context);
    let refreshed = match refresh_request_context(py, session, context) {
        Ok(refreshed) => refreshed,
        Err(error) => return Err(call.end_unrefreshed(py, error)),
    };
    // The levels of the logs of the call, and the call stack of its requests, as PySpark's
    // requests read them as they are built. A call that runs during another call of the thread, as
    // one of a progress handler or of a hook, carries the call stack of that call, as PySpark's
    // ends at the first frame of the pyspark package, of the call that runs it.
    let called = crate::client_log::take_called();
    let started = match enclosing_call_stack(true) {
        Some(call_stack) => {
            crate::client_log::start_nested_call(py, session, call_stack.clone());
            Ok(call_stack)
        }
        None => crate::client_log::start_call(py, session, called),
    };
    match started {
        Ok(call_stack) => keep_call_stack(call_stack),
        Err(error) => return Err(call.end_unrefreshed(py, error)),
    }
    call.of_copy(py, refreshed.copy_token.as_ref());
    // As PySpark's requests, which read `SPARK_CONNECT_USER_AGENT` of `os.environ` as each is
    // built: of the environment as the call starts, read under the GIL, as Python changes it.
    session.refresh_user_agent();
    let mut outcome = wait_interruptible(py, session, f, catch_interrupt_failure, counting);
    // As PySpark's `_handle_error`, which raises the error of a call as it is while the thread
    // handles another failure of the client (`inside_error_handling`), as of a call that a handler
    // of a log of that conversion makes: before it sets the operation ID of an execution on it,
    // and before its `finally` leaves `inside_error_handling` False, which it leaves True.
    let inside_error_handling = context.handles_errors() && session.inside_error_handling();
    if inside_error_handling {
        if let Ok(Err(failure)) = &mut outcome {
            failure.operation_id = None;
        }
    }
    let read_tags = refreshed.without_tags && session.has_thread_tags();
    let handled_error =
        context.handles_errors() && !inside_error_handling && matches!(outcome, Ok(Err(_)));
    let after = after_call(
        py,
        session,
        read_tags,
        handled_error,
        refreshed.copy_token,
        refreshed.snapshot,
    );
    let mut ended = call.end(py);
    ended.release_failure_errors(match &outcome {
        Ok(Err(failure)) => Some(failure),
        _ => None,
    });
    let raised = match ended.raised.take() {
        // An exception that a read for the call raised, which its wait did not raise, in place of
        // its outcome; the exception of a signal that ended the wait is raised all the same.
        Some(raised) => match outcome {
            Err(signal) => {
                raised.write_unraisable(py, None);
                Err(signal)
            }
            outcome => Err(replacing(py, raised, outcome)),
        },
        None => outcome,
    };
    // As PySpark's: the exception of the reset of the state of the call, which its `finally`
    // makes, replaces the one that the call raises, which becomes its context; the exception of
    // the read at the end of a call that ran during another, which arrives last, replaces both.
    let raised = match after {
        Ok(()) => raised,
        Err(after) => Err(replacing(py, after, raised)),
    };
    match ended.reread.take() {
        Some(reread) => Err(replacing(py, reread, raised)),
        None => raised,
    }
}

/// `raised`, which replaces `result`, the result of a call: the error that the call would raise
/// is its context, as Python chains an exception raised while another is handled.
fn replacing<T>(
    py: Python<'_>,
    raised: PyErr,
    result: PyResult<Result<T, Box<SparkError>>>,
) -> PyErr {
    match result {
        Ok(Ok(_)) => raised,
        Ok(Err(failure)) => crate::errors::with_context(
            py,
            raised,
            crate::errors::spark_error_to_py_exception(*failure),
        ),
        Err(error) => crate::errors::with_context(py, raised, error),
    }
}

/// The wait of [`run_interruptible`].
fn wait_interruptible<T: Send, E: Send + Into<Box<SparkError>>>(
    py: Python<'_>,
    session: &SparkSession,
    f: impl FnOnce() -> Result<T, E> + Send,
    catch_interrupt_failure: bool,
    counting: crate::gc::Counting,
) -> PyResult<Result<T, Box<SparkError>>> {
    let watch = Arc::new(SignalWatch {
        session: session.clone(),
        interrupted: AtomicBool::new(false),
        error: Mutex::new(None),
        interrupt_failure: Arc::new(Mutex::new(None)),
        interrupt_unbuilt: Mutex::new(None),
        converting: Mutex::new(None),
    });
    let poll: Arc<WaitWatch> = {
        let watch = Arc::clone(&watch);
        Arc::new(move || watch.poll())
    };
    let output = crate::client_log::of_its_own_logs(|| {
        crate::gc::run_detached(py, counting, || with_wait_watch(poll, f))
    });
    // Of a handler of a signal, or a KeyboardInterrupt that Python code of the client raised in
    // the call after the last check of the wait, as PySpark raises it out of the call there.
    let signal = watch
        .error
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take()
        .or_else(crate::deliver::take_interrupt);
    let interrupt_failure = watch
        .interrupt_failure
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take();
    let interrupt_unbuilt = watch
        .interrupt_unbuilt
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take();
    // As PySpark's: the exception of a signal while the failure of the call was converted, in
    // the `except Exception` clause of the call, has grpcio's error of the failure as its context,
    // and the operation ID of the execution of the failure, which the `except BaseException` of
    // its `_handle_error` sets on any exception that goes out of the conversion.
    let converting = watch
        .converting
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take();
    let signal = match (signal, converting) {
        (Some(signal), Some(conversion)) => {
            let signal = match crate::client::rpc_error_of(py, &conversion.failure) {
                Some(context) => crate::errors::with_context(py, signal, context),
                None => signal,
            };
            crate::errors::stamp_operation_id(
                signal.value(py).as_any(),
                conversion.operation_id.as_deref(),
            );
            Some(signal)
        }
        (signal, _) => signal,
    };
    let output: Result<T, Box<SparkError>> = output.map_err(Into::into);
    // As the `except KeyboardInterrupt` of PySpark's execution, of a KeyboardInterrupt that Python
    // code of the client raised in it, out of the iterator of its responses, as a handler of the
    // logs of a response or a retry policy raises the exception of a signal: its record, and then
    // the interrupt of the operation, of the attributes of the client of the call then. A signal
    // that the wait took is raised as such.
    if signal.is_none() && interrupt_unbuilt.is_none() {
        if let Err(err) = &output {
            if let Some(operation_id) = err.operation_id.clone() {
                if crate::errors::is_callback_keyboard_interrupt(py, err) {
                    if let Some(interrupt) = crate::errors::callback_exception(err) {
                        return Err(interrupt_for_handler(py, session, &operation_id, interrupt));
                    }
                }
            }
        }
    }
    // As PySpark's: the error of the build of the request of the interrupt replaces the
    // KeyboardInterrupt, which becomes its context, as `interrupt_operation` raises out of the
    // `except` clause, and no interrupt is sent.
    if let Some(unbuilt) = interrupt_unbuilt {
        let unbuilt = match signal {
            Some(error) => crate::errors::with_context(py, unbuilt, error),
            None => unbuilt,
        };
        if catch_interrupt_failure && unbuilt.is_instance_of::<pyo3::exceptions::PyException>(py) {
            return Ok(Err(Box::new(crate::errors::callback_error(unbuilt))));
        }
        return Err(unbuilt);
    }
    match signal {
        Some(error) => {
            // As PySpark's: the done call of a progress handler that raised while the signal ended
            // the call replaces the KeyboardInterrupt, which becomes its context, as the done call
            // runs in the `except KeyboardInterrupt` of PySpark's execution. Its exception is
            // released from the core's callbacks, so that it is not left behind.
            if let Err(err) = &output {
                if let Some(done) = crate::errors::callback_exception(err) {
                    // The context is set on the instance, as the implicit chaining of PySpark's
                    // `except` clause sets it.
                    let _ = done.value(py).setattr("__context__", error.value(py));
                    return Err(done);
                }
            }
            // As PySpark's: the error of the interrupt replaces the KeyboardInterrupt, which
            // becomes its context, as `interrupt_operation` raises out of the `except` clause.
            if let Some(failure) = interrupt_failure {
                if catch_interrupt_failure {
                    return Ok(Err(Box::new(failure)));
                }
                let failure = crate::errors::spark_error_to_py_exception(failure);
                return Err(crate::errors::with_context(py, failure, error));
            }
            Err(error)
        }
        None => Ok(output),
    }
}
