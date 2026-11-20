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
        py.check_signals()?;
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
        {
            let mut error = self.error.lock().unwrap_or_else(|e| e.into_inner());
            if error.is_none() {
                // Takes the GIL for the check only, as the wait runs without it. The exception of a
                // handler that is an `Exception` ends the part of the call that PySpark's `except
                // Exception` covers, if one runs, such as the lookup of the plan compression
                // settings, and the call goes on, as PySpark's does (see `runtime::catching`).
                let raised = Python::attach(|py| match py.check_signals() {
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
                });
                match raised {
                    None => return Watch::Continue,
                    Some((signal, keyboard_interrupt)) => {
                        *error = Some(signal);
                        // As PySpark's execution, whose `except KeyboardInterrupt` interrupts the
                        // operation on the server: the exception of another handler ends the
                        // wait at once, as it goes out of the execution as it is.
                        if !keyboard_interrupt {
                            return Watch::Abort;
                        }
                    }
                }
            } else if self.interrupted.load(Ordering::SeqCst) {
                // A failed interrupt ends the wait at once, as PySpark raises the error of the
                // interrupt without waiting for the server to end the execution.
                if self.interrupt_failure_is_set() {
                    return Watch::Abort;
                }
                // After the interrupt, a second signal ends the wait, as PySpark's wait does.
                return match Python::attach(|py| py.check_signals()) {
                    Ok(()) => Watch::Continue,
                    Err(_) => Watch::Abort,
                };
            }
        }
        match self.session.running_operation_id() {
            Some(operation_id) => {
                // The request of the interrupt is of the attributes of the client of the call as
                // they are now, as PySpark's `interrupt_operation` builds it once the
                // KeyboardInterrupt is raised (see `reread_nearest_call`).
                Python::attach(|py| {
                    if let Err(error) = reread_nearest_call_of(py, &self.session) {
                        raise_later(py, error);
                    }
                });
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
pub(crate) fn interruptible<T: Send, E: Send + Into<Box<SparkError>>>(
    py: Python<'_>,
    session: &SparkSession,
    f: impl FnOnce() -> Result<T, E> + Send,
) -> PyResult<T> {
    of_the_request(session, interruptible_outcome(py, session, f)?)
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
/// Without a signal, `f` returns its output.
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
            // Takes the GIL for the check only, as the wait runs without it.
            match Python::attach(|py| py.check_signals()) {
                Ok(()) => Watch::Continue,
                Err(raised) => {
                    *error = Some(raised);
                    Watch::Abort
                }
            }
        })
    };
    match py.detach(|| with_wait_watch(poll, f)) {
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
    of_the_request(
        session,
        run_interruptible(py, session, f, false, RequestContext::Identity)?,
    )
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
    run_interruptible(py, session, f, false, RequestContext::Identity)
}

/// As [`interruptible_without_user_context`], for the `CloneSession` of `cloneSession`, whose
/// failure PySpark's `clone` does not handle (see [`RequestContext::Clone`]).
pub(crate) fn interruptible_of_clone<T: Send, E: Send + Into<Box<SparkError>>>(
    py: Python<'_>,
    session: &SparkSession,
    f: impl FnOnce() -> Result<T, E> + Send,
) -> PyResult<T> {
    run_interruptible(py, session, f, false, RequestContext::Clone)?.to_pyerr()
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
    run_interruptible(py, session, f, false, context)?.to_pyerr()
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

    /// The mode of the refresher of the client (see `pyspark.sql.connect.client._RequestContext`):
    /// 2 of all of the context, raising its errors, 3 as 2 of the client of the session rather
    /// than of a copy whose method runs, 1 of the session ID and the user, raising theirs, and of
    /// the rest of none, 0 of all of it, of none, or no refresh.
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
    let Some(refresh) = refresher.downcast_ref::<Py<PyAny>>() else {
        return Ok(Refreshed::default());
    };
    let Some(mode) = context.mode() else {
        // A call that reads nothing of the client holds the contextvar of the copy of a client
        // whose method makes it, as a call that reads it does.
        let copy_token = refresh.bind(py).call_method0("_hold")?;
        return Ok(Refreshed {
            without_tags: false,
            copy_token: (!copy_token.is_none()).then(|| copy_token.unbind()),
        });
    };
    let refreshed = refresh.bind(py).call1((mode,))?;
    let (session_id, user_id, global, own, tags, copy_token, of_the_session): RefreshedContext<'_> =
        refreshed.extract()?;
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
) -> PyResult<()> {
    if !read_tags && !handled_error && copy_token.is_none() {
        return Ok(());
    }
    if let Some(refresher) = session.user_context_refresher() {
        if let Some(refresh) = refresher.downcast_ref::<Py<PyAny>>() {
            refresh
                .bind(py)
                .call_method1("_after_call", (read_tags, handled_error, copy_token))?;
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
/// collection of the garbage collector, keeps its own (see [`after_finalizer_request`]).
pub(crate) fn before_finalizer_request(session: &SparkSession) -> bool {
    // SAFETY: a read of the thread state of the current thread, which needs no lock.
    if unsafe { pyo3::ffi::PyGILState_GetThisThreadState() }.is_null() {
        return true;
    }
    Python::try_attach(|py| {
        match refresh_request_context(py, session, RequestContext::Finalizer) {
            Ok(refreshed) => {
                // An error of the reset of its state is not the request's.
                let _ = after_call(
                    py,
                    session,
                    refreshed.without_tags,
                    false,
                    refreshed.copy_token,
                );
                true
            }
            Err(_) => false,
        }
    })
    .unwrap_or(true)
}

thread_local! {
    /// The calls of [`run_interruptible`] that run on this thread, each during the one before it,
    /// as the call of a progress handler during the call that it reports on.
    static CALLS: std::cell::RefCell<Vec<CallOfThread>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// A call of [`run_interruptible`] that runs on this thread: the refresher of its session, which
/// tells the clients of the session apart from those of another, and the token of the contextvar
/// of the copy of a client whose method made it, of its refresh.
struct CallOfThread {
    refresher: Option<Arc<dyn std::any::Any + Send + Sync>>,
    copy_token: Option<Py<PyAny>>,
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
    /// Enter a call of `session`.
    fn enter(session: &'a SparkSession) -> Self {
        let call = CallOfThread {
            refresher: session.user_context_refresher(),
            copy_token: None,
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
    /// which it ran, if it ran during another, read again of its client. An error of the read
    /// that is not an `Exception`, as the `KeyboardInterrupt` of a signal handler that ran
    /// meanwhile, is the error of the call.
    fn end(mut self, py: Python<'_>) -> PyResult<()> {
        self.ended = true;
        if self.leave() {
            reread_nearest_call_of(py, self.session)?;
        }
        Ok(())
    }

    /// Take the call off the stack of the thread, and give the thread its context of before the
    /// call again, of a call that ran during another: whether it did.
    fn leave(&mut self) -> bool {
        // The token is released once the stack is no longer borrowed.
        let call = CALLS.with(|calls| calls.borrow_mut().pop());
        drop(call);
        match self.kept.take() {
            Some(kept) => {
                drop(kept);
                true
            }
            None => false,
        }
    }
}

impl Drop for Call<'_> {
    fn drop(&mut self) {
        if self.ended {
            return;
        }
        // A call that ends of an error of its refresh or of the reset of its state, which it
        // raises: the thread has its context of before it again, read again as at the end of
        // any other call, of an error that is not the call's.
        if self.leave() {
            let session = self.session;
            let _ = Python::try_attach(|py| {
                if let Err(error) = reread_nearest_call_of(py, session) {
                    raise_later(py, error);
                }
            });
        }
    }
}

/// Whether `call` is a call of the session whose refresher is `refresher`.
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

/// Give the requests of the calling thread the context of the nearest call of `session` that runs
/// on it, if any, read again of the attributes of its client as they are now (see
/// [`reread_nearest_call`]).
fn reread_nearest_call_of(py: Python<'_>, session: &SparkSession) -> PyResult<()> {
    match session.user_context_refresher() {
        Some(refresher) => reread_nearest_call(py, &refresher, session),
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
/// call, of the token of its refresh, else the session's. The values are read as those of the
/// artifacts are, of none of their errors: a value that cannot be read stays as it was, and so
/// does the whole context when the values cannot be taken. An error of the read that is not an
/// `Exception`, as the `KeyboardInterrupt` of a signal handler that ran meanwhile, is returned.
fn reread_nearest_call(
    py: Python<'_>,
    refresher: &Arc<dyn std::any::Any + Send + Sync>,
    target: &dyn ThreadRequestTarget,
) -> PyResult<()> {
    let token = CALLS.with(|calls| {
        // The stack is only borrowed for a moment, by no code that reads the client.
        let calls = calls.try_borrow().ok()?;
        let call = calls.iter().rev().find(|call| is_of(call, refresher))?;
        Some(call.copy_token.as_ref().map(|token| token.clone_ref(py)))
    });
    let Some(token) = token else {
        return Ok(());
    };
    let Some(refresh) = refresher.downcast_ref::<Py<PyAny>>() else {
        return Ok(());
    };
    let reread = || -> PyResult<()> {
        let values = refresh.bind(py).call_method1("_values_of_call", (token,))?;
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
            .transpose()?;
        target.give_request_context(
            session_id.unwrap_or_else(|| target.current_session_id()),
            user_id.unwrap_or_else(|| target.current_user_id()),
            global.unwrap_or_else(|| target.current_global()),
        );
        if let Some(own) = own {
            target.give_own(own);
        }
        if let Some(tags) = tags {
            target.give_tags(tags);
        }
        Ok(())
    };
    match reread() {
        Err(error) if !error.is_instance_of::<pyo3::exceptions::PyException>(py) => Err(error),
        _ => Ok(()),
    }
}

/// Raise `error`, of a re-read of the context of the requests of a thread that cannot raise it
/// (see [`reread_nearest_call`]), once the thread can: the `KeyboardInterrupt` of a signal handler
/// that ran during the read is raised again at the next check of the signals of the process, as
/// the signal arrived then, and another exception is reported as an unraisable one is.
fn raise_later(py: Python<'_>, error: PyErr) {
    if error.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py) {
        // SAFETY: as a signal that arrives, which needs no lock.
        unsafe { pyo3::ffi::PyErr_SetInterrupt() };
    } else {
        error.write_unraisable(py, None);
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
/// now (see [`reread_nearest_call`]). As [`before_finalizer_request`], a thread that is not one of
/// Python, or an interpreter that shuts down, has none to read, and the errors of the read are not
/// raised, as a finalizer's are not (see [`raise_later`]).
pub(crate) fn after_finalizer_request(session: &SparkSession) {
    // SAFETY: a read of the thread state of the current thread, which needs no lock.
    if unsafe { pyo3::ffi::PyGILState_GetThisThreadState() }.is_null() {
        return;
    }
    let _ = Python::try_attach(|py| {
        if let Err(error) = reread_nearest_call_of(py, session) {
            raise_later(py, error);
        }
    });
}

/// The hook of the requests of the failure of a call of `client`, before the client builds them
/// (see `spark_connect_core::client::set_failure_requests_hook`): the requests of the details of
/// the error and of the configs of its stack trace are of the context of the nearest call of the
/// session of `client` that runs on the calling thread, read again of the attributes of its client
/// as they are now (see [`reread_nearest_call`]), as PySpark's `_handle_error` builds them of
/// them. Only a thread of Python that runs such a call reads them, which a thread of the runtime
/// is not: the failure of the interrupt of a signal, and of a background read of the responses of
/// an execution, keep the context that they have. The errors of the read are not raised (see
/// [`raise_later`]).
pub(crate) fn before_failure_requests(client: &SparkConnectClient) {
    let Some(refresher) = client.user_context_refresher() else {
        return;
    };
    let of_a_call = CALLS.with(|calls| {
        calls
            .try_borrow()
            .is_ok_and(|calls| calls.iter().any(|call| is_of(call, &refresher)))
    });
    if !of_a_call {
        return;
    }
    // SAFETY: a read of the thread state of the current thread, which needs no lock.
    if unsafe { pyo3::ffi::PyGILState_GetThisThreadState() }.is_null() {
        return;
    }
    let _ = Python::try_attach(|py| {
        if let Err(error) = reread_nearest_call(py, &refresher, client) {
            raise_later(py, error);
        }
    });
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
    let call = Call::enter(session);
    let refreshed = refresh_request_context(py, session, context)?;
    call.of_copy(py, refreshed.copy_token.as_ref());
    let outcome = wait_interruptible(py, session, f, catch_interrupt_failure);
    let read_tags = refreshed.without_tags && session.has_thread_tags();
    let handled_error = context.handles_errors() && matches!(outcome, Ok(Err(_)));
    after_call(py, session, read_tags, handled_error, refreshed.copy_token)?;
    call.end(py)?;
    outcome
}

/// The wait of [`run_interruptible`].
fn wait_interruptible<T: Send, E: Send + Into<Box<SparkError>>>(
    py: Python<'_>,
    session: &SparkSession,
    f: impl FnOnce() -> Result<T, E> + Send,
    catch_interrupt_failure: bool,
) -> PyResult<Result<T, Box<SparkError>>> {
    let watch = Arc::new(SignalWatch {
        session: session.clone(),
        interrupted: AtomicBool::new(false),
        error: Mutex::new(None),
        interrupt_failure: Arc::new(Mutex::new(None)),
    });
    let poll: Arc<WaitWatch> = {
        let watch = Arc::clone(&watch);
        Arc::new(move || watch.poll())
    };
    let output = py.detach(|| with_wait_watch(poll, f));
    let signal = watch.error.lock().unwrap_or_else(|e| e.into_inner()).take();
    let interrupt_failure = watch
        .interrupt_failure
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .take();
    let output: Result<T, Box<SparkError>> = output.map_err(Into::into);
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
