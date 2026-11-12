//! Waits on the server that a signal of the process interrupts, as PySpark's waits are.
//!
//! The wait runs on the calling thread, as PySpark's does, so the progress handlers and the
//! other callbacks of the execution run there too. The wait polls the signals of the process,
//! which only the main thread of Python runs the handlers of. A handler that raises interrupts the
//! execution in flight of the calling thread on the server; the wait then ends with the server's
//! cancellation, and the exception of the handler is raised.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use pyo3::prelude::*;
use spark_connect::session::SparkSession;
use spark_connect_core::error::SparkError;
use spark_connect_core::runtime::{with_wait_watch, WaitWatch, Watch};

use crate::errors::ResultExt;

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
                // Takes the GIL for the check only, as the wait runs without it.
                match Python::attach(|py| py.check_signals()) {
                    Ok(()) => return Watch::Continue,
                    Err(signal) => *error = Some(signal),
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
/// it waits for the server. When a handler raises, the execution that the calling thread waits for
/// is interrupted, and the exception of the handler is raised once `f` has ended. Otherwise the
/// result of `f` is returned.
///
/// The execution is interrupted only when a wait of the calling thread is for one. The waits
/// that a signal ends at once, as PySpark's do (a retry sleep, or a call that the server does not
/// answer): the retries and the RPCs of `f` that are not answered when the first signal arrives
/// during an execution, and all of them when no execution is in flight, or after a second
/// signal. A streaming execution is still read after the first one, until the server cancels it.
/// A wait of `block_on_abortable` (the release of `stop()`) ends at once as well.
///
/// An execution that `f` ran then raises for the metrics of an Observation that PySpark does
/// not convert, as PySpark's does (`ObservedMetricsCheck`).
pub(crate) fn interruptible<T: Send, E: Send + Into<Box<SparkError>>>(
    py: Python<'_>,
    session: &SparkSession,
    f: impl FnOnce() -> Result<T, E> + Send,
) -> PyResult<T> {
    let observed = crate::observation::ObservedMetricsCheck::before(py, session);
    let output = interruptible_outcome(py, session, f)?.to_pyerr()?;
    if let Some(observed) = observed {
        observed.after(py, session)?;
    }
    Ok(output)
}

/// As [`interruptible`], but a failure of `f` is the inner result, which the caller handles, as
/// a warning. The outer error is the exception of a signal handler only, which is never handled
/// as a failure of the call: as PySpark's `except Exception` does not catch it.
pub(crate) fn interruptible_outcome<T: Send, E: Send + Into<Box<SparkError>>>(
    py: Python<'_>,
    session: &SparkSession,
    f: impl FnOnce() -> Result<T, E> + Send,
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
                let failure = crate::errors::spark_error_to_py_exception(failure);
                return Err(crate::errors::with_context(py, failure, error));
            }
            Err(error)
        }
        None => Ok(output),
    }
}
