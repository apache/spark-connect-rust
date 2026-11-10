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
}

impl SignalWatch {
    fn poll(&self) -> Watch {
        if self.interrupted.load(Ordering::SeqCst) {
            return Watch::Continue;
        }
        {
            let mut error = self.error.lock().unwrap_or_else(|e| e.into_inner());
            if error.is_none() {
                // Takes the GIL for the check only, as the wait runs without it.
                match Python::attach(|py| py.check_signals()) {
                    Ok(()) => return Watch::Continue,
                    Err(signal) => *error = Some(signal),
                }
            }
        }
        // The signal stays captured while no execution that the wait is for is in flight, so
        // that the first one the call starts is interrupted.
        match self.session.running_operation_id() {
            Some(operation_id) => {
                self.interrupted.store(true, Ordering::SeqCst);
                let session = self.session.clone();
                Watch::Interrupt(Box::pin(async move {
                    // A failed interrupt does not replace the exception of the signal.
                    let _ = session.interrupt_operation_async(&operation_id).await;
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
/// The execution is interrupted only when a wait of the calling thread is for one. A signal that
/// arrives while none is (such as during the analysis of a plan, or the plan compression before
/// an execution) is kept, and the first execution that `f` then waits for is interrupted; a
/// request that never starts one ends when its response arrives, and the exception is raised
/// then. A wait of `block_on_abortable` (the release of `stop()`) ends at once instead.
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
    });
    let poll: Arc<WaitWatch> = {
        let watch = Arc::clone(&watch);
        Arc::new(move || watch.poll())
    };
    let output = py.detach(|| with_wait_watch(poll, f));
    let signal = watch.error.lock().unwrap_or_else(|e| e.into_inner()).take();
    match signal {
        Some(error) => Err(error),
        None => Ok(output.map_err(Into::into)),
    }
}
