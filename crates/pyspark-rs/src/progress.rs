//! The progress handlers of the Python client, as PySpark Connect's client keeps them: the
//! `_progress_handlers` list of the client of each session, whose handlers are called with the
//! keyword arguments of `ProgressHandler` and the `StageInfo` of
//! `pyspark.sql.connect.shell.progress`, and the progress bars of
//! `SPARK_CONNECT_PROGRESS_BAR_ENABLED`, of its `Progress`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use spark_connect::progress::ProgressUpdate;
use spark_connect::session::WeakSparkSession;
use spark_connect::SparkSession;
use spark_connect_core::error::SparkError;

/// The progress bars of the executions of a session, and the number of the progress handlers of
/// its client.
pub(crate) struct SessionProgress {
    /// The `Progress` of each execution of a progress bar, by its operation ID.
    bars: Py<PyDict>,
    /// The number of the handlers of the `_progress_handlers` of the client of the session as its
    /// last call started, or as its methods changed them, so that an execution takes the GIL only
    /// for a handler or a bar (see [`set_handler_count`]).
    count: AtomicUsize,
}

/// The progress of each session of the client, by its session ID: every Python object of a
/// session has its handlers.
fn registry() -> &'static Mutex<HashMap<String, Arc<SessionProgress>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, Arc<SessionProgress>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

fn lock_registry() -> std::sync::MutexGuard<'static, HashMap<String, Arc<SessionProgress>>> {
    registry().lock().unwrap_or_else(|e| e.into_inner())
}

/// The progress of `session`: the first time, it registers the handler of the core session
/// that calls the Python handlers and drives the progress bars.
pub(crate) fn session_progress(py: Python<'_>, session: &SparkSession) -> Arc<SessionProgress> {
    let id = session.session_id().to_string();
    if let Some(progress) = lock_registry().get(&id) {
        return Arc::clone(progress);
    }
    // The Python objects are made before the registry is locked: making one can run the
    // garbage collector, and with it any Python code, which must not find the registry locked.
    let progress = Arc::new(SessionProgress {
        bars: PyDict::new(py).unbind(),
        count: AtomicUsize::new(0),
    });
    let mut registry = lock_registry();
    if let Some(progress) = registry.get(&id) {
        // Another thread registered the progress of the session meanwhile.
        return Arc::clone(progress);
    }
    let shared = Arc::clone(&progress);
    // A weak handle, which the handler of the core session does not keep the session alive of.
    let interrupted = session.downgrade();
    session.register_fallible_progress_handler(move |update| shared.notify(update, &interrupted));
    install_unreturned_error_handler();
    registry.insert(id, Arc::clone(&progress));
    progress
}

/// Have the core hand over the exceptions of the handlers that no caller is left to raise: that
/// of the done call of an execution dropped before its end goes to `sys.unraisablehook`, as
/// PySpark's generator reports it, and that of an execution that the exception of its done call
/// replaced is the `__context__` of the one the done call raises, as PySpark's `__exit__` does.
/// The exception that propagates while the execution is dropped is kept, as the finalization of
/// PySpark's generator keeps it (see `crate::deliver::preserving_pending_exception`).
fn install_unreturned_error_handler() {
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| {
        spark_connect::progress::set_unreturned_error_handler(|error| {
            use spark_connect::progress::UnreturnedError;
            Python::attach(|py| {
                crate::deliver::preserving_pending_exception(py, |py| match error {
                    UnreturnedError::DroppedDone(error) => {
                        // The generator of toLocalIterator may be freed by now, so no reference to
                        // it is left: its type is the object of the hook. Only that generator
                        // drops an execution that calls the handlers unfinished; the other
                        // iterators do not report progress.
                        let error = crate::errors::spark_error_to_py_exception(error);
                        match py.import("types").and_then(|t| t.getattr("GeneratorType")) {
                            Ok(generator) => error.write_unraisable(py, Some(&generator)),
                            Err(_) => error.write_unraisable(py, None),
                        }
                    }
                    UnreturnedError::Replaced { error, done } => {
                        let context = crate::errors::spark_error_to_py_exception(error);
                        crate::errors::set_callback_context(py, done, context);
                    }
                })
            })
        });
    });
}

/// `error`, the exception of a handler of a progress of an execution of `session`, once the
/// execution is interrupted on the server when it is a `KeyboardInterrupt` of a progress before the
/// done one: as the `except KeyboardInterrupt` of PySpark's execution, around the calls of its
/// handlers, interrupts the operation before it raises it, of the attributes of the client of the
/// execution then, and the error of the interrupt replaces it, which becomes its context (see
/// `interrupt::interrupt_for_handler`).
fn interrupted_by_handler(
    py: Python<'_>,
    session: &WeakSparkSession,
    update: &ProgressUpdate,
    error: PyErr,
) -> PyErr {
    if update.done || !error.is_instance_of::<pyo3::exceptions::PyKeyboardInterrupt>(py) {
        return error;
    }
    let (Some(session), Some(operation_id)) = (session.upgrade(), update.operation_id.as_deref())
    else {
        return error;
    };
    crate::interrupt::interrupt_for_handler(py, &session, operation_id, error)
}

/// Make `count`, the number of the progress handlers of the client of `session` as a call starts
/// (see `crate::interrupt::apply_call_settings`), or as its methods changed them, that of the
/// handlers that its executions call: of none, they do not take the GIL for them.
pub(crate) fn set_handler_count(py: Python<'_>, session: &SparkSession, count: usize) {
    if count == 0 && !lock_registry().contains_key(session.session_id()) {
        return;
    }
    session_progress(py, session)
        .count
        .store(count, Ordering::SeqCst);
}

/// The handlers of the `_progress_handlers` of the client of `session` as they are now, of the
/// state of its clients (see `_RequestContext` of `pyspark.sql.connect.client.core`), or of the
/// copy of a client whose method made the call of the execution, as PySpark's `Progress` reads
/// the list of the client of the execution at each progress: none of a session that is gone or has
/// none, and the `AttributeError` of a client that has no `_progress_handlers`.
fn client_handlers<'py>(
    py: Python<'py>,
    session: &WeakSparkSession,
) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let Some(session) = session.upgrade() else {
        return Ok(Vec::new());
    };
    let Some(state) = crate::interrupt::calling_copy_state(py, &session)
        .or_else(|| crate::state::client_state_of(py, &session))
    else {
        return Ok(Vec::new());
    };
    let handlers = match state.get_item("_progress_handlers")? {
        Some(handlers) => handlers,
        None => {
            return Err(pyo3::exceptions::PyAttributeError::new_err(
                "'SparkConnectClient' object has no attribute '_progress_handlers'",
            ))
        }
    };
    handlers.try_iter()?.collect()
}

/// Forget the progress of a stopped session.
pub(crate) fn forget_session(session: &SparkSession) {
    lock_registry().remove(session.session_id());
}

/// Whether the progress bar is enabled, as `pyspark.sql.connect.shell.progress_bar_enabled`.
fn progress_bar_enabled() -> bool {
    std::env::var("SPARK_CONNECT_PROGRESS_BAR_ENABLED").as_deref() == Ok("1")
}

impl SessionProgress {
    /// Call the handlers with `update`. The exception of a handler fails the execution that
    /// called it, and is raised in its caller, as PySpark's result loop raises it; that of the
    /// done call too, as PySpark's `Progress.__exit__` raises it from the execution. The done call
    /// of an execution that is dropped while an exception propagates keeps that exception, as the
    /// finalization of PySpark's generator does (see `crate::deliver::preserving_pending_exception`).
    fn notify(
        &self,
        update: &ProgressUpdate,
        session: &WeakSparkSession,
    ) -> Result<(), SparkError> {
        let bar = progress_bar_enabled();
        if self.count.load(Ordering::SeqCst) == 0 && !bar {
            return Ok(());
        }
        Python::attach(|py| {
            crate::deliver::preserving_pending_exception(py, |py| {
                self.notify_python(py, update, bar, session)
                    .map_err(|error| {
                        crate::errors::callback_error(interrupted_by_handler(
                            py, session, update, error,
                        ))
                    })
            })
        })
    }

    fn notify_python(
        &self,
        py: Python<'_>,
        update: &ProgressUpdate,
        bar: bool,
        session: &WeakSparkSession,
    ) -> PyResult<()> {
        let module = py.import("pyspark.sql.connect.shell.progress")?;
        let stage_info = module.getattr("StageInfo")?;
        let stages = PyList::empty(py);
        for stage in &update.stages {
            let kwargs = PyDict::new(py);
            kwargs.set_item("stage_id", stage.stage_id)?;
            kwargs.set_item("num_tasks", stage.num_tasks)?;
            kwargs.set_item("num_completed_tasks", stage.num_completed_tasks)?;
            kwargs.set_item("num_bytes_read", stage.num_bytes_read)?;
            kwargs.set_item("done", stage.done)?;
            stages.append(stage_info.call((), Some(&kwargs))?)?;
        }
        if bar {
            // The `Progress` of the execution, which prints its bar, as PySpark's client makes
            // one of each request.
            let bars = self.bars.bind(py);
            let key = update.operation_id.as_deref();
            let progress = match bars.get_item(key)? {
                Some(progress) => progress,
                None => {
                    let kwargs = PyDict::new(py);
                    kwargs.set_item("operation_id", key)?;
                    let progress = module.getattr("Progress")?.call((), Some(&kwargs))?;
                    bars.set_item(key, &progress)?;
                    progress
                }
            };
            if update.done {
                bars.del_item(key)?;
                progress.call_method0("finish")?;
            } else {
                progress.call_method1("update_ticks", (&stages, update.inflight_tasks, key))?;
            }
        }
        // The handlers of the list of the client as it is now, as PySpark's `Progress` iterates the
        // list of its client at each progress.
        let handlers = if self.count.load(Ordering::SeqCst) == 0 {
            Vec::new()
        } else {
            client_handlers(py, session)?
        };
        if handlers.is_empty() {
            return Ok(());
        }
        let kwargs = PyDict::new(py);
        kwargs.set_item("stages", &stages)?;
        kwargs.set_item("inflight_tasks", update.inflight_tasks)?;
        kwargs.set_item("operation_id", update.operation_id.as_deref())?;
        kwargs.set_item("done", update.done)?;
        for handler in handlers {
            handler.call((), Some(&kwargs))?;
        }
        Ok(())
    }
}
