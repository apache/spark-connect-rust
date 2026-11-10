//! The progress handlers of the Python client, as PySpark Connect's client keeps them: a list
//! of each session, whose handlers are called with the keyword arguments of `ProgressHandler`
//! and the `StageInfo` of `pyspark.sql.connect.shell.progress`, and the progress bars of
//! `SPARK_CONNECT_PROGRESS_BAR_ENABLED`, of its `Progress`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use spark_connect::progress::ProgressUpdate;
use spark_connect::SparkSession;
use spark_connect_core::error::SparkError;

/// The progress handlers of a session, and the progress bars of its executions.
pub(crate) struct SessionProgress {
    /// The handlers, as the `_progress_handlers` list of PySpark's client.
    handlers: Py<PyList>,
    /// The `Progress` of each execution of a progress bar, by its operation ID.
    bars: Py<PyDict>,
    /// The number of handlers, so that an execution takes the GIL only for a handler or a bar.
    count: AtomicUsize,
}

/// The progress of each session of the client, by its session ID: every Python object of a
/// session has its handlers.
fn registry() -> &'static Mutex<HashMap<String, Arc<SessionProgress>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, Arc<SessionProgress>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The progress of `session`: the first time, it registers the handler of the core session
/// that calls the Python handlers and drives the progress bars.
pub(crate) fn session_progress(py: Python<'_>, session: &SparkSession) -> Arc<SessionProgress> {
    let mut registry = registry().lock().unwrap_or_else(|e| e.into_inner());
    let id = session.session_id().to_string();
    if let Some(progress) = registry.get(&id) {
        return Arc::clone(progress);
    }
    let progress = Arc::new(SessionProgress {
        handlers: PyList::empty(py).unbind(),
        bars: PyDict::new(py).unbind(),
        count: AtomicUsize::new(0),
    });
    let shared = Arc::clone(&progress);
    session.register_fallible_progress_handler(move |update| shared.notify(update));
    install_unreturned_error_handler();
    registry.insert(id, Arc::clone(&progress));
    progress
}

/// Have the core hand over the exceptions of the handlers that no caller is left to raise: that
/// of the done call of an execution dropped before its end goes to `sys.unraisablehook`, as
/// PySpark's generator reports it, and that of an execution that the exception of its done call
/// replaced is the `__context__` of the one the done call raises, as PySpark's `__exit__` does.
fn install_unreturned_error_handler() {
    static INSTALL: std::sync::Once = std::sync::Once::new();
    INSTALL.call_once(|| {
        spark_connect::progress::set_unreturned_error_handler(|error| {
            use spark_connect::progress::UnreturnedError;
            Python::attach(|py| match error {
                UnreturnedError::DroppedDone(error) => {
                    // The generator of toLocalIterator may be freed by now, so no reference to it
                    // is left: its type is the object of the hook. Only that generator drops an
                    // execution that calls the handlers unfinished; the other iterators do not
                    // report progress.
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
        });
    });
}

/// Forget the progress of a stopped session.
pub(crate) fn forget_session(session: &SparkSession) {
    registry()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(session.session_id());
}

/// Whether the progress bar is enabled, as `pyspark.sql.connect.shell.progress_bar_enabled`.
fn progress_bar_enabled() -> bool {
    std::env::var("SPARK_CONNECT_PROGRESS_BAR_ENABLED").as_deref() == Ok("1")
}

impl SessionProgress {
    /// As PySpark's `register_progress_handler`: a handler is registered once.
    pub(crate) fn register(&self, py: Python<'_>, handler: &Bound<'_, PyAny>) -> PyResult<()> {
        let handlers = self.handlers.bind(py);
        if !handlers.contains(handler)? {
            handlers.append(handler)?;
        }
        self.count.store(handlers.len(), Ordering::SeqCst);
        Ok(())
    }

    /// As PySpark's `remove_progress_handler`: a `ValueError` for a handler that is not
    /// registered, of `list.remove`.
    pub(crate) fn remove(&self, py: Python<'_>, handler: &Bound<'_, PyAny>) -> PyResult<()> {
        let handlers = self.handlers.bind(py);
        handlers.call_method1("remove", (handler,))?;
        self.count.store(handlers.len(), Ordering::SeqCst);
        Ok(())
    }

    /// As PySpark's `clear_progress_handlers`.
    pub(crate) fn clear(&self, py: Python<'_>) -> PyResult<()> {
        self.handlers.bind(py).call_method0("clear")?;
        self.count.store(0, Ordering::SeqCst);
        Ok(())
    }

    /// Call the handlers with `update`. The exception of a handler fails the execution that
    /// called it, and is raised in its caller, as PySpark's result loop raises it; that of the
    /// done call too, as PySpark's `Progress.__exit__` raises it from the execution.
    fn notify(&self, update: &ProgressUpdate) -> Result<(), SparkError> {
        let bar = progress_bar_enabled();
        if self.count.load(Ordering::SeqCst) == 0 && !bar {
            return Ok(());
        }
        Python::attach(|py| {
            self.notify_python(py, update, bar)
                .map_err(crate::errors::callback_error)
        })
    }

    fn notify_python(&self, py: Python<'_>, update: &ProgressUpdate, bar: bool) -> PyResult<()> {
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
        // A copy of the handlers, as PySpark's `Progress` takes the list of the client.
        let handlers: Vec<Bound<'_, PyAny>> = self.handlers.bind(py).iter().collect();
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
