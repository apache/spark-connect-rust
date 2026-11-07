//! The UDF profiles of the Python client, as PySpark Connect's client collects them: a
//! `ConnectProfilerCollector` of each session, which the pickled updates of the Python
//! accumulators of its executions update, and of which `spark.profile` is a `Profile`.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use spark_connect::SparkSession;

/// The `ConnectProfilerCollector` of each session of the client, by its session ID: every
/// Python object of a session has its profiles.
fn registry() -> &'static Mutex<HashMap<String, Py<PyAny>>> {
    static REGISTRY: OnceLock<Mutex<HashMap<String, Py<PyAny>>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The `ConnectProfilerCollector` of `session`, as the `_profiler_collector` of PySpark's
/// client: the first time, it sets the handler of the core session that updates it.
pub(crate) fn profiler_collector(py: Python<'_>, session: &SparkSession) -> PyResult<Py<PyAny>> {
    let mut registry = registry().lock().unwrap_or_else(|e| e.into_inner());
    let id = session.session_id().to_string();
    if let Some(collector) = registry.get(&id) {
        return Ok(collector.clone_ref(py));
    }
    let collector = py
        .import("pyspark.sql.connect.profiler")?
        .getattr("ConnectProfilerCollector")?
        .call0()?
        .unbind();
    let shared = collector.clone_ref(py);
    session.set_python_accumulator_handler(move |update| {
        Python::attach(|py| {
            if let Err(error) = update_collector(py, shared.bind(py), update) {
                error.write_unraisable(py, None);
            }
        })
    });
    registry.insert(id, collector.clone_ref(py));
    Ok(collector)
}

/// Forget the profiles of a stopped session.
pub(crate) fn forget_session(session: &SparkSession) {
    registry()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(session.session_id());
}

/// Update `collector` with a pickled update of a Python accumulator, as PySpark Connect's
/// client: `(id, update)` of `specialAccumulatorSer`, of which those of the UDF profiler
/// (`SQL_UDF_PROFIER_V2`) update the collector, and the warning of one that does not unpickle.
fn update_collector(py: Python<'_>, collector: &Bound<'_, PyAny>, update: &[u8]) -> PyResult<()> {
    let accumulators = py.import("pyspark.accumulators")?;
    let unpickled = accumulators
        .getattr("specialAccumulatorSer")?
        .call_method1("loads", (PyBytes::new(py, update),));
    let unpickled = match unpickled {
        Ok(unpickled) => unpickled,
        Err(error) if error.is_instance(py, &py.import("pickle")?.getattr("UnpicklingError")?) => {
            // As PySpark's: an unexpected class or function in the metric is ignored.
            py.import("logging")?
                .call_method1("getLogger", ("pyspark.sql.connect.client.core",))?
                .call_method1(
                    "warning",
                    (format!(
                        "Error unpickling accumulator metric: {}",
                        error.value(py)
                    ),),
                )?;
            return Ok(());
        }
        Err(error) => return Err(error),
    };
    let (id, update): (Bound<'_, PyAny>, Bound<'_, PyAny>) = unpickled.extract()?;
    let profiler_v2 = accumulators
        .getattr("SpecialAccumulatorIds")?
        .getattr("SQL_UDF_PROFIER_V2")?;
    if id.eq(profiler_v2)? {
        collector.call_method1("_update", (update,))?;
    }
    Ok(())
}
