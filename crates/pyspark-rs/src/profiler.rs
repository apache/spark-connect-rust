//! The UDF profiles of the Python client, as PySpark Connect's client collects them: the
//! `ConnectProfilerCollector` of the `_profiler_collector` of the client of each session, which the
//! pickled updates of the Python accumulators of its executions update, and of which
//! `spark.profile` is a `Profile`.

use std::collections::HashSet;
use std::sync::{Mutex, OnceLock};

use pyo3::prelude::*;
use pyo3::types::PyBytes;
use spark_connect::SparkSession;

/// The sessions whose updates of the Python accumulators of their executions update the
/// `_profiler_collector` of their client, by their session IDs (see
/// [`install_collector_updates`]).
fn registry() -> &'static Mutex<HashSet<String>> {
    static REGISTRY: OnceLock<Mutex<HashSet<String>>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(HashSet::new()))
}

/// Make the pickled updates of the Python accumulators of the executions of `session` update the
/// `ConnectProfilerCollector` of the `_profiler_collector` of its client as it is then, as
/// PySpark's client updates its `_profiler_collector` of each update of a response, once for the
/// session.
pub(crate) fn install_collector_updates(session: &SparkSession) {
    let id = session.session_id().to_string();
    if !registry()
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(id)
    {
        return;
    }
    let weak = session.downgrade();
    session.set_python_accumulator_handler(move |update| {
        Python::attach(|py| {
            if let Err(error) = update_client_collector(py, &weak, update) {
                error.write_unraisable(py, None);
            }
        })
    });
}

/// Update the `_profiler_collector` of the client of `session` with `update` (see
/// [`update_collector`]), or of the copy of a client whose method made the call of the execution,
/// of the `AttributeError` of a client that has none, as PySpark's update reads it.
fn update_client_collector(
    py: Python<'_>,
    session: &spark_connect::session::WeakSparkSession,
    update: &[u8],
) -> PyResult<()> {
    let Some(session) = session.upgrade() else {
        return Ok(());
    };
    let Some(state) = crate::interrupt::calling_copy_state(py, &session)
        .or_else(|| crate::state::client_state_of(py, &session))
    else {
        return Ok(());
    };
    let collector = state.get_item("_profiler_collector")?.ok_or_else(|| {
        pyo3::exceptions::PyAttributeError::new_err(
            "'SparkConnectClient' object has no attribute '_profiler_collector'",
        )
    })?;
    update_collector(py, &collector, update)
}

/// Forget the updates of the profiles of a stopped session.
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
