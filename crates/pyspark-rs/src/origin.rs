//! As PySpark's `_with_origin`: the origin in user code of the expressions that an API function
//! builds, which the server reports in the `DataFrameQueryContext` of an error in them.

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use spark_connect::origin::{set_current, Origin, OriginGuard};

static UTILS: PyOnceLock<Py<PyModule>> = PyOnceLock::new();

/// While the returned guard lives, the expressions that this thread builds have the origin of
/// `fragment`, the name of the API function, at its call site in user code, when DataFrame
/// debugging is enabled. As PySpark's `_with_origin`, with the functions of
/// `pyspark.errors.utils` that it calls: `is_debugging_enabled`, which reads
/// `spark.python.sql.dataFrameDebugging.enabled` once, and `_capture_call_site`. An origin that
/// cannot be read is not set, as an expression is the same without one.
pub(crate) fn with_origin(fragment: &str) -> Option<OriginGuard> {
    Python::attach(|py| -> PyResult<Option<OriginGuard>> {
        let utils = UTILS
            .get_or_try_init(py, || py.import("pyspark.errors.utils").map(Bound::unbind))?
            .bind(py);
        if !utils.call_method0("is_debugging_enabled")?.is_truthy()? {
            return Ok(None);
        }
        let call_site: String = utils.call_method1("_capture_call_site", (1,))?.extract()?;
        Ok(Some(set_current(Origin {
            fragment: fragment.to_string(),
            call_site,
        })))
    })
    .unwrap_or(None)
}
