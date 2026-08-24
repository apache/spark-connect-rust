//! PyO3 wrapper for the Spark profiler collector.

use pyo3::prelude::*;
use spark_connect::profiler::ProfilerCollector;
use std::sync::Arc;

use crate::errors::ResultExt;

/// Python wrapper for the ProfilerCollector.
///
/// Exposed as `SparkSession.profile` to access profiler results.
#[pyclass(name = "ProfilerCollector")]
pub struct PyProfilerCollector {
    pub(crate) inner: Arc<ProfilerCollector>,
}

impl PyProfilerCollector {
    pub fn new(inner: Arc<ProfilerCollector>) -> Self {
        PyProfilerCollector { inner }
    }
}

#[pymethods]
impl PyProfilerCollector {
    /// Show the profile results for a given profiler ID, or all profiles if ID is None.
    ///
    /// Returns a formatted string representation of the profile data.
    #[pyo3(signature = (id=None))]
    fn show(&self, id: Option<i64>) -> String {
        self.inner.show(id)
    }

    /// Write profile results to a file.
    ///
    /// For a given profiler ID (or all profiles if ID is None), writes the profile data to the
    /// specified file path.
    #[pyo3(signature = (id=None, path="."))]
    fn dump(&self, py: Python<'_>, id: Option<i64>, path: &str) -> PyResult<()> {
        // Release the GIL during the file I/O operation
        py.detach(|| self.inner.dump(id, path)).to_pyerr()?;
        Ok(())
    }

    /// Clear collected profile results for a given ID, or all profiles if ID is None.
    #[pyo3(signature = (id=None))]
    fn clear(&self, id: Option<i64>) {
        self.inner.clear(id);
    }
}
