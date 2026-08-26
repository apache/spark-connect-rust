//! PyO3 wrapper for `spark_connect::observation::Observation` (`pyspark.sql.Observation`).

use std::sync::atomic::{AtomicU64, Ordering};

use pyo3::prelude::*;
use spark_connect::observation::Observation;

static OBS_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Python wrapper for a named metric Observation.
#[pyclass(name = "Observation")]
pub struct PyObservation {
    pub(crate) inner: Observation,
}

#[pymethods]
impl PyObservation {
    /// `Observation(name=None)` - an unnamed observation gets an auto-generated name,
    /// matching pyspark.
    #[new]
    #[pyo3(signature = (name=None))]
    fn new(name: Option<String>) -> Self {
        let n = name.unwrap_or_else(|| {
            format!("observation_{}", OBS_COUNTER.fetch_add(1, Ordering::SeqCst))
        });
        PyObservation {
            inner: Observation::new(&n),
        }
    }

    /// The observation name.
    #[getter]
    fn name(&self) -> String {
        self.inner.name().to_string()
    }

    /// The observed metrics (a dict), available after the observed action runs.
    #[getter]
    fn get(&self) -> std::collections::HashMap<String, String> {
        self.inner.get()
    }
}
