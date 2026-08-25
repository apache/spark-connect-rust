//! PyO3 wrapper for spark_connect::conf::RuntimeConf (`spark.conf`).

use pyo3::prelude::*;
use spark_connect::conf::RuntimeConf;

use crate::errors::ResultExt;

/// Python wrapper for the session runtime configuration.
#[pyclass(name = "RuntimeConf")]
pub struct PyRuntimeConf {
    conf: RuntimeConf,
}

impl PyRuntimeConf {
    pub fn new(conf: RuntimeConf) -> Self {
        PyRuntimeConf { conf }
    }
}

#[pymethods]
impl PyRuntimeConf {
    /// Set a configuration value (the value is coerced to its string form).
    fn set(&self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let v = value.str()?.to_string();
        self.conf.set(key, &v).to_pyerr()
    }

    /// Get a configuration value, returning `default` when unset.
    #[pyo3(signature = (key, default=None))]
    fn get(&self, key: &str, default: Option<String>) -> PyResult<Option<String>> {
        Ok(self.conf.get(key).to_pyerr()?.or(default))
    }

    /// Reset a configuration value to its default.
    fn unset(&self, key: &str) -> PyResult<()> {
        self.conf.unset(key).to_pyerr()
    }

    /// Whether the given configuration key is modifiable at runtime.
    #[pyo3(name = "isModifiable")]
    fn is_modifiable(&self, key: &str) -> PyResult<bool> {
        self.conf.is_modifiable(key).to_pyerr()
    }
}
