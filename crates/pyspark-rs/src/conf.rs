//! PyO3 wrapper for spark_connect::conf::RuntimeConf (`spark.conf`).

use pyo3::prelude::*;
use spark_connect::conf::RuntimeConf;

use crate::errors::ResultExt;

/// Python wrapper for the session runtime configuration.
#[pyclass(name = "RuntimeConf", module = "pyspark.sql.connect.conf")]
pub struct PyRuntimeConf {
    conf: RuntimeConf,
}

impl PyRuntimeConf {
    pub fn new(conf: RuntimeConf) -> Self {
        PyRuntimeConf { conf }
    }
}

// Every method round-trips to the server (`block_on(client.set_config/get_configs)`
// in the core), so each releases the GIL across the RPC via `py.detach`; otherwise a
// Python thread stays blocked for the whole call.
#[pymethods]
impl PyRuntimeConf {
    /// Set a configuration value. A bool lowercases to "true"/"false" (reference
    /// `to_str`); other values use their `str()`. (`None` would be unusual here;
    /// coerce_option_value maps it to no value, so fall back to "null".)
    fn set(&self, py: Python<'_>, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let v = crate::coerce_option_value(value)?.unwrap_or_else(|| "null".to_string());
        py.detach(|| self.conf.set(key, &v)).to_pyerr()
    }

    /// Get a configuration value. With a `default`, an unknown key returns the default
    /// (server `GetWithDefault`) rather than raising; without one, an unknown key raises,
    /// mirroring `RuntimeConfig.get`.
    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<String>) -> PyResult<Option<String>> {
        match default {
            Some(d) => py
                .detach(|| self.conf.get_with_default(key, Some(&d)))
                .to_pyerr(),
            None => py.detach(|| self.conf.get(key)).to_pyerr(),
        }
    }

    /// Reset a configuration value to its default.
    fn unset(&self, py: Python<'_>, key: &str) -> PyResult<()> {
        py.detach(|| self.conf.unset(key)).to_pyerr()
    }

    /// Whether the given configuration key is modifiable at runtime.
    #[pyo3(name = "isModifiable")]
    fn is_modifiable(&self, py: Python<'_>, key: &str) -> PyResult<bool> {
        py.detach(|| self.conf.is_modifiable(key)).to_pyerr()
    }

    /// All configuration values as a dict. Mirrors the `RuntimeConf.getAll` property.
    #[getter(getAll)]
    fn get_all<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let all = py.detach(|| self.conf.get_all()).to_pyerr()?;
        let dict = pyo3::types::PyDict::new(py);
        for (k, v) in all {
            dict.set_item(k, v)?;
        }
        Ok(dict)
    }
}
