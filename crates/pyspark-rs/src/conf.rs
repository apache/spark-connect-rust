//! PyO3 wrapper for spark_connect::conf::RuntimeConf (`spark.conf`).

use pyo3::prelude::*;
use spark_connect::conf::RuntimeConf;

use crate::errors::ResultExt;

/// Python wrapper for the session runtime configuration.
#[pyclass(
    name = "RuntimeConf",
    module = "pyspark.sql.connect.conf",
    weakref,
    dict
)]
pub struct PyRuntimeConf {
    conf: RuntimeConf,
}

impl PyRuntimeConf {
    pub fn new(conf: RuntimeConf) -> Self {
        PyRuntimeConf { conf }
    }
}

/// The `default` argument of `RuntimeConf.get`, where pyspark tells an omitted default apart
/// from None: `get(key, None)` returns None for an unset key, while `get(key)` raises.
pub enum ConfDefault {
    Omitted,
    Given(Py<PyAny>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for ConfDefault {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(ConfDefault::Given(obj.to_owned().unbind()))
    }
}

/// `obj`, a str, else PySpark's `NOT_EXPECTED_TYPE` of the argument `name`, as
/// `RuntimeConf._checkType`.
fn checked_str(obj: &Bound<'_, PyAny>, name: &str) -> PyResult<String> {
    match obj.cast::<pyo3::types::PyString>() {
        Ok(text) => Ok(text.to_cow()?.into_owned()),
        Err(_) => Err(crate::stat_arguments::not_expected_type(obj, name, "str")),
    }
}

/// The value of `RuntimeConf.set`, as PySpark sends it in a `KeyValue`: a bool as "true" or
/// "false", an int as its `str()`, a str, or UTF-8 bytes, as it is, and None as no value,
/// which the server rejects. Another value is the `TypeError` of protobuf, as in PySpark.
fn conf_value(value: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    use pyo3::types::{PyBool, PyBytes, PyInt, PyString};
    if value.is_none() {
        return Ok(None);
    }
    if let Ok(flag) = value.cast::<PyBool>() {
        return Ok(Some(
            if flag.is_true() { "true" } else { "false" }.to_string(),
        ));
    }
    if value.is_instance_of::<PyInt>() {
        return Ok(Some(value.str()?.to_cow()?.into_owned()));
    }
    if let Ok(text) = value.cast::<PyString>() {
        return Ok(Some(text.to_cow()?.into_owned()));
    }
    let shown = value.repr()?;
    if let Ok(bytes) = value.cast::<PyBytes>() {
        return String::from_utf8(bytes.as_bytes().to_vec())
            .map(Some)
            .map_err(|_| {
                pyo3::exceptions::PyValueError::new_err(format!(
                    "{shown} has type bytes, but isn't valid UTF-8 encoding. Non-UTF-8 strings \
                     must be converted to unicode objects before being added."
                ))
            });
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "Cannot set spark.connect.KeyValue.value to {shown}: {shown} has type {}, but expected \
         one of: (<class 'bytes'>, <class 'str'>) for field KeyValue.value",
        value.get_type().repr()?
    )))
}

/// Warn each warning of the server, such as of a deprecated or unsupported configuration, as
/// PySpark's `RuntimeConf` does with `warnings.warn`.
pub(crate) fn warn_all(py: Python<'_>, warnings: &[String]) -> PyResult<()> {
    let warn = py.import("warnings")?.getattr("warn")?;
    for warning in warnings {
        warn.call1((warning,))?;
    }
    Ok(())
}

// Every method round-trips to the server (`block_on(client.set_config/get_configs)`
// in the core), so each releases the GIL across the RPC via `py.detach`; otherwise a
// Python thread stays blocked for the whole call.
#[pymethods]
impl PyRuntimeConf {
    /// Set a configuration value, as PySpark's: a bool as "true" or "false", an int as its
    /// `str()`, a str as it is, and None as no value, which the server rejects. Warns each
    /// warning of the server.
    fn set(&self, py: Python<'_>, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let value = conf_value(value)?;
        let warnings = py
            .detach(|| {
                self.conf
                    .set_pairs(&[(key, value.as_deref())], None)
                    .map_err(Box::new)
            })
            .to_pyerr()?;
        warn_all(py, &warnings)
    }

    /// Get a configuration value. With a `default`, which may be None, an unknown key returns
    /// the default (server `GetWithDefault`) rather than raising; without one, an unknown key
    /// raises, mirroring `RuntimeConfig.get`, which raises `NOT_EXPECTED_TYPE` for a key, and
    /// a default other than None, that is not a str.
    #[pyo3(signature = (key, default=ConfDefault::Omitted))]
    fn get(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        default: ConfDefault,
    ) -> PyResult<Option<String>> {
        let key = checked_str(key, "key")?;
        match default {
            ConfDefault::Given(default) => {
                let default = default.bind(py);
                let default = if default.is_none() {
                    None
                } else {
                    Some(checked_str(default, "default")?)
                };
                py.detach(|| {
                    self.conf
                        .get_with_default(&key, default.as_deref())
                        .map_err(Box::new)
                })
                .to_pyerr()
            }
            ConfDefault::Omitted => py
                .detach(|| self.conf.get(&key).map_err(Box::new))
                .to_pyerr(),
        }
    }

    /// Reset a configuration value to its default. Warns each warning of the server, as
    /// PySpark's.
    fn unset(&self, py: Python<'_>, key: &str) -> PyResult<()> {
        let warnings = py
            .detach(|| self.conf.unset(key).map_err(Box::new))
            .to_pyerr()?;
        warn_all(py, &warnings)
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
