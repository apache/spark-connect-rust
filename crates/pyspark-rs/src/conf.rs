//! PyO3 wrapper for spark_connect::conf::RuntimeConf (`spark.conf`).

use pyo3::prelude::*;
use spark_connect::conf::RuntimeConf;
use spark_connect::session::{SparkSession, WeakSparkSession};

use crate::interrupt::interruptible;

// Python wrapper for the session runtime configuration.
#[doc = connect_doc!(RuntimeConf)]
#[pyclass(
    name = "RuntimeConf",
    module = "pyspark.sql.connect.conf",
    weakref,
    dict
)]
pub struct PyRuntimeConf {
    /// The session of the configuration, held weakly, as PySpark's `RuntimeConf` holds the
    /// client of its session and not the session object: once the program dropped the session,
    /// whose `__del__` closes its client in PySpark, the requests of the configuration raise
    /// `NO_ACTIVE_SESSION`.
    session: WeakSparkSession,
}

impl PyRuntimeConf {
    pub fn new(session: &SparkSession) -> Self {
        PyRuntimeConf {
            session: session.downgrade(),
        }
    }

    /// The session of the configuration, whose executions a signal of the process interrupts
    /// while a call of the configuration waits for the server, and its runtime configuration;
    /// once the session is dropped, the `NO_ACTIVE_SESSION` of PySpark's closed client.
    fn live(&self) -> PyResult<(SparkSession, RuntimeConf)> {
        match self.session.upgrade() {
            Some(session) => {
                let conf = session.conf();
                Ok((session, conf))
            }
            None => Err(crate::errors::spark_error_to_py_exception(
                spark_connect_core::error::SparkError::classed(
                    spark_connect_core::error::SparkErrorKind::Connect,
                    "NO_ACTIVE_SESSION",
                    &[],
                ),
            )),
        }
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
        Ok(text) => crate::str_argument(text.as_any()),
        Err(_) => Err(crate::stat_arguments::not_expected_type(obj, name, "str")),
    }
}

/// The value of `RuntimeConf.set`, as PySpark sends it in a `KeyValue`: a bool as "true" or
/// "false", an int as its `str()`, and None as no value, which the server rejects. Any other
/// value is set in the string field as it is, as [`crate::str_argument`] converts it.
fn conf_value(value: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    use pyo3::types::{PyBool, PyInt};
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
    crate::str_argument(value).map(Some)
}

/// Warn each warning of the server, such as of a deprecated or unsupported configuration, as
/// PySpark's `RuntimeConf` does with `warnings.warn`, of the line `line` of that call in
/// `pyspark.sql.connect.conf` (see `warn_as_pyspark`).
pub(crate) fn warn_all(py: Python<'_>, line: u32, warnings: &[String]) -> PyResult<()> {
    let category = py.get_type::<pyo3::exceptions::PyUserWarning>();
    for warning in warnings {
        crate::errors::warn_as_pyspark(py, "pyspark.sql.connect.conf", line, warning, &category)?;
    }
    Ok(())
}

// Every method round-trips to the server (`block_on(client.set_config/get_configs)`
// in the core), so each waits through `interruptible`, which releases the GIL across the
// RPC and lets a signal end the wait; otherwise a Python thread stays blocked for the
// whole call.
#[pymethods]
impl PyRuntimeConf {
    /// The shallow copy of the runtime configuration, as PySpark's `copy.copy`: of the same
    /// session, with the instance attributes of this one.
    fn __copy__(slf: &Bound<'_, Self>) -> PyResult<Py<Self>> {
        let py = slf.py();
        let copy = Py::new(
            py,
            PyRuntimeConf {
                session: slf.borrow().session.clone(),
            },
        )?;
        crate::reduce::copy_instance_dict(slf.as_any(), copy.bind(py).as_any())?;
        Ok(copy)
    }

    /// The runtime configuration does not pickle, as PySpark's, whose client holds a
    /// thread-local state that pickle cannot copy.
    fn __reduce__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::reduce::thread_local_reduce(slf.as_any())
    }

    // Set a configuration value, as PySpark's: a bool as "true" or "false", an int as its
    // `str()`, a str as it is, and None as no value, which the server rejects. Warns each
    // warning of the server.
    #[doc = connect_doc!(RuntimeConf.set)]
    fn set(
        &self,
        py: Python<'_>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        // A key of None is the empty key, as the constructor of PySpark's `KeyValue` leaves a
        // field of None unset.
        let key = if key.is_none() {
            String::new()
        } else {
            crate::str_argument(key)?
        };
        let value = conf_value(value)?;
        let (session, conf) = self.live()?;
        let warnings = interruptible(py, &session, || {
            conf.set_pairs(&[(key.as_str(), value.as_deref())], None)
                .map_err(Box::new)
        })?;
        // Of PySpark's `RuntimeConf.set`.
        warn_all(py, 45, &warnings)
    }

    // Get a configuration value. With a `default`, which may be None, an unknown key returns
    // the default (server `GetWithDefault`) rather than raising; without one, an unknown key
    // raises, mirroring `RuntimeConfig.get`, which raises `NOT_EXPECTED_TYPE` for a key, and
    // a default other than None, that is not a str.
    #[pyo3(signature = (key, default=ConfDefault::Omitted))]
    #[doc = connect_doc!(RuntimeConf.get)]
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
                let (session, conf) = self.live()?;
                interruptible(py, &session, || {
                    conf.get_with_default(&key, default.as_deref())
                        .map_err(Box::new)
                })
            }
            ConfDefault::Omitted => {
                let (session, conf) = self.live()?;
                interruptible(py, &session, || conf.get(&key).map_err(Box::new))
            }
        }
    }

    // Reset a configuration value to its default. Warns each warning of the server, as
    // PySpark's.
    #[doc = connect_doc!(RuntimeConf.unset)]
    fn unset(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<()> {
        let key = crate::str_argument(key)?;
        let (session, conf) = self.live()?;
        let warnings = interruptible(py, &session, || conf.unset(&key).map_err(Box::new))?;
        // Of PySpark's `RuntimeConf.unset`.
        warn_all(py, 100, &warnings)
    }

    /// Whether the given configuration key is modifiable at runtime.
    /// The values of the configurations `keys`, in one request, as PySpark Connect's
    /// `SparkConnectClient.get_configs`, which `SparkConnectClient` of the connect import path
    /// gives.
    fn _get_many(&self, py: Python<'_>, keys: Vec<String>) -> PyResult<Vec<Option<String>>> {
        let (session, conf) = self.live()?;
        interruptible(py, &session, || {
            let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
            conf.get_many(&keys).map_err(Box::new)
        })
    }

    /// The values of the configurations of `pairs` of a key and its default, in one request, as
    /// PySpark Connect's `SparkConnectClient.get_config_with_defaults`.
    fn _get_with_defaults(
        &self,
        py: Python<'_>,
        pairs: Vec<(String, Option<String>)>,
    ) -> PyResult<Vec<Option<String>>> {
        let (session, conf) = self.live()?;
        interruptible(py, &session, || {
            let pairs: Vec<(&str, Option<&str>)> = pairs
                .iter()
                .map(|(key, default)| (key.as_str(), default.as_deref()))
                .collect();
            conf.get_with_defaults(&pairs).map_err(Box::new)
        })
    }

    #[pyo3(name = "isModifiable")]
    #[doc = connect_doc!(RuntimeConf.isModifiable)]
    fn is_modifiable(&self, py: Python<'_>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        let key = crate::str_argument(key)?;
        let (session, conf) = self.live()?;
        interruptible(py, &session, || conf.is_modifiable(&key).map_err(Box::new))
    }

    // All configuration values as a dict. Mirrors the `RuntimeConf.getAll` property.
    #[getter(getAll)]
    #[doc = connect_doc!(RuntimeConf.getAll)]
    fn get_all<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let (session, conf) = self.live()?;
        let all = interruptible(py, &session, || conf.get_all().map_err(Box::new))?;
        let dict = pyo3::types::PyDict::new(py);
        for (k, v) in all {
            dict.set_item(k, v)?;
        }
        Ok(dict)
    }
}
