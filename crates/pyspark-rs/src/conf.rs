//! PyO3 wrapper for spark_connect::conf::RuntimeConf (`spark.conf`).

use pyo3::prelude::*;
use spark_connect::conf::RuntimeConf;
use spark_connect::session::SparkSession;

use crate::interrupt::interruptible_without_parts;

// Python wrapper for the session runtime configuration. As PySpark's plain class, it keeps its
// state in its `__dict__`, its `_client`, which its methods read when they make their requests, and
// it is subclassed and constructed of PySpark's arguments (see `crate::state`).
#[doc = connect_doc!(RuntimeConf)]
#[pyclass(
    name = "RuntimeConf",
    module = "pyspark.sql.connect.conf",
    subclass,
    weakref,
    dict
)]
pub struct PyRuntimeConf {}

/// What the requests of a runtime configuration are made of, of its `_client`.
enum Target<'py> {
    /// A client of this client, whose `config` is its class's: the session of the client, which
    /// it holds weakly, as PySpark's `RuntimeConf` holds the client of its session and not the
    /// session object, so that once the program dropped the session, whose `__del__` closes its
    /// client in PySpark, the requests raise `NO_ACTIVE_SESSION`; and its configuration.
    Native(SparkSession, RuntimeConf),
    /// Another object, whose `config` PySpark calls with each `ConfigRequest.Operation`.
    Client(Bound<'py, PyAny>),
}

impl PyRuntimeConf {
    /// The runtime configuration of `client`, as `spark.conf` makes it: `RuntimeConf(spark.client)`,
    /// of the client object, which PySpark's `__init__` keeps as its `_client`.
    pub(crate) fn of_client<'py>(
        client: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyRuntimeConf>> {
        let conf = Bound::new(client.py(), PyRuntimeConf {})?;
        Self::__init__(&conf, client)?;
        Ok(conf)
    }

    /// The target of the requests of the configuration, of its `_client`, which PySpark's methods
    /// read as they make their request, `self._client.config(operation)`: of the AttributeError of
    /// a configuration whose `__init__` never ran. A client of this client is of its session,
    /// unless its class overrides `config`, which is then called as PySpark calls it.
    fn target<'py>(slf: &Bound<'py, Self>) -> PyResult<Target<'py>> {
        let py = slf.py();
        let client = crate::state::attr(slf.as_any(), pyo3::intern!(py, "_client"))?;
        if let Some(session) = crate::state::client_session(&client) {
            if !crate::state::overrides(&client, pyo3::intern!(py, "config"))? {
                let conf = session.conf();
                return Ok(Target::Native(session, conf));
            }
        }
        Ok(Target::Client(client))
    }

    /// The session of the configuration and its runtime configuration, for the requests that only
    /// a client of this client makes (`_get_many`, `_get_with_defaults`).
    fn live(slf: &Bound<'_, Self>) -> PyResult<(SparkSession, RuntimeConf)> {
        match Self::target(slf)? {
            Target::Native(session, conf) => Ok((session, conf)),
            Target::Client(client) => Err(crate::state::foreign_client_error(&client)),
        }
    }
}

/// The result of `client.config(operation)`, of the `ConfigRequest.Operation` of PySpark's
/// `RuntimeConf` method `kind` of `args` (`pyspark.sql.connect._conf_operations`), as
/// PySpark's methods make their requests of a client of another class.
fn client_config<'py>(
    client: &Bound<'py, PyAny>,
    kind: &str,
    args: impl IntoPyObject<'py, Target = pyo3::types::PyTuple>,
) -> PyResult<Bound<'py, PyAny>> {
    use pyo3::BoundObject;
    let py = client.py();
    let args = args.into_pyobject(py).map_err(Into::into)?.into_bound();
    let operation = py
        .import("pyspark.sql.connect._conf_operations")?
        .getattr("config_operation")?
        .call1((kind,))?
        .call1(args)?;
    client.call_method1(pyo3::intern!(py, "config"), (operation,))
}

/// The value of the first pair of `result`, a `ConfigResult`, as PySpark's `result.pairs[0][1]`.
fn first_value(result: &Bound<'_, PyAny>) -> PyResult<Option<String>> {
    let py = result.py();
    result
        .getattr(pyo3::intern!(py, "pairs"))?
        .get_item(0)?
        .get_item(1)?
        .extract()
}

/// The warnings of `result`, a `ConfigResult`, as PySpark's `result.warnings`.
fn warnings_of(result: &Bound<'_, PyAny>) -> PyResult<Vec<String>> {
    let py = result.py();
    result
        .getattr(pyo3::intern!(py, "warnings"))?
        .try_iter()?
        .map(|warning| warning?.extract())
        .collect()
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
    /// A runtime configuration of no state, as the `object.__new__` of PySpark's plain class: of
    /// any arguments, which `__init__` takes, so that a subclass of another `__init__` is built as
    /// it is in PySpark. The class shows the signature of `__init__`, as `inspect.signature` of
    /// PySpark's class does.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs), text_signature = "(client)")]
    fn py_new(
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> Self {
        PyRuntimeConf {}
    }

    /// `RuntimeConf(client)`, as PySpark Connect's `__init__`, which keeps its argument as it is,
    /// as the `_client` of the configuration.
    fn __init__(slf: &Bound<'_, Self>, client: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        crate::state::assign(slf.as_any(), &[(pyo3::intern!(py, "_client"), client)])
    }

    /// The state of the configuration, its `__dict__`, as `object.__getstate__` gives that of
    /// PySpark's plain object: a copy shares its client, and a pickle fails at the client, which
    /// holds a thread-local state that pickle cannot copy, as PySpark's (see `state::getstate`).
    fn __getstate__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> {
        crate::state::getstate(slf.as_any())
    }

    /// The reduce of the configuration, as PySpark's plain object's: `copyreg.__newobj__` of its
    /// class, of its `__dict__` (see `state::reduce_ex`).
    fn __reduce_ex__<'py>(slf: &Bound<'py, Self>, protocol: i32) -> PyResult<Bound<'py, PyAny>> {
        crate::state::reduce_ex(slf.as_any(), protocol)
    }

    // Set a configuration value, as PySpark's: a bool as "true" or "false", an int as its
    // `str()`, a str as it is, and None as no value, which the server rejects. Warns each
    // warning of the server.
    #[doc = connect_doc!(RuntimeConf.set)]
    fn set(
        slf: &Bound<'_, Self>,
        key: &Bound<'_, PyAny>,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        let py = slf.py();
        // A key of None is the empty key, as the constructor of PySpark's `KeyValue` leaves a
        // field of None unset.
        let key = if key.is_none() {
            String::new()
        } else {
            crate::str_argument(key)?
        };
        let value = conf_value(value)?;
        let warnings = match Self::target(slf)? {
            Target::Native(session, conf) => interruptible_without_parts(py, &session, || {
                conf.set_pairs(&[(key.as_str(), value.as_deref())], None)
                    .map_err(Box::new)
            })?,
            Target::Client(client) => warnings_of(&client_config(&client, "set", (key, value))?)?,
        };
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
        slf: &Bound<'_, Self>,
        key: &Bound<'_, PyAny>,
        default: ConfDefault,
    ) -> PyResult<Option<String>> {
        let py = slf.py();
        let key = checked_str(key, "key")?;
        match default {
            ConfDefault::Given(default) => {
                let default = default.bind(py);
                let default = if default.is_none() {
                    None
                } else {
                    Some(checked_str(default, "default")?)
                };
                match Self::target(slf)? {
                    Target::Native(session, conf) => {
                        interruptible_without_parts(py, &session, || {
                            conf.get_with_default(&key, default.as_deref())
                                .map_err(Box::new)
                        })
                    }
                    Target::Client(client) => {
                        first_value(&client_config(&client, "get_with_default", (key, default))?)
                    }
                }
            }
            ConfDefault::Omitted => match Self::target(slf)? {
                Target::Native(session, conf) => {
                    interruptible_without_parts(py, &session, || conf.get(&key).map_err(Box::new))
                }
                Target::Client(client) => first_value(&client_config(&client, "get", (key,))?),
            },
        }
    }

    // Reset a configuration value to its default. Warns each warning of the server, as
    // PySpark's.
    #[doc = connect_doc!(RuntimeConf.unset)]
    fn unset(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<()> {
        let py = slf.py();
        let key = crate::str_argument(key)?;
        let warnings = match Self::target(slf)? {
            Target::Native(session, conf) => {
                interruptible_without_parts(py, &session, || conf.unset(&key).map_err(Box::new))?
            }
            Target::Client(client) => warnings_of(&client_config(&client, "unset", (key,))?)?,
        };
        // Of PySpark's `RuntimeConf.unset`.
        warn_all(py, 100, &warnings)
    }

    /// The values of the configurations `keys`, in one request, as PySpark Connect's
    /// `SparkConnectClient.get_configs`, which `SparkConnectClient` of the connect import path
    /// gives.
    fn _get_many(slf: &Bound<'_, Self>, keys: Vec<String>) -> PyResult<Vec<Option<String>>> {
        let (session, conf) = Self::live(slf)?;
        interruptible_without_parts(slf.py(), &session, || {
            let keys: Vec<&str> = keys.iter().map(String::as_str).collect();
            conf.get_many(&keys).map_err(Box::new)
        })
    }

    /// The values of the configurations of `pairs` of a key and its default, in one request, as
    /// PySpark Connect's `SparkConnectClient.get_config_with_defaults`.
    fn _get_with_defaults(
        slf: &Bound<'_, Self>,
        pairs: Vec<(String, Option<String>)>,
    ) -> PyResult<Vec<Option<String>>> {
        let (session, conf) = Self::live(slf)?;
        interruptible_without_parts(slf.py(), &session, || {
            let pairs: Vec<(&str, Option<&str>)> = pairs
                .iter()
                .map(|(key, default)| (key.as_str(), default.as_deref()))
                .collect();
            conf.get_with_defaults(&pairs).map_err(Box::new)
        })
    }

    // Whether the given configuration key is modifiable at runtime: PySpark's
    // `VALUE_NOT_ALLOWED` of a value of the server other than "true" and "false".
    #[pyo3(name = "isModifiable")]
    #[doc = connect_doc!(RuntimeConf.isModifiable)]
    fn is_modifiable(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>) -> PyResult<bool> {
        let py = slf.py();
        let key = crate::str_argument(key)?;
        match Self::target(slf)? {
            Target::Native(session, conf) => interruptible_without_parts(py, &session, || {
                conf.is_modifiable(&key).map_err(Box::new)
            }),
            Target::Client(client) => {
                match first_value(&client_config(&client, "is_modifiable", (key,))?)?.as_deref() {
                    Some("true") => Ok(true),
                    Some("false") => Ok(false),
                    _ => Err(crate::column::raise_pyspark(
                        py,
                        "PySparkValueError",
                        "VALUE_NOT_ALLOWED",
                        &[
                            ("arg_name", "result"),
                            ("allowed_values", "'true' or 'false'"),
                        ],
                    )),
                }
            }
        }
    }

    // All configuration values as a dict. Mirrors the `RuntimeConf.getAll` property.
    #[getter(getAll)]
    #[doc = connect_doc!(RuntimeConf.getAll)]
    fn get_all<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let py = slf.py();
        let dict = pyo3::types::PyDict::new(py);
        match Self::target(slf)? {
            Target::Native(session, conf) => {
                let all =
                    interruptible_without_parts(py, &session, || conf.get_all().map_err(Box::new))?;
                for (k, v) in all {
                    dict.set_item(k, v)?;
                }
            }
            Target::Client(client) => {
                // As PySpark's `for key, value in result.pairs: assert value is not None`.
                let result = client_config(&client, "get_all", ())?;
                for pair in result.getattr(pyo3::intern!(py, "pairs"))?.try_iter()? {
                    let pair = pair?;
                    let value = pair.get_item(1)?;
                    if value.is_none() {
                        return Err(pyo3::exceptions::PyAssertionError::new_err(()));
                    }
                    dict.set_item(pair.get_item(0)?, value)?;
                }
            }
        }
        Ok(dict)
    }
}
