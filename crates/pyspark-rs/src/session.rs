//! PyO3 wrapper for Spark Connect sessions.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use spark_connect::row::{Row, StructValue, Value};
use spark_connect::session::SparkSession;
use spark_connect::types::{DataType, StructField};
use spark_connect::udf::{CommonInlineUserDefinedFunctionExpression, PythonUDFPayload};
use spark_connect_core::artifact::{parse_artifact, UnsupportedArtifact};

use crate::catalog::PyCatalog;
use crate::dataframe::PyDataFrame;
use crate::datasource::PyDataSourceRegistration;
use crate::errors::ResultExt;
use crate::interrupt::interruptible;
use crate::resource::PyResourceProfile;
use crate::streaming::{PyDataStreamReader, PyStreamingQueryManager};

/// Python wrapper for a Spark session builder.
///
/// Mirrors `pyspark.sql.SparkSession.Builder`: each method sets its option on the builder and
/// returns it, and `SparkSession.builder` is a new builder of each access.
#[pyclass(name = "Builder", module = "pyspark.sql.session", from_py_object)]
#[derive(Clone)]
pub struct PySparkSessionBuilder {
    remote_url: Option<String>,
    /// Runtime configs set via `.config(...)`, applied after the session connects.
    configs: Vec<(String, Option<String>)>,
    /// The hook factories of `_registerHook`, which a new session calls with itself, as
    /// PySpark's builder does.
    hook_factories: Vec<std::sync::Arc<Py<PyAny>>>,
    /// The channel builder of `channelBuilder`, which the session connects with, instead of
    /// `remote()`: PySpark's `_channel_builder`, which is not an option.
    channel_builder: Option<std::sync::Arc<Py<PyAny>>>,
}

impl PySparkSessionBuilder {
    pub fn new() -> Self {
        PySparkSessionBuilder {
            remote_url: None,
            configs: Vec::new(),
            hook_factories: Vec::new(),
            channel_builder: None,
        }
    }
}

/// `SparkSession.builder`, as PySpark's `classproperty(lambda cls: cls.Builder())`: each access
/// of the class, or of a session, is a new builder, so the options of one do not leak into
/// another.
#[pyclass(name = "classproperty", module = "pyspark.sql.session")]
pub struct PyBuilderProperty;

#[pymethods]
impl PyBuilderProperty {
    fn __get__(
        &self,
        _instance: Option<&Bound<'_, PyAny>>,
        _owner: Option<&Bound<'_, PyAny>>,
    ) -> PySparkSessionBuilder {
        PySparkSessionBuilder::new()
    }
}

#[pymethods]
impl PySparkSessionBuilder {
    /// Construct a fresh builder, so `SparkSession.Builder()` works like pyspark
    /// (in addition to the `SparkSession.builder` class attribute).
    #[new]
    fn py_new() -> Self {
        PySparkSessionBuilder::new()
    }

    /// Set the remote Spark Connect server URL. Returns the builder (chainable).
    /// Mirrors `SparkSession.Builder.remote(url)`, and the Connect session builder's
    /// `remote(location="sc://localhost")`: the URL is `url` or the keyword `location`.
    #[pyo3(signature = (url=None, *, location=None))]
    fn remote<'py>(
        mut slf: PyRefMut<'py, Self>,
        url: Option<&str>,
        location: Option<&str>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let url = match (url, location) {
            (Some(_), Some(_)) => {
                return Err(pyo3::exceptions::PyTypeError::new_err(
                    "remote() got multiple values for the URL: give url or location",
                ))
            }
            (Some(url), None) | (None, Some(url)) => url,
            (None, None) => "sc://localhost",
        };
        let py = slf.py();
        slf.set_option(py, "spark.remote", url.to_string())?;
        Ok(slf)
    }

    /// Set a config option, or several via `map`. Mirrors
    /// `SparkSession.Builder.config(key=None, value=None, *, map=None, conf=None)`.
    /// `spark.remote` sets the connect endpoint; other keys are applied as runtime confs
    /// on connect. `conf` accepts a `SparkConf` (its `getAll()` pairs are applied).
    #[pyo3(signature = (key=None, value=None, map=None, conf=None))]
    fn config<'py>(
        mut slf: PyRefMut<'py, Self>,
        key: Option<&str>,
        value: Option<&Bound<'_, PyAny>>,
        map: Option<&Bound<'_, PyDict>>,
        conf: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        if let Some(k) = key {
            let v = match value {
                Some(v) => crate::coerce_option_value(v)?,
                None => None,
            };
            slf.set_option_value(py, k, v)?;
        }
        if let Some(m) = map {
            for (k, v) in m.iter() {
                let ks = k.str()?.to_string();
                let vs = crate::coerce_option_value(&v)?;
                slf.set_option_value(py, &ks, vs)?;
            }
        }
        if let Some(c) = conf {
            // A SparkConf: apply its (key, value) pairs via getAll().
            for pair in c.call_method0("getAll")?.try_iter()? {
                let pair = pair?;
                let k: String = pair.get_item(0)?.str()?.to_string();
                let v: String = pair.get_item(1)?.str()?.to_string();
                slf.set_option(py, &k, v)?;
            }
        }
        Ok(slf)
    }

    /// `appName` - the option `spark.app.name`, as PySpark's builder sets. Like any option,
    /// it is set on the session silently, so it has no effect on a running server.
    #[pyo3(name = "appName")]
    fn app_name<'py>(mut slf: PyRefMut<'py, Self>, name: &str) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        slf.set_option(py, "spark.app.name", name.to_string())?;
        Ok(slf)
    }

    /// `master` - the option `spark.master`, as PySpark's builder sets: a session of Spark
    /// Connect does not take it, and with a remote it raises
    /// `CANNOT_CONFIGURE_SPARK_CONNECT_MASTER`.
    fn master<'py>(mut slf: PyRefMut<'py, Self>, master: &str) -> PyResult<PyRefMut<'py, Self>> {
        let py = slf.py();
        slf.set_option(py, "spark.master", master.to_string())?;
        Ok(slf)
    }

    /// `_registerHook` - register a factory of a `SparkSession.Hook`, as PySpark's builder: a
    /// session that the builder creates calls it with itself, and calls the `on_execute_plan`
    /// of the hook with each `ExecutePlanRequest` of its executions before it sends it.
    #[pyo3(name = "_registerHook")]
    fn register_hook<'py>(
        mut slf: PyRefMut<'py, Self>,
        hook_factory: &Bound<'_, PyAny>,
    ) -> PyRefMut<'py, Self> {
        slf.hook_factories
            .push(std::sync::Arc::new(hook_factory.clone().unbind()));
        slf
    }

    /// `channelBuilder` - accept a Spark Connect ChannelBuilder and reconstruct an
    /// `sc://` connection URL from its endpoint (host/port) and connection params, then
    /// use it as the native-transport remote. Mirrors `SparkSession.Builder.channelBuilder`
    /// (custom gRPC channels/interceptors themselves are a Python-transport concept and
    /// do not apply to the native Rust transport, but the endpoint + params are honored).
    #[pyo3(name = "channelBuilder")]
    #[allow(non_snake_case)]
    fn channel_builder<'py>(
        mut slf: PyRefMut<'py, Self>,
        channelBuilder: &Bound<'_, PyAny>,
    ) -> PyResult<PyRefMut<'py, Self>> {
        // A builder without an endpoint is an error here, not when the session is created; the
        // URL itself is read then, with the parameters of the builder as they are by then.
        url_of_channel_builder(channelBuilder)?;
        slf.channel_builder = Some(std::sync::Arc::new(channelBuilder.clone().unbind()));
        Ok(slf)
    }

    /// `enableHiveSupport` - a no-op for a Connect client (the remote server's
    /// catalog is fixed); accepted for API parity.
    #[pyo3(name = "enableHiveSupport")]
    fn enable_hive_support(slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf
    }

    /// Get the active session of this thread, or else the default session, or create one
    /// (`getOrCreate`), and set the options of the builder on it. Mirrors PySpark's: the URL
    /// is that of `remote()` or else of `SPARK_REMOTE`, and an existing session is returned
    /// whatever its URL.
    #[pyo3(name = "getOrCreate")]
    fn get_or_create(&self, py: Python<'_>) -> PyResult<Py<PySparkSession>> {
        let (session, created) = {
            let _creating = session_lock();
            match active_session(py).or_else(|| default_session(py)) {
                Some(session) => (session, false),
                // The URL is needed only to create a session, as PySpark's getOrCreate.
                None => (new_session(py, &self.connection_url(py)?)?, true),
            }
        };
        if created {
            self.install_hooks(py, &session)?;
        }
        self.apply_options(py, &session.borrow(py).session)?;
        log_client_banner(py);
        Ok(session)
    }

    /// `create()` - a new session, which is the default session, and the active one of this
    /// thread, only when they are not set. Mirrors PySpark's.
    fn create(&self, py: Python<'_>) -> PyResult<Py<PySparkSession>> {
        let url = self.connection_url(py)?;
        let session = {
            let _creating = session_lock();
            new_session(py, &url)?
        };
        self.install_hooks(py, &session)?;
        self.apply_options(py, &session.borrow(py).session)?;
        log_client_banner(py);
        Ok(session)
    }
}

impl PySparkSessionBuilder {
    /// The hooks of the factories of `_registerHook` for `session`, a session that the builder
    /// created, as PySpark's `SparkSession(hook_factories=...)` makes them.
    fn install_hooks(&self, py: Python<'_>, session: &Py<PySparkSession>) -> PyResult<()> {
        let factories: Vec<Py<PyAny>> = self
            .hook_factories
            .iter()
            .map(|factory| factory.clone_ref(py))
            .collect();
        install_hook_factories(py, &factories, session)
    }

    /// The connection URL of the session that the builder creates: that of the channel builder
    /// of `channelBuilder`, or else of `remote()` or `SPARK_REMOTE`. As PySpark's `create`, it
    /// is an error to give both a channel builder and a URL.
    fn connection_url(&self, py: Python<'_>) -> PyResult<String> {
        match &self.channel_builder {
            Some(_) if self.remote_url.is_some() => Err(crate::column::raise_pyspark(
                py,
                "PySparkValueError",
                "SESSION_NEED_CONN_STR_OR_BUILDER",
                &[],
            )),
            Some(channel_builder) => url_of_channel_builder(channel_builder.bind(py)),
            None => self.url(py),
        }
    }

    /// Set an option, capturing `spark.remote` as the endpoint, and validate the startup URLs
    /// after it, as PySpark's `config`.
    fn set_option(&mut self, py: Python<'_>, key: &str, value: String) -> PyResult<()> {
        self.set_option_value(py, key, Some(value))
    }

    /// Set an option whose value may be None, which PySpark keeps in its options: such a key is
    /// not set on the session, as `_set_all` sends it without a value.
    fn set_option_value(
        &mut self,
        py: Python<'_>,
        key: &str,
        value: Option<String>,
    ) -> PyResult<()> {
        if key == "spark.remote" {
            self.remote_url = value;
        } else {
            self.configs.push((key.to_string(), value));
        }
        self.validate_startup_urls(py)
    }

    /// As PySpark's `_validate_startup_urls`: a master (`spark.master`, or else `MASTER`) with a
    /// remote (`spark.remote`, or else `SPARK_REMOTE`) raises
    /// `CANNOT_CONFIGURE_SPARK_CONNECT_MASTER`, and a remote other than `SPARK_REMOTE`, with
    /// `SPARK_LOCAL_REMOTE`, raises `CANNOT_CONFIGURE_SPARK_CONNECT` unless it is a local one.
    fn validate_startup_urls(&self, py: Python<'_>) -> PyResult<()> {
        let master = self
            .configs
            .iter()
            .rev()
            .find(|(key, _)| key == "spark.master")
            .and_then(|(_, value)| value.clone())
            .or_else(|| std::env::var("MASTER").ok());
        let remote = self
            .remote_url
            .clone()
            .or_else(|| std::env::var("SPARK_REMOTE").ok());
        if let (Some(master), Some(remote)) = (&master, &remote) {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkRuntimeError",
                "CANNOT_CONFIGURE_SPARK_CONNECT_MASTER",
                &[("master_url", master), ("connect_url", remote)],
            ));
        }
        if let (Some(remote), Ok(existing), Ok(_)) = (
            &self.remote_url,
            std::env::var("SPARK_REMOTE"),
            std::env::var("SPARK_LOCAL_REMOTE"),
        ) {
            if &existing != remote && !remote.starts_with("local") {
                return Err(crate::column::raise_pyspark(
                    py,
                    "PySparkRuntimeError",
                    "CANNOT_CONFIGURE_SPARK_CONNECT",
                    &[("existing_url", &existing), ("new_url", remote)],
                ));
            }
        }
        Ok(())
    }

    /// The URL of the session: that of `remote()`, or else of `SPARK_REMOTE`, as PySpark's
    /// builder of a Connect-only client, which raises `CONNECT_URL_NOT_SET` without either.
    fn url(&self, py: Python<'_>) -> PyResult<String> {
        match self
            .remote_url
            .clone()
            .or_else(|| std::env::var("SPARK_REMOTE").ok())
        {
            Some(url) => Ok(url),
            None => Err(connect_url_not_set(py)),
        }
    }

    /// Set the options of the builder on `session`, as PySpark's `_apply_options`: in one
    /// request whose failures are ignored, such as those of static configurations, and
    /// without `spark.master`, which a session does not take.
    // SparkError is large, as the core crate allows for its API.
    #[allow(clippy::result_large_err)]
    fn apply_options(&self, py: Python<'_>, session: &SparkSession) -> PyResult<()> {
        // A later value of a key replaces an earlier one, as the options of PySpark's builder do.
        let mut options: Vec<(String, Option<String>)> = Vec::new();
        for (key, value) in self.configs.iter().filter(|(key, _)| key != "spark.master") {
            match options.iter_mut().find(|(k, _)| k == key) {
                Some(entry) => entry.1 = value.clone(),
                None => options.push((key.clone(), value.clone())),
            }
        }
        if options.is_empty() {
            return Ok(());
        }
        let pairs: Vec<(&str, Option<&str>)> = options
            .iter()
            .map(|(key, value)| (key.as_str(), value.as_deref()))
            .collect();
        let conf = session.conf();
        let warnings = py
            .detach(|| conf.set_pairs(&pairs, Some(true)).map_err(Box::new))
            .to_pyerr()?;
        // As PySpark's `_set_all`, which warns each warning of the server.
        crate::conf::warn_all(py, &warnings)
    }
}

/// The `sc://` URL of the channel builder `channel_builder`, which has the `host` and the `port`
/// of its endpoint and the `_params` of its connection, as PySpark's `ChannelBuilder` does. Its
/// parameters are percent-encoded, as the URL parser of the transport decodes them.
pub(crate) fn url_of_channel_builder(channel_builder: &Bound<'_, PyAny>) -> PyResult<String> {
    // Endpoint: the builder's host and port, which DefaultChannelBuilder exposes.
    let host: String = channel_builder
        .getattr("host")
        .and_then(|h| h.extract::<String>())
        .map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "channelBuilder must expose a `host` (e.g. DefaultChannelBuilder); \
                 otherwise configure the endpoint via .remote(url)",
            )
        })?;
    let port: i64 = channel_builder
        .getattr("port")
        .and_then(|p| p.extract::<i64>())
        .unwrap_or(15002);
    let mut url = format!("sc://{host}:{port}");
    // Preserve connection params (token/use_ssl/user_id/...) if present as `_params`.
    if let Ok(params) = channel_builder.getattr("_params") {
        if let Ok(dict) = params.cast::<pyo3::types::PyDict>() {
            if !dict.is_empty() {
                let mut parts: Vec<String> = Vec::new();
                for (k, v) in dict.iter() {
                    parts.push(format!(
                        "{}={}",
                        k.str()?,
                        quote_param(&v.str()?.to_string())
                    ));
                }
                url.push_str("/;");
                url.push_str(&parts.join(";"));
            }
        }
    }
    Ok(url)
}

/// `value`, percent-encoded outside the unreserved characters, so that it is one value of a
/// `;`-separated URL parameter.
fn quote_param(value: &str) -> String {
    value
        .bytes()
        .map(|b| {
            if b.is_ascii_alphanumeric() || b"-._~".contains(&b) {
                (b as char).to_string()
            } else {
                format!("%{b:02X}")
            }
        })
        .collect()
}

/// `url`, with the parameter `user_id` set to `user`, unless it has one, which takes precedence,
/// as PySpark's `SparkSession(userId=...)` gives it to the connection.
fn with_user_id(url: &str, user: &str) -> PyResult<String> {
    if spark_connect_core::channel::ChannelBuilder::parse(url)
        .to_pyerr()?
        .get("user_id")
        .is_some()
    {
        return Ok(url.to_string());
    }
    let param = format!("user_id={}", quote_param(user));
    let base = url.trim_end_matches('/');
    Ok(if base.contains(';') {
        format!("{base};{param}")
    } else {
        format!("{base}/;{param}")
    })
}

/// The `CONNECT_URL_NOT_SET` error of a builder without a URL: see `_ConnectUrlNotSetError`.
fn connect_url_not_set(py: Python<'_>) -> PyErr {
    let error = py
        .import("pyspark.sql.connect.session")
        .and_then(|module| module.getattr("_connect_url_not_set_error"))
        .and_then(|make| make.call0());
    match error {
        Ok(error) => PyErr::from_value(error),
        Err(error) => error,
    }
}

/// Install on `session` the hooks that `factories` make of it, each called with the session, as
/// PySpark's `SparkSession(hook_factories=...)`.
fn install_hook_factories(
    py: Python<'_>,
    factories: &[Py<PyAny>],
    session: &Py<PySparkSession>,
) -> PyResult<()> {
    if factories.is_empty() {
        return Ok(());
    }
    let hooks = factories
        .iter()
        .map(|factory| {
            factory
                .bind(py)
                .call1((session.clone_ref(py),))
                .map(Bound::unbind)
        })
        .collect::<PyResult<Vec<_>>>()?;
    let hook: spark_connect::ExecutePlanHook = std::sync::Arc::new(move |request| {
        Python::attach(|py| call_execute_plan_hooks(py, &hooks, request))
            .map_err(crate::errors::callback_error)
    });
    session.borrow(py).session.set_execute_plan_hook(Some(hook));
    Ok(())
}

/// The default session, which is the class attribute `_default_session` of `SparkSession`, as
/// PySpark's: the first session created, of every thread, until it stops. The attribute is the
/// one value, which the Python code may also set, as PySpark's tests do.
fn default_slot(py: Python<'_>) -> Option<Py<PySparkSession>> {
    let value = py
        .get_type::<PySparkSession>()
        .getattr("_default_session")
        .ok()?;
    if value.is_none() {
        return None;
    }
    value.cast_into::<PySparkSession>().ok().map(Bound::unbind)
}

/// Set the default session, as the class attribute `_default_session`.
fn set_default_slot(py: Python<'_>, session: Option<Py<PySparkSession>>) {
    let _ = py
        .get_type::<PySparkSession>()
        .setattr("_default_session", session);
}

thread_local! {
    /// The active session of this thread, until it stops. Mirrors the thread-local
    /// `SparkSession._active_session`.
    static ACTIVE_SESSION: std::cell::RefCell<Option<Py<PySparkSession>>> =
        const { std::cell::RefCell::new(None) };
}

/// The request that `hooks` return of `request`, as PySpark's client calls its session hooks:
/// each `on_execute_plan` with the request of the one before, as a `pb2.ExecutePlanRequest`
/// of the generated `pyspark.sql.connect.proto`, of the operation ID of `request`.
fn call_execute_plan_hooks(
    py: Python<'_>,
    hooks: &[Py<PyAny>],
    request: spark_connect_proto::ExecutePlanRequest,
) -> PyResult<spark_connect_proto::ExecutePlanRequest> {
    use prost::Message;
    let operation_id = request.operation_id.clone();
    let mut message = py
        .import("pyspark.sql.connect.proto")?
        .getattr("ExecutePlanRequest")?
        .call_method1(
            "FromString",
            (pyo3::types::PyBytes::new(py, &request.encode_to_vec()),),
        )?;
    for hook in hooks {
        message = hook.bind(py).call_method1("on_execute_plan", (message,))?;
        if let Some(operation_id) = &operation_id {
            message.setattr("operation_id", operation_id)?;
        }
    }
    let bytes = message.call_method0("SerializeToString")?;
    let mut request = spark_connect_proto::ExecutePlanRequest::decode(
        bytes.cast::<pyo3::types::PyBytes>()?.as_bytes(),
    )
    .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    // A hook may put in the plan of a DataFrame, which the Python plan sends as a placeholder.
    spark_connect::plan::resolve_request(&mut request)
        .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
    Ok(request)
}

/// Held while a session is got or created, as PySpark's `SparkSession._lock`, so that threads
/// get the same default session. Taken with the GIL, which is not released while it is held.
fn session_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|e| e.into_inner())
}

/// `session`, unless it stopped.
fn unless_stopped(py: Python<'_>, session: &Py<PySparkSession>) -> Option<Py<PySparkSession>> {
    (!session.borrow(py).session.is_stopped()).then(|| session.clone_ref(py))
}

/// The active session of this thread, unless it stopped.
fn active_session(py: Python<'_>) -> Option<Py<PySparkSession>> {
    ACTIVE_SESSION.with(|active| {
        active
            .borrow()
            .as_ref()
            .and_then(|session| unless_stopped(py, session))
    })
}

/// The default session, unless it stopped.
fn default_session(py: Python<'_>) -> Option<Py<PySparkSession>> {
    default_slot(py).and_then(|session| unless_stopped(py, &session))
}

/// A new session of `url`, which is the default session, and the active session of this
/// thread, when they are not set, as PySpark's `_set_default_and_active_session`.
fn new_session(py: Python<'_>, url: &str) -> PyResult<Py<PySparkSession>> {
    // The channel connects lazily, so this does not wait on the server.
    let session = SparkSession::builder()
        .remote(url)
        .get_or_create()
        .to_pyerr()?;
    let session = session_object(py, &session)?;
    set_default_and_active(py, &session);
    Ok(session)
}

/// The Python session objects by session ID, held weakly, so that the registry keeps none
/// alive. Sessions made apart may share a session ID, such as two of an explicit `session_id`,
/// so an ID has a list of them.
type SessionObjects = std::collections::HashMap<String, Vec<Py<PyAny>>>;

/// The registry of the Python session objects.
fn session_objects() -> &'static std::sync::Mutex<SessionObjects> {
    static OBJECTS: std::sync::OnceLock<std::sync::Mutex<SessionObjects>> =
        std::sync::OnceLock::new();
    OBJECTS.get_or_init(|| std::sync::Mutex::new(SessionObjects::new()))
}

/// The session object that the weak reference `weak` refers to, unless it was dropped.
fn live_session_object(py: Python<'_>, weak: &Py<PyAny>) -> Option<Py<PySparkSession>> {
    let object = weak.bind(py).call0().ok()?;
    let session = object.cast::<PySparkSession>().ok()?;
    Some(session.clone().unbind())
}

/// The live object of `session` itself in `registry`, if any, and not that of another session
/// of its session ID.
fn registered_session_object(
    py: Python<'_>,
    registry: &SessionObjects,
    session: &SparkSession,
) -> Option<Py<PySparkSession>> {
    registry
        .get(session.session_id())?
        .iter()
        .filter_map(|weak| live_session_object(py, weak))
        .find(|object| {
            object
                .bind(py)
                .try_borrow()
                .is_ok_and(|object| object.session.is_same_session(session))
        })
}

/// The Python object of `session`, one per session as PySpark has them: its live object, or
/// else a new one, which the registry holds weakly. So the DataFrames, the streams and the
/// clients of a session give back the session's object, as PySpark's do.
pub(crate) fn session_object(
    py: Python<'_>,
    session: &SparkSession,
) -> PyResult<Py<PySparkSession>> {
    {
        let registry = session_objects().lock().unwrap_or_else(|e| e.into_inner());
        if let Some(object) = registered_session_object(py, &registry, session) {
            return Ok(object);
        }
    }
    let object = Py::new(py, PySparkSession::new(session.clone()))?;
    let weak = py
        .import("weakref")?
        .call_method1("ref", (&object,))?
        .unbind();
    let mut registry = session_objects().lock().unwrap_or_else(|e| e.into_inner());
    // Another thread may have made the object of this session meanwhile; then it is that one.
    if let Some(existing) = registered_session_object(py, &registry, session) {
        return Ok(existing);
    }
    registry.retain(|_, objects| {
        objects.retain(|weak| live_session_object(py, weak).is_some());
        !objects.is_empty()
    });
    registry
        .entry(session.session_id().to_string())
        .or_default()
        .push(weak);
    Ok(object)
}

/// Make `session` the default session, and the active session of this thread, when they are
/// not set, as PySpark's `SparkSession._set_default_and_active_session`.
fn set_default_and_active(py: Python<'_>, session: &Py<PySparkSession>) {
    if default_slot(py).is_none() {
        set_default_slot(py, Some(session.clone_ref(py)));
    }
    ACTIVE_SESSION.with(|active| {
        let mut active = active.borrow_mut();
        if active.is_none() {
            *active = Some(session.clone_ref(py));
        }
    });
}

/// Emit a one-time INFO log identifying this as the Rust-backed Spark Connect
/// client, so it is discoverable at runtime which client is in use -- and issues
/// get filed against this project, not the reference `pyspark-client`. Logged the
/// first time a session connects, via Python's `logging` (so it honors the
/// application's logging configuration); best-effort and never fails the connect.
fn log_client_banner(py: Python<'_>) {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let _ = (|| -> PyResult<()> {
            let version: String = py
                .import("pyspark")
                .and_then(|m| m.getattr("__version__"))
                .and_then(|v| v.extract())
                .unwrap_or_default();
            let logger = py
                .import("logging")?
                .call_method1("getLogger", ("pyspark",))?;
            let msg = format!(
                "pyspark-client-rust {version} active: Spark Connect client backed by the native \
                 Rust engine (tonic), a drop-in for pyspark-client -- not the reference Python \
                 client. https://apache.github.io/spark-connect-rust/which-client/"
            );
            logger.call_method1("info", (msg,))?;
            Ok(())
        })();
    });
}

/// The active session of the current thread, as PySpark's `SparkSession._active_session`, a
/// thread-local whose `session` attribute is the session: the `ACTIVE_SESSION` of this thread.
#[pyclass(name = "_ActiveSession", module = "pyspark.sql.session", frozen)]
pub struct PyActiveSession;

#[pymethods]
impl PyActiveSession {
    /// The active session of this thread, which may be stopped, as the attribute of the
    /// thread-local of PySpark.
    #[getter]
    fn session(&self, py: Python<'_>) -> Option<Py<PySparkSession>> {
        ACTIVE_SESSION.with(|active| {
            active
                .borrow()
                .as_ref()
                .map(|session| session.clone_ref(py))
        })
    }

    /// Set the active session of this thread; None clears it.
    #[setter]
    fn set_session(&self, session: Option<Py<PySparkSession>>) {
        ACTIVE_SESSION.with(|active| *active.borrow_mut() = session);
    }
}

/// `SparkSession.Hook` - injects behavior around plan execution, as PySpark's
/// `SparkSession.Hook`: a session calls the `on_execute_plan` of each hook of the factories of
/// `_registerHook` with each `ExecutePlanRequest`, as a message of the generated
/// `pyspark.sql.connect.proto`, before it sends it, and sends the request that it returns. The
/// base hook returns the request as it is.
#[pyclass(name = "Hook", module = "pyspark.sql.connect.session", subclass)]
pub struct PyHook;

#[pymethods]
impl PyHook {
    /// A hook, of the arguments of the `__init__` of a subclass, which `__new__` leaves to it,
    /// as `object.__new__` does for PySpark's hooks.
    #[new]
    #[pyo3(signature = (*_args, **_kwargs))]
    fn new(
        _args: &Bound<'_, pyo3::types::PyTuple>,
        _kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
    ) -> Self {
        PyHook
    }

    /// Called before sending an ExecutePlanRequest; returns the (unchanged) request.
    fn on_execute_plan<'py>(&self, request: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        request
    }
}

/// Python wrapper for a Spark session.
/// A copy of the tags of a thread, which the threads that `_share_thread_tags` gives it share,
/// for `pyspark.util.inheritable_thread_target`.
#[pyclass(name = "_SharedTags", module = "pyspark.sql.session", frozen)]
pub struct PySharedTags(spark_connect::session::SharedTags);

/// A copy of the user context extensions of a thread, which the threads that
/// `_share_thread_user_context_extensions` gives it share, for
/// `pyspark.util.inheritable_thread_target`.
#[pyclass(
    name = "_SharedUserContextExtensions",
    module = "pyspark.sql.session",
    frozen
)]
pub struct PySharedUserContextExtensions(spark_connect_core::client::SharedUserContextExtensions);

/// A tag of `addTag` or `removeTag`: `CANNOT_BE_NONE` for None, as PySpark Connect's.
fn tag_of(py: Python<'_>, tag: &Bound<'_, PyAny>) -> PyResult<String> {
    if tag.is_none() {
        return Err(crate::column::raise_pyspark(
            py,
            "PySparkValueError",
            "CANNOT_BE_NONE",
            &[("arg_name", "Spark Connect tag")],
        ));
    }
    tag.extract()
}

#[pyclass(name = "SparkSession", module = "pyspark.sql.session", weakref, dict)]
pub struct PySparkSession {
    pub(crate) session: SparkSession,
    /// The `StreamingQueryManager` of `spark.streams`, made at its first access.
    streams: std::sync::OnceLock<Py<PyStreamingQueryManager>>,
    /// Whether `stop()` releases the session on the server, as PySpark's
    /// `release_session_on_close`, which tests set to False to skip the release.
    release_session_on_close: std::sync::atomic::AtomicBool,
}

struct PandasArrowColumn {
    data_type: DataType,
    arrays: Vec<arrow::array::ArrayRef>,
    field: arrow::datatypes::Field,
}

struct PandasDecimalContext<'a> {
    dtype: &'a str,
    label: &'a str,
    safecheck: Option<bool>,
}

struct PandasArrowSettings {
    options: spark_connect::ArrowInputOptions,
    safecheck: bool,
    infer_dict_as_map: bool,
}

impl PandasArrowSettings {
    fn helper_options(&self) -> (&str, bool, bool) {
        (
            self.options.timezone().unwrap_or("UTC"),
            self.options.use_large_var_types() == Some("true"),
            self.options.timestamp_type() == Some("TIMESTAMP_NTZ"),
        )
    }

    fn prepare_table<'py>(
        &self,
        helper: &Bound<'py, PyAny>,
        frame: &Bound<'py, PyAny>,
    ) -> PyResult<(Bound<'py, PyAny>, Option<Vec<String>>)> {
        let (timezone, large, timestamp_ntz) = self.helper_options();
        helper
            .getattr("prepare_inferred_pandas_table")?
            .call1((
                frame,
                timezone,
                self.safecheck,
                large,
                self.infer_dict_as_map,
                timestamp_ntz,
            ))?
            .extract()
    }

    fn prepare_row_column<'py>(
        &self,
        helper: &Bound<'py, PyAny>,
        values: &Bound<'py, PyAny>,
        name: &str,
    ) -> PyResult<(DataType, Bound<'py, PyAny>)> {
        let (timezone, large, timestamp_ntz) = self.helper_options();
        let result = helper
            .getattr("prepare_inferred_pandas_row_column")?
            .call1((
                values,
                name,
                timezone,
                self.safecheck,
                large,
                self.infer_dict_as_map,
                timestamp_ntz,
            ))?;
        let (spark_type, table): (Bound<'py, PyAny>, Bound<'py, PyAny>) = result.extract()?;
        Ok((crate::types::py_to_data_type(&spark_type)?, table))
    }
}

struct PandasRowColumns {
    columns: Vec<Option<PandasArrowColumn>>,
    selected: Vec<bool>,
    errors: Vec<Option<PyErr>>,
    settings: Option<PandasArrowSettings>,
    row_count: usize,
}

impl PandasRowColumns {
    fn empty(width: usize, row_count: usize) -> Self {
        Self {
            columns: (0..width).map(|_| None).collect(),
            selected: vec![false; width],
            errors: (0..width).map(|_| None).collect(),
            settings: None,
            row_count,
        }
    }
}

impl PySparkSession {
    pub fn new(session: SparkSession) -> Self {
        // The progress of the session, for the progress bars of its executions, and its
        // profiles, which the updates of its first execution update too.
        Python::attach(|py| {
            crate::progress::session_progress(py, &session);
            if let Err(error) = crate::profiler::profiler_collector(py, &session) {
                error.write_unraisable(py, None);
            }
        });
        PySparkSession {
            session,
            streams: std::sync::OnceLock::new(),
            release_session_on_close: std::sync::atomic::AtomicBool::new(true),
        }
    }

    /// A DataFrame of local rows, as PySpark Connect creates one: the schema is given, or
    /// inferred by merging the types of every row, and each value is converted to the type
    /// of its field. A list of names names the inferred fields and is applied with `toDF`.
    fn create_from_local(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        spec: Spec,
    ) -> PyResult<PyDataFrame> {
        use crate::infer;

        let (given, mut cols) = match spec {
            Spec::None => (None, None),
            Spec::Names(names) => (None, Some(names)),
            Spec::Struct(fields) => (Some(fields), None),
            Spec::Atomic(atomic) => (Some(vec![infer::value_field(atomic)]), None),
        };
        // Empty data with a length needs a schema to type its relation, and gets no toDF.
        if data.len().is_ok_and(|n| n == 0) {
            let Some(fields) = given else {
                return Err(crate::column::raise_pyspark(
                    py,
                    "PySparkValueError",
                    "CANNOT_INFER_EMPTY_SCHEMA",
                    &[],
                ));
            };
            let schema = DataType::Struct { fields };
            let df = py
                .detach(|| {
                    self.session
                        .create_dataframe(Vec::new(), schema)
                        .map_err(Box::new)
                })
                .to_pyerr()?;
            return Ok(PyDataFrame::new(df));
        }
        let mut items: Vec<Bound<'_, PyAny>> = data.try_iter()?.collect::<PyResult<_>>()?;
        let Some(first) = items.first().cloned() else {
            // As PySpark takes the first row of data without a length.
            return Err(pyo3::exceptions::PyIndexError::new_err(
                "list index out of range",
            ));
        };
        // As PySpark: dict rows are sorted by their keys, and rows that are none of a Row, a
        // tuple, a list, a dict or an object, such as 1, are rows of that single value.
        let single = if first.is_instance_of::<PyDict>() {
            for item in items.iter_mut() {
                if !item.is_none() {
                    *item = infer::sorted_dict(item)?;
                }
            }
            false
        } else {
            // A VariantVal of PySpark has a `__dict__`, so that it is taken as a row, which
            // PySpark's `to_row` rejects, as `row_values` does.
            !(first.is_instance_of::<pyo3::types::PyTuple>()
                || first.is_instance_of::<PyList>()
                || first.is_instance_of::<crate::values::PyVariantVal>()
                || first.hasattr("__dict__")?)
        };

        let fields = match given {
            Some(fields) => fields,
            None => {
                // The configurations are read only when a value depends on them.
                let confs = infer::LazyConfs::new(self.session.conf());
                let options = infer::Options::new(&confs);
                let mut merged: Option<DataType> = None;
                for item in &items {
                    let row_type = if single {
                        let items = infer::positional(py, vec![item.clone()], cols.as_mut());
                        infer::infer_fields(py, items, options)?
                    } else if item.is_instance_of::<crate::values::PyVariantVal>() {
                        // PySpark infers the type of its `__dict__`, then rejects the row.
                        return Err(infer::variant_row(py));
                    } else {
                        infer::infer_schema(item, cols.as_mut(), options)?
                    };
                    merged = Some(match merged {
                        None => row_type,
                        Some(merged) => infer::merge_type(py, merged, row_type)?,
                    });
                }
                match merged {
                    Some(schema) if infer::has_null_type(&schema) => {
                        return Err(crate::column::raise_pyspark(
                            py,
                            "PySparkValueError",
                            "CANNOT_DETERMINE_TYPE",
                            &[],
                        ));
                    }
                    Some(DataType::Struct { fields }) => fields,
                    _ => unreachable!("rows are inferred as structs"),
                }
            }
        };

        // As PySpark: every row is taken apart first, then each column is converted.
        let mut parts: Vec<Vec<Bound<'_, PyAny>>> = Vec::with_capacity(items.len());
        for item in &items {
            parts.push(if single {
                if fields.len() != 1 {
                    return Err(infer::axis_length_mismatch(py, fields.len(), 1));
                }
                vec![item.clone()]
            } else {
                infer::row_values(item, &fields)?
            });
        }
        let mut values: Vec<Vec<Value>> = (0..parts.len())
            .map(|_| Vec::with_capacity(fields.len()))
            .collect();
        {
            let _quanta = infer::DecimalQuanta::new();
            for (i, field) in fields.iter().enumerate() {
                for (row, out) in parts.iter().zip(values.iter_mut()) {
                    out.push(infer::convert(&row[i], &field.data_type, field.nullable)?);
                }
            }
        }
        // Then, as PySpark, the schema converts to Arrow, which rejects a null type that is not
        // nullable, and has no char, varchar, year-month or calendar interval type.
        if infer::has_non_nullable_null_type(&fields) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "A null type field may not be non-nullable",
            ));
        }
        for field in &fields {
            if let Some(err) = infer::unsupported_arrow_type(py, &field.data_type)? {
                return Err(err);
            }
        }
        // The names, with those that inference added, must name every column.
        if let Some(cols) = &cols {
            if cols.len() != fields.len() {
                return Err(infer::axis_length_mismatch(py, cols.len(), fields.len()));
            }
        }
        let names: std::sync::Arc<[String]> = fields.iter().map(|f| f.name.clone()).collect();
        let rows: Vec<Row> = values
            .into_iter()
            .map(|values| StructValue::new(names.clone(), values).into())
            .collect();
        let schema = DataType::Struct { fields };
        let df = py
            .detach(|| {
                self.session
                    .create_dataframe(rows, schema)
                    .map_err(Box::new)
            })
            .to_pyerr()?;
        // As pyspark, column names are applied with toDF. The local relation already carries
        // them, so this only changes the plan, which the server resolves into a Project that
        // is not local; it keeps isLocal() as in pyspark and is not redundant.
        Ok(PyDataFrame::new(match cols.take() {
            Some(cols) if !cols.is_empty() => df.to_df(cols.iter().map(String::as_str).collect()),
            _ => df,
        }))
    }

    /// A DataFrame of a non-empty pandas DataFrame under an explicit struct schema of the
    /// same width, converted as PySpark does: each column by its Arrow conversion (session
    /// time zone, safe casts), then the table cast to the schema, which the importer
    /// sends as Arrow. None when the width differs, which the row path reports.
    fn create_from_pandas_arrow(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        schema: &Bound<'_, PyAny>,
        fields: &[StructField],
    ) -> PyResult<Option<PyDataFrame>> {
        if fields.is_empty() || data.len()? == 0 || data.getattr("columns")?.len()? != fields.len()
        {
            return Ok(None);
        }
        // The caller's schema object is checked, before any configuration request is sent.
        let helper = py.import("pyspark.sql._pandas_input")?;
        if !helper
            .getattr("accepts_pandas_table")?
            .call1((data, schema))?
            .is_truthy()?
        {
            return Ok(None);
        }
        let options = py.detach(|| self.session.arrow_input_options().to_pyerr())?;
        // The option is always set, as the importer requires it.
        let timezone = options.timezone().unwrap_or("UTC").to_owned();
        let large = options.use_large_var_types() == Some("true");
        let safecheck = py
            .detach(|| {
                self.session
                    .conf()
                    .get("spark.sql.execution.pandas.convertToArrowArraySafely")
                    .map_err(Box::new)
            })
            .to_pyerr()?
            .as_deref()
            == Some("true");
        // The caller's StructType, which the guard checked, is converted as given; a DDL
        // string is parsed into the same fields.
        let schema_py = if schema
            .extract::<PyRef<crate::types::PyStructType>>()
            .is_ok()
        {
            schema.clone().unbind()
        } else {
            crate::types::data_type_to_py(
                py,
                &DataType::Struct {
                    fields: fields.to_vec(),
                },
            )?
        };
        let table = helper.getattr("prepare_pandas_table")?.call1((
            data,
            schema_py.bind(py),
            timezone,
            safecheck,
            large,
        ))?;
        // As PySpark Connect, the table, cast to the Arrow types of the schema of deduplicated
        // nested field names, is sent with the schema as given, which the server applies. It is
        // not normalized again as an Arrow table of the caller, which may not repeat the names
        // of the fields of a struct (DUPLICATED_FIELD_NAME_IN_ARROW_STRUCT).
        let spark_schema = crate::types::py_to_data_type(schema_py.bind(py))?;
        let batches = crate::pyarrow_import::table_batches(&table)?;
        let df = py
            .detach(|| {
                self.session
                    .create_dataframe_from_arrow_with_options(batches, Some(spark_schema), &options)
                    .map_err(Box::new)
            })
            .to_pyerr()?;
        // As the other pandas paths, the explicit names are applied with toDF.
        Ok(Some(PyDataFrame::new(df.to_df(
            fields.iter().map(|field| field.name.as_str()).collect(),
        ))))
    }

    /// A DataFrame of a pandas DataFrame without a schema, or of names, as PySpark Connect
    /// converts one: each column by PySpark's Arrow conversion, of the type that Arrow infers
    /// (`prepare_inferred_pandas_table`), sent without a schema, for the server to infer the
    /// types of the Arrow data. The columns are named with `toDF`: by their labels, as strings,
    /// or by the names, and `_n` for the n-th column past them.
    ///
    /// None for a frame that the row path converts: an empty one, a customized one
    /// (`accepts_pandas_frame`), and any without pyarrow, which the row path does not need.
    /// Without pyarrow, that path retains native Row struct inference; with it,
    /// customized Row columns can use Arrow after the frame's normalization hooks.
    fn create_from_pandas_inferred(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        spec: &Spec,
    ) -> PyResult<Option<PyDataFrame>> {
        let names = match spec {
            Spec::None => None,
            Spec::Names(names) => Some(names),
            Spec::Struct(_) | Spec::Atomic(_) => return Ok(None),
        };
        if data.len()? == 0 {
            return Ok(None);
        }
        let helper = py.import("pyspark.sql._pandas_input")?;
        if !helper.getattr("pyarrow_available")?.call0()?.is_truthy()?
            || !helper
                .getattr("accepts_pandas_frame")?
                .call1((data,))?
                .is_truthy()?
        {
            return Ok(None);
        }
        let width = data.getattr("columns")?.len()?;
        let mut columns: Vec<String> = match names {
            Some(names) => names.clone(),
            None => {
                let mut labels = Vec::with_capacity(width);
                for label in data.getattr("columns")?.try_iter()? {
                    let label = label?;
                    labels.push(match label.cast::<pyo3::types::PyString>() {
                        Ok(label) => label.to_cow()?.into_owned(),
                        Err(_) => label.str()?.to_cow()?.into_owned(),
                    });
                }
                if labels.is_empty() {
                    return Err(crate::column::raise_pyspark(
                        py,
                        "PySparkValueError",
                        "CANNOT_INFER_EMPTY_SCHEMA",
                        &[],
                    ));
                }
                labels
            }
        };
        if columns.len() < width {
            columns.extend((columns.len()..width).map(|i| format!("_{}", i + 1)));
        }
        let settings = self.pandas_arrow_settings(py, names.is_none())?;
        let (table, struct_names) = settings.prepare_table(&helper, data)?;
        if let Some(struct_names) = struct_names {
            columns = struct_names;
        }
        let actual: usize = table.getattr("num_columns")?.extract()?;
        if columns.len() != actual {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkValueError",
                "AXIS_LENGTH_MISMATCH",
                &[
                    ("expected_length", &columns.len().to_string()),
                    ("actual_length", &actual.to_string()),
                ],
            ));
        }
        let batches = crate::pyarrow_import::table_batches(&table)?;
        let df = py
            .detach(|| {
                self.session
                    .create_dataframe_from_arrow_with_options(batches, None, &settings.options)
                    .map_err(Box::new)
            })
            .to_pyerr()?;
        Ok(Some(PyDataFrame::new(if columns.is_empty() {
            df
        } else {
            df.to_df(columns.iter().map(String::as_str).collect())
        })))
    }

    /// A DataFrame of a numpy array, as PySpark Connect's `createDataFrame` builds it: an Arrow
    /// table of a column of each column of the array, sent with the schema, if any, for the
    /// server to cast to. The columns are named by the given names, else `value` for a
    /// 1-dimensional array or an array of one column, else `_1`, `_2`, ...
    ///
    /// It raises `INVALID_NDARRAY_DIMENSION` for an array of another dimension, and
    /// `AXIS_LENGTH_MISMATCH` for names, or a schema, of another number of columns. An empty
    /// array is empty data, of the schema, if any.
    fn create_from_ndarray(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        spec: Spec,
    ) -> PyResult<PyDataFrame> {
        let ndim: usize = data.getattr("ndim")?.extract()?;
        if ndim != 1 && ndim != 2 {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkValueError",
                "INVALID_NDARRAY_DIMENSION",
                &[("dimensions", "1 or 2")],
            ));
        }
        if data.len()? == 0 {
            return self.create_from_local(py, PyList::empty(py).as_any(), spec);
        }
        let width: usize = if ndim == 1 {
            1
        } else {
            data.getattr("shape")?.get_item(1)?.extract()?
        };
        let (names, num_cols, schema) = match spec {
            Spec::Names(names) => (Some(names), None, None),
            Spec::Struct(fields) => (None, Some(fields.len()), Some(DataType::Struct { fields })),
            Spec::Atomic(atomic) => (None, Some(1), Some(atomic)),
            Spec::None => (None, None, None),
        };
        let names = names.unwrap_or_else(|| {
            if width == 1 {
                vec!["value".to_string()]
            } else {
                (1..=width).map(|i| format!("_{i}")).collect()
            }
        });
        let axis_length_mismatch = |expected: usize| {
            crate::column::raise_pyspark(
                py,
                "PySparkValueError",
                "AXIS_LENGTH_MISMATCH",
                &[
                    ("expected_length", &expected.to_string()),
                    ("actual_length", &width.to_string()),
                ],
            )
        };
        if names.len() != width {
            return Err(axis_length_mismatch(names.len()));
        }
        let table = py
            .import("pyspark.sql._arrow_input")?
            .getattr("ndarray_table")?
            .call1((data, names))?;
        if let Some(expected) = num_cols.filter(|&expected| expected != width) {
            return Err(axis_length_mismatch(expected));
        }
        let batches = crate::pyarrow_import::table_batches(&table)?;
        let options = py.detach(|| self.session.arrow_input_options().to_pyerr())?;
        let df = py
            .detach(|| {
                self.session
                    .create_dataframe_from_arrow_with_options(batches, schema, &options)
                    .map_err(Box::new)
            })
            .to_pyerr()?;
        Ok(PyDataFrame::new(df))
    }

    /// A DataFrame of a pandas DataFrame: its rows as lists with NaN and NaT as None, typed
    /// by the first non-null value of each column, and its column labels, as strings, or the
    /// given names, applied with `toDF`.
    /// With PyArrow, schema-free columns containing Rows use whole-column Arrow inference
    /// and encoding, and the frame is sent through the Arrow importer.
    fn create_from_pandas(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        spec: Spec,
    ) -> PyResult<PyDataFrame> {
        // Without a struct schema, PySpark infers the Arrow type of each column, which has
        // no int64 for a Python int beyond the 64-bit range.
        let inferred = !matches!(spec, Spec::Struct(_));
        let infer_dict_as_map_allowed = matches!(spec, Spec::None);
        let mut columns = Vec::new();
        for label in data.getattr("columns")?.try_iter()? {
            columns.push(label?.str()?.extract::<String>()?);
        }
        // The dtype of each column, which types a column without values, as Arrow does.
        let mut dtypes = Vec::new();
        for dtype in data.getattr("dtypes")?.try_iter()? {
            dtypes.push(dtype?.str()?.extract::<String>()?);
        }
        // df.astype(object).where(df.notna(), None).values.tolist() -> rows with None for NA.
        let notna = data.call_method0("notna")?;
        let obj = data.call_method1("astype", ("object",))?;
        let replaced = obj.call_method1("where", (notna, py.None()))?;
        let values = replaced.getattr("values")?.call_method0("tolist")?;
        let data = values.cast_into::<PyList>()?;
        let empty = data.is_empty();
        let cannot_infer_empty = || {
            crate::column::raise_pyspark(py, "PySparkValueError", "CANNOT_INFER_EMPTY_SCHEMA", &[])
        };
        // The given fields, and the names that toDF applies, which empty data does not get.
        // The labels, which errors of a column name, as PySpark names its pandas.Series.
        let labels = columns.clone();
        let (given, names) = match spec {
            Spec::Atomic(atomic) if !empty => {
                // pyspark reports str() of the parsed type, which is its repr.
                let data_type = crate::types::data_type_to_py(py, &atomic)?
                    .bind(py)
                    .repr()?
                    .extract::<String>()?;
                return Err(crate::column::raise_pyspark(
                    py,
                    "PySparkTypeError",
                    "UNSUPPORTED_DATA_TYPE_FOR_ARROW",
                    &[("data_type", &data_type)],
                ));
            }
            Spec::Atomic(atomic) => (Some(vec![crate::infer::value_field(atomic)]), None),
            Spec::Struct(fields) => {
                let names = fields.iter().map(|f| f.name.clone()).collect();
                (Some(fields), (!empty).then_some(names))
            }
            _ if empty => return Err(cannot_infer_empty()),
            Spec::None if columns.is_empty() => return Err(cannot_infer_empty()),
            Spec::None => (None, Some(columns)),
            Spec::Names(mut names) => {
                // Extended with `_2`, `_3`, ... for more columns, as pyspark does.
                let width = columns.len();
                let start = names.len() + 1;
                names.extend((start..=width).map(|i| format!("_{i}")));
                (None, Some(names))
            }
        };
        let row_names: std::sync::Arc<[String]> = match (&given, &names) {
            (Some(fields), _) => fields.iter().map(|f| f.name.clone()).collect(),
            (None, Some(names)) => names.clone().into(),
            (None, None) => Vec::new().into(),
        };
        // Arrow treats Rows in an inferred pandas column as sequences, without reading
        // their field names. Adapt only those columns after the frame's normalization
        // hooks; explicit schemas and the other native column converters stay unchanged.
        let PandasRowColumns {
            columns: mut row_columns,
            selected: selected_row_columns,
            errors: mut column_errors,
            settings: row_settings,
            row_count,
        } = if given.is_none() {
            self.prepare_pandas_row_columns(
                py,
                &data,
                &row_names,
                &labels,
                &dtypes,
                infer_dict_as_map_allowed,
            )?
        } else {
            PandasRowColumns::empty(row_names.len(), data.len())
        };
        let has_row_columns = selected_row_columns.iter().any(|selected| *selected);
        let mut rows: Vec<Row> = Vec::with_capacity(row_count);
        for row in data.iter() {
            if has_row_columns && rows.len() >= row_count {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "normalized pandas row count changed during conversion",
                ));
            }
            let values = row
                .try_iter()?
                .enumerate()
                .map(|(column_index, v)| {
                    let v = v?;
                    if has_row_columns && column_index >= row_names.len() {
                        // Row detection validated plain list widths before column conversion.
                        // A hook can change a list before traversal discovers its final width.
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "normalized pandas row width changed during conversion",
                        ));
                    }
                    if selected_row_columns.get(column_index) == Some(&true) {
                        return Ok(Value::Null);
                    }
                    if column_errors.get(column_index).is_some_and(Option::is_some) {
                        // Do not run later scalar hooks in an already failing column.
                        return Ok(Value::Null);
                    }
                    let converted = (|| {
                        if inferred {
                            if let Ok(i) = v.cast_exact::<pyo3::types::PyInt>() {
                                if i.extract::<i64>().is_err() {
                                    return Err(pyo3::exceptions::PyOverflowError::new_err(
                                        "Python int too large to convert to C long",
                                    ));
                                }
                            }
                        }
                        py_to_value(&v)
                    })();
                    match converted {
                        Err(error)
                            if has_row_columns
                                && error.is_instance_of::<pyo3::exceptions::PyException>(py) =>
                        {
                            // Run native scalar conversion at most once per cell, then
                            // report errors in column order alongside Arrow failures.
                            // Decimal typing can still reread the original Python values.
                            if let Some(slot) = column_errors.get_mut(column_index) {
                                *slot = Some(error);
                                Ok(Value::Null)
                            } else {
                                Err(error)
                            }
                        }
                        other => other,
                    }
                })
                .collect::<PyResult<Vec<_>>>()?;
            if values.len() != row_names.len() {
                // As pyspark, which checks every row against the width of the schema.
                return Err(crate::infer::axis_length_mismatch(
                    py,
                    row_names.len(),
                    values.len(),
                ));
            }
            rows.push(StructValue::new(row_names.clone(), values).into());
        }
        if has_row_columns && rows.len() != row_count {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "normalized pandas row count changed during conversion",
            ));
        }
        let converted_row_count = rows.len();
        // Move the validated rows into columns once. Encoding each native column at
        // its field position preserves error priority over failures in later Row columns.
        let mut native_columns = if has_row_columns {
            let mut columns: Vec<Vec<Value>> = (0..row_names.len())
                .map(|_| Vec::with_capacity(converted_row_count))
                .collect();
            for row in std::mem::take(&mut rows) {
                for (column, value) in columns.iter_mut().zip(row.into_values()) {
                    column.push(value);
                }
            }
            Some(columns)
        } else {
            None
        };
        // A given struct wins; otherwise each field takes the type of its first non-null
        // value (empty input has a struct schema, checked above), but for these, which are
        // typed as Arrow types a pandas column, as PySpark does: a column of Decimals takes
        // the precision and scale of all its values, and a column without values the type of
        // its dtype. With PyArrow, Row columns already carry whole-column Arrow types.
        let mut arrow_columns = Vec::new();
        let mut arrow_fields = Vec::new();
        let fields = match given {
            Some(fields) => fields,
            None => {
                let mut fields: Vec<StructField> = Vec::with_capacity(row_names.len());
                for (i, name) in row_names.iter().enumerate() {
                    if let Some(error) = column_errors[i].take() {
                        return Err(error);
                    }
                    if let Some(PandasArrowColumn {
                        data_type,
                        arrays,
                        field: arrow_field,
                    }) = row_columns[i].take()
                    {
                        fields.push(StructField {
                            name: name.clone(),
                            data_type,
                            nullable: arrow_field.is_nullable(),
                            metadata: Default::default(),
                        });
                        arrow_columns.push(arrays);
                        arrow_fields.push(arrow_field.with_name(name.clone()));
                        continue;
                    }
                    // As PySpark, whose Arrow conversion of the column fails for a null map key.
                    let cells = || pandas_column_values(&rows, native_columns.as_deref(), i);
                    if cells().any(has_null_map_key) {
                        return Err(PyErr::from_value(
                            py.import("pyspark.errors")?
                                .getattr("PySparkTypeError")?
                                .call1((format!(
                                    "Exception thrown when converting pandas.Series ({}) with name \
                                     '{}' to Arrow Array (None).",
                                    dtypes.get(i).map_or("object", String::as_str),
                                    labels.get(i).unwrap_or(name),
                                ),))?,
                        ));
                    }
                    let first = cells().find(|value| !matches!(value, Value::Null));
                    let data_type = if !cells().any(|v| matches!(v, Value::Decimal { .. })) {
                        match first {
                            Some(value) => {
                                let mut data_type = value_to_datatype(value);
                                if has_decimal_type(&data_type) {
                                    let column: Vec<&Value> = cells().collect();
                                    widen_pandas_decimal_scales(&mut data_type, &column)?;
                                }
                                data_type
                            }
                            None => {
                                pandas_dtype_type(dtypes.get(i).map_or("object", String::as_str))
                            }
                        }
                    } else if cells().all(|v| matches!(v, Value::Decimal { .. } | Value::Null)) {
                        pandas_decimal_type(&data, i, cells())?
                    } else {
                        // Decimals with other values, which PySpark's conversion leaves to
                        // Arrow: it types the column, takes ints as Decimals and raises for
                        // the others, so the column is typed and converted by Arrow here too.
                        let column = data
                            .iter()
                            .map(|row| row.get_item(i))
                            .collect::<PyResult<Vec<_>>>()?;
                        let typed = py
                            .import("pyspark.sql._pandas_input")?
                            .getattr("arrow_decimal_column")?
                            .call1((
                                column,
                                dtypes.get(i).map_or("object", String::as_str),
                                labels.get(i).unwrap_or(name),
                            ))?;
                        let (precision, scale, values): (i32, i32, Vec<Bound<'_, PyAny>>) =
                            typed.extract()?;
                        if native_columns.is_some() && values.len() != converted_row_count {
                            return Err(pyo3::exceptions::PyValueError::new_err(
                                "normalized pandas column has an unexpected row count",
                            ));
                        }
                        let values = values
                            .iter()
                            .map(|v| Ok(py_to_value(v)?))
                            .collect::<PyResult<Vec<Value>>>()?;
                        if let Some(columns) = native_columns.as_mut() {
                            columns[i] = values;
                        } else {
                            rows = std::mem::take(&mut rows)
                                .into_iter()
                                .zip(values)
                                .map(|(row, value)| {
                                    let mut cells = row.into_values();
                                    cells[i] = value;
                                    StructValue::new(row_names.clone(), cells).into()
                                })
                                .collect();
                        }
                        DataType::Decimal { precision, scale }
                    };
                    let field = StructField {
                        name: name.clone(),
                        data_type,
                        nullable: true,
                        metadata: Default::default(),
                    };
                    if let Some(columns) = native_columns.as_mut() {
                        self.check_pandas_decimal_column(
                            py,
                            columns[i].iter(),
                            &field,
                            PandasDecimalContext {
                                dtype: dtypes.get(i).map_or("object", String::as_str),
                                label: labels.get(i).unwrap_or(name),
                                safecheck: row_settings.as_ref().map(|settings| settings.safecheck),
                            },
                        )?;
                        let column_names: std::sync::Arc<[String]> = vec![name.clone()].into();
                        let column_rows: Vec<Row> = std::mem::take(&mut columns[i])
                            .into_iter()
                            .map(|value| StructValue::new(column_names.clone(), vec![value]).into())
                            .collect();
                        let column_schema = DataType::Struct {
                            fields: vec![field.clone()],
                        };
                        let batch = py
                            .detach(|| {
                                spark_connect::session::rows_to_arrow_batch(
                                    &column_rows,
                                    &column_schema,
                                )
                                .map_err(Box::new)
                            })
                            .to_pyerr()?;
                        arrow_columns.push(vec![batch.column(0).clone()]);
                        arrow_fields.push(batch.schema().field(0).clone());
                    }
                    fields.push(field);
                }
                fields
            }
        };
        if !has_row_columns {
            for (i, field) in fields.iter().enumerate() {
                self.check_pandas_decimal_column(
                    py,
                    rows.iter().filter_map(|row| row.get(i)),
                    field,
                    PandasDecimalContext {
                        dtype: dtypes.get(i).map_or("object", String::as_str),
                        label: labels.get(i).unwrap_or(&field.name),
                        safecheck: None,
                    },
                )?;
            }
        }
        let schema = DataType::Struct { fields };
        let df = if has_row_columns {
            // Columns may have different chunk boundaries. Slice their buffers at
            // shared boundaries without copying or narrowing variable-width arrays.
            let arrow_schema = std::sync::Arc::new(arrow::datatypes::Schema::new(arrow_fields));
            let mut chunks = vec![0usize; arrow_columns.len()];
            let mut offsets = vec![0usize; arrow_columns.len()];
            // Use the validated Rust row count, not a list that scalar hooks may change.
            let mut remaining = converted_row_count;
            let mut batches = Vec::new();
            while remaining > 0 {
                let mut length = remaining;
                for (i, column) in arrow_columns.iter().enumerate() {
                    while column
                        .get(chunks[i])
                        .is_some_and(|array| offsets[i] == array.len())
                    {
                        chunks[i] += 1;
                        offsets[i] = 0;
                    }
                    let array = column.get(chunks[i]).ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(
                            "normalized pandas column has an unexpected row count",
                        )
                    })?;
                    length = length.min(array.len() - offsets[i]);
                }
                let arrays = arrow_columns
                    .iter()
                    .enumerate()
                    .map(|(i, column)| column[chunks[i]].slice(offsets[i], length))
                    .collect();
                batches.push(
                    arrow::record_batch::RecordBatch::try_new(arrow_schema.clone(), arrays)
                        .map_err(|error| {
                            pyo3::exceptions::PyValueError::new_err(error.to_string())
                        })?,
                );
                for offset in &mut offsets {
                    *offset += length;
                }
                remaining -= length;
            }
            let settings = row_settings.ok_or_else(|| {
                pyo3::exceptions::PyRuntimeError::new_err("missing pandas Row Arrow options")
            })?;
            py.detach(|| {
                self.session
                    .create_dataframe_from_arrow_with_options(
                        batches,
                        Some(schema),
                        &settings.options,
                    )
                    .map_err(Box::new)
            })
            .to_pyerr()?
        } else {
            py.detach(|| {
                self.session
                    .create_dataframe(rows, schema)
                    .map_err(Box::new)
            })
            .to_pyerr()?
        };
        Ok(PyDataFrame::new(match names {
            Some(names) if !names.is_empty() => {
                df.to_df(names.iter().map(String::as_str).collect())
            }
            _ => df,
        }))
    }

    fn pandas_arrow_settings(
        &self,
        py: Python<'_>,
        allow_dict_as_map: bool,
    ) -> PyResult<PandasArrowSettings> {
        let options = py.detach(|| self.session.arrow_input_options().to_pyerr())?;
        let conf = |key: &'static str| {
            py.detach(|| self.session.conf().get(key).map_err(Box::new))
                .to_pyerr()
                .map(|value| value.as_deref() == Some("true"))
        };
        Ok(PandasArrowSettings {
            options,
            safecheck: conf("spark.sql.execution.pandas.convertToArrowArraySafely")?,
            infer_dict_as_map: allow_dict_as_map
                && conf("spark.sql.execution.pandas.inferPandasDictAsMap")?,
        })
    }

    /// Prepare only selected Row columns, retaining ordinary failures by column index.
    /// Non-Exception control flow propagates immediately, as it does in Python helpers.
    fn prepare_pandas_row_columns(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyList>,
        row_names: &[String],
        labels: &[String],
        dtypes: &[String],
        allow_dict_as_map: bool,
    ) -> PyResult<PandasRowColumns> {
        let mut prepared = PandasRowColumns::empty(row_names.len(), data.len());
        // Only a column of Python objects can hold a Row: walk the cells of those alone,
        // so that frames of numeric, boolean, string or datetime columns pay nothing.
        let candidates: Vec<usize> = (0..row_names.len())
            .filter(|&i| {
                matches!(
                    pandas_dtype_type(dtypes.get(i).map_or("object", String::as_str)),
                    DataType::Null
                )
            })
            .collect();
        if candidates.is_empty() {
            return Ok(prepared);
        }
        let helper = py.import("pyspark.sql._pandas_input")?;
        if !helper.getattr("pyarrow_available")?.call0()?.is_truthy()? {
            return Ok(prepared);
        }
        let columns: Vec<(usize, Bound<'_, PyAny>)> = helper
            .getattr("inferred_pandas_row_columns")?
            .call1((data, row_names.len(), candidates))?
            .extract()?;
        let Some((first_index, _)) = columns.first() else {
            return Ok(prepared);
        };
        for (index, _) in &columns {
            let selected = prepared.selected.get_mut(*index).ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err(
                    "inferred pandas Row column has an unexpected index",
                )
            })?;
            *selected = true;
        }
        let settings = match self.pandas_arrow_settings(py, allow_dict_as_map) {
            Ok(settings) => settings,
            Err(error) if error.is_instance_of::<pyo3::exceptions::PyException>(py) => {
                prepared.errors[*first_index] = Some(error);
                return Ok(prepared);
            }
            Err(error) => return Err(error),
        };
        for (index, values) in columns {
            let column = (|| -> PyResult<PandasArrowColumn> {
                if values.is_instance_of::<pyo3::exceptions::PyBaseException>() {
                    return Err(PyErr::from_value(values));
                }
                let (data_type, table) = settings.prepare_row_column(
                    &helper,
                    &values,
                    labels.get(index).unwrap_or(&row_names[index]),
                )?;
                let batches = crate::pyarrow_import::table_batches(&table)?;
                if batches.iter().any(|batch| batch.num_columns() != 1)
                    || batches.iter().map(|batch| batch.num_rows()).sum::<usize>()
                        != prepared.row_count
                {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "inferred pandas Row column has an unexpected shape",
                    ));
                }
                let field = batches[0].schema().field(0).clone();
                Ok(PandasArrowColumn {
                    data_type,
                    arrays: batches
                        .iter()
                        .map(|batch| batch.column(0).clone())
                        .collect(),
                    field,
                })
            })();
            match column {
                Ok(column) => prepared.columns[index] = Some(column),
                Err(error) if error.is_instance_of::<pyo3::exceptions::PyException>(py) => {
                    prepared.errors[index] = Some(error);
                }
                Err(error) => return Err(error),
            }
        }
        prepared.settings = Some(settings);
        Ok(prepared)
    }

    /// Keep a Decimal column exact, without invoking its original Python objects again.
    fn check_pandas_decimal_column<'a>(
        &self,
        py: Python<'_>,
        values: impl IntoIterator<Item = &'a Value>,
        field: &StructField,
        context: PandasDecimalContext<'_>,
    ) -> PyResult<()> {
        let DataType::Decimal { precision, scale } = field.data_type else {
            return Ok(());
        };
        let fits = |value: &Value| match value {
            Value::Decimal { value, .. } => {
                spark_connect::session::decimal_str_to_unscaled(value, precision, scale).is_ok()
            }
            _ => true,
        };
        if values.into_iter().all(fits) {
            return Ok(());
        }
        let PandasDecimalContext {
            dtype,
            label,
            safecheck,
        } = context;
        let mut message = format!(
            "Exception thrown when converting pandas.Series ({dtype}) with name '{label}' to \
             Arrow Array (decimal128({precision}, {scale}))."
        );
        // Row frames reuse the Arrow settings already read for the whole frame. Other
        // native frames retain their existing read-on-error behavior and default.
        let safecheck = safecheck.unwrap_or_else(|| {
            py.detach(|| {
                self.session
                    .conf()
                    .get("spark.sql.execution.pandas.convertToArrowArraySafely")
                    .map_err(Box::new)
            })
            .map_or(true, |value| value.is_none_or(|value| value == "true"))
        });
        if safecheck {
            message.push_str(
                " It can be caused by overflows or other unsafe conversions warned by Arrow. \
                 Arrow safe type check can be disabled by using SQL config \
                 `spark.sql.execution.pandas.convertToArrowArraySafely`.",
            );
        }
        Err(PyErr::from_value(
            py.import("pyspark.errors")?
                .getattr("PySparkValueError")?
                .call1((message,))?,
        ))
    }
}

/// Whether `obj` is an instance of `module.class`, without importing `module` when it is not
/// loaded yet: an instance can only exist once its module was imported. A `None` entry in
/// `sys.modules` blocks the import, so it counts as not loaded.
fn is_loaded_instance(
    py: Python<'_>,
    obj: &Bound<'_, PyAny>,
    module: &str,
    class: &str,
) -> PyResult<bool> {
    let modules = py.import("sys")?.getattr("modules")?;
    match modules.cast::<PyDict>()?.get_item(module)? {
        Some(m) if !m.is_none() => obj.is_instance(&m.getattr(class)?),
        _ => Ok(false),
    }
}

/// A `createDataFrame` schema, as PySpark Connect reads one.
enum Spec {
    None,
    /// Column names, which name the inferred fields and are applied with `toDF`.
    Names(Vec<String>),
    /// The fields of a struct type, or of a DDL string of one.
    Struct(Vec<StructField>),
    /// An atomic type, of the single column `value`.
    Atomic(DataType),
}

/// The schema argument of `createDataFrame`: None, a list or tuple of names, or a struct or
/// atomic type, given as a type or a DDL string. Other types raise `NOT_EXPECTED_TYPE`.
fn schema_spec(
    py: Python<'_>,
    session: &spark_connect::session::SparkSession,
    schema: Option<&Bound<'_, PyAny>>,
) -> PyResult<Spec> {
    let Some(schema) = schema.filter(|s| !s.is_none()) else {
        return Ok(Spec::None);
    };
    let not_expected = |arg_type: &str| {
        crate::column::raise_pyspark(
            py,
            "PySparkTypeError",
            "NOT_EXPECTED_TYPE",
            &[
                ("arg_name", "schema"),
                ("expected_type", "list, None or StructType"),
                ("arg_type", arg_type),
            ],
        )
    };
    if schema.is_instance_of::<PyList>() || schema.is_instance_of::<pyo3::types::PyTuple>() {
        let mut names = Vec::new();
        for name in schema.try_iter()? {
            let name = name?;
            names.push(match name.cast::<pyo3::types::PyString>() {
                Ok(name) => name.to_cow()?.into_owned(),
                // PySpark encodes a name that is not a string, which most values cannot do.
                Err(_) => name.call_method1("encode", ("utf-8",))?.str()?.to_string(),
            });
        }
        return Ok(Spec::Names(names));
    }
    if !schema.is_instance_of::<pyo3::types::PyString>()
        && !schema.is_instance_of::<crate::types::PyDataType>()
    {
        return Err(not_expected(&crate::infer::type_name(schema)?));
    }
    Ok(match crate::types::schema_data_type(py, session, schema)? {
        DataType::Struct { fields } => Spec::Struct(fields),
        atomic if crate::infer::is_atomic(&atomic) => Spec::Atomic(atomic),
        other => return Err(not_expected(&crate::infer::class_name(&other))),
    })
}

#[pymethods]
impl PySparkSession {
    /// The session builder, reached as `SparkSession.builder`: a new builder of each access, as
    /// PySpark's `classproperty`.
    #[classattr]
    fn builder() -> PyBuilderProperty {
        PyBuilderProperty
    }

    /// Create a DataFrame representing a range of integers.
    ///
    /// Mirrors `pyspark.sql.SparkSession.range(start, end=None, step=1, numPartitions=None)`:
    /// called with a single positional arg it means `range(0, start)`.
    /// Each argument is taken by its `int()`, as PySpark's `range` takes them.
    #[pyo3(signature = (start, end=None, step=RangeStep::Omitted, numPartitions=None))]
    #[allow(non_snake_case)]
    fn range(
        &self,
        start: &Bound<'_, PyAny>,
        end: Option<&Bound<'_, PyAny>>,
        step: RangeStep,
        numPartitions: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        let (start, end, step, num_partitions) = range_arguments(start, end, step, numPartitions)?;
        let df = self
            .session
            .range_full(start, end, step, num_partitions)
            .to_pyerr()?;
        Ok(PyDataFrame::new(df))
    }

    /// Execute a SQL query and return a DataFrame, as PySpark Connect does. `args` binds
    /// parameters: a list fills positional `?` parameters and a dict named `:name` ones, each
    /// value as a literal (a string arg is a string literal, never a column reference).
    ///
    /// Keyword arguments format the query, as `str.format` does, with the formatter of
    /// `pyspark.sql.connect.sql_formatter`: a DataFrame becomes the name of a reference that
    /// is sent with the query, a column reference its name, a string its SQL literal, and
    /// another value its string.
    #[pyo3(signature = (sqlQuery, args=None, **kwargs))]
    fn sql(
        slf: &Bound<'_, Self>,
        sqlQuery: &str,
        args: Option<Bound<'_, PyAny>>,
        kwargs: Option<Bound<'_, PyDict>>,
    ) -> PyResult<PyDataFrame> {
        fn to_lit(v: &Bound<'_, PyAny>) -> PyResult<spark_connect::expression::Expression> {
            use spark_connect::expression::{Expression, LiteralExpression};
            if let Ok(sv) = v.extract::<String>() {
                return Ok(Expression::literal(LiteralExpression::string(sv)));
            }
            Ok(crate::functions::to_column(v)?.expression().clone())
        }
        let py = slf.py();
        let mut pos: Vec<spark_connect::expression::Expression> = Vec::new();
        let mut named: std::collections::HashMap<String, spark_connect::expression::Expression> =
            std::collections::HashMap::new();
        if let Some(a) = args {
            if let Ok(d) = a.cast::<PyDict>() {
                for (k, v) in d.iter() {
                    // As PySpark, which asserts that a name is a string.
                    let k = k
                        .cast::<pyo3::types::PyString>()
                        .map_err(|_| pyo3::exceptions::PyAssertionError::new_err(()))?;
                    named.insert(k.to_cow()?.into_owned(), to_lit(&v)?);
                }
            } else if let Ok(lst) = a.cast::<PyList>() {
                for v in lst.iter() {
                    pos.push(to_lit(&v)?);
                }
            } else {
                return Err(crate::column::raise_pyspark(
                    py,
                    "PySparkTypeError",
                    "INVALID_TYPE",
                    &[
                        ("arg_name", "args"),
                        ("arg_type", &a.get_type().name()?.to_string()),
                    ],
                ));
            }
        }
        let mut query = sqlQuery.to_string();
        let mut references = Vec::new();
        if let Some(kwargs) = kwargs.filter(|kwargs| !kwargs.is_empty()) {
            let formatter = py
                .import("pyspark.sql.connect.sql_formatter")?
                .getattr("SQLStringFormatter")?
                .call1((slf,))?;
            query = formatter
                .call_method("format", (sqlQuery,), Some(&kwargs))?
                .extract()?;
            for view in formatter.getattr("_temp_views")?.try_iter()? {
                let (df, name): (PyRef<'_, PyDataFrame>, String) = view?.extract()?;
                references.push((df.dataframe.clone(), name));
            }
        }
        let session = slf.borrow().session.clone();
        // Within a pipeline flow function, the statement is not run here: the server analyzes
        // its relation in the pipeline once the flow is defined, as PySpark's server returns
        // the relation without running it within a `PipelineAnalysisContext`.
        if crate::pipelines::in_flow_function() {
            return Ok(PyDataFrame::new(
                session.sql_without_running(&query, pos, named, references),
            ));
        }
        // The statement runs to completion at once, so other Python threads go on meanwhile,
        // to interrupt it for one.
        let df = interruptible(py, &session, || {
            session
                .sql_with_references(&query, pos, named, references)
                .map_err(Box::new)
        })?;
        Ok(PyDataFrame::new(df))
    }

    /// Create a DataFrame from local data (a list of rows, a pandas DataFrame, or a pyarrow
    /// Table) and an optional schema, as in pyspark.
    ///
    /// Local rows are inferred, converted and named as PySpark Connect does it, in
    /// `crate::infer`. The columns of a pandas DataFrame, or the field names of its schema,
    /// are applied with `toDF`, and its fields take the type of their first non-null value.
    #[pyo3(signature = (data, schema=None, samplingRatio=None, verifySchema=None))]
    fn createDataFrame(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        schema: Option<&Bound<'_, PyAny>>,
        samplingRatio: Option<&Bound<'_, PyAny>>,
        verifySchema: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        if data.is_none() {
            return Err(pyo3::exceptions::PyAssertionError::new_err(()));
        }
        if data.is_instance_of::<PyDataFrame>() {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkTypeError",
                "INVALID_TYPE",
                &[("arg_name", "data"), ("arg_type", "DataFrame")],
            ));
        }
        // As PySpark Connect, which ignores both.
        for (name, value) in [
            ("samplingRatio", samplingRatio),
            ("verifySchema", verifySchema),
        ] {
            if value.is_some_and(|v| !v.is_none()) {
                py.import("warnings")?.call_method1(
                    "warn",
                    (format!(
                        "'{name}' is ignored. It is not supported with Spark Connect."
                    ),),
                )?;
            }
        }

        // A caller holding a Table, or a pandas DataFrame, has already imported its module.
        if is_loaded_instance(py, data, "pyarrow", "Table")? {
            return crate::pyarrow_import::create_dataframe(py, &self.session, data, schema);
        }
        let spec = schema_spec(py, &self.session, schema)?;
        if is_loaded_instance(py, data, "numpy", "ndarray")? {
            return self.create_from_ndarray(py, data, spec);
        }
        if is_loaded_instance(py, data, "pandas", "DataFrame")? {
            // Guarded integer columns share the existing Arrow importer. Schema
            // validation above and unsupported pandas conversions keep master's
            // normal ordering and row fallback.
            let table = py
                .import("pyspark.sql._pandas_input")?
                .getattr("prepare_int64_table")?
                .call1((data, schema))?;
            if !table.is_none() {
                let frame =
                    crate::pyarrow_import::create_dataframe(py, &self.session, &table, schema)?;
                // pandas applies explicit field names with toDF even though the
                // local relation already carries them. Empty input stays local.
                return Ok(match &spec {
                    Spec::Struct(fields) if !fields.is_empty() && table.len()? > 0 => {
                        PyDataFrame::new(
                            frame
                                .dataframe
                                .to_df(fields.iter().map(|field| field.name.as_str()).collect()),
                        )
                    }
                    _ => frame,
                });
            }
            let helper = py.import("pyspark.sql._pandas_input")?;
            let columns = helper.getattr("timestamp_columns")?.call1((data, schema))?;
            if !columns.is_none() {
                // Only a naive column needs the session time zone; aware columns carry theirs.
                let aware = helper
                    .getattr("timestamp_columns_are_aware")?
                    .call1((&columns,))?
                    .is_truthy()?;
                let timezone = if aware {
                    None
                } else {
                    py.detach(|| {
                        self.session
                            .conf()
                            .get("spark.sql.session.timeZone")
                            .to_pyerr()
                    })?
                };
                let table = helper
                    .getattr("prepare_timestamp_table")?
                    .call1((columns, schema, timezone))?;
                if !table.is_none() {
                    let frame =
                        crate::pyarrow_import::create_dataframe(py, &self.session, &table, schema)?;
                    return Ok(match &spec {
                        Spec::Struct(fields) if !fields.is_empty() && table.len()? > 0 => {
                            PyDataFrame::new(
                                frame.dataframe.to_df(
                                    fields.iter().map(|field| field.name.as_str()).collect(),
                                ),
                            )
                        }
                        _ => frame,
                    });
                }
            }
            let ipc = py
                .import("pyspark.sql._pandas_input")?
                .getattr("prepare_float64_input")?
                .call1((data, schema))?;
            if !ipc.is_none() {
                let fields = match &spec {
                    Spec::Struct(fields) => fields,
                    _ => {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "float64 input requires a struct schema",
                        ));
                    }
                };
                let schema = DataType::Struct {
                    fields: fields.clone(),
                };
                let ipc: Vec<u8> = ipc.extract()?;
                let frame = py
                    .detach(|| {
                        self.session
                            .create_dataframe_from_inline_arrow_ipc(ipc, schema)
                            .map_err(Box::new)
                    })
                    .to_pyerr()?;
                // Match pandas row input: explicit nonempty field names travel
                // through toDF as well as the local relation's logical schema.
                return Ok(PyDataFrame::new(
                    frame.to_df(fields.iter().map(|field| field.name.as_str()).collect()),
                ));
            }
            let ipc = py
                .import("pyspark.sql._pandas_input")?
                .getattr("prepare_string_input")?
                .call1((data, schema))?;
            // Columns kept in pandas' Arrow string storage come as an Arrow table, which the
            // importer sends, or uploads at the relation cache threshold.
            if !ipc.is_none() && !ipc.is_instance_of::<pyo3::types::PyBytes>() {
                let frame =
                    crate::pyarrow_import::create_dataframe(py, &self.session, &ipc, schema)?;
                return Ok(match &spec {
                    Spec::Struct(fields) if !fields.is_empty() && ipc.len()? > 0 => {
                        PyDataFrame::new(
                            frame
                                .dataframe
                                .to_df(fields.iter().map(|field| field.name.as_str()).collect()),
                        )
                    }
                    _ => frame,
                });
            }
            if !ipc.is_none() {
                let fields = match &spec {
                    Spec::Struct(fields) => fields,
                    _ => {
                        return Err(pyo3::exceptions::PyValueError::new_err(
                            "string input requires a struct schema",
                        ));
                    }
                };
                let schema = DataType::Struct {
                    fields: fields.clone(),
                };
                let ipc: Vec<u8> = ipc.extract()?;
                let frame = py
                    .detach(|| {
                        self.session
                            .create_dataframe_from_inline_arrow_ipc(ipc, schema)
                            .map_err(Box::new)
                    })
                    .to_pyerr()?;
                // Match pandas row input: explicit nonempty field names travel
                // through toDF as well as the local relation's logical schema.
                return Ok(PyDataFrame::new(
                    frame.to_df(fields.iter().map(|field| field.name.as_str()).collect()),
                ));
            }
            let table = py
                .import("pyspark.sql._pandas_input")?
                .getattr("prepare_date_table")?
                .call1((data, schema))?;
            if !table.is_none() {
                let frame =
                    crate::pyarrow_import::create_dataframe(py, &self.session, &table, schema)?;
                return Ok(match &spec {
                    Spec::Struct(fields) if !fields.is_empty() && table.len()? > 0 => {
                        PyDataFrame::new(
                            frame
                                .dataframe
                                .to_df(fields.iter().map(|field| field.name.as_str()).collect()),
                        )
                    }
                    _ => frame,
                });
            }
            // An explicit struct schema converts with PySpark's Arrow conversion, which
            // reports the same values and errors. Other schemas keep the row path.
            if let (Spec::Struct(fields), Some(schema)) = (&spec, schema) {
                if let Some(frame) = self.create_from_pandas_arrow(py, data, schema, fields)? {
                    return Ok(frame);
                }
            }
            if let Some(frame) = self.create_from_pandas_inferred(py, data, &spec)? {
                return Ok(frame);
            }
            return self.create_from_pandas(py, data, spec);
        }
        // Preserve schema parsing, ignored-argument warnings, and pandas/Arrow routing.
        // Only a native StructType opts into this explicit integer optimization;
        // DDL, inferred schemas and unsupported values keep schema-driven conversion.
        if schema.is_some_and(|s| s.extract::<PyRef<crate::types::PyStructType>>().is_ok()) {
            if let Spec::Struct(fields) = &spec {
                if let Ok(data) = data.cast::<PyList>() {
                    if let Some(columns) = exact_long_tuple_columns(data, fields) {
                        let schema = DataType::Struct {
                            fields: fields.clone(),
                        };
                        let df = py
                            .detach(|| {
                                self.session
                                    .create_dataframe_from_columns(columns, schema)
                                    .map_err(Box::new)
                            })
                            .to_pyerr()?;
                        return Ok(PyDataFrame::new(df));
                    }
                }
            }
        }
        self.create_from_local(py, data, spec)
    }

    /// Get the catalog API (`spark.catalog`).
    #[getter]
    #[pyo3(name = "catalog")]
    fn catalog(&self) -> PyCatalog {
        PyCatalog::new(self.session.catalog())
    }

    /// Runtime configuration (`spark.conf`).
    #[getter]
    #[pyo3(name = "conf")]
    fn conf(&self) -> crate::conf::PyRuntimeConf {
        crate::conf::PyRuntimeConf::new(self.session.conf())
    }

    /// Table-valued functions namespace (`spark.tvf`).
    #[getter]
    #[pyo3(name = "tvf")]
    fn tvf(&self) -> crate::tvf::PyTableValuedFunction {
        crate::tvf::PyTableValuedFunction::new(self.session.tvf())
    }

    /// Copy a local file to the Spark-managed filesystem (`copyFromLocalToFs`), with the errors
    /// of PySpark Connect's: `NO_SCHEMA_AND_DRIVER_DEFAULT_SCHEME` for a destination of a scheme,
    /// an `AssertionError` for a local path that is not a file, and for a destination that is
    /// not an absolute path.
    #[pyo3(name = "copyFromLocalToFs")]
    fn copy_from_local_to_fs(
        &self,
        py: Python<'_>,
        local_path: &str,
        dest_path: &str,
    ) -> PyResult<()> {
        let scheme: String = py
            .import("urllib.parse")?
            .call_method1("urlparse", (dest_path,))?
            .getattr("scheme")?
            .extract()?;
        if !scheme.is_empty() {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkValueError",
                "NO_SCHEMA_AND_DRIVER_DEFAULT_SCHEME",
                &[("arg_name", "dest_path")],
            ));
        }
        let is_file: bool = py
            .import("pathlib")?
            .getattr("Path")?
            .call1((local_path,))?
            .call_method0("absolute")?
            .call_method0("is_file")?
            .extract()?;
        if !is_file {
            return Err(pyo3::exceptions::PyAssertionError::new_err(
                "local path must be a file path.",
            ));
        }
        let is_absolute: bool = py
            .import("pathlib")?
            .getattr("Path")?
            .call1((dest_path,))?
            .call_method0("is_absolute")?
            .extract()?;
        if !is_absolute {
            return Err(pyo3::exceptions::PyAssertionError::new_err(
                "destination FS path must be an absolute path.",
            ));
        }
        py.detach(|| {
            self.session
                .copy_from_local_to_fs(local_path, dest_path)
                .map_err(Box::new)
        })
        .to_pyerr()
    }

    /// As PySpark Connect's: the attributes of the JVM of a classic session raise
    /// `JVM_ATTRIBUTE_NOT_SUPPORTED`, and the others are not attributes.
    fn __getattr__(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyAny>> {
        if ["_jsc", "_jconf", "_jvm", "_jsparkSession", "sparkContext"].contains(&name) {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkAttributeError",
                "JVM_ATTRIBUTE_NOT_SUPPORTED",
                &[("attr_name", name)],
            ));
        }
        Err(pyo3::exceptions::PyAttributeError::new_err(format!(
            "'SparkSession' object has no attribute '{name}'"
        )))
    }

    /// DataFrameReader for batch reads (`spark.read`).
    #[getter]
    #[pyo3(name = "read")]
    fn read(&self) -> crate::readwriter::PyDataFrameReader {
        crate::readwriter::PyDataFrameReader::new(self.session.read())
    }

    /// Get the DataStreamReader for reading streaming data (`spark.readStream`).
    #[getter]
    #[pyo3(name = "readStream")]
    fn read_stream(&self) -> PyDataStreamReader {
        let reader = self.session.read_stream();
        PyDataStreamReader::new(reader)
    }

    /// Get the StreamingQueryManager for managing active streaming queries (`spark.streams`):
    /// the same object at each access, as PySpark Connect's.
    ///
    /// The native manager is of the Rust-side listener bus of the session, so
    /// `addListener`/`removeListener`/`close` are implemented in the core (Rust clients get the
    /// feature too).
    #[getter]
    #[pyo3(name = "streams")]
    fn streams(&self, py: Python<'_>) -> PyResult<Py<PyStreamingQueryManager>> {
        if let Some(manager) = self.streams.get() {
            return Ok(manager.clone_ref(py));
        }
        let manager = Py::new(
            py,
            PyStreamingQueryManager::new(self.session.streams(), self.session.clone()),
        )?;
        Ok(self.streams.get_or_init(|| manager).clone_ref(py))
    }

    /// Stop this Spark session. Best-effort and idempotent, mirroring pyspark: a failure
    /// to release the session on the server is logged, not raised. Also clears the
    /// active session and the cached `SQLContext` if they belong to this session. Listener
    /// callbacks are not joined; the current event's listener snapshot may finish afterward.
    fn stop(&self, py: Python<'_>) -> PyResult<()> {
        crate::progress::forget_session(&self.session);
        crate::profiler::forget_session(&self.session);
        let released = if self
            .release_session_on_close
            .load(std::sync::atomic::Ordering::SeqCst)
        {
            py.detach(|| self.session.stop().map_err(Box::new))
        } else {
            // As PySpark's stop: without the release, the client is closed all the same.
            py.detach(|| self.session.close());
            Ok(())
        };
        if let Err(e) = released {
            let logger = py
                .import("logging")?
                .call_method1("getLogger", ("pyspark",))?;
            logger.call_method1(
                "warning",
                (format!(
                    "session.stop(): Session could not be released. Error: {e}"
                ),),
            )?;
        }
        // As PySpark's, a stopped session is no longer the default session, or the active
        // session of this thread.
        let sid = self.session.session_id();
        let is_this = |s: &Py<PySparkSession>| s.borrow(py).session.session_id() == sid;
        if default_slot(py).as_ref().is_some_and(is_this) {
            set_default_slot(py, None);
        }
        ACTIVE_SESSION.with(|active| {
            let mut active = active.borrow_mut();
            if active.as_ref().is_some_and(is_this) {
                *active = None;
            }
        });
        // Only touch the SQLContext cache if the module was ever imported (as pyspark does).
        let modules = py.import("sys")?.getattr("modules")?;
        if let Ok(ctx_mod) = modules.get_item("pyspark.sql.connect.context") {
            ctx_mod.call_method1("_reset_instantiated_context", (sid,))?;
        }
        Ok(())
    }

    /// Enable `with SparkSession.builder.(...).getOrCreate() as session:` syntax.
    fn __enter__(slf: Py<Self>) -> Py<Self> {
        slf
    }

    /// Stop the session when leaving a `with` block (mirrors pyspark 4.3.0).
    #[pyo3(signature = (exc_type=None, exc_value=None, traceback=None))]
    fn __exit__(
        &self,
        py: Python<'_>,
        exc_type: Option<Py<PyAny>>,
        exc_value: Option<Py<PyAny>>,
        traceback: Option<Py<PyAny>>,
    ) -> PyResult<()> {
        let _ = (exc_type, exc_value, traceback);
        self.stop(py)
    }

    /// Get the UDF registration API (`spark.udf`). Returns the
    /// `pyspark.sql.udf.UDFRegistration` bound to this session, so it can register
    /// Python UDFs and Java UDFs/UDAFs (registerJavaFunction / registerJavaUDAF).
    #[getter(udf)]
    fn get_udf<'a>(slf: &Bound<'a, Self>, py: Python<'a>) -> PyResult<Bound<'a, PyAny>> {
        let cls = py.import("pyspark.sql.udf")?.getattr("UDFRegistration")?;
        cls.call1((slf,))
    }

    /// Register a Java UDF/UDAF by class name (used by
    /// `UDFRegistration.registerJavaFunction` / `registerJavaUDAF`), of the return type
    /// `return_type`, as PySpark's `JavaUDF` sends it: a DDL string unparsed, for the server
    /// to parse, and a DataType as itself, else `UNSUPPORTED_OPERATION` of "data type ...".
    #[pyo3(name = "_registerJavaFunction", signature = (name, java_class_name, return_type=None, aggregate=false))]
    #[allow(non_snake_case)]
    fn register_java_function_py(
        &self,
        py: Python<'_>,
        name: &str,
        java_class_name: &str,
        return_type: Option<&Bound<'_, PyAny>>,
        aggregate: bool,
    ) -> PyResult<()> {
        let return_type = match return_type {
            None => None,
            Some(ddl) if ddl.is_instance_of::<pyo3::types::PyString>() => {
                Some(DataType::Unparsed {
                    data_type_string: ddl.extract()?,
                })
            }
            Some(data_type) if data_type.is_instance_of::<crate::types::PyDataType>() => {
                Some(crate::types::py_to_data_type(data_type)?)
            }
            Some(other) => {
                return Err(crate::column::raise_pyspark(
                    py,
                    "PySparkValueError",
                    "UNSUPPORTED_OPERATION",
                    &[("operation", &format!("data type {}", other.str()?))],
                ))
            }
        };
        interruptible(py, &self.session, || {
            self.session
                .register_java_function_with(name, java_class_name, return_type, aggregate)
                .map_err(Box::new)
        })
    }

    /// Register a Python UDF by name so it resolves in SQL (used by
    /// `UDFRegistration.register`). Mirrors the reference `client.register_udf`: build a
    /// `CommonInlineUserDefinedFunction` carrying a `PythonUDF` (cloudpickled command +
    /// output type + eval type) with NO arguments, and send it as the `RegisterFunction`
    /// command so the server's session function registry resolves `name` in SQL.
    #[pyo3(name = "_registerPythonUdf", signature = (name, return_type, eval_type, command, python_ver, deterministic=true))]
    #[allow(non_snake_case)]
    fn register_python_udf_py(
        &self,
        py: Python<'_>,
        name: &str,
        return_type: &Bound<'_, PyAny>,
        eval_type: i32,
        command: Vec<u8>,
        python_ver: String,
        deterministic: bool,
    ) -> PyResult<()> {
        let return_data_type = crate::types::py_to_data_type(return_type)?;
        let payload = PythonUDFPayload::new(return_data_type, eval_type, command, python_ver);
        let udf_expr = CommonInlineUserDefinedFunctionExpression::new(
            name.to_string(),
            deterministic,
            Vec::new(),
            payload,
        );
        interruptible(py, &self.session, || {
            self.session.register_function(udf_expr).map_err(Box::new)
        })
    }

    /// Register a Python UDTF by name so SQL calls it (used by `UDTFRegistration.register`).
    /// Mirrors the reference `client.register_udtf`: the `CommonInlineUserDefinedTableFunction`
    /// of a `PythonUDTF` (cloudpickled handler class, return type and eval type) with NO
    /// arguments, sent as the `register_table_function` command. As for a UDTF call, a DDL
    /// return type is left to the server to parse.
    #[pyo3(name = "_registerPythonUdtf", signature = (name, return_type, eval_type, command, python_ver, deterministic=false))]
    // SparkError is large, as the core crate allows for its API.
    #[allow(clippy::too_many_arguments, clippy::result_large_err)]
    fn register_python_udtf_py(
        &self,
        py: Python<'_>,
        name: &str,
        return_type: Option<&Bound<'_, PyAny>>,
        eval_type: i32,
        command: Vec<u8>,
        python_ver: String,
        deterministic: bool,
    ) -> PyResult<()> {
        let return_type = return_type
            .map(|value| match value.extract::<String>() {
                Ok(ddl) => Ok(DataType::Unparsed {
                    data_type_string: ddl,
                }),
                Err(_) => crate::types::py_to_data_type(value),
            })
            .transpose()?;
        interruptible(py, &self.session, || {
            self.session.register_table_function(
                name,
                return_type.as_ref(),
                eval_type,
                &command,
                &python_ver,
                deterministic,
            )
        })
    }

    #[pyo3(name = "sessionId")]
    fn session_id_camel(&self) -> String {
        self.session.session_id().to_string()
    }

    /// Add the extension of the bytes of a `google.protobuf.Any`, `extension`, to the user context
    /// of the requests of this session: of all the threads, if `is_global`, else of the current
    /// thread. Its ID: for the user context extensions of `client`.
    fn _add_user_context_extension(&self, extension: &[u8], is_global: bool) -> PyResult<String> {
        use prost::Message;
        let extension = prost_types::Any::decode(extension)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        Ok(if is_global {
            self.session.add_global_user_context_extension(extension)
        } else {
            self.session
                .add_threadlocal_user_context_extension(extension)
        })
    }

    /// The user context extensions of this session, the global ones if `is_global`, else those
    /// of the current thread, as their IDs and the bytes of each `google.protobuf.Any`: for
    /// `client`.
    fn _user_context_extensions<'py>(
        &self,
        py: Python<'py>,
        is_global: bool,
    ) -> Vec<(String, Bound<'py, pyo3::types::PyBytes>)> {
        use prost::Message;
        let extensions = if is_global {
            self.session.global_user_context_extensions()
        } else {
            self.session.threadlocal_user_context_extensions()
        };
        extensions
            .into_iter()
            .map(|(id, extension)| {
                (
                    id,
                    pyo3::types::PyBytes::new(py, &extension.encode_to_vec()),
                )
            })
            .collect()
    }

    /// Remove the user context extension of `extension_id`, for `client`.
    fn _remove_user_context_extension(&self, extension_id: &str) {
        self.session.remove_user_context_extension(extension_id);
    }

    /// Remove the user context extensions of the current thread and the global ones, for
    /// `client`.
    fn _clear_user_context_extensions(&self) {
        self.session.clear_user_context_extensions();
    }

    /// The responses of the server to `command`, the bytes of a `spark.connect.Command` built
    /// outside the client, such as the ML commands of `pyspark.ml`, as the bytes of each
    /// `ExecutePlanResponse`: for the `execute_command` of `client`.
    fn _execute_command<'py>(
        &self,
        py: Python<'py>,
        command: &[u8],
    ) -> PyResult<Vec<Bound<'py, pyo3::types::PyBytes>>> {
        use prost::Message;
        let command = spark_connect_proto::Command::decode(command)
            .map_err(|error| pyo3::exceptions::PyValueError::new_err(error.to_string()))?;
        let responses = interruptible(py, &self.session, || {
            self.session
                .execute_command_proto(command)
                .map_err(Box::new)
        })?;
        Ok(responses
            .iter()
            .map(|response| pyo3::types::PyBytes::new(py, &response.encode_to_vec()))
            .collect())
    }

    /// The client of PySpark Connect's `session.client`, as far as the modules of PySpark that
    /// build their own messages, such as `pyspark.ml`, use it: of their commands and of the ML
    /// cache, over the Rust transport (`pyspark.sql.connect.client`).
    #[getter]
    fn client(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        Ok(slf
            .py()
            .import("pyspark.sql.connect.client")?
            .getattr("_client_of")?
            .call1((slf,))?
            .unbind())
    }

    /// Release the session on the server, without stopping it, for `client.release_session`.
    fn _release_session(&self, py: Python<'_>) -> PyResult<()> {
        let session = self.session.clone();
        py.detach(|| session.release_session().map_err(Box::new))
            .to_pyerr()
    }

    /// Close the session without releasing it on the server, for `client.close`.
    /// Do not join listener callbacks; the current event's listener snapshot may finish afterward.
    fn _close(&self, py: Python<'_>) {
        py.detach(|| self.session.close());
    }

    /// The client of the session, as PySpark Connect's `SparkSession._client`, which the
    /// modules of PySpark read, such as the `TorchDistributor` of `pyspark.ml`: `client`.
    #[getter]
    fn _client(slf: &Bound<'_, Self>) -> PyResult<Py<PyAny>> {
        Self::client(slf)
    }

    /// The session id (property form, mirrors `SparkSession.session_id`).
    #[getter]
    fn session_id(&self) -> String {
        self.session.session_id().to_string()
    }

    /// Whether this session has been stopped. Mirrors `SparkSession.is_stopped`.
    #[getter]
    fn is_stopped(&self) -> bool {
        self.session.is_stopped()
    }

    /// `SparkSession(connection, userId=None, hook_factories=None)`, as PySpark Connect's: a
    /// session of the URL or channel builder `connection`, which is neither the default nor the
    /// active session, as PySpark's constructor does not make it either.
    #[new]
    #[pyo3(signature = (connection, userId=None, hook_factories=None))]
    #[allow(non_snake_case)]
    fn py_new(
        py: Python<'_>,
        connection: &Bound<'_, PyAny>,
        userId: Option<String>,
        hook_factories: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Py<Self>> {
        let url = match connection.extract::<String>() {
            Ok(url) => url,
            Err(_) => url_of_channel_builder(connection)?,
        };
        let url = match userId {
            Some(user) => with_user_id(&url, &user)?,
            None => url,
        };
        let session = py
            .detach(|| {
                SparkSession::builder()
                    .remote(&url)
                    .get_or_create()
                    .map_err(Box::new)
            })
            .to_pyerr()?;
        let session = session_object(py, &session)?;
        let factories = match hook_factories {
            Some(factories) => factories
                .try_iter()?
                .map(|factory| factory.map(Bound::unbind))
                .collect::<PyResult<Vec<_>>>()?,
            None => Vec::new(),
        };
        install_hook_factories(py, &factories, &session)?;
        log_client_banner(py);
        Ok(session)
    }

    /// Whether `stop()` releases the session on the server, as PySpark's
    /// `release_session_on_close`, which is True unless set otherwise.
    #[getter]
    fn release_session_on_close(&self) -> bool {
        self.release_session_on_close
            .load(std::sync::atomic::Ordering::SeqCst)
    }

    #[setter]
    fn set_release_session_on_close(&self, value: bool) {
        self.release_session_on_close
            .store(value, std::sync::atomic::Ordering::SeqCst);
    }

    /// The builder CLASS, reached as `SparkSession.Builder` (mirrors pyspark, which
    /// exposes the nested `Builder` type so `SparkSession.Builder()` works).
    #[classattr]
    #[allow(non_snake_case)]
    fn Builder(py: Python<'_>) -> Py<pyo3::types::PyType> {
        py.get_type::<PySparkSessionBuilder>().unbind()
    }

    /// The class attribute `_default_session`, the default session or None, as PySpark's:
    /// `_get_default_session` reads it, and `stop()` clears it. It is set as a class attribute
    /// (`set_default_slot`), so Python code that sets it changes the session that is default.
    #[classattr]
    fn _default_session(py: Python<'_>) -> Py<PyAny> {
        py.None()
    }

    /// The class attribute `_active_session`, whose `session` is the active session of the
    /// current thread, as PySpark's thread-local `SparkSession._active_session`.
    #[classattr]
    fn _active_session(py: Python<'_>) -> PyResult<Py<PyActiveSession>> {
        Py::new(py, PyActiveSession)
    }

    /// The class attribute `_lock`, PySpark's `threading.RLock` of the session registry, for the
    /// Python code that takes it. The drop-in serializes the creation of sessions with its own
    /// lock, as `getOrCreate` does with `_lock`.
    #[classattr]
    fn _lock(py: Python<'_>) -> PyResult<Py<PyAny>> {
        Ok(py.import("threading")?.call_method0("RLock")?.unbind())
    }

    /// The default session, unless it stopped, as PySpark's `_get_default_session`.
    #[staticmethod]
    #[pyo3(name = "_get_default_session")]
    fn get_default_session(py: Python<'_>) -> Option<Py<PySparkSession>> {
        default_session(py)
    }

    /// The host of the endpoint of the session, as PySpark's `SparkConnectClient.host`.
    fn _endpoint_host(&self) -> String {
        self.session.host().to_string()
    }

    /// The server-side session ID that the responses of the session carried, or None before the
    /// first response, as PySpark's `SparkConnectClient._server_session_id`.
    fn _observed_server_session_id(&self) -> Option<String> {
        self.session.server_session_id()
    }

    /// Set the server-side session ID that the session observed, as PySpark's
    /// `SparkConnectClient._server_session_id` is settable.
    fn _set_observed_server_session_id(&self, id: Option<String>) {
        self.session.set_server_session_id(id);
    }

    /// The execution hook base class, reached as `SparkSession.Hook` (mirrors pyspark).
    #[classattr]
    #[allow(non_snake_case)]
    fn Hook(py: Python<'_>) -> Py<pyo3::types::PyType> {
        py.get_type::<PyHook>().unbind()
    }

    /// UDTF registration accessor, mirroring `SparkSession.udtf`: the `UDTFRegistration`
    /// of this session, whose `register(name, f)` registers the UDTF `f` as `name`.
    #[getter]
    fn udtf<'a>(slf: &Bound<'a, Self>) -> PyResult<Bound<'a, PyAny>> {
        slf.py()
            .import("pyspark.sql.udtf")?
            .getattr("UDTFRegistration")?
            .call1((slf,))
    }

    /// The active session of this thread, or None. Mirrors `SparkSession.getActiveSession`.
    #[staticmethod]
    #[pyo3(name = "getActiveSession")]
    fn get_active_session(py: Python<'_>) -> Option<Py<PySparkSession>> {
        active_session(py)
    }

    /// The active session of this thread, if its ID is `session_id`, as PySpark Connect's
    /// `SparkSession._getActiveSessionIfMatches`, which gives back a pickled session: without
    /// an active session, as on a Python worker of the server, it raises `NO_ACTIVE_SESSION`,
    /// and for another one, `PySparkAssertionError`.
    #[staticmethod]
    #[pyo3(name = "_getActiveSessionIfMatches")]
    fn get_active_session_if_matches(
        py: Python<'_>,
        session_id: &str,
    ) -> PyResult<Py<PySparkSession>> {
        let Some(session) = active_session(py) else {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkRuntimeError",
                "NO_ACTIVE_SESSION",
                &[],
            ));
        };
        let active_id = session.borrow(py).session.session_id().to_string();
        if active_id != session_id {
            let error = py
                .import("pyspark.errors")?
                .getattr("PySparkAssertionError")?
                .call1((format!(
                    "Expected session ID does not match active session ID: \
                     {session_id} != {active_id}"
                ),))?;
            return Err(PyErr::from_value(error));
        }
        Ok(session)
    }

    /// A session pickles, as PySpark Connect's: as a function of its ID, which cloudpickle
    /// pickles by value, that gives back the active session of that ID
    /// (`_getActiveSessionIfMatches`).
    fn __reduce__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        py.import("pyspark.sql.connect.session")?
            .getattr("_reduce_session")?
            .call1((self.session.session_id().to_string(),))
    }

    /// The type of the DDL string `ddl`, as the server parses it, as PySpark Connect's
    /// `SparkSession._parse_ddl`, of a `DDLParse` of an `AnalyzePlanRequest`.
    fn _parse_ddl(&self, py: Python<'_>, ddl: &str) -> PyResult<Py<PyAny>> {
        let data_type = py
            .detach(|| self.session.parse_ddl(ddl).map_err(Box::new))
            .to_pyerr()?;
        crate::types::data_type_to_py(py, &data_type)
    }

    /// The active session of this thread, or else the default session; errors if neither.
    /// Mirrors `SparkSession.active`, which raises `NO_ACTIVE_OR_DEFAULT_SESSION`.
    #[staticmethod]
    fn active(py: Python<'_>) -> PyResult<Py<PySparkSession>> {
        active_session(py)
            .or_else(|| default_session(py))
            .ok_or_else(|| {
                crate::column::raise_pyspark(
                    py,
                    "PySparkRuntimeError",
                    "NO_ACTIVE_OR_DEFAULT_SESSION",
                    &[],
                )
            })
    }

    /// Register a handler of the progress of the executions of the session, as PySpark
    /// Connect's: once, called with the keyword arguments of `ProgressHandler` (`stages`,
    /// `inflight_tasks`, `operation_id` and `done`) for each progress of an execution, and
    /// once more, as done, when it ends.
    #[pyo3(name = "registerProgressHandler")]
    fn register_progress_handler(
        &self,
        py: Python<'_>,
        handler: &Bound<'_, PyAny>,
    ) -> PyResult<()> {
        crate::progress::session_progress(py, &self.session).register(py, handler)
    }

    /// Remove a progress handler, as PySpark Connect's: a `ValueError` for one that is not
    /// registered.
    #[pyo3(name = "removeProgressHandler")]
    fn remove_progress_handler(&self, py: Python<'_>, handler: &Bound<'_, PyAny>) -> PyResult<()> {
        crate::progress::session_progress(py, &self.session).remove(py, handler)
    }

    /// Remove all progress handlers. Mirrors `clearProgressHandlers`.
    #[pyo3(name = "clearProgressHandlers")]
    fn clear_progress_handlers(&self, py: Python<'_>) -> PyResult<()> {
        crate::progress::session_progress(py, &self.session).clear(py)
    }

    /// The UDF profiles of the session, as PySpark Connect's `spark.profile`: a `Profile` of its
    /// `ConnectProfilerCollector`, of `show`, `dump`, `render` and `clear`.
    #[getter]
    fn profile(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let collector = crate::profiler::profiler_collector(py, &self.session)?;
        Ok(py
            .import("pyspark.sql.profiler")?
            .getattr("Profile")?
            .call1((collector,))?
            .unbind())
    }

    /// The `ConnectProfilerCollector` of the session, as PySpark Connect's.
    #[getter(_profiler_collector)]
    fn profiler_collector(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        crate::profiler::profiler_collector(py, &self.session)
    }

    /// Get the data source registration accessor.
    ///
    /// Mirrors `SparkSession.dataSource.register`. Used to register custom data sources.
    #[getter]
    fn dataSource(&self, py: Python<'_>) -> PyResult<PyDataSourceRegistration> {
        Ok(PyDataSourceRegistration::new(session_object(
            py,
            &self.session,
        )?))
    }

    /// Build and register a resource profile with the server.
    ///
    /// Returns the server-assigned profile ID that can be used with `DataFrame.withResources()`.
    #[pyo3(name = "buildResourceProfile")]
    fn build_resource_profile(&self, py: Python<'_>, profile: &PyResourceProfile) -> PyResult<i32> {
        py.detach(|| {
            self.session
                .build_resource_profile(&profile.inner)
                .map_err(Box::new)
        })
        .to_pyerr()
    }

    #[getter]
    fn version(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.session.version().map_err(Box::new))
            .to_pyerr()
    }

    /// The DataFrame of a table, as PySpark Connect's: `NOT_EXPECTED_TYPE` for a name that is not
    /// a str.
    fn table(&self, tableName: &Bound<'_, PyAny>) -> PyResult<PyDataFrame> {
        let Ok(name) = tableName.cast::<pyo3::types::PyString>() else {
            return Err(crate::stat_arguments::not_expected_type(
                tableName,
                "tableName",
                "str",
            ));
        };
        Ok(PyDataFrame::new(
            self.session.table(&name.to_cow()?).to_pyerr()?,
        ))
    }

    // Reference: emptyDataFrame(schema) -> createDataFrame([], schema). We accept an
    // optional schema (a permissive superset): with a schema we build a typed empty frame
    // via createDataFrame; without one we return the core's schema-less empty frame.
    #[pyo3(name = "emptyDataFrame", signature = (schema=None))]
    fn empty_data_frame(
        &self,
        py: Python<'_>,
        schema: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        match schema {
            Some(s) => {
                let empty = PyList::empty(py);
                self.createDataFrame(py, empty.as_any(), Some(s), None, None)
            }
            None => Ok(PyDataFrame::new(
                self.session.empty_data_frame().to_pyerr()?,
            )),
        }
    }

    #[pyo3(name = "interruptAll")]
    fn interrupt_all(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        py.detach(|| self.session.interrupt_all().map_err(Box::new))
            .to_pyerr()
    }

    #[pyo3(name = "interruptTag")]
    fn interrupt_tag(&self, py: Python<'_>, tag: &str) -> PyResult<Vec<String>> {
        py.detach(|| self.session.interrupt_tag(tag).map_err(Box::new))
            .to_pyerr()
    }

    #[pyo3(name = "interruptOperation")]
    fn interrupt_operation(&self, py: Python<'_>, op_id: &str) -> PyResult<Vec<String>> {
        py.detach(|| self.session.interrupt_operation(op_id))
            .to_pyerr()
    }

    /// Add a tag of the current thread, with the errors of PySpark Connect's tags: `CANNOT_BE_NONE`
    /// for None, and those of [`SparkSession::check_tag`].
    #[pyo3(name = "addTag")]
    fn add_tag(&self, py: Python<'_>, tag: &Bound<'_, PyAny>) -> PyResult<()> {
        let tag = tag_of(py, tag)?;
        self.session.add_tag(&tag).to_pyerr()
    }

    /// Remove a tag of the current thread, with the errors of `addTag`, as PySpark Connect's.
    #[pyo3(name = "removeTag")]
    fn remove_tag(&self, py: Python<'_>, tag: &Bound<'_, PyAny>) -> PyResult<()> {
        let tag = tag_of(py, tag)?;
        SparkSession::check_tag(&tag).to_pyerr()?;
        self.session.remove_tag(&tag);
        Ok(())
    }

    /// Make `session` the default session, and the active session of this thread, when they
    /// are not set, as PySpark's, which `inheritable_thread_target` calls in a new thread.
    #[staticmethod]
    #[pyo3(name = "_set_default_and_active_session")]
    fn set_default_and_active_session(py: Python<'_>, session: Py<PySparkSession>) {
        set_default_and_active(py, &session);
    }

    /// The DDL of the schema `struct`, as PySpark Connect's `_to_ddl`: the server converts
    /// its JSON.
    #[pyo3(name = "_to_ddl")]
    fn to_ddl(&self, py: Python<'_>, r#struct: &Bound<'_, PyAny>) -> PyResult<String> {
        let json: String = r#struct.call_method0("json")?.extract()?;
        let session = self.session.clone();
        py.detach(|| session.json_to_ddl(&json).map_err(Box::new))
            .to_pyerr()
    }

    /// The tags of the current thread, a set, as PySpark's `getTags`.
    #[pyo3(name = "getTags")]
    fn get_tags<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PySet>> {
        pyo3::types::PySet::new(py, self.session.get_tags())
    }

    #[pyo3(name = "clearTags")]
    fn clear_tags(&self) {
        self.session.clear_tags()
    }

    /// A copy of the tags of the current thread, which `pyspark.util.inheritable_thread_target`
    /// makes when it decorates a function, as PySpark Connect's copies the `thread_local` of the
    /// client.
    fn _copy_thread_tags(&self) -> PySharedTags {
        PySharedTags(self.session.copy_thread_tags())
    }

    /// Make `tags` those of the current thread, which shares them with the other threads given
    /// them, as `pyspark.util.inheritable_thread_target` gives every thread that runs its
    /// function the copy of the `thread_local` of the client of PySpark Connect.
    fn _share_thread_tags(&self, tags: &PySharedTags) {
        self.session.share_thread_tags(&tags.0)
    }

    /// A copy of the user context extensions of the current thread, which
    /// `pyspark.util.inheritable_thread_target` makes when it decorates a function, as PySpark
    /// Connect's deep-copies the `thread_local` of the client.
    fn _copy_thread_user_context_extensions(&self) -> PySharedUserContextExtensions {
        PySharedUserContextExtensions(self.session.copy_threadlocal_user_context_extensions())
    }

    /// Make `extensions` the user context extensions of the current thread, which it shares with
    /// the other threads given them, as `pyspark.util.inheritable_thread_target` sets the copy of
    /// the `thread_local` of the client in each thread that runs its function.
    fn _share_thread_user_context_extensions(&self, extensions: &PySharedUserContextExtensions) {
        self.session
            .share_threadlocal_user_context_extensions(&extensions.0)
    }

    /// Add local files, or `file:` URIs, as artifacts of the session, as PySpark's
    /// `addArtifacts(*path, pyfile=False, archive=False, file=False)`: each path is named by
    /// the flags and its extension, and the errors for a path that is not supported are
    /// PySpark's (see `spark_connect_core::artifact::parse_artifact`). A class file is refused,
    /// as PySpark's ArtifactManager does not upload one.
    #[pyo3(name = "addArtifacts", signature = (*path, pyfile=false, archive=false, file=false))]
    fn add_artifacts(
        &self,
        py: Python<'_>,
        path: Vec<String>,
        pyfile: bool,
        archive: bool,
        file: bool,
    ) -> PyResult<()> {
        if [pyfile, archive, file].iter().filter(|flag| **flag).count() > 1 {
            return Err(crate::column::raise_pyspark(
                py,
                "PySparkValueError",
                "INVALID_MULTIPLE_ARGUMENT_CONDITIONS",
                &[
                    ("arg_names", "'pyfile', 'archive' and/or 'file'"),
                    ("condition", "True together"),
                ],
            ));
        }
        // As PySpark's _parse_artifacts, every path is parsed, and a local file opened, before
        // any upload.
        for p in &path {
            let (name, local) = match parse_artifact(p, pyfile, archive, file) {
                Ok(parsed) => parsed,
                Err(unsupported) => return Err(unsupported_artifact_error(py, unsupported)),
            };
            if name.starts_with("classes/") {
                let format = UnsupportedArtifact::Format(local.to_string_lossy().into_owned());
                return Err(unsupported_artifact_error(py, format));
            }
            check_local_artifact(py, &name, &local, pyfile)?;
        }
        let refs: Vec<&str> = path.iter().map(|s| s.as_str()).collect();
        py.detach(|| {
            self.session
                .add_artifacts_with(&refs, pyfile, archive, file)
                .map_err(Box::new)
        })
        .to_pyerr()
    }

    /// `addArtifact` is `addArtifacts`, as in PySpark.
    #[pyo3(name = "addArtifact", signature = (*path, pyfile=false, archive=false, file=false))]
    fn add_artifact(
        &self,
        py: Python<'_>,
        path: Vec<String>,
        pyfile: bool,
        archive: bool,
        file: bool,
    ) -> PyResult<()> {
        self.add_artifacts(py, path, pyfile, archive, file)
    }

    #[pyo3(name = "newSession")]
    fn new_session(&self, py: Python<'_>) -> PyResult<Py<PySparkSession>> {
        session_object(py, &self.session.new_session())
    }

    /// Clone this session on the server, as PySpark's `cloneSession(new_session_id=None)`: a
    /// session of a copy of the server-side state of this one, its SQL configurations,
    /// temporary views, registered functions and catalog state, of the id `new_session_id`,
    /// a UUID, if one is given.
    #[pyo3(name = "cloneSession", signature = (new_session_id=None))]
    fn clone_session(
        &self,
        py: Python<'_>,
        new_session_id: Option<String>,
    ) -> PyResult<Py<PySparkSession>> {
        let session = py
            .detach(|| {
                self.session
                    .clone_session_with_id(new_session_id.as_deref())
                    .map_err(Box::new)
            })
            .to_pyerr()?;
        session_object(py, &session)
    }
}

/// PySpark's error for a path of `addArtifacts` that it does not upload.
fn unsupported_artifact_error(py: Python<'_>, unsupported: UnsupportedArtifact) -> PyErr {
    match unsupported {
        UnsupportedArtifact::Format(local) => crate::column::raise_pyspark(
            py,
            "PySparkRuntimeError",
            "UNSUPPORTED_OPERATION",
            &[("operation", &format!("{local} file format"))],
        ),
        UnsupportedArtifact::Scheme(scheme) => crate::column::raise_pyspark(
            py,
            "PySparkRuntimeError",
            "UNSUPPORTED_OPERATION",
            &[("operation", &format!("{scheme} scheme"))],
        ),
        UnsupportedArtifact::HashInArchivePath => crate::column::raise_pyspark(
            py,
            "PySparkValueError",
            "VALUE_ALLOWED",
            &[("arg_name", "artifact path"), ("disallowed_value", "#")],
        ),
    }
}

/// The client side of parsing a local artifact, as PySpark's `_parse_artifacts` does it: a
/// Python file, zip, egg or jar of `pyfile` is importable by the client, and the file is opened
/// as PySpark's `LocalFile` opens it, so that a missing file or a directory raises the error of
/// `open` before any upload.
fn check_local_artifact(
    py: Python<'_>,
    name: &str,
    local: &std::path::Path,
    pyfile: bool,
) -> PyResult<()> {
    let local = local.to_string_lossy();
    let importable = pyfile && name.starts_with("pyfiles/");
    if importable && !name.ends_with(".py") {
        py.import("sys")?
            .getattr("path")?
            .call_method1("insert", (1, local.as_ref()))?;
    }
    py.import("builtins")?
        .call_method1("open", (local.as_ref(), "rb"))?
        .call_method0("close")?;
    if importable {
        py.import("importlib")?.call_method0("invalidate_caches")?;
    }
    Ok(())
}

/// Whether `value` has a map with a null key, at any depth.
fn has_null_map_key(value: &Value) -> bool {
    match value {
        Value::Map(entries) => entries
            .iter()
            .any(|(key, value)| matches!(key, Value::Null) || has_null_map_key(value)),
        Value::List(items) => items.iter().any(has_null_map_key),
        Value::Struct(row) => row.values().iter().any(has_null_map_key),
        _ => false,
    }
}

/// The type of a pandas column without values, by its dtype, as Arrow types it
/// (`pa.Schema.from_pandas`): `void` for an object column of None, as for one of another
/// dtype that Spark has no type for.
fn pandas_dtype_type(dtype: &str) -> DataType {
    match dtype {
        "float64" | "Float64" => DataType::Double,
        "float32" | "Float32" => DataType::Float,
        "int64" | "Int64" => DataType::Long,
        "int32" | "Int32" => DataType::Integer,
        "int16" | "Int16" => DataType::Short,
        "int8" | "Int8" => DataType::Byte,
        "bool" | "boolean" => DataType::Boolean,
        "string" | "str" => crate::infer::string_type(),
        dtype if dtype.starts_with("datetime64") => DataType::Timestamp,
        dtype if dtype.starts_with("timedelta64") => DataType::DayTimeInterval {
            start_field: 0,
            end_field: 3,
        },
        _ => DataType::Null,
    }
}

/// Read one pandas column from either retained rows or the transposed Row-frame values.
/// The row vector is empty when columns are present; these views never overlap.
fn pandas_column_values<'a>(
    rows: &'a [Row],
    columns: Option<&'a [Vec<Value>]>,
    index: usize,
) -> impl Iterator<Item = &'a Value> {
    rows.iter().filter_map(move |row| row.get(index)).chain(
        columns
            .into_iter()
            .flat_map(move |columns| columns[index].iter()),
    )
}

/// The decimal type of column `i` of pandas rows, as Arrow infers it from all its Decimals:
/// the most integer digits and the largest scale of the values.
fn pandas_decimal_type<'a>(
    data: &Bound<'_, PyList>,
    i: usize,
    converted: impl IntoIterator<Item = &'a Value>,
) -> PyResult<DataType> {
    // The digits and exponent of an exact C Decimal come from the text that `py_to_value`
    // took of it, rather than from `as_tuple()`, which builds a DecimalTuple of every digit.
    // Subclasses and pure-Python Decimals can override either, so keep their `as_tuple()`.
    let (mut integer_digits, mut scale) = (0i64, 0i64);
    let c_decimal = crate::infer::c_decimal_type(data.py())?;
    for (row, converted) in data.iter().zip(converted) {
        let value = row.get_item(i)?;
        if value.is_none() {
            continue;
        }
        let from_text = match converted {
            Value::Decimal { value: text, .. }
                if crate::infer::is_exact_c_decimal(&value, c_decimal) =>
            {
                decimal_text_digits(text)
            }
            _ => None,
        };
        let (digits, exponent) = match from_text {
            Some(parts) => parts,
            None => {
                let parts = value.call_method0("as_tuple")?;
                let digits = parts.getattr("digits")?.len()? as i64;
                // NaN and infinities have no exponent of digits, which Arrow cannot type either.
                let Ok(exponent) = parts.getattr("exponent")?.extract::<i64>() else {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Decimal {} has no precision to type a column by",
                        value.str()?
                    )));
                };
                (digits, exponent)
            }
        };
        let (precision, value_scale) = if exponent < 0 {
            (digits.max(-exponent), -exponent)
        } else {
            (digits + exponent, 0)
        };
        integer_digits = integer_digits.max(precision - value_scale);
        scale = scale.max(value_scale);
    }
    let precision = integer_digits + scale;
    if precision > 38 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Decimal precision {precision} is out of the range of DecimalType, up to 38"
        )));
    }
    Ok(DataType::Decimal {
        precision: precision.max(1) as i32,
        scale: scale as i32,
    })
}

/// Whether a type contains Decimal, to prune unrelated nested inference subtrees.
fn has_decimal_type(data_type: &DataType) -> bool {
    match data_type {
        DataType::Decimal { .. } => true,
        DataType::Array { element_type, .. } => has_decimal_type(element_type),
        DataType::Struct { fields } => fields
            .iter()
            .any(|field| has_decimal_type(&field.data_type)),
        DataType::Map {
            key_type,
            value_type,
            ..
        } => has_decimal_type(key_type) || has_decimal_type(value_type),
        _ => false,
    }
}

/// Preserve the existing nested shape and precision, but use every observed Decimal's
/// scale. Choosing only the first scale would make the exact encoder reject a later value
/// such as 1.25 after 1.5. This pass uses already converted values and invokes no Python hooks.
/// Callers prune types without Decimal before allocating child references.
/// Raises `ValueError` when widening increases a scale beyond its precision; an
/// unchanged, already-invalid first scale is left to the encoder.
fn widen_pandas_decimal_scales(data_type: &mut DataType, values: &[&Value]) -> PyResult<()> {
    match data_type {
        DataType::Decimal { precision, scale } => {
            let initial_scale = *scale;
            for value in values {
                if let Value::Decimal {
                    scale: Some(observed),
                    ..
                } = value
                {
                    *scale = (*scale).max(*observed);
                }
            }
            // Reject invalid types introduced by widening before the encoder can cast the
            // scale to i8. Keep the inherited handling of an already-invalid first value.
            if *scale > *precision && *scale > initial_scale {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Decimal scale is out of the range of DecimalType, up to {precision}"
                )));
            }
        }
        DataType::Array { element_type, .. } => {
            let children = values
                .iter()
                .filter_map(|value| match value {
                    Value::List(items) => Some(items.iter()),
                    _ => None,
                })
                .flatten()
                .collect::<Vec<_>>();
            widen_pandas_decimal_scales(element_type, &children)?;
        }
        DataType::Struct { fields } => {
            let in_order = values
                .iter()
                .map(|value| match value {
                    Value::Struct(row) => {
                        row.len() == fields.len()
                            && row
                                .fields()
                                .iter()
                                .zip(fields.iter())
                                .all(|(name, field)| *name == field.name)
                    }
                    _ => false,
                })
                .collect::<Vec<_>>();
            for (index, field) in fields.iter_mut().enumerate() {
                if !has_decimal_type(&field.data_type) {
                    continue;
                }
                let children = values
                    .iter()
                    .zip(&in_order)
                    .filter_map(|(value, &in_order)| match value {
                        Value::Struct(row) => {
                            if in_order {
                                row.get(index)
                            } else {
                                // Match the encoder: by field name, else by position,
                                // so widening inspects the same value it will encode.
                                row.get_by_name(&field.name).or_else(|| row.get(index))
                            }
                        }
                        // An earlier Row can establish the Struct type while a later
                        // dict in the same pandas column is converted to Value::Map.
                        Value::Map(entries) => entries.iter().find_map(|(key, value)| match key {
                            Value::String(name) if *name == field.name => Some(value),
                            _ => None,
                        }),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                widen_pandas_decimal_scales(&mut field.data_type, &children)?;
            }
        }
        DataType::Map {
            key_type,
            value_type,
            ..
        } => {
            if has_decimal_type(key_type) {
                let keys = values
                    .iter()
                    .filter_map(|value| match value {
                        Value::Map(entries) => Some(entries.iter()),
                        _ => None,
                    })
                    .flatten()
                    .map(|(key, _)| key)
                    .collect::<Vec<_>>();
                widen_pandas_decimal_scales(key_type, &keys)?;
            }
            if has_decimal_type(value_type) {
                let children = values
                    .iter()
                    .filter_map(|value| match value {
                        Value::Map(entries) => Some(entries.iter()),
                        _ => None,
                    })
                    .flatten()
                    .map(|(_, value)| value)
                    .collect::<Vec<_>>();
                widen_pandas_decimal_scales(value_type, &children)?;
            }
        }
        _ => {}
    }
    Ok(())
}

/// Infer the Spark `DataType` for a `createDataFrame` column from its (first) value.
/// Nested collections are inferred recursively; nulls and empty collections use
/// String as a fallback.
fn value_to_datatype(v: &Value) -> DataType {
    let utf8 = || DataType::String {
        collation: "UTF8_BINARY".to_string(),
    };
    match v {
        Value::Bool(_) => DataType::Boolean,
        Value::Byte(_) => DataType::Byte,
        Value::Short(_) => DataType::Short,
        Value::Integer(_) => DataType::Integer,
        Value::Long(_) => DataType::Long,
        Value::Float(_) => DataType::Float,
        Value::Double(_) => DataType::Double,
        Value::String(_) => utf8(),
        Value::Binary(_) => DataType::Binary,
        Value::Date(_) => DataType::Date,
        Value::Timestamp(_) => DataType::Timestamp,
        Value::TimestampNtz(_) => DataType::TimestampNtz,
        // As PySpark infers TimeType() of a datetime.time.
        Value::Time(_) => DataType::Time { precision: 6 },
        Value::DayTimeInterval(_) => DataType::DayTimeInterval {
            start_field: 0,
            end_field: 3,
        },
        Value::Decimal {
            precision, scale, ..
        } => DataType::Decimal {
            precision: precision.unwrap_or(38),
            scale: scale.unwrap_or(0),
        },
        // Nested inference (schema=None): infer element/field/value types recursively so a
        // list/dict/Row value produces the right Array/Struct/Map type.
        Value::List(items) => {
            let element_type = items
                .iter()
                .find(|x| !matches!(x, Value::Null))
                .map(value_to_datatype)
                .unwrap_or_else(utf8);
            DataType::Array {
                element_type: Box::new(element_type),
                contains_null: true,
            }
        }
        Value::Struct(row) => DataType::Struct {
            fields: row
                .fields()
                .iter()
                .zip(row.values())
                .map(|(n, val)| spark_connect::types::StructField {
                    name: n.clone(),
                    data_type: value_to_datatype(val),
                    nullable: true,
                    metadata: Default::default(),
                })
                .collect(),
        },
        Value::Map(m) => {
            // The type of the first non-null key or value, as the other fields take theirs.
            let first_type = |values: Vec<&Value>| {
                values
                    .into_iter()
                    .find(|x| !matches!(x, Value::Null))
                    .map(value_to_datatype)
                    .unwrap_or_else(utf8)
            };
            DataType::Map {
                key_type: Box::new(first_type(m.iter().map(|(k, _)| k).collect())),
                value_type: Box::new(first_type(m.iter().map(|(_, v)| v).collect())),
                value_contains_null: true,
            }
        }
        Value::Variant { .. } => DataType::Variant,
        Value::Null => utf8(),
    }
}

/// Build columns only for exact tuples of exact signed integers and None.
/// Unsupported containers, values, overflow, widths and non-nullable nulls use
/// schema-driven local conversion, preserving its validation and error ordering.
#[inline(never)]
fn exact_long_tuple_columns(
    data: &Bound<'_, PyList>,
    fields: &[StructField],
) -> Option<Vec<arrow::array::ArrayRef>> {
    use arrow::array::Int64Builder;
    use pyo3::types::{PyInt, PyTuple};
    use std::sync::Arc;

    if data.cast_exact::<PyList>().is_err()
        || data.is_empty()
        || fields.is_empty()
        || fields.iter().any(|field| field.data_type != DataType::Long)
    {
        return None;
    }
    let mut builders: Vec<_> = fields
        .iter()
        .map(|_| Int64Builder::with_capacity(data.len()))
        .collect();
    for item in data.iter() {
        let Ok(tuple) = item.cast_exact::<PyTuple>() else {
            return None;
        };
        if tuple.len() != builders.len() {
            return None;
        }
        for ((value, builder), field) in tuple.iter().zip(&mut builders).zip(fields) {
            if value.is_none() {
                if !field.nullable {
                    return None;
                }
                builder.append_null();
            } else if let Ok(integer) = value.cast_exact::<PyInt>() {
                let Ok(integer) = integer.extract::<i64>() else {
                    return None;
                };
                builder.append_value(integer);
            } else {
                return None;
            }
        }
    }
    Some(
        builders
            .iter_mut()
            .map(|builder| Arc::new(builder.finish()) as arrow::array::ArrayRef)
            .collect(),
    )
}

/// Convert a Python value to a Rust Value.
pub(crate) fn py_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    // Check for None
    if obj.is_none() {
        return Ok(Value::Null);
    }

    // Exact built-in strings cannot implement numeric conversion protocols.
    // Keep subclasses and strings that fail UTF-8 extraction on the generic path.
    if let Ok(s) = obj.cast_exact::<pyo3::types::PyString>() {
        if let Ok(s) = s.extract::<String>() {
            return Ok(Value::String(s));
        }
    }

    // Exact built-in numbers need neither bool coercion nor Decimal detection.
    // Keep subclasses and overflowing ints on the existing generic path.
    if let Ok(i) = obj.cast_exact::<pyo3::types::PyInt>() {
        if let Ok(i) = i.extract::<i64>() {
            return Ok(Value::Long(i));
        }
    }
    if let Ok(f) = obj.cast_exact::<pyo3::types::PyFloat>() {
        return Ok(Value::Double(f.value()));
    }

    // Exact built-in containers cannot implement numeric protocols. Convert them
    // before Decimal/numeric probing; subclasses retain the generic precedence below.
    if let Ok(list) = obj.cast_exact::<pyo3::types::PyList>() {
        let items = list
            .iter()
            .map(|item| py_to_value(&item))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(Value::List(items));
    }
    if let Ok(tuple) = obj.cast_exact::<pyo3::types::PyTuple>() {
        let items = tuple
            .iter()
            .map(|item| py_to_value(&item))
            .collect::<PyResult<Vec<_>>>()?;
        return Ok(Value::List(items));
    }

    // Exact dates cannot implement numeric protocols or override toordinal. Skip
    // the generic Decimal/numeric probes; subclasses (including datetime) must
    // retain the fallback ordering below. PyDate also supports abi3 wheels.
    if let Ok(date) = obj.cast_exact::<pyo3::types::PyDate>() {
        let ordinal: i64 = date.call_method0("toordinal")?.extract()?;
        return Ok(Value::Date((ordinal - 719163i64) as i32));
    }

    // Check for bool (before int because bool is a subclass of int in Python)
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(Value::Bool(b));
    }

    // Check for decimal.Decimal BEFORE int/float: Decimal defines __float__/__int__, so
    // extract::<f64>()/<i64>() would succeed and mis-tag a Decimal as Double/Long.
    if let Some(v) = decimal_to_value(obj)? {
        return Ok(v);
    }

    // Check for int
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::Long(i));
    }
    // An int beyond a long is not a double, as PySpark's literal of it: it is an error.
    if obj.is_instance_of::<pyo3::types::PyInt>() {
        return Err(crate::functions::value_not_between(obj.py()));
    }

    // Check for float
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(Value::Double(f));
    }

    // Check for str
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Value::String(s));
    }

    // Check for bytes/bytearray SPECIFICALLY (not `extract::<Vec<u8>>()`, which also
    // matches a Python list of small ints like [1,2,3] and would mis-tag arrays as binary).
    if let Ok(b) = obj.cast::<pyo3::types::PyBytes>() {
        return Ok(Value::Binary(b.as_bytes().to_vec()));
    }
    if let Ok(ba) = obj.cast::<pyo3::types::PyByteArray>() {
        return Ok(Value::Binary(ba.to_vec()));
    }

    // list / tuple -> array (recursively converted)
    if let Ok(list) = obj.cast::<pyo3::types::PyList>() {
        let mut items = Vec::with_capacity(list.len());
        for item in list.iter() {
            items.push(py_to_value(&item)?);
        }
        return Ok(Value::List(items));
    }
    if let Ok(tuple) = obj.cast::<pyo3::types::PyTuple>() {
        let mut items = tuple
            .iter()
            .map(|item| py_to_value(&item))
            .collect::<PyResult<Vec<_>>>()?;
        // Named Rows (tuple subclasses with `__fields__`) become structs on the generic
        // object path, e.g. customized pandas object columns. Unnamed tuples stay arrays.
        if !obj.is_exact_instance_of::<pyo3::types::PyTuple>() {
            match obj.getattr(pyo3::intern!(obj.py(), "__fields__")) {
                Ok(fields) => {
                    let mut fields: Vec<String> = fields.extract()?;
                    // Row factories may receive fewer values than names, and __fields__
                    // is mutable. Pair them as schema inference does, without panicking
                    // in the native Row constructor when their lengths differ.
                    fields.truncate(items.len());
                    items.truncate(fields.len());
                    return Ok(Value::Struct(Row::new(fields, items).into()));
                }
                Err(error)
                    if error.is_instance_of::<pyo3::exceptions::PyAttributeError>(obj.py()) => {}
                Err(error) => return Err(error),
            }
        }
        return Ok(Value::List(items));
    }

    // dict -> map, its entries in order with keys and values of their own types
    if let Ok(dict) = obj.cast::<pyo3::types::PyDict>() {
        let mut map = Vec::with_capacity(dict.len());
        for (k, v) in dict.iter() {
            map.push((py_to_value(&k)?, py_to_value(&v)?));
        }
        return Ok(Value::Map(map));
    }

    // Check for datetime.datetime (before datetime.date, since datetime is a subclass of date)
    let py = obj.py();
    let datetime_mod = py.import("datetime").ok();
    if let Some(dt_mod) = &datetime_mod {
        if let Ok(datetime_cls) = dt_mod.getattr("datetime") {
            if obj.is_instance(&datetime_cls)? {
                // A naive datetime represents session-local wall-clock; encode it via the
                // POSIX instant (.timestamp() interprets a naive value in the local zone).
                // A pandas Timestamp is a datetime subclass but its .timestamp() treats a
                // naive value as UTC, which disagrees with datetime/`lit(..)`; normalize it
                // to a pure datetime first so createDataFrame() and lit() encode identically.
                let micros = datetime_to_micros(obj)?;
                return Ok(Value::Timestamp(micros));
            }
        }
    }

    // datetime.time -> nanoseconds since midnight, as TimeType.toInternal.
    if let Some(dt_mod) = &datetime_mod {
        if let Ok(time_cls) = dt_mod.getattr("time") {
            if obj.is_instance(&time_cls)? {
                return Ok(Value::Time(crate::infer::time_nanos(obj)?));
            }
        }
    }

    // datetime.timedelta -> a day-time interval, as DayTimeIntervalType.toInternal.
    if let Some(dt_mod) = &datetime_mod {
        if let Ok(timedelta_cls) = dt_mod.getattr("timedelta") {
            if obj.is_instance(&timedelta_cls)? {
                return Ok(Value::DayTimeInterval(crate::infer::timedelta_micros(obj)?));
            }
        }
    }

    // Check for datetime.date (after datetime.datetime)
    if let Some(dt_mod) = &datetime_mod {
        if let Ok(date_cls) = dt_mod.getattr("date") {
            if obj.is_instance(&date_cls)? {
                // Get the ordinal and convert to days since epoch
                let ordinal: i64 = obj.call_method0("toordinal")?.extract()?;
                // 719163 is the ordinal of 1970-01-01
                let days = (ordinal - 719163i64) as i32;
                return Ok(Value::Date(days));
            }
        }
    }

    // Anything else is not a supported literal; error rather than silently coercing to a wrong value.
    let type_name = obj
        .get_type()
        .name()
        .map(|n| n.to_string())
        .unwrap_or_else(|_| "<unknown>".to_string());
    Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
        "unsupported value type for createDataFrame: {type_name}"
    )))
}

/// Encode a Python `datetime.datetime` as microseconds since the Unix epoch (UTC),
/// consistently for both plain `datetime` and `pandas.Timestamp`.
///
/// `datetime.timestamp()` interprets a naive value in the local zone (the convention Spark
/// uses for a naive value in a session-local TIMESTAMP, and the one `functions.lit(..)`
/// follows). `pandas.Timestamp.timestamp()` instead treats a naive value as UTC, which would
/// make `createDataFrame(pandas_df)` disagree with `lit(..)` by the local offset. To keep the
/// two paths identical we first downcast a pandas Timestamp to a plain `datetime` via
/// `to_pydatetime()`. Timezone-aware values are unaffected (both encode the same instant).
pub(crate) fn datetime_to_micros(obj: &Bound<'_, PyAny>) -> PyResult<i64> {
    let normalized = if obj.hasattr("to_pydatetime").unwrap_or(false) {
        obj.call_method0("to_pydatetime")?
    } else {
        obj.clone()
    };
    let timestamp_f64: f64 = normalized.call_method0("timestamp")?.extract()?;
    Ok((timestamp_f64 * 1_000_000.0).round() as i64)
}

/// If `obj` is an instance of the cached Decimal class (C `_decimal.Decimal`, else the
/// public fallback), return it as a `Value::Decimal` (string value + scale from its
/// exponent); otherwise `None`. See the class-rebinding entry in
/// `dev/parity/known-differences.md`. Checked before int/float in `py_to_value` because
/// Decimal defines `__float__`/`__int__`.
fn decimal_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Option<Value>> {
    let py = obj.py();
    // Cache the accelerator class or its absence. Interpreters without it use the
    // public Python class, whose exact instances retain the as_tuple hook path.
    let (c_decimal, decimal_cls) = crate::infer::decimal_classes(py)?;
    if !obj.is_instance(decimal_cls.as_any())? {
        return Ok(None);
    }
    // As written, in scientific notation for `1E+2` or `1.5E-7`: the Arrow encoder parses
    // either notation, in work bounded by the length of the text, which a fixed-point form
    // of `1E+1000000000` would not be.
    let dec_str: String = obj.str()?.extract()?;
    let scale = if crate::infer::is_exact_c_decimal(obj, c_decimal) {
        // Exact C Decimal strings preserve the coefficient's trailing zeros and exponent.
        // Subclasses and pure-Python Decimals can override __str__ or as_tuple, so keep
        // their tuple hooks.
        decimal_string_scale(&dec_str)
    } else {
        let as_tuple = obj.call_method0("as_tuple")?;
        let mut scale = None;
        if let Ok(exp_obj) = as_tuple.get_item(2) {
            if !exp_obj.is_none() {
                if let Ok(exp) = exp_obj.extract::<i64>() {
                    if exp <= 0 {
                        scale = Some((-exp) as i32);
                    }
                }
            }
        }
        scale
    };
    Ok(Some(Value::Decimal {
        value: dec_str,
        precision: None,
        scale,
    }))
}

/// The scale of `Decimal.as_tuple()`, from the text of an exact Decimal: the negated exponent,
/// or `None` for a positive exponent and for NaN and the infinities.
fn decimal_string_scale(value: &str) -> Option<i32> {
    let (_, exponent) = decimal_text_digits(value)?;
    // As the scale of `as_tuple()`, including its i32 cast.
    (exponent <= 0).then(|| (-exponent) as i32)
}

/// The number of digits and the exponent of `Decimal.as_tuple()`, from the text (`str()`) of
/// an exact Decimal, which keeps every digit of its coefficient and its exponent: `123.4500`
/// has 7 digits and the exponent -4, `1E+3` 1 and 3, `0.000` 1 and -3. `None` for NaN and the
/// infinities, which have no such exponent.
fn decimal_text_digits(value: &str) -> Option<(i64, i64)> {
    let unsigned = value.trim_start_matches(['-', '+']);
    if !unsigned.as_bytes().first()?.is_ascii_digit() {
        return None;
    }
    let (coefficient, exponent) = match unsigned.split_once(['E', 'e']) {
        Some((coefficient, exponent)) => (coefficient, exponent.parse::<i64>().ok()?),
        None => (unsigned, 0),
    };
    let (integer, fraction) = coefficient.split_once('.').unwrap_or((coefficient, ""));
    // The digits from the first nonzero one; zero has the single digit 0.
    let leading_zeros = integer.len() - integer.trim_start_matches('0').len();
    let leading_zeros = if leading_zeros == integer.len() {
        leading_zeros + (fraction.len() - fraction.trim_start_matches('0').len())
    } else {
        leading_zeros
    };
    let digits = (integer.len() + fraction.len() - leading_zeros).max(1) as i64;
    Some((digits, exponent - fraction.len() as i64))
}

/// The `step` of `range`, which PySpark defaults to 1, told apart from None, whose `int()`
/// raises.
pub(crate) enum RangeStep {
    Omitted,
    Given(Py<PyAny>),
}

impl<'a, 'py> FromPyObject<'a, 'py> for RangeStep {
    type Error = PyErr;

    fn extract(obj: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        Ok(RangeStep::Given(obj.to_owned().unbind()))
    }
}

/// The start, end, step and number of partitions of `spark.range` and `spark.tvf.range`, as
/// PySpark's `range` takes them: the `int()` of each, of the number of partitions first, with
/// `range(n)` from 0 to `n`. A value out of the range of its proto field is the `ValueError` of
/// protobuf.
pub(crate) fn range_arguments(
    start: &Bound<'_, PyAny>,
    end: Option<&Bound<'_, PyAny>>,
    step: RangeStep,
    num_partitions: Option<&Bound<'_, PyAny>>,
) -> PyResult<(i64, i64, i64, Option<i32>)> {
    let py = start.py();
    let int = |value: &Bound<'_, PyAny>| -> PyResult<i64> {
        let int = py.get_type::<pyo3::types::PyInt>().call1((value,))?;
        int.extract::<i64>().map_err(|_| value_out_of_range(&int))
    };
    let num_partitions = match num_partitions {
        None => None,
        Some(value) => {
            let int = py.get_type::<pyo3::types::PyInt>().call1((value,))?;
            Some(int.extract::<i32>().map_err(|_| value_out_of_range(&int))?)
        }
    };
    let (start, end) = match end {
        Some(end) => (int(start)?, int(end)?),
        None => (0, int(start)?),
    };
    let step = match step {
        RangeStep::Omitted => 1,
        RangeStep::Given(step) => int(step.bind(py))?,
    };
    Ok((start, end, step, num_partitions))
}

/// The `ValueError` of protobuf for an int out of the range of its field.
fn value_out_of_range(int: &Bound<'_, PyAny>) -> PyErr {
    let shown = int.str().map(|text| text.to_string()).unwrap_or_default();
    pyo3::exceptions::PyValueError::new_err(format!("Value out of range: {shown}"))
}

#[cfg(test)]
mod decimal_text_tests {
    use super::{decimal_string_scale, decimal_text_digits};

    #[test]
    fn coefficient_digits_and_exponent_match_decimal_tuple() {
        for (text, digits, exponent) in [
            ("0", 1, 0),
            ("0.000", 1, -3),
            ("-0E-7", 1, -7),
            ("1E+3", 1, 3),
            ("1.0E+4", 2, 3),
            ("123.4500", 7, -4),
            ("0.001234500", 7, -9),
            ("1.5E-7", 2, -8),
        ] {
            assert_eq!(
                decimal_text_digits(text),
                Some((digits, exponent)),
                "{text}"
            );
        }
    }

    #[test]
    fn scale_preserves_trailing_zeros_and_nonpositive_exponents() {
        for (text, scale) in [
            ("0", Some(0)),
            ("0.000", Some(3)),
            ("-0E-7", Some(7)),
            ("123.4500", Some(4)),
            ("1.5E-7", Some(8)),
            ("1E+3", None),
            ("1.0E+4", None),
        ] {
            assert_eq!(decimal_string_scale(text), scale, "{text}");
        }
    }

    #[test]
    fn nonfinite_values_have_no_numeric_exponent() {
        for text in ["NaN", "sNaN", "Infinity", "-Infinity"] {
            assert_eq!(decimal_text_digits(text), None, "{text}");
            assert_eq!(decimal_string_scale(text), None, "{text}");
        }
    }
}

#[cfg(test)]
mod connection_url_tests {
    use super::{quote_param, with_user_id};

    #[test]
    fn a_parameter_value_is_percent_encoded_outside_the_unreserved_characters() {
        assert_eq!(quote_param("a-b_c.d~e9"), "a-b_c.d~e9");
        assert_eq!(quote_param("a b;c=d%"), "a%20b%3Bc%3Dd%25");
    }

    #[test]
    fn user_id_is_added_to_the_url_unless_it_has_one() {
        assert_eq!(
            with_user_id("sc://h", "bob").unwrap(),
            "sc://h/;user_id=bob"
        );
        assert_eq!(
            with_user_id("sc://h/", "bob").unwrap(),
            "sc://h/;user_id=bob"
        );
        assert_eq!(
            with_user_id("sc://h/;token=t", "bob").unwrap(),
            "sc://h/;token=t;user_id=bob"
        );
        // The user ID of the connection string takes precedence, as PySpark's does.
        assert_eq!(
            with_user_id("sc://h/;user_id=ann", "bob").unwrap(),
            "sc://h/;user_id=ann"
        );
    }

    #[test]
    fn a_user_id_with_separators_reads_back_as_given() {
        let url = with_user_id("sc://h:114", "a b;c=d").unwrap();
        let parsed = spark_connect_core::channel::ChannelBuilder::parse(&url).unwrap();
        assert_eq!(parsed.get("user_id"), Some("a b;c=d"));
        assert_eq!(parsed.host(), "h");
        assert_eq!(parsed.port(), 114);
    }
}
