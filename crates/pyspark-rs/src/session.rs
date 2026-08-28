//! PyO3 wrapper for Spark Connect sessions.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use spark_connect::row::{Row, Value};
use spark_connect::session::SparkSession;
use spark_connect::types::{DataType, StructField};

use crate::catalog::PyCatalog;
use crate::dataframe::PyDataFrame;
use crate::datasource::PyDataSourceRegistration;
use crate::errors::ResultExt;
use crate::profiler::PyProfilerCollector;
use crate::resource::PyResourceProfile;
use crate::streaming::{PyDataStreamReader, PyStreamingQueryManager};

/// Python wrapper for a Spark session builder.
///
/// Mirrors `pyspark.sql.SparkSession.Builder`: chainable and reached via the
/// `SparkSession.builder` class attribute.
#[pyclass(name = "SparkSessionBuilder")]
#[derive(Clone)]
pub struct PySparkSessionBuilder {
    remote_url: Option<String>,
    /// Runtime configs set via `.config(...)`, applied after the session connects.
    configs: Vec<(String, String)>,
}

impl PySparkSessionBuilder {
    pub fn new() -> Self {
        PySparkSessionBuilder {
            remote_url: None,
            configs: Vec::new(),
        }
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
    fn remote(&self, url: &str) -> PySparkSessionBuilder {
        let mut b = self.clone();
        b.remote_url = Some(url.to_string());
        b
    }

    /// Set a config option, or several via `map`. Mirrors
    /// `SparkSession.Builder.config(key=None, value=None, *, map=None)`. `spark.remote`
    /// sets the connect endpoint; other keys are applied as runtime confs on connect.
    #[pyo3(signature = (key=None, value=None, map=None))]
    fn config(
        &self,
        key: Option<&str>,
        value: Option<&Bound<'_, PyAny>>,
        map: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<PySparkSessionBuilder> {
        let mut b = self.clone();
        if let Some(k) = key {
            let v = match value {
                Some(v) => crate::coerce_option_value(v)?.unwrap_or_default(),
                None => String::new(),
            };
            b.set_conf(k, v);
        }
        if let Some(m) = map {
            for (k, v) in m.iter() {
                let ks = k.str()?.to_string();
                let vs = crate::coerce_option_value(&v)?.unwrap_or_default();
                b.set_conf(&ks, vs);
            }
        }
        Ok(b)
    }

    /// `appName` - accepted for API parity; a Connect client cannot rename an
    /// already-running remote server, so this is a no-op (chainable), matching
    /// pyspark's Connect behavior.
    #[pyo3(name = "appName")]
    fn app_name(&self, _name: &str) -> PySparkSessionBuilder {
        self.clone()
    }

    /// `master` - not applicable to a remote Connect session (no local cluster to
    /// point at); accepted and ignored for API parity, like pyspark Connect.
    fn master(&self, _url: &str) -> PySparkSessionBuilder {
        self.clone()
    }

    /// `channelBuilder` - accept a Spark Connect ChannelBuilder and reconstruct an
    /// `sc://` connection URL from its endpoint (host/port) and connection params, then
    /// use it as the native-transport remote. Mirrors `SparkSession.Builder.channelBuilder`
    /// (custom gRPC channels/interceptors themselves are a Python-transport concept and
    /// do not apply to the native Rust transport, but the endpoint + params are honored).
    #[pyo3(name = "channelBuilder")]
    #[allow(non_snake_case)]
    fn channel_builder(
        &self,
        channelBuilder: &Bound<'_, PyAny>,
    ) -> PyResult<PySparkSessionBuilder> {
        // Endpoint: prefer the builder's host/port (DefaultChannelBuilder exposes both).
        let host: String = channelBuilder
            .getattr("host")
            .and_then(|h| h.extract::<String>())
            .map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "channelBuilder must expose a `host` (e.g. DefaultChannelBuilder); \
                     otherwise configure the endpoint via .remote(url)",
                )
            })?;
        let port: i64 = channelBuilder
            .getattr("port")
            .and_then(|p| p.extract::<i64>())
            .unwrap_or(15002);
        let mut url = format!("sc://{host}:{port}");
        // Preserve connection params (token/use_ssl/user_id/...) if present as `_params`.
        if let Ok(params) = channelBuilder.getattr("_params") {
            if let Ok(dict) = params.downcast::<pyo3::types::PyDict>() {
                if !dict.is_empty() {
                    let mut parts: Vec<String> = Vec::new();
                    for (k, v) in dict.iter() {
                        parts.push(format!("{}={}", k.str()?, v.str()?));
                    }
                    url.push_str("/;");
                    url.push_str(&parts.join(";"));
                }
            }
        }
        let mut b = self.clone();
        b.remote_url = Some(url);
        Ok(b)
    }

    /// `enableHiveSupport` - a no-op for a Connect client (the remote server's
    /// catalog is fixed); accepted for API parity.
    #[pyo3(name = "enableHiveSupport")]
    fn enable_hive_support(&self) -> PySparkSessionBuilder {
        self.clone()
    }

    /// Build and get or create the session (`getOrCreate`).
    #[pyo3(name = "getOrCreate")]
    fn get_or_create(&self, py: Python<'_>) -> PyResult<PySparkSession> {
        let url = self.remote_url.clone().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Must call .remote(url) before .getOrCreate()")
        })?;

        let builder = SparkSession::builder().remote(&url);
        let session = py.detach(|| builder.get_or_create()).to_pyerr()?;
        // Apply builder configs as runtime confs now that we have a session.
        let conf = session.conf();
        for (k, v) in &self.configs {
            py.detach(|| conf.set(k, v)).to_pyerr()?;
        }
        *active_slot().lock().unwrap() = Some(session.clone());
        Ok(PySparkSession::new(session))
    }

    /// `create()` - builds a brand-new session. For Connect this is equivalent to
    /// getOrCreate against the configured remote.
    fn create(&self, py: Python<'_>) -> PyResult<PySparkSession> {
        self.get_or_create(py)
    }
}

impl PySparkSessionBuilder {
    /// Record a config pair, capturing `spark.remote` as the endpoint.
    fn set_conf(&mut self, key: &str, value: String) {
        if key == "spark.remote" {
            self.remote_url = Some(value);
        } else {
            self.configs.push((key.to_string(), value));
        }
    }
}

/// Process-global "active" session (mirrors `SparkSession.active`/`getActiveSession`).
fn active_slot() -> &'static std::sync::Mutex<Option<SparkSession>> {
    static SLOT: std::sync::OnceLock<std::sync::Mutex<Option<SparkSession>>> =
        std::sync::OnceLock::new();
    SLOT.get_or_init(|| std::sync::Mutex::new(None))
}

/// Python wrapper for a Spark session.
#[pyclass(name = "SparkSession")]
pub struct PySparkSession {
    pub(crate) session: SparkSession,
}

impl PySparkSession {
    pub fn new(session: SparkSession) -> Self {
        PySparkSession { session }
    }
}

#[pymethods]
impl PySparkSession {
    /// The session builder, reached as `SparkSession.builder` (class attribute,
    /// matching pyspark).
    #[classattr]
    fn builder() -> PySparkSessionBuilder {
        PySparkSessionBuilder::new()
    }

    /// Create a DataFrame representing a range of integers.
    ///
    /// Mirrors `pyspark.sql.SparkSession.range(start, end=None, step=1, numPartitions=None)`:
    /// called with a single positional arg it means `range(0, start)`.
    #[pyo3(signature = (start, end=None, step=1, numPartitions=None))]
    #[allow(non_snake_case)]
    fn range(
        &self,
        start: i64,
        end: Option<i64>,
        step: i64,
        numPartitions: Option<i32>,
    ) -> PyResult<PyDataFrame> {
        let (start, end) = match end {
            Some(e) => (start, e),
            None => (0, start),
        };
        let df = self
            .session
            .range_full(start, end, step, numPartitions)
            .to_pyerr()?;
        Ok(PyDataFrame::new(df))
    }

    /// Execute a SQL query and return a DataFrame.
    fn sql(&self, query: &str) -> PyResult<PyDataFrame> {
        let df = self.session.sql(query).to_pyerr()?;
        Ok(PyDataFrame::new(df))
    }

    /// Create a DataFrame from a list of tuples/lists and an optional schema.
    ///
    /// If schema is a list of strings, those are used as column names (all types inferred as string).
    /// If schema is None, columns are named col0, col1, etc. and all types are string.
    #[pyo3(signature = (data, schema=None))]
    fn createDataFrame(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyList>,
        schema: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        use crate::row::PyRow;

        // Resolve the schema: a list of column names (types inferred), a DDL string or
        // a StructType (names + types explicit), or None (names from Row / default).
        enum Spec {
            None,
            Names(Vec<String>),
            Struct(Vec<StructField>),
        }
        let spec = match schema {
            None => Spec::None,
            Some(s) => {
                if let Ok(names) = s.extract::<Vec<String>>() {
                    Spec::Names(names)
                } else {
                    // Any DataType object (StructType, PyDataType) or a DDL string,
                    // resolved through the shared converter. A struct becomes the
                    // schema directly; a scalar AtomicType becomes a single "value"
                    // column (matching pyspark createDataFrame(data, AtomicType())).
                    match crate::types::py_to_data_type(s)? {
                        DataType::Struct { fields } => Spec::Struct(fields),
                        atomic => Spec::Struct(vec![spark_connect::types::StructField {
                            name: "value".to_string(),
                            data_type: atomic,
                            nullable: true,
                            metadata: std::collections::BTreeMap::new(),
                        }]),
                    }
                }
            }
        };

        let spec_names: Option<Vec<String>> = match &spec {
            Spec::Names(n) => Some(n.clone()),
            Spec::Struct(f) => Some(f.iter().map(|x| x.name.clone()).collect()),
            Spec::None => None,
        };

        // Parse each item into values; a list/tuple carries no names, a Row carries its own.
        let mut rows: Vec<Row> = vec![];
        for item in data.iter() {
            let (values, row_names): (Vec<Value>, Option<Vec<String>>) =
                if let Ok(pyrow) = item.extract::<PyRef<PyRow>>() {
                    (
                        pyrow.row.values().to_vec(),
                        Some(pyrow.row.fields().to_vec()),
                    )
                } else if let Ok(lst) = item.cast::<PyList>() {
                    (
                        lst.iter()
                            .map(|v| py_to_value(&v))
                            .collect::<PyResult<_>>()?,
                        None,
                    )
                } else if let Ok(tuple) = item.cast::<pyo3::types::PyTuple>() {
                    (
                        tuple
                            .iter()
                            .map(|v| py_to_value(&v))
                            .collect::<PyResult<_>>()?,
                        None,
                    )
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "Each row must be a list, tuple, or Row",
                    ));
                };
            let names = spec_names
                .clone()
                .or(row_names)
                .unwrap_or_else(|| (0..values.len()).map(|i| format!("col{}", i)).collect());
            rows.push(Row::new(names, values));
        }

        if rows.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Data is empty",
            ));
        }

        // An explicit struct schema wins; otherwise infer field types from the first row.
        let schema_dtype = match spec {
            Spec::Struct(fields) => DataType::Struct { fields },
            _ => {
                let fields: Vec<StructField> = rows[0]
                    .fields()
                    .iter()
                    .enumerate()
                    .map(|(i, name)| StructField {
                        name: name.clone(),
                        data_type: rows[0].get(i).map(value_to_datatype).unwrap_or(
                            DataType::String {
                                collation: "UTF8_BINARY".to_string(),
                            },
                        ),
                        nullable: true,
                        metadata: Default::default(),
                    })
                    .collect();
                DataType::Struct { fields }
            }
        };

        let df = py
            .detach(|| self.session.create_dataframe(rows, schema_dtype))
            .to_pyerr()?;
        Ok(PyDataFrame::new(df))
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

    /// Copy a local file to the Spark-managed filesystem (`copyFromLocalToFs`).
    #[pyo3(name = "copyFromLocalToFs")]
    fn copy_from_local_to_fs(
        &self,
        py: Python<'_>,
        local_path: &str,
        dest_path: &str,
    ) -> PyResult<()> {
        py.detach(|| self.session.copy_from_local_to_fs(local_path, dest_path))
            .to_pyerr()
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

    /// Get the StreamingQueryManager for managing active streaming queries (`spark.streams`).
    ///
    /// The native manager owns a Rust-side listener bus, so `addListener`/`removeListener`/
    /// `close` are implemented in the core (Rust clients get the feature too).
    #[getter]
    #[pyo3(name = "streams")]
    fn streams(&self) -> PyStreamingQueryManager {
        PyStreamingQueryManager::new(self.session.streams())
    }

    /// Stop this Spark session.
    fn stop(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| self.session.stop()).to_pyerr()
    }

    /// Get the UDF registration API. Returns a Python object that implements spark.udf.register.
    #[pyo3(name = "udf")]
    fn get_udf<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyAny>> {
        // Create a simple UDFRegistration wrapper object with a register method
        let code = c"
class _UDFRegistration:
    def __init__(self):
        pass

    def register(self, name, f, returnType=None):
        # Import locally to avoid circular imports
        from pyspark.sql.udf import UserDefinedFunction
        from pyspark.sql.types import StringType
        if returnType is None:
            returnType = StringType()
        return UserDefinedFunction(f, returnType, 100, name)

_UDFRegistration()
";
        let udf_reg = py.eval(code, None, None)?;
        Ok(udf_reg)
    }

    #[pyo3(name = "sessionId")]
    fn session_id_camel(&self) -> String {
        self.session.session_id().to_string()
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

    /// The builder CLASS, reached as `SparkSession.Builder` (mirrors pyspark, which
    /// exposes the nested `Builder` type so `SparkSession.Builder()` works).
    #[classattr]
    #[allow(non_snake_case)]
    fn Builder(py: Python<'_>) -> Py<pyo3::types::PyType> {
        py.get_type::<PySparkSessionBuilder>().unbind()
    }

    /// UDTF registration accessor, mirroring `SparkSession.udtf.register(name, cls)`.
    /// Returns a registration object whose `register` cloudpickles the Python UDTF
    /// class and returns a bound, name-registered table function.
    #[getter]
    fn udtf<'a>(&self, py: Python<'a>) -> PyResult<Bound<'a, PyAny>> {
        py.import("pyspark.sql.udtf")?
            .getattr("UDTFRegistration")?
            .call0()
    }

    /// The active session for this process, or None. Mirrors `SparkSession.getActiveSession`.
    #[staticmethod]
    #[pyo3(name = "getActiveSession")]
    fn get_active_session() -> Option<PySparkSession> {
        active_slot()
            .lock()
            .unwrap()
            .clone()
            .map(PySparkSession::new)
    }

    /// The active session; errors if none. Mirrors `SparkSession.active`.
    #[staticmethod]
    fn active() -> PyResult<PySparkSession> {
        Self::get_active_session().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("No active SparkSession found")
        })
    }

    /// Register a progress handler invoked on each execution-progress message.
    /// Returns an id for `removeProgressHandler`. Mirrors `registerProgressHandler`.
    #[pyo3(name = "registerProgressHandler")]
    fn register_progress_handler(&self, handler: Py<PyAny>) -> u64 {
        self.session.register_progress_handler(move |progress| {
            Python::attach(|py| {
                let dict = pyo3::types::PyDict::new(py);
                let _ = dict.set_item("num_inflight_tasks", progress.num_inflight_tasks);
                let stages = PyList::empty(py);
                for s in &progress.stages {
                    let sd = pyo3::types::PyDict::new(py);
                    let _ = sd.set_item("stage_id", s.stage_id);
                    let _ = sd.set_item("num_tasks", s.num_tasks);
                    let _ = sd.set_item("num_completed_tasks", s.num_completed_tasks);
                    let _ = sd.set_item("input_bytes_read", s.input_bytes_read);
                    let _ = sd.set_item("done", s.done);
                    let _ = stages.append(sd);
                }
                let _ = dict.set_item("stages", stages);
                // Best-effort: ignore handler errors so progress never breaks execution.
                let _ = handler.call1(py, (dict,));
            });
        })
    }

    /// Remove a progress handler by id. Mirrors `removeProgressHandler`.
    #[pyo3(name = "removeProgressHandler")]
    fn remove_progress_handler(&self, id: u64) {
        self.session.remove_progress_handler(id);
    }

    /// Remove all progress handlers. Mirrors `clearProgressHandlers`.
    #[pyo3(name = "clearProgressHandlers")]
    fn clear_progress_handlers(&self) {
        self.session.clear_progress_handlers();
    }

    /// Get the profiler collector for this session.
    ///
    /// Mirrors `SparkSession.profile`. Profile data is accumulated across query
    /// executions and can be shown, dumped, or cleared via the returned collector.
    #[getter]
    fn profile(&self) -> PyProfilerCollector {
        PyProfilerCollector::new(self.session.profiler())
    }

    /// Get the data source registration accessor.
    ///
    /// Mirrors `SparkSession.dataSource.register`. Used to register custom data sources.
    #[getter]
    fn dataSource(&self) -> PyDataSourceRegistration {
        PyDataSourceRegistration::new(PySparkSession::new(self.session.clone()))
    }

    /// Build and register a resource profile with the server.
    ///
    /// Returns the server-assigned profile ID that can be used with `DataFrame.withResources()`.
    #[pyo3(name = "buildResourceProfile")]
    fn build_resource_profile(&self, py: Python<'_>, profile: &PyResourceProfile) -> PyResult<i32> {
        py.detach(|| self.session.build_resource_profile(&profile.inner))
            .to_pyerr()
    }

    fn version(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.session.version()).to_pyerr()
    }

    fn table(&self, table_name: &str) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(self.session.table(table_name).to_pyerr()?))
    }

    #[pyo3(name = "emptyDataFrame")]
    fn empty_data_frame(&self) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(
            self.session.empty_data_frame().to_pyerr()?,
        ))
    }

    #[pyo3(name = "interruptAll")]
    fn interrupt_all(&self, py: Python<'_>) -> PyResult<Vec<String>> {
        py.detach(|| self.session.interrupt_all()).to_pyerr()
    }

    #[pyo3(name = "interruptTag")]
    fn interrupt_tag(&self, py: Python<'_>, tag: &str) -> PyResult<Vec<String>> {
        py.detach(|| self.session.interrupt_tag(tag)).to_pyerr()
    }

    #[pyo3(name = "interruptOperation")]
    fn interrupt_operation(&self, py: Python<'_>, operation_id: &str) -> PyResult<Vec<String>> {
        py.detach(|| self.session.interrupt_operation(operation_id))
            .to_pyerr()
    }

    #[pyo3(name = "addTag")]
    fn add_tag(&self, tag: &str) -> PyResult<()> {
        self.session.add_tag(tag).to_pyerr()
    }

    #[pyo3(name = "removeTag")]
    fn remove_tag(&self, tag: &str) {
        self.session.remove_tag(tag)
    }

    #[pyo3(name = "getTags")]
    fn get_tags(&self) -> Vec<String> {
        self.session.get_tags()
    }

    #[pyo3(name = "clearTags")]
    fn clear_tags(&self) {
        self.session.clear_tags()
    }

    // Mirrors reference `SparkSession.addArtifacts(*path)` - variadic positional paths.
    #[pyo3(name = "addArtifacts", signature = (*paths))]
    fn add_artifacts(&self, py: Python<'_>, paths: Vec<String>) -> PyResult<()> {
        let refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
        py.detach(|| self.session.add_artifacts(&refs)).to_pyerr()
    }

    #[pyo3(name = "addArtifact")]
    fn add_artifact(&self, py: Python<'_>, path: &str) -> PyResult<()> {
        py.detach(|| self.session.add_artifact(path)).to_pyerr()
    }

    #[pyo3(name = "newSession")]
    fn new_session(&self) -> PySparkSession {
        PySparkSession::new(self.session.new_session())
    }

    #[pyo3(name = "cloneSession")]
    fn clone_session(&self) -> PySparkSession {
        PySparkSession::new(self.session.clone_session())
    }
}

/// Infer the Spark `DataType` for a `createDataFrame` column from its (first) value.
/// Scalars, Date/Timestamp, and Decimal map precisely; Null and nested collections
/// fall back to String (createDataFrame here does not deep-infer nested schemas).
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
        Value::Decimal {
            precision, scale, ..
        } => DataType::Decimal {
            precision: precision.unwrap_or(38),
            scale: scale.unwrap_or(0),
        },
        _ => utf8(),
    }
}

/// Convert a Python value to a Rust Value.
pub(crate) fn py_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
    // Check for None
    if obj.is_none() {
        return Ok(Value::Null);
    }

    // Check for bool (before int because bool is a subclass of int in Python)
    if let Ok(b) = obj.extract::<bool>() {
        return Ok(Value::Bool(b));
    }

    // Check for int
    if let Ok(i) = obj.extract::<i64>() {
        return Ok(Value::Long(i));
    }

    // Check for float
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(Value::Double(f));
    }

    // Check for str
    if let Ok(s) = obj.extract::<String>() {
        return Ok(Value::String(s));
    }

    // Check for bytes
    if let Ok(b) = obj.extract::<Vec<u8>>() {
        return Ok(Value::Binary(b));
    }

    // list / tuple -> array (recursively converted)
    if let Ok(list) = obj.downcast::<pyo3::types::PyList>() {
        let mut items = Vec::with_capacity(list.len());
        for item in list.iter() {
            items.push(py_to_value(&item)?);
        }
        return Ok(Value::List(items));
    }
    if let Ok(tuple) = obj.downcast::<pyo3::types::PyTuple>() {
        let mut items = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            items.push(py_to_value(&item)?);
        }
        return Ok(Value::List(items));
    }

    // dict -> map (keys coerced to their string form, values recursively converted)
    if let Ok(dict) = obj.downcast::<pyo3::types::PyDict>() {
        let mut map = std::collections::BTreeMap::new();
        for (k, v) in dict.iter() {
            map.insert(k.str()?.to_string(), py_to_value(&v)?);
        }
        return Ok(Value::Map(map));
    }

    // Check for datetime.datetime (before datetime.date, since datetime is a subclass of date)
    let py = obj.py();
    let datetime_mod = py.import("datetime").ok();
    if let Some(dt_mod) = &datetime_mod {
        if let Ok(datetime_cls) = dt_mod.getattr("datetime") {
            if obj.is_instance(&datetime_cls)? {
                // Call .timestamp() to get POSIX seconds, then convert to microseconds
                let timestamp_f64: f64 = obj.call_method0("timestamp")?.extract()?;
                let micros = (timestamp_f64 * 1_000_000.0) as i64;
                return Ok(Value::Timestamp(micros));
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

    // Check for decimal.Decimal
    let decimal_mod = py.import("decimal").ok();
    if let Some(dec_mod) = &decimal_mod {
        if let Ok(decimal_cls) = dec_mod.getattr("Decimal") {
            if obj.is_instance(&decimal_cls)? {
                // Get the string representation
                let dec_str: String = obj.str()?.extract()?;
                // Try to get precision and scale from as_tuple()
                let as_tuple = obj.call_method0("as_tuple")?;
                let mut scale: Option<i32> = None;
                if let Ok(exp_obj) = as_tuple.get_item(2) {
                    if !exp_obj.is_none() {
                        if let Ok(exp) = exp_obj.extract::<i64>() {
                            if exp <= 0 {
                                scale = Some((-exp) as i32);
                            }
                        }
                    }
                }
                return Ok(Value::Decimal {
                    value: dec_str,
                    precision: None,
                    scale,
                });
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
