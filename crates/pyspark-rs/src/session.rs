//! PyO3 wrapper for Spark Connect sessions.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use spark_connect::row::{Row, Value};
use spark_connect::session::SparkSession;
use spark_connect::types::{DataType, StructField};
use spark_connect::udf::{CommonInlineUserDefinedFunctionExpression, PythonUDFPayload};

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
#[pyclass(
    name = "SparkSessionBuilder",
    module = "pyspark.sql.session",
    from_py_object
)]
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
    /// `SparkSession.Builder.config(key=None, value=None, *, map=None, conf=None)`.
    /// `spark.remote` sets the connect endpoint; other keys are applied as runtime confs
    /// on connect. `conf` accepts a `SparkConf` (its `getAll()` pairs are applied).
    #[pyo3(signature = (key=None, value=None, map=None, conf=None))]
    fn config(
        &self,
        key: Option<&str>,
        value: Option<&Bound<'_, PyAny>>,
        map: Option<&Bound<'_, PyDict>>,
        conf: Option<&Bound<'_, PyAny>>,
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
        if let Some(c) = conf {
            // A SparkConf: apply its (key, value) pairs via getAll().
            for pair in c.call_method0("getAll")?.try_iter()? {
                let pair = pair?;
                let k: String = pair.get_item(0)?.str()?.to_string();
                let v: String = pair.get_item(1)?.str()?.to_string();
                b.set_conf(&k, v);
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
            if let Ok(dict) = params.cast::<pyo3::types::PyDict>() {
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
#[pyclass(name = "SparkSession", module = "pyspark.sql.session")]
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

    /// Execute a SQL query and return a DataFrame. `args` binds parameters: a list fills
    /// positional `?` parameters, a dict fills named (`:name`) parameters. Values are always
    /// literals (a string arg is a string literal, never a column reference).
    #[pyo3(signature = (sqlQuery, args=None))]
    fn sql(&self, sqlQuery: &str, args: Option<Bound<'_, PyAny>>) -> PyResult<PyDataFrame> {
        fn to_lit(v: &Bound<'_, PyAny>) -> PyResult<spark_connect::expression::Expression> {
            use spark_connect::expression::{Expression, LiteralExpression};
            if let Ok(sv) = v.extract::<String>() {
                return Ok(Expression::Literal(LiteralExpression::string(sv)));
            }
            Ok(crate::functions::to_column(v)?.expression().clone())
        }
        let mut pos: Vec<spark_connect::expression::Expression> = Vec::new();
        let mut named: std::collections::HashMap<String, spark_connect::expression::Expression> =
            std::collections::HashMap::new();
        if let Some(a) = args {
            if let Ok(d) = a.cast::<PyDict>() {
                for (k, v) in d.iter() {
                    named.insert(k.extract::<String>()?, to_lit(&v)?);
                }
            } else if let Ok(lst) = a.cast::<PyList>() {
                for v in lst.iter() {
                    pos.push(to_lit(&v)?);
                }
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "args must be a list (positional) or dict (named)",
                ));
            }
        }
        let df = self
            .session
            .sql_with_args(sqlQuery, pos, named)
            .to_pyerr()?;
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
        data: &Bound<'_, PyAny>,
        schema: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyDataFrame> {
        use crate::row::PyRow;

        // Accept a pandas DataFrame (as pyspark does): convert it to a list of row lists
        // with NaN/NaT mapped to None, and take its column names as the default schema.
        // pandas-on-Spark always passes an explicit `schema`, so this mainly needs to
        // deliver the row values; the list path below handles the rest.
        let mut pandas_columns: Option<Vec<String>> = None;
        let is_pandas = py
            .import("pandas")
            .and_then(|m| m.getattr("DataFrame"))
            .and_then(|c| data.is_instance(&c))
            .unwrap_or(false);
        let data_owned: Bound<'_, PyList>;
        let data: &Bound<'_, PyList> = if is_pandas {
            pandas_columns = Some(data.getattr("columns")?.call_method0("tolist")?.extract()?);
            // df.astype(object).where(df.notna(), None).values.tolist() -> rows with None for NA.
            let notna = data.call_method0("notna")?;
            let obj = data.call_method1("astype", ("object",))?;
            let replaced = obj.call_method1("where", (notna, py.None()))?;
            let values = replaced.getattr("values")?.call_method0("tolist")?;
            data_owned = values.cast_into::<PyList>()?;
            &data_owned
        } else {
            data.cast::<PyList>()?
        };

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
                } else if let Ok(dict) = item.cast::<PyDict>() {
                    // A dict row: field names are its keys (sorted, mirroring pyspark's
                    // `_infer_schema` for dicts), values in that key order.
                    let mut pairs: Vec<(String, Bound<'_, PyAny>)> = Vec::with_capacity(dict.len());
                    for (k, v) in dict.iter() {
                        pairs.push((k.str()?.to_string(), v));
                    }
                    pairs.sort_by(|a, b| a.0.cmp(&b.0));
                    let names: Vec<String> = pairs.iter().map(|(k, _)| k.clone()).collect();
                    let values: Vec<Value> = pairs
                        .iter()
                        .map(|(_, v)| py_to_value(v))
                        .collect::<PyResult<_>>()?;
                    (values, Some(names))
                } else {
                    // A scalar row (int/str/float/...): a single-field row, matching
                    // `createDataFrame([1, 2, 3], IntegerType())`. The field name comes from
                    // the schema when given, else defaults below.
                    (vec![py_to_value(&item)?], None)
                };
            let names = spec_names
                .clone()
                .or(row_names)
                .or_else(|| pandas_columns.clone())
                .unwrap_or_else(|| (0..values.len()).map(|i| format!("col{}", i)).collect());
            rows.push(Row::new(names, values));
        }

        // An explicit struct schema wins; otherwise infer field types from the first row
        // (which requires at least one row -- an empty dataset needs an explicit schema,
        // matching pyspark which creates an empty DataFrame from schema + [] data).
        let schema_dtype = match spec {
            Spec::Struct(fields) => DataType::Struct { fields },
            _ => {
                if rows.is_empty() {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "can not infer schema from empty dataset",
                    ));
                }
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

    /// Get the UDF registration API (`spark.udf`). Returns the
    /// `pyspark.sql.udf.UDFRegistration` bound to this session, so it can register
    /// Python UDFs and Java UDFs/UDAFs (registerJavaFunction / registerJavaUDAF).
    #[getter(udf)]
    fn get_udf<'a>(slf: &Bound<'a, Self>, py: Python<'a>) -> PyResult<Bound<'a, PyAny>> {
        let cls = py.import("pyspark.sql.udf")?.getattr("UDFRegistration")?;
        cls.call1((slf,))
    }

    /// Register a Java UDF/UDAF by class name (used by
    /// `UDFRegistration.registerJavaFunction` / `registerJavaUDAF`).
    #[pyo3(name = "_registerJavaFunction", signature = (name, java_class_name, return_type=None, aggregate=false))]
    #[allow(non_snake_case)]
    fn register_java_function_py(
        &self,
        py: Python<'_>,
        name: &str,
        java_class_name: &str,
        return_type: Option<&str>,
        aggregate: bool,
    ) -> PyResult<()> {
        py.detach(|| {
            self.session
                .register_java_function(name, java_class_name, return_type, aggregate)
        })
        .to_pyerr()
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
        py.detach(|| self.session.register_function(udf_expr))
            .to_pyerr()
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

    /// The low-level Connect client. In the reference client this is the gRPC
    /// `SparkConnectClient`; here the transport is Rust, so we expose a minimal stub
    /// carrying the bits test/util code touches (`_server_session_id`, `_cleanup_ml_cache`),
    /// mirroring `SparkSession.client` enough for harnesses like `ReusedConnectTestCase`.
    #[getter]
    fn client(&self) -> PyConnectClientStub {
        PyConnectClientStub {
            session_id: self.session.session_id().to_string(),
        }
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
    fn remove_progress_handler(&self, handler: u64) {
        self.session.remove_progress_handler(handler);
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

    #[getter]
    fn version(&self, py: Python<'_>) -> PyResult<String> {
        py.detach(|| self.session.version()).to_pyerr()
    }

    fn table(&self, tableName: &str) -> PyResult<PyDataFrame> {
        Ok(PyDataFrame::new(self.session.table(tableName).to_pyerr()?))
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
    fn interrupt_operation(&self, py: Python<'_>, op_id: &str) -> PyResult<Vec<String>> {
        py.detach(|| self.session.interrupt_operation(op_id))
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

    #[pyo3(name = "addArtifact", signature = (*path, pyfile=false, archive=false, file=false))]
    fn add_artifact(
        &self,
        py: Python<'_>,
        path: Vec<String>,
        pyfile: bool,
        archive: bool,
        file: bool,
    ) -> PyResult<()> {
        // pyfile/archive/file are accepted for PySpark signature parity; artifact
        // classification is handled by the Rust transport / server.
        let _ = (pyfile, archive, file);
        let refs: Vec<&str> = path.iter().map(|s| s.as_str()).collect();
        py.detach(|| self.session.add_artifacts(&refs)).to_pyerr()
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

/// Minimal stand-in for the reference `SparkConnectClient`, returned by
/// `SparkSession.client`. The real client is the Rust transport; this only surfaces the
/// members that test/utility code (e.g. `ReusedConnectTestCase`) touches.
#[pyclass(name = "SparkConnectClientStub", module = "pyspark.sql.connect.client")]
pub struct PyConnectClientStub {
    session_id: String,
}

#[pymethods]
impl PyConnectClientStub {
    /// The server-side session id (property, mirrors `client._server_session_id`).
    #[getter]
    fn _server_session_id(&self) -> String {
        self.session_id.clone()
    }

    /// No-op: the Rust transport has no client-side ML cache to clear.
    fn _cleanup_ml_cache(&self) {}
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
        Value::Struct(fields) => DataType::Struct {
            fields: fields
                .iter()
                .map(|(n, val)| spark_connect::types::StructField {
                    name: n.clone(),
                    data_type: value_to_datatype(val),
                    nullable: true,
                    metadata: Default::default(),
                })
                .collect(),
        },
        Value::Map(m) => {
            let value_type = m
                .values()
                .find(|x| !matches!(x, Value::Null))
                .map(value_to_datatype)
                .unwrap_or_else(utf8);
            DataType::Map {
                key_type: Box::new(utf8()),
                value_type: Box::new(value_type),
                value_contains_null: true,
            }
        }
        Value::Variant { .. } => DataType::Variant,
        Value::Null => utf8(),
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

    // Check for decimal.Decimal BEFORE int/float: Decimal defines __float__/__int__, so
    // extract::<f64>()/<i64>() would succeed and mis-tag a Decimal as Double/Long.
    if let Some(v) = decimal_to_value(obj)? {
        return Ok(v);
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
        let mut items = Vec::with_capacity(tuple.len());
        for item in tuple.iter() {
            items.push(py_to_value(&item)?);
        }
        return Ok(Value::List(items));
    }

    // pyspark Row -> struct: its named fields (values already core Values), so a nested
    // Row inside createDataFrame data (e.g. Row(s=Row(x=1))) converts recursively.
    if let Ok(pyrow) = obj.extract::<PyRef<crate::row::PyRow>>() {
        let fields = pyrow.row.fields();
        let values = pyrow.row.values();
        let mut out = Vec::with_capacity(values.len());
        for (i, v) in values.iter().enumerate() {
            let name = fields
                .get(i)
                .cloned()
                .unwrap_or_else(|| format!("_{}", i + 1));
            out.push((name, v.clone()));
        }
        return Ok(Value::Struct(out));
    }

    // dict -> map (keys coerced to their string form, values recursively converted)
    if let Ok(dict) = obj.cast::<pyo3::types::PyDict>() {
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

/// If `obj` is a `decimal.Decimal`, return it as a `Value::Decimal` (string value + scale
/// from `as_tuple()`); otherwise `None`. Checked before int/float in `py_to_value` because
/// Decimal defines `__float__`/`__int__`.
fn decimal_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Option<Value>> {
    let py = obj.py();
    let dec_mod = match py.import("decimal") {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };
    let decimal_cls = dec_mod.getattr("Decimal")?;
    if !obj.is_instance(&decimal_cls)? {
        return Ok(None);
    }
    let dec_str: String = obj.str()?.extract()?;
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
    Ok(Some(Value::Decimal {
        value: dec_str,
        precision: None,
        scale,
    }))
}
