//! PyO3 wrapper for Spark Connect sessions.

use pyo3::prelude::*;
use pyo3::types::PyList;
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
}

impl PySparkSessionBuilder {
    pub fn new() -> Self {
        PySparkSessionBuilder { remote_url: None }
    }
}

#[pymethods]
impl PySparkSessionBuilder {
    /// Set the remote Spark Connect server URL. Returns the builder (chainable).
    fn remote(&self, url: &str) -> PySparkSessionBuilder {
        let mut b = self.clone();
        b.remote_url = Some(url.to_string());
        b
    }

    /// `appName` - accepted for API parity; Connect derives the app name server-side.
    #[pyo3(name = "appName")]
    fn app_name(&self, _name: &str) -> PySparkSessionBuilder {
        self.clone()
    }

    /// Build and get or create the session (`getOrCreate`).
    #[pyo3(name = "getOrCreate")]
    fn get_or_create(&self) -> PyResult<PySparkSession> {
        let url = self.remote_url.clone().ok_or_else(|| {
            pyo3::exceptions::PyValueError::new_err("Must call .remote(url) before .getOrCreate()")
        })?;

        let builder = SparkSession::builder().remote(&url);
        let session = builder.get_or_create().to_pyerr()?;
        *active_slot().lock().unwrap() = Some(session.clone());
        Ok(PySparkSession::new(session))
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
        // Parse the data into rows
        let mut rows = vec![];
        let mut num_cols: Option<usize> = None;

        for item in data.iter() {
            let row_list = if let Ok(lst) = item.cast::<PyList>() {
                lst.iter().collect::<Vec<_>>()
            } else {
                // Try to extract as tuple
                if let Ok(tuple) = item.cast::<pyo3::types::PyTuple>() {
                    tuple.iter().collect::<Vec<_>>()
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "Each row must be a list or tuple",
                    ));
                }
            };

            if let Some(nc) = num_cols {
                if row_list.len() != nc {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "All rows must have the same number of columns",
                    ));
                }
            } else {
                num_cols = Some(row_list.len());
            }

            let mut values = vec![];
            for val_py in row_list {
                let val = py_to_value(&val_py)?;
                values.push(val);
            }

            let field_names = if let Some(schema_obj) = schema {
                // Parse schema to get field names
                if let Ok(names) = schema_obj.extract::<Vec<String>>() {
                    names
                } else {
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                        "Schema must be a list of strings",
                    ));
                }
            } else {
                // Generate default names
                (0..values.len()).map(|i| format!("col{}", i)).collect()
            };

            rows.push(Row::new(field_names, values));
        }

        if rows.is_empty() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "Data is empty",
            ));
        }

        // Build schema
        let schema_dtype = if let Some(schema_obj) = schema {
            if let Ok(names) = schema_obj.extract::<Vec<String>>() {
                // Build a struct with string fields
                let fields: Vec<StructField> = names
                    .into_iter()
                    .map(|name| StructField {
                        name,
                        data_type: DataType::String {
                            collation: "UTF8_BINARY".to_string(),
                        },
                        nullable: true,
                        metadata: Default::default(),
                    })
                    .collect();
                DataType::Struct { fields }
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                    "Schema must be a list of strings",
                ));
            }
        } else {
            // Generate default schema with string types
            let names = &rows[0].fields();
            let fields: Vec<StructField> = names
                .iter()
                .map(|name| StructField {
                    name: name.clone(),
                    data_type: DataType::String {
                        collation: "UTF8_BINARY".to_string(),
                    },
                    nullable: true,
                    metadata: Default::default(),
                })
                .collect();
            DataType::Struct { fields }
        };

        let df = self
            .session
            .create_dataframe(rows, schema_dtype)
            .to_pyerr()?;
        Ok(PyDataFrame::new(df))
    }

    /// Get the catalog API.
    #[pyo3(name = "catalog")]
    fn catalog(&self) -> PyCatalog {
        PyCatalog::new(self.session.catalog())
    }

    /// Get the DataStreamReader for reading streaming data.
    #[pyo3(name = "readStream")]
    fn read_stream(&self) -> PyDataStreamReader {
        let reader = self.session.read_stream();
        PyDataStreamReader::new(reader)
    }

    /// Get the StreamingQueryManager for managing active streaming queries.
    #[pyo3(name = "streams")]
    fn streams(&self) -> PyStreamingQueryManager {
        let manager = self.session.streams();
        PyStreamingQueryManager::new(manager)
    }

    /// Stop this Spark session.
    fn stop(&self) -> PyResult<()> {
        self.session.stop().to_pyerr()
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
    fn session_id(&self) -> String {
        self.session.session_id().to_string()
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
    fn build_resource_profile(&self, profile: &PyResourceProfile) -> PyResult<i32> {
        self.session
            .build_resource_profile(&profile.inner)
            .to_pyerr()
    }

    fn version(&self) -> PyResult<String> {
        self.session.version().to_pyerr()
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
    fn interrupt_all(&self) -> PyResult<Vec<String>> {
        self.session.interrupt_all().to_pyerr()
    }

    #[pyo3(name = "interruptTag")]
    fn interrupt_tag(&self, tag: &str) -> PyResult<Vec<String>> {
        self.session.interrupt_tag(tag).to_pyerr()
    }

    #[pyo3(name = "interruptOperation")]
    fn interrupt_operation(&self, operation_id: &str) -> PyResult<Vec<String>> {
        self.session.interrupt_operation(operation_id).to_pyerr()
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

    #[pyo3(name = "addArtifacts")]
    fn add_artifacts(&self, paths: Vec<String>) -> PyResult<()> {
        let refs: Vec<&str> = paths.iter().map(|s| s.as_str()).collect();
        self.session.add_artifacts(&refs).to_pyerr()
    }

    #[pyo3(name = "addArtifact")]
    fn add_artifact(&self, path: &str) -> PyResult<()> {
        self.session.add_artifact(path).to_pyerr()
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

/// Convert a Python value to a Rust Value.
fn py_to_value(obj: &Bound<'_, PyAny>) -> PyResult<Value> {
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
