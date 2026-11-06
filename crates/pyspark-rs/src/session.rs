//! PyO3 wrapper for Spark Connect sessions.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use spark_connect::row::{Row, StructValue, Value};
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
    /// Mirrors `SparkSession.Builder.remote(location="sc://localhost")`.
    #[pyo3(signature = (location="sc://localhost"))]
    fn remote(&self, location: &str) -> PySparkSessionBuilder {
        let mut b = self.clone();
        b.remote_url = Some(location.to_string());
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
    fn app_name(&self, name: &str) -> PySparkSessionBuilder {
        let _ = name;
        self.clone()
    }

    /// `master` - not applicable to a remote Connect session (no local cluster to
    /// point at); accepted and ignored for API parity, like pyspark Connect.
    fn master(&self, master: &str) -> PySparkSessionBuilder {
        let _ = master;
        self.clone()
    }

    /// `_registerHook` - register an execution-hook factory (mirrors pyspark's Builder).
    /// Accepted for API parity; this native-transport client sends ExecutePlanRequests
    /// from Rust, so registered `Hook.on_execute_plan` callbacks are not invoked (see
    /// `PyHook`). Chainable, matching the builder pattern.
    #[pyo3(name = "_registerHook")]
    fn register_hook(&self, hook_factory: &Bound<'_, PyAny>) -> PySparkSessionBuilder {
        let _ = hook_factory;
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
        log_client_banner(py);
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

/// `SparkSession.Hook` - injects behavior around plan execution (mirrors pyspark's
/// `SparkSession.Hook`). In pyspark, `on_execute_plan` rewrites the Python
/// `ExecutePlanRequest` before the gRPC client sends it. This client uses a native Rust
/// transport that builds and sends the request in Rust (there is no Python request to
/// rewrite), so the hook is a passthrough - the same architectural reason
/// `SparkSession.client` is absent. Exposed for API parity and subclassing.
#[pyclass(name = "Hook", module = "pyspark.sql.connect.session", subclass)]
pub struct PyHook;

#[pymethods]
impl PyHook {
    #[new]
    fn new() -> Self {
        PyHook
    }

    /// Called before sending an ExecutePlanRequest; returns the (unchanged) request.
    fn on_execute_plan<'py>(&self, request: Bound<'py, PyAny>) -> Bound<'py, PyAny> {
        request
    }
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
            !(first.is_instance_of::<crate::row::PyRow>()
                || first.is_instance_of::<pyo3::types::PyTuple>()
                || first.is_instance_of::<PyList>()
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
        for (i, field) in fields.iter().enumerate() {
            for (row, out) in parts.iter().zip(values.iter_mut()) {
                out.push(infer::convert(&row[i], &field.data_type, field.nullable)?);
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

    /// A DataFrame of a pandas DataFrame: its rows as lists with NaN and NaT as None, typed
    /// by the first non-null value of each column, and its column labels, as strings, or the
    /// given names, applied with `toDF`.
    fn create_from_pandas(
        &self,
        py: Python<'_>,
        data: &Bound<'_, PyAny>,
        spec: Spec,
    ) -> PyResult<PyDataFrame> {
        let mut columns = Vec::new();
        for label in data.getattr("columns")?.try_iter()? {
            columns.push(label?.str()?.extract::<String>()?);
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
        let row_names: Vec<String> = match (&given, &names) {
            (Some(fields), _) => fields.iter().map(|f| f.name.clone()).collect(),
            (None, Some(names)) => names.clone(),
            (None, None) => Vec::new(),
        };
        let mut rows: Vec<Row> = Vec::with_capacity(data.len());
        for row in data.iter() {
            let values = row
                .try_iter()?
                .map(|v| py_to_value(&v?))
                .collect::<PyResult<Vec<_>>>()?;
            if values.len() != row_names.len() {
                // As pyspark, which checks every row against the width of the schema.
                return Err(crate::infer::axis_length_mismatch(
                    py,
                    row_names.len(),
                    values.len(),
                ));
            }
            rows.push(Row::new(row_names.clone(), values));
        }
        // A given struct wins; otherwise each field takes the type of its first non-null
        // value (empty input has a struct schema, checked above).
        let fields = match given {
            Some(fields) => fields,
            None => {
                let mut fields: Vec<StructField> = Vec::with_capacity(row_names.len());
                for (i, name) in row_names.iter().enumerate() {
                    let Some(value) = rows
                        .iter()
                        .filter_map(|row| row.get(i))
                        .find(|value| !matches!(value, Value::Null))
                    else {
                        return Err(crate::column::raise_pyspark(
                            py,
                            "PySparkValueError",
                            "CANNOT_DETERMINE_TYPE",
                            &[],
                        ));
                    };
                    fields.push(StructField {
                        name: name.clone(),
                        data_type: value_to_datatype(value),
                        nullable: true,
                        metadata: Default::default(),
                    });
                }
                fields
            }
        };
        let schema = DataType::Struct { fields };
        let df = py
            .detach(|| {
                self.session
                    .create_dataframe(rows, schema)
                    .map_err(Box::new)
            })
            .to_pyerr()?;
        Ok(PyDataFrame::new(match names {
            Some(names) if !names.is_empty() => {
                df.to_df(names.iter().map(String::as_str).collect())
            }
            _ => df,
        }))
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
fn schema_spec(py: Python<'_>, schema: Option<&Bound<'_, PyAny>>) -> PyResult<Spec> {
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
    Ok(match crate::types::py_to_data_type(schema)? {
        DataType::Struct { fields } => Spec::Struct(fields),
        atomic if crate::infer::is_atomic(&atomic) => Spec::Atomic(atomic),
        other => return Err(not_expected(&crate::infer::class_name(&other))),
    })
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
        let df = slf
            .borrow()
            .session
            .sql_with_references(&query, pos, named, references)
            .to_pyerr()?;
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
        let spec = schema_spec(py, schema)?;
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
            let ipc = py
                .import("pyspark.sql._pandas_input")?
                .getattr("prepare_string_input")?
                .call1((data, schema))?;
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
                let frame = self
                    .session
                    .create_dataframe_from_inline_arrow_ipc(ipc.extract()?, schema);
                // Match pandas row input: explicit nonempty field names travel
                // through toDF as well as the local relation's logical schema.
                return Ok(PyDataFrame::new(
                    frame.to_df(fields.iter().map(|field| field.name.as_str()).collect()),
                ));
            }
            return self.create_from_pandas(py, data, spec);
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

    /// Stop this Spark session. Best-effort and idempotent, mirroring pyspark: a failure
    /// to release the session on the server is logged, not raised. Also clears the
    /// active session and the cached `SQLContext` if they belong to this session.
    fn stop(&self, py: Python<'_>) -> PyResult<()> {
        if let Err(e) = py.detach(|| self.session.stop()) {
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
        let sid = self.session.session_id();
        {
            let mut slot = active_slot().lock().unwrap();
            if slot.as_ref().is_some_and(|s| s.session_id() == sid) {
                *slot = None;
            }
        }
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

    /// The execution hook base class, reached as `SparkSession.Hook` (mirrors pyspark).
    #[classattr]
    #[allow(non_snake_case)]
    fn Hook(py: Python<'_>) -> Py<pyo3::types::PyType> {
        py.get_type::<PyHook>().unbind()
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

    // Mirrors reference `SparkSession.addArtifacts(*path, pyfile=False, archive=False,
    // file=False)`. The pyfile/archive/file flags are accepted for signature parity;
    // artifact classification is handled by the Rust transport / server.
    #[pyo3(name = "addArtifacts", signature = (*path, pyfile=false, archive=false, file=false))]
    fn add_artifacts(
        &self,
        py: Python<'_>,
        path: Vec<String>,
        pyfile: bool,
        archive: bool,
        file: bool,
    ) -> PyResult<()> {
        let _ = (pyfile, archive, file);
        let refs: Vec<&str> = path.iter().map(|s| s.as_str()).collect();
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

    // Reference: cloneSession(new_session_id=None). The optional id is accepted for
    // signature parity; the Rust core assigns the cloned session's id itself and does not
    // support overriding it. See PARITY_GAP_ANALYSIS.md.
    #[pyo3(name = "cloneSession", signature = (new_session_id=None))]
    fn clone_session(&self, new_session_id: Option<String>) -> PySparkSession {
        let _ = new_session_id;
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
/// Nested collections are inferred recursively; nulls and empty collections use
/// String as a fallback. Interval input conversion is tracked in SPARK-60659.
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
        return Ok(Value::Struct(pyrow.to_row(obj.py())?.into()));
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
