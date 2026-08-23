//! PyO3 wrapper for Spark Connect sessions.

use pyo3::prelude::*;
use pyo3::types::PyList;
use spark_connect::row::{Row, Value};
use spark_connect::session::SparkSession;
use spark_connect::types::{DataType, StructField};

use crate::catalog::PyCatalog;
use crate::dataframe::PyDataFrame;
use crate::errors::ResultExt;

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
        Ok(PySparkSession::new(session))
    }
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

    // For now, treat everything else as a string
    Ok(Value::String(obj.to_string()))
}
