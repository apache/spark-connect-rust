//! Error mapping from Rust SparkError to Python exceptions.

use pyo3::exceptions;
use pyo3::prelude::*;
use spark_connect_core::error::{SparkError, SparkErrorKind};

/// Maps a SparkErrorKind to the corresponding pyspark.errors exception class name.
/// Returns a string that can be used to retrieve the class from the pyspark.errors module.
fn kind_to_pyspark_class_name(kind: SparkErrorKind) -> &'static str {
    match kind {
        // PySpark builtin-like errors (these also subclass the Python builtins)
        SparkErrorKind::ValueError => "PySparkValueError",
        SparkErrorKind::TypeError => "PySparkTypeError",
        SparkErrorKind::IndexError => "PySparkIndexError",
        SparkErrorKind::AttributeError => "PySparkAttributeError",
        SparkErrorKind::KeyError => "PySparkKeyError",
        SparkErrorKind::RuntimeError => "PySparkRuntimeError",
        SparkErrorKind::NotImplementedError => "PySparkNotImplementedError",
        SparkErrorKind::AssertionError => "PySparkAssertionError",
        SparkErrorKind::PicklingError => "PySparkPicklingError",
        SparkErrorKind::ImportError => "PySparkImportError",

        // PySpark-specific exceptions
        SparkErrorKind::Analysis => "AnalysisException",
        SparkErrorKind::SessionNotSame => "SessionNotSameException",
        SparkErrorKind::TempTableAlreadyExists => "TempTableAlreadyExistsException",
        SparkErrorKind::Parse => "ParseException",
        SparkErrorKind::IllegalArgument => "IllegalArgumentException",
        SparkErrorKind::Arithmetic => "ArithmeticException",
        SparkErrorKind::UnsupportedOperation => "UnsupportedOperationException",
        SparkErrorKind::ArrayIndexOutOfBounds => "ArrayIndexOutOfBoundsException",
        SparkErrorKind::DateTime => "DateTimeException",
        SparkErrorKind::NumberFormat => "NumberFormatException",
        SparkErrorKind::StreamingQuery => "StreamingQueryException",
        SparkErrorKind::StreamingPythonRunnerInitialization => {
            "StreamingPythonRunnerInitializationException"
        }
        SparkErrorKind::QueryExecution => "QueryExecutionException",
        SparkErrorKind::Python => "PythonException",
        SparkErrorKind::SparkRuntime => "SparkRuntimeException",
        SparkErrorKind::SparkUpgrade => "SparkUpgradeException",
        SparkErrorKind::SparkNoSuchElement => "SparkNoSuchElementException",
        SparkErrorKind::Unknown => "UnknownException",
        SparkErrorKind::PickleException => "PickleException",

        // Connect-specific exceptions: these don't exist in the vendored pyspark.errors,
        // so we fall back to PySparkRuntimeError
        SparkErrorKind::Connect => "PySparkRuntimeError",
        SparkErrorKind::ConnectGrpc => "PySparkRuntimeError",
        SparkErrorKind::InvalidPlanInput => "PySparkRuntimeError",
    }
}

/// Maps a SparkError to a Python exception, attempting to raise the correct typed
/// exception based on the error kind. Falls back to PyRuntimeError if import fails.
pub fn spark_error_to_py_exception(err: SparkError) -> PyErr {
    let msg = err.message();
    let class_name = kind_to_pyspark_class_name(err.kind);

    // Try to acquire the GIL and import the exception class from pyspark.errors
    Python::attach(|py| {
        // First, verify and get the exception class from pyspark.errors
        let errors_module = match py.import("pyspark.errors") {
            Ok(m) => m,
            Err(_) => {
                // If import fails, return PyRuntimeError as fallback
                return exceptions::PyRuntimeError::new_err(msg);
            }
        };

        // Get the exception class from the module
        match errors_module.getattr(class_name) {
            Ok(exc_class) => {
                // Try to instantiate the exception class by calling it with the message
                // This creates a PyObject exception instance
                match exc_class.call1((msg.clone(),)) {
                    Ok(exc_instance) => {
                        // Convert the exception instance to a PyErr.
                        // We use unsafe code to construct the PyErr from the exception value.
                        unsafe { PyErr::from_value(exc_instance) }
                    }
                    Err(_) => {
                        // If instantiation fails, fall back to PyRuntimeError
                        exceptions::PyRuntimeError::new_err(msg)
                    }
                }
            }
            Err(_) => {
                // If class not found, fall back to PyRuntimeError
                exceptions::PyRuntimeError::new_err(msg)
            }
        }
    })
}

/// Helper to convert Result<T, SparkError> to PyResult<T>.
pub trait ResultExt<T> {
    fn to_pyerr(self) -> PyResult<T>;
}

impl<T> ResultExt<T> for Result<T, SparkError> {
    fn to_pyerr(self) -> PyResult<T> {
        self.map_err(spark_error_to_py_exception)
    }
}
