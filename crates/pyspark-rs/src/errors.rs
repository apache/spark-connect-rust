//! Error mapping from Rust SparkError to Python exceptions.

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use spark_connect_core::error::{ServerErrorInfo, SparkError, SparkErrorKind};

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
        // so we fall back to PySparkRuntimeError. An error that the server raised is built by
        // `server_error_to_py` instead.
        SparkErrorKind::Connect => "PySparkRuntimeError",
        SparkErrorKind::ConnectGrpc => "PySparkRuntimeError",
        SparkErrorKind::InvalidPlanInput => "PySparkRuntimeError",
        SparkErrorKind::Spark => "PySparkRuntimeError",
    }
}

/// The exception for an error that the server raised, as the reference client raises it:
/// `pyspark.errors.exceptions.connect._convert_server_error` picks and builds it, as the
/// reference client's `_convert_exception` does, from what the transport read of the error's
/// `ErrorInfo` and `FetchErrorDetails`.
fn server_error_to_py<'py>(
    py: Python<'py>,
    err: &SparkError,
    server: &ServerErrorInfo,
) -> PyResult<Bound<'py, PyAny>> {
    let contexts = PyList::empty(py);
    for context in &err.contexts {
        let fields = PyDict::new(py);
        fields.set_item("context_type", context.context_type.to_string())?;
        fields.set_item("object_type", &context.object_type)?;
        fields.set_item("object_name", &context.object_name)?;
        fields.set_item("start_index", context.start_index)?;
        fields.set_item("stop_index", context.stop_index)?;
        fields.set_item("fragment", &context.fragment)?;
        fields.set_item("call_site", &context.call_site)?;
        fields.set_item("summary", &context.summary)?;
        contexts.append(fields)?;
    }
    let breaking_change = match &server.breaking_change {
        Some(change) => {
            let info = PyDict::new(py);
            info.set_item("migration_message", &change.migration_message)?;
            info.set_item("needs_audit", change.needs_audit)?;
            if let Some((key, value)) = &change.mitigation_config {
                let config = PyDict::new(py);
                config.set_item("key", key)?;
                config.set_item("value", value)?;
                info.set_item("mitigation_config", config)?;
            }
            Some(info)
        }
        None => None,
    };
    let kwargs = PyDict::new(py);
    kwargs.set_item("classes", &server.classes)?;
    kwargs.set_item("sql_state", err.sql_state.as_deref())?;
    kwargs.set_item(
        "error_class",
        (!err.error_class.is_empty()).then_some(err.error_class.as_str()),
    )?;
    kwargs.set_item("reason", &server.reason)?;
    kwargs.set_item("message", &err.message)?;
    // In the order that the server gave the parameters, as the reference client keeps it.
    let parameters = PyDict::new(py);
    for name in &server.parameter_order {
        if let Some(value) = err.params.get(name) {
            parameters.set_item(name, value)?;
        }
    }
    for (name, value) in &err.params {
        if !parameters.contains(name)? {
            parameters.set_item(name, value)?;
        }
    }
    kwargs.set_item("message_parameters", parameters)?;
    kwargs.set_item("stacktrace", err.server_stacktrace.as_deref())?;
    kwargs.set_item("display_server_stacktrace", server.display_stacktrace)?;
    kwargs.set_item("contexts", contexts)?;
    // A server error always comes from a gRPC status; UNKNOWN is the reference default.
    kwargs.set_item("grpc_status_code", err.grpc_code.unwrap_or(2))?;
    kwargs.set_item("breaking_change_info", breaking_change)?;
    py.import("pyspark.errors.exceptions.connect")?.call_method(
        "_convert_server_error",
        (),
        Some(&kwargs),
    )
}

/// Maps a SparkError to a Python exception, attempting to raise the correct typed
/// exception based on the error kind. Falls back to PyRuntimeError if import fails.
pub fn spark_error_to_py_exception(err: SparkError) -> PyErr {
    let msg = err.message();
    let operation_id = err.operation_id.clone();
    let class_name = kind_to_pyspark_class_name(err.kind);

    // Try to acquire the GIL and import the exception class from pyspark.errors
    Python::attach(|py| {
        if let Some(server) = &err.server {
            // Should the connect classes not import, fall back to the class by kind below.
            if let Ok(exc_instance) = server_error_to_py(py, &err, server) {
                if let Some(op) = &operation_id {
                    let _ = exc_instance.setattr("_operation_id", op);
                }
                return PyErr::from_value(exc_instance);
            }
        }
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
                        // Mirror the reference client, which stamps `_operation_id` on
                        // errors raised for an ExecutePlan (read via `operation_id`).
                        if let Some(op) = &operation_id {
                            let _ = exc_instance.setattr("_operation_id", op);
                        }
                        // Convert the exception instance to a PyErr.
                        // We use unsafe code to construct the PyErr from the exception value.
                        PyErr::from_value(exc_instance)
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

/// For a boxed error, as a closure that runs without the GIL can return one, which keeps
/// its result small.
impl<T> ResultExt<T> for Result<T, Box<SparkError>> {
    fn to_pyerr(self) -> PyResult<T> {
        self.map_err(|e| spark_error_to_py_exception(*e))
    }
}
