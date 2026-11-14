//! Error mapping from Rust SparkError to Python exceptions.

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use spark_connect_core::error::{ServerErrorInfo, SparkError, SparkErrorKind};

/// Raise a `pyspark.errors` exception (`class_name`) carrying the given `error_class`
/// condition and message parameters, mirroring the reference Column guard-rails.
pub(crate) fn raise_pyspark(
    py: Python<'_>,
    class_name: &str,
    error_class: &str,
    params: &[(&str, &str)],
) -> PyErr {
    let build = || -> PyResult<PyErr> {
        let cls = py.import("pyspark.errors")?.getattr(class_name)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("errorClass", error_class)?;
        let mp = PyDict::new(py);
        for (k, v) in params {
            mp.set_item(*k, *v)?;
        }
        kwargs.set_item("messageParameters", mp)?;
        let exc = cls.call((), Some(&kwargs))?;
        Ok(PyErr::from_value(exc))
    };
    build().unwrap_or_else(|e| e)
}

/// PySpark's `NOT_EXPECTED_TYPE` error of the argument `name`, `value`, which should be of
/// the type `expected`.
pub(crate) fn not_expected_type(value: &Bound<'_, PyAny>, name: &str, expected: &str) -> PyErr {
    let type_name = value
        .get_type()
        .name()
        .map(|name| name.to_string())
        .unwrap_or_default();
    raise_pyspark(
        value.py(),
        "PySparkTypeError",
        "NOT_EXPECTED_TYPE",
        &[
            ("arg_name", name),
            ("expected_type", expected),
            ("arg_type", &type_name),
        ],
    )
}

/// Maps a SparkErrorKind to the corresponding pyspark.errors exception class name.
/// Returns a string that can be used to retrieve the class from the pyspark.errors module.
fn kind_to_pyspark_class_name(kind: SparkErrorKind) -> &'static str {
    match kind {
        // PySpark builtin-like errors (these also subclass the Python builtins)
        SparkErrorKind::BuiltinTypeError => "TypeError",
        SparkErrorKind::BuiltinOverflowError => "OverflowError",
        SparkErrorKind::ValueError => "PySparkValueError",
        SparkErrorKind::BuiltinValueError => "ValueError",
        SparkErrorKind::TypeError => "PySparkTypeError",
        SparkErrorKind::IndexError => "PySparkIndexError",
        SparkErrorKind::AttributeError => "PySparkAttributeError",
        SparkErrorKind::KeyError => "PySparkKeyError",
        SparkErrorKind::RuntimeError => "PySparkRuntimeError",
        SparkErrorKind::RecursionError => "RecursionError",
        SparkErrorKind::UnicodeDecode => "UnicodeDecodeError",
        SparkErrorKind::UnicodeEncode => "UnicodeEncodeError",
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
        // A `SparkConnectException` of a plain message: `spark_error_to_py_exception` raises it.
        SparkErrorKind::ConnectMessage => "PySparkRuntimeError",
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

/// The exception for a gRPC error that carries no error of the server, such as an
/// `UNAVAILABLE` of an unreachable server or a `PERMISSION_DENIED` of a proxy, as the
/// reference client's `_handle_rpc_error` raises it: a `SparkConnectGrpcException` of the
/// status code.
fn grpc_error_to_py<'py>(py: Python<'py>, message: &str, code: i32) -> PyResult<Bound<'py, PyAny>> {
    let connect = py.import("pyspark.errors.exceptions.connect")?;
    let kwargs = PyDict::new(py);
    kwargs.set_item("message", message)?;
    kwargs.set_item(
        "grpc_status_code",
        connect.call_method1("_status_code", (code,))?,
    )?;
    connect
        .getattr("SparkConnectGrpcException")?
        .call((), Some(&kwargs))
}

/// The error class of the error of the core of an exception of a Python callback that the core
/// calls, such as a hook of `SparkSession.Hook`: its `callback_id` is that of the exception, which
/// the error raises as it is.
const PYTHON_CALLBACK_ERROR: &str = "PYTHON_CALLBACK_ERROR";

/// The exceptions of the Python callbacks that the core calls, by their IDs, until the errors of
/// the core that carry them are raised.
static CALLBACK_EXCEPTIONS: std::sync::LazyLock<
    std::sync::Mutex<std::collections::HashMap<u64, PyErr>>,
> = std::sync::LazyLock::new(Default::default);

static NEXT_CALLBACK_EXCEPTION: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// The error of the core of `exception`, the exception of a Python callback that the core calls,
/// which [`spark_error_to_py_exception`] raises as it is: as PySpark raises the exception of a
/// hook from the execution that calls it.
pub(crate) fn callback_error(exception: PyErr) -> SparkError {
    let id = NEXT_CALLBACK_EXCEPTION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    CALLBACK_EXCEPTIONS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .insert(id, exception);
    SparkError::classed(
        SparkErrorKind::Python,
        PYTHON_CALLBACK_ERROR,
        &[("callback_id", &id.to_string())],
    )
}

/// The ID of the exception of the Python callback that `err` is the error of, if it is one.
fn callback_id(err: &SparkError) -> Option<u64> {
    if err.error_class != PYTHON_CALLBACK_ERROR {
        return None;
    }
    err.params.get("callback_id")?.parse::<u64>().ok()
}

/// The exception of the Python callback of `err`, if it is the error of one, which is then
/// released: the exception is raised by the caller instead of `err`.
pub(crate) fn callback_exception(err: &SparkError) -> Option<PyErr> {
    let id = callback_id(err)?;
    CALLBACK_EXCEPTIONS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&id)
}

/// Make `context` the `__context__` of the exception of the Python callback that `err` is the
/// error of, as PySpark's `Progress.__exit__` makes the exception of its body the context of the
/// one that it raises. An error that is not the error of a callback has no exception to change.
pub(crate) fn set_callback_context(py: Python<'_>, err: &SparkError, context: PyErr) {
    let Some(id) = callback_id(err) else {
        return;
    };
    // Taken out of the lock: dropping the context that the exception had can run Python code.
    let exception = CALLBACK_EXCEPTIONS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&id)
        .map(|exception| exception.clone_ref(py));
    if let Some(exception) = exception {
        set_context(py, exception.value(py), context);
    }
}

/// Make `context` the `__context__` of `exception`, as Python makes the exception that is being
/// handled the context of one that is raised: through the C API, which cannot fail, rather than
/// by setting the attribute, which the `__setattr__` of the class of the exception could refuse.
/// An exception is never made its own context.
fn set_context(
    py: Python<'_>,
    exception: &Bound<'_, pyo3::exceptions::PyBaseException>,
    context: PyErr,
) {
    let context = context.into_value(py);
    if context.as_ptr() == exception.as_ptr() {
        return;
    }
    // SAFETY: both are exception instances, held with the interpreter attached, and
    // `PyException_SetContext` steals the reference to the context that `into_ptr` gives it.
    unsafe { pyo3::ffi::PyException_SetContext(exception.as_ptr(), context.into_ptr()) }
}

/// `raised`, with `secondary` as its `__context__`: an exception that kept the client from
/// raising a more specific exception than `raised` is not dropped, but chained to the one that
/// is raised in its place, as Python chains an exception raised while another is handled.
pub(crate) fn with_context(py: Python<'_>, raised: PyErr, secondary: PyErr) -> PyErr {
    set_context(py, raised.value(py), secondary);
    raised
}

/// Stamp `_operation_id` on an exception of an ExecutePlan, as the reference client does (read
/// through `operation_id`). The id is auxiliary: an exception that refuses the attribute is
/// still the one to raise, so the refusal does not replace it, and goes to
/// `sys.unraisablehook` instead of being dropped.
fn stamp_operation_id(exception: &Bound<'_, PyAny>, operation_id: Option<&str>) {
    if let Some(operation_id) = operation_id {
        if let Err(error) = exception.setattr("_operation_id", operation_id) {
            error.write_unraisable(exception.py(), Some(exception));
        }
    }
}

/// The UnicodeDecodeError of a string of the result that is not valid UTF-8, as PySpark raises
/// it: the error of Python's own decoder of its bytes, so that the message is the same.
fn unicode_decode_error(value: &[u8], err: &SparkError) -> PyErr {
    Python::attach(
        |py| match PyBytes::new(py, value).call_method1("decode", ("utf-8",)) {
            Err(exception) => exception,
            Ok(_) => exceptions::PyRuntimeError::new_err(err.message()),
        },
    )
}

/// The UnicodeEncodeError of a str that UTF-8 cannot encode, as PySpark raises it when it builds
/// the plan: the error of Python's own encoder of the str, which the UTF-32 bytes of `err` give
/// back.
fn unicode_encode_error(utf32le: &[u8], err: &SparkError) -> PyErr {
    Python::attach(|py| {
        let text = PyBytes::new(py, utf32le).call_method1("decode", ("utf-32-le", "surrogatepass"));
        match text.and_then(|text| text.call_method1("encode", ("utf-8",))) {
            Err(exception) => exception,
            Ok(_) => exceptions::PyRuntimeError::new_err(err.message()),
        }
    })
}

/// Maps a SparkError to a Python exception, attempting to raise the correct typed
/// exception based on the error kind. Falls back to PyRuntimeError if import fails.
pub fn spark_error_to_py_exception(err: SparkError) -> PyErr {
    if let Some(exception) = callback_exception(&err) {
        return exception;
    }
    if err.kind == SparkErrorKind::RecursionError {
        // A builtin, as PySpark's recursion limit raises it.
        return pyo3::exceptions::PyRecursionError::new_err(err.message());
    }
    if err.kind == SparkErrorKind::BuiltinValueError {
        // A builtin, as the `int()` or `urllib` of the connection string raises it.
        return exceptions::PyValueError::new_err(err.message());
    }
    if err.kind == SparkErrorKind::BuiltinTypeError {
        // A builtin, as the protobuf of a plan's argument raises it.
        return exceptions::PyTypeError::new_err(err.message());
    }
    if err.kind == SparkErrorKind::BuiltinOverflowError {
        // A builtin, as the protobuf of a plan's int argument raises it.
        return exceptions::PyOverflowError::new_err(err.message());
    }
    if let Some(value) = err.invalid_utf8_value() {
        return unicode_decode_error(&value, &err);
    }
    if let Some(value) = err.unencodable_value() {
        return unicode_encode_error(&value, &err);
    }
    let msg = err.message();
    let operation_id = err.operation_id.clone();
    let class_name = kind_to_pyspark_class_name(err.kind);

    Python::attach(|py| {
        // An error of the server, or of gRPC, is the exception of the connect classes. Should
        // making it fail, the error is raised as the class of its kind below, with the
        // exception of the failure as its `__context__`.
        let connect_error = match (&err.server, err.grpc_code) {
            (Some(server), _) => Some(server_error_to_py(py, &err, server)),
            (None, Some(code)) => Some(grpc_error_to_py(py, &msg, code)),
            (None, None) => None,
        };
        let connect_failure = match connect_error {
            Some(Ok(exc_instance)) => {
                stamp_operation_id(&exc_instance, operation_id.as_deref());
                return PyErr::from_value(exc_instance);
            }
            Some(Err(failure)) => Some(failure),
            None => None,
        };
        match kind_exception(py, &err, class_name, &msg) {
            Ok(exc_instance) => {
                stamp_operation_id(&exc_instance, operation_id.as_deref());
                let raised = PyErr::from_value(exc_instance);
                match connect_failure {
                    Some(failure) => with_context(py, raised, failure),
                    None => raised,
                }
            }
            // The class of the kind does not import, or does not take the message: a
            // RuntimeError of the message, with the exception of the failure as its context,
            // which the failure of the connect classes, if any, came before.
            Err(failure) => {
                let failure = match connect_failure {
                    Some(earlier) => with_context(py, failure, earlier),
                    None => failure,
                };
                with_context(py, exceptions::PyRuntimeError::new_err(msg), failure)
            }
        }
    })
}

/// The exception of `err` of the class of its kind, `class_name` of `pyspark.errors`.
fn kind_exception<'py>(
    py: Python<'py>,
    err: &SparkError,
    class_name: &str,
    msg: &str,
) -> PyResult<Bound<'py, PyAny>> {
    let errors_module = py.import("pyspark.errors")?;
    // A classed Connect error, such as `NO_ACTIVE_SESSION`, and a Connect error of a plain
    // message are `SparkConnectException`s, as the reference client raises them.
    let connect_class = err.kind == SparkErrorKind::ConnectMessage
        || (err.kind == SparkErrorKind::Connect && !err.error_class.is_empty());
    let exc_class = if connect_class {
        py.import("pyspark.errors.exceptions.connect")?
            .getattr("SparkConnectException")?
    } else {
        errors_module.getattr(class_name)?
    };
    // Of the error class and message parameters of an error of a PySpark error class,
    // otherwise of the message.
    match classed_exception(&exc_class, err) {
        Some(exc_instance) => Ok(exc_instance),
        None => exc_class.call1((msg,)),
    }
}

/// The exception of `exc_class` for an error of a PySpark error class that the client raised,
/// as PySpark raises it: of the error class and the message parameters, from which the
/// exception renders its message, so that `getCondition` and `getMessageParameters` return
/// them. None for an error without an error class, or of one that PySpark does not know.
fn classed_exception<'py>(
    exc_class: &Bound<'py, PyAny>,
    err: &SparkError,
) -> Option<Bound<'py, PyAny>> {
    let error_class = err.get_condition()?;
    let py = exc_class.py();
    let parameters = PyDict::new(py);
    for (key, value) in err.get_message_parameters().unwrap_or_default() {
        parameters.set_item(key, value).ok()?;
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("errorClass", error_class).ok()?;
    kwargs.set_item("messageParameters", parameters).ok()?;
    exc_class.call((), Some(&kwargs)).ok()
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
