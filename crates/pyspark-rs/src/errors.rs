//! Error mapping from Rust SparkError to Python exceptions.

use pyo3::exceptions;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList, PyString};
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

/// Raise a `pyspark.errors` exception (`class_name`) of the `error_class` condition alone, as
/// PySpark raises some of them without `messageParameters`, such as `MALFORMED_VARIANT` of the
/// conversion of local rows: its `getMessageParameters()` is None, where that of
/// [`raise_pyspark`] of no parameters is `{}`.
pub(crate) fn raise_pyspark_condition(
    py: Python<'_>,
    class_name: &str,
    error_class: &str,
) -> PyErr {
    let build = || -> PyResult<PyErr> {
        let cls = py.import("pyspark.errors")?.getattr(class_name)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("errorClass", error_class)?;
        Ok(PyErr::from_value(cls.call((), Some(&kwargs))?))
    };
    build().unwrap_or_else(|e| e)
}

/// As [`raise_pyspark`], of parameters that are Python objects, as PySpark formats them into its
/// message: a str that UTF-8 cannot encode, such as one of a lone surrogate, shows as it is.
pub(crate) fn raise_pyspark_of(
    py: Python<'_>,
    class_name: &str,
    error_class: &str,
    params: &[(&str, &Bound<'_, PyAny>)],
) -> PyErr {
    let build = || -> PyResult<PyErr> {
        let cls = py.import("pyspark.errors")?.getattr(class_name)?;
        let kwargs = PyDict::new(py);
        kwargs.set_item("errorClass", error_class)?;
        let mp = PyDict::new(py);
        for (k, v) in params {
            mp.set_item(*k, v)?;
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

/// Warn `message` of the warning class `category` as PySpark's `warnings.warn` at the line `line`
/// of its module `module` warns it, of which the drop-in's module of that name stands for
/// PySpark's: of the file of the drop-in's module, of PySpark's line, and of the module and its
/// registry of the warnings that it showed, as `warnings.warn` gives them of PySpark's frame. So
/// the filters of a module (`filterwarnings("ignore", module="pyspark")`) and of a line, and the
/// actions of a location (`"default"`, `"module"`), take it as they take PySpark's, where the frame
/// of the caller is of the user's module. The drop-in's modules end before PySpark's lines, so
/// the warning shows no source line, where PySpark's shows the line of its call.
pub(crate) fn warn_as_pyspark(
    py: Python<'_>,
    module: &str,
    line: u32,
    message: &str,
    category: &Bound<'_, pyo3::types::PyType>,
) -> PyResult<()> {
    let shim = py.import(module)?;
    let registry = shim.dict().call_method1(
        pyo3::intern!(py, "setdefault"),
        ("__warningregistry__", PyDict::new(py)),
    )?;
    py.import("warnings")?.call_method1(
        "warn_explicit",
        (
            message,
            category,
            shim.getattr("__file__")?,
            line,
            module,
            registry,
        ),
    )?;
    Ok(())
}

/// Maps a SparkErrorKind to the corresponding pyspark.errors exception class name.
/// Returns a string that can be used to retrieve the class from the pyspark.errors module.
fn kind_to_pyspark_class_name(kind: SparkErrorKind) -> &'static str {
    match kind {
        // PySpark builtin-like errors (these also subclass the Python builtins)
        SparkErrorKind::BuiltinTypeError => "TypeError",
        SparkErrorKind::BuiltinOverflowError => "OverflowError",
        SparkErrorKind::BuiltinRuntimeError => "RuntimeError",
        SparkErrorKind::BuiltinAttributeError => "AttributeError",
        SparkErrorKind::BuiltinAssertionError => "AssertionError",
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
/// released: the exception is raised by the caller instead of `err`. Of a kept error
/// ([`kept_error`]), a new exception of it, as each build of PySpark's plan raises a new one.
pub(crate) fn callback_exception(err: &SparkError) -> Option<PyErr> {
    if let Some(exception) = kept_exception(err) {
        return Some(exception);
    }
    let id = callback_id(err)?;
    CALLBACK_EXCEPTIONS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .remove(&id)
}

/// The error class of the error of the core of an exception that a plan keeps, which each
/// request of the plan raises ([`kept_error`]).
const PYTHON_KEPT_ERROR: &str = "PYTHON_KEPT_ERROR";

/// The owners of the exceptions that plans keep, by the IDs of their errors: each is the
/// `Py<PyAny>` of a tuple of the plan of the Python client whose build raised and the exception,
/// which the plan of the core holds as the owner of its relation, so the entry goes with the plan.
static KEPT_EXCEPTIONS: std::sync::LazyLock<
    std::sync::Mutex<
        std::collections::HashMap<u64, std::sync::Weak<dyn std::any::Any + Send + Sync>>,
    >,
> = std::sync::LazyLock::new(Default::default);

/// The error of the core of `exception`, an `Exception` that `plan`, a plan of the Python client,
/// raised when it was built, which a plan of the core keeps, and each request of that plan raises
/// anew, as PySpark raises a new exception at each request that builds the plan
/// ([`fresh_exception`]): the owner that it gives, of the tuple of `plan` and the exception,
/// keeps both while the plan of the core holds the owner (as the owner of a relation, see
/// `DataFrame::from_unbuilt_relation`), which reports it to the garbage collector of Python. The
/// exception is kept as a copy of no traceback, of copies of its cause and its context of none
/// either, so that it keeps no frame of the build alive with the plan, and each exception of a
/// request is one of its own, which keeps the frames of that request alone. A pickle of an object
/// of the plan carries `plan`, which its unpickle builds again (see [`kept_plan`]).
pub(crate) fn kept_error(
    py: Python<'_>,
    plan: &Bound<'_, PyAny>,
    exception: PyErr,
) -> PyResult<(SparkError, std::sync::Arc<dyn std::any::Any + Send + Sync>)> {
    let value = exception.into_value(py).into_bound(py).into_any();
    let kept = fresh_exception(&value);
    let owner: std::sync::Arc<dyn std::any::Any + Send + Sync> = std::sync::Arc::new(
        pyo3::types::PyTuple::new(py, [plan.clone(), kept])?
            .into_any()
            .unbind(),
    );
    let id = NEXT_CALLBACK_EXCEPTION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut kept = KEPT_EXCEPTIONS.lock().unwrap_or_else(|e| e.into_inner());
    kept.retain(|_, owner| owner.strong_count() > 0);
    kept.insert(id, std::sync::Arc::downgrade(&owner));
    let error = SparkError::classed(
        SparkErrorKind::Python,
        PYTHON_KEPT_ERROR,
        &[("kept_id", &id.to_string())],
    );
    Ok((error, owner))
}

/// The error of the core of an unpickled Column of a subquery of `plan`, a plan of the Python
/// client whose build raised when the Column was pickled, which the pickle carried, and the owner
/// that keeps the plan while the Column holds it ([`kept_error`], of no exception): each request
/// of the Column builds the plan of the session of the request ([`of_request_session`]), as
/// PySpark's Column builds the plan objects of its subqueries at each request, of the client of
/// the request, and raises the error of that build, a new one ([`kept_exception`]); and a pickle
/// of the Column carries `plan` again ([`kept_plan`]).
pub(crate) fn unpickled_plan_error(
    py: Python<'_>,
    plan: &Bound<'_, PyAny>,
) -> PyResult<(SparkError, std::sync::Arc<dyn std::any::Any + Send + Sync>)> {
    let owner: std::sync::Arc<dyn std::any::Any + Send + Sync> = std::sync::Arc::new(
        pyo3::types::PyTuple::new(py, [plan.clone(), py.None().into_bound(py)])?
            .into_any()
            .unbind(),
    );
    let id = NEXT_CALLBACK_EXCEPTION.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let mut kept = KEPT_EXCEPTIONS.lock().unwrap_or_else(|e| e.into_inner());
    kept.retain(|_, owner| owner.strong_count() > 0);
    kept.insert(id, std::sync::Arc::downgrade(&owner));
    let error = SparkError::classed(
        SparkErrorKind::Python,
        PYTHON_KEPT_ERROR,
        &[("kept_id", &id.to_string())],
    );
    Ok((error, owner))
}

/// The item `index` of the owner of `err`, if it is a kept error ([`kept_error`]) of a plan that
/// is alive: the plan of the Python client (0), or the exception that it raised (1), None of a
/// plan that a request builds again ([`unpickled_plan_error`]).
fn kept_item<'py>(py: Python<'py>, err: &SparkError, index: usize) -> Option<Bound<'py, PyAny>> {
    if err.error_class != PYTHON_KEPT_ERROR {
        return None;
    }
    let id = err.params.get("kept_id")?.parse::<u64>().ok()?;
    let owner = KEPT_EXCEPTIONS
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .get(&id)?
        .upgrade()?;
    let held = owner.downcast_ref::<Py<PyAny>>()?.bind(py);
    held.get_item(index).ok()
}

/// The plan of the Python client whose build raised the error that a plan keeps of `err`, if it is
/// a kept error of a plan that is alive: what a pickle of an object of the plan carries, as
/// PySpark's pickle carries its plan objects, which its unpickle builds again, of a new error of
/// its own.
pub(crate) fn kept_plan<'py>(py: Python<'py>, err: &SparkError) -> Option<Bound<'py, PyAny>> {
    kept_item(py, err, 0)
}

thread_local! {
    /// The sessions of the requests whose errors this thread converts now, and of the builds of
    /// plans of the Python client that it runs, innermost last (see [`of_request_session`]).
    static REQUEST_SESSIONS: std::cell::RefCell<Vec<spark_connect::session::SparkSession>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Run `convert`, which converts the errors of a request of `session`, or builds a plan of the
/// Python client of it, of `session` as the session of the request: so the plan that a Column
/// of its subquery builds again at a request ([`unpickled_plan_error`]) is built of the session
/// of the request, as PySpark's Column builds its plan object of the client of the request.
pub(crate) fn of_request_session<T>(
    session: &spark_connect::session::SparkSession,
    convert: impl FnOnce() -> T,
) -> T {
    struct Left;
    impl Drop for Left {
        fn drop(&mut self) {
            REQUEST_SESSIONS.with(|sessions| {
                sessions.borrow_mut().pop();
            });
        }
    }
    REQUEST_SESSIONS.with(|sessions| sessions.borrow_mut().push(session.clone()));
    let _left = Left;
    convert()
}

/// As [`of_request_session`], of the session of `client`, the client that a plan or an expression
/// is built of (see `session::session_of_client`), or of the session of the build that runs now
/// when `client` has none.
pub(crate) fn of_client_session<T>(client: &Bound<'_, PyAny>, convert: impl FnOnce() -> T) -> T {
    match crate::session::session_of_client(client) {
        Some(session) => of_request_session(&session, convert),
        None => convert(),
    }
}

/// A new exception of the one that a plan keeps of `err`, if it is a kept error of a plan that is
/// alive ([`fresh_exception`]); of a plan that a request builds again ([`unpickled_plan_error`]),
/// the error of its build now, of the session of the request, as PySpark's request builds the
/// plan of the client of the request, or of the active session outside a request; of a plan
/// that builds, the error of the object over it whose plan the pickle has not (see
/// `dataframe::derived_plan_error`).
fn kept_exception(err: &SparkError) -> Option<PyErr> {
    if err.error_class != PYTHON_KEPT_ERROR {
        return None;
    }
    Python::attach(|py| {
        let kept = kept_item(py, err, 1)?;
        if !kept.is_none() {
            return Some(PyErr::from_value(fresh_exception(&kept)));
        }
        let plan = kept_item(py, err, 0)?;
        Some(built_plan_exception(&plan).unwrap_or_else(|error| error))
    })
}

/// The exception of the error of the build of `plan`, a plan of the Python client whose build
/// raised when an object over it was pickled, of the session of the request now, or of the
/// active session outside a request (see [`kept_exception`]).
fn built_plan_exception(plan: &Bound<'_, PyAny>) -> PyResult<PyErr> {
    let py = plan.py();
    let request = REQUEST_SESSIONS.with(|sessions| sessions.borrow().last().cloned());
    let session = match request {
        Some(session) => crate::session::session_object(py, &session)?.into_bound(py),
        None => {
            let active = py
                .import("pyspark.sql.connect.session")?
                .getattr("SparkSession")?
                .call_method0("getActiveSession")?;
            match active.cast_into::<crate::session::PySparkSession>() {
                Ok(session) => session,
                Err(_) => {
                    return Ok(spark_error_to_py_exception(SparkError::classed(
                        SparkErrorKind::RuntimeError,
                        "NO_ACTIVE_SESSION",
                        &[],
                    )))
                }
            }
        }
    };
    // The exception is made while the DataFrame of the plan, which keeps the error of its build,
    // is alive.
    let (dataframe, error) = crate::dataframe::rebuilt_python_plan(plan, &session, true)?;
    let exception =
        spark_error_to_py_exception(error.unwrap_or_else(crate::dataframe::derived_plan_error));
    drop(dataframe);
    Ok(exception)
}

/// The levels of the chain of causes and contexts of an exception that [`fresh_exception`]
/// copies; those below are not kept.
const FRESH_CHAIN_LEVELS: usize = 64;

/// A new exception of `kept`, as PySpark's plan raises a new one each time that it is built: its
/// copy, as `copy.copy` makes it, of its arguments and its state; or, of a class that does not
/// take its arguments back, as PySpark's error classes do not take their message, the exception
/// of its class made of its arguments, without its `__init__`, and of its state. Of copies of the
/// cause and the context of `kept`, made so in turn, and of no traceback, neither the copy nor
/// those of its chain, which the raise gives the copy. `kept` itself when neither makes one.
fn fresh_exception<'py>(kept: &Bound<'py, PyAny>) -> Bound<'py, PyAny> {
    fresh_exception_of_chain(kept, 0)
}

fn fresh_exception_of_chain<'py>(kept: &Bound<'py, PyAny>, level: usize) -> Bound<'py, PyAny> {
    let py = kept.py();
    let class = kept.get_type();
    let copied = py
        .import("copy")
        .and_then(|copy| copy.call_method1("copy", (kept,)))
        .ok()
        .filter(|copy| copy.get_type().is(&class) && !copy.is(kept));
    let fresh = copied.or_else(|| {
        let args = kept.getattr("args").ok()?;
        let args = args.cast::<pyo3::types::PyTuple>().ok()?;
        let made = class
            .getattr("__new__")
            .and_then(|new| {
                new.call1(pyo3::types::PyTuple::new(
                    py,
                    std::iter::once(class.clone().into_any())
                        .chain(args.iter())
                        .collect::<Vec<_>>(),
                )?)
            })
            .ok()
            .filter(|made| made.get_type().is(&class))?;
        if let (Ok(state), Ok(dict)) = (kept.getattr("__dict__"), made.getattr("__dict__")) {
            dict.call_method1("update", (state,)).ok()?;
        }
        Some(made)
    });
    let Some(fresh) = fresh else {
        return kept.clone();
    };
    for name in ["__cause__", "__context__"] {
        let chained = kept
            .getattr(name)
            .ok()
            .filter(|chained| !chained.is_none() && level < FRESH_CHAIN_LEVELS)
            .map(|chained| fresh_exception_of_chain(&chained, level + 1));
        let _ = fresh.setattr(name, chained);
    }
    if let Ok(value) = kept.getattr("__suppress_context__") {
        let _ = fresh.setattr("__suppress_context__", value);
    }
    let _ = fresh.call_method1("with_traceback", (py.None(),));
    fresh
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
        // As PySpark's `_handle_error`, which stamps the operation ID of an execution on any
        // exception of it, such as that of the warning of its spent retries of a filter of
        // "error".
        Python::attach(|py| {
            stamp_operation_id(exception.value(py).as_any(), err.operation_id.as_deref())
        });
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
    if err.kind == SparkErrorKind::BuiltinRuntimeError {
        // A builtin, as the thread pool of PySpark's releases raises it at interpreter shutdown.
        return exceptions::PyRuntimeError::new_err(err.message());
    }
    if err.kind == SparkErrorKind::BuiltinAttributeError {
        // A builtin, as PySpark's plan raises it for an argument that lacks a method it calls.
        return exceptions::PyAttributeError::new_err(err.message());
    }
    if err.kind == SparkErrorKind::BuiltinAssertionError {
        // A builtin, as a bare `assert` of PySpark's client raises it, of no arguments, and of
        // the operation ID of the execution, which its `_handle_error` sets.
        let message = err.message();
        let error = if message.is_empty() {
            exceptions::PyAssertionError::new_err(())
        } else {
            exceptions::PyAssertionError::new_err(message)
        };
        Python::attach(|py| {
            stamp_operation_id(error.value(py).as_any(), err.operation_id.as_deref())
        });
        return error;
    }
    if let Some(value) = err.invalid_utf8_value() {
        return unicode_decode_error(&value, &err);
    }
    if let Some(value) = err.unencodable_value() {
        return unicode_encode_error(&value, &err);
    }
    // PySpark's message, where the core's differs: grpcio's text of a gRPC failure without an
    // error of the server.
    let msg = err.pyspark_message.clone().unwrap_or_else(|| err.message());
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
        parameters.set_item(key, param_value(py, &value)).ok()?;
    }
    let kwargs = PyDict::new(py);
    kwargs.set_item("errorClass", error_class).ok()?;
    kwargs.set_item("messageParameters", parameters).ok()?;
    exc_class.call((), Some(&kwargs)).ok()
}

/// The text of a message parameter of an error of the core that the binding makes of `value`, a
/// Python str, which the exception of the error has as it is ([`param_value`]): its text, or, for
/// a str that UTF-8 cannot encode, such as one of a lone surrogate, which a Rust string cannot
/// hold, a NUL and the hex of its UTF-32 little-endian bytes. A str that starts with a NUL is held
/// so too, so that its text is not taken for such bytes.
pub(crate) fn str_param(value: &Bound<'_, PyString>) -> PyResult<String> {
    if let Ok(text) = value.to_cow() {
        if !text.starts_with('\0') {
            return Ok(text.into_owned());
        }
    }
    let utf32le = value.call_method1("encode", ("utf-32-le", "surrogatepass"))?;
    let mut text = String::from('\0');
    for byte in utf32le.cast::<PyBytes>()?.as_bytes() {
        text.push_str(&format!("{byte:02x}"));
    }
    Ok(text)
}

/// The Python str of `text`, the text of a message parameter of an error of the core: the str of
/// the bytes that [`str_param`] holds, or else the text as it is.
fn param_value<'py>(py: Python<'py>, text: &str) -> Bound<'py, PyAny> {
    let held = text.strip_prefix('\0').and_then(|hex| {
        let bytes = (0..hex.len())
            .step_by(2)
            .map(|at| u8::from_str_radix(hex.get(at..at + 2)?, 16).ok())
            .collect::<Option<Vec<u8>>>()?;
        PyBytes::new(py, &bytes)
            .call_method1("decode", ("utf-32-le", "surrogatepass"))
            .ok()
    });
    held.unwrap_or_else(|| PyString::new(py, text).into_any())
}

/// The error that raises `err` again, an exception of one of PySpark's error classes, such as
/// the `PySparkAttributeError` that a Row raises of an attribute that it does not have: of its
/// class, error class and message parameters, of which [`classed_exception`] builds the exception
/// as PySpark built it. None for another exception, such as one of a subclass, or one without an
/// error class or of message parameters that are not all str.
pub(crate) fn pyspark_class_error(py: Python<'_>, err: &PyErr) -> PyResult<Option<SparkError>> {
    const KINDS: [SparkErrorKind; 10] = [
        SparkErrorKind::ValueError,
        SparkErrorKind::TypeError,
        SparkErrorKind::IndexError,
        SparkErrorKind::AttributeError,
        SparkErrorKind::KeyError,
        SparkErrorKind::RuntimeError,
        SparkErrorKind::NotImplementedError,
        SparkErrorKind::AssertionError,
        SparkErrorKind::PicklingError,
        SparkErrorKind::ImportError,
    ];
    let errors_module = py.import("pyspark.errors")?;
    let value = err.value(py);
    let mut kind = None;
    for candidate in KINDS {
        let class = errors_module.getattr(kind_to_pyspark_class_name(candidate))?;
        if value.get_type().is(&class) {
            kind = Some(candidate);
            break;
        }
    }
    let Some(kind) = kind else {
        return Ok(None);
    };
    let Ok(error_class) = value.call_method0("getCondition")?.extract::<String>() else {
        return Ok(None);
    };
    let mut error = SparkError::classed(kind, &error_class, &[]);
    let parameters = value.call_method0("getMessageParameters")?;
    if !parameters.is_none() {
        let Ok(parameters) = parameters.cast::<PyDict>() else {
            return Ok(None);
        };
        for (key, parameter) in parameters.iter() {
            let (Ok(key), Ok(parameter)) = (key.extract::<String>(), parameter.extract::<String>())
            else {
                return Ok(None);
            };
            error.params.insert(key, parameter);
        }
    }
    Ok(Some(error))
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
