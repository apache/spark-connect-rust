//! Error types mirroring PySpark's error-class framework.
//!
//! PySpark raises typed exceptions carrying an ``errorClass`` and
//! ``messageParameters`` (see ``pyspark/errors/exceptions``). We preserve the
//! error class so the PyO3 layer can re-raise the exact Python exception type
//! with the same class/parameters, giving byte-compatible error messages.
//!
//! The error-conditions registry is embedded at build time from
//! `error-conditions.json`, allowing us to render error messages with parameter
//! substitution matching Python's `PySparkException.getMessage()` exactly.

use prost::Message;
use std::collections::BTreeMap;
use std::collections::HashMap;
use std::fmt;
use tonic::Status;

// Embed error-conditions.json at compile time for offline message rendering.
const ERROR_CONDITIONS_JSON: &str = include_str!("../error-conditions.json");

// ============================================================================
// Query context types mirroring PySpark's QueryContext
// ============================================================================

/// The type of query context (SQL or DataFrame).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryContextType {
    SQL,
    DataFrame,
}

impl fmt::Display for QueryContextType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            QueryContextType::SQL => write!(f, "SQL"),
            QueryContextType::DataFrame => write!(f, "DataFrame"),
        }
    }
}

/// Query context describing where an error occurred.
#[derive(Debug, Clone)]
pub struct QueryContext {
    pub context_type: QueryContextType,
    pub object_type: String,
    pub object_name: String,
    pub start_index: i32,
    pub stop_index: i32,
    pub fragment: String,
    pub call_site: String,
    pub summary: String,
}

impl QueryContext {
    /// Create a new QueryContext.
    pub fn new(
        context_type: QueryContextType,
        object_type: String,
        object_name: String,
        start_index: i32,
        stop_index: i32,
        fragment: String,
        call_site: String,
        summary: String,
    ) -> Self {
        Self {
            context_type,
            object_type,
            object_name,
            start_index,
            stop_index,
            fragment,
            call_site,
            summary,
        }
    }

    /// Get the context type.
    pub fn context_type(&self) -> QueryContextType {
        self.context_type
    }

    /// Get the object type (e.g., "VIEW", or empty string for main query).
    pub fn object_type(&self) -> &str {
        &self.object_type
    }

    /// Get the object name (e.g., view name, or empty string for main query).
    pub fn object_name(&self) -> &str {
        &self.object_name
    }

    /// Get the starting index in the query text (0-based).
    pub fn start_index(&self) -> i32 {
        self.start_index
    }

    /// Get the stopping index in the query text (0-based).
    pub fn stop_index(&self) -> i32 {
        self.stop_index
    }

    /// Get the corresponding fragment of the query.
    pub fn fragment(&self) -> &str {
        &self.fragment
    }

    /// Get the user code (call site of the API) that caused the exception.
    pub fn call_site(&self) -> &str {
        &self.call_site
    }

    /// Get a summary of the exception cause.
    pub fn summary(&self) -> &str {
        &self.summary
    }
}

// ============================================================================
// Minimal prost message definitions for google.rpc types (needed for error parsing)
// ============================================================================

/// Minimal google.rpc.Status for decoding error details.
#[derive(Clone, PartialEq, Message)]
struct RpcStatus {
    #[prost(int32, tag = "1")]
    code: i32,
    #[prost(string, tag = "2")]
    message: String,
    #[prost(message, repeated, tag = "3")]
    details: Vec<RpcAny>,
}

/// Minimal google.rpc.Any for decoding packed types.
#[derive(Clone, PartialEq, Message)]
struct RpcAny {
    #[prost(string, tag = "1")]
    type_url: String,
    #[prost(bytes, tag = "2")]
    value: Vec<u8>,
}

/// Minimal google.rpc.ErrorInfo for structured error details.
#[derive(Clone, PartialEq, Message)]
struct RpcErrorInfo {
    #[prost(string, tag = "1")]
    reason: String,
    #[prost(string, tag = "2")]
    domain: String,
    #[prost(map = "string, string", tag = "3")]
    metadata: HashMap<String, String>,
}

/// What a server error carries besides its message, condition, parameters, SQL state, stack
/// trace and query contexts, as PySpark Connect reads it from the error's `ErrorInfo` and, when
/// the server keeps more, from `FetchErrorDetails`.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ServerErrorInfo {
    /// The JVM class of the exception and then its superclasses (`classes`).
    pub classes: Vec<String>,
    /// `ErrorInfo.reason`.
    pub reason: String,
    /// The id that the server keeps the error's details under (`errorId`), if it keeps them.
    pub error_id: Option<String>,
    /// The names of the message parameters in the order that the server gave them, which the
    /// reference client keeps.
    pub parameter_order: Vec<String>,
    /// Whether the message shows the server's stack trace, as
    /// `spark.sql.connect.serverStacktrace.enabled` or `spark.sql.pyspark.jvmStacktrace.enabled`
    /// decides.
    pub display_stacktrace: bool,
    /// The breaking change that the error comes from, if the server reports one.
    pub breaking_change: Option<BreakingChangeInfo>,
    /// The kind of the call that failed and the text of grpcio's error of it, `str()` of
    /// PySpark's `RpcError`, as PySpark Connect handles that error as it builds its exception,
    /// which the query context loggers of the Python binding log as the exception. None of an
    /// error that no call failed with. Not a stable API.
    #[doc(hidden)]
    pub rpc_error: Option<(GrpcCall, String)>,
}

/// A breaking change that an error comes from. Mirrors what
/// `SparkConnectGrpcException.getBreakingChangeInfo()` returns.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BreakingChangeInfo {
    /// How to migrate.
    pub migration_message: Vec<String>,
    /// The config, as a key and a value, that restores the previous behavior.
    pub mitigation_config: Option<(String, String)>,
    /// Whether the change needs an audit.
    pub needs_audit: bool,
}

/// A structured error carrying a PySpark error class and its message parameters.
#[derive(Debug, Clone)]
pub struct SparkError {
    /// Which Python exception type to raise at the PyO3 boundary.
    pub kind: SparkErrorKind,
    /// PySpark error class (e.g. "INVALID_CONNECT_URL"), or empty for plain messages.
    pub error_class: String,
    /// Message parameters keyed by name, as in ``messageParameters``.
    pub params: BTreeMap<String, String>,
    /// A pre-rendered message, used when no error class applies.
    pub message: String,
    /// SQL state code, if known (from the error-conditions registry).
    pub sql_state: Option<String>,
    /// Query contexts where the error occurred.
    pub contexts: Vec<QueryContext>,
    /// Server-side stack trace, if available.
    pub server_stacktrace: Option<String>,
    /// Raw gRPC status code, if this error originated from a gRPC Status.
    pub grpc_code: Option<i32>,
    /// Raw `grpc-status-details-bin` (a serialized `google.rpc.Status`), if present.
    /// Lets the PyO3 layer reconstruct the exact typed pyspark exception.
    pub grpc_details: Option<Vec<u8>>,
    /// The ExecutePlan operation id the error was raised for, if known. Backs
    /// `SparkConnectException.operation_id` (new in pyspark 4.3.0).
    pub operation_id: Option<String>,
    /// For an error that the server raised, what it sent about it besides the fields above.
    /// Boxed, so that it does not make every `Result` larger.
    pub server: Option<Box<ServerErrorInfo>>,
    /// The message of PySpark's exception of this error, where it is not
    /// [`SparkError::message`]: for a gRPC failure without an error of the server, the text of
    /// grpcio's error of the call or the message of its `google.rpc.Status`, as PySpark's
    /// `_handle_rpc_error` raises it (see [`SparkError::from_grpc_status_of`]). The Python
    /// binding raises it; the message of the Rust API is `[code] details`.
    #[doc(hidden)]
    pub pyspark_message: Option<String>,
    /// grpcio's error of a failed call that PySpark's client leaves to grpc, as it does not
    /// convert it (`cloneSession`, `copyFromLocalToFs`, and the caching of artifacts), which the
    /// Python binding raises as grpcio's `RpcError`; or, of an error of the kind
    /// [`SparkErrorKind::BuiltinValueError`], the error of the call whose `google.rpc.Status` does
    /// not match it, or whose details are not one, of which PySpark raises the `ValueError` of
    /// `rpc_status.from_call`, or protobuf's `DecodeError` of `Error parsing message with type
    /// 'google.rpc.Status': ...`, as it handles it, its `__context__` (see `from_call_error`).
    /// Only in the dialect of grpcio (see [`crate::set_grpcio_dialect`]); the Rust client converts
    /// these failures as any other.
    #[doc(hidden)]
    pub grpcio: Option<Box<GrpcioError>>,
}

/// The PySpark exception hierarchy leaf we map to.
///
/// These variants correspond to the Python exception classes in
/// `pyspark/errors/exceptions/base.py` and `pyspark/errors/exceptions/connect.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SparkErrorKind {
    /// ``pyspark.errors.PySparkValueError``
    ValueError,
    /// ``pyspark.errors.PySparkTypeError``
    TypeError,
    /// ``pyspark.errors.PySparkIndexError``
    IndexError,
    /// ``pyspark.errors.PySparkAttributeError``
    AttributeError,
    /// ``pyspark.errors.PySparkKeyError``
    KeyError,
    /// ``pyspark.errors.PySparkRuntimeError``
    RuntimeError,
    /// ``pyspark.errors.PySparkNotImplementedError``
    NotImplementedError,
    /// ``pyspark.errors.PySparkAssertionError``
    AssertionError,
    /// ``pyspark.errors.PySparkPicklingError``
    PicklingError,
    /// ``pyspark.errors.PySparkImportError``
    ImportError,
    /// ``pyspark.errors.exceptions.connect.SparkConnectException`` and subclasses
    Connect,
    /// ``pyspark.errors.exceptions.connect.SparkConnectException`` of a plain message, which has
    /// no error class: the reference client raises it for a malformed result, such as a chunk of
    /// an Arrow batch that is out of order.
    ConnectMessage,
    /// ``pyspark.errors.exceptions.connect.SparkConnectGrpcException``
    ConnectGrpc,
    /// ``pyspark.errors.exceptions.base.AnalysisException``
    Analysis,
    /// ``pyspark.errors.exceptions.base.SessionNotSameException``
    SessionNotSame,
    /// ``pyspark.errors.exceptions.base.TempTableAlreadyExistsException``
    TempTableAlreadyExists,
    /// ``pyspark.errors.exceptions.base.ParseException``
    Parse,
    /// ``pyspark.errors.exceptions.base.IllegalArgumentException``
    IllegalArgument,
    /// ``pyspark.errors.exceptions.base.ArithmeticException``
    Arithmetic,
    /// ``pyspark.errors.exceptions.base.UnsupportedOperationException``
    UnsupportedOperation,
    /// ``pyspark.errors.exceptions.base.ArrayIndexOutOfBoundsException``
    ArrayIndexOutOfBounds,
    /// ``pyspark.errors.exceptions.base.DateTimeException``
    DateTime,
    /// ``pyspark.errors.exceptions.base.NumberFormatException``
    NumberFormat,
    /// ``pyspark.errors.exceptions.base.StreamingQueryException``
    StreamingQuery,
    /// ``pyspark.errors.exceptions.base.StreamingPythonRunnerInitializationException``
    StreamingPythonRunnerInitialization,
    /// ``pyspark.errors.exceptions.base.QueryExecutionException``
    QueryExecution,
    /// ``pyspark.errors.exceptions.base.PythonException``
    Python,
    /// ``RecursionError``, a Python builtin: the executions nested in callbacks are too deep.
    RecursionError,
    /// ``UnicodeDecodeError``, a Python builtin: a string of the result is not valid UTF-8.
    UnicodeDecode,
    /// ``UnicodeEncodeError``, a Python builtin: a str of a request that UTF-8 cannot encode, such
    /// as a lone surrogate, which the reference client raises when it builds the plan.
    UnicodeEncode,
    /// ``ValueError``, a Python builtin: a connection-string value that `int()` or `urllib`
    /// rejects, which the reference client raises unwrapped rather than as a `PySparkValueError`.
    BuiltinValueError,
    /// ``TypeError``, a Python builtin: an argument that a protobuf field does not take, which the
    /// reference client raises when it builds the plan of a request, unwrapped.
    BuiltinTypeError,
    /// ``OverflowError``, a Python builtin: an int that a protobuf field does not take, as
    /// [`SparkErrorKind::BuiltinTypeError`] is raised.
    BuiltinOverflowError,
    /// ``RuntimeError``, a Python builtin: the release of an execution once the interpreter has
    /// begun to shut down, which the thread pool of the reference client's releases refuses.
    BuiltinRuntimeError,
    /// ``AttributeError``, a Python builtin: an argument that lacks a method that the reference
    /// client calls when it builds the plan of a request, such as the `to_plan` of a join's `on`.
    BuiltinAttributeError,
    /// ``AssertionError``, a Python builtin: an assertion of the reference client that fails, such
    /// as that the Observation that an execution is given of the name of observed metrics is used.
    BuiltinAssertionError,
    /// ``pyspark.errors.exceptions.base.SparkRuntimeException``
    SparkRuntime,
    /// ``pyspark.errors.exceptions.base.SparkUpgradeException``
    SparkUpgrade,
    /// ``pyspark.errors.exceptions.base.SparkNoSuchElementException``
    SparkNoSuchElement,
    /// ``pyspark.errors.exceptions.base.UnknownException``
    Unknown,
    /// ``pyspark.errors.exceptions.connect.InvalidPlanInput``
    InvalidPlanInput,
    /// ``pyspark.errors.exceptions.connect.PickleException``
    PickleException,
    /// ``pyspark.errors.exceptions.connect.SparkException``
    Spark,
}

/// The bytes as lowercase hex, to keep them in the string parameters of an error.
fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn hex_decode(hex: &str) -> Option<Vec<u8>> {
    hex.as_bytes()
        .chunks(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).ok()?, 16).ok())
        .collect()
}

impl SparkError {
    /// Create a ValueError-kind error with an error class and parameters.
    pub fn value(error_class: &str, params: &[(&str, &str)]) -> Self {
        Self::classed(SparkErrorKind::ValueError, error_class, params)
    }

    /// Create a Connect-kind error with a plain message (no error class).
    pub fn connect_msg(message: impl Into<String>) -> Self {
        Self {
            kind: SparkErrorKind::Connect,
            error_class: String::new(),
            params: BTreeMap::new(),
            message: message.into(),
            sql_state: None,
            contexts: Vec::new(),
            server_stacktrace: None,
            grpc_code: None,
            grpc_details: None,
            operation_id: None,
            server: None,
            pyspark_message: None,
            grpcio: None,
        }
    }

    /// Create a ConnectMessage-kind error with a plain message (no error class), which is a
    /// `SparkConnectException` in Python.
    pub fn connect_exception_msg(message: impl Into<String>) -> Self {
        Self {
            kind: SparkErrorKind::ConnectMessage,
            ..Self::connect_msg(message)
        }
    }

    /// Create a ValueError-kind error with a plain message (no error class).
    ///
    /// For invalid client-side configuration (e.g. a malformed connection string)
    /// that has no registered error class. Surfaces as `PySparkValueError` on the
    /// Python side, matching how the reference client reports connection-string
    /// problems, rather than the generic runtime kind.
    pub fn value_msg(message: impl Into<String>) -> Self {
        Self {
            kind: SparkErrorKind::ValueError,
            error_class: String::new(),
            params: BTreeMap::new(),
            message: message.into(),
            sql_state: None,
            contexts: Vec::new(),
            server_stacktrace: None,
            grpc_code: None,
            grpc_details: None,
            operation_id: None,
            server: None,
            pyspark_message: None,
            grpcio: None,
        }
    }

    /// Create a `BuiltinValueError`-kind error with a plain message: a `ValueError` of the Python
    /// builtin, which is what `int()` and `urllib` raise, as the reference client lets them.
    pub fn builtin_value_msg(message: impl Into<String>) -> Self {
        let mut error = Self::value_msg(message);
        error.kind = SparkErrorKind::BuiltinValueError;
        error
    }

    /// Create a `BuiltinAttributeError`-kind error with a plain message: an `AttributeError` of
    /// the Python builtin, as the reference client raises it when it builds a plan of an object
    /// that lacks the method that it calls.
    pub fn builtin_attribute(message: impl Into<String>) -> Self {
        let mut error = Self::value_msg(message);
        error.kind = SparkErrorKind::BuiltinAttributeError;
        error
    }

    /// Create a `BuiltinAssertionError`-kind error of no message: the `AssertionError` of a bare
    /// `assert` of the reference client, which it lets out of an execution.
    pub fn builtin_assertion() -> Self {
        let mut error = Self::value_msg("");
        error.kind = SparkErrorKind::BuiltinAssertionError;
        error
    }

    /// Create a `BuiltinRuntimeError`-kind error with a plain message: a `RuntimeError` of the
    /// Python builtin, which is what `ThreadPoolExecutor.submit` raises once the interpreter has
    /// begun to shut down, as the reference client lets it.
    pub fn builtin_runtime_msg(message: impl Into<String>) -> Self {
        let mut error = Self::value_msg(message);
        error.kind = SparkErrorKind::BuiltinRuntimeError;
        error
    }

    /// Create a RecursionError-kind error with a plain message (no error class).
    pub fn recursion_msg(message: impl Into<String>) -> Self {
        Self {
            kind: SparkErrorKind::RecursionError,
            error_class: String::new(),
            params: BTreeMap::new(),
            message: message.into(),
            sql_state: None,
            contexts: Vec::new(),
            server_stacktrace: None,
            grpc_code: None,
            grpc_details: None,
            operation_id: None,
            server: None,
            pyspark_message: None,
            grpcio: None,
        }
    }

    /// Create an UnicodeDecode-kind error for the string `value` of the result column `column`,
    /// which is not valid UTF-8. The bytes are kept, so that the Python client can raise the error
    /// of Python's own decoder for them.
    pub fn invalid_utf8(column: usize, value: &[u8]) -> Self {
        let mut error = Self::classed(
            SparkErrorKind::UnicodeDecode,
            "",
            &[
                ("column", &column.to_string()),
                ("value", &hex_encode(value)),
            ],
        );
        let reason = std::str::from_utf8(value)
            .err()
            .map_or_else(String::new, |e| e.to_string());
        error.message = format!("Invalid UTF-8 in column {column} of the result: {reason}");
        error
    }

    /// The result column of the string of an error from `invalid_utf8`.
    pub fn utf8_column(&self) -> Option<usize> {
        if self.kind != SparkErrorKind::UnicodeDecode {
            return None;
        }
        self.params.get("column")?.parse().ok()
    }

    /// The bytes of the string of an error from `invalid_utf8`.
    pub fn invalid_utf8_value(&self) -> Option<Vec<u8>> {
        if self.kind != SparkErrorKind::UnicodeDecode {
            return None;
        }
        hex_decode(self.params.get("value")?)
    }

    /// Create an UnicodeEncode-kind error for a str that UTF-8 cannot encode, given as the UTF-32
    /// little-endian bytes of its code points, which keep the surrogates. The bytes are kept, so
    /// that the Python client can raise the error of Python's own encoder for them.
    pub fn unencodable_str(utf32le: &[u8]) -> Self {
        let mut error = Self::classed(
            SparkErrorKind::UnicodeEncode,
            "",
            &[("value", &hex_encode(utf32le))],
        );
        error.message = "A string that UTF-8 cannot encode".to_string();
        error
    }

    /// The UTF-32 little-endian bytes of the str of an error from `unencodable_str`.
    pub fn unencodable_value(&self) -> Option<Vec<u8>> {
        if self.kind != SparkErrorKind::UnicodeEncode {
            return None;
        }
        hex_decode(self.params.get("value")?)
    }

    /// Create an error of a specific kind with an error class and parameters.
    pub fn classed(kind: SparkErrorKind, error_class: &str, params: &[(&str, &str)]) -> Self {
        Self {
            kind,
            error_class: error_class.to_string(),
            params: params
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
            message: String::new(),
            sql_state: None,
            contexts: Vec::new(),
            server_stacktrace: None,
            grpc_code: None,
            grpc_details: None,
            operation_id: None,
            server: None,
            pyspark_message: None,
            grpcio: None,
        }
    }

    /// Get the rendered message for this error, performing template substitution.
    ///
    /// If an error_class is set, looks up the message template in error-conditions.json,
    /// substitutes `<param_name>` placeholders with values from `params`, and returns
    /// `[ERROR_CLASS] rendered_message`. Otherwise, returns the pre-set message. The message
    /// of an error that the server raised is the server's, which already names the condition.
    ///
    /// Mirrors `PySparkException.getMessage()` from `pyspark/errors/exceptions/base.py`.
    pub fn message(&self) -> String {
        if self.error_class.is_empty() || self.server.is_some() {
            return self.message.clone();
        }

        // Try to render from error-conditions registry
        if let Ok(rendered) = render_message(&self.error_class, &self.params) {
            format!("[{}] {}", self.error_class, rendered)
        } else {
            // Fallback: format as-is if error class not found
            if self.message.is_empty() {
                format!("[{}]", self.error_class)
            } else {
                format!("[{}] {}", self.error_class, self.message)
            }
        }
    }

    /// Get the SQL state code for this error, if known.
    pub fn sql_state(&self) -> Option<String> {
        if self.sql_state.is_some() {
            self.sql_state.clone()
        } else if !self.error_class.is_empty() {
            get_sql_state(&self.error_class).map(|s| s.to_string())
        } else {
            None
        }
    }

    /// Convert the gRPC Status of a failed call of one response to a SparkError: see
    /// [`SparkError::from_grpc_status_of`].
    pub fn from_grpc_status(status: Status) -> Self {
        Self::from_grpc_status_of(status, GrpcCall::Unary)
    }

    /// Convert the gRPC Status of a failed call of the kind `call` to a SparkError.
    ///
    /// A status with an `ErrorInfo` is an error that the server raised. As
    /// `pyspark.errors.exceptions.connect.convert_exception` does, its exception classes,
    /// condition, message parameters, SQL state and stack trace are read from the
    /// `ErrorInfo`, and its kind from the classes. [`SparkError::with_details`] completes it
    /// with what `FetchErrorDetails` returns. The code is the one grpc-python reports (see
    /// [`crate::retries::grpc_python_code`]), so that a dropped connection is `UNAVAILABLE`.
    ///
    /// Any other status is the error of the message `[code] details`. Its hidden
    /// `pyspark_message`, which the Python binding raises, is the message as
    /// `SparkConnectClient._handle_rpc_error` gives it: the message of its `google.rpc.Status`,
    /// if it has one, else the text of grpcio's error of the call, `str()` of its `RpcError`, of
    /// the class of `call`.
    pub fn from_grpc_status_of(status: Status, call: GrpcCall) -> Self {
        // A response that the client refused, whose kept messages nest past the bound of what it
        // holds (`spark_connect_proto::MAX_KEPT_DEPTH`): its `RecursionError`.
        if let Some(too_deep) = spark_connect_proto::kept_too_deep(&status) {
            return Self::recursion_msg(too_deep.to_string());
        }
        let message = crate::client::status_details(&status);
        let code = crate::retries::grpc_python_code(&status);
        let details = status.details().to_vec();
        let (mut err, pyspark_message) = if code == tonic::Code::DeadlineExceeded {
            // As `SparkConnectClient._handle_rpc_error`, which raises this for a call past its
            // per-RPC deadline before it reads the error of the status.
            let deadline_exceeded = |error: String| {
                format!(
                    "{error}: RPC deadline exceeded. The client applies per-RPC timeouts to \
                     prevent silent hangs on broken connections. Deadlines can be configured via \
                     the rpc_deadlines parameter of SparkConnectClient. To disable all \
                     deadlines: SparkConnectClient(url, rpc_deadlines=RpcDeadlines.disabled())."
                )
            };
            (
                Self::connect_msg(deadline_exceeded(format!("[{code}] {message}"))),
                Some(deadline_exceeded(grpc_call_text(call, code, &message))),
            )
        } else {
            match parse_error_info(&details) {
                Some(info) => {
                    let mut err = Self::from_error_info(info, message);
                    if let Some(server) = err.server.as_mut() {
                        server.rpc_error = Some((call, grpcio_status(&status).text(call)));
                    }
                    (err, None)
                }
                None => {
                    // The `grpc-status-details-bin` of the status is the `google.rpc.Status` of
                    // `rpc_status.from_call`, whose message is that of the call.
                    let pyspark_message = if !details.is_empty() {
                        message.clone()
                    } else {
                        grpcio_status(&status).text(call)
                    };
                    let err = Self::connect_msg(format!("[{code}] {message}"));
                    (err, Some(pyspark_message))
                }
            }
        };
        err.pyspark_message = pyspark_message;
        // Preserve the raw status so the PyO3 layer can rebuild the exact pyspark exception.
        err.grpc_code = Some(code as i32);
        if !details.is_empty() {
            err.grpc_details = Some(details);
        }
        err
    }

    /// An error that the server raised, from its `ErrorInfo`.
    fn from_error_info(info: RpcErrorInfo, message: String) -> Self {
        let mut metadata = info.metadata;
        let classes = metadata
            .get("classes")
            .and_then(|classes| serde_json::from_str(classes).ok())
            .unwrap_or_default();
        let params = metadata
            .get("messageParameters")
            .and_then(|params| serde_json::from_str(params).ok())
            .unwrap_or_default();
        let parameter_order = metadata
            .get("messageParameters")
            .map(|params| parameter_names(params))
            .unwrap_or_default();
        let mut err = Self {
            kind: SparkErrorKind::Unknown,
            error_class: metadata.remove("errorClass").unwrap_or_default(),
            params,
            message,
            sql_state: metadata.remove("sqlState"),
            contexts: Vec::new(),
            server_stacktrace: metadata.remove("stackTrace"),
            grpc_code: None,
            grpc_details: None,
            operation_id: None,
            server: Some(Box::new(ServerErrorInfo {
                classes,
                reason: info.reason,
                error_id: metadata.remove("errorId"),
                parameter_order,
                ..Default::default()
            })),
            pyspark_message: None,
            grpcio: None,
        };
        err.kind = err.server_kind();
        err
    }

    /// The error of collecting the metrics of an Observation, which the server reports in
    /// its `ObservedMetrics`, as PySpark's `convert_observation_errors` makes it: the root
    /// error of `errors`, as a server error of the details of `FetchErrorDetails`.
    pub fn from_observation_errors(
        root_error_idx: i32,
        errors: Vec<spark_connect_proto::fetch_error_details_response::Error>,
    ) -> Self {
        let Some(root) = usize::try_from(root_error_idx)
            .ok()
            .and_then(|index| errors.get(index))
        else {
            return Self::connect_msg("Observation error: invalid root_error_idx");
        };
        let throwable = root.spark_throwable.clone().unwrap_or_default();
        let mut err = Self {
            kind: SparkErrorKind::Unknown,
            error_class: throwable.error_class.clone().unwrap_or_default(),
            params: BTreeMap::new(),
            message: root.message.clone(),
            sql_state: throwable.sql_state.clone(),
            contexts: Vec::new(),
            server_stacktrace: None,
            grpc_code: None,
            grpc_details: None,
            operation_id: None,
            server: Some(Box::new(ServerErrorInfo {
                classes: root.error_type_hierarchy.clone(),
                ..Default::default()
            })),
            pyspark_message: None,
            grpcio: None,
        };
        err.kind = err.server_kind();
        let details = spark_connect_proto::FetchErrorDetailsResponse {
            root_error_idx: Some(root_error_idx),
            errors,
            ..Default::default()
        };
        err.with_details(Some(details), false)
    }

    /// Completes an error that the server raised, as
    /// `pyspark.errors.exceptions.connect._convert_exception` does: with the error's details
    /// as `FetchErrorDetails` returned them, if it did, and whether the message shows the
    /// server's stack trace. Without the details, the message shows the stack trace only if
    /// the `ErrorInfo` had one. Any other error is returned as is.
    pub fn with_details(
        mut self,
        details: Option<spark_connect_proto::FetchErrorDetailsResponse>,
        display_stacktrace: bool,
    ) -> Self {
        if self.server.is_none() {
            return self;
        }
        let root = details.and_then(|details| {
            let index = usize::try_from(details.root_error_idx?).ok()?;
            (index < details.errors.len()).then_some((index, details.errors))
        });
        let display_stacktrace = match root {
            Some((index, errors)) => {
                let root = &errors[index];
                self.message = root.message.clone();
                self.server_stacktrace = Some(jvm_stacktrace(index, &errors));
                let throwable = root.spark_throwable.clone().unwrap_or_default();
                if self.error_class.is_empty() {
                    if let Some(error_class) = throwable.error_class {
                        self.error_class = error_class;
                    }
                }
                self.params = throwable.message_parameters.into_iter().collect();
                self.contexts = throwable
                    .query_contexts
                    .into_iter()
                    .map(QueryContext::from)
                    .collect();
                if let Some(server) = self.server.as_mut() {
                    server.breaking_change =
                        throwable.breaking_change_info.map(BreakingChangeInfo::from);
                }
                display_stacktrace
            }
            None => {
                display_stacktrace
                    && self
                        .server_stacktrace
                        .as_deref()
                        .is_some_and(|stacktrace| !stacktrace.is_empty())
            }
        };
        if let Some(server) = self.server.as_mut() {
            server.display_stacktrace = display_stacktrace;
        }
        self.kind = self.server_kind();
        self
    }

    /// The kind of an error that the server raised, from its exception classes. As
    /// `_convert_exception` picks the exception, the first class, from the most specific one,
    /// that `EXCEPTION_CLASS_MAPPING` maps decides it, and an error with none is unknown.
    fn server_kind(&self) -> SparkErrorKind {
        let classes = self
            .server
            .as_ref()
            .map(|server| server.classes.as_slice())
            .unwrap_or_default();
        if classes
            .iter()
            .any(|class| class == "org.apache.spark.api.python.PythonException")
        {
            return SparkErrorKind::Python;
        }
        classes
            .iter()
            .find_map(|class| exception_class_kind(class, &self.message))
            .unwrap_or(SparkErrorKind::Unknown)
    }

    /// Attach the ExecutePlan operation id this error was raised for.
    pub fn with_operation_id(mut self, operation_id: impl Into<String>) -> Self {
        self.operation_id = Some(operation_id.into());
        self
    }

    /// Get the error condition (error class).
    ///
    /// Mirrors `PySparkException.getCondition()` from PySpark.
    pub fn get_condition(&self) -> Option<String> {
        if self.error_class.is_empty() {
            None
        } else {
            Some(self.error_class.clone())
        }
    }

    /// Get the error class (deprecated, use `get_condition` instead).
    ///
    /// Mirrors `PySparkException.getErrorClass()` from PySpark (deprecated).
    pub fn get_error_class(&self) -> Option<String> {
        self.get_condition()
    }

    /// Get the message parameters.
    ///
    /// Mirrors `PySparkException.getMessageParameters()` from PySpark.
    pub fn get_message_parameters(&self) -> Option<BTreeMap<String, String>> {
        if self.params.is_empty() && self.error_class.is_empty() {
            None
        } else {
            Some(self.params.clone())
        }
    }

    /// Get the SQL state code, if known.
    ///
    /// Mirrors `PySparkException.getSqlState()` from PySpark.
    pub fn get_sql_state(&self) -> Option<String> {
        if self.sql_state.is_some() {
            self.sql_state.clone()
        } else if !self.error_class.is_empty() {
            get_sql_state(&self.error_class).map(|s| s.to_string())
        } else {
            None
        }
    }

    /// Get the rendered error message.
    ///
    /// Mirrors `PySparkException.getMessage()` from PySpark.
    pub fn get_message(&self) -> String {
        self.message()
    }

    /// Get the query contexts where this error occurred.
    ///
    /// Mirrors `PySparkException.getQueryContext()` from PySpark.
    pub fn get_query_context(&self) -> Vec<QueryContext> {
        self.contexts.clone()
    }

    /// Get the server-side stack trace, if available.
    pub fn get_stacktrace(&self) -> Option<String> {
        self.server_stacktrace.clone()
    }

    /// Raw gRPC status code, if this error came from a gRPC Status.
    pub fn grpc_code(&self) -> Option<i32> {
        self.grpc_code
    }

    /// Raw `grpc-status-details-bin` bytes (serialized `google.rpc.Status`), if present.
    pub fn grpc_details(&self) -> Option<&[u8]> {
        self.grpc_details.as_deref()
    }
}

impl std::fmt::Display for SparkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message())
    }
}

impl std::error::Error for SparkError {}

pub type Result<T> = std::result::Result<T, SparkError>;

// ============================================================================
// gRPC error detail parsing
// ============================================================================

/// The kind of a gRPC call, which names the class of grpcio's error of its failure, as PySpark
/// raises it: `_InactiveRpcError` for a call of one response, `_MultiThreadedRendezvous` for a
/// call of a stream of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrpcCall {
    /// A call of one response, such as `Config`, `AnalyzePlan`, or `AddArtifacts` of a stream of
    /// requests.
    Unary,
    /// A call of a stream of responses: `ExecutePlan` and `ReattachExecute`.
    ServerStreaming,
}

/// The name of `code` in grpcio's `StatusCode` and in gRPC's C core.
fn status_code_name(code: tonic::Code) -> &'static str {
    use tonic::Code;
    match code {
        Code::Ok => "OK",
        Code::Cancelled => "CANCELLED",
        Code::Unknown => "UNKNOWN",
        Code::InvalidArgument => "INVALID_ARGUMENT",
        Code::DeadlineExceeded => "DEADLINE_EXCEEDED",
        Code::NotFound => "NOT_FOUND",
        Code::AlreadyExists => "ALREADY_EXISTS",
        Code::PermissionDenied => "PERMISSION_DENIED",
        Code::ResourceExhausted => "RESOURCE_EXHAUSTED",
        Code::FailedPrecondition => "FAILED_PRECONDITION",
        Code::Aborted => "ABORTED",
        Code::OutOfRange => "OUT_OF_RANGE",
        Code::Unimplemented => "UNIMPLEMENTED",
        Code::Internal => "INTERNAL",
        Code::Unavailable => "UNAVAILABLE",
        Code::DataLoss => "DATA_LOSS",
        Code::Unauthenticated => "UNAUTHENTICATED",
    }
}

/// The text of grpcio's error of a failed call of the kind `call`, `str()` of its `RpcError`,
/// which PySpark's errors of a status without a `google.rpc.Status` show: the class of the error,
/// the code, the details, which are the message of the status, and the `debug_error_string` of
/// gRPC's C core, which is the name of the code and the details, as grpcio 1.85 gives it. The
/// details of a failure that the client detects itself, such as a connection that is refused,
/// are tonic's message, where grpcio's are those of gRPC's C core, such as `failed to connect to
/// all addresses; last error: ...`.
fn grpc_call_text(call: GrpcCall, code: tonic::Code, details: &str) -> String {
    let name = status_code_name(code);
    let debug_error_string = if details.is_empty() {
        name.to_string()
    } else {
        format!("{name}:{details}")
    };
    grpc_call_text_of(call, code, details, &debug_error_string)
}

/// The text of grpcio's error of a failed call of the kind `call`, of the code, the details and
/// the `debug_error_string` given.
fn grpc_call_text_of(
    call: GrpcCall,
    code: tonic::Code,
    details: &str,
    debug_error_string: &str,
) -> String {
    let class = match call {
        GrpcCall::Unary => "_InactiveRpcError",
        GrpcCall::ServerStreaming => "_MultiThreadedRendezvous",
    };
    let name = status_code_name(code);
    format!(
        "<{class} of RPC that terminated with:\n\tstatus = StatusCode.{name}\n\tdetails = \
         \"{details}\"\n\tdebug_error_string = \"{debug_error_string}\"\n>"
    )
}

/// The details of grpcio's error of a response that it fails to deserialize, of no
/// `debug_error_string` (`None`), as PySpark raises it of a corrupt response.
const DESERIALIZE_FAILURE: &str = "Exception deserializing response!";

/// What grpcio's error of a failed call holds of its status: the code, as grpc-python reports it
/// (see [`crate::retries::grpc_python_code`]), the details, as grpcio gives them (see
/// [`crate::client::status_details`]), and the `debug_error_string` of gRPC's C core, the name of
/// the code and the details; but of a response that the client failed to decode, grpcio's of a
/// response that it fails to deserialize, `Exception deserializing response!` of no
/// `debug_error_string`.
pub(crate) struct GrpcioStatus {
    pub(crate) code: tonic::Code,
    pub(crate) details: String,
    pub(crate) debug_error_string: Option<String>,
}

impl GrpcioStatus {
    /// `str()` of grpcio's error of a failed call of the kind `call` of this status, whose
    /// `debug_error_string` reads `None` when it has none.
    pub(crate) fn text(&self, call: GrpcCall) -> String {
        grpc_call_text_of(
            call,
            self.code,
            &self.details,
            self.debug_error_string.as_deref().unwrap_or("None"),
        )
    }
}

/// What grpcio's error of the failed call that ended with `status` holds of it (see
/// [`GrpcioStatus`]).
pub(crate) fn grpcio_status(status: &Status) -> GrpcioStatus {
    let code = crate::retries::grpc_python_code(status);
    if is_decode_failure(status) {
        return GrpcioStatus {
            code,
            details: DESERIALIZE_FAILURE.to_string(),
            debug_error_string: None,
        };
    }
    let details = crate::client::status_details(status);
    let name = status_code_name(code);
    let debug_error_string = if details.is_empty() {
        name.to_string()
    } else {
        format!("{name}:{details}")
    };
    GrpcioStatus {
        code,
        details,
        debug_error_string: Some(debug_error_string),
    }
}

/// Whether `status` is the client's own of a response that it failed to decode, of the codec of
/// the generated client, which tags it (see [`spark_connect_proto::ClientCodec`]): a status of the
/// same code and text that a server sends, as of its failure to decode a request, is not.
fn is_decode_failure(status: &Status) -> bool {
    status.code() == tonic::Code::Internal && spark_connect_proto::is_client_decode_error(status)
}

/// grpcio's error of a failed call: its code, its details and its `debug_error_string`, its
/// trailing metadata, and its text, `str()` and `repr()` of PySpark's `RpcError`, which its logs
/// show. Not a stable API.
#[doc(hidden)]
#[derive(Debug, Clone)]
pub struct GrpcioError {
    /// The kind of the call, which names the class of the error.
    pub call: GrpcCall,
    /// The code, as grpc-python reports it (see [`crate::retries::grpc_python_code`]).
    pub code: tonic::Code,
    /// The details, as grpcio gives them (see [`crate::client::status_details`]).
    pub details: String,
    /// The `debug_error_string` of gRPC's C core: the name of the code and the details, or None
    /// of a response that the client failed to decode, as grpcio's error of a response that it
    /// fails to deserialize has none.
    pub debug_error_string: Option<String>,
    /// The trailing metadata of the call: the ASCII entries, then the binary ones, of which the
    /// `grpc-status-details-bin` of a `google.rpc.Status`.
    pub trailing_metadata: Vec<(String, Vec<u8>)>,
    /// `str()` of the error: `<_InactiveRpcError of RPC that terminated with: ...>`.
    pub text: String,
}

/// grpcio's error of the failed call of the kind `call` that ended with `status`. Not a stable
/// API.
#[doc(hidden)]
pub fn grpcio_error(status: &Status, call: GrpcCall) -> GrpcioError {
    let grpcio = grpcio_status(status);
    let mut trailing_metadata = Vec::new();
    // As grpcio's metadata, of which gRPC's core keeps the headers that it reads itself.
    let reserved = |key: &str| {
        key.starts_with(':')
            || matches!(
                key,
                "content-type"
                    | "te"
                    | "user-agent"
                    | "host"
                    | "content-encoding"
                    | "accept-encoding"
                    | "grpc-encoding"
                    | "grpc-accept-encoding"
                    | "grpc-status"
                    | "grpc-message"
                    | "grpc-timeout"
                    | "grpc-internal-encoding-request"
                    | "grpc-previous-rpc-attempts"
                    | "grpc-retry-pushback-ms"
                    | "grpc-server-stats-bin"
                    | "grpc-tags-bin"
                    | "grpc-trace-bin"
            )
    };
    for entry in status.metadata().iter() {
        let key = match &entry {
            tonic::metadata::KeyAndValueRef::Ascii(key, _) => key.as_str(),
            tonic::metadata::KeyAndValueRef::Binary(key, _) => key.as_str(),
        };
        if reserved(key) {
            continue;
        }
        match entry {
            tonic::metadata::KeyAndValueRef::Ascii(key, value) => trailing_metadata
                .push((key.as_str().to_string(), value.as_encoded_bytes().to_vec())),
            tonic::metadata::KeyAndValueRef::Binary(key, value) => {
                if let Ok(value) = value.to_bytes() {
                    trailing_metadata.push((key.as_str().to_string(), value.to_vec()));
                }
            }
        }
    }
    // As gRPC's core gives the call details that are not base64, empty.
    if !status.details().is_empty() || crate::metadata::has_undecoded_details(status) {
        trailing_metadata.push((
            "grpc-status-details-bin".to_string(),
            status.details().to_vec(),
        ));
    }
    GrpcioError {
        call,
        code: grpcio.code,
        text: grpcio.text(call),
        details: grpcio.details,
        debug_error_string: grpcio.debug_error_string,
        trailing_metadata,
    }
}

/// `str()` of grpcio's error of the failed call of the kind `call` that ended with `status`, as
/// PySpark's logs show it.
pub(crate) fn grpcio_error_text(status: &Status, call: GrpcCall) -> String {
    grpcio_status(status).text(call)
}

/// The text of protobuf's `DecodeError` of details of a status that are not a `google.rpc.Status`,
/// as upb gives it, which grpc_status's `rpc_status.from_call` raises (see [`from_call_error`]).
pub(crate) const STATUS_DECODE_ERROR: &str =
    "Error parsing message with type 'google.rpc.Status': Wire format was corrupt";

/// In the dialect of grpcio, the error that `grpc_status.rpc_status.from_call` raises of grpcio's
/// error of the failed call of the kind `call` that ended with `status`, as PySpark's client reads
/// the status of each failed call of it as it handles it (`_handle_rpc_error`), and so do its
/// `DefaultPolicy`, of a `RetryInfo`, and the iterator of a reattachable execution
/// (`_call_iter`), which raise it in place of the error, its context: the `ValueError` of a
/// `google.rpc.Status`, of its `grpc-status-details-bin`, that is not of its code or of its
/// details, as of metadata past gRPC's soft limit, whose call fails `RESOURCE_EXHAUSTED` of the
/// details of an error of the server, or of details that are not base64, which gRPC's core gives
/// the call empty, a status of the code `OK` (see [`crate::metadata`]); or protobuf's
/// `DecodeError` of details that are not a `google.rpc.Status` ([`STATUS_DECODE_ERROR`]). The
/// text of the `ValueError` is grpc_status's: `Code in Status proto (StatusCode.INTERNAL) doesn't
/// match status code (StatusCode.RESOURCE_EXHAUSTED)`, of a code that is not one of gRPC's
/// `Invalid status code <code>`, or `Message in Status proto (...) doesn't match status details
/// (...)`. Both are errors of the kind [`SparkErrorKind::BuiltinValueError`] of grpcio's error of
/// the call (see [`SparkError::grpcio`]), which the Python binding raises as their classes. None
/// of a status of no details, or of details that the status matches, and out of the dialect of
/// grpcio.
pub(crate) fn from_call_error(status: &Status, call: GrpcCall) -> Option<SparkError> {
    from_call_error_in(status, call, crate::connector::grpcio_dialect())
}

/// As [`from_call_error`], in the dialect of grpcio if `grpcio`.
fn from_call_error_in(status: &Status, call: GrpcCall, grpcio: bool) -> Option<SparkError> {
    if !grpcio || (status.details().is_empty() && !crate::metadata::has_undecoded_details(status)) {
        return None;
    }
    let message = match RpcStatus::decode(status.details()) {
        Err(_) => STATUS_DECODE_ERROR.to_string(),
        Ok(proto) => {
            let grpcio = grpcio_status(status);
            if proto.code != grpcio.code as i32 {
                if (0..=16).contains(&proto.code) {
                    format!(
                        "Code in Status proto (StatusCode.{}) doesn't match status code \
                         (StatusCode.{})",
                        status_code_name(tonic::Code::from_i32(proto.code)),
                        status_code_name(grpcio.code)
                    )
                } else {
                    format!("Invalid status code {}", proto.code)
                }
            } else if proto.message != grpcio.details {
                format!(
                    "Message in Status proto ({}) doesn't match status details ({})",
                    proto.message, grpcio.details
                )
            } else {
                return None;
            }
        }
    };
    let mut error = SparkError::builtin_value_msg(message);
    error.grpcio = Some(Box::new(grpcio_error(status, call)));
    Some(error)
}

/// A `google.rpc.ErrorInfo` of the entries of its metadata as they are on the wire.
#[derive(Clone, PartialEq, Message)]
struct RpcErrorInfoEntries {
    #[prost(string, tag = "1")]
    reason: String,
    #[prost(string, tag = "2")]
    domain: String,
    #[prost(message, repeated, tag = "3")]
    metadata: Vec<MetadataEntry>,
}

/// An entry of the map of the metadata of an `ErrorInfo`, as it is on the wire.
#[derive(Clone, PartialEq, Message)]
struct MetadataEntry {
    #[prost(string, tag = "1")]
    key: String,
    #[prost(string, tag = "2")]
    value: String,
}

/// The text of the `google.rpc.ErrorInfo` of the `grpc-status-details-bin` `details` of a status,
/// if it has one, as `str()` of Python's protobuf (upb) prints it, which PySpark's client logs
/// (`Received ErrorInfo: ...`): the text format of the fields that are set, of the entries of the
/// metadata in the order of upb, of the octal escapes of the bytes that are not printable ASCII.
pub(crate) fn error_info_text(details: &[u8]) -> Option<String> {
    if details.is_empty() {
        return None;
    }
    let info = RpcStatus::decode(details)
        .ok()?
        .details
        .into_iter()
        .filter(|detail| detail.type_url == "type.googleapis.com/google.rpc.ErrorInfo")
        .find_map(|detail| RpcErrorInfoEntries::decode(&detail.value[..]).ok())?;
    let mut text = String::new();
    let field = |text: &mut String, indent: &str, name: &str, value: &str| {
        if !value.is_empty() {
            text.push_str(&format!(
                "{indent}{name}: \"{}\"\n",
                text_format_escape(value)
            ));
        }
    };
    field(&mut text, "", "reason", &info.reason);
    field(&mut text, "", "domain", &info.domain);
    // A map of the entries, of the last of a key, in the order of upb's text encoder, which
    // sorts the keys by their bytes, descending, and a key before the longer ones it begins.
    let mut entries: Vec<MetadataEntry> = Vec::new();
    for entry in info.metadata {
        entries.retain(|kept| kept.key != entry.key);
        entries.push(entry);
    }
    entries.sort_by(|a, b| {
        let (a, b) = (a.key.as_bytes(), b.key.as_bytes());
        let common = a.len().min(b.len());
        a[..common]
            .cmp(&b[..common])
            .reverse()
            .then(a.len().cmp(&b.len()))
    });
    for entry in &entries {
        text.push_str("metadata {\n");
        field(&mut text, "  ", "key", &entry.key);
        field(&mut text, "  ", "value", &entry.value);
        text.push_str("}\n");
    }
    Some(text)
}

/// `value` escaped as Python protobuf's `text_format` escapes a string (`CEscape`, not as UTF-8):
/// each byte of its UTF-8 that is not printable ASCII is an octal escape.
fn text_format_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
    for &byte in value.as_bytes() {
        match byte {
            b'\n' => escaped.push_str("\\n"),
            b'\r' => escaped.push_str("\\r"),
            b'\t' => escaped.push_str("\\t"),
            b'"' => escaped.push_str("\\\""),
            b'\'' => escaped.push_str("\\'"),
            b'\\' => escaped.push_str("\\\\"),
            0x20..=0x7e => escaped.push(char::from(byte)),
            _ => escaped.push_str(&format!("\\{byte:03o}")),
        }
    }
    escaped
}

/// The `google.rpc.ErrorInfo` of a gRPC status, from its `grpc-status-details-bin`: the
/// protobuf-encoded `google.rpc.Status` whose details hold Any-packed messages, each decoded
/// within the limit of the nesting of the responses (see `spark_connect_proto::ClientCodec`).
fn parse_error_info(details: &[u8]) -> Option<RpcErrorInfo> {
    use spark_connect_proto::decode_status_detail;
    if details.is_empty() {
        return None;
    }
    decode_status_detail::<RpcStatus>("google.rpc.Status", details)?
        .details
        .into_iter()
        .filter(|detail| detail.type_url == "type.googleapis.com/google.rpc.ErrorInfo")
        .find_map(|detail| decode_status_detail("google.rpc.ErrorInfo", &detail.value))
}

/// The names in a JSON object of message parameters, in their order. A JSON object that does
/// not parse has none.
fn parameter_names(json: &str) -> Vec<String> {
    struct Names(Vec<String>);
    impl<'de> serde::Deserialize<'de> for Names {
        fn deserialize<D: serde::Deserializer<'de>>(
            deserializer: D,
        ) -> std::result::Result<Self, D::Error> {
            struct NamesVisitor;
            impl<'de> serde::de::Visitor<'de> for NamesVisitor {
                type Value = Names;
                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("a JSON object")
                }
                fn visit_map<A: serde::de::MapAccess<'de>>(
                    self,
                    mut map: A,
                ) -> std::result::Result<Names, A::Error> {
                    let mut names = Vec::new();
                    while let Some((name, _)) = map.next_entry::<String, serde::de::IgnoredAny>()? {
                        names.push(name);
                    }
                    Ok(Names(names))
                }
            }
            deserializer.deserialize_map(NamesVisitor)
        }
    }
    serde_json::from_str::<Names>(json)
        .map(|names| names.0)
        .unwrap_or_default()
}

/// The kind of the exception that a JVM exception class maps to, as
/// `pyspark.errors.exceptions.connect.EXCEPTION_CLASS_MAPPING` and
/// `THIRD_PARTY_EXCEPTION_CLASS_MAPPING` map it, or `None` for a class that they do not map.
fn exception_class_kind(class: &str, message: &str) -> Option<SparkErrorKind> {
    use SparkErrorKind::*;
    Some(match class {
        "org.apache.spark.sql.catalyst.parser.ParseException" => Parse,
        "org.apache.spark.sql.AnalysisException" => Analysis,
        "org.apache.spark.sql.streaming.StreamingQueryException" => StreamingQuery,
        "org.apache.spark.sql.execution.QueryExecutionException" => QueryExecution,
        "java.lang.NumberFormatException" => NumberFormat,
        "java.lang.IllegalArgumentException" => IllegalArgument,
        "java.lang.ArithmeticException" => Arithmetic,
        "java.lang.UnsupportedOperationException" => UnsupportedOperation,
        "java.lang.ArrayIndexOutOfBoundsException" => ArrayIndexOutOfBounds,
        "java.time.DateTimeException" => DateTime,
        "org.apache.spark.SparkRuntimeException" => SparkRuntime,
        "org.apache.spark.SparkUpgradeException" => SparkUpgrade,
        "org.apache.spark.api.python.PythonException" => Python,
        "org.apache.spark.SparkNoSuchElementException" => SparkNoSuchElement,
        // A SparkException from a failure to pickle on the server.
        "org.apache.spark.SparkException" if message.contains("net.razorvine.pickle.PickleException") => {
            PickleException
        }
        "org.apache.spark.SparkException" => Spark,
        "org.apache.spark.sql.connect.common.InvalidPlanInput" => InvalidPlanInput,
        "org.apache.spark.api.python.StreamingPythonRunner$StreamingPythonRunnerInitializationException" => {
            StreamingPythonRunnerInitialization
        }
        _ => return None,
    })
}

/// The JVM stack trace of the error at `index` in `FetchErrorDetails`' errors, with its
/// causes. Mirrors `pyspark.errors.exceptions.connect._extract_jvm_stacktrace`.
fn jvm_stacktrace(
    index: usize,
    errors: &[spark_connect_proto::fetch_error_details_response::Error],
) -> String {
    let mut lines: Vec<String> = Vec::new();
    let mut next = Some(index);
    // A cause is another error of the list; stop at a cycle.
    let mut seen = std::collections::HashSet::new();
    while let Some(error) = next.filter(|i| seen.insert(*i)).and_then(|i| errors.get(i)) {
        let class = error
            .error_type_hierarchy
            .first()
            .map(String::as_str)
            .unwrap_or_default();
        if lines.is_empty() {
            lines.push(class.to_string());
        } else {
            lines.push(format!("Caused by: {}: {}", class, error.message));
        }
        for element in &error.stack_trace {
            lines.push(format!(
                "\tat {}.{}({}:{})",
                element.declaring_class,
                element.method_name,
                element.file_name.as_deref().unwrap_or_default(),
                element.line_number
            ));
        }
        next = error.cause_idx.and_then(|i| usize::try_from(i).ok());
    }
    lines.join("\n")
}

impl From<spark_connect_proto::fetch_error_details_response::QueryContext> for QueryContext {
    fn from(context: spark_connect_proto::fetch_error_details_response::QueryContext) -> Self {
        use spark_connect_proto::fetch_error_details_response::query_context::ContextType;
        let context_type = if context.context_type == ContextType::Dataframe as i32 {
            QueryContextType::DataFrame
        } else {
            QueryContextType::SQL
        };
        QueryContext::new(
            context_type,
            context.object_type,
            context.object_name,
            context.start_index,
            context.stop_index,
            context.fragment,
            context.call_site,
            context.summary,
        )
    }
}

impl From<spark_connect_proto::fetch_error_details_response::BreakingChangeInfo>
    for BreakingChangeInfo
{
    fn from(info: spark_connect_proto::fetch_error_details_response::BreakingChangeInfo) -> Self {
        BreakingChangeInfo {
            migration_message: info.migration_message,
            mitigation_config: info
                .mitigation_config
                .map(|config| (config.key, config.value)),
            // As in pyspark, a change that does not say needs an audit.
            needs_audit: info.needs_audit.unwrap_or(true),
        }
    }
}

// ============================================================================
// Error conditions registry parsing and message rendering
// ============================================================================

/// Render an error message by substituting parameters into the template.
///
/// Looks up the error class in the error-conditions registry, finds the message
/// template, and substitutes `<param_name>` placeholders with values from the
/// params map. Returns the rendered message without the error class prefix.
fn render_message(
    error_class: &str,
    params: &BTreeMap<String, String>,
) -> std::result::Result<String, String> {
    let template = get_message_template(error_class)?;

    // Find all placeholders in the template
    let mut result = template.clone();

    // Use regex to find and replace all <name> placeholders
    // Pattern: <([a-zA-Z0-9_-]+)>
    let pattern = regex::Regex::new(r"<([a-zA-Z0-9_\-]+)>").map_err(|e| e.to_string())?;

    for cap in pattern.captures_iter(&template) {
        if let Some(param_name) = cap.get(1) {
            let name = param_name.as_str();
            if let Some(value) = params.get(name) {
                let placeholder = format!("<{}>", name);
                result = result.replace(&placeholder, value);
            }
        }
    }

    Ok(result)
}

/// Get the message template for an error class from the registry.
fn get_message_template(error_class: &str) -> std::result::Result<String, String> {
    // Parse error_class which may be "MAIN_CLASS" or "MAIN_CLASS.SUB_CLASS"
    let parts: Vec<&str> = error_class.split('.').collect();

    let json_obj = parse_error_conditions_json()
        .map_err(|e| format!("Failed to parse error-conditions.json: {}", e))?;

    match parts.len() {
        1 => {
            let main_class = parts[0];
            if let Some(entry) = json_obj.get(main_class) {
                if let Some(msg_array) = entry.get("message") {
                    if let Some(msg_list) = msg_array.as_array() {
                        let message_parts: Vec<String> = msg_list
                            .iter()
                            .filter_map(|m| m.as_str().map(|s| s.to_string()))
                            .collect();
                        return Ok(message_parts.join("\n"));
                    }
                }
            }
            Err(format!("Error class not found: {}", main_class))
        }
        2 => {
            let main_class = parts[0];
            let sub_class = parts[1];
            if let Some(entry) = json_obj.get(main_class) {
                // Get main message
                let mut message = String::new();
                if let Some(msg_array) = entry.get("message") {
                    if let Some(msg_list) = msg_array.as_array() {
                        let message_parts: Vec<String> = msg_list
                            .iter()
                            .filter_map(|m| m.as_str().map(|s| s.to_string()))
                            .collect();
                        message = message_parts.join("\n");
                    }
                }

                // Get sub-class message if it exists
                if let Some(subclasses) = entry.get("sub_class") {
                    if let Some(sub) = subclasses.get(sub_class) {
                        if let Some(sub_msg_array) = sub.get("message") {
                            if let Some(sub_msg_list) = sub_msg_array.as_array() {
                                let sub_message_parts: Vec<String> = sub_msg_list
                                    .iter()
                                    .filter_map(|m| m.as_str().map(|s| s.to_string()))
                                    .collect();
                                if !message.is_empty() {
                                    message.push(' ');
                                }
                                message.push_str(&sub_message_parts.join("\n"));
                            }
                        }
                    }
                }

                if !message.is_empty() {
                    Ok(message)
                } else {
                    Err(format!("No message found for error class: {}", error_class))
                }
            } else {
                Err(format!("Main error class not found: {}", main_class))
            }
        }
        _ => Err(format!("Invalid error class format: {}", error_class)),
    }
}

/// Get the SQL state code for an error class from the registry.
fn get_sql_state(error_class: &str) -> Option<String> {
    let parts: Vec<&str> = error_class.split('.').collect();

    let json_obj = parse_error_conditions_json().ok()?;

    match parts.len() {
        1 => {
            let main_class = parts[0];
            json_obj
                .get(main_class)
                .and_then(|entry| entry.get("sqlState"))
                .and_then(|state| state.as_str())
                .map(|s| s.to_string())
        }
        2 => {
            let main_class = parts[0];
            let sub_class = parts[1];
            json_obj
                .get(main_class)
                .and_then(|entry| entry.get("sub_class"))
                .and_then(|subclasses| subclasses.get(sub_class))
                .and_then(|sub| sub.get("sqlState"))
                .and_then(|state| state.as_str())
                .map(|s| s.to_string())
        }
        _ => None,
    }
}

/// Parse the embedded error-conditions.json file.
fn parse_error_conditions_json(
) -> std::result::Result<serde_json::Map<String, serde_json::Value>, String> {
    let json: serde_json::Value = serde_json::from_str(ERROR_CONDITIONS_JSON)
        .map_err(|e| format!("Failed to parse JSON: {}", e))?;
    match json {
        serde_json::Value::Object(map) => Ok(map),
        _ => Err("Expected JSON object at root".to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_with_operation_id() {
        let err = SparkError::connect_msg("boom");
        assert_eq!(err.operation_id, None);
        let err = err.with_operation_id("op-123");
        assert_eq!(err.operation_id.as_deref(), Some("op-123"));
        let from_status = SparkError::from_grpc_status(Status::internal("x"));
        assert_eq!(from_status.operation_id, None);
    }

    #[test]
    fn a_builtin_assertion_has_no_message_and_keeps_its_operation_id() {
        // As the AssertionError of a bare `assert` of PySpark's client, which its `_handle_error`
        // raises with the operation ID of the execution.
        let err = SparkError::builtin_assertion().with_operation_id("op-1");
        assert_eq!(err.kind, SparkErrorKind::BuiltinAssertionError);
        assert_eq!(err.message(), "");
        assert_eq!(err.get_condition(), None);
        assert_eq!(err.operation_id.as_deref(), Some("op-1"));
    }

    #[test]
    fn observation_errors_are_their_root_errors_as_pyspark() {
        use spark_connect_proto::fetch_error_details_response::{Error, SparkThrowable};
        let error = |message: &str, class: &str| Error {
            error_type_hierarchy: vec![class.to_string(), "java.lang.Throwable".to_string()],
            message: message.to_string(),
            spark_throwable: Some(SparkThrowable {
                error_class: Some("USER_RAISED_EXCEPTION".to_string()),
                sql_state: Some("P0001".to_string()),
                ..Default::default()
            }),
            ..Default::default()
        };
        let errors = vec![
            error("cause", "java.lang.Exception"),
            error(
                "[USER_RAISED_EXCEPTION] test error",
                "org.apache.spark.SparkRuntimeException",
            ),
        ];
        let err = SparkError::from_observation_errors(1, errors.clone());
        assert_eq!(err.kind, SparkErrorKind::SparkRuntime);
        assert_eq!(
            err.get_condition().as_deref(),
            Some("USER_RAISED_EXCEPTION")
        );
        assert_eq!(err.get_sql_state().as_deref(), Some("P0001"));
        assert!(err.message().contains("test error"));
        // As PySpark's convert_observation_errors, of an index out of the errors.
        for index in [-1, 2] {
            let err = SparkError::from_observation_errors(index, errors.clone());
            assert_eq!(err.message(), "Observation error: invalid root_error_idx");
        }
    }

    #[test]
    fn test_render_simple_message() {
        let mut params = BTreeMap::new();
        params.insert("arg_name".to_string(), "x".to_string());

        let result = render_message("CANNOT_BE_NONE", &params).expect("should render");
        assert_eq!(result, "Argument `x` cannot be None.");
    }

    #[test]
    fn test_error_message_format() {
        let mut params = BTreeMap::new();
        params.insert("arg_name".to_string(), "x".to_string());

        let err = SparkError::classed(
            SparkErrorKind::ValueError,
            "CANNOT_BE_NONE",
            &[("arg_name", "x")],
        );

        let msg = err.message();
        assert_eq!(msg, "[CANNOT_BE_NONE] Argument `x` cannot be None.");
    }

    #[test]
    fn test_error_display() {
        let err = SparkError::classed(
            SparkErrorKind::ValueError,
            "CANNOT_BE_NONE",
            &[("arg_name", "test_arg")],
        );

        let display = format!("{}", err);
        assert_eq!(
            display,
            "[CANNOT_BE_NONE] Argument `test_arg` cannot be None."
        );
    }

    #[test]
    fn test_connect_msg() {
        let err = SparkError::connect_msg("Connection failed");
        let msg = err.message();
        assert_eq!(msg, "Connection failed");
    }

    #[test]
    fn test_get_sql_state_known() {
        let sql_state = get_sql_state("ATTRIBUTE_NOT_SUPPORTED");
        assert_eq!(sql_state, Some("0A000".to_string()));
    }

    #[test]
    fn test_get_sql_state_unknown() {
        let sql_state = get_sql_state("UNKNOWN_ERROR_CLASS_XXXXX");
        assert_eq!(sql_state, None);
    }

    #[test]
    fn test_multiple_params() {
        let result = render_message(
            "INVALID_CONNECT_URL",
            &[(
                "detail".to_string(),
                "The URL must start with 'sc://'".to_string(),
            )]
            .iter()
            .cloned()
            .collect(),
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_full_error_message_with_params() {
        // Test: CANNOT_BE_NONE
        let err = SparkError::classed(
            SparkErrorKind::ValueError,
            "CANNOT_BE_NONE",
            &[("arg_name", "my_arg")],
        );
        assert_eq!(
            err.message(),
            "[CANNOT_BE_NONE] Argument `my_arg` cannot be None."
        );

        // Test: ARGUMENT_REQUIRED
        let err = SparkError::classed(
            SparkErrorKind::ValueError,
            "ARGUMENT_REQUIRED",
            &[("arg_name", "foo"), ("condition", "x > 0")],
        );
        assert_eq!(
            err.message(),
            "[ARGUMENT_REQUIRED] Argument `foo` is required when x > 0."
        );

        // Test: INVALID_CONNECT_URL
        let err = SparkError::classed(
            SparkErrorKind::RuntimeError,
            "INVALID_CONNECT_URL",
            &[("detail", "The URL must start with sc://")],
        );
        assert_eq!(
            err.message(),
            "[INVALID_CONNECT_URL] Invalid URL for Spark Connect: The URL must start with sc://"
        );

        // Test: ATTRIBUTE_NOT_CALLABLE
        let err = SparkError::classed(
            SparkErrorKind::ValueError,
            "ATTRIBUTE_NOT_CALLABLE",
            &[("attr_name", "compute"), ("obj_name", "MyClass")],
        );
        assert_eq!(
            err.message(),
            "[ATTRIBUTE_NOT_CALLABLE] Attribute `compute` in provided object `MyClass` is not callable."
        );
    }

    #[test]
    fn test_error_without_error_class() {
        let mut err = SparkError::connect_msg("Custom error message");
        assert_eq!(err.message(), "Custom error message");

        // Also test when message field is set but error_class is empty
        err.message = "Another message".to_string();
        assert_eq!(err.message(), "Another message");
    }

    #[test]
    fn test_error_kind_variants() {
        // Ensure all error kinds are defined and accessible
        let _ = SparkErrorKind::ValueError;
        let _ = SparkErrorKind::TypeError;
        let _ = SparkErrorKind::IndexError;
        let _ = SparkErrorKind::AttributeError;
        let _ = SparkErrorKind::KeyError;
        let _ = SparkErrorKind::RuntimeError;
        let _ = SparkErrorKind::NotImplementedError;
        let _ = SparkErrorKind::AssertionError;
        let _ = SparkErrorKind::PicklingError;
        let _ = SparkErrorKind::ImportError;
        let _ = SparkErrorKind::Connect;
        let _ = SparkErrorKind::ConnectMessage;
        let _ = SparkErrorKind::ConnectGrpc;
        let _ = SparkErrorKind::Analysis;
        let _ = SparkErrorKind::Parse;
        let _ = SparkErrorKind::Python;
        let _ = SparkErrorKind::Unknown;
    }

    #[test]
    fn test_message_rendering_parity_with_python() {
        // These test cases are verified to match PySpark's str(e) output exactly.
        // Run these in Python to verify:
        //   from pyspark.errors import PySparkValueError, PySparkRuntimeError
        //   e = PySparkValueError(errorClass='CANNOT_BE_NONE', messageParameters={'arg_name':'x'})
        //   print(str(e))  # Should be: [CANNOT_BE_NONE] Argument `x` cannot be None.

        // Test 1: CANNOT_BE_NONE
        let err = SparkError::classed(
            SparkErrorKind::ValueError,
            "CANNOT_BE_NONE",
            &[("arg_name", "x")],
        );
        assert_eq!(
            err.message(),
            "[CANNOT_BE_NONE] Argument `x` cannot be None."
        );

        // Test 2: ARGUMENT_REQUIRED
        let err = SparkError::classed(
            SparkErrorKind::ValueError,
            "ARGUMENT_REQUIRED",
            &[("arg_name", "foo"), ("condition", "x > 0")],
        );
        assert_eq!(
            err.message(),
            "[ARGUMENT_REQUIRED] Argument `foo` is required when x > 0."
        );

        // Test 3: INVALID_CONNECT_URL
        let err = SparkError::classed(
            SparkErrorKind::RuntimeError,
            "INVALID_CONNECT_URL",
            &[("detail", "test")],
        );
        assert_eq!(
            err.message(),
            "[INVALID_CONNECT_URL] Invalid URL for Spark Connect: test"
        );

        // Test 4: ATTRIBUTE_NOT_CALLABLE
        let err = SparkError::classed(
            SparkErrorKind::ValueError,
            "ATTRIBUTE_NOT_CALLABLE",
            &[("attr_name", "compute"), ("obj_name", "MyClass")],
        );
        assert_eq!(
            err.message(),
            "[ATTRIBUTE_NOT_CALLABLE] Attribute `compute` in provided object `MyClass` is not callable."
        );
    }

    /// A gRPC status of an error that the server raised, with an `ErrorInfo` of `metadata`.
    fn server_status(message: &str, metadata: &[(&str, &str)]) -> Status {
        let error_info = RpcErrorInfo {
            reason: "org.apache.spark.sql.catalyst.parser.ParseException".to_string(),
            domain: "org.apache.spark".to_string(),
            metadata: metadata
                .iter()
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        };
        let status = RpcStatus {
            code: tonic::Code::Internal as i32,
            message: message.to_string(),
            details: vec![RpcAny {
                type_url: "type.googleapis.com/google.rpc.ErrorInfo".to_string(),
                value: error_info.encode_to_vec(),
            }],
        };
        Status::with_details(
            tonic::Code::Internal,
            message,
            bytes::Bytes::from(status.encode_to_vec()),
        )
    }

    fn classes(classes: &[&str]) -> String {
        serde_json::to_string(classes).unwrap()
    }

    #[test]
    fn a_server_error_reads_its_error_info() {
        let message = "\n[PARSE_SYNTAX_ERROR] Syntax error at or near 'SELEC'. SQLSTATE: 42601";
        let err = SparkError::from_grpc_status(server_status(
            message,
            &[
                (
                    "classes",
                    &classes(&[
                        "org.apache.spark.sql.catalyst.parser.ParseException",
                        "org.apache.spark.sql.AnalysisException",
                    ]),
                ),
                ("errorClass", "PARSE_SYNTAX_ERROR"),
                ("sqlState", "42601"),
                ("messageParameters", r#"{"error":"'SELEC'","hint":""}"#),
                ("errorId", "error-1"),
            ],
        ));
        // A ParseException is also an AnalysisException; the most specific class decides.
        assert_eq!(err.kind, SparkErrorKind::Parse);
        assert_eq!(err.get_condition().as_deref(), Some("PARSE_SYNTAX_ERROR"));
        assert_eq!(err.get_sql_state().as_deref(), Some("42601"));
        // The parameters are the message's, not the rest of the metadata.
        let params: Vec<_> = err.params.iter().collect();
        assert_eq!(
            params,
            [
                (&"error".to_string(), &"'SELEC'".to_string()),
                (&"hint".to_string(), &String::new())
            ]
        );
        // The server's message already names the condition.
        assert_eq!(err.message(), message);
        assert_eq!(err.to_string(), message);
        let server = err.server.as_ref().unwrap();
        assert_eq!(
            server.reason,
            "org.apache.spark.sql.catalyst.parser.ParseException"
        );
        assert_eq!(server.error_id.as_deref(), Some("error-1"));
        assert!(!server.display_stacktrace);
        assert_eq!(err.grpc_code(), Some(tonic::Code::Internal as i32));
        assert!(err.grpc_details().is_some());
    }

    #[test]
    fn a_server_error_keeps_the_order_of_its_parameters() {
        let err = SparkError::from_grpc_status(server_status(
            "[INVALID_CONF_VALUE.TYPE_MISMATCH] ...",
            &[(
                "messageParameters",
                r#"{"confName":"spark.sql.shuffle.partitions","confValue":"abc","confType":"int"}"#,
            )],
        ));
        assert_eq!(
            err.server.unwrap().parameter_order,
            ["confName", "confValue", "confType"]
        );
        assert!(parameter_names("not json").is_empty());
        assert!(parameter_names("[1, 2]").is_empty());
    }

    #[test]
    fn the_first_class_that_maps_decides_the_kind_of_a_server_error() {
        let kind = |message: &str, hierarchy: &[&str]| {
            SparkError::from_grpc_status(server_status(
                message,
                &[("classes", &classes(hierarchy))],
            ))
            .kind
        };
        assert_eq!(
            kind(
                "x",
                &[
                    "org.apache.spark.sql.catalyst.ExtendedAnalysisException",
                    "org.apache.spark.sql.AnalysisException",
                    "java.lang.Exception",
                ]
            ),
            SparkErrorKind::Analysis
        );
        assert_eq!(
            kind(
                "x",
                &[
                    "org.apache.spark.SparkNumberFormatException",
                    "java.lang.NumberFormatException",
                    "java.lang.IllegalArgumentException",
                ]
            ),
            SparkErrorKind::NumberFormat
        );
        assert_eq!(
            kind(
                "x",
                &[
                    "org.apache.spark.SparkArithmeticException",
                    "java.lang.ArithmeticException"
                ]
            ),
            SparkErrorKind::Arithmetic
        );
        // A PythonException anywhere in the hierarchy decides it.
        assert_eq!(
            kind(
                "x",
                &[
                    "org.apache.spark.SparkException",
                    "org.apache.spark.api.python.PythonException"
                ]
            ),
            SparkErrorKind::Python
        );
        assert_eq!(
            kind("x", &["org.apache.spark.SparkException"]),
            SparkErrorKind::Spark
        );
        assert_eq!(
            kind(
                "net.razorvine.pickle.PickleException: expected zero arguments",
                &["org.apache.spark.SparkException"]
            ),
            SparkErrorKind::PickleException
        );
        assert_eq!(
            kind(
                "x",
                &["org.apache.spark.sql.connect.common.InvalidPlanInput"]
            ),
            SparkErrorKind::InvalidPlanInput
        );
        assert_eq!(
            kind("x", &["java.lang.IllegalStateException"]),
            SparkErrorKind::Unknown
        );
        assert_eq!(kind("x", &[]), SparkErrorKind::Unknown);
    }

    #[test]
    fn details_complete_a_server_error() {
        use spark_connect_proto::fetch_error_details_response::{
            query_context::ContextType, BreakingChangeInfo as ProtoChange, Error, MitigationConfig,
            QueryContext as ProtoContext, SparkThrowable, StackTraceElement,
        };
        use spark_connect_proto::FetchErrorDetailsResponse;
        let element = |method: &str, file: Option<&str>, line| StackTraceElement {
            declaring_class: "org.apache.spark.Expr".to_string(),
            method_name: method.to_string(),
            file_name: file.map(str::to_string),
            line_number: line,
        };
        let details = FetchErrorDetailsResponse {
            root_error_idx: Some(1),
            errors: vec![
                Error {
                    error_type_hierarchy: vec!["java.lang.RuntimeException".to_string()],
                    message: "the cause".to_string(),
                    stack_trace: vec![element("cause", Some("Cause.scala"), 3)],
                    ..Default::default()
                },
                Error {
                    error_type_hierarchy: vec![
                        "org.apache.spark.SparkArithmeticException".to_string(),
                        "java.lang.ArithmeticException".to_string(),
                    ],
                    message: "[DIVIDE_BY_ZERO] Division by zero.".to_string(),
                    stack_trace: vec![
                        element("eval", Some("Expr.scala"), 12),
                        element("native", None, -2),
                    ],
                    cause_idx: Some(0),
                    spark_throwable: Some(SparkThrowable {
                        error_class: Some("DIVIDE_BY_ZERO".to_string()),
                        message_parameters: [("config".to_string(), "ansi".to_string())].into(),
                        query_contexts: vec![ProtoContext {
                            context_type: ContextType::Sql as i32,
                            fragment: "1/0".to_string(),
                            start_index: 7,
                            stop_index: 9,
                            ..Default::default()
                        }],
                        sql_state: Some("22012".to_string()),
                        breaking_change_info: Some(ProtoChange {
                            migration_message: vec!["Set the config.".to_string()],
                            mitigation_config: Some(MitigationConfig {
                                key: "spark.sql.ansi.enabled".to_string(),
                                value: "false".to_string(),
                            }),
                            needs_audit: None,
                        }),
                    }),
                },
            ],
            ..Default::default()
        };
        // The ErrorInfo has neither a condition nor a stack trace; the details have both.
        let err = SparkError::from_grpc_status(server_status(
            "[DIVIDE_BY_ZERO] Division by ze...",
            &[
                (
                    "classes",
                    &classes(&[
                        "org.apache.spark.SparkArithmeticException",
                        "java.lang.ArithmeticException",
                    ]),
                ),
                ("errorId", "error-2"),
            ],
        ))
        .with_details(Some(details), true);
        assert_eq!(err.kind, SparkErrorKind::Arithmetic);
        assert_eq!(err.message(), "[DIVIDE_BY_ZERO] Division by zero.");
        assert_eq!(err.get_condition().as_deref(), Some("DIVIDE_BY_ZERO"));
        assert_eq!(err.params.get("config").map(String::as_str), Some("ansi"));
        assert_eq!(
            err.get_stacktrace().as_deref(),
            Some(
                "org.apache.spark.SparkArithmeticException\n\
                 \tat org.apache.spark.Expr.eval(Expr.scala:12)\n\
                 \tat org.apache.spark.Expr.native(:-2)\n\
                 Caused by: java.lang.RuntimeException: the cause\n\
                 \tat org.apache.spark.Expr.cause(Cause.scala:3)"
            )
        );
        let contexts = err.get_query_context();
        assert_eq!(contexts.len(), 1);
        assert_eq!(contexts[0].context_type(), QueryContextType::SQL);
        assert_eq!(contexts[0].fragment(), "1/0");
        assert_eq!(
            (contexts[0].start_index(), contexts[0].stop_index()),
            (7, 9)
        );
        let server = err.server.as_ref().unwrap();
        assert!(server.display_stacktrace);
        assert_eq!(
            server.breaking_change,
            Some(BreakingChangeInfo {
                migration_message: vec!["Set the config.".to_string()],
                mitigation_config: Some((
                    "spark.sql.ansi.enabled".to_string(),
                    "false".to_string()
                )),
                needs_audit: true,
            })
        );
    }

    #[test]
    fn without_details_a_server_error_shows_a_stack_trace_only_if_it_has_one() {
        let status = |metadata: &[(&str, &str)]| {
            SparkError::from_grpc_status(server_status("[X] message", metadata))
        };
        let with_trace = status(&[("stackTrace", "org.apache.spark.X\n\tat ...")]);
        let shown = with_trace.clone().with_details(None, true);
        assert!(shown.server.as_ref().unwrap().display_stacktrace);
        assert_eq!(shown.message(), "[X] message");
        assert!(
            !with_trace
                .with_details(None, false)
                .server
                .unwrap()
                .display_stacktrace
        );
        assert!(
            !status(&[])
                .with_details(None, true)
                .server
                .unwrap()
                .display_stacktrace
        );
        // A root error index out of range is no details.
        let out_of_range = spark_connect_proto::FetchErrorDetailsResponse {
            root_error_idx: Some(3),
            ..Default::default()
        };
        let err = status(&[]).with_details(Some(out_of_range), true);
        assert_eq!(err.message(), "[X] message");
        assert_eq!(err.get_stacktrace(), None);
    }

    #[test]
    fn a_cause_cycle_ends_the_stack_trace() {
        use spark_connect_proto::fetch_error_details_response::Error;
        let error = |cause| Error {
            error_type_hierarchy: vec!["java.lang.Exception".to_string()],
            message: "loop".to_string(),
            cause_idx: Some(cause),
            ..Default::default()
        };
        assert_eq!(
            jvm_stacktrace(0, &[error(1), error(0)]),
            "java.lang.Exception\nCaused by: java.lang.Exception: loop"
        );
    }

    #[test]
    fn a_status_without_an_error_info_is_not_a_server_error() {
        let err = SparkError::from_grpc_status(Status::unavailable("connection refused"));
        assert!(err.server.is_none());
        assert_eq!(err.kind, SparkErrorKind::Connect);
        assert_eq!(err.grpc_code(), Some(tonic::Code::Unavailable as i32));
        let message = err.message();
        assert_eq!(
            message,
            "[The service is currently unavailable] connection refused"
        );
        // Details are only for a server error.
        let unchanged = err.with_details(None, true);
        assert!(unchanged.server.is_none());
        assert_eq!(unchanged.message(), message);
    }

    #[test]
    fn a_status_without_an_error_info_is_the_text_of_the_error_of_grpcio_as_pyspark() {
        // PySpark's message is that of `_handle_rpc_error`, of the `str()` of grpcio's RpcError:
        // of the class of the kind of the call, and of the `debug_error_string` of the code and
        // the details, which is the code alone of no details, as grpcio 1.85.1 gives them for a
        // status of the server. The message of the Rust API is the code and the details.
        let message = |status: Status, call| {
            let err = SparkError::from_grpc_status_of(status.clone(), call);
            assert_eq!(
                err.message(),
                format!("[{}] {}", status.code(), status.message())
            );
            err.pyspark_message.unwrap()
        };
        assert_eq!(
            message(
                Status::permission_denied(r#"denied by "the" proxy"#),
                GrpcCall::Unary
            ),
            "<_InactiveRpcError of RPC that terminated with:\n\
             \tstatus = StatusCode.PERMISSION_DENIED\n\
             \tdetails = \"denied by \"the\" proxy\"\n\
             \tdebug_error_string = \"PERMISSION_DENIED:denied by \"the\" proxy\"\n>"
        );
        assert_eq!(
            message(Status::internal(""), GrpcCall::ServerStreaming),
            "<_MultiThreadedRendezvous of RPC that terminated with:\n\
             \tstatus = StatusCode.INTERNAL\n\tdetails = \"\"\n\
             \tdebug_error_string = \"INTERNAL\"\n>"
        );
        assert_eq!(
            message(Status::unknown("line one\nline two é"), GrpcCall::Unary),
            "<_InactiveRpcError of RPC that terminated with:\n\
             \tstatus = StatusCode.UNKNOWN\n\tdetails = \"line one\nline two é\"\n\
             \tdebug_error_string = \"UNKNOWN:line one\nline two é\"\n>"
        );
        // Of a response that the client fails to decode (the status of prost's error that the
        // codec of the client tags), grpcio's error of a response that it fails to deserialize,
        // as PySpark raises it.
        let undecodable = "failed to decode Protobuf message: ExecutePlanResponse.session_id: \
                           buffer underflow";
        let tagged = |mut status: Status| {
            status.set_source(std::sync::Arc::new(spark_connect_proto::ClientDecodeError));
            status
        };
        for (call, class) in [
            (GrpcCall::ServerStreaming, "_MultiThreadedRendezvous"),
            (GrpcCall::Unary, "_InactiveRpcError"),
        ] {
            assert_eq!(
                message(tagged(Status::internal(undecodable)), call),
                format!(
                    "<{class} of RPC that terminated with:\n\tstatus = StatusCode.INTERNAL\n\
                     \tdetails = \"Exception deserializing response!\"\n\
                     \tdebug_error_string = \"None\"\n>"
                )
            );
        }
        // Not of a status of the server of the same text, as a server that fails to decode a
        // request sends it, nor of another code, nor of a `google.rpc.Status`.
        assert_eq!(
            message(Status::internal(undecodable), GrpcCall::Unary),
            format!(
                "<_InactiveRpcError of RPC that terminated with:\n\
                 \tstatus = StatusCode.INTERNAL\n\tdetails = \"{undecodable}\"\n\
                 \tdebug_error_string = \"INTERNAL:{undecodable}\"\n>"
            )
        );
        assert!(
            message(tagged(Status::unknown(undecodable)), GrpcCall::Unary).contains(undecodable)
        );
        let of_server = Status::with_details(
            tonic::Code::Internal,
            undecodable,
            bytes::Bytes::from_static(&[8, 13]),
        );
        assert_eq!(message(of_server, GrpcCall::Unary), undecodable);
        // Of a `google.rpc.Status` without an `ErrorInfo`, here of the code 14 alone, the message
        // of the status, which `rpc_status.from_call` checks is that of the call.
        let without_info = Status::with_details(
            tonic::Code::Unavailable,
            "retry later",
            bytes::Bytes::from_static(&[8, 14]),
        );
        assert_eq!(
            message(without_info, GrpcCall::ServerStreaming),
            "retry later"
        );
    }

    #[test]
    fn the_error_info_of_a_status_is_the_text_of_upb_as_pyspark_logs_it() {
        // As `str()` of the ErrorInfo of Python's protobuf (upb), which PySpark's client logs: the
        // fields that are set, the entries of the map sorted by upb's comparison of their keys,
        // descending, a key before the longer ones that it begins, of the last value of a key,
        // and the octal escapes of the bytes that are not printable ASCII.
        let mut metadata = vec![
            ("x1", "one"),
            ("x10", "ten"),
            ("x2", "two"),
            ("classes", "[\"a\"]"),
            ("x1", "last"),
            ("quote", "it's \\ é\n"),
        ];
        let entries = metadata
            .drain(..)
            .map(|(key, value)| MetadataEntry {
                key: key.to_string(),
                value: value.to_string(),
            })
            .collect();
        let info = RpcErrorInfoEntries {
            reason: "org.apache.spark.SparkException".to_string(),
            domain: String::new(),
            metadata: entries,
        };
        let status = RpcStatus {
            code: 13,
            message: "boom".to_string(),
            details: vec![RpcAny {
                type_url: "type.googleapis.com/google.rpc.ErrorInfo".to_string(),
                value: info.encode_to_vec(),
            }],
        };
        assert_eq!(
            error_info_text(&status.encode_to_vec()).unwrap(),
            "reason: \"org.apache.spark.SparkException\"\n\
             metadata {\n  key: \"x2\"\n  value: \"two\"\n}\n\
             metadata {\n  key: \"x1\"\n  value: \"last\"\n}\n\
             metadata {\n  key: \"x10\"\n  value: \"ten\"\n}\n\
             metadata {\n  key: \"quote\"\n  value: \"it\\'s \\\\ \\303\\251\\n\"\n}\n\
             metadata {\n  key: \"classes\"\n  value: \"[\\\"a\\\"]\"\n}\n"
        );
        assert_eq!(error_info_text(&[]), None);
    }

    #[test]
    fn grpcio_s_error_of_a_response_that_does_not_decode_has_no_debug_error_string() {
        // As grpcio's error of a response that it fails to deserialize, which a policy, the logs
        // and the calls left to grpc see: its details, of no `debug_error_string`, whose text
        // reads `None`, of a unary call and of a stream alike.
        let mut undecodable = Status::internal("failed to decode Protobuf message: invalid varint");
        undecodable.set_source(std::sync::Arc::new(spark_connect_proto::ClientDecodeError));
        for (call, class) in [
            (GrpcCall::Unary, "_InactiveRpcError"),
            (GrpcCall::ServerStreaming, "_MultiThreadedRendezvous"),
        ] {
            let error = grpcio_error(&undecodable, call);
            assert_eq!(error.code, tonic::Code::Internal);
            assert_eq!(error.details, "Exception deserializing response!");
            assert_eq!(error.debug_error_string, None);
            let text = format!(
                "<{class} of RPC that terminated with:\n\tstatus = StatusCode.INTERNAL\n\
                 \tdetails = \"Exception deserializing response!\"\n\
                 \tdebug_error_string = \"None\"\n>"
            );
            assert_eq!(error.text, text);
            assert_eq!(grpcio_error_text(&undecodable, call), text);
        }
        // A status of the server of the same text is the server's, of its debug string.
        let of_server = Status::internal("failed to decode Protobuf message: invalid varint");
        let error = grpcio_error(&of_server, GrpcCall::Unary);
        assert_eq!(error.details, of_server.message());
        assert_eq!(
            error.debug_error_string.as_deref(),
            Some("INTERNAL:failed to decode Protobuf message: invalid varint")
        );
    }

    #[test]
    fn grpcio_s_error_of_a_status_keeps_the_metadata_that_grpcio_gives() {
        // As grpcio's `RpcError`, whose trailing metadata does not hold the headers that gRPC's
        // core reads itself, and holds the `google.rpc.Status` of the details.
        let mut metadata = tonic::metadata::MetadataMap::new();
        metadata.insert("content-type", "application/grpc".parse().unwrap());
        metadata.insert("x-mine", "kept".parse().unwrap());
        let status = Status::with_details_and_metadata(
            tonic::Code::Internal,
            "boom",
            bytes::Bytes::from_static(b"details"),
            metadata,
        );
        let error = grpcio_error(&status, GrpcCall::ServerStreaming);
        assert_eq!(error.code, tonic::Code::Internal);
        assert_eq!(
            (error.details.as_str(), error.debug_error_string.as_deref()),
            ("boom", Some("INTERNAL:boom"))
        );
        let keys: Vec<&str> = error
            .trailing_metadata
            .iter()
            .map(|(key, _)| key.as_str())
            .collect();
        assert_eq!(keys, ["x-mine", "grpc-status-details-bin"]);
        assert!(error
            .text
            .starts_with("<_MultiThreadedRendezvous of RPC that terminated with:"));
    }

    #[test]
    fn test_accessor_methods_get_condition() {
        // Test with error class
        let err = SparkError::classed(
            SparkErrorKind::ValueError,
            "CANNOT_BE_NONE",
            &[("arg_name", "test")],
        );
        assert_eq!(err.get_condition(), Some("CANNOT_BE_NONE".to_string()));
        assert_eq!(err.get_error_class(), Some("CANNOT_BE_NONE".to_string()));

        // Test without error class
        let err = SparkError::connect_msg("Plain message");
        assert_eq!(err.get_condition(), None);
        assert_eq!(err.get_error_class(), None);
    }

    #[test]
    fn test_accessor_methods_get_message_parameters() {
        // Test with parameters
        let err = SparkError::classed(
            SparkErrorKind::ValueError,
            "CANNOT_BE_NONE",
            &[("arg_name", "x"), ("other", "y")],
        );
        let params = err.get_message_parameters();
        assert!(params.is_some());
        let params = params.unwrap();
        assert_eq!(params.get("arg_name"), Some(&"x".to_string()));
        assert_eq!(params.get("other"), Some(&"y".to_string()));

        // Test without parameters
        let err = SparkError::connect_msg("Plain message");
        assert_eq!(err.get_message_parameters(), None);
    }

    #[test]
    fn test_accessor_methods_get_sql_state() {
        // Test with known SQL state
        let err = SparkError::classed(
            SparkErrorKind::ValueError,
            "ATTRIBUTE_NOT_SUPPORTED",
            &[("attr_name", "test")],
        );
        let sql_state = err.get_sql_state();
        assert_eq!(sql_state, Some("0A000".to_string()));

        // Test without SQL state
        let err = SparkError::classed(
            SparkErrorKind::ValueError,
            "CANNOT_BE_NONE",
            &[("arg_name", "x")],
        );
        let sql_state = err.get_sql_state();
        assert_eq!(sql_state, None);
    }

    #[test]
    fn test_accessor_methods_get_message() {
        let err = SparkError::classed(
            SparkErrorKind::ValueError,
            "CANNOT_BE_NONE",
            &[("arg_name", "x")],
        );
        assert_eq!(
            err.get_message(),
            "[CANNOT_BE_NONE] Argument `x` cannot be None."
        );

        // Test without error class
        let err = SparkError::connect_msg("Custom message");
        assert_eq!(err.get_message(), "Custom message");
    }

    #[test]
    fn test_accessor_methods_get_query_context() {
        let mut err = SparkError::connect_msg("Test error");
        assert!(err.get_query_context().is_empty());

        // Add a query context
        let ctx = QueryContext::new(
            QueryContextType::SQL,
            "VIEW".to_string(),
            "my_view".to_string(),
            0,
            10,
            "SELECT * FROM".to_string(),
            "file.py:10".to_string(),
            "In DataFrame operation".to_string(),
        );
        err.contexts.push(ctx);
        let contexts = err.get_query_context();
        assert_eq!(contexts.len(), 1);
        assert_eq!(contexts[0].context_type(), QueryContextType::SQL);
        assert_eq!(contexts[0].object_type(), "VIEW");
        assert_eq!(contexts[0].object_name(), "my_view");
    }

    #[test]
    fn test_query_context_creation() {
        let ctx = QueryContext::new(
            QueryContextType::DataFrame,
            "".to_string(),
            "".to_string(),
            5,
            15,
            "df.select(col('x'))".to_string(),
            "test.py:42".to_string(),
            "Selecting column x".to_string(),
        );
        assert_eq!(ctx.context_type(), QueryContextType::DataFrame);
        assert_eq!(ctx.start_index(), 5);
        assert_eq!(ctx.stop_index(), 15);
        assert_eq!(ctx.fragment(), "df.select(col('x'))");
        assert_eq!(ctx.call_site(), "test.py:42");
        assert_eq!(ctx.summary(), "Selecting column x");
    }

    #[test]
    fn test_stacktrace_accessor() {
        let mut err = SparkError::connect_msg("Error with stack trace");
        assert_eq!(err.get_stacktrace(), None);

        err.server_stacktrace = Some("at org.apache.spark.sql...".to_string());
        assert_eq!(
            err.get_stacktrace(),
            Some("at org.apache.spark.sql...".to_string())
        );
    }

    #[test]
    fn test_parity_cannot_be_none() {
        // Verify that message rendering matches PySpark exactly
        // Expected in PySpark: [CANNOT_BE_NONE] Argument `x` cannot be None.
        let err = SparkError::classed(
            SparkErrorKind::ValueError,
            "CANNOT_BE_NONE",
            &[("arg_name", "x")],
        );
        assert_eq!(
            err.message(),
            "[CANNOT_BE_NONE] Argument `x` cannot be None."
        );
        assert_eq!(err.get_condition(), Some("CANNOT_BE_NONE".to_string()));
        let params = err.get_message_parameters().unwrap();
        assert_eq!(params.get("arg_name"), Some(&"x".to_string()));
    }

    #[test]
    fn test_parity_argument_required() {
        // Expected in PySpark: [ARGUMENT_REQUIRED] Argument `foo` is required when x > 0.
        let err = SparkError::classed(
            SparkErrorKind::ValueError,
            "ARGUMENT_REQUIRED",
            &[("arg_name", "foo"), ("condition", "x > 0")],
        );
        assert_eq!(
            err.message(),
            "[ARGUMENT_REQUIRED] Argument `foo` is required when x > 0."
        );
    }

    #[test]
    fn test_parity_invalid_connect_url() {
        // Expected in PySpark: [INVALID_CONNECT_URL] Invalid URL for Spark Connect: <detail>
        let err = SparkError::classed(
            SparkErrorKind::RuntimeError,
            "INVALID_CONNECT_URL",
            &[("detail", "must start with sc://")],
        );
        assert_eq!(
            err.message(),
            "[INVALID_CONNECT_URL] Invalid URL for Spark Connect: must start with sc://"
        );
    }

    #[test]
    fn test_parity_attribute_not_callable() {
        // Expected in PySpark: [ATTRIBUTE_NOT_CALLABLE] Attribute `compute` in provided object `MyClass` is not callable.
        let err = SparkError::classed(
            SparkErrorKind::ValueError,
            "ATTRIBUTE_NOT_CALLABLE",
            &[("attr_name", "compute"), ("obj_name", "MyClass")],
        );
        assert_eq!(
            err.message(),
            "[ATTRIBUTE_NOT_CALLABLE] Attribute `compute` in provided object `MyClass` is not callable."
        );
    }

    #[test]
    fn test_analysis_exception_mapping() {
        let err = SparkError::classed(
            SparkErrorKind::Analysis,
            "ANALYSIS_ERROR",
            &[("message", "Table not found")],
        );
        assert_eq!(err.kind, SparkErrorKind::Analysis);
        assert_eq!(err.get_condition(), Some("ANALYSIS_ERROR".to_string()));
    }

    #[test]
    fn deadline_exceeded_is_raised_as_pyspark_does() {
        // As `_handle_rpc_error`: a call past its per-RPC deadline, whatever the details of
        // its status, is the SparkConnectGrpcException of the deadline.
        let status = Status::with_details(
            tonic::Code::DeadlineExceeded,
            "Deadline Exceeded",
            bytes::Bytes::from_static(b"not an error info"),
        );
        let err = SparkError::from_grpc_status(status);
        assert_eq!(err.kind, SparkErrorKind::Connect);
        assert_eq!(err.grpc_code, Some(tonic::Code::DeadlineExceeded as i32));
        assert!(err.server.is_none());
        assert!(
            err.message.starts_with(
                "[Deadline expired before operation could complete] Deadline Exceeded: RPC \
                 deadline exceeded. The client applies per-RPC timeouts"
            ),
            "{}",
            err.message
        );
        assert!(err
            .message
            .ends_with("SparkConnectClient(url, rpc_deadlines=RpcDeadlines.disabled())."));
        // PySpark's is of the text of grpcio's error of the call, `f"{error}: RPC deadline
        // exceeded..."`, as grpcio 1.85.1 gives it for a call past its deadline.
        let pyspark_message = err.pyspark_message.unwrap();
        assert_eq!(
            pyspark_message,
            err.message.replacen(
                "[Deadline expired before operation could complete] Deadline Exceeded",
                "<_InactiveRpcError of RPC that terminated with:\n\
                 \tstatus = StatusCode.DEADLINE_EXCEEDED\n\tdetails = \"Deadline Exceeded\"\n\
                 \tdebug_error_string = \"DEADLINE_EXCEEDED:Deadline Exceeded\"\n>",
                1
            )
        );
    }

    #[test]
    fn test_illegal_argument_exception_mapping() {
        let err = SparkError::classed(SparkErrorKind::IllegalArgument, "ILLEGAL_ARGUMENT", &[]);
        assert_eq!(err.kind, SparkErrorKind::IllegalArgument);
        assert_eq!(err.get_condition(), Some("ILLEGAL_ARGUMENT".to_string()));
    }

    #[test]
    fn test_arithmetic_exception_mapping() {
        let err = SparkError::classed(
            SparkErrorKind::Arithmetic,
            "ARITHMETIC_ERROR",
            &[("message", "Division by zero")],
        );
        assert_eq!(err.kind, SparkErrorKind::Arithmetic);
    }

    #[test]
    fn test_unsupported_operation_exception_mapping() {
        let err = SparkError::classed(
            SparkErrorKind::UnsupportedOperation,
            "UNSUPPORTED_OPERATION",
            &[],
        );
        assert_eq!(err.kind, SparkErrorKind::UnsupportedOperation);
    }

    #[test]
    fn test_query_execution_exception_mapping() {
        let err = SparkError::classed(
            SparkErrorKind::QueryExecution,
            "QUERY_EXECUTION_ERROR",
            &[("message", "Stage failed")],
        );
        assert_eq!(err.kind, SparkErrorKind::QueryExecution);
    }

    #[test]
    fn test_streaming_query_exception_mapping() {
        let err = SparkError::classed(SparkErrorKind::StreamingQuery, "STREAMING_QUERY_ERROR", &[]);
        assert_eq!(err.kind, SparkErrorKind::StreamingQuery);
    }

    #[test]
    fn test_python_exception_mapping() {
        let err = SparkError::classed(
            SparkErrorKind::Python,
            "PYTHON_ERROR",
            &[("message", "Python worker failed")],
        );
        assert_eq!(err.kind, SparkErrorKind::Python);
    }

    #[test]
    fn test_spark_runtime_exception_mapping() {
        let err = SparkError::classed(SparkErrorKind::SparkRuntime, "SPARK_RUNTIME_ERROR", &[]);
        assert_eq!(err.kind, SparkErrorKind::SparkRuntime);
    }

    #[test]
    fn test_connect_grpc_exception_mapping() {
        let err = SparkError::classed(
            SparkErrorKind::ConnectGrpc,
            "RESPONSE_ALREADY_RECEIVED",
            &[],
        );
        assert_eq!(err.kind, SparkErrorKind::ConnectGrpc);
    }

    #[test]
    fn test_invalid_plan_input_exception_mapping() {
        let err = SparkError::classed(
            SparkErrorKind::InvalidPlanInput,
            "INVALID_PLAN_INPUT",
            &[("message", "Invalid plan")],
        );
        assert_eq!(err.kind, SparkErrorKind::InvalidPlanInput);
    }

    #[test]
    fn a_status_whose_google_rpc_status_does_not_match_it_is_grpc_status_s_value_error() {
        // As `grpc_status.rpc_status.from_call` of grpcio's error, which PySpark's client calls as
        // it reads a failure: of its code first, then of its details, of grpc_status's texts, and
        // of grpcio's error of the call as the context of the ValueError.
        let proto = |code: i32, message: &str| {
            RpcStatus {
                code,
                message: message.to_string(),
                details: Vec::new(),
            }
            .encode_to_vec()
        };
        let mismatch = |status: &Status, call| {
            from_call_error_in(status, call, true).map(|error| {
                assert_eq!(error.kind, SparkErrorKind::BuiltinValueError);
                let grpcio = error.grpcio.as_deref().unwrap();
                assert_eq!(grpcio.call, call);
                assert_eq!(grpcio.code, status.code());
                assert!(grpcio
                    .trailing_metadata
                    .iter()
                    .any(|(key, value)| key == "grpc-status-details-bin"
                        && value[..] == status.details()[..]));
                error.message()
            })
        };
        let code = Status::with_details(
            tonic::Code::ResourceExhausted,
            "failed",
            proto(13, "failed").into(),
        );
        assert_eq!(
            mismatch(&code, GrpcCall::ServerStreaming).as_deref(),
            Some(
                "Code in Status proto (StatusCode.INTERNAL) doesn't match status code \
                 (StatusCode.RESOURCE_EXHAUSTED)"
            )
        );
        let invalid =
            Status::with_details(tonic::Code::Internal, "failed", proto(99, "failed").into());
        assert_eq!(
            mismatch(&invalid, GrpcCall::Unary).as_deref(),
            Some("Invalid status code 99")
        );
        let message =
            Status::with_details(tonic::Code::Internal, "failed", proto(13, "other").into());
        assert_eq!(
            mismatch(&message, GrpcCall::Unary).as_deref(),
            Some("Message in Status proto (other) doesn't match status details (failed)")
        );
        // Of details that are not a `google.rpc.Status`, protobuf's DecodeError.
        let undecodable = Status::with_details(tonic::Code::Internal, "boom", b"abc"[..].into());
        assert_eq!(
            mismatch(&undecodable, GrpcCall::Unary).as_deref(),
            Some(STATUS_DECODE_ERROR)
        );
        // Of details that were not base64, which gRPC's core gives the call empty: a status of
        // the code OK.
        let mut empty = Status::internal("boom");
        empty.set_source(std::sync::Arc::new(crate::metadata::UndecodedDetails));
        assert_eq!(
            mismatch(&empty, GrpcCall::ServerStreaming).as_deref(),
            Some(
                "Code in Status proto (StatusCode.OK) doesn't match status code \
                 (StatusCode.INTERNAL)"
            )
        );
        // A status that its details match, one of no details, and the Rust client's dialect.
        let matched =
            Status::with_details(tonic::Code::Internal, "failed", proto(13, "failed").into());
        assert!(mismatch(&matched, GrpcCall::Unary).is_none());
        assert!(mismatch(&Status::internal("failed"), GrpcCall::Unary).is_none());
        assert!(from_call_error_in(&code, GrpcCall::Unary, false).is_none());
        assert!(grpcio_error(&Status::internal("failed"), GrpcCall::Unary)
            .trailing_metadata
            .is_empty());
    }
}
