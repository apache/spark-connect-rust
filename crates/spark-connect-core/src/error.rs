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
    /// ``ValueError``, a Python builtin: a connection-string value that `int()` or `urllib`
    /// rejects, which the reference client raises unwrapped rather than as a `PySparkValueError`.
    BuiltinValueError,
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
        }
    }

    /// Create a `BuiltinValueError`-kind error with a plain message: a `ValueError` of the Python
    /// builtin, which is what `int()` and `urllib` raise, as the reference client lets them.
    pub fn builtin_value_msg(message: impl Into<String>) -> Self {
        let mut error = Self::value_msg(message);
        error.kind = SparkErrorKind::BuiltinValueError;
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

    /// Convert a gRPC Status to a SparkError.
    ///
    /// A status with an `ErrorInfo` is an error that the server raised. As
    /// `pyspark.errors.exceptions.connect.convert_exception` does, its exception classes,
    /// condition, message parameters, SQL state and stack trace are read from the
    /// `ErrorInfo`, and its kind from the classes. [`SparkError::with_details`] completes it
    /// with what `FetchErrorDetails` returns. The code is the one grpc-python reports (see
    /// [`crate::retries::grpc_python_code`]), so that a dropped connection is `UNAVAILABLE`.
    pub fn from_grpc_status(status: Status) -> Self {
        let message = status.message().to_string();
        let code = crate::retries::grpc_python_code(&status);
        let details = status.details().to_vec();
        let mut err = if code == tonic::Code::DeadlineExceeded {
            // As `SparkConnectClient._handle_rpc_error`, which raises this for a call past its
            // per-RPC deadline before it reads the error of the status.
            Self::connect_msg(format!(
                "[{}] {}: RPC deadline exceeded. The client applies per-RPC timeouts to \
                 prevent silent hangs on broken connections. Deadlines can be configured via \
                 the rpc_deadlines parameter of SparkConnectClient. To disable all deadlines: \
                 SparkConnectClient(url, rpc_deadlines=RpcDeadlines.disabled()).",
                code, message
            ))
        } else {
            match parse_error_info(&details) {
                Some(info) => Self::from_error_info(info, message),
                None => Self::connect_msg(format!("[{}] {}", code, message)),
            }
        };
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

/// The `google.rpc.ErrorInfo` of a gRPC status, from its `grpc-status-details-bin`: the
/// protobuf-encoded `google.rpc.Status` whose details hold Any-packed messages.
fn parse_error_info(details: &[u8]) -> Option<RpcErrorInfo> {
    if details.is_empty() {
        return None;
    }
    RpcStatus::decode(details)
        .ok()?
        .details
        .into_iter()
        .filter(|detail| detail.type_url == "type.googleapis.com/google.rpc.ErrorInfo")
        .find_map(|detail| RpcErrorInfo::decode(&detail.value[..]).ok())
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
        assert!(message.ends_with("connection refused"), "{message}");
        // Details are only for a server error.
        let unchanged = err.with_details(None, true);
        assert!(unchanged.server.is_none());
        assert_eq!(unchanged.message(), message);
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
}
