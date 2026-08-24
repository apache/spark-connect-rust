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
        }
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
        }
    }

    /// Get the rendered message for this error, performing template substitution.
    ///
    /// If an error_class is set, looks up the message template in error-conditions.json,
    /// substitutes `<param_name>` placeholders with values from `params`, and returns
    /// `[ERROR_CLASS] rendered_message`. Otherwise, returns the pre-set message.
    ///
    /// Mirrors `PySparkException.getMessage()` from `pyspark/errors/exceptions/base.py`.
    pub fn message(&self) -> String {
        if self.error_class.is_empty() {
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

    /// Convert a gRPC Status to a SparkError, extracting error class and parameters from metadata.
    ///
    /// Mirrors `convert_exception` from `pyspark.errors.exceptions.connect`.
    pub fn from_grpc_status(status: Status) -> Self {
        // Extract error metadata from trailer if present
        let message = status.message().to_string();
        let code = status.code() as i32;

        // Try to parse details from the Status (details is a byte slice)
        let details = status.details().to_vec();
        let mut err = if details.is_empty() {
            Self::connect_msg(format!("[{}] {}", status.code(), message))
        } else {
            // Decode grpc-status-details-bin as a google.rpc.Status with Any-packed ErrorInfo.
            match parse_error_info_from_details(&details) {
                Some((error_class, params, sql_state, server_stacktrace)) => Self {
                    kind: classify_error_kind(&error_class),
                    error_class,
                    params,
                    message,
                    sql_state,
                    contexts: Vec::new(),
                    server_stacktrace,
                    grpc_code: None,
                    grpc_details: None,
                },
                None => Self::connect_msg(format!("[{}] {}", status.code(), message)),
            }
        };
        // Preserve the raw status so the PyO3 layer can rebuild the exact pyspark exception.
        err.grpc_code = Some(code);
        if !details.is_empty() {
            err.grpc_details = Some(details);
        }
        err
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
        if let Some(condition) = self.get_condition() {
            format!("[{}] {}", condition, self.message())
        } else {
            self.message()
        }
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

/// Parse ErrorInfo from gRPC status details bytes.
///
/// Expects the details to be a protobuf-encoded google.rpc.Status containing
/// Any-packed google.rpc.ErrorInfo messages. Returns (error_class, params, sql_state, stacktrace) if found.
fn parse_error_info_from_details(
    details: &[u8],
) -> Option<(
    String,
    BTreeMap<String, String>,
    Option<String>,
    Option<String>,
)> {
    // Decode the outer google.rpc.Status from details
    let rpc_status = RpcStatus::decode(details).ok()?;

    // Look for an ErrorInfo in the details list
    for any_detail in &rpc_status.details {
        // google.rpc.ErrorInfo type URL
        if any_detail.type_url == "type.googleapis.com/google.rpc.ErrorInfo" {
            // Decode the Any.value as google.rpc.ErrorInfo
            if let Ok(error_info) = RpcErrorInfo::decode(&any_detail.value[..]) {
                // Extract errorClass and other metadata
                let error_class = error_info
                    .metadata
                    .get("errorClass")
                    .cloned()
                    .unwrap_or_else(|| error_info.reason.clone());

                let sql_state = error_info.metadata.get("sqlState").cloned();
                let stacktrace = error_info.metadata.get("stackTrace").cloned();

                // Convert HashMap to BTreeMap
                let params: BTreeMap<String, String> = error_info
                    .metadata
                    .iter()
                    .map(|(k, v)| (k.clone(), v.clone()))
                    .collect();

                // Return the parsed error info
                return Some((error_class, params, sql_state, stacktrace));
            }
        }
    }

    None
}

/// Classify error_class string into a SparkErrorKind.
///
/// Maps Spark error classes to the corresponding Python exception type.
fn classify_error_kind(error_class: &str) -> SparkErrorKind {
    if error_class.starts_with("ANALYSIS") {
        SparkErrorKind::Analysis
    } else if error_class.starts_with("PARSE") {
        SparkErrorKind::Parse
    } else if error_class.starts_with("ILLEGAL") {
        SparkErrorKind::IllegalArgument
    } else if error_class.starts_with("UNSUPPORTED") {
        SparkErrorKind::UnsupportedOperation
    } else if error_class.starts_with("ARITHMETIC") {
        SparkErrorKind::Arithmetic
    } else if error_class.contains("NOT_IMPLEMENTED") {
        SparkErrorKind::NotImplementedError
    } else if error_class.contains("RUNTIME") {
        SparkErrorKind::RuntimeError
    } else if error_class.contains("VALUE") {
        SparkErrorKind::ValueError
    } else {
        match error_class {
            "INVALID_CONNECT_URL" => SparkErrorKind::RuntimeError,
            "SYNTAX_ERROR" => SparkErrorKind::Parse,
            "INVALID_PLAN_INPUT" => SparkErrorKind::InvalidPlanInput,
            "PICKLE_ERROR" => SparkErrorKind::PickleException,
            "RESPONSE_ALREADY_RECEIVED" | "INVALID_HANDLE" => SparkErrorKind::ConnectGrpc,
            _ => SparkErrorKind::RuntimeError, // Default fallback
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

    #[test]
    fn test_parse_error_info_from_grpc_status() {
        // Create a synthetic google.rpc.ErrorInfo
        let mut error_metadata = HashMap::new();
        error_metadata.insert("errorClass".to_string(), "ANALYSIS_ERROR".to_string());
        error_metadata.insert("sqlState".to_string(), "42601".to_string());
        error_metadata.insert(
            "messageParameters".to_string(),
            r#"{"message":"test error"}"#.to_string(),
        );
        error_metadata.insert(
            "stackTrace".to_string(),
            "at org.apache.spark.sql....".to_string(),
        );

        let error_info = RpcErrorInfo {
            reason: "ANALYSIS_ERROR".to_string(),
            domain: "org.apache.spark.connect".to_string(),
            metadata: error_metadata,
        };

        // Pack into Any
        let error_info_bytes = error_info.encode_to_vec();
        let any = RpcAny {
            type_url: "type.googleapis.com/google.rpc.ErrorInfo".to_string(),
            value: error_info_bytes,
        };

        // Create outer Status
        let rpc_status = RpcStatus {
            code: 0,
            message: "Analysis error".to_string(),
            details: vec![any],
        };

        let status_bytes = rpc_status.encode_to_vec();

        // Parse it back
        let result = parse_error_info_from_details(&status_bytes);
        assert!(result.is_some());

        let (error_class, params, sql_state, stacktrace) = result.unwrap();
        assert_eq!(error_class, "ANALYSIS_ERROR");
        assert_eq!(sql_state, Some("42601".to_string()));
        assert_eq!(stacktrace, Some("at org.apache.spark.sql....".to_string()));
        assert!(params.contains_key("errorClass"));
    }

    #[test]
    fn test_classify_error_kind() {
        assert_eq!(
            classify_error_kind("ANALYSIS_ERROR"),
            SparkErrorKind::Analysis
        );
        assert_eq!(classify_error_kind("PARSE_ERROR"), SparkErrorKind::Parse);
        assert_eq!(
            classify_error_kind("ILLEGAL_ARGUMENT"),
            SparkErrorKind::IllegalArgument
        );
        assert_eq!(
            classify_error_kind("ARITHMETIC_ERROR"),
            SparkErrorKind::Arithmetic
        );
        assert_eq!(
            classify_error_kind("UNSUPPORTED_OPERATION"),
            SparkErrorKind::UnsupportedOperation
        );
        assert_eq!(
            classify_error_kind("INVALID_PLAN_INPUT"),
            SparkErrorKind::InvalidPlanInput
        );
        assert_eq!(
            classify_error_kind("RESPONSE_ALREADY_RECEIVED"),
            SparkErrorKind::ConnectGrpc
        );
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
            "[CANNOT_BE_NONE] [CANNOT_BE_NONE] Argument `x` cannot be None."
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
