//! What PySpark's client logs of its calls to the logger of Spark Connect
//! (`pyspark.sql.connect.logging`): its plans, its requests and its responses, which the client
//! builds and reads here, of the levels and the messages of PySpark's records, at the points of each
//! call where PySpark's `SparkConnectClient` logs them.
//!
//! The listener of the process, which a binding sets, tells whether a level is logged, as of the
//! call of the calling thread, and writes the records. A record of a level that is not logged is
//! not made: no message is encoded or formatted for it. Without a listener, nothing is logged. The
//! error of a record, of a handler or a filter of the logger that raised, fails the call at the
//! record, as the exception of PySpark's `logger.debug(...)` goes out of its call there. Not a
//! stable API.

use std::borrow::Cow;
use std::sync::{Arc, RwLock};

use crate::error::{Result, SparkError};
use crate::retries::Delivery;

/// The level of a record, of Python's `logging`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Level {
    /// `logging.DEBUG`.
    Debug,
    /// `logging.INFO`.
    Info,
    /// `logging.WARNING`.
    Warning,
    /// `logging.ERROR`.
    Error,
}

impl Level {
    /// The number of the level in Python's `logging`.
    pub fn number(self) -> i32 {
        match self {
            Level::Debug => 10,
            Level::Info => 20,
            Level::Warning => 30,
            Level::Error => 40,
        }
    }
}

/// The message of a record.
#[derive(Clone, Debug)]
pub enum Message {
    /// A text, as it is.
    Text(String),
    /// `prefix` and then the text of `message`, as PySpark's `SparkConnectClient._proto_to_string`
    /// writes it: on one line, of the protobuf runtime of the binding, after its truncation to 8
    /// levels and to short strings and bytes (`_truncate`) when `truncated`.
    Proto {
        /// The text before the message.
        prefix: Cow<'static, str>,
        /// The message.
        message: Encoded,
        /// Whether the message is truncated first, as `_proto_to_string(p, True)` truncates it.
        truncated: bool,
    },
    /// `text`, logged with the argument `error`, as PySpark's `logger.debug(text, e)` logs it: the
    /// text has no placeholder for it, so that the formatting of the record fails, as Python's
    /// `logging` reports it (`--- Logging error ---`). Of the error of a call that the client
    /// catches as PySpark's `except Exception` does, which a listener logs only of an error of an
    /// `Exception`, as PySpark's catches only those.
    WithError {
        /// The text, of no placeholder.
        text: &'static str,
        /// The error of the argument, of the exception that PySpark's client caught.
        error: Box<SparkError>,
    },
}

/// A message of Spark Connect, encoded, of the full name of its type, as
/// `spark.connect.ExecutePlanRequest`.
#[derive(Clone, Debug)]
pub struct Encoded {
    /// The full name of the type of the message.
    pub type_name: &'static str,
    /// The encoding of the message.
    pub bytes: Vec<u8>,
}

/// The stack that the encoding of a message takes for its levels, as the clone of a request does
/// (see `REQUEST_STACK` of the client).
const ENCODE_STACK: usize = if cfg!(debug_assertions) {
    16 << 20
} else {
    4 << 20
};

impl Encoded {
    /// `message` of the type `type_name`, encoded on a stack that holds its levels.
    pub fn of<M: prost::Message>(type_name: &'static str, message: &M) -> Self {
        Encoded {
            type_name,
            bytes: stacker::maybe_grow(ENCODE_STACK, ENCODE_STACK, || message.encode_to_vec()),
        }
    }
}

/// What a binding does of the records of the client: it tells whether a level is logged, and
/// writes a record. Not a stable API.
pub trait Listener: Send + Sync {
    /// Whether the logger writes a record of `level`, as PySpark's `logger.isEnabledFor` tells,
    /// read once per call: as of the call of the calling thread, else as of the last call.
    fn enabled_for(&self, level: Level) -> bool;

    /// Write a record of `level` and `message`, as PySpark's client logs it on the thread of the
    /// call: at once on a thread of the binding's interpreter, else the delivery of the record,
    /// which the call awaits before it goes on (see [`Delivery`]). The error of the record, of the
    /// delivery, fails the call at the record.
    fn log(&self, level: Level, message: Message) -> Delivery;
}

/// The listener of the process.
static LISTENER: RwLock<Option<Arc<dyn Listener>>> = RwLock::new(None);

/// Set the listener of the records of the process, or none.
pub fn set_listener(listener: Option<Arc<dyn Listener>>) {
    *LISTENER.write().unwrap_or_else(|e| e.into_inner()) = listener;
}

fn listener() -> Option<Arc<dyn Listener>> {
    LISTENER.read().unwrap_or_else(|e| e.into_inner()).clone()
}

/// Whether a record of `level` is logged.
pub fn enabled(level: Level) -> bool {
    listener().is_some_and(|listener| listener.enabled_for(level))
}

/// Log the message that `message` makes, of `level`, when a record of it is logged: the delivery
/// of the record, which the caller awaits before it goes on, as a record is written only once its
/// delivery is polled, and whose error fails the call at the record. `message` is not called
/// otherwise.
#[must_use = "a record is written once its delivery is awaited"]
pub fn log(level: Level, message: impl FnOnce() -> Message) -> Delivery {
    let listener = listener()?;
    if !listener.enabled_for(level) {
        return None;
    }
    listener.log(level, message())
}

/// As [`log`], awaited: the error of the record.
pub async fn logged(level: Level, message: impl FnOnce() -> Message) -> Result<()> {
    match log(level, message) {
        Some(delivery) => delivery.await,
        None => Ok(()),
    }
}

/// As [`log`], of a caller that is not async: the delivery is waited for on the calling thread,
/// of the error of the record. A delivery that is done at once, as of a record that a listener
/// wrote on the calling thread, which failed, is not waited for, so that a caller that a wait for
/// the server polls, as its watch of the signals, logs without a wait of its own.
pub fn log_blocking(level: Level, message: impl FnOnce() -> Message) -> Result<()> {
    let Some(mut delivery) = log(level, message) else {
        return Ok(());
    };
    let mut context = std::task::Context::from_waker(std::task::Waker::noop());
    match delivery.as_mut().poll(&mut context) {
        std::task::Poll::Ready(logged) => logged,
        std::task::Poll::Pending => crate::runtime::block_on(delivery),
    }
}

/// Log the text that `text` makes at the debug level, as [`log_blocking`].
pub fn debug_blocking(text: impl FnOnce() -> String) -> Result<()> {
    log_blocking(Level::Debug, || Message::Text(text()))
}

thread_local! {
    /// Whether the records of this thread are written on it, a thread of a client of its own,
    /// which runs the callbacks of the user's code and may wait for the binding's interpreter, as
    /// the thread of a listener bus (see [`write_records_on_this_thread`]).
    static WRITTEN_HERE: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Have the records of the calls of the calling thread written on it, rather than delivered from
/// it, as the records of a call of a thread of the runtime are: the thread of a client of its own
/// that runs the callbacks of the user's code, as the listener bus of streaming queries runs them
/// on the thread that reads its events, as PySpark's `_query_event_handler` logs the responses
/// that it reads and runs the listeners on its thread. Not a stable API.
#[doc(hidden)]
pub fn write_records_on_this_thread() {
    WRITTEN_HERE.with(|here| here.set(true));
}

/// Whether the records of the calling thread are written on it (see
/// [`write_records_on_this_thread`]). Not a stable API.
#[doc(hidden)]
pub fn records_written_here() -> bool {
    WRITTEN_HERE.with(std::cell::Cell::get)
}

/// The full names of the messages that the client logs.
pub mod types {
    /// `spark.connect.Plan`.
    pub const PLAN: &str = "spark.connect.Plan";
    /// `spark.connect.Command`.
    pub const COMMAND: &str = "spark.connect.Command";
    /// `spark.connect.ExecutePlanRequest`.
    pub const EXECUTE_PLAN_REQUEST: &str = "spark.connect.ExecutePlanRequest";
    /// `spark.connect.ExecutePlanResponse`.
    pub const EXECUTE_PLAN_RESPONSE: &str = "spark.connect.ExecutePlanResponse";
}

/// What PySpark's `to_table`, `to_table_as_iterator` and `to_pandas` log of the plan that they
/// execute, `plan`, as `to_proto` built it, compressed (`Executing plan ...`), truncated.
pub fn executing_plan(plan: Option<&spark_connect_proto::Plan>) -> Message {
    let message = match plan {
        Some(plan) => Encoded::of(types::PLAN, plan),
        None => Encoded::of(types::PLAN, &spark_connect_proto::Plan::default()),
    };
    Message::Proto {
        prefix: "Executing plan ".into(),
        message,
        truncated: true,
    }
}

/// What PySpark's client logs of an execution, of the method of its `SparkConnectClient` that the
/// execution stands for. Each but [`ExecutionLog::Execute`] and [`ExecutionLog::Silent`] logs
/// the request (`ExecuteAndFetchAsIterator. Request: ...`), as it is before the hook of the
/// executions runs, and each response as the stream reads it
/// (`ExecuteAndFetchAsIterator. Response received: ...`, `Received the schema.`, ...).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExecutionLog {
    /// `to_table`, of `collect`, `toArrow` and the other actions of a DataFrame:
    /// `Executing plan ...` of the plan as it is sent, compressed, then `ExecuteAndFetch`.
    ToTable,
    /// `to_pandas`, whose `Executing plan ...` the caller logged before it read the settings of
    /// its conversion (see [`executing_plan`]): `ExecuteAndFetch`.
    Fetch,
    /// `to_table_as_iterator`, of `toLocalIterator`: `Executing plan ...`.
    ToTableAsIterator,
    /// `execute_command`: `Execute command for command ...` of the command before it is
    /// compressed, then `ExecuteAndFetch`.
    #[default]
    ExecuteCommand,
    /// `execute_command_as_iterator`, of the listeners of streaming queries:
    /// `Execute command as iterator for command ...`.
    ExecuteCommandAsIterator,
    /// `_execute`, of the registrations of functions, table functions, data sources and Java
    /// functions: `Execute`, and nothing of the request or of its responses.
    Execute,
    /// Nothing, of an execution that PySpark's client does not log.
    Silent,
}

impl ExecutionLog {
    /// Whether the request and the responses of the execution are logged.
    pub fn logs_responses(self) -> bool {
        !matches!(self, ExecutionLog::Execute | ExecutionLog::Silent)
    }
}

/// What PySpark's client logs of an analysis, of the method of its `SparkConnectClient` that the
/// analysis stands for, of its plan as it is sent, compressed. `explain_string` logs its plan of the
/// mode as it is given it, before the mode is checked, which its caller logs (see
/// [`explain_plan`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AnalysisLog {
    /// `schema`: `Schema for plan: ...`.
    Schema,
    /// Nothing, as `_analyze` logs nothing of the other analyses.
    #[default]
    Silent,
}

/// What PySpark's `explain_string` logs of the plan that it explains, `plan`, as `to_proto` built
/// it, compressed, of the mode as it is given it, `mode` (`Explain (mode=...) for plan ...`),
/// truncated, before its `_analyze` checks the mode.
pub fn explain_plan(mode: &str, plan: &spark_connect_proto::Plan) -> Message {
    Message::Proto {
        prefix: format!("Explain (mode={mode}) for plan ").into(),
        message: Encoded::of(types::PLAN, plan),
        truncated: true,
    }
}

/// The integer that Python's `int()` reads of `text`, or None when it raises: a sign, decimal digits
/// that single underscores may separate, and whitespace around them, as PySpark's
/// `int(plan_compression_threshold_str)` reads the threshold of the plan compression. Of the ASCII
/// digits that a server sends, and of the range of an `i128`.
pub(crate) fn python_int(text: &str) -> Option<i128> {
    let text = text.trim_matches(|c: char| c.is_whitespace());
    let (negative, digits) = match text.as_bytes().first() {
        Some(b'-') => (true, &text[1..]),
        Some(b'+') => (false, &text[1..]),
        _ => (false, text),
    };
    if digits.is_empty() || digits.starts_with('_') || digits.ends_with('_') {
        return None;
    }
    let mut value: i128 = 0;
    let mut underscore = false;
    for c in digits.chars() {
        if c == '_' {
            if underscore {
                return None;
            }
            underscore = true;
            continue;
        }
        underscore = false;
        let digit = c.to_digit(10)?;
        value = value.checked_mul(10)?.checked_add(i128::from(digit))?;
    }
    Some(if negative { -value } else { value })
}

/// The text of a record of the logs of the tests, of the calls of the calling thread, with a
/// listener of the test's own: the level and the message of each, the text of a message of a proto
/// its type and whether it is truncated. The tests that listen take turns, and a listener of
/// another test does not see what this one logs.
#[cfg(test)]
pub(crate) fn logged_by<T>(level: Level, f: impl FnOnce() -> T) -> (T, Vec<(Level, String)>) {
    static TURN: std::sync::Mutex<()> = std::sync::Mutex::new(());
    struct Recorder {
        level: Level,
        thread: std::thread::ThreadId,
        records: std::sync::Mutex<Vec<(Level, String)>>,
    }
    impl Listener for Recorder {
        fn enabled_for(&self, level: Level) -> bool {
            level >= self.level && std::thread::current().id() == self.thread
        }
        fn log(&self, level: Level, message: Message) -> Delivery {
            let text = match message {
                Message::Text(text) => text,
                Message::Proto {
                    prefix,
                    message,
                    truncated,
                } => format!("{prefix}<{} truncated={truncated}>", message.type_name),
                Message::WithError { text, error } => format!("{text} % ({})", error.message()),
            };
            self.records.lock().unwrap().push((level, text));
            None
        }
    }
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let recorder = Arc::new(Recorder {
        level,
        thread: std::thread::current().id(),
        records: std::sync::Mutex::new(Vec::new()),
    });
    set_listener(Some(recorder.clone()));
    let output = f();
    set_listener(None);
    let records = recorder.records.lock().unwrap().clone();
    (output, records)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_record_of_a_level_that_is_not_logged_is_not_made() {
        let ((), records) = logged_by(Level::Info, || {
            log_blocking(Level::Debug, || panic!("not made")).unwrap();
            debug_blocking(|| panic!("not made")).unwrap();
            log_blocking(Level::Info, || Message::Text("logged".to_string())).unwrap();
        });
        assert_eq!(records, vec![(Level::Info, "logged".to_string())]);
        // Without a listener, nothing is made.
        assert!(!enabled(Level::Error));
        log_blocking(Level::Error, || panic!("not made")).unwrap();
    }

    #[test]
    fn the_records_of_a_thread_are_written_on_it_once_it_asks_for_it() {
        assert!(!records_written_here());
        let here = std::thread::spawn(|| {
            write_records_on_this_thread();
            records_written_here()
        });
        assert!(here.join().unwrap());
        assert!(!records_written_here());
    }

    #[test]
    fn python_int_reads_what_int_reads() {
        assert_eq!(python_int("10485760"), Some(10485760));
        assert_eq!(python_int(" -1 "), Some(-1));
        assert_eq!(python_int("+1_000"), Some(1000));
        assert_eq!(python_int("007"), Some(7));
        assert_eq!(python_int("\t5\n"), Some(5));
        for invalid in ["", "-", "1__0", "_1", "1_", "1.5", "0x10", "1e3", "- 1"] {
            assert_eq!(python_int(invalid), None, "{invalid:?}");
        }
    }
}
