// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! The records of PySpark's client of its plans, its requests and its responses, which the
//! transport builds and reads (see `spark_connect_core::client_log`), written to the logger of
//! Spark Connect (`pyspark.sql.connect.logging`) as PySpark's client writes them: of its methods,
//! `logger.debug(...)` and the others, of the same messages, the messages of protobuf of the text of
//! PySpark's `SparkConnectClient._proto_to_string` of the runtime's messages
//! (`pyspark.sql.connect.client.core._proto_text`).
//!
//! The levels that the logger writes are read as each call of the client starts, of
//! `logger.isEnabledFor`, and hold for the records of the transport until the next call starts:
//! a record of a level that the logger does not write is not made, and nothing is encoded or
//! formatted for it. And the call stack of the user's code, of
//! `SPARK_CONNECT_DEBUG_CLIENT_CALL_STACK`, is read then, which the requests of the call carry.
//! The exception of a record, of a handler or a filter of the logger, fails the call at the record,
//! as it goes out of PySpark's `logger.debug(...)`.

use std::sync::atomic::{AtomicI32, Ordering};

use pyo3::prelude::*;
use pyo3::sync::PyOnceLock;
use spark_connect::session::SparkSession;
use spark_connect_core::client_log::{Encoded, Level, Listener, Message};
use spark_connect_core::error::SparkError;
use spark_connect_core::retries::Delivery;

/// The least level of `logging` that the logger writes, as of the start of the last call of the
/// process, or `NONE` when it writes none.
static THRESHOLD: AtomicI32 = AtomicI32::new(NONE);

/// A level above every level of `logging`: the logger writes no record.
const NONE: i32 = i32::MAX;

/// The logger of Spark Connect, `pyspark.sql.connect.logging.logger`, as PySpark's modules of the
/// client import it: the logger as it was when the extension first logged.
static LOGGER: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

/// The logger of Spark Connect.
pub(crate) fn logger(py: Python<'_>) -> PyResult<&Bound<'_, PyAny>> {
    LOGGER
        .get_or_try_init(py, || {
            Ok::<_, PyErr>(
                py.import("pyspark.sql.connect.logging")?
                    .getattr("logger")?
                    .unbind(),
            )
        })
        .map(|logger| logger.bind(py))
}

/// Read the levels that the logger writes, as a call of the client starts: the levels of the
/// records of the transport until the next call starts. A logger that cannot be read writes none.
pub(crate) fn read_levels(py: Python<'_>) {
    let threshold = threshold_of(py).unwrap_or(NONE);
    THRESHOLD.store(threshold, Ordering::Relaxed);
}

/// The least of PySpark's levels that the logger writes, as `logger.isEnabledFor` tells: none of a
/// disabled logger, as the logger is unless `SPARK_CONNECT_LOG_LEVEL` is set, which its attribute
/// tells at once.
fn threshold_of(py: Python<'_>) -> PyResult<i32> {
    let logger = logger(py)?;
    if logger.getattr(pyo3::intern!(py, "disabled"))?.is_truthy()? {
        return Ok(NONE);
    }
    for level in [Level::Debug, Level::Info, Level::Warning, Level::Error] {
        if logger
            .call_method1(pyo3::intern!(py, "isEnabledFor"), (level.number(),))?
            .is_truthy()?
        {
            return Ok(level.number());
        }
    }
    Ok(NONE)
}

thread_local! {
    /// Whether the logs of the call of this thread are silenced: once a KeyboardInterrupt
    /// interrupted the execution that it waits for, whose responses and failure the client reads
    /// until the server ends it, where PySpark's client raises the KeyboardInterrupt at once and
    /// reads, and logs, nothing more of it (see [`interrupted`]).
    static SILENCED: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Whether the logs of the call of the calling thread are silenced (see [`SILENCED`]).
pub(crate) fn silenced() -> bool {
    SILENCED.with(std::cell::Cell::get)
}

/// Log that a KeyboardInterrupt interrupts the execution of `operation_id` that the call of the
/// calling thread waits for, as the `except KeyboardInterrupt` of PySpark's
/// `_execute_and_fetch_as_iterator` logs it (`Interrupt request received for operation=...`)
/// before it interrupts it, and silence the logs of the rest of the call, which PySpark's does not
/// read. The error is the exception of the record, which replaces the KeyboardInterrupt, and with
/// which nothing is interrupted, nor silenced, as it goes out of PySpark's `except` clause.
pub(crate) fn interrupted(operation_id: &str) -> Result<(), SparkError> {
    interrupt_requested(operation_id)?;
    SILENCED.with(|silenced| silenced.set(true));
    Ok(())
}

/// Log that a KeyboardInterrupt interrupts the execution of `operation_id`, as [`interrupted`]
/// does, of an execution that the call reads no more of.
pub(crate) fn interrupt_requested(operation_id: &str) -> Result<(), SparkError> {
    spark_connect_core::client_log::debug_blocking(|| {
        format!("Interrupt request received for operation={operation_id}")
    })
}

/// Run `f`, a wait of a call of the calling thread, of the logs of its own: of none silenced as it
/// starts, and of those of the call that it runs during as they were once it ends.
pub(crate) fn of_its_own_logs<T>(f: impl FnOnce() -> T) -> T {
    let before = SILENCED.with(|silenced| silenced.replace(false));
    let output = f();
    SILENCED.with(|silenced| silenced.set(before));
    output
}

/// What the client does of the records of the transport: it writes them to the logger, on the
/// thread of the call, or on the delivery thread, which the call waits for, of the exception of a
/// record, which fails the call (see [`crate::client::delivered`]).
struct ConnectLogger;

impl Listener for ConnectLogger {
    fn enabled_for(&self, level: Level) -> bool {
        level.number() >= THRESHOLD.load(Ordering::Relaxed) && !silenced()
    }

    fn log(&self, level: Level, message: Message) -> Delivery {
        crate::client::delivered(false, move |py| write(py, level, message))
    }
}

/// Write a record of `level` and `message` to the logger, of the method of the level: the
/// exception of a handler or a filter of the logger.
pub(crate) fn write(py: Python<'_>, level: Level, message: Message) -> PyResult<()> {
    let logger = logger(py)?;
    let method = match level {
        Level::Debug => pyo3::intern!(py, "debug"),
        Level::Info => pyo3::intern!(py, "info"),
        Level::Warning => pyo3::intern!(py, "warning"),
        Level::Error => pyo3::intern!(py, "error"),
    };
    match message {
        Message::Text(text) => {
            logger.call_method1(method, (text,))?;
        }
        Message::Proto {
            prefix,
            message,
            truncated,
        } => {
            let text = proto_text(py, &message, truncated)?;
            logger.call_method1(method, (format!("{prefix}{text}"),))?;
        }
        Message::WithError { text, error } => {
            // As PySpark's `except Exception`, which catches the exception that it logs: an
            // exception that is not an `Exception` is not caught, nor logged.
            if crate::errors::is_base_exception_only(&error) {
                return Ok(());
            }
            let exception = crate::errors::exception_of(py, &error);
            logger.call_method1(method, (text, exception.value(py)))?;
        }
    }
    Ok(())
}

/// The text of `message` as PySpark's `SparkConnectClient._proto_to_string` writes it, truncated
/// when `truncated`, of the message of the runtime of `pyspark.sql.connect.proto` of its encoding,
/// of any depth (see `pyspark.sql.connect.client.core._proto_text`).
fn proto_text(py: Python<'_>, message: &Encoded, truncated: bool) -> PyResult<String> {
    let bytes = pyo3::types::PyBytes::new(py, &message.bytes);
    crate::on_grown_stack(|| {
        py.import("pyspark.sql.connect.client.core")?
            .call_method1(
                pyo3::intern!(py, "_proto_text"),
                (message.type_name, bytes, truncated),
            )?
            .extract()
    })
}

/// Write the records of the transport to the logger of Spark Connect, as the extension loads.
pub(crate) fn listen() {
    spark_connect_core::client_log::set_listener(Some(std::sync::Arc::new(ConnectLogger)));
}

/// What a call of the client is, of which the call stack of its requests names the innermost
/// frame of the user's code (see `pyspark.sql.connect.client.core._call_stack_of_call`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Called {
    /// A method or an attribute of the extension, which has no frame: of the name that the source
    /// of its call names.
    Method,
    /// An attribute that is a `functools.cached_property` of PySpark's, of its name, as
    /// `DataFrame.isStreaming` and `SparkSession.version`: PySpark's stack has the frame of the
    /// `__get__` of `functools`, which calls the function of the attribute.
    CachedProperty(&'static str),
    /// A finalizer, which the garbage collector runs where the last reference goes, as PySpark's
    /// `__del__`.
    Finalizer,
}

thread_local! {
    /// What the call that starts next on this thread is (see [`of_cached_property`]).
    static CALLED: std::cell::Cell<Called> = const { std::cell::Cell::new(Called::Method) };
}

/// Run `f`, the call of an attribute of the extension that is a `functools.cached_property` of
/// PySpark of the name `name`, as the call that `f` starts (see [`Called::CachedProperty`]).
pub(crate) fn of_cached_property<T>(name: &'static str, f: impl FnOnce() -> T) -> T {
    let before = CALLED.with(|called| called.replace(Called::CachedProperty(name)));
    let output = f();
    CALLED.with(|called| called.set(before));
    output
}

/// What the call that starts on this thread is, which is then a method of the extension for the
/// calls that start during it.
pub(crate) fn take_called() -> Called {
    CALLED.with(|called| called.replace(Called::Method))
}

/// Read, as a call of `session` starts, the levels that the logger writes (see [`read_levels`])
/// and the call stack of the user's code that the requests of the call carry, as PySpark's
/// requests carry the call stack of their build when `SPARK_CONNECT_DEBUG_CLIENT_CALL_STACK` is
/// set (see [`call_stack`]), of the call of `called`: the call stack. The error is that of the
/// read of the call stack, which fails the call, as it fails the build of PySpark's request.
pub(crate) fn start_call(
    py: Python<'_>,
    session: &SparkSession,
    called: Called,
) -> PyResult<Option<prost_types::Any>> {
    read_levels(py);
    let call_stack = call_stack(py, called)?;
    session.set_thread_call_stack(call_stack.clone());
    Ok(call_stack)
}

/// Read, as a call of `session` starts during another call of its thread, as of a progress
/// handler, a hook or a finalizer, the levels that the logger writes, and give the requests of the
/// call `call_stack`, the call stack of that call: PySpark's stack of a call that its own call
/// runs ends at the first frame of the pyspark package of that call, and the stack is read once
/// as a call starts.
pub(crate) fn start_nested_call(
    py: Python<'_>,
    session: &SparkSession,
    call_stack: Option<prost_types::Any>,
) {
    read_levels(py);
    session.set_thread_call_stack(call_stack);
}

/// Whether the requests carry the call stack of the user's code, as PySpark's
/// `os.getenv("SPARK_CONNECT_DEBUG_CLIENT_CALL_STACK", "false").lower() in ("true", "1")` tells it,
/// of the environment of the process as the call starts.
fn call_stack_enabled() -> bool {
    std::env::var_os("SPARK_CONNECT_DEBUG_CLIENT_CALL_STACK").is_some_and(|value| {
        matches!(
            value.to_string_lossy().to_lowercase().as_str(),
            "true" | "1"
        )
    })
}

/// The call stack of the user's code of the call of `called`, as PySpark's
/// `_build_call_stack_trace` packs it into a `google.protobuf.Any` of a
/// `FetchErrorDetailsResponse.Error`, of the frames that
/// `pyspark.sql.connect.client.core._call_stack_of_call` reads, if the requests carry it.
fn call_stack(py: Python<'_>, called: Called) -> PyResult<Option<prost_types::Any>> {
    if !call_stack_enabled() {
        return Ok(None);
    }
    let read = py
        .import("pyspark.sql.connect.client.core")?
        .getattr(pyo3::intern!(py, "_call_stack_of_call"))?;
    let frames = match called {
        Called::Method => read.call0()?,
        Called::CachedProperty(name) => {
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("cached_property", name)?;
            read.call((), Some(&kwargs))?
        }
        Called::Finalizer => {
            let kwargs = pyo3::types::PyDict::new(py);
            kwargs.set_item("innermost", "__del__")?;
            read.call((), Some(&kwargs))?
        }
    };
    let frames: Vec<(String, String, i32)> = frames.extract()?;
    if frames.is_empty() {
        return Ok(None);
    }
    use prost::Message as _;
    use spark_connect_proto::fetch_error_details_response::{Error, StackTraceElement};
    let error = Error {
        stack_trace: frames
            .into_iter()
            .map(|(method_name, file_name, line_number)| StackTraceElement {
                // As PySpark's, which knows no class of a frame.
                declaring_class: String::new(),
                method_name,
                file_name: Some(file_name),
                line_number,
            })
            .collect(),
        ..Default::default()
    };
    Ok(Some(prost_types::Any {
        type_url: "type.googleapis.com/spark.connect.FetchErrorDetailsResponse.Error".to_string(),
        value: error.encode_to_vec(),
    }))
}
