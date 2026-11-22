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

//! The requests of PySpark Connect's `SparkConnectClient` (`pyspark.sql.connect.client.core`)
//! that it sends of the messages it is given, such as the plans of `to_table` and `schema`, over
//! the Rust transport of its session: the analyses, the configuration requests and the
//! executions, of the bytes of the messages of `pyspark.sql.connect.proto` that the client builds
//! as PySpark's does, and the settings of the client that the transport honors, such as its retry
//! policy.

use prost::Message;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use spark_connect::dataframe::PlanExecution;
// The messages of the client, with the type URLs of placeholders of their encodings, of which the
// core finds those of their relations as it walks them.
use spark_connect::plan::BuiltOutside;
use spark_connect::session::{ExecutionInfo, SparkSession};
use spark_connect_core::client::{ResultChunking, RpcDeadlines};
use spark_connect_core::client_log::ExecutionLog;
use spark_connect_core::error::GrpcCall;
use spark_connect_core::error::SparkError;
use spark_connect_core::retries::{
    BackgroundFailure, BackgroundRequest, CallRetries, Decision, Delivery, Handled, LongSleep,
    RetriesExceeded, RetriesListener, RetryDecider, RetryPolicy,
};
use spark_connect_proto as proto;

use crate::interrupt::interruptible;

/// The message of `bytes`, decoded on a stack that holds its levels, as a plan of any depth may be.
fn decode<M: Message + Default + Send>(bytes: &[u8]) -> PyResult<M> {
    crate::on_grown_stack(|| M::decode(bytes))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// The bytes of the `AnalyzePlanResponse` of the server to `request`, the bytes of an
/// `AnalyzePlanRequest` that the client built, as PySpark's `SparkConnectClient._analyze` sends
/// it, of the session `session`: of its plans as they are, as PySpark's `_analyze` sends the plans
/// that it is given.
pub(crate) fn analyze_plan<'py>(
    py: Python<'py>,
    session: &SparkSession,
    request: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let request = BuiltOutside::decoded(request, decode::<proto::AnalyzePlanRequest>)?;
    let response = interruptible(py, session, || {
        crate::on_grown_stack(|| session.analyze_plan_proto(request)).map_err(Box::new)
    })?;
    Ok(PyBytes::new(py, &crate::encode_on_grown_stack(response)))
}

/// Log `prefix` and the plan `plan`, the bytes of a `Plan` that the client was given, of its
/// placeholders replaced as `session` sends it, truncated, at the debug level, as PySpark's
/// `to_pandas`, `schema` and `explain_string` log the plan that they are given as they start
/// (`Executing plan ...`, `Schema for plan: ...` and `Explain (mode=...) for plan ...`): of a
/// caller that checked that the logger writes it. The error is the exception of the record, as it
/// goes out of PySpark's log.
pub(crate) fn log_plan(
    py: Python<'_>,
    session: &SparkSession,
    prefix: String,
    plan: &[u8],
) -> PyResult<()> {
    use spark_connect_core::client_log::{types, Encoded, Level, Message};
    let plan = BuiltOutside::decoded(plan, decode::<proto::Plan>)?;
    let encoded = crate::on_grown_stack(|| {
        session
            .resolved_plan_proto(plan)
            .map(|plan| Encoded::of(types::PLAN, &plan))
    })
    .map_err(crate::errors::spark_error_to_py_exception)?;
    crate::client_log::write(
        py,
        Level::Debug,
        Message::Proto {
            prefix: prefix.into(),
            message: encoded,
            truncated: true,
        },
    )
}

/// The bytes of `plan`, the bytes of a `Plan` of a relation or a command that the client sets in
/// a plan, compressed as `session` compresses its plans, after the lookup of the plan compression
/// of its first plan, as PySpark's `_try_compress_and_set_plan` compresses the plan of `to_proto`,
/// with what it logs then; None when it is not compressed, which the client sets as it is (see
/// `SparkSession::compressed_plan_proto`).
pub(crate) fn compressed_plan<'py>(
    py: Python<'py>,
    session: &SparkSession,
    plan: &[u8],
) -> PyResult<Option<Bound<'py, PyBytes>>> {
    let plan = BuiltOutside::decoded(plan, decode::<proto::Plan>)?;
    let compress =
        || crate::on_grown_stack(|| session.compressed_plan_proto(plan)).map_err(Box::new);
    // As PySpark's `_try_compress_and_set_plan`, which reads nothing of its client once the
    // settings are looked up, as it sends nothing: the user, the session ID and the extensions of
    // the client are read only for the request of the lookup.
    let compressed = if session.known_plan_compression().is_some() {
        crate::interrupt::interruptible_continuation(py, session, compress)?
    } else {
        interruptible(py, session, compress)?
    };
    Ok(compressed.map(|plan| PyBytes::new(py, &plan.encode_to_vec())))
}

/// The bytes of the `ConfigResponse` of the server to `request`, the bytes of a `ConfigRequest`
/// that the client built, as PySpark's `SparkConnectClient.config` sends it, of `session`.
pub(crate) fn config<'py>(
    py: Python<'py>,
    session: &SparkSession,
    request: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let request: proto::ConfigRequest = decode(request)?;
    let response = interruptible(py, session, || {
        session.config_proto(request).map_err(Box::new)
    })?;
    Ok(PyBytes::new(py, &response.encode_to_vec()))
}

/// Start the execution of `plan`, the bytes of a `Plan` that the client was given, of `session`,
/// whose observed metrics set the Observations of `observations` by their names, and whose
/// progress reaches the progress handlers when `reported`.
///
/// It logs what the method of PySpark's client of the name `log` logs of the execution: `to_table`,
/// `to_table_as_iterator`, `execute_command` and `execute_command_as_iterator`, and `fetch` for the
/// execution of `to_pandas`, which logged its plan as it started (see [`log_plan`]). Of none, a
/// plan of a relation logs as `to_table`, and one of a command as `execute_command`. The plan of
/// `to_table`, `to_table_as_iterator` and `fetch` is sent as it is, as PySpark's client sends the
/// plan that it is given, and the command of `execute_command` and `execute_command_as_iterator`
/// compressed, as it sets the command that it is given in its plan.
pub(crate) fn execute_plan(
    py: Python<'_>,
    session: &SparkSession,
    plan: &[u8],
    observations: Vec<(String, Bound<'_, PyAny>)>,
    reported: bool,
    log: Option<&str>,
) -> PyResult<PyExecutePlanResponses> {
    let log = match log {
        None => None,
        Some("to_table") => Some(ExecutionLog::ToTable),
        Some("to_table_as_iterator") => Some(ExecutionLog::ToTableAsIterator),
        Some("fetch") => Some(ExecutionLog::Fetch),
        Some("execute_command") => Some(ExecutionLog::ExecuteCommand),
        Some("execute_command_as_iterator") => Some(ExecutionLog::ExecuteCommandAsIterator),
        Some(other) => return Err(PyValueError::new_err(format!("unknown log: {other}"))),
    };
    let plan = BuiltOutside::decoded(plan, decode::<proto::Plan>)?;
    let observations = observations
        .into_iter()
        .map(|(name, observation)| {
            let observation = observation.cast::<crate::observation::PyObservation>()?;
            Ok((name, crate::observation::state_of_execution(observation)?))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let execution = interruptible(py, session, || {
        crate::on_grown_stack(|| match log {
            Some(log) => session.execute_plan_proto_logged(plan, observations, reported, log),
            None => session.execute_plan_proto(plan, observations, reported),
        })
        .map_err(Box::new)
    })?;
    Ok(PyExecutePlanResponses {
        session: session.clone(),
        operation_id: execution.operation_id().to_string(),
        execution: Some(execution),
    })
}

/// The responses of an execution that `SparkConnectClient` started of a plan it was given (see
/// `PlanExecution`), as PySpark's `_execute_and_fetch_as_iterator` reads them: each `next` gives
/// the bytes of a response, with the `PlanMetrics` of its metrics and the `PlanObservedMetrics` of
/// its observed metrics, as `DataFrame.executionInfo` has them, which the client builds the
/// `ExecutionInfo` of the execution of. Closing it, or reading it to its end, ends the execution.
// PySpark has no such class: it is of the extension, which has it, and not of PySpark's modules of
// the client, whose names are PySpark's.
#[pyclass(name = "ExecutePlanResponses", module = "pyspark._pyspark")]
pub struct PyExecutePlanResponses {
    session: SparkSession,
    /// The execution, until it ends or is closed.
    execution: Option<PlanExecution>,
    operation_id: String,
}

#[pymethods]
impl PyExecutePlanResponses {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// The next response, as the bytes of an `ExecutePlanResponse`, its plan metrics and its
    /// observed metrics, or the end of the iteration once the execution ended.
    #[allow(clippy::type_complexity)]
    fn __next__<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<Option<(Bound<'py, PyBytes>, Bound<'py, PyList>, Bound<'py, PyList>)>> {
        let Some(execution) = self.execution.as_mut() else {
            return Ok(None);
        };
        let session = self.session.clone();
        // As PySpark's iterator of the responses, whose reattaches and releases copy the request
        // of the execution: nothing of the client is read again.
        let next = crate::interrupt::interruptible_continuation(py, &session, || {
            execution.next_response().map_err(Box::new)
        });
        let response = match next {
            Ok(Some(response)) => response,
            Ok(None) => {
                self.finish(py);
                return Ok(None);
            }
            Err(error) => {
                self.finish(py);
                return Err(error);
            }
        };
        let of_response = ExecutionInfo {
            metrics: response.metrics.clone(),
            // Of literals of up to `spark_connect_proto::MAX_KEPT_DEPTH` levels.
            observed_metrics: if response.observed_metrics.is_empty() {
                Vec::new()
            } else {
                crate::on_grown_stack(|| response.observed_metrics.clone())
            },
            operation_id: None,
        };
        let (metrics, observed) = crate::dataframe::execution_metrics(py, &of_response)?;
        let bytes = PyBytes::new(py, &crate::encode_on_grown_stack(response));
        Ok(Some((bytes, metrics, observed)))
    }

    /// End the execution, releasing it on the server if it is still running.
    fn close(&mut self, py: Python<'_>) {
        self.finish(py);
    }

    /// The operation ID of the execution, which the client assigned to its request.
    #[getter]
    fn operation_id(&self) -> &str {
        &self.operation_id
    }
}

impl PyExecutePlanResponses {
    /// End the execution and drop it without the GIL: its release and the done call of its
    /// progress handlers, which take the GIL themselves.
    fn finish(&mut self, py: Python<'_>) {
        if let Some(execution) = self.execution.take() {
            py.detach(move || drop(execution));
        }
    }
}

/// The retry policy of `session`, as the keyword arguments of PySpark's `RetryPolicy`, the
/// `max_retry_exception_elapsed_time` of its client, in seconds, or None without a bound, and
/// `retries_failures`, whether a policy of the client retries the failures (see
/// [`RetryPolicy::retries_failures`]).
pub(crate) fn retry_policy<'py>(
    py: Python<'py>,
    session: &SparkSession,
) -> PyResult<Bound<'py, PyDict>> {
    let policy = session.retry_policy();
    let kwargs = PyDict::new(py);
    kwargs.set_item("max_retries", policy.max_retries)?;
    kwargs.set_item("initial_backoff", policy.initial_backoff_ms)?;
    kwargs.set_item("max_backoff", policy.max_backoff_ms)?;
    kwargs.set_item("backoff_multiplier", policy.backoff_multiplier)?;
    kwargs.set_item("jitter", policy.jitter_ms)?;
    kwargs.set_item("min_jitter_threshold", policy.min_jitter_threshold_ms)?;
    kwargs.set_item(
        "recognize_server_retry_delay",
        policy.recognize_server_retry_delay,
    )?;
    kwargs.set_item("max_server_retry_delay", policy.max_server_retry_delay_ms)?;
    kwargs.set_item(
        "max_retry_exception_elapsed_time",
        policy
            .max_retry_exception_elapsed_time_ms
            .map(|ms| ms as f64 / 1000.0),
    )?;
    kwargs.set_item("retries_failures", policy.retries_failures)?;
    Ok(kwargs)
}

/// The parameters of a retry policy as PySpark's `RetryPolicy` has them, which the transport
/// retries the RPCs of a session with (see [`RetryPolicy`]): the backoffs, the jitter and the
/// delays in milliseconds.
pub(crate) struct RetryParameters {
    pub(crate) max_retries: Option<u32>,
    pub(crate) initial_backoff: u64,
    pub(crate) max_backoff: Option<u64>,
    pub(crate) backoff_multiplier: f64,
    pub(crate) jitter: u64,
    pub(crate) min_jitter_threshold: u64,
    pub(crate) recognize_server_retry_delay: bool,
    pub(crate) max_server_retry_delay: Option<u64>,
    /// The bound of the retries of a `RetryException`, in seconds, or None without one.
    pub(crate) max_retry_exception_elapsed_time: Option<f64>,
    /// Whether the failures are retried, false when no policy of PySpark's client is a
    /// `DefaultPolicy`.
    pub(crate) retries_failures: bool,
}

/// What the client does when the retries of a call of the transport are spent, where PySpark's
/// `Retrying._wait` warns `[RETRIES_EXCEEDED]`: it warns as PySpark does, with
/// `_warn_retries_exceeded` of `pyspark.sql.connect.client.core`, on the thread of the call, and
/// the exception of the warning, of a filter of `"error"`, is the error of the call in place of
/// its failure, as PySpark raises it out of `_wait`. A call of a thread of the runtime, which no
/// caller waits for, warns on the delivery thread, as PySpark's warns from the thread that sends
/// it, and its failure stays its error. It logs to the logger of Spark Connect what PySpark's
/// `Retrying` logs of the retries of the calls, and the failure of a request that the transport
/// sent in the background as PySpark's sender logs it.
struct RetriesWarning;

/// Warn as PySpark's `Retrying._wait` warns once retries are spent, with `_warn_retries_exceeded`
/// of `pyspark.sql.connect.client.core`: the exception of the warning, of a filter of `"error"`.
fn warn_retries_exceeded(py: Python<'_>, exceeded: RetriesExceeded) -> PyResult<()> {
    py.import("pyspark.sql.connect.client.core")?.call_method1(
        "_warn_retries_exceeded",
        (exceeded == RetriesExceeded::RetryException,),
    )?;
    Ok(())
}

/// The levels of `logging` of the logs of the client.
pub(crate) const LOG_INFO: i32 = 20;
const LOG_ERROR: i32 = 40;

/// Whether the logger of Spark Connect (`pyspark.sql.connect.logging`), which is disabled unless
/// `SPARK_CONNECT_LOG_LEVEL` is set, writes a log of `level`.
pub(crate) fn connect_logger_enabled_for(py: Python<'_>, level: i32) -> PyResult<bool> {
    py.import("pyspark.sql.connect.logging")?
        .getattr("logger")?
        .call_method1("isEnabledFor", (level,))?
        .extract()
}

/// Log `message` at the level `level` to the logger of Spark Connect
/// (`pyspark.sql.connect.logging`), as PySpark's client logs it, on the thread of the caller when
/// it is a thread of Python, else on the delivery thread. The exception of the log, of a handler
/// or a filter of the logger, is the error of the call, as it goes out of PySpark's log; of a
/// request that the transport sends in the background, which no caller waits for, it is dropped,
/// but for the KeyboardInterrupt of a signal (see [`crate::deliver::keep_interrupt`]).
fn log_to_connect_logger(level: &'static str, message: String) -> Result<(), SparkError> {
    let log = move |py: Python<'_>| -> PyResult<()> {
        py.import("pyspark.sql.connect.logging")?
            .getattr("logger")?
            .call_method1(level, (message,))?;
        Ok(())
    };
    if crate::deliver::on_python_thread() || spark_connect_core::client_log::records_written_here()
    {
        return match Python::try_attach(log) {
            Some(Err(error)) => Err(crate::errors::callback_error(error)),
            _ => Ok(()),
        };
    }
    crate::deliver::run_dropping_errors(log);
    Ok(())
}

impl RetriesListener for RetriesWarning {
    fn retries_exceeded(&self, exceeded: RetriesExceeded) -> spark_connect_core::error::Result<()> {
        // On a thread of the runtime, of a request that no caller waits for, the warning is the
        // delivery thread's, and the failure stays the error of the request.
        if !crate::deliver::on_python_thread() {
            // The exception of the warning, of a filter of "error", is dropped, as PySpark's
            // senders in the background catch it (`except Exception`).
            crate::deliver::run(move |py| drop(warn_retries_exceeded(py, exceeded)));
            return Ok(());
        }
        Python::attach(|py| warn_retries_exceeded(py, exceeded))
            .map_err(crate::errors::callback_error)
    }

    fn caught(&self, error: SparkError) -> Option<SparkError> {
        // As PySpark's `except Exception`, which drops the exception of the warning, or of a
        // policy, but lets a `BaseException` only out, as a KeyboardInterrupt.
        if crate::errors::is_base_exception_only(&error) {
            return Some(error);
        }
        drop(crate::errors::callback_exception(&error));
        None
    }

    fn background_failure(
        &self,
        request: BackgroundRequest,
        exceeded: Option<RetriesExceeded>,
        failure: BackgroundFailure<'_>,
    ) {
        let failure = match failure {
            BackgroundFailure::Rpc(text) => Ok(text.to_string()),
            BackgroundFailure::Error(error) => Err(error),
        };
        crate::deliver::run(move |py| {
            // As PySpark's sender, which warns from its thread once the retries are spent, and
            // catches the failure, or the exception of the warning of a filter of "error", or of
            // a policy, which it logs as `{e}`.
            let error = match exceeded.map(|exceeded| warn_retries_exceeded(py, exceeded)) {
                Some(Err(raised)) => raised.value(py).to_string(),
                _ => match failure {
                    Ok(text) => text,
                    Err(error) => match crate::errors::callback_exception(&error) {
                        Some(raised) => raised.value(py).to_string(),
                        None => error.message(),
                    },
                },
            };
            let logged = match request {
                // `ReleaseExecute failed with exception: ...` of PySpark's iterator, and
                // `RemoveRemoteCachedRelation failed with exception: ...` of its finalizer.
                BackgroundRequest::ReleaseExecute => {
                    format!("ReleaseExecute failed with exception: {error}.")
                }
                BackgroundRequest::RemoveCachedRemoteRelation => {
                    format!("RemoveRemoteCachedRelation failed with exception: {error}.")
                }
            };
            if let Err(error) = py
                .import("pyspark.sql.connect.logging")
                .and_then(|logging| logging.getattr("logger"))
                .and_then(|logger| logger.call_method1("warning", (logged,)))
            {
                crate::deliver::keep_interrupt(py, &error);
            }
        });
    }

    fn retrying(&self, error: &str, wait_ms: i128, policy: &str) -> Result<(), SparkError> {
        if crate::client_log::silenced() {
            return Ok(());
        }
        log_to_connect_logger(
            "debug",
            format!("Got error: {error}. Will retry after {wait_ms} ms (policy: {policy})"),
        )
    }

    fn retrying_at_once(&self, error: &str) -> Result<(), SparkError> {
        if crate::client_log::silenced() {
            return Ok(());
        }
        log_to_connect_logger("debug", format!("Got error: {error}. Retrying."))
    }

    fn given_up(&self, error: &str) -> Result<(), SparkError> {
        if crate::client_log::silenced() {
            return Ok(());
        }
        log_to_connect_logger("debug", format!("Given up on retrying. error: {error}"))
    }

    fn server_retry_delay(&self, delay_ms: i128) -> Result<(), SparkError> {
        if crate::client_log::silenced() {
            return Ok(());
        }
        log_to_connect_logger(
            "debug",
            format!("The server has sent a retry delay of {delay_ms} ms."),
        )
    }

    fn sleep_long(&self, wait_ms: i128, watched: bool) -> LongSleep {
        // As PySpark's `Retrying`, which sleeps `wait_time / 1000`: the true division of the int of
        // the wait, which the f64 gives exactly up to 2^53 ms, and past it of the same error.
        sleep_of_python(wait_ms as f64 / 1000.0, watched)
    }

    fn rpc_error(&self, error: &spark_connect_core::error::GrpcioError) -> Delivery {
        // Of the execution that a KeyboardInterrupt interrupted, whose failure PySpark's client
        // does not read.
        if crate::client_log::silenced() {
            return None;
        }
        let error = error.clone();
        // A handler of the log may wait forever on the error (see `may_wait_forever`).
        let apart = may_wait_forever(&error);
        delivered(apart, move |py| {
            // The error of grpcio, of `pyspark.sql.connect.client.core`, which imports grpc, is
            // made only for a log that is written.
            if !connect_logger_enabled_for(py, LOG_ERROR)? {
                return Ok(());
            }
            py.import("pyspark.sql.connect.client.core")?
                .call_method1("_log_rpc_error", grpc_error_args(py, &error))?;
            Ok(())
        })
    }

    fn error_info(&self, text: &str) -> Delivery {
        if crate::client_log::silenced() {
            return None;
        }
        let message = format!("Received ErrorInfo: {text}");
        delivered(false, move |py| {
            py.import("pyspark.sql.connect.logging")?
                .getattr("logger")?
                .call_method1("debug", (message,))?;
            Ok(())
        })
    }

    fn raised_in_handling(&self, error: SparkError, handled: Handled<'_>) -> SparkError {
        // As the log is in an `except` clause that the `except KeyboardInterrupt` of PySpark's
        // execution does not catch, as its `except Exception` that converts a failure: a
        // KeyboardInterrupt of it interrupts nothing (see `crate::interrupt::interruptible`).
        let error = crate::errors::raised_in_handling(error);
        match handled {
            // As PySpark's `_handle_rpc_error`, which logs while the `except` of the call handles
            // grpcio's `RpcError`: the exception of a log has it as its context.
            Handled::Rpc(failure) => Python::attach(|py| {
                if let Some(context) = rpc_error_of(py, failure) {
                    crate::errors::set_callback_context(py, &error, context);
                }
            }),
            // As the `except Exception as e` of PySpark's lookup of the plan compression, whose
            // log of the exception that it caught is in it: the exception of the log has that
            // exception, the argument of the log, as its context.
            Handled::Error(caught) => Python::attach(|py| {
                let context = crate::errors::exception_of(py, caught);
                crate::errors::set_callback_context(py, &error, context);
            }),
        }
        error
    }
}

/// The sleep of `seconds` before a retry of Python's `time.sleep`, as PySpark's `Retrying` sleeps
/// it, which raises its error at once of a wait that it cannot hold (see
/// [`RetriesListener::sleep_long`]): on the thread of the call, a thread of Python, whose wait is
/// `watched` when a signal ends it; of a request that no caller waits for, on a thread of the
/// runtime, as PySpark's sender sleeps it on a thread of Python, on a thread of its own, as the
/// sleep may last decades, which would hold the delivery thread meanwhile.
fn sleep_of_python(seconds: f64, watched: bool) -> LongSleep {
    let sleep = move |py: Python<'_>| -> PyResult<()> {
        py.import("time")?.call_method1("sleep", (seconds,))?;
        Ok(())
    };
    if !crate::deliver::on_python_thread() {
        return Box::pin(crate::deliver::call_apart(move |py| {
            sleep(py).map_err(crate::errors::callback_error)
        }));
    }
    let slept = Python::attach(|py| match sleep(py) {
        Ok(()) => Some(Ok(())),
        // The KeyboardInterrupt of a signal that came while it slept is the watch's, which ends the
        // wait of the core as of a signal that came while it waited: the execution in flight is
        // interrupted, as PySpark's `except KeyboardInterrupt` interrupts it.
        Err(error) if watched && crate::deliver::keep_interrupt(py, &error) => None,
        Err(error) => Some(Err(crate::errors::callback_error(error))),
    });
    Box::pin(std::future::ready(slept))
}

/// Run `log`, a log of a call, as PySpark's client logs it on the thread of the call: at once on a
/// thread of Python, or on a thread whose records are written on it (see
/// `spark_connect_core::client_log::records_written_here`), and else on the delivery thread, or on
/// a thread of its own (`apart`, see [`crate::deliver::call_apart`]), of the delivery that the call
/// awaits, so that the log precedes what the call does next, as PySpark's client logs on the
/// thread of the call before it goes on. The exception of the log, of a handler or a filter of the
/// logger, is the error of the delivery, which fails the call at the log, as the exception of
/// PySpark's log goes out of its call (see [`crate::errors::callback_error`]).
pub(crate) fn delivered(
    apart: bool,
    log: impl FnOnce(Python<'_>) -> PyResult<()> + Send + 'static,
) -> Delivery {
    let log = move |py: Python<'_>| log(py).map_err(crate::errors::callback_error);
    if crate::deliver::on_python_thread() || spark_connect_core::client_log::records_written_here()
    {
        // An interpreter that exits writes nothing more.
        return match Python::try_attach(log) {
            Some(Err(error)) => Some(Box::pin(std::future::ready(Err(error)))),
            _ => None,
        };
    }
    Some(if apart {
        Box::pin(async move { crate::deliver::call_apart(log).await.unwrap_or(Ok(())) })
    } else {
        Box::pin(async move { crate::deliver::call(log).await.unwrap_or(Ok(())) })
    })
}

/// grpcio's error of the failed RPC `failure`, as the call of PySpark's client raises it
/// (`_rpc_error` of `pyspark.sql.connect.client.core`): the context of an exception raised while
/// PySpark's client handles it. None of one that Python code may wait forever on (see
/// [`may_wait_forever`]), which nothing would read then.
pub(crate) fn rpc_error_of(
    py: Python<'_>,
    failure: &spark_connect_core::error::GrpcioError,
) -> Option<PyErr> {
    if may_wait_forever(failure) {
        return None;
    }
    py.import("pyspark.sql.connect.client.core")
        .and_then(|core| core.call_method1("_rpc_error", grpc_error_args(py, failure)))
        .ok()
        .map(PyErr::from_value)
}

/// Whether Python code that is given grpcio's `error` may wait forever on it: grpcio's error of a
/// stream of responses, `_MultiThreadedRendezvous`, of no `debug_error_string`, as of a response
/// that the client failed to decode, whose `debug_error_string()` waits for one, which nothing
/// sets, as grpcio's does. Such code runs apart from the delivery thread, where it would hold
/// every later delivery of the process, as it holds only its own call in grpcio (see
/// [`crate::deliver::call_apart`]).
fn may_wait_forever(error: &spark_connect_core::error::GrpcioError) -> bool {
    error.call == GrpcCall::ServerStreaming && error.debug_error_string.is_none()
}

/// The arguments of `_grpc_error` of `pyspark.sql.connect.client.core` of grpcio's `error`:
/// whether its call is of a stream of responses, its code, its details, its `debug_error_string`,
/// None when it has none, its trailing metadata, and its text, of which an error of no grpcio is
/// made.
#[allow(clippy::type_complexity)]
pub(crate) fn grpc_error_args<'py>(
    py: Python<'py>,
    error: &spark_connect_core::error::GrpcioError,
) -> (
    bool,
    i32,
    String,
    Option<String>,
    Vec<(String, Bound<'py, PyBytes>)>,
    String,
) {
    (
        error.call == GrpcCall::ServerStreaming,
        error.code as i32,
        error.details.clone(),
        error.debug_error_string.clone(),
        error
            .trailing_metadata
            .iter()
            .map(|(key, value)| (key.clone(), PyBytes::new(py, value)))
            .collect(),
        error.text.clone(),
    )
}

/// The retries of the calls of a session whose retry policies the transport does not retry with
/// itself, of their own methods, which PySpark's `Retrying` decides, as PySpark's client asks it:
/// `_CallRetries` of `pyspark.sql.connect.client.core`, a `Retrying` of the policies of each call,
/// of grpcio's `RpcError` of each failed attempt. The Python of a call of a thread of Python runs
/// on that thread, the `to_state` of the policies as the call starts; that of a request that the
/// transport sends in the background runs on the delivery thread, from its first failure on.
pub(crate) struct PolicyRetries {
    /// The list of the policies, which a thread of the runtime clones without the GIL.
    policies: std::sync::Arc<Py<PyAny>>,
}

/// The retries of one call of [`PolicyRetries`]: its `_CallRetries`, once it is made, or the error
/// of its making, which fails the call at its first failure.
struct PolicyCallRetries {
    policies: std::sync::Arc<Py<PyAny>>,
    streaming: bool,
    retries: Option<PyResult<Py<PyAny>>>,
}

/// The least wait before a retry of decades, in seconds, of 2^30 s (34 years), which is slept of
/// Python's `time.sleep` (see [`sleep_of_python`]).
const LONG_WAIT_SECONDS: f64 = (1u64 << 30) as f64;

/// The `_CallRetries` of `policies`, a `Retrying` of their states.
fn call_retries(py: Python<'_>, policies: &Py<PyAny>) -> PyResult<Py<PyAny>> {
    Ok(py
        .import("pyspark.sql.connect.client.core")?
        .getattr("_CallRetries")?
        .call1((policies.clone_ref(py),))?
        .unbind())
}

impl RetryDecider for PolicyRetries {
    fn start(&self, call: GrpcCall) -> Box<dyn CallRetries> {
        // The `to_state` of the policies, on the thread of the call, as it starts; a thread of
        // the runtime does not wait for the GIL, and makes them at its first failure. Of a request
        // that a drop sends, as the release of a stream that a frame that CPython unwinds drops,
        // of the exception that propagates kept (see `crate::deliver::preserving_pending_exception`).
        let retries = if crate::deliver::on_python_thread() {
            Python::try_attach(|py| {
                crate::deliver::preserving_pending_exception(py, |py| {
                    call_retries(py, &self.policies)
                })
            })
        } else {
            None
        };
        Box::new(PolicyCallRetries {
            policies: std::sync::Arc::clone(&self.policies),
            streaming: call == GrpcCall::ServerStreaming,
            retries,
        })
    }
}

impl CallRetries for PolicyCallRetries {
    fn decide<'a>(
        &'a mut self,
        status: &'a tonic::Status,
    ) -> std::pin::Pin<Box<dyn std::future::Future<Output = Decision> + Send + 'a>> {
        let streaming = self.streaming;
        let call = if streaming {
            GrpcCall::ServerStreaming
        } else {
            GrpcCall::Unary
        };
        let error = spark_connect_core::error::grpcio_error(status, call);
        let apart = may_wait_forever(&error);
        let policies = std::sync::Arc::clone(&self.policies);
        let retries = self.retries.take();
        Box::pin(async move {
            let decide_of = move |py: Python<'_>| {
                let retries = match retries {
                    Some(retries) => retries,
                    None => call_retries(py, &policies),
                };
                let decided = match &retries {
                    Ok(retries) => decide(py, retries, streaming, &error),
                    Err(failed) => Err(failed.clone_ref(py)),
                };
                let decision = match decided {
                    // A wait that `time.sleep` refuses, negative or NaN, or one of decades, which
                    // is slept of it, as PySpark's `Retrying` sleeps it.
                    Ok(Some(seconds)) if !(0.0..LONG_WAIT_SECONDS).contains(&seconds) => {
                        Err(seconds)
                    }
                    Ok(Some(seconds)) => Ok(Decision::Retry(
                        std::time::Duration::try_from_secs_f64(seconds)
                            .unwrap_or(std::time::Duration::MAX),
                    )),
                    Ok(None) => Ok(Decision::GiveUp),
                    Err(raised) => Ok(Decision::Failed(Box::new(crate::errors::callback_error(
                        raised,
                    )))),
                };
                (decision, retries)
            };
            // A policy may wait forever on the error (see `may_wait_forever`).
            let decided = if apart {
                crate::deliver::call_apart(decide_of).await
            } else {
                crate::deliver::call(decide_of).await
            };
            match decided {
                Some((decision, retries)) => {
                    self.retries = Some(retries);
                    match decision {
                        Ok(decision) => decision,
                        Err(seconds) => {
                            let watched = spark_connect_core::runtime::in_watched_wait();
                            match sleep_of_python(seconds, watched).await {
                                Some(Ok(())) => Decision::Retry(std::time::Duration::ZERO),
                                Some(Err(error)) => Decision::Failed(Box::new(error)),
                                // The watch ends the wait, of the signal that ended the sleep.
                                None => Decision::Retry(std::time::Duration::MAX),
                            }
                        }
                    }
                }
                // The interpreter exits: the failure is the error of the call.
                None => Decision::GiveUp,
            }
        })
    }
}

/// What the `_CallRetries` `retries` decides of the attempt that failed of grpcio's `error`: the
/// wait in seconds before the retry, or None when the failure is the error of the call.
fn decide(
    py: Python<'_>,
    retries: &Py<PyAny>,
    streaming: bool,
    error: &spark_connect_core::error::GrpcioError,
) -> PyResult<Option<f64>> {
    let metadata: Vec<(&str, Bound<'_, PyBytes>)> = error
        .trailing_metadata
        .iter()
        .map(|(key, value)| (key.as_str(), PyBytes::new(py, value)))
        .collect();
    retries
        .bind(py)
        .call_method1(
            "decide",
            (
                streaming,
                error.code as i32,
                error.details.as_str(),
                error.debug_error_string.as_deref(),
                metadata,
                error.text.as_str(),
            ),
        )?
        .extract()
}

/// Make PySpark's `Retrying` of `policies` decide the retries of the calls of `session` from now
/// on, or the policy of the transport with None (see [`PolicyRetries`]).
pub(crate) fn set_retry_decider(session: &SparkSession, policies: Option<Bound<'_, PyList>>) {
    let decider = policies.map(|policies| {
        std::sync::Arc::new(PolicyRetries {
            policies: std::sync::Arc::new(policies.into_any().unbind()),
        }) as std::sync::Arc<dyn RetryDecider>
    });
    session.set_retry_decider(decider);
}

/// Make the client warn as PySpark's when the retries of a call are spent (see
/// [`RetriesWarning`]).
pub(crate) fn warn_of_spent_retries() {
    spark_connect_core::retries::set_listener(Some(std::sync::Arc::new(RetriesWarning)));
}

/// Make the client log what its channels log as grpcio does, as PySpark's client shows it: the
/// line of gRPC's C core on stderr, written at once by the thread that logs it, of the channel or
/// of its connections, as gRPC's core writes it (see [`grpc_log_line`]). As grpcio reads
/// `GRPC_VERBOSITY` when it initializes, the variable as the extension loads decides what is
/// logged: nothing for `NONE`, the errors only for `ERROR`, and the information too otherwise,
/// such as the failure of a TLS handshake, as grpcio does by default.
pub(crate) fn log_grpc_errors() {
    use spark_connect_core::grpc_log::Severity;
    let verbosity = std::env::var("GRPC_VERBOSITY").unwrap_or_default();
    if verbosity.eq_ignore_ascii_case("NONE") {
        return;
    }
    let errors_only = verbosity.eq_ignore_ascii_case("ERROR");
    spark_connect_core::grpc_log::set_listener(Some(std::sync::Arc::new(
        move |logged: &spark_connect_core::grpc_log::Logged| {
            if errors_only && logged.severity != Severity::Error {
                return;
            }
            let location = logged.location;
            let file = location
                .file()
                .rsplit(['/', '\\'])
                .next()
                .unwrap_or_default()
                .to_string();
            write_grpc_log(
                logged.severity.letter(),
                file,
                location.line(),
                logged.message.clone(),
            );
        },
    )));
}

/// Write the line of gRPC's core of `message`, of the severity `severity` and of the source
/// `file` and `line`, to stderr, at once, as gRPC's core writes its log, whatever the thread: to
/// the descriptor itself, without the lock of Rust's stderr, which a thread of the runtime that
/// logs while another thread forks would hold forever in the child of the fork.
#[cfg(unix)]
fn write_grpc_log(severity: char, file: String, line: u32, message: String) {
    let entry = grpc_log_line(severity, &file, line, &message);
    let mut unwritten = entry.as_bytes();
    while !unwritten.is_empty() {
        // SAFETY: writes the bytes of `unwritten`, which it is given the length of, to stderr.
        let written = unsafe { libc::write(2, unwritten.as_ptr().cast(), unwritten.len()) };
        match usize::try_from(written) {
            Ok(written) if written > 0 => unwritten = &unwritten[written..],
            _ if std::io::Error::last_os_error().kind() == std::io::ErrorKind::Interrupted => {}
            // As gRPC's core, which goes on when stderr is closed.
            _ => break,
        }
    }
}

/// As on Unix, by the Python of the client, which has the local time.
#[cfg(not(unix))]
fn write_grpc_log(severity: char, file: String, line: u32, message: String) {
    // An error that cannot be logged is dropped, as gRPC's core goes on.
    crate::deliver::run_dropping_errors(move |py| {
        py.import("pyspark.sql.connect.client.core")?.call_method1(
            "_log_grpc_error",
            (file, line, message, severity.to_string()),
        )?;
        Ok(())
    });
}

/// The line of gRPC's core of `message`: `E<month><day> <time> <thread> <file>:<line>]
/// <message>`, of the local time to the microsecond and the native ID of the calling thread, as
/// `threading.get_native_id()` gives it, where gRPC's names the source of its own call.
#[cfg(unix)]
fn grpc_log_line(severity: char, file: &str, line: u32, message: &str) -> String {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let seconds = now.as_secs() as libc::time_t;
    // SAFETY: `localtime_r` writes the local time of `seconds` to the `tm` it is given, which is
    // read only once it returned it.
    let tm = unsafe {
        let mut tm = std::mem::MaybeUninit::<libc::tm>::zeroed();
        libc::localtime_r(&seconds, tm.as_mut_ptr());
        tm.assume_init()
    };
    format!(
        "{severity}{:02}{:02} {:02}:{:02}:{:02}.{:06} {:7} {file}:{line}] {message}\n",
        tm.tm_mon + 1,
        tm.tm_mday,
        tm.tm_hour,
        tm.tm_min,
        tm.tm_sec,
        now.subsec_micros(),
        native_thread_id(),
    )
}

/// The native ID of the calling thread, as `threading.get_native_id()` gives it: of the system
/// call, as the wheels link a glibc older than its `gettid` wrapper (2.30).
#[cfg(target_os = "linux")]
fn native_thread_id() -> u64 {
    // SAFETY: the `gettid` system call takes no argument and cannot fail.
    unsafe { libc::syscall(libc::SYS_gettid) as u64 }
}

#[cfg(all(unix, not(target_os = "linux")))]
fn native_thread_id() -> u64 {
    let mut id = 0u64;
    // SAFETY: writes the ID of the calling thread to `id`.
    #[cfg(target_os = "macos")]
    unsafe {
        libc::pthread_threadid_np(0, &mut id);
    }
    #[cfg(not(target_os = "macos"))]
    {
        // SAFETY: always safe to call.
        id = unsafe { libc::pthread_self() } as u64;
    }
    id
}

/// Set the retry policy of `session` to `parameters`, from the calls that start from now on.
pub(crate) fn set_retry_policy(session: &SparkSession, parameters: RetryParameters) {
    session.set_retry_policy(RetryPolicy {
        max_retries: parameters.max_retries,
        initial_backoff_ms: parameters.initial_backoff,
        max_backoff_ms: parameters.max_backoff,
        backoff_multiplier: parameters.backoff_multiplier,
        jitter_ms: parameters.jitter,
        min_jitter_threshold_ms: parameters.min_jitter_threshold,
        recognize_server_retry_delay: parameters.recognize_server_retry_delay,
        max_server_retry_delay_ms: parameters.max_server_retry_delay,
        max_retry_exception_elapsed_time_ms: parameters
            .max_retry_exception_elapsed_time
            .map(|seconds| (seconds.max(0.0) * 1000.0) as u64),
        retries_failures: parameters.retries_failures,
    });
}

/// The RPC deadlines of `deadlines`, the deadlines in seconds by the names of the fields of
/// PySpark's `RpcDeadlines`, None for no deadline.
pub(crate) fn rpc_deadlines(deadlines: &Bound<'_, PyDict>) -> PyResult<RpcDeadlines> {
    let of = |name: &str| -> PyResult<Option<std::time::Duration>> {
        let seconds: Option<f64> = match deadlines.get_item(name)? {
            Some(value) => value.extract()?,
            None => None,
        };
        seconds
            .map(|seconds| {
                std::time::Duration::try_from_secs_f64(seconds)
                    .map_err(|error| PyValueError::new_err(format!("{name}: {error}")))
            })
            .transpose()
    };
    Ok(RpcDeadlines {
        reattachable_execute_plan: of("reattachable_execute_plan")?,
        reattach_execute: of("reattach_execute")?,
        analyze_plan: of("analyze_plan")?,
        add_artifacts: of("add_artifacts")?,
        config: of("config")?,
        interrupt: of("interrupt")?,
        release_session: of("release_session")?,
        artifact_status: of("artifact_status")?,
        fetch_error_details: of("fetch_error_details")?,
        clone_session: of("clone_session")?,
        get_status: of("get_status")?,
    })
}

/// The result chunking of the executions of a client of `allow_arrow_batch_chunking` and
/// `preferred_arrow_chunk_size`, as PySpark's `SparkConnectClient` takes them.
pub(crate) fn result_chunking(
    allow_arrow_batch_chunking: bool,
    preferred_arrow_chunk_size: Option<i64>,
) -> ResultChunking {
    ResultChunking {
        allow_arrow_batch_chunking,
        preferred_arrow_chunk_size,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_the_error_of_a_stream_of_no_debug_string_may_wait_forever() {
        // grpcio's `debug_error_string()` of a stream waits for a debug string that a response
        // that does not decode never gets; a unary error, or one of a debug string, does not.
        let error =
            |call, debug_error_string: Option<&str>| spark_connect_core::error::GrpcioError {
                call,
                code: tonic::Code::Internal,
                details: "Exception deserializing response!".to_string(),
                debug_error_string: debug_error_string.map(str::to_string),
                trailing_metadata: Vec::new(),
                text: String::new(),
            };
        assert!(may_wait_forever(&error(GrpcCall::ServerStreaming, None)));
        assert!(!may_wait_forever(&error(GrpcCall::Unary, None)));
        assert!(!may_wait_forever(&error(
            GrpcCall::ServerStreaming,
            Some("INTERNAL:boom")
        )));
    }
}
