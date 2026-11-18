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

//! The requests of PySpark Connect's `SparkConnectClient` (`pyspark.sql.connect.client`) that it
//! sends of the messages it is given, such as the plans of `to_table` and `schema`, over the Rust
//! transport of its session: the analyses, the configuration requests and the executions, of the
//! bytes of the messages of `pyspark.sql.connect.proto` that the client builds as PySpark's does,
//! and the settings of the client that the transport honors, such as its retry policy.

use prost::Message;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict, PyList};
use spark_connect::dataframe::PlanExecution;
use spark_connect::session::{ExecutionInfo, SparkSession};
use spark_connect_core::client::{ResultChunking, RpcDeadlines};
use spark_connect_core::error::SparkError;
use spark_connect_core::retries::{RetriesExceeded, RetriesListener, RetryPolicy};
use spark_connect_proto as proto;

use crate::interrupt::interruptible;

/// The message of `bytes`, decoded on a stack that holds its levels, as a plan of any depth may be.
fn decode<M: Message + Default + Send>(bytes: &[u8]) -> PyResult<M> {
    crate::on_grown_stack(|| M::decode(bytes))
        .map_err(|error| PyValueError::new_err(error.to_string()))
}

/// The bytes of the `AnalyzePlanResponse` of the server to `request`, the bytes of an
/// `AnalyzePlanRequest` that the client built, as PySpark's `SparkConnectClient._analyze` sends
/// it, of the session `session`.
pub(crate) fn analyze_plan<'py>(
    py: Python<'py>,
    session: &SparkSession,
    request: &[u8],
) -> PyResult<Bound<'py, PyBytes>> {
    let request: proto::AnalyzePlanRequest = decode(request)?;
    let response = interruptible(py, session, || {
        crate::on_grown_stack(|| session.analyze_plan_proto(request)).map_err(Box::new)
    })?;
    Ok(PyBytes::new(py, &crate::encode_on_grown_stack(response)))
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
pub(crate) fn execute_plan(
    py: Python<'_>,
    session: &SparkSession,
    plan: &[u8],
    observations: Vec<(String, Bound<'_, PyAny>)>,
    reported: bool,
) -> PyResult<PyExecutePlanResponses> {
    let plan: proto::Plan = decode(plan)?;
    let observations = observations
        .into_iter()
        .map(|(name, observation)| {
            let observation = observation.cast::<crate::observation::PyObservation>()?;
            Ok((name, observation.borrow().inner.clone()))
        })
        .collect::<PyResult<Vec<_>>>()?;
    let execution = interruptible(py, session, || {
        crate::on_grown_stack(|| session.execute_plan_proto(plan, observations, reported))
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
#[pyclass(name = "ExecutePlanResponses", module = "pyspark.sql.connect.client")]
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
        let next = interruptible(py, &session, || execution.next_response().map_err(Box::new));
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
            observed_metrics: response.observed_metrics.clone(),
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
/// `_warn_retries_exceeded` of `pyspark.sql.connect.client`, on the thread of the call, and the
/// exception of the warning, of a filter of `"error"`, is the error of the call in place of its
/// failure, as PySpark raises it out of `_wait`.
struct RetriesWarning;

impl RetriesListener for RetriesWarning {
    fn retries_exceeded(&self, exceeded: RetriesExceeded) -> spark_connect_core::error::Result<()> {
        // Only on a thread of Python, as that of a call of the client is: a thread of the runtime,
        // which runs what no caller waits for, has no thread state of Python, and does not wait
        // for the GIL.
        // SAFETY: reads the thread state that the interpreter keeps for this thread, without the
        // GIL, which it does not need.
        if unsafe { pyo3::ffi::PyGILState_GetThisThreadState() }.is_null() {
            return Ok(());
        }
        Python::attach(|py| {
            py.import("pyspark.sql.connect.client")?.call_method1(
                "_warn_retries_exceeded",
                (exceeded == RetriesExceeded::RetryException,),
            )?;
            Ok(())
        })
        .map_err(crate::errors::callback_error)
    }

    fn caught(&self, error: SparkError) {
        // As PySpark's `except Exception`, which drops the exception of the warning.
        drop(crate::errors::callback_exception(&error));
    }
}

/// Make the client warn as PySpark's when the retries of a call are spent (see
/// [`RetriesWarning`]).
pub(crate) fn warn_of_spent_retries() {
    spark_connect_core::retries::set_listener(Some(std::sync::Arc::new(RetriesWarning)));
}

/// Make the client log the errors of its channels that grpcio logs, as PySpark's client shows
/// them: the line of gRPC's C core on stderr, which `_log_grpc_error` of
/// `pyspark.sql.connect.client` writes, on the thread of the channel. Unless `GRPC_VERBOSITY` is
/// `NONE` when the extension loads, as grpcio reads it when it initializes and then logs no error.
pub(crate) fn log_grpc_errors() {
    let silenced = std::env::var("GRPC_VERBOSITY")
        .is_ok_and(|verbosity| verbosity.eq_ignore_ascii_case("NONE"));
    if silenced {
        return;
    }
    spark_connect_core::grpc_log::set_listener(Some(std::sync::Arc::new(
        |logged: &spark_connect_core::grpc_log::Logged| {
            // Only on a thread of Python, as that of a channel of a session is: a thread of the
            // runtime has no thread state of Python, and does not wait for the GIL.
            // SAFETY: reads the thread state that the interpreter keeps for this thread, without
            // the GIL, which it does not need.
            if unsafe { pyo3::ffi::PyGILState_GetThisThreadState() }.is_null() {
                return;
            }
            let location = logged.location;
            let file = location
                .file()
                .rsplit(['/', '\\'])
                .next()
                .unwrap_or_default();
            Python::attach(|py| {
                // An error that cannot be logged is dropped, as gRPC's core goes on.
                let _ = py.import("pyspark.sql.connect.client").and_then(|client| {
                    client.call_method1(
                        "_log_grpc_error",
                        (file, location.line(), logged.message.as_str()),
                    )
                });
            });
        },
    )));
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
