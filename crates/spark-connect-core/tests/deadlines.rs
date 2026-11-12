// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

//! Per-RPC deadlines that expire before the response headers, and the reattach to an
//! execution that the server does not know. No Spark server required: a loopback tonic
//! service stalls its calls before they send headers, as a stalled connection does.

mod common;

use std::collections::{HashSet, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use common::{FakeServer, FakeService};
use futures::{stream, Stream};
use spark_connect_core::{ChannelBuilder, RetryPolicy, RpcDeadlines, SparkConnectClient};
use spark_connect_proto::execute_plan_response::{ResponseType, ResultComplete};
use spark_connect_proto::release_execute_request::Release;
use spark_connect_proto::{
    ConfigRequest, ConfigResponse, ExecutePlanRequest, ExecutePlanResponse, ReattachExecuteRequest,
    ReleaseExecuteRequest, ReleaseExecuteResponse, UserContext,
};
use tonic::codegen::BoxFuture;
use tonic::metadata::MetadataMap;
use tonic::server::{ServerStreamingService, UnaryService};
use tonic::{Code, Request, Response, Status};
use tonic_prost::ProstCodec;

/// The per-RPC deadline of the tests: short, so that the stalls end quickly.
const DEADLINE: Duration = Duration::from_millis(200);
/// Bound of every RPC of a test, so that a call that waits forever fails the test.
const WAIT: Duration = Duration::from_secs(5);
const OPERATION: &str = "00000000-0000-4000-8000-000000000002";
type Responses = Pin<Box<dyn Stream<Item = Result<ExecutePlanResponse, Status>> + Send>>;

async fn bounded<T>(future: impl Future<Output = T>) -> T {
    tokio::time::timeout(WAIT, future)
        .await
        .expect("test RPC waited past the bound")
}

/// What the server does with one call: stall before its headers, fail the call, or stream
/// these items (an error item ends the stream with that status).
enum Step {
    Stall,
    Fail(Status),
    Stream(Vec<Result<ExecutePlanResponse, Status>>),
}

fn operation_not_found() -> Status {
    // As the server reports an execution it never started: an INTERNAL status whose message
    // carries the error class.
    Status::internal(format!(
        "[INVALID_HANDLE.OPERATION_NOT_FOUND] Operation not found: {OPERATION}"
    ))
}

fn session_not_found() -> Status {
    // As the server reports a session it does not know, with the same shape.
    Status::internal(
        "[INVALID_HANDLE.SESSION_NOT_FOUND] Session not found: 00000000-0000-4000-8000-000000000001",
    )
}

#[derive(Clone)]
struct CapturedRelease {
    request: ReleaseExecuteRequest,
}

struct State {
    /// What each ExecutePlan call does, in order. When none is left, it completes the result.
    execute_script: Mutex<VecDeque<Step>>,
    /// What each ReattachExecute call does, in order. When none is left, it completes the
    /// result of an execution that the server knows, and fails with OPERATION_NOT_FOUND for
    /// one it does not.
    reattach_script: Mutex<VecDeque<Step>>,
    /// The operations that the server started, that is, whose ExecutePlan streamed.
    known: Mutex<HashSet<String>>,
    /// The `grpc-timeout` of each call, in order.
    timeouts: Mutex<Vec<(&'static str, Option<String>)>>,
    executes: Mutex<usize>,
    reattaches: Mutex<Vec<ReattachExecuteRequest>>,
    /// When each ReattachExecute call arrived, in order.
    reattached_at: Mutex<Vec<Instant>>,
    releases: Mutex<Vec<CapturedRelease>>,
}

impl State {
    fn new(execute: Vec<Step>, reattach: Vec<Step>) -> Self {
        State {
            execute_script: Mutex::new(execute.into()),
            reattach_script: Mutex::new(reattach.into()),
            known: Mutex::new(HashSet::new()),
            timeouts: Mutex::new(Vec::new()),
            executes: Mutex::new(0),
            reattaches: Mutex::new(Vec::new()),
            reattached_at: Mutex::new(Vec::new()),
            releases: Mutex::new(Vec::new()),
        }
    }

    fn record_timeout(&self, call: &'static str, metadata: &MetadataMap) {
        let timeout = metadata
            .get("grpc-timeout")
            .map(|value| value.to_str().unwrap().to_string());
        self.timeouts.lock().unwrap().push((call, timeout));
    }

    fn release_all_count(&self) -> usize {
        self.releases
            .lock()
            .unwrap()
            .iter()
            .filter(|r| matches!(r.request.release, Some(Release::ReleaseAll(_))))
            .count()
    }
}

struct Fixture {
    state: Arc<State>,
    _server: FakeServer,
    client: SparkConnectClient,
}

impl Fixture {
    /// A client with `deadline` for the calls of a reattachable execution and for Config, and
    /// the given retry policy.
    async fn start(execute: Vec<Step>, reattach: Vec<Step>, retry_policy: RetryPolicy) -> Self {
        let state = Arc::new(State::new(execute, reattach));
        let server = service(state.clone()).serve().await;
        let address = server.address;
        let builder = ChannelBuilder::parse(&format!(
            "sc://localhost:{}/;user_id=transport-user;token=fixture-token;session_id=00000000-0000-4000-8000-000000000001",
            address.port()
        ))
        .unwrap();
        let mut client = bounded(SparkConnectClient::connect(&builder))
            .await
            .unwrap();
        client.set_retry_policy(retry_policy);
        client.set_rpc_deadlines(RpcDeadlines {
            reattachable_execute_plan: Some(DEADLINE),
            reattach_execute: Some(DEADLINE),
            config: Some(DEADLINE),
            ..RpcDeadlines::default()
        });
        Self {
            state,
            _server: server,
            client,
        }
    }

    /// Waits until the release of the whole execution is sent once, and checks that it is not
    /// sent again.
    async fn assert_release_all_count(&self, expected: usize) {
        let state = self.state.clone();
        bounded(async {
            while state.release_all_count() < expected {
                tokio::time::sleep(Duration::from_millis(5)).await;
            }
        })
        .await;
        tokio::time::sleep(Duration::from_millis(100)).await;
        assert_eq!(self.state.release_all_count(), expected);
    }
}

fn execute_request() -> ExecutePlanRequest {
    ExecutePlanRequest {
        session_id: "00000000-0000-4000-8000-000000000001".into(),
        user_context: Some(UserContext {
            user_id: "request-user".into(),
            ..Default::default()
        }),
        operation_id: Some(OPERATION.into()),
        client_type: Some("deadline-regression".into()),
        ..Default::default()
    }
}

fn response(id: &str, complete: bool) -> ExecutePlanResponse {
    ExecutePlanResponse {
        // The server echoes the session ID of the request.
        session_id: execute_request().session_id,
        response_id: id.into(),
        response_type: complete.then_some(ResponseType::ResultComplete(ResultComplete {})),
        ..Default::default()
    }
}

/// A stream of `items`, as the server sends them.
fn streamed(items: Vec<ExecutePlanResponse>) -> Step {
    Step::Stream(items.into_iter().map(Ok).collect())
}

/// The retry policy of the tests: as the cleanup tests, with short backoffs.
fn quick_retries() -> RetryPolicy {
    RetryPolicy {
        max_retries: Some(2),
        initial_backoff_ms: 1,
        max_backoff_ms: Some(2),
        ..RetryPolicy::default()
    }
}

/// Drains `stream`, returning the ids of its responses, or the error that ends it.
async fn drain(
    stream: &mut spark_connect_core::client::ReattachableResponseStream,
) -> Result<Vec<String>, spark_connect_core::SparkError> {
    let mut ids = Vec::new();
    while let Some(response) = bounded(stream.message()).await? {
        ids.push(response.response_id);
    }
    Ok(ids)
}

#[tokio::test]
async fn stalled_first_execute_plan_reattaches_then_resends_the_execution() {
    // As the reference: the first ExecutePlan sends no headers, so its deadline passes. The
    // execution is reattached, and as the server does not know it, it is sent again, and its
    // results are returned.
    let fixture = Fixture::start(
        vec![
            Step::Stall,
            streamed(vec![response("r1", false), response("r2", true)]),
        ],
        vec![],
        quick_retries(),
    )
    .await;
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    let ids = drain(&mut stream).await.unwrap();
    assert_eq!(ids, ["r1", "r2"]);

    assert_eq!(*fixture.state.executes.lock().unwrap(), 2);
    let reattaches = fixture.state.reattaches.lock().unwrap().clone();
    assert_eq!(reattaches.len(), 1);
    assert_eq!(reattaches[0].last_response_id, None);
    let timeouts = fixture.state.timeouts.lock().unwrap().clone();
    let calls: Vec<_> = timeouts.iter().map(|(call, _)| *call).collect();
    assert_eq!(calls, ["ExecutePlan", "ReattachExecute", "ExecutePlan"]);
    assert!(
        timeouts.iter().all(|(_, timeout)| timeout.is_some()),
        "{timeouts:?}"
    );
    fixture.assert_release_all_count(1).await;
}

#[tokio::test]
async fn stalled_reattach_is_reattached_again() {
    // As the reference: a ReattachExecute that sends no headers past its deadline is retried,
    // the execution going on on the server, and the results after the last response read are
    // returned.
    let fixture = Fixture::start(
        vec![streamed(vec![response("r1", false)])],
        vec![Step::Stall, streamed(vec![response("r2", true)])],
        quick_retries(),
    )
    .await;
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    let ids = drain(&mut stream).await.unwrap();
    assert_eq!(ids, ["r1", "r2"]);

    let reattaches = fixture.state.reattaches.lock().unwrap().clone();
    let resumed: Vec<_> = reattaches
        .iter()
        .map(|r| r.last_response_id.clone())
        .collect();
    assert_eq!(resumed, [Some("r1".to_string()), Some("r1".to_string())]);
    assert_eq!(*fixture.state.executes.lock().unwrap(), 1);
    fixture.assert_release_all_count(1).await;
}

#[tokio::test]
async fn unary_call_past_its_deadline_is_deadline_exceeded() {
    // As the reference: a unary call stalled past its deadline raises DEADLINE_EXCEEDED, with
    // the message about the per-RPC timeouts.
    let fixture = Fixture::start(vec![], vec![], quick_retries()).await;
    let error = bounded(fixture.client.config(ConfigRequest::default()))
        .await
        .unwrap_err();
    assert_eq!(
        error.grpc_code,
        Some(Code::DeadlineExceeded as i32),
        "{error:?}"
    );
    assert!(
        error.message.contains("RPC deadline exceeded"),
        "unexpected error: {error:?}"
    );
    assert!(!error.message.contains("Timeout expired"), "{error:?}");
}

#[tokio::test]
async fn deadlines_too_long_for_the_grpc_timeout_are_sent_and_the_calls_run() {
    // A deadline that gRPC cannot encode is never reached, so the execution and its reattach
    // run to their results, instead of the client panicking while it sets the deadline.
    let mut fixture = Fixture::start(
        vec![streamed(vec![response("r1", false)])],
        vec![streamed(vec![response("r2", true)])],
        quick_retries(),
    )
    .await;
    fixture.client.set_rpc_deadlines(RpcDeadlines {
        reattachable_execute_plan: Some(Duration::MAX),
        reattach_execute: Some(Duration::MAX),
        ..RpcDeadlines::default()
    });
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    let ids = drain(&mut stream).await.unwrap();
    assert_eq!(ids, ["r1", "r2"]);

    assert_eq!(*fixture.state.executes.lock().unwrap(), 1);
    assert_eq!(fixture.state.reattaches.lock().unwrap().len(), 1);
    fixture.assert_release_all_count(1).await;
}

#[tokio::test]
async fn execute_plan_past_its_deadline_is_bounded_as_the_reference_retries() {
    // As the reference: the deadline of the initial ExecutePlan is a RetryException, which is
    // the error once the bound of the retries has passed. With no bound, the error is the
    // deadline, and nothing is reattached.
    let fixture = Fixture::start(
        vec![Step::Stall],
        vec![],
        RetryPolicy {
            max_retry_exception_elapsed_time_ms: Some(0),
            ..quick_retries()
        },
    )
    .await;
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    let error = bounded(stream.message()).await.unwrap_err();
    assert_eq!(
        error.grpc_code,
        Some(Code::DeadlineExceeded as i32),
        "{error:?}"
    );
    assert!(error.message.contains("RPC deadline exceeded"), "{error:?}");
    assert!(fixture.state.reattaches.lock().unwrap().is_empty());
    assert_eq!(*fixture.state.executes.lock().unwrap(), 1);
    fixture.assert_release_all_count(1).await;
}

#[tokio::test]
async fn reattach_past_its_deadline_is_the_error_once_the_bound_passes() {
    // As the reference: a reattach past its deadline is retried, and past the bound of those
    // retries, the deadline is the error and the execution is released.
    let fixture = Fixture::start(
        vec![streamed(vec![response("r1", false)])],
        vec![Step::Stall],
        RetryPolicy {
            max_retry_exception_elapsed_time_ms: Some(0),
            ..quick_retries()
        },
    )
    .await;
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    assert_eq!(
        bounded(stream.message())
            .await
            .unwrap()
            .unwrap()
            .response_id,
        "r1"
    );
    let error = bounded(stream.message()).await.unwrap_err();
    assert_eq!(
        error.grpc_code,
        Some(Code::DeadlineExceeded as i32),
        "{error:?}"
    );
    assert_eq!(fixture.state.reattaches.lock().unwrap().len(), 1);
    fixture.assert_release_all_count(1).await;
}

#[tokio::test]
async fn reattach_that_fails_after_it_opens_is_a_retry_of_the_policy_as_pyspark() {
    // As PySpark's `_has_next`, which reads the stream and reattaches in one attempt of a
    // `Retrying` of its own for each response: a stream that fails after it opened, also one of
    // a reattach, is an attempt of the policy, after the policy's backoff, and its failure is
    // the error once the retries are spent. A response read starts the attempts again.
    let reset = || Err(Status::unavailable("connection reset"));
    let fails_after_opening = || Step::Stream(vec![reset()]);
    let fixture = Fixture::start(
        vec![Step::Stream(vec![Ok(response("r1", false)), reset()])],
        vec![
            fails_after_opening(),
            fails_after_opening(),
            Step::Stream(vec![Ok(response("r2", false)), reset()]),
            fails_after_opening(),
            fails_after_opening(),
            fails_after_opening(),
            // Not reached: the retries are spent before.
            streamed(vec![response("complete", true)]),
        ],
        RetryPolicy {
            max_retries: Some(3),
            initial_backoff_ms: 10,
            max_backoff_ms: Some(40),
            ..quick_retries()
        },
    )
    .await;
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    let read = |response: Option<ExecutePlanResponse>| response.unwrap().response_id;
    assert_eq!(read(bounded(stream.message()).await.unwrap()), "r1");
    let first = Instant::now();
    assert_eq!(read(bounded(stream.message()).await.unwrap()), "r2");
    let second = Instant::now();
    let error = bounded(stream.message()).await.unwrap_err();
    assert_eq!(error.grpc_code, Some(Code::Unavailable as i32), "{error:?}");

    // Three reattaches for each response, each after the backoff of its attempt: 10, 20 and
    // 40 ms from the failure before it.
    let at = fixture.state.reattached_at.lock().unwrap().clone();
    assert_eq!(at.len(), 6, "{at:?}");
    for (start, reattaches) in [(first, &at[..3]), (second, &at[3..])] {
        let mut failed = start;
        for (reattached, backoff) in reattaches.iter().zip([10, 20, 40]) {
            let waited = reattached.duration_since(failed);
            assert!(waited >= Duration::from_millis(backoff), "{waited:?}");
            failed = *reattached;
        }
    }
    assert_eq!(*fixture.state.executes.lock().unwrap(), 1);
    fixture.assert_release_all_count(1).await;
}

#[tokio::test]
async fn operation_not_found_before_any_response_resends_the_execute_plan() {
    // As the reference: the ExecutePlan of the execution streams no response, so it is
    // reattached, and the reattach reports that the server does not know the operation, which
    // is then sent again, its results returned.
    let fixture = Fixture::start(
        vec![
            streamed(vec![]),
            streamed(vec![response("r1", false), response("r2", true)]),
        ],
        vec![Step::Stream(vec![Err(operation_not_found())])],
        quick_retries(),
    )
    .await;
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    let ids = drain(&mut stream).await.unwrap();
    assert_eq!(ids, ["r1", "r2"]);

    assert_eq!(*fixture.state.executes.lock().unwrap(), 2);
    let reattaches = fixture.state.reattaches.lock().unwrap().clone();
    assert_eq!(reattaches.len(), 1);
    assert_eq!(reattaches[0].last_response_id, None);
    fixture.assert_release_all_count(1).await;
}

#[tokio::test]
async fn operation_not_found_in_the_reattach_call_resends_the_execute_plan() {
    // As the reference: the reattach call itself fails with OPERATION_NOT_FOUND, before any
    // response was read: the ExecutePlan is sent again.
    let fixture = Fixture::start(
        vec![
            streamed(vec![]),
            streamed(vec![response("r1", false), response("r2", true)]),
        ],
        vec![Step::Fail(operation_not_found())],
        quick_retries(),
    )
    .await;
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    let ids = drain(&mut stream).await.unwrap();
    assert_eq!(ids, ["r1", "r2"]);
    assert_eq!(*fixture.state.executes.lock().unwrap(), 2);
    assert_eq!(fixture.state.reattaches.lock().unwrap().len(), 1);
    fixture.assert_release_all_count(1).await;
}

#[tokio::test]
async fn operation_not_found_after_a_response_is_the_error_and_is_not_resent() {
    // As the reference, which raises RESPONSE_ALREADY_RECEIVED here: once a response was read,
    // the execution is not sent again, and the stream fails.
    let fixture = Fixture::start(
        vec![streamed(vec![response("r1", false)])],
        vec![Step::Fail(operation_not_found())],
        quick_retries(),
    )
    .await;
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    assert_eq!(
        bounded(stream.message())
            .await
            .unwrap()
            .unwrap()
            .response_id,
        "r1"
    );
    let error = bounded(stream.message()).await.unwrap_err();
    assert_response_already_received(&error, "INVALID_HANDLE.OPERATION_NOT_FOUND");
    assert_eq!(*fixture.state.executes.lock().unwrap(), 1);
    fixture.assert_release_all_count(1).await;
}

/// The error of a reattach to an execution that the server does not know after a response was
/// read, as the reference: `RESPONSE_ALREADY_RECEIVED` of a `PySparkRuntimeError`.
fn assert_response_already_received(error: &spark_connect_core::SparkError, error_type: &str) {
    assert_eq!(
        error.kind,
        spark_connect_core::error::SparkErrorKind::RuntimeError,
        "{error:?}"
    );
    assert_eq!(error.error_class, "RESPONSE_ALREADY_RECEIVED", "{error:?}");
    assert_eq!(
        error.params.get("error_type").map(String::as_str),
        Some(error_type),
        "{error:?}"
    );
}

#[tokio::test]
async fn session_not_found_before_any_response_resends_the_execute_plan() {
    // As the reference, which reads SESSION_NOT_FOUND as OPERATION_NOT_FOUND: no response was
    // read, so the ExecutePlan is sent again, and its results are returned.
    let fixture = Fixture::start(
        vec![
            streamed(vec![]),
            streamed(vec![response("r1", false), response("r2", true)]),
        ],
        vec![Step::Stream(vec![Err(session_not_found())])],
        quick_retries(),
    )
    .await;
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    let ids = drain(&mut stream).await.unwrap();
    assert_eq!(ids, ["r1", "r2"]);

    assert_eq!(*fixture.state.executes.lock().unwrap(), 2);
    assert_eq!(fixture.state.reattaches.lock().unwrap().len(), 1);
    fixture.assert_release_all_count(1).await;
}

#[tokio::test]
async fn session_not_found_after_a_response_is_response_already_received() {
    // As the reference: once a response was read, SESSION_NOT_FOUND is the error of the reattach,
    // not a reason to send the execution again.
    let fixture = Fixture::start(
        vec![streamed(vec![response("r1", false)])],
        vec![Step::Stream(vec![Err(session_not_found())])],
        quick_retries(),
    )
    .await;
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    assert_eq!(
        bounded(stream.message())
            .await
            .unwrap()
            .unwrap()
            .response_id,
        "r1"
    );
    let error = bounded(stream.message()).await.unwrap_err();
    assert_response_already_received(&error, "INVALID_HANDLE.SESSION_NOT_FOUND");
    assert_eq!(*fixture.state.executes.lock().unwrap(), 1);
    fixture.assert_release_all_count(1).await;
}

#[tokio::test]
async fn session_not_found_in_the_reattach_call_after_a_response_is_response_already_received() {
    // As the reference, for the reattach call itself failing with SESSION_NOT_FOUND.
    let fixture = Fixture::start(
        vec![streamed(vec![response("r1", false)])],
        vec![Step::Fail(session_not_found())],
        quick_retries(),
    )
    .await;
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    assert_eq!(
        bounded(stream.message())
            .await
            .unwrap()
            .unwrap()
            .response_id,
        "r1"
    );
    let error = bounded(stream.message()).await.unwrap_err();
    assert_response_already_received(&error, "INVALID_HANDLE.SESSION_NOT_FOUND");
    assert_eq!(*fixture.state.executes.lock().unwrap(), 1);
    fixture.assert_release_all_count(1).await;
}

#[tokio::test]
async fn policy_retry_of_a_reattach_is_not_cut_by_the_retry_exception_bound() {
    // As the reference's RetryPolicyState: the elapsed bound of the RetryExceptions (here 0, as
    // if an hour had passed) does not cut a retry of the policy. The reattach call fails with
    // UNAVAILABLE, and it is retried with the policy's attempts, here reattaching with success.
    let fixture = Fixture::start(
        vec![streamed(vec![response("r1", false)])],
        vec![Step::Fail(Status::unavailable("server restarting"))],
        RetryPolicy {
            max_retry_exception_elapsed_time_ms: Some(0),
            ..quick_retries()
        },
    )
    .await;
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    let ids = drain(&mut stream).await.unwrap();
    assert_eq!(ids, ["r1", "complete"]);

    let reattaches = fixture.state.reattaches.lock().unwrap().clone();
    assert_eq!(reattaches.len(), 2);
    assert!(reattaches
        .iter()
        .all(|r| r.last_response_id.as_deref() == Some("r1")));
    assert_eq!(*fixture.state.executes.lock().unwrap(), 1);
    fixture.assert_release_all_count(1).await;
}

#[tokio::test]
async fn error_of_a_server_with_details_bounds_the_config_read() {
    // As the reference: the stack trace of an error is shown as its config says, read with the
    // config deadline. A config that never answers is not waited for, and the error is raised.
    let fixture = Fixture::start(vec![], vec![], quick_retries()).await;
    let status = Status::with_details(Code::Internal, "boom", error_info_details().into());
    let error = bounded(fixture.client.error_from_status(status)).await;
    assert!(error.message.contains("boom"), "{error:?}");

    let timeouts = fixture.state.timeouts.lock().unwrap().clone();
    assert!(
        timeouts
            .iter()
            .any(|(call, timeout)| *call == "Config" && timeout.is_some()),
        "{timeouts:?}"
    );
}

/// A `google.rpc.Status` with an `ErrorInfo` detail, as the server sends the errors it raises.
fn error_info_details() -> Vec<u8> {
    use prost::Message;

    #[derive(Clone, PartialEq, prost::Message)]
    struct ErrorInfo {
        #[prost(string, tag = "1")]
        reason: String,
        #[prost(string, tag = "2")]
        domain: String,
    }
    #[derive(Clone, PartialEq, prost::Message)]
    struct RpcStatus {
        #[prost(int32, tag = "1")]
        code: i32,
        #[prost(string, tag = "2")]
        message: String,
        #[prost(message, repeated, tag = "3")]
        details: Vec<prost_types::Any>,
    }
    RpcStatus {
        code: Code::Internal as i32,
        message: "boom".into(),
        details: vec![prost_types::Any {
            type_url: "type.googleapis.com/google.rpc.ErrorInfo".into(),
            value: ErrorInfo {
                reason: "BOOM".into(),
                domain: "org.apache.spark".into(),
            }
            .encode_to_vec(),
        }],
    }
    .encode_to_vec()
}

/// The loopback service: ExecutePlan, ReattachExecute, ReleaseExecute and Config.
fn service(state: Arc<State>) -> FakeService {
    let (execute, reattach, release) = (state.clone(), state.clone(), state.clone());
    FakeService::new()
        .server_streaming("ExecutePlan", ProstCodec::default, move |_| {
            Execute(execute.clone())
        })
        .server_streaming("ReattachExecute", ProstCodec::default, move |_| {
            Reattach(reattach.clone())
        })
        .unary("ReleaseExecute", ProstCodec::default, move |_| {
            ReleaseRpc(release.clone())
        })
        .unary("Config", ProstCodec::default, move |_| {
            Config(state.clone())
        })
}

/// Serves the next step of `script`, a call that stalls never returning its headers.
async fn serve(step: Step) -> Result<Response<Responses>, Status> {
    match step {
        Step::Stall => {
            std::future::pending::<()>().await;
            unreachable!("a stalled call never completes")
        }
        Step::Fail(status) => Err(status),
        Step::Stream(items) => {
            let responses: Responses = Box::pin(stream::iter(items));
            Ok(Response::new(responses))
        }
    }
}

struct Execute(Arc<State>);

impl ServerStreamingService<ExecutePlanRequest> for Execute {
    type Response = ExecutePlanResponse;
    type ResponseStream = Responses;
    type Future = BoxFuture<Response<Responses>, Status>;

    fn call(&mut self, request: Request<ExecutePlanRequest>) -> Self::Future {
        let state = self.0.clone();
        state.record_timeout("ExecutePlan", request.metadata());
        *state.executes.lock().unwrap() += 1;
        let operation = request.get_ref().operation_id.clone().unwrap_or_default();
        let step = state
            .execute_script
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| streamed(vec![response("complete", true)]));
        if matches!(step, Step::Stream(_)) {
            state.known.lock().unwrap().insert(operation);
        }
        Box::pin(serve(step))
    }
}

struct Reattach(Arc<State>);

impl ServerStreamingService<ReattachExecuteRequest> for Reattach {
    type Response = ExecutePlanResponse;
    type ResponseStream = Responses;
    type Future = BoxFuture<Response<Responses>, Status>;

    fn call(&mut self, request: Request<ReattachExecuteRequest>) -> Self::Future {
        let state = self.0.clone();
        state.record_timeout("ReattachExecute", request.metadata());
        let request = request.into_inner();
        state.reattaches.lock().unwrap().push(request.clone());
        state.reattached_at.lock().unwrap().push(Instant::now());
        let step = state
            .reattach_script
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| {
                if state.known.lock().unwrap().contains(&request.operation_id) {
                    streamed(vec![response("complete", true)])
                } else {
                    Step::Fail(operation_not_found())
                }
            });
        Box::pin(serve(step))
    }
}

struct ReleaseRpc(Arc<State>);

impl tonic::server::UnaryService<ReleaseExecuteRequest> for ReleaseRpc {
    type Response = ReleaseExecuteResponse;
    type Future = BoxFuture<Response<ReleaseExecuteResponse>, Status>;

    fn call(&mut self, request: Request<ReleaseExecuteRequest>) -> Self::Future {
        let request = request.into_inner();
        self.0.releases.lock().unwrap().push(CapturedRelease {
            request: request.clone(),
        });
        Box::pin(async move {
            Ok(Response::new(ReleaseExecuteResponse {
                session_id: request.session_id,
                operation_id: Some(request.operation_id),
                ..Default::default()
            }))
        })
    }
}

struct Config(Arc<State>);

impl UnaryService<ConfigRequest> for Config {
    type Response = ConfigResponse;
    type Future = BoxFuture<Response<ConfigResponse>, Status>;

    fn call(&mut self, request: Request<ConfigRequest>) -> Self::Future {
        self.0.record_timeout("Config", request.metadata());
        // Stalled before its headers, as a connection that never answers.
        Box::pin(async {
            std::future::pending::<()>().await;
            unreachable!("a stalled call never completes")
        })
    }
}

#[tokio::test]
async fn reattach_of_a_stream_whose_client_was_closed_raises_no_active_session() {
    // As the reference: an execution that is still read when its session is stopped, and that
    // needs a reattach, raises NO_ACTIVE_SESSION from the stub of the closed client, instead of
    // reattaching. The stream ends with that error, and the server gets no reattach.
    let fixture = Fixture::start(
        vec![streamed(vec![response("r1", false)])],
        vec![streamed(vec![response("r2", true)])],
        quick_retries(),
    )
    .await;
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    let first = bounded(stream.message()).await.unwrap().unwrap();
    assert_eq!(first.response_id, "r1");

    // The stream ends after r1, so the next read reattaches, which the closed client refuses.
    fixture.client.close();
    let error = bounded(stream.message()).await.unwrap_err();
    assert_eq!(
        error.get_condition().as_deref(),
        Some("NO_ACTIVE_SESSION"),
        "{error:?}"
    );
    assert!(bounded(stream.message()).await.unwrap().is_none());

    assert!(fixture.state.reattaches.lock().unwrap().is_empty());
    assert_eq!(*fixture.state.executes.lock().unwrap(), 1);
}

#[tokio::test]
async fn retry_of_an_execution_whose_client_was_closed_raises_no_active_session() {
    // As the reference: each attempt of a call goes through the stub of the client, which
    // raises NO_ACTIVE_SESSION once the client is closed. An execution that waits to retry when
    // its session is stopped raises that error at its next attempt, and sends nothing more.
    let fixture = Fixture::start(
        (0..1000)
            .map(|_| Step::Fail(Status::unavailable("server unavailable")))
            .collect(),
        vec![],
        RetryPolicy {
            max_retries: Some(1000),
            initial_backoff_ms: 20,
            max_backoff_ms: Some(20),
            ..RetryPolicy::default()
        },
    )
    .await;
    let client = fixture.client.clone();
    let call = tokio::spawn(async move {
        client
            .execute_plan_reattachable(execute_request())
            .await
            .map(|_| ())
    });
    // The execution is retried before its session stops.
    let state = fixture.state.clone();
    bounded(async {
        while *state.executes.lock().unwrap() < 2 {
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
    })
    .await;

    fixture.client.close();
    let error = bounded(call).await.unwrap().unwrap_err();
    assert_eq!(
        error.get_condition().as_deref(),
        Some("NO_ACTIVE_SESSION"),
        "{error:?}"
    );
    let sent = *fixture.state.executes.lock().unwrap();
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert_eq!(*fixture.state.executes.lock().unwrap(), sent);
}
