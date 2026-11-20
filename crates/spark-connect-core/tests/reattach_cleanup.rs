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

//! Real-wire lifecycle tests; no Spark server or generated server bindings required.

mod common;

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::{FakeServer, FakeService};
use futures::{stream, Stream, StreamExt};
use spark_connect_core::{
    ChannelBuilder, ResultChunking, RetryPolicy, RpcDeadlines, SparkConnectClient,
};
use spark_connect_proto::execute_plan_response::{ResponseType, ResultComplete};
use spark_connect_proto::release_execute_request::Release;
use spark_connect_proto::{
    ExecutePlanRequest, ExecutePlanResponse, ReattachExecuteRequest, ReleaseExecuteRequest,
    ReleaseExecuteResponse, UserContext,
};
use tokio::sync::{oneshot, Notify};
use tonic::codegen::BoxFuture;
use tonic::metadata::MetadataMap;
use tonic::server::{ServerStreamingService, UnaryService};
use tonic::{Code, Request, Response, Status};
use tonic_prost::ProstCodec;

const WAIT: Duration = Duration::from_secs(3);
type Responses = Pin<Box<dyn Stream<Item = Result<ExecutePlanResponse, Status>> + Send>>;

async fn bounded<T>(future: impl Future<Output = T>) -> T {
    tokio::time::timeout(WAIT, future)
        .await
        .expect("test RPC timed out")
}

#[derive(Clone, Copy)]
enum Scenario {
    Complete,
    Open,
    Error,
    Eof,
    Disconnect,
    /// A retryable INTERNAL status carrying Spark's cursor-disconnect marker.
    CursorDisconnect,
    /// The ExecutePlan call fails itself, with a trailers-only response, as for an
    /// analysis error.
    Reject,
}

#[derive(Clone)]
struct CapturedRelease {
    request: ReleaseExecuteRequest,
    metadata: MetadataMap,
}

#[derive(Clone)]
struct CapturedExecute {
    request: ExecutePlanRequest,
    metadata: MetadataMap,
}

#[derive(Clone)]
struct CapturedReattach {
    request: ReattachExecuteRequest,
    metadata: MetadataMap,
}

struct State {
    scenario: Scenario,
    executions: Mutex<Vec<CapturedExecute>>,
    releases: Mutex<Vec<CapturedRelease>>,
    reattaches: Mutex<Vec<CapturedReattach>>,
    /// What each ReattachExecute call does, in order: fail with a code, or stream responses.
    /// When none is left, a call completes the result.
    reattach_script: Mutex<VecDeque<Result<Vec<ExecutePlanResponse>, Code>>>,
    release_errors: Mutex<VecDeque<Code>>,
    release_finished: AtomicUsize,
    block_release: AtomicBool,
    block_reattach: AtomicBool,
    release_gate: Notify,
    reattach_gate: Notify,
    changed: Notify,
    /// The `grpc-timeout` of each ExecutePlan and ReattachExecute call, in order.
    timeouts: Mutex<Vec<(&'static str, Option<String>)>>,
}

impl State {
    fn record_timeout(&self, call: &'static str, metadata: &MetadataMap) {
        let timeout = metadata
            .get("grpc-timeout")
            .map(|value| value.to_str().unwrap().to_string());
        self.timeouts.lock().unwrap().push((call, timeout));
    }

    fn all_releases(&self) -> Vec<CapturedRelease> {
        self.releases
            .lock()
            .unwrap()
            .iter()
            .filter(|r| matches!(r.request.release, Some(Release::ReleaseAll(_))))
            .cloned()
            .collect()
    }
}

struct Fixture {
    state: Arc<State>,
    _server: FakeServer,
    client: SparkConnectClient,
}

impl Fixture {
    async fn start(scenario: Scenario) -> Self {
        let state = Arc::new(State {
            scenario,
            executions: Mutex::new(Vec::new()),
            releases: Mutex::new(Vec::new()),
            reattaches: Mutex::new(Vec::new()),
            reattach_script: Mutex::new(VecDeque::new()),
            release_errors: Mutex::new(VecDeque::new()),
            release_finished: AtomicUsize::new(0),
            block_release: AtomicBool::new(false),
            block_reattach: AtomicBool::new(false),
            release_gate: Notify::new(),
            reattach_gate: Notify::new(),
            changed: Notify::new(),
            timeouts: Mutex::new(Vec::new()),
        });
        let server = service(state.clone()).serve().await;
        let address = server.address;
        let builder = ChannelBuilder::parse(&format!(
            "sc://localhost:{}/;user_id=transport-user;token=fixture-token;x-cleanup=fixture-value;session_id=00000000-0000-4000-8000-000000000001",
            address.port()
        ))
        .unwrap();
        let mut client = bounded(SparkConnectClient::connect(&builder))
            .await
            .unwrap();
        client.set_retry_policy(RetryPolicy {
            max_retries: Some(2),
            initial_backoff_ms: 1,
            max_backoff_ms: Some(2),
            ..RetryPolicy::default()
        });
        Self {
            state,
            _server: server,
            client,
        }
    }

    async fn wait_for(&self, predicate: impl Fn(&State) -> bool) {
        tokio::time::timeout(WAIT, async {
            loop {
                let changed = self.state.changed.notified();
                if predicate(&self.state) {
                    return;
                }
                changed.await;
            }
        })
        .await
        .expect("fixture did not observe the expected RPC");
    }

    async fn assert_release_all_count(&self, expected: usize) {
        self.wait_for(|s| s.all_releases().len() >= expected).await;
        assert_eq!(self.state.all_releases().len(), expected);
        assert!(
            tokio::time::timeout(
                Duration::from_millis(100),
                self.wait_for(|s| s.all_releases().len() > expected)
            )
            .await
            .is_err(),
            "execution received more ReleaseAll requests than expected"
        );
    }
}

fn execute_request() -> ExecutePlanRequest {
    ExecutePlanRequest {
        session_id: "00000000-0000-4000-8000-000000000001".into(),
        user_context: Some(UserContext {
            user_id: "request-user".into(),
            ..Default::default()
        }),
        operation_id: Some("00000000-0000-4000-8000-000000000002".into()),
        client_type: Some("cleanup-regression".into()),
        ..Default::default()
    }
}

fn response(id: &str, complete: bool) -> ExecutePlanResponse {
    ExecutePlanResponse {
        // The server echoes the session ID of the request.
        session_id: "00000000-0000-4000-8000-000000000001".into(),
        response_id: id.into(),
        response_type: complete.then_some(ResponseType::ResultComplete(ResultComplete {})),
        ..Default::default()
    }
}

fn assert_identity(captured: &CapturedRelease) {
    let original = execute_request();
    assert_eq!(captured.request.session_id, original.session_id);
    assert_eq!(captured.request.user_context, original.user_context);
    assert_eq!(
        Some(captured.request.operation_id.clone()),
        original.operation_id
    );
    assert_eq!(captured.request.client_type, original.client_type);
    assert_metadata(&captured.metadata);
}

fn assert_metadata(metadata: &MetadataMap) {
    assert_eq!(
        metadata.get("authorization").unwrap(),
        "Bearer fixture-token"
    );
    // As PySpark's, which sends the user ID in the user context only.
    assert!(metadata.get("user_id").is_none());
    assert_eq!(metadata.get("x-cleanup").unwrap(), "fixture-value");
    // tonic's own header, as grpcio sends its own: the user agent is the client type.
    let user_agent = metadata.get("user-agent").unwrap().to_str().unwrap();
    assert!(
        user_agent.starts_with("tonic/") && !user_agent.contains("spark/"),
        "{user_agent}"
    );
}

/// The loopback service: ExecutePlan, ReattachExecute and ReleaseExecute.
fn service(state: Arc<State>) -> FakeService {
    let (execute, reattach) = (state.clone(), state.clone());
    FakeService::new()
        .server_streaming("ExecutePlan", ProstCodec::default, move |_| {
            Execute(execute.clone())
        })
        .server_streaming("ReattachExecute", ProstCodec::default, move |_| {
            Reattach(reattach.clone())
        })
        .unary("ReleaseExecute", ProstCodec::default, move |_| {
            ReleaseRpc(state.clone())
        })
}

struct Execute(Arc<State>);

impl ServerStreamingService<ExecutePlanRequest> for Execute {
    type Response = ExecutePlanResponse;
    type ResponseStream = Responses;
    type Future = BoxFuture<Response<Responses>, Status>;

    fn call(&mut self, request: Request<ExecutePlanRequest>) -> Self::Future {
        self.0.record_timeout("ExecutePlan", request.metadata());
        let metadata = request.metadata().clone();
        self.0.executions.lock().unwrap().push(CapturedExecute {
            request: request.into_inner(),
            metadata,
        });
        if let Scenario::Reject = self.0.scenario {
            return Box::pin(async { Err(Status::invalid_argument("plan sentinel")) });
        }
        let responses: Responses = match self.0.scenario {
            Scenario::Complete => Box::pin(stream::iter([Ok(response("complete", true))])),
            Scenario::Error => Box::pin(stream::iter([Err(Status::invalid_argument(
                "query sentinel",
            ))])),
            Scenario::Open => {
                Box::pin(stream::iter([Ok(response("first", false))]).chain(stream::pending()))
            }
            Scenario::Eof => Box::pin(stream::iter([Ok(response("first", false))])),
            Scenario::Disconnect => Box::pin(stream::iter([
                Ok(response("first", false)),
                Err(Status::unavailable("stream disconnected")),
            ])),
            Scenario::CursorDisconnect => Box::pin(stream::iter([
                Ok(response("first", false)),
                Err(Status::internal("INVALID_CURSOR.DISCONNECTED")),
            ])),
            Scenario::Reject => unreachable!("rejected before streaming"),
        };
        Box::pin(async { Ok(Response::new(responses)) })
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
        let metadata = request.metadata().clone();
        state.reattaches.lock().unwrap().push(CapturedReattach {
            request: request.into_inner(),
            metadata,
        });
        state.changed.notify_waiters();
        Box::pin(async move {
            if state.block_reattach.load(Ordering::SeqCst) {
                state.reattach_gate.notified().await;
            }
            let scripted = state.reattach_script.lock().unwrap().pop_front();
            let responses = match scripted {
                Some(Err(code)) => return Err(Status::new(code, "reattach sentinel")),
                Some(Ok(responses)) => responses,
                None => vec![response("complete", true)],
            };
            let responses: Responses = Box::pin(stream::iter(responses.into_iter().map(Ok)));
            Ok(Response::new(responses))
        })
    }
}

struct ReleaseRpc(Arc<State>);

impl UnaryService<ReleaseExecuteRequest> for ReleaseRpc {
    type Response = ReleaseExecuteResponse;
    type Future = BoxFuture<Response<ReleaseExecuteResponse>, Status>;

    fn call(&mut self, request: Request<ReleaseExecuteRequest>) -> Self::Future {
        let state = self.0.clone();
        let metadata = request.metadata().clone();
        let request = request.into_inner();
        state.releases.lock().unwrap().push(CapturedRelease {
            request: request.clone(),
            metadata,
        });
        state.changed.notify_waiters();
        Box::pin(async move {
            if state.block_release.load(Ordering::SeqCst) {
                state.release_gate.notified().await;
            }
            let error = state.release_errors.lock().unwrap().pop_front();
            state.release_finished.fetch_add(1, Ordering::SeqCst);
            state.changed.notify_waiters();
            match error {
                Some(code) => Err(Status::new(code, "cleanup sentinel")),
                None => Ok(Response::new(ReleaseExecuteResponse {
                    session_id: request.session_id,
                    operation_id: Some(request.operation_id),
                    ..Default::default()
                })),
            }
        })
    }
}

#[tokio::test]
async fn completion_releases_once_without_waiting_for_release_reply() {
    let fixture = Fixture::start(Scenario::Complete).await;
    fixture.state.block_release.store(true, Ordering::SeqCst);
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    let result = tokio::time::timeout(Duration::from_secs(1), stream.message())
        .await
        .expect("completion waited for cleanup")
        .unwrap()
        .unwrap();
    assert!(matches!(
        result.response_type,
        Some(ResponseType::ResultComplete(_))
    ));
    let captured = fixture.state.executions.lock().unwrap()[0].clone();
    assert_metadata(&captured.metadata);
    let executed = captured.request;
    let original = execute_request();
    assert_eq!(executed.session_id, original.session_id);
    assert_eq!(executed.user_context, original.user_context);
    assert_eq!(executed.operation_id, original.operation_id);
    assert_eq!(executed.client_type, original.client_type);
    assert!(executed.request_options.iter().any(|option| matches!(
        &option.request_option,
        Some(spark_connect_proto::execute_plan_request::request_option::RequestOption::ReattachOptions(options))
            if options.reattachable
    )));
    // Observe terminal cleanup while the stream is alive: Drop must not mask it.
    fixture.wait_for(|s| !s.all_releases().is_empty()).await;
    assert_eq!(fixture.state.release_finished.load(Ordering::SeqCst), 0);
    assert_identity(&fixture.state.all_releases()[0]);
    assert!(bounded(stream.message()).await.unwrap().is_none());
    fixture.state.release_gate.notify_one();
    fixture
        .wait_for(|s| s.release_finished.load(Ordering::SeqCst) == 1)
        .await;
    drop(stream);
    fixture.assert_release_all_count(1).await;
}

#[tokio::test]
async fn dropping_unfinished_stream_releases_inside_runtime() {
    assert_unfinished_drop(false).await;
}

#[tokio::test]
async fn dropping_unfinished_stream_releases_outside_runtime() {
    assert_unfinished_drop(true).await;
}

async fn assert_unfinished_drop(outside_runtime: bool) {
    let fixture = Fixture::start(Scenario::Open).await;
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    assert_eq!(
        bounded(stream.message())
            .await
            .unwrap()
            .unwrap()
            .response_id,
        "first"
    );
    fixture
        .wait_for(|s| !s.releases.lock().unwrap().is_empty())
        .await;
    let until = fixture.state.releases.lock().unwrap()[0].clone();
    assert_identity(&until);
    assert!(matches!(until.request.release,
        Some(Release::ReleaseUntil(ref until)) if until.response_id == "first"));
    if outside_runtime {
        let (finished, completion) = oneshot::channel();
        drop(std::thread::spawn(move || {
            drop(stream);
            let _ = finished.send(());
        }));
        tokio::time::timeout(WAIT, completion)
            .await
            .expect("drop did not return")
            .expect("drop thread panicked");
    } else {
        drop(stream);
    }
    fixture.assert_release_all_count(1).await;
    assert_identity(&fixture.state.all_releases()[0]);
}

#[tokio::test]
async fn terminal_failure_releases_and_preserves_the_original_error() {
    let fixture = Fixture::start(Scenario::Error).await;
    fixture
        .state
        .release_errors
        .lock()
        .unwrap()
        .push_back(Code::PermissionDenied);
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    let error = bounded(stream.message()).await.unwrap_err().to_string();
    assert!(
        error.contains("query sentinel"),
        "unexpected error: {error}"
    );
    fixture
        .wait_for(|s| s.release_finished.load(Ordering::SeqCst) == 1)
        .await;
    assert!(bounded(stream.message()).await.unwrap().is_none());
    drop(stream);
    fixture.assert_release_all_count(1).await;
}

#[tokio::test]
async fn transient_release_failure_retries_without_changing_success() {
    let fixture = Fixture::start(Scenario::Complete).await;
    fixture
        .state
        .release_errors
        .lock()
        .unwrap()
        .push_back(Code::Unavailable);
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    assert!(matches!(
        bounded(stream.message())
            .await
            .unwrap()
            .unwrap()
            .response_type,
        Some(ResponseType::ResultComplete(_))
    ));
    fixture
        .wait_for(|s| s.release_finished.load(Ordering::SeqCst) == 2)
        .await;
    let releases = fixture.state.all_releases();
    assert_eq!(releases.len(), 2);
    assert_eq!(releases[0].request, releases[1].request);
    releases.iter().for_each(assert_identity);
    assert!(bounded(stream.message()).await.unwrap().is_none());
    drop(stream);
    fixture.assert_release_all_count(2).await;
}

#[tokio::test]
async fn permanent_release_failure_does_not_retry_or_change_success() {
    let fixture = Fixture::start(Scenario::Complete).await;
    fixture
        .state
        .release_errors
        .lock()
        .unwrap()
        .push_back(Code::PermissionDenied);
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    assert!(matches!(
        bounded(stream.message())
            .await
            .unwrap()
            .unwrap()
            .response_type,
        Some(ResponseType::ResultComplete(_))
    ));
    fixture
        .wait_for(|s| s.release_finished.load(Ordering::SeqCst) == 1)
        .await;
    assert!(bounded(stream.message()).await.unwrap().is_none());
    drop(stream);
    fixture.assert_release_all_count(1).await;
}

#[tokio::test]
async fn eof_reattaches_before_releasing_the_execution() {
    assert_detached_reattachment(Scenario::Eof).await;
}

#[tokio::test]
async fn unavailable_reattaches_before_releasing_the_execution() {
    assert_detached_reattachment(Scenario::Disconnect).await;
}

#[tokio::test]
async fn disconnected_cursor_reattaches_before_releasing_the_execution() {
    assert_detached_reattachment(Scenario::CursorDisconnect).await;
}

async fn assert_detached_reattachment(scenario: Scenario) {
    let fixture = Fixture::start(scenario).await;
    fixture.state.block_reattach.store(true, Ordering::SeqCst);
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    assert_eq!(
        bounded(stream.message())
            .await
            .unwrap()
            .unwrap()
            .response_id,
        "first"
    );
    let next = tokio::spawn(async move {
        let result = bounded(stream.message()).await;
        (stream, result)
    });
    fixture
        .wait_for(|s| !s.reattaches.lock().unwrap().is_empty())
        .await;
    let captured = fixture.state.reattaches.lock().unwrap()[0].clone();
    let reattach = captured.request;
    let original = execute_request();
    assert_eq!(reattach.session_id, original.session_id);
    assert_eq!(reattach.user_context, original.user_context);
    assert_eq!(reattach.client_type, original.client_type);
    assert_metadata(&captured.metadata);
    assert_eq!(reattach.operation_id, original.operation_id.unwrap());
    assert_eq!(reattach.last_response_id.as_deref(), Some("first"));
    assert!(
        tokio::time::timeout(
            Duration::from_millis(100),
            fixture.wait_for(|s| !s.all_releases().is_empty())
        )
        .await
        .is_err(),
        "released before reattachment completed"
    );
    fixture.state.reattach_gate.notify_one();
    let (stream, result) = tokio::time::timeout(WAIT, next).await.unwrap().unwrap();
    assert!(matches!(
        result.unwrap().unwrap().response_type,
        Some(ResponseType::ResultComplete(_))
    ));
    fixture.wait_for(|s| !s.all_releases().is_empty()).await;
    drop(stream);
    fixture.assert_release_all_count(1).await;
}

#[tokio::test]
async fn permanent_reattach_failure_releases_and_preserves_the_error() {
    let fixture = Fixture::start(Scenario::Eof).await;
    fixture
        .state
        .reattach_script
        .lock()
        .unwrap()
        .push_back(Err(Code::PermissionDenied));
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    assert_eq!(
        bounded(stream.message())
            .await
            .unwrap()
            .unwrap()
            .response_id,
        "first"
    );
    let error = bounded(stream.message()).await.unwrap_err();
    assert!(error.message.contains("reattach sentinel"), "{error:?}");
    assert_eq!(error.grpc_code, Some(Code::PermissionDenied as i32));
    assert_eq!(error.operation_id, execute_request().operation_id);
    // Observe failure cleanup before Drop can send a fallback release.
    fixture.wait_for(|s| !s.all_releases().is_empty()).await;
    assert_identity(&fixture.state.all_releases()[0]);
    assert!(bounded(stream.message()).await.unwrap().is_none());
    drop(stream);
    fixture.assert_release_all_count(1).await;
    assert_eq!(fixture.state.reattaches.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn initial_failure_releases_the_execution() {
    // As the reference iterator releases an execution whose first ExecutePlan call fails,
    // such as for an analysis error, rather than leave it until its detached timeout.
    let fixture = Fixture::start(Scenario::Reject).await;
    let result = bounded(fixture.client.execute_plan_reattachable(execute_request())).await;
    let Err(error) = result else {
        panic!("the rejected ExecutePlan returned a stream");
    };
    let error = error.to_string();
    assert!(error.contains("plan sentinel"), "unexpected error: {error}");
    fixture.assert_release_all_count(1).await;
    assert_identity(&fixture.state.all_releases()[0]);
}

/// Whether `request` asks for a reattachable execution.
fn asks_to_reattach(request: &ExecutePlanRequest) -> bool {
    use spark_connect_proto::execute_plan_request::request_option::RequestOption;
    request.request_options.iter().any(|option| {
        matches!(
            &option.request_option,
            Some(RequestOption::ReattachOptions(_))
        )
    })
}

#[tokio::test]
async fn an_execution_that_is_not_reattachable_is_sent_again_and_never_released() {
    // As PySpark's client with `use_reattachable_execute` False: the request has no reattach
    // option and the ExecutePlan no deadline, a stream that fails in a way that the policy
    // retries sends the ExecutePlan again, under the retries of the whole execution (two here),
    // and nothing is reattached or released.
    let fixture = Fixture::start(Scenario::Disconnect).await;
    fixture.client.set_reattachable_execute(false);
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    let mut ids = Vec::new();
    let error = loop {
        match bounded(stream.message()).await {
            Ok(Some(response)) => ids.push(response.response_id),
            Ok(None) => panic!("the failed execution ended"),
            Err(error) => break error,
        }
    };
    assert!(error.to_string().contains("stream disconnected"), "{error}");
    assert_eq!(ids, ["first", "first", "first"]);
    let executions = fixture.state.executions.lock().unwrap().clone();
    assert_eq!(executions.len(), 3);
    assert!(executions.iter().all(|e| !asks_to_reattach(&e.request)));
    assert!(fixture
        .state
        .timeouts
        .lock()
        .unwrap()
        .iter()
        .all(|(call, timeout)| *call == "ExecutePlan" && timeout.is_none()));
    drop(stream);
    assert!(fixture.state.reattaches.lock().unwrap().is_empty());
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(fixture.state.releases.lock().unwrap().is_empty());
}

#[tokio::test]
async fn an_execution_that_is_not_reattachable_ends_with_its_stream() {
    // The stream of an execution that is not reattachable ends with the call, without a
    // ResultComplete, and a client of a new session keeps the setting, as PySpark's
    // `newSession` keeps `use_reattachable_execute`; it can be turned on again.
    let fixture = Fixture::start(Scenario::Eof).await;
    fixture.client.set_reattachable_execute(false);
    assert!(!fixture.client.with_new_session_id().reattachable_execute());
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    assert_eq!(
        bounded(stream.message())
            .await
            .unwrap()
            .unwrap()
            .response_id,
        "first"
    );
    assert!(bounded(stream.message()).await.unwrap().is_none());
    drop(stream);
    assert!(fixture.state.reattaches.lock().unwrap().is_empty());
    tokio::time::sleep(Duration::from_millis(100)).await;
    assert!(fixture.state.releases.lock().unwrap().is_empty());

    fixture.client.set_reattachable_execute(true);
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    while bounded(stream.message()).await.unwrap().is_some() {}
    assert!(asks_to_reattach(
        &fixture.state.executions.lock().unwrap()[1].request
    ));
    assert_eq!(fixture.state.reattaches.lock().unwrap().len(), 1);
    drop(stream);
    fixture.assert_release_all_count(1).await;
}

#[tokio::test]
async fn the_result_chunking_of_the_client_is_that_of_its_requests() {
    // As PySpark's `allow_arrow_batch_chunking` and `preferred_arrow_chunk_size`, sent in the
    // options of each ExecutePlanRequest; a client of a new session has the default ones.
    let mut fixture = Fixture::start(Scenario::Complete).await;
    let chunking = ResultChunking {
        allow_arrow_batch_chunking: false,
        preferred_arrow_chunk_size: Some(4096),
    };
    fixture.client.set_result_chunking(chunking);
    assert_eq!(
        fixture.client.with_new_session_id().result_chunking(),
        ResultChunking::default()
    );
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    while bounded(stream.message()).await.unwrap().is_some() {}
    use spark_connect_proto::execute_plan_request::request_option::RequestOption;
    let request = fixture.state.executions.lock().unwrap()[0].request.clone();
    let sent: Vec<_> = request
        .request_options
        .iter()
        .filter_map(|option| match &option.request_option {
            Some(RequestOption::ResultChunkingOptions(options)) => Some((
                options.allow_arrow_batch_chunking,
                options.preferred_arrow_chunk_size,
            )),
            _ => None,
        })
        .collect();
    assert_eq!(sent, [(false, Some(4096))]);
}

#[tokio::test]
async fn a_replaced_retry_policy_retries_the_calls_that_start_after_it() {
    // As PySpark's `set_retry_policies` on a client in use: the clones of the client share
    // the policy, which the next call reads; a client of a new session has the default one.
    let fixture = Fixture::start(Scenario::Disconnect).await;
    let clone = fixture.client.clone();
    clone.replace_retry_policy(RetryPolicy::no_retries());
    assert_eq!(fixture.client.retry_policy().max_retries, Some(0));
    assert_eq!(
        fixture
            .client
            .with_new_session_id()
            .retry_policy()
            .max_retries,
        RetryPolicy::default().max_retries
    );
    fixture.client.set_reattachable_execute(false);
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    assert!(bounded(stream.message()).await.unwrap().is_some());
    assert!(bounded(stream.message()).await.is_err());
    assert_eq!(fixture.state.executions.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn each_response_retries_its_reattaches_afresh() {
    // Each response read has the retries of the policy, as the reference iterator's
    // `_has_next` makes a `Retrying` of its own: three reattaches that fail once each read
    // the whole result, though the policy allows two retries.
    let fixture = Fixture::start(Scenario::Eof).await;
    {
        let mut script = fixture.state.reattach_script.lock().unwrap();
        for id in ["second", "third"] {
            script.push_back(Err(Code::Unavailable));
            script.push_back(Ok(vec![response(id, false)]));
        }
        script.push_back(Err(Code::Unavailable));
        script.push_back(Ok(vec![response("complete", true)]));
    }
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    let mut ids = Vec::new();
    while let Some(response) = bounded(stream.message()).await.unwrap() {
        ids.push(response.response_id);
    }
    assert_eq!(ids, ["first", "second", "third", "complete"]);
    assert_eq!(fixture.state.reattaches.lock().unwrap().len(), 6);
    fixture.assert_release_all_count(1).await;
}

/// The per-RPC deadline of the execution streams of the tests of their deadlines. It runs from
/// the call, so it covers the first response of a stream too, which the fixture sends at once and
/// which must arrive before it: a second leaves a loaded machine, of the tests of this file at once,
/// the time to deliver it, and the deadline still ends a stream that stays open after it within
/// `WAIT`.
const STREAM_DEADLINE: Duration = Duration::from_secs(1);

/// The deadlines of PySpark, but of `stream` for the streams of an execution.
fn stream_deadlines(stream: Duration) -> RpcDeadlines {
    RpcDeadlines {
        reattachable_execute_plan: Some(stream),
        reattach_execute: Some(stream),
        ..RpcDeadlines::default()
    }
}

#[tokio::test]
async fn execution_streams_carry_their_deadlines_and_reattach_past_them() {
    // As the reference iterator: the stream of a reattachable ExecutePlan, and that of each
    // ReattachExecute, has its per-RPC deadline, after which the execution goes on on the
    // server, so the iteration reattaches to resume its results.
    let mut fixture = Fixture::start(Scenario::Open).await;
    fixture
        .client
        .set_rpc_deadlines(stream_deadlines(STREAM_DEADLINE));
    {
        let mut script = fixture.state.reattach_script.lock().unwrap();
        // A stream that stalls too, then the rest of the result.
        script.push_back(Ok(vec![response("second", false)]));
        script.push_back(Ok(vec![response("complete", true)]));
    }
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    let mut ids = Vec::new();
    while let Some(response) = bounded(stream.message()).await.unwrap() {
        ids.push(response.response_id);
    }
    assert_eq!(ids, ["first", "second", "complete"]);
    let reattaches = fixture.state.reattaches.lock().unwrap().clone();
    let resumed: Vec<_> = reattaches
        .iter()
        .map(|r| r.request.last_response_id.clone())
        .collect();
    assert_eq!(
        resumed,
        [Some("first".to_string()), Some("second".to_string())]
    );
    let timeouts = fixture.state.timeouts.lock().unwrap().clone();
    assert_eq!(timeouts.len(), 3);
    assert!(
        timeouts.iter().all(|(_, timeout)| timeout.is_some()),
        "{timeouts:?}"
    );
    fixture.assert_release_all_count(1).await;
}

#[tokio::test]
async fn reattaches_past_deadlines_are_bounded_as_the_reference_retries() {
    // The reattaches after deadlines of a response are bounded by the elapsed time of the
    // reference `RetryException`s: past it, the deadline is the error, as PySpark raises.
    let mut fixture = Fixture::start(Scenario::Open).await;
    fixture
        .client
        .set_rpc_deadlines(stream_deadlines(STREAM_DEADLINE));
    fixture.client.set_retry_policy(RetryPolicy {
        max_retries: Some(2),
        initial_backoff_ms: 1,
        max_backoff_ms: Some(2),
        max_retry_exception_elapsed_time_ms: Some(0),
        ..RetryPolicy::default()
    });
    let started = std::time::Instant::now();
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    assert_eq!(
        bounded(stream.message())
            .await
            .unwrap()
            .unwrap()
            .response_id,
        "first"
    );
    let error = bounded(stream.message()).await.unwrap_err();
    assert!(error.message.contains("RPC deadline exceeded"), "{error:?}");
    assert_eq!(error.grpc_code, Some(Code::DeadlineExceeded as i32));
    // It is the deadline of the call that ended the stream, not a deadline of the test.
    assert!(
        started.elapsed() >= STREAM_DEADLINE,
        "{:?}",
        started.elapsed()
    );
    let timeouts = fixture.state.timeouts.lock().unwrap().clone();
    assert_eq!(timeouts.len(), 1, "{timeouts:?}");
    assert!(fixture.state.reattaches.lock().unwrap().is_empty());
    fixture.assert_release_all_count(1).await;
}

#[tokio::test]
async fn disabled_deadlines_send_no_timeout() {
    // As `RpcDeadlines.disabled()`: the calls have no deadline.
    let mut fixture = Fixture::start(Scenario::Complete).await;
    fixture.client.set_rpc_deadlines(RpcDeadlines::disabled());
    let mut stream = bounded(fixture.client.execute_plan_reattachable(execute_request()))
        .await
        .unwrap();
    while bounded(stream.message()).await.unwrap().is_some() {}
    let timeouts = fixture.state.timeouts.lock().unwrap().clone();
    assert_eq!(timeouts, [("ExecutePlan", None)]);
    assert_eq!(
        RpcDeadlines::default().analyze_plan,
        Some(Duration::from_secs(3600))
    );
    assert_eq!(
        RpcDeadlines::default().config,
        Some(Duration::from_secs(600))
    );
}
