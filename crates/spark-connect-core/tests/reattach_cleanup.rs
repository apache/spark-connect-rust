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

use std::collections::VecDeque;
use std::convert::Infallible;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use futures::{stream, Stream, StreamExt};
use spark_connect_core::{ChannelBuilder, RetryPolicy, SparkConnectClient};
use spark_connect_proto::execute_plan_response::{ResponseType, ResultComplete};
use spark_connect_proto::release_execute_request::Release;
use spark_connect_proto::{
    ExecutePlanRequest, ExecutePlanResponse, ReattachExecuteRequest, ReleaseExecuteRequest,
    ReleaseExecuteResponse, UserContext,
};
use tokio::net::TcpListener;
use tokio::sync::Notify;
use tonic::body::Body;
use tonic::codegen::{http, BoxFuture, Service};
use tonic::metadata::MetadataMap;
use tonic::server::{NamedService, ServerStreamingService, UnaryService};
use tonic::transport::Server;
use tonic::{Code, Request, Response, Status};

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
}

#[derive(Clone)]
struct CapturedRelease {
    request: ReleaseExecuteRequest,
    metadata: MetadataMap,
}

struct State {
    scenario: Scenario,
    releases: Mutex<Vec<CapturedRelease>>,
    reattaches: Mutex<Vec<ReattachExecuteRequest>>,
    release_errors: Mutex<VecDeque<Code>>,
    release_finished: AtomicUsize,
    block_release: AtomicBool,
    block_reattach: AtomicBool,
    release_gate: Notify,
    reattach_gate: Notify,
    changed: Notify,
}

impl State {
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
    server: tokio::task::JoinHandle<()>,
    client: SparkConnectClient,
}

impl Fixture {
    async fn start(scenario: Scenario) -> Self {
        let state = Arc::new(State {
            scenario,
            releases: Mutex::new(Vec::new()),
            reattaches: Mutex::new(Vec::new()),
            release_errors: Mutex::new(VecDeque::new()),
            release_finished: AtomicUsize::new(0),
            block_release: AtomicBool::new(false),
            block_reattach: AtomicBool::new(false),
            release_gate: Notify::new(),
            reattach_gate: Notify::new(),
            changed: Notify::new(),
        });
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let incoming = stream::unfold(listener, |listener| async {
            Some((listener.accept().await.map(|(socket, _)| socket), listener))
        });
        let service = TestService(state.clone());
        let server = tokio::spawn(async move {
            Server::builder()
                .add_service(service)
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        let builder = ChannelBuilder::parse(&format!(
            "sc://{address}/;user_id=transport-user;token=fixture-token;x-cleanup=fixture-value"
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
            server,
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
        self.wait_for(|s| s.all_releases().len() == expected).await;
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

impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.abort();
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
    assert_eq!(
        captured.metadata.get("authorization").unwrap(),
        "Bearer fixture-token"
    );
    assert_eq!(captured.metadata.get("user_id").unwrap(), "transport-user");
    assert_eq!(captured.metadata.get("x-cleanup").unwrap(), "fixture-value");
    assert!(captured.metadata.get("user-agent").is_some());
}

#[derive(Clone)]
struct TestService(Arc<State>);

impl NamedService for TestService {
    const NAME: &'static str = "spark.connect.SparkConnectService";
}

impl Service<http::Request<Body>> for TestService {
    type Response = http::Response<Body>;
    type Error = Infallible;
    type Future = BoxFuture<Self::Response, Self::Error>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: http::Request<Body>) -> Self::Future {
        let state = self.0.clone();
        match request.uri().path() {
            "/spark.connect.SparkConnectService/ExecutePlan" => Box::pin(async move {
                let mut grpc = tonic::server::Grpc::new(tonic_prost::ProstCodec::default());
                Ok(grpc.server_streaming(Execute(state), request).await)
            }),
            "/spark.connect.SparkConnectService/ReattachExecute" => Box::pin(async move {
                let mut grpc = tonic::server::Grpc::new(tonic_prost::ProstCodec::default());
                Ok(grpc.server_streaming(Reattach(state), request).await)
            }),
            "/spark.connect.SparkConnectService/ReleaseExecute" => Box::pin(async move {
                let mut grpc = tonic::server::Grpc::new(tonic_prost::ProstCodec::default());
                Ok(grpc.unary(ReleaseRpc(state), request).await)
            }),
            _ => Box::pin(async {
                Ok(http::Response::builder()
                    .status(200)
                    .header("grpc-status", "12")
                    .header("content-type", "application/grpc")
                    .body(Body::empty())
                    .unwrap())
            }),
        }
    }
}

struct Execute(Arc<State>);

impl ServerStreamingService<ExecutePlanRequest> for Execute {
    type Response = ExecutePlanResponse;
    type ResponseStream = Responses;
    type Future = BoxFuture<Response<Responses>, Status>;

    fn call(&mut self, _: Request<ExecutePlanRequest>) -> Self::Future {
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
        state.reattaches.lock().unwrap().push(request.into_inner());
        state.changed.notify_waiters();
        Box::pin(async move {
            if state.block_reattach.load(Ordering::SeqCst) {
                state.reattach_gate.notified().await;
            }
            let responses: Responses = Box::pin(stream::iter([Ok(response("complete", true))]));
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
async fn dropping_unfinished_stream_releases_inside_or_outside_runtime() {
    for outside_runtime in [false, true] {
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
            std::thread::spawn(move || drop(stream)).join().unwrap();
        } else {
            drop(stream);
        }
        fixture.assert_release_all_count(1).await;
        assert_identity(&fixture.state.all_releases()[0]);
    }
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
async fn detached_stream_reattaches_before_releasing_the_execution() {
    for scenario in [Scenario::Eof, Scenario::Disconnect] {
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
        let reattach = fixture.state.reattaches.lock().unwrap()[0].clone();
        assert_eq!(
            reattach.operation_id,
            execute_request().operation_id.unwrap()
        );
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
}
