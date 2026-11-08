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

//! The user context of typed requests on the wire, against a loopback gRPC service that
//! records the bytes of every request; no Spark server required.

use std::collections::VecDeque;
use std::convert::Infallible;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use futures::{stream, Stream, StreamExt};
use prost::Message;
use spark_connect_core::bytes_codec::BytesCodec;
use spark_connect_core::{ChannelBuilder, RetryPolicy, SparkConnectClient};
use spark_connect_proto::execute_plan_response::{ResponseType, ResultComplete};
use spark_connect_proto::release_execute_request::Release;
use spark_connect_proto::{
    AnalyzePlanRequest, ConfigRequest, ExecutePlanRequest, ExecutePlanResponse,
    FetchErrorDetailsRequest, InterruptRequest, ReattachExecuteRequest, ReleaseExecuteRequest,
    UserContext,
};
use tokio::net::TcpListener;
use tokio::sync::Notify;
use tonic::body::Body;
use tonic::codegen::{http, BoxFuture, Service};
use tonic::server::{NamedService, ServerStreamingService, UnaryService};
use tonic::transport::Server;
use tonic::{Request, Response, Status};

const WAIT: Duration = Duration::from_secs(5);
const SERVICE: &str = "/spark.connect.SparkConnectService/";
type Frames = Pin<Box<dyn Stream<Item = Result<Vec<u8>, Status>> + Send>>;

/// What an ExecutePlan call streams. A ReattachExecute call streams the rest of the result.
#[derive(Clone, Copy)]
enum Execution {
    /// The whole result.
    Complete,
    /// A first response, then nothing until the client goes away.
    Open,
    /// A first response, then the stream drops, so that the client reattaches.
    Disconnect,
}

struct State {
    executions: Mutex<VecDeque<Execution>>,
    /// The method and the bytes of every request, in the order they arrived.
    requests: Mutex<Vec<(String, Vec<u8>)>>,
    changed: Notify,
}

impl State {
    fn record(&self, method: &str, request: Vec<u8>) {
        self.requests
            .lock()
            .unwrap()
            .push((method.to_string(), request));
        self.changed.notify_waiters();
    }

    fn requests_of(&self, method: &str) -> Vec<Vec<u8>> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|(m, _)| m == method)
            .map(|(_, request)| request.clone())
            .collect()
    }
}

struct Fixture {
    state: Arc<State>,
    server: tokio::task::JoinHandle<()>,
    address: std::net::SocketAddr,
}

impl Fixture {
    async fn start(executions: impl IntoIterator<Item = Execution>) -> Self {
        let state = Arc::new(State {
            executions: Mutex::new(executions.into_iter().collect()),
            requests: Mutex::new(Vec::new()),
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
        Self {
            state,
            server,
            address,
        }
    }

    /// A client of the fixture, with `user_id` in its connection string when given.
    async fn client(&self, user_id: Option<&str>) -> SparkConnectClient {
        let url = match user_id {
            Some(user_id) => format!("sc://{}/;user_id={user_id}", self.address),
            None => format!("sc://{}", self.address),
        };
        let builder = ChannelBuilder::parse(&url).unwrap();
        let mut client = tokio::time::timeout(WAIT, SparkConnectClient::connect(&builder))
            .await
            .expect("connecting timed out")
            .unwrap();
        client.set_retry_policy(RetryPolicy {
            max_retries: Some(2),
            initial_backoff_ms: 1,
            max_backoff_ms: Some(2),
            ..RetryPolicy::default()
        });
        client
    }

    /// The decoded requests of `method`, once `done` holds for them. ReleaseExecute requests
    /// are sent in the background, so they are waited for rather than counted at once.
    async fn wait_for<T: Message + Default>(
        &self,
        method: &str,
        done: impl Fn(&[T]) -> bool,
    ) -> Vec<T> {
        tokio::time::timeout(WAIT, async {
            loop {
                let changed = self.state.changed.notified();
                let requests: Vec<T> = self
                    .state
                    .requests_of(method)
                    .iter()
                    .map(|bytes| T::decode(bytes.as_slice()).unwrap())
                    .collect();
                if done(&requests) {
                    return requests;
                }
                changed.await;
            }
        })
        .await
        .unwrap_or_else(|_| panic!("the fixture did not receive the expected {method} requests"))
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        self.server.abort();
    }
}

fn response(id: &str, complete: bool) -> Vec<u8> {
    ExecutePlanResponse {
        response_id: id.into(),
        response_type: complete.then_some(ResponseType::ResultComplete(ResultComplete {})),
        ..Default::default()
    }
    .encode_to_vec()
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
        let method = request.uri().path().trim_start_matches(SERVICE).to_string();
        let state = self.0.clone();
        Box::pin(async move {
            let mut grpc = tonic::server::Grpc::new(BytesCodec);
            Ok(match method.as_str() {
                "ExecutePlan" | "ReattachExecute" => {
                    grpc.server_streaming(Streamed(state, method), request)
                        .await
                }
                _ => grpc.unary(Unary(state, method), request).await,
            })
        })
    }
}

/// ExecutePlan and ReattachExecute.
struct Streamed(Arc<State>, String);

impl ServerStreamingService<Vec<u8>> for Streamed {
    type Response = Vec<u8>;
    type ResponseStream = Frames;
    type Future = BoxFuture<Response<Frames>, Status>;

    fn call(&mut self, request: Request<Vec<u8>>) -> Self::Future {
        self.0.record(&self.1, request.into_inner());
        let execution = match self.1.as_str() {
            "ExecutePlan" => self.0.executions.lock().unwrap().pop_front(),
            _ => None,
        };
        let frames: Frames = match execution.unwrap_or(Execution::Complete) {
            Execution::Complete => Box::pin(stream::iter([Ok(response("complete", true))])),
            Execution::Open => {
                Box::pin(stream::iter([Ok(response("first", false))]).chain(stream::pending()))
            }
            Execution::Disconnect => Box::pin(stream::iter([
                Ok(response("first", false)),
                Err(Status::unavailable("stream disconnected")),
            ])),
        };
        Box::pin(async { Ok(Response::new(frames)) })
    }
}

/// The unary RPCs, each answered with its empty response message.
struct Unary(Arc<State>, String);

impl UnaryService<Vec<u8>> for Unary {
    type Response = Vec<u8>;
    type Future = BoxFuture<Response<Vec<u8>>, Status>;

    fn call(&mut self, request: Request<Vec<u8>>) -> Self::Future {
        self.0.record(&self.1, request.into_inner());
        Box::pin(async { Ok(Response::new(Vec::new())) })
    }
}

fn extension() -> prost_types::Any {
    prost_types::Any {
        type_url: "example.org/context".into(),
        value: vec![1, 2, 3],
    }
}

fn is_release_all(request: &ReleaseExecuteRequest) -> bool {
    matches!(request.release, Some(Release::ReleaseAll(_)))
}

/// Sends every typed request of the client with `context` as its user context, and returns
/// the user contexts that the server received, by method.
async fn send_typed_requests(
    fixture: &Fixture,
    client: &SparkConnectClient,
    context: &Option<UserContext>,
) -> Vec<(&'static str, Option<UserContext>)> {
    let execute = ExecutePlanRequest {
        session_id: client.session_id().to_string(),
        user_context: context.clone(),
        ..Default::default()
    };
    let mut stream = client.execute_plan(execute.clone()).await.unwrap();
    while stream.message().await.unwrap().is_some() {}
    let mut stream = client.execute_plan_reattachable(execute).await.unwrap();
    while stream.message().await.unwrap().is_some() {}
    client
        .analyze_plan(AnalyzePlanRequest {
            user_context: context.clone(),
            ..Default::default()
        })
        .await
        .unwrap();
    client
        .config(ConfigRequest {
            user_context: context.clone(),
            ..Default::default()
        })
        .await
        .unwrap();
    client
        .interrupt(InterruptRequest {
            user_context: context.clone(),
            ..Default::default()
        })
        .await
        .unwrap();
    client
        .fetch_error_details(FetchErrorDetailsRequest {
            user_context: context.clone(),
            ..Default::default()
        })
        .await
        .unwrap();
    let mut stream = client
        .reattach_execute(ReattachExecuteRequest {
            user_context: context.clone(),
            operation_id: "typed-operation".into(),
            ..Default::default()
        })
        .await
        .unwrap();
    while stream.message().await.unwrap().is_some() {}
    client
        .release_execute(ReleaseExecuteRequest {
            user_context: context.clone(),
            operation_id: "typed-operation".into(),
            ..Default::default()
        })
        .await
        .unwrap();

    let mut received = Vec::new();
    for request in fixture
        .wait_for::<ExecutePlanRequest>("ExecutePlan", |r| r.len() == 2)
        .await
    {
        received.push(("ExecutePlan", request.user_context));
    }
    for request in fixture
        .wait_for::<AnalyzePlanRequest>("AnalyzePlan", |r| r.len() == 1)
        .await
    {
        received.push(("AnalyzePlan", request.user_context));
    }
    for request in fixture
        .wait_for::<ConfigRequest>("Config", |r| r.len() == 1)
        .await
    {
        received.push(("Config", request.user_context));
    }
    for request in fixture
        .wait_for::<InterruptRequest>("Interrupt", |r| r.len() == 1)
        .await
    {
        received.push(("Interrupt", request.user_context));
    }
    for request in fixture
        .wait_for::<FetchErrorDetailsRequest>("FetchErrorDetails", |r| r.len() == 1)
        .await
    {
        received.push(("FetchErrorDetails", request.user_context));
    }
    for request in fixture
        .wait_for::<ReattachExecuteRequest>("ReattachExecute", |r| r.len() == 1)
        .await
    {
        received.push(("ReattachExecute", request.user_context));
    }
    // The typed release, and the release of the completed reattachable execution, which the
    // client sends in the background.
    let releases = fixture
        .wait_for::<ReleaseExecuteRequest>("ReleaseExecute", |r| {
            r.iter().any(|r| r.operation_id == "typed-operation")
                && r.iter()
                    .any(|r| r.operation_id != "typed-operation" && is_release_all(r))
        })
        .await;
    for request in releases {
        received.push(("ReleaseExecute", request.user_context));
    }
    received
}

#[tokio::test]
async fn typed_requests_take_a_missing_user_id_from_the_connection() {
    let partial = UserContext {
        user_name: "name".into(),
        extensions: vec![extension()],
        ..Default::default()
    };
    let explicit = UserContext {
        user_id: "explicit".into(),
        user_name: "name".into(),
        extensions: vec![extension()],
    };
    for (given, expected) in [
        (
            None,
            UserContext {
                user_id: "configured".into(),
                ..Default::default()
            },
        ),
        (
            Some(UserContext::default()),
            UserContext {
                user_id: "configured".into(),
                ..Default::default()
            },
        ),
        (
            Some(partial.clone()),
            UserContext {
                user_id: "configured".into(),
                ..partial
            },
        ),
        // An explicit user ID wins over the connection's.
        (Some(explicit.clone()), explicit),
    ] {
        let fixture = Fixture::start([Execution::Complete, Execution::Complete]).await;
        let client = fixture.client(Some("configured")).await;
        for (method, context) in send_typed_requests(&fixture, &client, &given).await {
            assert_eq!(context.as_ref(), Some(&expected), "{method} for {given:?}");
        }
    }
}

#[tokio::test]
async fn typed_requests_keep_their_user_context_with_an_empty_user_id() {
    let explicit = UserContext {
        user_id: "explicit".into(),
        user_name: "name".into(),
        extensions: vec![extension()],
    };
    for (given, expected) in [
        // The message is still sent, as for every RPC that the client builds itself. An
        // empty `user_id` in the connection string is no user, whatever SPARK_USER or USER is.
        (None, UserContext::default()),
        (Some(UserContext::default()), UserContext::default()),
        (Some(explicit.clone()), explicit),
    ] {
        let fixture = Fixture::start([Execution::Complete, Execution::Complete]).await;
        let client = fixture.client(Some("")).await;
        assert_eq!(client.user_id(), Some(""));
        for (method, context) in send_typed_requests(&fixture, &client, &given).await {
            assert_eq!(context.as_ref(), Some(&expected), "{method} for {given:?}");
        }
    }
}

/// The ReattachExecute and ReleaseExecute requests of a reattachable execution take the
/// user context of its ExecutePlan request, including the user ID taken from the connection.
#[tokio::test]
async fn reattach_and_release_take_the_user_context_of_the_execution() {
    let configured = Some(UserContext {
        user_id: "configured".into(),
        ..Default::default()
    });

    // A stream that drops after its first response: the client releases that response,
    // reattaches, and releases the execution once its result is complete.
    let fixture = Fixture::start([Execution::Disconnect]).await;
    let client = fixture.client(Some("configured")).await;
    let mut stream = client
        .execute_plan_reattachable(ExecutePlanRequest::default())
        .await
        .unwrap();
    let mut ids = Vec::new();
    while let Some(response) = stream.message().await.unwrap() {
        ids.push(response.response_id);
    }
    assert_eq!(ids, ["first", "complete"]);
    let execute = fixture
        .wait_for::<ExecutePlanRequest>("ExecutePlan", |r| r.len() == 1)
        .await;
    assert_eq!(execute[0].user_context, configured);
    let reattaches = fixture
        .wait_for::<ReattachExecuteRequest>("ReattachExecute", |r| r.len() == 1)
        .await;
    assert_eq!(reattaches[0].user_context, configured);
    assert_eq!(reattaches[0].last_response_id.as_deref(), Some("first"));
    let releases = fixture
        .wait_for::<ReleaseExecuteRequest>("ReleaseExecute", |r| {
            r.iter().any(is_release_all) && r.iter().any(|r| !is_release_all(r))
        })
        .await;
    for release in &releases {
        assert_eq!(release.user_context, configured);
        assert_eq!(
            Some(&release.operation_id),
            execute[0].operation_id.as_ref()
        );
    }
    drop(stream);

    // A stream dropped before its result is complete releases its execution.
    let fixture = Fixture::start([Execution::Open]).await;
    let client = fixture.client(Some("configured")).await;
    let mut stream = client
        .execute_plan_reattachable(ExecutePlanRequest::default())
        .await
        .unwrap();
    assert_eq!(
        stream.message().await.unwrap().unwrap().response_id,
        "first"
    );
    drop(stream);
    let releases = fixture
        .wait_for::<ReleaseExecuteRequest>("ReleaseExecute", |r| r.iter().any(is_release_all))
        .await;
    for release in &releases {
        assert_eq!(release.user_context, configured);
    }
}

#[tokio::test]
async fn raw_requests_are_sent_byte_exact() {
    let fixture = Fixture::start([Execution::Complete]).await;
    let client = fixture.client(Some("configured")).await;
    // Without a user context, and with an unknown field (2047, varint 1) that must survive,
    // as the request is not decoded and encoded again.
    let unknown_field = [0xf8, 0x7f, 0x01];
    let mut execute = ExecutePlanRequest::default().encode_to_vec();
    let mut analyze = AnalyzePlanRequest::default().encode_to_vec();
    let mut reattach = ReattachExecuteRequest::default().encode_to_vec();
    for request in [&mut execute, &mut analyze, &mut reattach] {
        request.extend_from_slice(&unknown_field);
    }
    let mut stream = client.execute_plan_raw(execute.clone()).await.unwrap();
    while stream.message().await.unwrap().is_some() {}
    client.analyze_plan_raw(analyze.clone()).await.unwrap();
    let mut stream = client.reattach_execute_raw(reattach.clone()).await.unwrap();
    while stream.message().await.unwrap().is_some() {}
    assert_eq!(fixture.state.requests_of("ExecutePlan"), [execute]);
    assert_eq!(fixture.state.requests_of("AnalyzePlan"), [analyze]);
    assert_eq!(fixture.state.requests_of("ReattachExecute"), [reattach]);
}
