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

//! The server-side session of a client, against a loopback gRPC service that answers with the
//! server-side session ID of the server, which a test changes to restart it: the observed ID of
//! the requests, the check of the responses, and the close of the client on a restart.

mod common;

use std::collections::HashMap;
use std::collections::VecDeque;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use common::{FakeServer, FakeService};
use futures::{stream, Stream};
use prost::Message;
use spark_connect_core::bytes_codec::BytesCodec;
use spark_connect_core::{ChannelBuilder, RetryPolicy, SparkConnectClient, SparkErrorKind};
use spark_connect_proto::execute_plan_response::{ResponseType, ResultComplete};
use spark_connect_proto::{
    ConfigRequest, ConfigResponse, ExecutePlanRequest, ExecutePlanResponse, ReattachExecuteRequest,
    ReleaseExecuteRequest, ReleaseExecuteResponse,
};
use tonic::codegen::BoxFuture;
use tonic::server::{ServerStreamingService, UnaryService};
use tonic::{Code, Request, Response, Status};

const FIRST_SERVER_ID: &str = "server-1";
const RESTARTED_SERVER_ID: &str = "server-2";
type Frames = Pin<Box<dyn Stream<Item = Result<Vec<u8>, Status>> + Send>>;

struct State {
    /// The server-side session ID that the responses carry.
    server_session_id: Mutex<String>,
    /// Whether the next ExecutePlan streams its first response, and then drops.
    drop_next_execute: Mutex<bool>,
    /// The errors that the next calls of a method fail with, in order.
    failures: Mutex<VecDeque<(String, Status)>>,
    /// The method and the bytes of every request, in the order they arrived.
    requests: Mutex<Vec<(String, Vec<u8>)>>,
}

impl State {
    fn record(&self, method: &str, request: Vec<u8>) {
        self.requests
            .lock()
            .unwrap()
            .push((method.to_string(), request));
    }

    fn requests_of<T: Message + Default>(&self, method: &str) -> Vec<T> {
        self.requests
            .lock()
            .unwrap()
            .iter()
            .filter(|(m, _)| m == method)
            .map(|(_, bytes)| T::decode(bytes.as_slice()).unwrap())
            .collect()
    }

    fn server_session_id(&self) -> String {
        self.server_session_id.lock().unwrap().clone()
    }

    fn failure(&self, method: &str) -> Option<Status> {
        let mut failures = self.failures.lock().unwrap();
        let index = failures.iter().position(|(m, _)| m == method)?;
        failures.remove(index).map(|(_, status)| status)
    }

    fn take_drop_next_execute(&self) -> bool {
        std::mem::take(&mut *self.drop_next_execute.lock().unwrap())
    }
}

struct Fixture {
    state: Arc<State>,
    _server: FakeServer,
    address: std::net::SocketAddr,
}

impl Fixture {
    async fn start() -> Self {
        let state = Arc::new(State {
            server_session_id: Mutex::new(FIRST_SERVER_ID.to_string()),
            drop_next_execute: Mutex::new(false),
            failures: Mutex::new(VecDeque::new()),
            requests: Mutex::new(Vec::new()),
        });
        let server = service(state.clone()).serve().await;
        let address = server.address;
        Self {
            state,
            _server: server,
            address,
        }
    }

    /// A client of the fixture, with a session of its own.
    async fn client(&self) -> SparkConnectClient {
        let builder = ChannelBuilder::parse(&format!("sc://{}", self.address)).unwrap();
        let mut client = SparkConnectClient::connect(&builder).await.unwrap();
        client.set_retry_policy(RetryPolicy {
            max_retries: Some(2),
            initial_backoff_ms: 1,
            max_backoff_ms: Some(2),
            ..RetryPolicy::default()
        });
        client
    }
}

/// The session ID of a request of `method`, which the server echoes in its response.
fn session_of(method: &str, bytes: &[u8]) -> String {
    match method {
        "ExecutePlan" => ExecutePlanRequest::decode(bytes).unwrap().session_id,
        "ReattachExecute" => ReattachExecuteRequest::decode(bytes).unwrap().session_id,
        "ReleaseExecute" => ReleaseExecuteRequest::decode(bytes).unwrap().session_id,
        "Config" => ConfigRequest::decode(bytes).unwrap().session_id,
        _ => String::new(),
    }
}

/// The response of the whole result of an execution, as the server sends it.
fn result_response(session_id: &str, server_session_id: &str) -> Vec<u8> {
    ExecutePlanResponse {
        session_id: session_id.into(),
        server_side_session_id: server_session_id.into(),
        response_id: "complete".into(),
        response_type: Some(ResponseType::ResultComplete(ResultComplete {})),
        ..Default::default()
    }
    .encode_to_vec()
}

/// The first response of an execution that drops before its result is complete.
fn first_response(session_id: &str, server_session_id: &str) -> Vec<u8> {
    ExecutePlanResponse {
        session_id: session_id.into(),
        server_side_session_id: server_session_id.into(),
        response_id: "first".into(),
        ..Default::default()
    }
    .encode_to_vec()
}

/// The loopback service: ExecutePlan and ReattachExecute stream, and every other method is
/// unary.
fn service(state: Arc<State>) -> FakeService {
    let streamed = {
        let state = state.clone();
        move |method: &str| Streamed(state.clone(), method.to_string())
    };
    FakeService::new()
        .server_streaming("ExecutePlan", || BytesCodec, streamed.clone())
        .server_streaming("ReattachExecute", || BytesCodec, streamed)
        .fallback_unary(
            || BytesCodec,
            move |method| Unary(state.clone(), method.to_string()),
        )
}

/// ExecutePlan and ReattachExecute.
struct Streamed(Arc<State>, String);

impl ServerStreamingService<Vec<u8>> for Streamed {
    type Response = Vec<u8>;
    type ResponseStream = Frames;
    type Future = BoxFuture<Response<Frames>, Status>;

    fn call(&mut self, request: Request<Vec<u8>>) -> Self::Future {
        let bytes = request.into_inner();
        self.0.record(&self.1, bytes.clone());
        if let Some(status) = self.0.failure(&self.1) {
            return Box::pin(async { Err(status) });
        }
        let session_id = session_of(&self.1, &bytes);
        let server_session_id = self.0.server_session_id();
        let items: Vec<Result<Vec<u8>, Status>> =
            if self.1 == "ExecutePlan" && self.0.take_drop_next_execute() {
                vec![
                    Ok(first_response(&session_id, &server_session_id)),
                    Err(Status::unavailable("stream disconnected")),
                ]
            } else {
                vec![Ok(result_response(&session_id, &server_session_id))]
            };
        let frames: Frames = Box::pin(stream::iter(items));
        Box::pin(async { Ok(Response::new(frames)) })
    }
}

/// The unary RPCs that the tests use: Config and ReleaseExecute answer with their IDs.
struct Unary(Arc<State>, String);

impl UnaryService<Vec<u8>> for Unary {
    type Response = Vec<u8>;
    type Future = BoxFuture<Response<Vec<u8>>, Status>;

    fn call(&mut self, request: Request<Vec<u8>>) -> Self::Future {
        let bytes = request.into_inner();
        self.0.record(&self.1, bytes.clone());
        if let Some(status) = self.0.failure(&self.1) {
            return Box::pin(async { Err(status) });
        }
        let session_id = session_of(&self.1, &bytes);
        let server_side_session_id = self.0.server_session_id();
        let body = match self.1.as_str() {
            "Config" => ConfigResponse {
                session_id,
                server_side_session_id,
                ..Default::default()
            }
            .encode_to_vec(),
            "ReleaseExecute" => ReleaseExecuteResponse {
                session_id,
                server_side_session_id,
                ..Default::default()
            }
            .encode_to_vec(),
            _ => Vec::new(),
        };
        Box::pin(async { Ok(Response::new(body)) })
    }
}

/// A `google.rpc.Status` with an `ErrorInfo` of `error_class`, as the server sends the errors it
/// raises.
fn error_with_class(message: &str, error_class: &str) -> Status {
    error_with_class_and_id(message, error_class, None)
}

/// [`error_with_class`] that also names the `errorId` under which the server keeps the details.
fn error_with_class_and_id(message: &str, error_class: &str, error_id: Option<&str>) -> Status {
    #[derive(Clone, PartialEq, prost::Message)]
    struct ErrorInfo {
        #[prost(string, tag = "1")]
        reason: String,
        #[prost(string, tag = "2")]
        domain: String,
        #[prost(map = "string, string", tag = "3")]
        metadata: HashMap<String, String>,
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
    let mut metadata = HashMap::from([("errorClass".to_string(), error_class.to_string())]);
    if let Some(id) = error_id {
        metadata.insert("errorId".to_string(), id.to_string());
    }
    let info = ErrorInfo {
        reason: "SESSION_CHANGED".into(),
        domain: "org.apache.spark".into(),
        metadata,
    };
    let details = RpcStatus {
        code: Code::Internal as i32,
        message: message.into(),
        details: vec![prost_types::Any {
            type_url: "type.googleapis.com/google.rpc.ErrorInfo".into(),
            value: info.encode_to_vec(),
        }],
    }
    .encode_to_vec();
    Status::with_details(Code::Internal, message, details.into())
}

fn config_request(client: &SparkConnectClient) -> ConfigRequest {
    ConfigRequest {
        session_id: client.session_id().to_string(),
        ..Default::default()
    }
}

fn execute_request(client: &SparkConnectClient) -> ExecutePlanRequest {
    ExecutePlanRequest {
        session_id: client.session_id().to_string(),
        ..Default::default()
    }
}

/// Runs an execution of the client to its end, as the session does.
async fn run_execution(client: &SparkConnectClient) {
    let mut stream = client
        .execute_plan_reattachable(execute_request(client))
        .await
        .unwrap();
    while stream.message().await.unwrap().is_some() {}
}

#[tokio::test]
async fn requests_carry_the_server_side_session_id_observed_in_the_responses() {
    let fixture = Fixture::start().await;
    let client = fixture.client().await;

    // No response has set the ID yet, so the first request carries none.
    client.config(config_request(&client)).await.unwrap();
    client.config(config_request(&client)).await.unwrap();
    run_execution(&client).await;

    // An execution that drops after its first response: the reattach takes the observed ID of
    // the execution, as the reference client's reattach does.
    *fixture.state.drop_next_execute.lock().unwrap() = true;
    run_execution(&client).await;

    let configs = fixture.state.requests_of::<ConfigRequest>("Config");
    assert_eq!(
        configs
            .iter()
            .map(|r| r.client_observed_server_side_session_id.clone())
            .collect::<Vec<_>>(),
        [None, Some(FIRST_SERVER_ID.to_string())]
    );
    let executes = fixture
        .state
        .requests_of::<ExecutePlanRequest>("ExecutePlan");
    assert_eq!(executes.len(), 2);
    for request in &executes {
        assert_eq!(
            request.client_observed_server_side_session_id.as_deref(),
            Some(FIRST_SERVER_ID)
        );
    }
    let reattaches = fixture
        .state
        .requests_of::<ReattachExecuteRequest>("ReattachExecute");
    assert_eq!(reattaches.len(), 1);
    assert_eq!(
        reattaches[0]
            .client_observed_server_side_session_id
            .as_deref(),
        Some(FIRST_SERVER_ID)
    );
}

#[tokio::test]
async fn a_restarted_server_raises_and_closes_the_client() {
    let fixture = Fixture::start().await;
    let client = fixture.client().await;
    let clone = client.clone();
    client.config(config_request(&client)).await.unwrap();

    // The server restarts, so its responses carry another server-side session ID.
    *fixture.state.server_session_id.lock().unwrap() = RESTARTED_SERVER_ID.to_string();
    let error = client
        .config(config_request(&client))
        .await
        .expect_err("a response of a restarted server must raise");
    assert_eq!(error.kind, SparkErrorKind::AssertionError);
    assert_eq!(
        error.message,
        "Received incorrect server side session identifier for request. Please create a new \
         Spark Session to reconnect. (server-2 != server-1)"
    );
    assert!(client.is_closed() && clone.is_closed());

    // The client is closed from then on, and so are its clones, as PySpark's `_stub` raises.
    let error = client
        .config(config_request(&client))
        .await
        .expect_err("a closed client must raise");
    assert_eq!(error.get_condition().as_deref(), Some("NO_ACTIVE_SESSION"));
    let error = clone
        .config(config_request(&clone))
        .await
        .expect_err("a clone of a closed client must raise");
    assert_eq!(error.get_condition().as_deref(), Some("NO_ACTIVE_SESSION"));
    let error = client
        .execute_plan_reattachable(execute_request(&client))
        .await
        .err()
        .expect("an execution of a closed client must raise");
    assert_eq!(error.get_condition().as_deref(), Some("NO_ACTIVE_SESSION"));
}

#[tokio::test]
async fn a_restart_that_the_plan_compression_lookup_sees_sends_no_plan_as_pyspark() {
    use spark_connect_proto::analyze_plan_request::{Analyze, Schema};
    use spark_connect_proto::{plan, AnalyzePlanRequest, Plan, Relation};

    // As PySpark, which reads its `_stub` for an execution or an analysis once the plan is
    // built: the lookup of the plan compression of the first plan gets a response of the
    // restarted server, which closes the client, so the call raises NO_ACTIVE_SESSION and sends
    // nothing.
    let fixture = Fixture::start().await;
    let a_plan = || {
        Some(Plan {
            op_type: Some(plan::OpType::Root(Relation::default())),
        })
    };
    let mut conditions = Vec::new();
    for call in ["reattachable ExecutePlan", "ExecutePlan", "AnalyzePlan"] {
        *fixture.state.server_session_id.lock().unwrap() = FIRST_SERVER_ID.to_string();
        let client = fixture.client().await;
        client.config(config_request(&client)).await.unwrap();
        *fixture.state.server_session_id.lock().unwrap() = RESTARTED_SERVER_ID.to_string();
        let execution = ExecutePlanRequest {
            plan: a_plan(),
            ..execute_request(&client)
        };
        let error = match call {
            "reattachable ExecutePlan" => client.execute_plan_reattachable(execution).await.err(),
            "ExecutePlan" => client.execute_plan(execution).await.err(),
            _ => client
                .analyze_plan(AnalyzePlanRequest {
                    session_id: client.session_id().to_string(),
                    analyze: Some(Analyze::Schema(Schema { plan: a_plan() })),
                    ..Default::default()
                })
                .await
                .err(),
        };
        assert!(client.is_closed(), "{call}");
        conditions.push((call, error.and_then(|error| error.get_condition())));
    }
    let no_active_session = Some("NO_ACTIVE_SESSION".to_string());
    assert_eq!(
        conditions,
        [
            ("reattachable ExecutePlan", no_active_session.clone()),
            ("ExecutePlan", no_active_session.clone()),
            ("AnalyzePlan", no_active_session),
        ]
    );
    // Of each client, its Config and the lookup, and nothing after them.
    let methods: Vec<String> = fixture
        .state
        .requests
        .lock()
        .unwrap()
        .iter()
        .map(|(method, _)| method.clone())
        .collect();
    assert_eq!(methods, ["Config"; 6]);
}

#[tokio::test]
async fn a_session_changed_error_closes_the_client() {
    let fixture = Fixture::start().await;
    let client = fixture.client().await;
    fixture.state.failures.lock().unwrap().push_back((
        "Config".to_string(),
        error_with_class(
            "The existing Spark server driver instance has restarted. Please reconnect.",
            "INVALID_HANDLE.SESSION_CHANGED",
        ),
    ));

    let error = client
        .config(config_request(&client))
        .await
        .expect_err("a session change must raise");
    assert_eq!(
        error.get_condition().as_deref(),
        Some("INVALID_HANDLE.SESSION_CHANGED")
    );
    assert!(client.is_closed());

    let error = client
        .config(config_request(&client))
        .await
        .expect_err("a closed client must raise");
    assert_eq!(error.get_condition().as_deref(), Some("NO_ACTIVE_SESSION"));
}

#[tokio::test]
async fn a_session_changed_error_that_has_details_raises_no_active_session() {
    // The client is closed by the error, and the fetch of the details of the error goes through
    // it: PySpark raises the `NO_ACTIVE_SESSION` of that fetch from the call that failed.
    let fixture = Fixture::start().await;
    let client = fixture.client().await;
    fixture.state.failures.lock().unwrap().push_back((
        "Config".to_string(),
        error_with_class_and_id(
            "The existing Spark server driver instance has restarted. Please reconnect.",
            "INVALID_HANDLE.SESSION_CHANGED",
            Some("error-1"),
        ),
    ));

    let error = client
        .config(config_request(&client))
        .await
        .expect_err("a session change must raise");
    assert_eq!(error.get_condition().as_deref(), Some("NO_ACTIVE_SESSION"));
    assert!(client.is_closed());
    assert!(fixture
        .state
        .requests_of::<spark_connect_proto::FetchErrorDetailsRequest>("FetchErrorDetails")
        .is_empty());
}

#[tokio::test]
async fn a_client_without_session_tracking_passes_the_ids_of_its_requests_through() {
    // The transport-injection stub sends the requests of the reference client's session, which
    // tracks its server-side session itself, as over the grpcio stub that does no tracking.
    let fixture = Fixture::start().await;
    let mut client = fixture.client().await;
    client.set_session_tracking(false);
    let request = ConfigRequest {
        session_id: "the-reference-session".into(),
        client_observed_server_side_session_id: Some("observed-by-the-reference".into()),
        ..Default::default()
    };

    // A response of another session than the client's is not an error.
    client.config(request.clone()).await.unwrap();
    // Nor is a response of a restarted server, which does not close the client.
    *fixture.state.server_session_id.lock().unwrap() = RESTARTED_SERVER_ID.to_string();
    client.config(request.clone()).await.unwrap();
    // A session change is the error of its call, and does not close the client either.
    fixture.state.failures.lock().unwrap().push_back((
        "Config".to_string(),
        error_with_class(
            "The existing Spark server driver instance has restarted. Please reconnect.",
            "INVALID_HANDLE.SESSION_CHANGED",
        ),
    ));
    let error = client
        .config(request.clone())
        .await
        .expect_err("the session change is the error of its call");
    assert_eq!(
        error.get_condition().as_deref(),
        Some("INVALID_HANDLE.SESSION_CHANGED")
    );
    assert!(!client.is_closed());
    client.config(request.clone()).await.unwrap();

    // The requests keep the observed ID that the caller set. (The client reads the stack trace
    // confs of the error of the session change in a request of its own session.)
    let configs: Vec<ConfigRequest> = fixture
        .state
        .requests_of::<ConfigRequest>("Config")
        .into_iter()
        .filter(|config| config.session_id == "the-reference-session")
        .collect();
    assert_eq!(configs.len(), 4);
    for config in &configs {
        assert_eq!(
            config.client_observed_server_side_session_id.as_deref(),
            Some("observed-by-the-reference")
        );
    }
}

#[tokio::test]
async fn executions_ask_the_server_to_chunk_arrow_batches() {
    use spark_connect_proto::execute_plan_request::request_option::RequestOption;
    use spark_connect_proto::ResultChunkingOptions;

    let fixture = Fixture::start().await;
    let client = fixture.client().await;

    // A reattachable execution that drops after its first response, which is reattached, and a
    // plain ExecutePlan. The reattach has no request options to carry the option in.
    *fixture.state.drop_next_execute.lock().unwrap() = true;
    run_execution(&client).await;
    let mut stream = client.execute_plan(execute_request(&client)).await.unwrap();
    while stream.message().await.unwrap().is_some() {}

    let executes = fixture
        .state
        .requests_of::<ExecutePlanRequest>("ExecutePlan");
    assert_eq!(executes.len(), 2);
    for request in &executes {
        let chunking: Vec<ResultChunkingOptions> = request
            .request_options
            .iter()
            .filter_map(|option| match &option.request_option {
                Some(RequestOption::ResultChunkingOptions(options)) => Some(*options),
                _ => None,
            })
            .collect();
        assert_eq!(
            chunking,
            [ResultChunkingOptions {
                allow_arrow_batch_chunking: true,
                preferred_arrow_chunk_size: None,
            }]
        );
    }
}
