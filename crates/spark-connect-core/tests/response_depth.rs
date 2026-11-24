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

//! The responses of the client nest within the limit of the levels of messages of prost's and
//! upb's decoders, of a loopback service whose responses nest as deep as a test asks: the typed
//! responses of the generated client and the raw responses of the passthrough RPCs decode up to
//! 100 levels below them, and a deeper one is the client's error of prost's limit, which is
//! PySpark's error of a response that gRPC fails to deserialize, of any depth, without a crash.
//! Within a limit that the Python binding raises, a response decodes to it, but of kept messages
//! nested at most `MAX_KEPT_DEPTH` levels.

mod common;

use std::pin::Pin;
use std::sync::{Arc, Mutex};

use common::{FakeServer, FakeService};
use futures::{stream, Stream};
use prost::Message;
use spark_connect_core::bytes_codec::BytesCodec;
use spark_connect_core::error::GrpcCall;
use spark_connect_core::{ChannelBuilder, SparkConnectClient, SparkError, SparkErrorKind};
use spark_connect_proto::analyze_plan_request::{Analyze, SparkVersion};
use spark_connect_proto::config_request::operation::OpType;
use spark_connect_proto::config_request::{GetAll, Operation};
use spark_connect_proto::{
    AnalyzePlanRequest, ConfigRequest, ExecutePlanRequest, ReattachExecuteRequest,
};
use tonic::codegen::BoxFuture;
use tonic::server::{ServerStreamingService, UnaryService};
use tonic::{Request, Response, Status};

type Frames = Pin<Box<dyn Stream<Item = Result<Vec<u8>, Status>> + Send>>;

/// The tests share the limit of the process (`spark_connect_proto::set_depth_limit`), which some
/// raise: one at a time.
static SERIAL: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn varint(mut value: u64) -> Vec<u8> {
    let mut out = Vec::new();
    while value >= 0x80 {
        out.push((value as u8) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
    out
}

fn key(number: u32, wire_type: u8) -> Vec<u8> {
    varint((u64::from(number) << 3) | u64::from(wire_type))
}

fn delimited(number: u32, value: &[u8]) -> Vec<u8> {
    [key(number, 2), varint(value.len() as u64), value.to_vec()].concat()
}

/// `levels` messages, each the field `numbers[level % numbers.len()]` of the one above, around
/// `inner`, built without recursion.
fn nested(levels: usize, numbers: &[u32], inner: &[u8]) -> Vec<u8> {
    let number = |level: usize| numbers[level % numbers.len()];
    let mut sizes = vec![0; levels];
    let mut size = inner.len();
    for level in (0..levels).rev() {
        sizes[level] = size;
        size += key(number(level), 2).len() + varint(size as u64).len();
    }
    let mut out = Vec::with_capacity(size);
    for (level, &size) in sizes.iter().enumerate() {
        out.extend(key(number(level), 2));
        out.extend(varint(size as u64));
    }
    out.extend_from_slice(inner);
    out
}

/// Unknown groups (of the field 1000) nested `levels` deep, below the message that holds them.
fn groups(levels: usize) -> Vec<u8> {
    [key(1000, 3).repeat(levels), key(1000, 4).repeat(levels)].concat()
}

/// A DataType of `arrays` nested arrays of a long (DataType.array 20, Array.element_type 1, and
/// DataType.long 7 of an empty Long): the Long is `2 * arrays + 1` levels below it.
fn arrays(arrays: usize) -> Vec<u8> {
    nested(2 * arrays, &[20, 1], &delimited(7, &[]))
}

/// What the service answers, which a test sets.
#[derive(Default)]
struct State {
    /// The fields of the AnalyzePlanResponse, the ConfigResponse of a request of no keys
    /// (GetAll), and the ExecutePlanResponse (and ReattachExecute's) after the IDs.
    analyze: Mutex<Vec<u8>>,
    config: Mutex<Vec<u8>>,
    execute: Mutex<Vec<u8>>,
    /// The status that AnalyzePlan fails with instead, of its details.
    analyze_failure: Mutex<Option<(String, Vec<u8>)>>,
}

/// The session ID of a request, its field 1 in every request.
fn session_of(bytes: &[u8]) -> Vec<u8> {
    // ConfigRequest stands for any request: session_id is field 1 of each.
    ConfigRequest::decode(bytes)
        .map(|request| delimited(1, request.session_id.as_bytes()))
        .unwrap_or_default()
}

struct Unary(Arc<State>, String);

impl UnaryService<Vec<u8>> for Unary {
    type Response = Vec<u8>;
    type Future = BoxFuture<Response<Vec<u8>>, Status>;

    fn call(&mut self, request: Request<Vec<u8>>) -> Self::Future {
        let bytes = request.into_inner();
        let head = session_of(&bytes);
        // The server-side session ID of each response: AnalyzePlanResponse 15, ConfigResponse 4.
        let server = |number| delimited(number, b"server");
        let body = match self.1.as_str() {
            "AnalyzePlan" => {
                if let Some((message, details)) = self.0.analyze_failure.lock().unwrap().clone() {
                    let status =
                        Status::with_details(tonic::Code::Internal, message, details.into());
                    return Box::pin(async { Err(status) });
                }
                [head, server(15), self.0.analyze.lock().unwrap().clone()].concat()
            }
            // The deep response of a request of all the settings, and none of the others, such
            // as the client's lookup of the plan compression.
            "Config" => {
                let all = ConfigRequest::decode(bytes.as_slice()).is_ok_and(|request| {
                    matches!(
                        request.operation.and_then(|o| o.op_type),
                        Some(OpType::GetAll(_))
                    )
                });
                let config = if all {
                    self.0.config.lock().unwrap().clone()
                } else {
                    Vec::new()
                };
                [head, server(4), config].concat()
            }
            _ => head,
        };
        Box::pin(async { Ok(Response::new(body)) })
    }
}

struct Streamed(Arc<State>);

impl ServerStreamingService<Vec<u8>> for Streamed {
    type Response = Vec<u8>;
    type ResponseStream = Frames;
    type Future = BoxFuture<Response<Frames>, Status>;

    fn call(&mut self, request: Request<Vec<u8>>) -> Self::Future {
        // ExecutePlanResponse.server_side_session_id (15).
        let head = [session_of(&request.into_inner()), delimited(15, b"server")].concat();
        let first = [head.clone(), self.0.execute.lock().unwrap().clone()].concat();
        // ExecutePlanResponse.result_complete (14).
        let complete = [head, delimited(14, &[])].concat();
        let frames: Frames = Box::pin(stream::iter(vec![Ok(first), Ok(complete)]));
        Box::pin(async { Ok(Response::new(frames)) })
    }
}

struct Fixture {
    state: Arc<State>,
    _server: FakeServer,
    client: SparkConnectClient,
}

impl Fixture {
    async fn start() -> Self {
        let state = Arc::new(State::default());
        let streamed = {
            let state = state.clone();
            move |_: &str| Streamed(state.clone())
        };
        let unary = {
            let state = state.clone();
            move |method: &str| Unary(state.clone(), method.to_string())
        };
        let server = FakeService::new()
            .server_streaming("ExecutePlan", || BytesCodec, streamed.clone())
            .server_streaming("ReattachExecute", || BytesCodec, streamed)
            .fallback_unary(|| BytesCodec, unary)
            .serve()
            .await;
        let builder = ChannelBuilder::parse(&format!("sc://{}", server.address)).unwrap();
        let client = SparkConnectClient::connect(&builder).await.unwrap();
        Self {
            state,
            _server: server,
            client,
        }
    }

    fn analyze_request(&self) -> AnalyzePlanRequest {
        AnalyzePlanRequest {
            session_id: self.client.session_id().to_string(),
            analyze: Some(Analyze::SparkVersion(SparkVersion {})),
            ..Default::default()
        }
    }

    async fn analyze(&self, fields: Vec<u8>) -> Result<(), SparkError> {
        *self.state.analyze.lock().unwrap() = fields;
        self.client
            .analyze_plan(self.analyze_request())
            .await
            .map(drop)
    }

    async fn config(&self, fields: Vec<u8>) -> Result<(), SparkError> {
        *self.state.config.lock().unwrap() = fields;
        let request = ConfigRequest {
            session_id: self.client.session_id().to_string(),
            operation: Some(Operation {
                op_type: Some(OpType::GetAll(GetAll { prefix: None })),
            }),
            ..Default::default()
        };
        self.client.config(request).await.map(drop)
    }

    /// The responses of an ExecutePlan, of its first of `fields`, read to the end, or the error
    /// of the stream, as the client converts the status of a stream.
    async fn execute(&self, fields: Vec<u8>) -> Result<usize, SparkError> {
        *self.state.execute.lock().unwrap() = fields;
        let request = ExecutePlanRequest {
            session_id: self.client.session_id().to_string(),
            operation_id: Some("00000000-0000-4000-8000-000000000001".into()),
            plan: Some(spark_connect_proto::Plan {
                op_type: Some(spark_connect_proto::plan::OpType::Root(
                    spark_connect_proto::Relation::default(),
                )),
            }),
            ..Default::default()
        };
        let stream = self.client.execute_plan(request).await?;
        read(stream).await
    }

    async fn reattach(&self, fields: Vec<u8>) -> Result<usize, SparkError> {
        *self.state.execute.lock().unwrap() = fields;
        let request = ReattachExecuteRequest {
            session_id: self.client.session_id().to_string(),
            operation_id: "00000000-0000-4000-8000-000000000001".into(),
            ..Default::default()
        };
        let stream = self.client.reattach_execute(request).await?;
        read(stream).await
    }

    /// The raw RPCs: of the bytes of their requests, the bytes of their responses, unparsed.
    async fn execute_raw(&self, fields: Vec<u8>) -> Result<usize, Status> {
        *self.state.execute.lock().unwrap() = fields;
        let request = ExecutePlanRequest {
            session_id: self.client.session_id().to_string(),
            ..Default::default()
        };
        let mut stream = self
            .client
            .execute_plan_raw(request.encode_to_vec())
            .await
            .unwrap();
        let mut count = 0;
        while stream.message().await?.is_some() {
            count += 1;
        }
        Ok(count)
    }

    async fn reattach_raw(&self, fields: Vec<u8>) -> Result<usize, Status> {
        *self.state.execute.lock().unwrap() = fields;
        let request = ReattachExecuteRequest {
            session_id: self.client.session_id().to_string(),
            ..Default::default()
        };
        let mut stream = self
            .client
            .reattach_execute_raw(request.encode_to_vec())
            .await
            .unwrap();
        let mut count = 0;
        while stream.message().await?.is_some() {
            count += 1;
        }
        Ok(count)
    }

    async fn analyze_raw(&self, fields: Vec<u8>) -> Result<Vec<u8>, SparkError> {
        *self.state.analyze.lock().unwrap() = fields;
        self.client
            .analyze_plan_raw(self.analyze_request().encode_to_vec())
            .await
    }
}

async fn read(
    mut stream: tonic::Streaming<spark_connect_proto::ExecutePlanResponse>,
) -> Result<usize, SparkError> {
    let mut count = 0;
    loop {
        match stream.message().await {
            Ok(Some(_)) => count += 1,
            Ok(None) => return Ok(count),
            Err(status) => {
                return Err(SparkError::from_grpc_status_of(
                    status,
                    GrpcCall::ServerStreaming,
                ))
            }
        }
    }
}

/// PySpark's error of a response that gRPC's deserializer refuses, of a call of `class`.
fn deserializing_error(class: &str) -> String {
    format!(
        "<{class} of RPC that terminated with:\n\tstatus = StatusCode.INTERNAL\n\tdetails = \
         \"Exception deserializing response!\"\n\tdebug_error_string = \"None\"\n>"
    )
}

/// That `error` is the client's of a response past the limit: the Rust API's of prost's error of
/// its limit, `prost`, and PySpark's error of the call of `class` that the binding raises.
fn assert_past_the_limit(error: &SparkError, prost: &str, class: &str) {
    assert_eq!(error.message(), format!("[Internal error] {prost}"));
    assert_eq!(error.grpc_code(), Some(tonic::Code::Internal as i32));
    assert_eq!(
        error.pyspark_message.as_deref(),
        Some(deserializing_error(class).as_str())
    );
}

const NO_FIELDS: &str = "failed to decode Protobuf message: recursion limit reached";

#[tokio::test]
async fn typed_responses_decode_within_100_levels_and_fail_past_them_as_prost() {
    let _serial = SERIAL.lock().await;
    let fixture = Fixture::start().await;
    // Unknown groups are levels of upb's and prost's decoders, of no field that prost names.
    fixture.analyze(groups(100)).await.unwrap();
    let error = fixture.analyze(groups(101)).await.unwrap_err();
    assert_past_the_limit(&error, NO_FIELDS, "_InactiveRpcError");
    fixture.config(groups(100)).await.unwrap();
    let error = fixture.config(groups(101)).await.unwrap_err();
    assert_past_the_limit(&error, NO_FIELDS, "_InactiveRpcError");
    assert_eq!(fixture.execute(groups(100)).await.unwrap(), 2);
    let error = fixture.execute(groups(101)).await.unwrap_err();
    assert_past_the_limit(&error, NO_FIELDS, "_MultiThreadedRendezvous");
    assert_eq!(fixture.reattach(groups(100)).await.unwrap(), 2);
    let error = fixture.reattach(groups(101)).await.unwrap_err();
    assert_past_the_limit(&error, NO_FIELDS, "_MultiThreadedRendezvous");

    // A schema of nested arrays: AnalyzePlanResponse.schema (2) -> Schema.schema (1) -> the
    // DataType at level 2, whose Long of 49 arrays is at 101.
    let schema = |arrays_of: usize| delimited(2, &delimited(1, &arrays(arrays_of)));
    fixture.analyze(schema(48)).await.unwrap();
    let error = fixture.analyze(schema(49)).await.unwrap_err();
    let prost = format!(
        "failed to decode Protobuf message: DataType.kind: {}Schema.schema: \
         AnalyzePlanResponse.result: recursion limit reached",
        "Array.element_type: DataType.kind: ".repeat(49)
    );
    assert_past_the_limit(&error, &prost, "_InactiveRpcError");
    // ExecutePlanResponse.schema (7): the DataType at level 1, whose Long of 49 arrays is at 100.
    assert_eq!(fixture.execute(delimited(7, &arrays(49))).await.unwrap(), 2);
    let error = fixture
        .execute(delimited(7, &arrays(50)))
        .await
        .unwrap_err();
    let prost = format!(
        "failed to decode Protobuf message: {}ExecutePlanResponse.schema: recursion limit \
         reached",
        "Array.element_type: DataType.kind: ".repeat(50)
    );
    assert_past_the_limit(&error, &prost, "_MultiThreadedRendezvous");
}

#[tokio::test]
async fn raw_responses_are_scanned_of_the_type_of_the_responses_of_their_rpc() {
    let _serial = SERIAL.lock().await;
    let fixture = Fixture::start().await;
    assert_eq!(fixture.execute_raw(groups(100)).await.unwrap(), 2);
    assert_eq!(fixture.reattach_raw(groups(100)).await.unwrap(), 2);
    fixture.analyze_raw(groups(100)).await.unwrap();
    for status in [
        fixture.execute_raw(groups(101)).await.unwrap_err(),
        fixture.reattach_raw(groups(101)).await.unwrap_err(),
    ] {
        assert_eq!(status.code(), tonic::Code::Internal);
        assert_eq!(status.message(), NO_FIELDS);
        assert!(spark_connect_proto::is_client_decode_error(&status));
    }
    let error = fixture.analyze_raw(groups(101)).await.unwrap_err();
    assert_past_the_limit(&error, NO_FIELDS, "_InactiveRpcError");
    // Of the schema of the response of each RPC.
    let prost = format!(
        "failed to decode Protobuf message: {}ExecutePlanResponse.schema: recursion limit \
         reached",
        "Array.element_type: DataType.kind: ".repeat(50)
    );
    let status = fixture
        .execute_raw(delimited(7, &arrays(50)))
        .await
        .unwrap_err();
    assert_eq!(status.message(), prost);
    let error = fixture
        .analyze_raw(delimited(2, &delimited(1, &arrays(49))))
        .await
        .unwrap_err();
    assert!(error
        .message()
        .ends_with("Schema.schema: AnalyzePlanResponse.result: recursion limit reached"));
}

#[tokio::test]
async fn a_response_of_a_million_levels_fails_without_a_crash() {
    let _serial = SERIAL.lock().await;
    let fixture = Fixture::start().await;
    let deep = groups(1_000_000);
    let error = fixture.analyze(deep.clone()).await.unwrap_err();
    assert_past_the_limit(&error, NO_FIELDS, "_InactiveRpcError");
    let error = fixture.execute(deep.clone()).await.unwrap_err();
    assert_past_the_limit(&error, NO_FIELDS, "_MultiThreadedRendezvous");
    assert!(fixture.execute_raw(deep).await.is_err());
    // Of known messages too.
    let error = fixture
        .execute(delimited(7, &arrays(500_000)))
        .await
        .unwrap_err();
    assert!(error
        .message()
        .ends_with("ExecutePlanResponse.schema: recursion limit reached"));
}

/// Restores the limit of the process when it drops.
struct Limit(u32);

impl Limit {
    fn raise(levels: u32) -> Self {
        let before = spark_connect_proto::depth_limit();
        spark_connect_proto::set_depth_limit(levels);
        Limit(before)
    }
}

impl Drop for Limit {
    fn drop(&mut self) {
        spark_connect_proto::set_depth_limit(self.0);
    }
}

/// Whether prost of this build has its own limit of 100 levels: it has none of the
/// `no-recursion-limit` feature, which the Python binding and the tests of `apache-spark-connect`
/// enable, and `cargo test` unifies with the crates that it builds. Of its limit, prost fails what
/// a raised limit of the client takes.
fn prost_has_limit() -> bool {
    let response = delimited(7, &arrays(60));
    spark_connect_proto::ExecutePlanResponse::decode(response.as_slice()).is_err()
}

#[tokio::test]
async fn a_raised_limit_decodes_to_it_on_a_grown_stack_of_kept_messages_up_to_the_bound() {
    let _serial = SERIAL.lock().await;
    if prost_has_limit() {
        return;
    }
    let fixture = Fixture::start().await;
    // upb's limit once SetAllowOversizeProtos(True) is called, which the Python binding follows.
    let _limit = Limit::raise(65535);
    fixture.analyze(groups(65535)).await.unwrap();
    assert_eq!(fixture.execute(groups(65535)).await.unwrap(), 2);
    let error = fixture.execute(groups(65536)).await.unwrap_err();
    assert_past_the_limit(&error, NO_FIELDS, "_MultiThreadedRendezvous");
    // Kept messages nest up to MAX_KEPT_DEPTH (1024), and past it are the client's
    // RecursionError: the Long of 511 arrays of an ExecutePlanResponse.schema is at level 1024.
    assert_eq!(spark_connect_proto::MAX_KEPT_DEPTH, 1024);
    assert_eq!(
        fixture.execute(delimited(7, &arrays(511))).await.unwrap(),
        2
    );
    let error = fixture
        .execute(delimited(7, &arrays(512)))
        .await
        .unwrap_err();
    assert_eq!(error.kind, SparkErrorKind::RecursionError);
    assert_eq!(
        error.message(),
        "maximum recursion depth exceeded while decoding a response; the messages of a \
         response may be nested at most 1024 levels deep"
    );
    assert_eq!(error.pyspark_message, None);
    let error = fixture
        .analyze(delimited(2, &delimited(1, &arrays(600))))
        .await
        .unwrap_err();
    assert_eq!(error.kind, SparkErrorKind::RecursionError);
    let status = fixture
        .execute_raw(delimited(7, &arrays(512)))
        .await
        .unwrap_err();
    assert!(spark_connect_proto::kept_too_deep(&status).is_some());
    assert!(!spark_connect_proto::is_client_decode_error(&status));
}

#[tokio::test]
async fn the_details_of_a_failure_decode_within_the_limit() {
    let _serial = SERIAL.lock().await;
    let fixture = Fixture::start().await;
    // google.rpc.Status: code=1, message=2, details=3 of an Any (type_url=1, value=2) of an
    // ErrorInfo (reason=1, domain=2, metadata=3 of entries of key=1 and value=2).
    let info = [
        delimited(1, b"reason"),
        delimited(
            3,
            &[delimited(1, b"errorClass"), delimited(2, b"X")].concat(),
        ),
    ]
    .concat();
    let any = [
        delimited(1, b"type.googleapis.com/google.rpc.ErrorInfo"),
        delimited(2, &info),
    ]
    .concat();
    let status = |extra: &[u8]| {
        [
            key(1, 0),
            varint(13),
            delimited(2, b"failed"),
            delimited(3, &[any.clone(), extra.to_vec()].concat()),
        ]
        .concat()
    };
    // Of no more levels than a header of HTTP/2 holds.
    for (extra, read) in [
        (groups(99), true),
        (groups(100), false),
        (groups(3000), false),
    ] {
        *fixture.state.analyze_failure.lock().unwrap() = Some(("failed".into(), status(&extra)));
        let error = fixture.analyze(Vec::new()).await.unwrap_err();
        // The ErrorInfo of a status within the limit is read; past it, the status is of no
        // ErrorInfo, as one of details that do not decode.
        assert_eq!(error.server.is_some(), read, "{} levels", extra.len() / 4);
        assert_eq!(error.error_class == "X", read);
    }
    *fixture.state.analyze_failure.lock().unwrap() = None;
}

#[test]
fn a_call_on_a_thread_of_a_small_stack_polls_on_a_stack_of_its_own() {
    // The polls of the futures of an RPC (tonic's, the deadline's and the retries') take frames of
    // tens of KiB each, more than a thread of 128 KiB holds, such as a Python thread of a small
    // `threading.stack_size`: a wait of such a thread polls on a segment of its own. Of a response
    // within the limit, and, of a raised limit, of 65535 levels and of kept messages at the bound,
    // whose decode and drop take a stack of their own too.
    let _serial = SERIAL.blocking_lock();
    let fixture = std::sync::Arc::new(spark_connect_core::runtime::block_on(Fixture::start()));
    let on_small_thread = |fields: Vec<u8>| {
        let fixture = fixture.clone();
        std::thread::Builder::new()
            .stack_size(128 << 10)
            .spawn(move || {
                let analyzed =
                    spark_connect_core::runtime::block_on(fixture.analyze(fields.clone()));
                let executed = spark_connect_core::runtime::block_on(fixture.execute(fields));
                (analyzed.map(drop), executed.map(drop))
            })
            .unwrap()
            .join()
            .unwrap()
    };
    let (analyzed, executed) = on_small_thread(groups(100));
    assert!(analyzed.is_ok() && executed.is_ok());
    let (analyzed, executed) = on_small_thread(groups(101));
    assert!(analyzed.is_err() && executed.is_err());
    if prost_has_limit() {
        return;
    }
    let _limit = Limit::raise(65535);
    let (analyzed, executed) = on_small_thread(groups(65535));
    assert!(analyzed.is_ok() && executed.is_ok());
    // ExecutePlanResponse.schema of 511 arrays, of its Long at level 1024 (the AnalyzePlanResponse
    // of the same fields reads them as an unknown field 7 of the response).
    let (_, executed) = on_small_thread(delimited(7, &arrays(511)));
    assert!(executed.is_ok());
}
