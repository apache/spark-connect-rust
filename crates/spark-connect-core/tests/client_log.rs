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

//! The records of PySpark's client that the client logs of its calls (see
//! `spark_connect_core::client_log`): of the plan compression, the plans, the requests and the
//! responses of the executions, the plans of the analyses, the deadlines of the streams and the
//! errors, in PySpark's order. No Spark server required: a loopback tonic service answers the
//! calls. The listener is of the process, so the tests take turns.

mod common;

use std::collections::{HashMap, VecDeque};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Duration;

use common::{FakeServer, FakeService};
use futures::{stream, Stream, StreamExt};
use prost::Message as _;
use spark_connect_core::client_log::{AnalysisLog, ExecutionLog, Level, Listener, Message};
use spark_connect_core::retries::Delivery;
use spark_connect_core::{ChannelBuilder, RetryPolicy, RpcDeadlines, SparkConnectClient};
use spark_connect_proto::execute_plan_response::{
    ArrowBatch, Metrics, ObservedMetrics, ResponseType, ResultComplete, SqlCommandResult,
};
use spark_connect_proto::{
    analyze_plan_request, analyze_plan_response, command, plan, AnalyzePlanRequest,
    AnalyzePlanResponse, Command, ConfigRequest, ConfigResponse, DataType, ExecutePlanRequest,
    ExecutePlanResponse, KeyValue, Plan, ReattachExecuteRequest, Relation, ReleaseExecuteRequest,
    ReleaseExecuteResponse, SqlCommand, UserContext,
};
use tonic::codegen::BoxFuture;
use tonic::server::{ServerStreamingService, UnaryService};
use tonic::{Code, Request, Response, Status};
use tonic_prost::ProstCodec;

const SESSION: &str = "00000000-0000-4000-8000-000000000001";
const OPERATION: &str = "00000000-0000-4000-8000-000000000002";
const WAIT: Duration = Duration::from_secs(5);
type Responses = Pin<Box<dyn Stream<Item = Result<ExecutePlanResponse, Status>> + Send>>;

async fn bounded<T>(future: impl Future<Output = T>) -> T {
    tokio::time::timeout(WAIT, future)
        .await
        .expect("test RPC waited past the bound")
}

/// The records of the listener of a test, of the level that it logs from, and the text that a
/// record starts with whose delivery fails, once, as the exception of a handler of the logger.
struct Recorder {
    level: Level,
    records: Mutex<Vec<(Level, Message)>>,
    failing: Mutex<Option<&'static str>>,
}

impl Listener for Recorder {
    fn enabled_for(&self, level: Level) -> bool {
        level >= self.level
    }

    fn log(&self, level: Level, message: Message) -> Delivery {
        let fails = {
            let mut failing = self.failing.lock().unwrap();
            let fails = failing.is_some_and(|prefix| text(&message).starts_with(prefix));
            if fails {
                *failing = None;
            }
            fails
        };
        self.records.lock().unwrap().push((level, message));
        fails.then(|| -> Pin<Box<dyn Future<Output = _> + Send>> {
            Box::pin(std::future::ready(Err(
                spark_connect_core::error::SparkError::value_msg("the handler raised"),
            )))
        })
    }
}

/// The listener of the process, a recorder of the records from `level`, while the turn of the test
/// is held.
struct Recording {
    recorder: Arc<Recorder>,
    _turn: MutexGuard<'static, ()>,
}

impl Recording {
    fn start(level: Level) -> Self {
        static TURN: Mutex<()> = Mutex::new(());
        let turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
        let recorder = Arc::new(Recorder {
            level,
            records: Mutex::new(Vec::new()),
            failing: Mutex::new(None),
        });
        spark_connect_core::client_log::set_listener(Some(recorder.clone()));
        Recording {
            recorder,
            _turn: turn,
        }
    }

    /// The records so far, of their level and message, of a message of a proto as its prefix and
    /// the type of the message, and whether it is truncated.
    fn texts(&self) -> Vec<(Level, String)> {
        self.recorder
            .records
            .lock()
            .unwrap()
            .iter()
            .map(|(level, message)| (*level, text(message)))
            .collect()
    }

    /// Fail the delivery of the next record whose text starts with `prefix`.
    fn fail_at(&self, prefix: &'static str) {
        *self.recorder.failing.lock().unwrap() = Some(prefix);
    }

    /// The encoded messages of the records so far whose text starts with `prefix`.
    fn messages(&self, prefix: &str) -> Vec<Vec<u8>> {
        self.recorder
            .records
            .lock()
            .unwrap()
            .iter()
            .filter_map(|(_, message)| match message {
                Message::Proto {
                    prefix: of,
                    message,
                    ..
                } if of.as_ref() == prefix => Some(message.bytes.clone()),
                _ => None,
            })
            .collect()
    }
}

impl Drop for Recording {
    fn drop(&mut self) {
        spark_connect_core::client_log::set_listener(None);
    }
}

fn text(message: &Message) -> String {
    match message {
        Message::Text(text) => text.clone(),
        Message::Proto {
            prefix,
            message,
            truncated,
        } => format!(
            "{prefix}<{}{}>",
            message.type_name,
            if *truncated { ", truncated" } else { "" }
        ),
        Message::WithError { text, error } => format!("{text} % <{:?}>", error.grpc_code),
    }
}

fn texts(expected: &[(Level, &str)]) -> Vec<(Level, String)> {
    expected
        .iter()
        .map(|(level, text)| (*level, text.to_string()))
        .collect()
}

/// What the server answers.
#[derive(Default)]
struct State {
    /// The values of the confs of the plan compression of the lookup, or None to fail it.
    compression: Mutex<Option<(String, String)>>,
    /// The requests of the ExecutePlans, the lookups and the releases, as the server got them.
    executed: Mutex<Vec<ExecutePlanRequest>>,
    looked_up: std::sync::atomic::AtomicUsize,
    released: std::sync::atomic::AtomicUsize,
    /// The responses of each ExecutePlan, in order, or None to stall it, or the status to end it
    /// with after them.
    executes: Mutex<VecDeque<Script>>,
    reattaches: Mutex<VecDeque<Script>>,
}

/// The responses of a call, then the end of the stream: complete, stalled or failed.
struct Script {
    responses: Vec<ExecutePlanResponse>,
    end: End,
}

enum End {
    Complete,
    Stall,
    Fail(Status),
}

fn service(state: Arc<State>) -> FakeService {
    let (execute, reattach, release, config) = (state.clone(), state.clone(), state.clone(), state);
    FakeService::new()
        .server_streaming("ExecutePlan", ProstCodec::default, move |_| {
            Execute(execute.clone())
        })
        .server_streaming("ReattachExecute", ProstCodec::default, move |_| {
            Reattach(reattach.clone())
        })
        .unary("ReleaseExecute", ProstCodec::default, move |_| {
            Release(release.clone())
        })
        .unary("Config", ProstCodec::default, move |_| {
            Config(config.clone())
        })
        .unary("AnalyzePlan", ProstCodec::default, |_| Analyze)
}

fn of_script(script: Option<Script>) -> Responses {
    let Some(script) = script else {
        return Box::pin(stream::iter(vec![Ok(response(
            ResponseType::ResultComplete(ResultComplete {}),
        ))]));
    };
    let items = script.responses.into_iter().map(Ok);
    match script.end {
        End::Complete => Box::pin(stream::iter(items.collect::<Vec<_>>())),
        End::Stall => Box::pin(stream::iter(items).chain(stream::pending())),
        End::Fail(status) => Box::pin(stream::iter(items).chain(stream::iter([Err(status)]))),
    }
}

struct Execute(Arc<State>);

impl ServerStreamingService<ExecutePlanRequest> for Execute {
    type Response = ExecutePlanResponse;
    type ResponseStream = Responses;
    type Future = BoxFuture<Response<Responses>, Status>;

    fn call(&mut self, request: Request<ExecutePlanRequest>) -> Self::Future {
        self.0.executed.lock().unwrap().push(request.into_inner());
        let script = self.0.executes.lock().unwrap().pop_front();
        Box::pin(async move { Ok(Response::new(of_script(script))) })
    }
}

struct Reattach(Arc<State>);

impl ServerStreamingService<ReattachExecuteRequest> for Reattach {
    type Response = ExecutePlanResponse;
    type ResponseStream = Responses;
    type Future = BoxFuture<Response<Responses>, Status>;

    fn call(&mut self, _: Request<ReattachExecuteRequest>) -> Self::Future {
        let script = self.0.reattaches.lock().unwrap().pop_front();
        Box::pin(async move { Ok(Response::new(of_script(script))) })
    }
}

struct Release(Arc<State>);

impl UnaryService<ReleaseExecuteRequest> for Release {
    type Response = ReleaseExecuteResponse;
    type Future = BoxFuture<Response<ReleaseExecuteResponse>, Status>;

    fn call(&mut self, _: Request<ReleaseExecuteRequest>) -> Self::Future {
        self.0
            .released
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        Box::pin(async { Ok(Response::new(ReleaseExecuteResponse::default())) })
    }
}

struct Config(Arc<State>);

impl UnaryService<ConfigRequest> for Config {
    type Response = ConfigResponse;
    type Future = BoxFuture<Response<ConfigResponse>, Status>;

    fn call(&mut self, request: Request<ConfigRequest>) -> Self::Future {
        self.0
            .looked_up
            .fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let compression = self.0.compression.lock().unwrap().clone();
        let keys = match request.into_inner().operation.and_then(|op| op.op_type) {
            Some(spark_connect_proto::config_request::operation::OpType::Get(get)) => get.keys,
            _ => Vec::new(),
        };
        Box::pin(async move {
            let Some((threshold, algorithm)) = compression else {
                return Err(Status::internal("[SQL_CONF_NOT_FOUND]"));
            };
            let values = [threshold, algorithm];
            Ok(Response::new(ConfigResponse {
                session_id: SESSION.into(),
                pairs: keys
                    .into_iter()
                    .zip(values)
                    .map(|(key, value)| KeyValue {
                        key,
                        value: Some(value),
                    })
                    .collect(),
                ..Default::default()
            }))
        })
    }
}

struct Analyze;

impl UnaryService<AnalyzePlanRequest> for Analyze {
    type Response = AnalyzePlanResponse;
    type Future = BoxFuture<Response<AnalyzePlanResponse>, Status>;

    fn call(&mut self, _: Request<AnalyzePlanRequest>) -> Self::Future {
        Box::pin(async {
            Ok(Response::new(AnalyzePlanResponse {
                session_id: SESSION.into(),
                result: Some(analyze_plan_response::Result::Schema(
                    analyze_plan_response::Schema {
                        schema: Some(DataType::default()),
                    },
                )),
                ..Default::default()
            }))
        })
    }
}

fn response(response_type: ResponseType) -> ExecutePlanResponse {
    ExecutePlanResponse {
        session_id: SESSION.into(),
        operation_id: OPERATION.into(),
        response_id: uuid_like(),
        response_type: Some(response_type),
        ..Default::default()
    }
}

fn uuid_like() -> String {
    static NEXT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(1);
    format!(
        "r{}",
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    )
}

fn chunk(index: i64, data: &[u8]) -> ExecutePlanResponse {
    response(ResponseType::ArrowBatch(ArrowBatch {
        row_count: 3,
        data: data.to_vec().into(),
        start_offset: Some(0),
        chunk_index: Some(index),
        num_chunks_in_batch: Some(2),
    }))
}

struct Fixture {
    state: Arc<State>,
    _server: FakeServer,
    client: SparkConnectClient,
}

impl Fixture {
    async fn start(compression: Option<(&str, &str)>) -> Self {
        let state = Arc::new(State {
            compression: Mutex::new(compression.map(|(t, a)| (t.into(), a.into()))),
            ..Default::default()
        });
        let server = service(state.clone()).serve().await;
        let builder = ChannelBuilder::parse(&format!(
            "sc://localhost:{}/;user_id=logged-user;session_id={SESSION}",
            server.address.port()
        ))
        .unwrap();
        let mut client = bounded(SparkConnectClient::connect(&builder))
            .await
            .unwrap();
        client.set_retry_policy(RetryPolicy {
            max_retries: Some(1),
            initial_backoff_ms: 1,
            max_backoff_ms: Some(2),
            ..RetryPolicy::default()
        });
        Fixture {
            state,
            _server: server,
            client,
        }
    }

    fn script(&self, responses: Vec<ExecutePlanResponse>, end: End) {
        self.state
            .executes
            .lock()
            .unwrap()
            .push_back(Script { responses, end });
    }
}

fn relation_request() -> ExecutePlanRequest {
    ExecutePlanRequest {
        session_id: SESSION.into(),
        user_context: Some(UserContext::default()),
        operation_id: Some(OPERATION.into()),
        plan: Some(Plan {
            op_type: Some(plan::OpType::Root(Relation::default())),
        }),
        ..Default::default()
    }
}

fn command_request() -> ExecutePlanRequest {
    ExecutePlanRequest {
        plan: Some(Plan {
            op_type: Some(plan::OpType::Command(Command {
                command_type: Some(command::CommandType::SqlCommand(SqlCommand {
                    input: Some(Relation::default()),
                    ..Default::default()
                })),
            })),
        }),
        ..relation_request()
    }
}

async fn drain(stream: &mut spark_connect_core::client::ReattachableResponseStream) -> usize {
    let mut responses = 0;
    while bounded(stream.message()).await.unwrap().is_some() {
        responses += 1;
    }
    responses
}

#[tokio::test]
async fn an_execution_logs_its_plan_its_request_and_each_response_as_pyspark() {
    // As PySpark's `to_table`: the lookup of the plan compression of the first plan, the plan
    // as it is sent, `ExecuteAndFetch`, the request before the reattach option, and each response
    // as it is read, with what it has of a schema, an Arrow batch or its chunks, metrics and
    // observed metrics, as `_execute_and_fetch_as_iterator` logs them.
    let fixture = Fixture::start(Some(("10485760", "ZSTD"))).await;
    let schema = ExecutePlanResponse {
        schema: Some(DataType::default()),
        ..response(ResponseType::SqlCommandResult(SqlCommandResult::default()))
    };
    let metrics = ExecutePlanResponse {
        metrics: Some(Metrics::default()),
        observed_metrics: vec![ObservedMetrics::default()],
        response_type: None,
        ..response(ResponseType::ResultComplete(ResultComplete {}))
    };
    let responses = vec![
        schema,
        chunk(0, &[1, 2, 3]),
        chunk(1, &[4, 5]),
        metrics,
        response(ResponseType::ResultComplete(ResultComplete {})),
    ];
    fixture.script(responses.clone(), End::Complete);
    let recording = Recording::start(Level::Debug);
    let mut stream = bounded(
        fixture
            .client
            .execute_plan_reattachable_logged(relation_request(), ExecutionLog::ToTable),
    )
    .await
    .unwrap();
    assert_eq!(drain(&mut stream).await, 5);
    let received = "ExecuteAndFetchAsIterator. Response received: \
                    <spark.connect.ExecutePlanResponse>";
    assert_eq!(
        recording.texts(),
        texts(&[
            (
                Level::Debug,
                "Plan compression threshold: 10485760, algorithm: ZSTD"
            ),
            (
                Level::Debug,
                "Executing plan <spark.connect.Plan, truncated>"
            ),
            (Level::Debug, "ExecuteAndFetch"),
            (
                Level::Debug,
                "ExecuteAndFetchAsIterator. Request: <spark.connect.ExecutePlanRequest>"
            ),
            (Level::Debug, received),
            (Level::Debug, "Received the schema."),
            (Level::Debug, "Received the SQL command result."),
            (Level::Debug, received),
            (
                Level::Debug,
                "Received arrow batch rows=3 Number of chunks in batch=2 Chunk index=0 size=3"
            ),
            (Level::Debug, received),
            (
                Level::Debug,
                "Received arrow batch rows=3 Number of chunks in batch=2 Chunk index=1 size=2"
            ),
            (Level::Debug, received),
            (Level::Debug, "Received metric batch."),
            (Level::Debug, "Received observed metric batch."),
            (Level::Debug, received),
        ])
    );
    // The request as PySpark logs it: of the result chunking option, not yet of the reattach
    // option, which the iterator adds.
    let [request] = recording
        .messages("ExecuteAndFetchAsIterator. Request: ")
        .try_into()
        .unwrap();
    let request = ExecutePlanRequest::decode(&request[..]).unwrap();
    assert_eq!(request.request_options.len(), 1);
    assert_eq!(request.operation_id.as_deref(), Some(OPERATION));
    assert_eq!(request.user_context.unwrap().user_id, "logged-user");
    // Each response as the server sent it.
    let logged: Vec<ExecutePlanResponse> = recording
        .messages("ExecuteAndFetchAsIterator. Response received: ")
        .iter()
        .map(|bytes| ExecutePlanResponse::decode(&bytes[..]).unwrap())
        .collect();
    assert_eq!(logged, responses);
}

#[tokio::test]
async fn a_command_logs_before_its_compression_and_a_registration_logs_execute() {
    // As PySpark's `execute_command`, which logs the command before it sets it in the plan, with
    // the lookup of the plan compression; `execute_command_as_iterator`, of no `ExecuteAndFetch`;
    // and `_execute` of the registrations, which logs `Execute` and nothing of its request or
    // of its responses.
    let fixture = Fixture::start(Some(("10485760", "ZSTD"))).await;
    let recording = Recording::start(Level::Debug);
    let complete = || vec![response(ResponseType::ResultComplete(ResultComplete {}))];
    for log in [
        ExecutionLog::ExecuteCommand,
        ExecutionLog::ExecuteCommandAsIterator,
    ] {
        fixture.script(complete(), End::Complete);
        let mut stream = bounded(
            fixture
                .client
                .execute_plan_reattachable_logged(command_request(), log),
        )
        .await
        .unwrap();
        drain(&mut stream).await;
    }
    fixture.script(complete(), End::Complete);
    let mut stream = bounded(
        fixture
            .client
            .execute_plan_reattachable_uncompressed(command_request()),
    )
    .await
    .unwrap();
    drain(&mut stream).await;
    let request = "ExecuteAndFetchAsIterator. Request: <spark.connect.ExecutePlanRequest>";
    let received = "ExecuteAndFetchAsIterator. Response received: \
                    <spark.connect.ExecutePlanResponse>";
    assert_eq!(
        recording.texts(),
        texts(&[
            (
                Level::Debug,
                "Execute command for command <spark.connect.Command, truncated>"
            ),
            (
                Level::Debug,
                "Plan compression threshold: 10485760, algorithm: ZSTD"
            ),
            (Level::Debug, "ExecuteAndFetch"),
            (Level::Debug, request),
            (Level::Debug, received),
            (
                Level::Debug,
                "Execute command as iterator for command <spark.connect.Command, truncated>"
            ),
            (Level::Debug, request),
            (Level::Debug, received),
            (Level::Debug, "Execute"),
        ])
    );
    let commands = recording.messages("Execute command for command ");
    assert_eq!(
        Command::decode(&commands[0][..]).unwrap(),
        match command_request().plan.unwrap().op_type.unwrap() {
            plan::OpType::Command(command) => command,
            _ => unreachable!(),
        }
    );
}

#[tokio::test]
async fn a_failed_lookup_is_logged_with_its_error_and_compression_of_its_sizes() {
    // As PySpark's lookup, whose `except Exception` logs the exception that it caught as the
    // argument of a text of no placeholder for it; and a plan over the threshold, compressed, of
    // its sizes, and one that compression does not make smaller.
    let failed = Fixture::start(None).await;
    let recording = Recording::start(Level::Debug);
    failed.script(vec![], End::Complete);
    let mut stream = bounded(
        failed
            .client
            .execute_plan_reattachable_logged(relation_request(), ExecutionLog::Fetch),
    )
    .await
    .unwrap();
    drain(&mut stream).await;
    let texts_of_failure = recording.texts();
    assert_eq!(
        texts_of_failure[0],
        (
            Level::Debug,
            format!(
                "Plan compression is disabled because the server does not support it. % \
                 <Some({})>",
                Code::Internal as i32
            )
        )
    );
    assert_eq!(
        texts_of_failure[1],
        (Level::Debug, "ExecuteAndFetch".into())
    );
    drop(recording);

    let compressing = Fixture::start(Some(("4", "ZSTD"))).await;
    let recording = Recording::start(Level::Debug);
    let of_query = |query: &str| Relation {
        rel_type: Some(spark_connect_proto::relation::RelType::Sql(
            spark_connect_proto::Sql {
                query: query.into(),
                ..Default::default()
            },
        )),
        ..Default::default()
    };
    // The sizes of the encoding of a relation and of its frame of zstd, of the level of
    // python-zstandard's `compress`.
    let sizes = |relation: &Relation| {
        let encoded = relation.encode_to_vec();
        let compressed = zstd::bulk::compress(&encoded, zstd::DEFAULT_COMPRESSION_LEVEL).unwrap();
        (encoded.len(), compressed.len())
    };
    let (original, compressed) = sizes(&of_query(&"select 1".repeat(64)));
    let large = ExecutePlanRequest {
        plan: Some(Plan {
            op_type: Some(plan::OpType::Root(of_query(&"select 1".repeat(64)))),
        }),
        ..relation_request()
    };
    compressing.script(vec![], End::Complete);
    let mut stream = bounded(
        compressing
            .client
            .execute_plan_reattachable_logged(large, ExecutionLog::ToTable),
    )
    .await
    .unwrap();
    drain(&mut stream).await;
    let logged = recording.texts();
    assert_eq!(
        logged[0],
        (
            Level::Debug,
            "Plan compression threshold: 4, algorithm: ZSTD".into()
        )
    );
    let ratio = 1.0 - compressed as f64 / original as f64;
    assert!(
        logged[1].1.starts_with(&format!(
            "Plan compression: original_size={original}, compressed_size={compressed}, \
             saving_ratio={ratio:.2}, duration_s="
        )),
        "{logged:?}"
    );
    assert_eq!(
        logged[2].1,
        "Executing plan <spark.connect.Plan, truncated>"
    );
    // A plan that compression does not make smaller is sent as it is.
    let (original, compressed) = sizes(&of_query("select 1"));
    assert!(compressed >= original);
    let small = ExecutePlanRequest {
        operation_id: Some("00000000-0000-4000-8000-000000000003".into()),
        plan: Some(Plan {
            op_type: Some(plan::OpType::Root(of_query("select 1"))),
        }),
        ..relation_request()
    };
    compressing.script(vec![], End::Complete);
    let before = recording.texts().len();
    let mut stream = bounded(
        compressing
            .client
            .execute_plan_reattachable_logged(small, ExecutionLog::ToTable),
    )
    .await
    .unwrap();
    drain(&mut stream).await;
    let logged = &recording.texts()[before..];
    let ratio = 1.0 - compressed as f64 / original as f64;
    assert!(
        logged[0].1.starts_with(&format!(
            "Plan compression: original_size={original}, compressed_size={compressed}, \
             saving_ratio={ratio:.2}, duration_s="
        )),
        "{logged:?}"
    );
    assert_eq!(
        logged[1].1,
        "Plan compression not effective. Using original plan."
    );
}

#[tokio::test]
async fn the_deadline_of_a_stream_is_logged_before_it_is_reattached() {
    // As the `_call_iter` of PySpark's iterator, which logs the deadline of the stream that
    // passed, of the last response read, before the RetryException that reattaches.
    let mut fixture = Fixture::start(Some(("10485760", "ZSTD"))).await;
    fixture.client.set_rpc_deadlines(RpcDeadlines {
        reattachable_execute_plan: Some(Duration::from_millis(200)),
        reattach_execute: Some(Duration::from_millis(200)),
        ..RpcDeadlines::default()
    });
    let first = response(ResponseType::SqlCommandResult(SqlCommandResult::default()));
    let first_id = first.response_id.clone();
    fixture.script(vec![first], End::Stall);
    let recording = Recording::start(Level::Debug);
    let mut stream = bounded(
        fixture
            .client
            .execute_plan_reattachable_logged(relation_request(), ExecutionLog::ToTable),
    )
    .await
    .unwrap();
    assert_eq!(drain(&mut stream).await, 2);
    let deadline = format!(
        "Deadline exceeded on stream for operation {OPERATION}; will reattach. (last response: \
         {first_id})"
    );
    let logged = recording.texts();
    let at = logged
        .iter()
        .position(|(_, text)| *text == deadline)
        .unwrap_or_else(|| panic!("{logged:?}"));
    // After the first response, and before the response of the reattach.
    assert_eq!(logged[at - 1].1, "Received the SQL command result.");
    assert!(logged[at + 1]
        .1
        .starts_with("ExecuteAndFetchAsIterator. Response received"));
}

#[tokio::test]
async fn an_analysis_logs_its_plan_and_nothing_is_logged_below_the_level() {
    // As PySpark's `schema`, of the plan as it is sent, and its `explain_string`, of the mode as it
    // is given it, which its caller logs, of a plan that it sends as it is; and nothing of a level
    // that the logger does not write, of which no record is made.
    let fixture = Fixture::start(Some(("10485760", "ZSTD"))).await;
    let recording = Recording::start(Level::Debug);
    let analysis = |analyze| AnalyzePlanRequest {
        session_id: SESSION.into(),
        analyze: Some(analyze),
        ..Default::default()
    };
    let plan = || {
        Some(Plan {
            op_type: Some(plan::OpType::Root(Relation::default())),
        })
    };
    bounded(fixture.client.analyze_plan_logged(
        analysis(analyze_plan_request::Analyze::Schema(
            analyze_plan_request::Schema { plan: plan() },
        )),
        AnalysisLog::Schema,
    ))
    .await
    .unwrap();
    let explained = plan().unwrap();
    spark_connect_core::client_log::logged(Level::Debug, || {
        spark_connect_core::client_log::explain_plan("COST", &explained)
    })
    .await
    .unwrap();
    bounded(fixture.client.analyze_plan_as_is(
        analysis(analyze_plan_request::Analyze::Explain(
            analyze_plan_request::Explain {
                plan: Some(explained),
                explain_mode: analyze_plan_request::explain::ExplainMode::Cost as i32,
            },
        )),
        AnalysisLog::Silent,
    ))
    .await
    .unwrap();
    // `_analyze` itself logs nothing.
    bounded(
        fixture
            .client
            .analyze_plan(analysis(analyze_plan_request::Analyze::Schema(
                analyze_plan_request::Schema { plan: plan() },
            ))),
    )
    .await
    .unwrap();
    assert_eq!(
        recording.texts(),
        texts(&[
            (
                Level::Debug,
                "Plan compression threshold: 10485760, algorithm: ZSTD"
            ),
            (
                Level::Debug,
                "Schema for plan: <spark.connect.Plan, truncated>"
            ),
            (
                Level::Debug,
                "Explain (mode=COST) for plan <spark.connect.Plan, truncated>"
            ),
        ])
    );
    drop(recording);

    let quiet = Fixture::start(Some(("10485760", "ZSTD"))).await;
    let recording = Recording::start(Level::Info);
    quiet.script(vec![], End::Complete);
    let mut stream = bounded(
        quiet
            .client
            .execute_plan_reattachable_logged(relation_request(), ExecutionLog::ToTable),
    )
    .await
    .unwrap();
    drain(&mut stream).await;
    assert_eq!(recording.texts(), Vec::new());
}

#[tokio::test]
async fn a_plan_that_the_server_cannot_parse_logs_that_compression_is_disabled() {
    // As PySpark's `_handle_rpc_error`, which logs, as information, that it disables the plan
    // compression of the session for a plan that the server cannot parse.
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
    let info = ErrorInfo {
        reason: "org.apache.spark.sql.connect.planner.InvalidPlanInput".into(),
        domain: "org.apache.spark".into(),
        metadata: HashMap::from([(
            "errorClass".to_string(),
            "CONNECT_INVALID_PLAN.CANNOT_PARSE".to_string(),
        )]),
    };
    let details = RpcStatus {
        code: Code::Internal as i32,
        message: "no parse".into(),
        details: vec![prost_types::Any {
            type_url: "type.googleapis.com/google.rpc.ErrorInfo".into(),
            value: info.encode_to_vec(),
        }],
    }
    .encode_to_vec();
    let fixture = Fixture::start(Some(("10485760", "ZSTD"))).await;
    fixture.script(
        vec![],
        End::Fail(Status::with_details(
            Code::Internal,
            "no parse",
            details.into(),
        )),
    );
    let recording = Recording::start(Level::Info);
    let mut stream = bounded(
        fixture
            .client
            .execute_plan_reattachable_logged(relation_request(), ExecutionLog::ToTable),
    )
    .await
    .unwrap();
    assert!(bounded(stream.message()).await.is_err());
    assert_eq!(
        recording.texts(),
        texts(&[(
            Level::Info,
            "Disabling plan compression for the session due to CONNECT_INVALID_PLAN.CANNOT_PARSE \
             error."
        )])
    );
}

/// The plan of a request of a relation of a SQL query of `query`, of no plan ID.
fn query_request(query: &str) -> ExecutePlanRequest {
    ExecutePlanRequest {
        plan: Some(Plan {
            op_type: Some(plan::OpType::Root(Relation {
                rel_type: Some(spark_connect_proto::relation::RelType::Sql(
                    spark_connect_proto::Sql {
                        query: query.into(),
                        ..Default::default()
                    },
                )),
                ..Default::default()
            })),
        }),
        ..relation_request()
    }
}

#[tokio::test]
async fn a_record_that_fails_fails_the_call_there_as_pyspark() {
    // As the exception of a handler of PySpark's logger, which goes out of its `logger.debug(...)`:
    // a record before the request fails the execution before it is sent; a record of a response
    // fails the stream, of the operation ID of the execution, which is released.
    let fixture = Fixture::start(Some(("10485760", "ZSTD"))).await;
    let recording = Recording::start(Level::Debug);
    recording.fail_at("ExecuteAndFetch");
    let failed = bounded(
        fixture
            .client
            .execute_plan_reattachable_logged(relation_request(), ExecutionLog::ToTable),
    )
    .await
    .err()
    .expect("the record fails the execution");
    assert_eq!(failed.message(), "the handler raised");
    assert!(fixture.state.executed.lock().unwrap().is_empty());
    assert_eq!(
        recording.texts(),
        texts(&[
            (
                Level::Debug,
                "Plan compression threshold: 10485760, algorithm: ZSTD"
            ),
            (
                Level::Debug,
                "Executing plan <spark.connect.Plan, truncated>"
            ),
            (Level::Debug, "ExecuteAndFetch"),
        ])
    );

    let schema = ExecutePlanResponse {
        schema: Some(DataType::default()),
        ..response(ResponseType::SqlCommandResult(SqlCommandResult::default()))
    };
    fixture.script(vec![schema], End::Stall);
    recording.fail_at("Received the schema.");
    let mut stream = bounded(
        fixture
            .client
            .execute_plan_reattachable_logged(relation_request(), ExecutionLog::ToTable),
    )
    .await
    .unwrap();
    let failed = bounded(stream.message())
        .await
        .expect_err("the record fails the stream");
    assert_eq!(failed.message(), "the handler raised");
    assert_eq!(failed.operation_id.as_deref(), Some(OPERATION));
    for _ in 0..100 {
        if fixture
            .state
            .released
            .load(std::sync::atomic::Ordering::SeqCst)
            > 0
        {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert!(
        fixture
            .state
            .released
            .load(std::sync::atomic::Ordering::SeqCst)
            > 0
    );
    assert!(bounded(stream.message()).await.unwrap().is_none());
}

#[tokio::test]
async fn a_failed_record_of_the_settings_disables_the_compression_as_pyspark() {
    // As the `except Exception` of PySpark's lookup, around its log of the settings that it read:
    // the error of that record disables the plan compression, of the error logged, and the call
    // goes on, of its plan sent as it is.
    let fixture = Fixture::start(Some(("4", "ZSTD"))).await;
    let recording = Recording::start(Level::Debug);
    recording.fail_at("Plan compression threshold");
    fixture.script(vec![], End::Complete);
    let large = query_request(&"select 1".repeat(64));
    let mut stream = bounded(
        fixture
            .client
            .execute_plan_reattachable_logged(large, ExecutionLog::ToTable),
    )
    .await
    .unwrap();
    drain(&mut stream).await;
    let logged = recording.texts();
    assert_eq!(
        logged[..3],
        texts(&[
            (
                Level::Debug,
                "Plan compression threshold: 4, algorithm: ZSTD"
            ),
            (
                Level::Debug,
                "Plan compression is disabled because the server does not support it. % <None>"
            ),
            (
                Level::Debug,
                "Executing plan <spark.connect.Plan, truncated>"
            ),
        ])
    );
    let sent = fixture.state.executed.lock().unwrap()[0].clone();
    assert!(matches!(
        sent.plan.and_then(|plan| plan.op_type),
        Some(plan::OpType::Root(_))
    ));
}

#[tokio::test]
async fn a_closed_client_writes_the_records_of_a_call_before_it_raises_as_pyspark() {
    // As PySpark's client, which reads its `_stub`, of the NO_ACTIVE_SESSION of a closed client,
    // only when it sends: an execution logs its plan or its command, `ExecuteAndFetch` and its
    // request first, and an analysis its plan, and the lookup of the plan compression of the first
    // plan fails of the closed client, which disables the compression, of the error logged. The
    // error of an execution carries its operation ID, as the `_handle_error` of PySpark's
    // execution sets it. Nothing is sent.
    let fixture = Fixture::start(Some(("10485760", "ZSTD"))).await;
    let recording = Recording::start(Level::Debug);
    fixture.client.set_closed(true);
    for (request, log) in [
        (relation_request(), ExecutionLog::ToTable),
        (command_request(), ExecutionLog::ExecuteCommand),
    ] {
        let failed = bounded(
            fixture
                .client
                .execute_plan_reattachable_logged(request, log),
        )
        .await
        .err()
        .expect("a closed client raises");
        assert_eq!(failed.error_class, "NO_ACTIVE_SESSION");
        assert_eq!(failed.operation_id.as_deref(), Some(OPERATION));
    }
    let failed = bounded(fixture.client.analyze_plan_logged(
        AnalyzePlanRequest {
            session_id: SESSION.into(),
            analyze: Some(analyze_plan_request::Analyze::Schema(
                analyze_plan_request::Schema {
                    plan: Some(Plan {
                        op_type: Some(plan::OpType::Root(Relation::default())),
                    }),
                },
            )),
            ..Default::default()
        },
        AnalysisLog::Schema,
    ))
    .await
    .expect_err("a closed client raises");
    assert_eq!(failed.error_class, "NO_ACTIVE_SESSION");
    assert_eq!(
        recording.texts(),
        texts(&[
            (
                Level::Debug,
                "Plan compression is disabled because the server does not support it. % <None>"
            ),
            (
                Level::Debug,
                "Executing plan <spark.connect.Plan, truncated>"
            ),
            (Level::Debug, "ExecuteAndFetch"),
            (
                Level::Debug,
                "ExecuteAndFetchAsIterator. Request: <spark.connect.ExecutePlanRequest>"
            ),
            (
                Level::Debug,
                "Execute command for command <spark.connect.Command, truncated>"
            ),
            (Level::Debug, "ExecuteAndFetch"),
            (
                Level::Debug,
                "ExecuteAndFetchAsIterator. Request: <spark.connect.ExecutePlanRequest>"
            ),
            (
                Level::Debug,
                "Schema for plan: <spark.connect.Plan, truncated>"
            ),
        ])
    );
    assert!(fixture.state.executed.lock().unwrap().is_empty());
    assert_eq!(
        fixture
            .state
            .looked_up
            .load(std::sync::atomic::Ordering::SeqCst),
        0
    );
}

/// A listener of the retries of the calls of the process that tells what the records whose
/// handler raised were written as the client handled (see `RetriesListener::raised_in_handling`),
/// and whose record of grpcio's error of a failed call (`GRPC Error received`) raises once, when
/// asked to, as a handler of the logger of PySpark's `_handle_rpc_error` raises.
#[derive(Default)]
struct Handling {
    handled: Mutex<Vec<String>>,
    failing_rpc_record: std::sync::atomic::AtomicBool,
}

impl spark_connect_core::retries::RetriesListener for Handling {
    fn retries_exceeded(
        &self,
        _exceeded: spark_connect_core::retries::RetriesExceeded,
    ) -> spark_connect_core::error::Result<()> {
        Ok(())
    }

    fn caught(
        &self,
        _error: spark_connect_core::error::SparkError,
    ) -> Option<spark_connect_core::error::SparkError> {
        None
    }

    fn rpc_error(&self, _error: &spark_connect_core::error::GrpcioError) -> Delivery {
        if !self
            .failing_rpc_record
            .swap(false, std::sync::atomic::Ordering::SeqCst)
        {
            return None;
        }
        Some(Box::pin(std::future::ready(Err(
            spark_connect_core::error::SparkError::value_msg("the handler of the record raised"),
        ))))
    }

    fn raised_in_handling(
        &self,
        error: spark_connect_core::error::SparkError,
        handled: spark_connect_core::retries::Handled<'_>,
    ) -> spark_connect_core::error::SparkError {
        use spark_connect_core::retries::Handled;
        let handled = match handled {
            Handled::Rpc(failure) => format!("rpc {:?}", failure.code),
            Handled::Error(caught) => format!("error {}", caught.message()),
        };
        self.handled.lock().unwrap().push(handled);
        error
    }
}

#[tokio::test]
async fn the_error_of_a_record_of_the_handling_of_a_failure_is_raised_in_handling_it() {
    // As PySpark's records in the `except` clauses that handle the failure of a call: of the
    // lookup of the plan compression (`except Exception as e`, of the error of the lookup) and of
    // `_handle_rpc_error` (of grpcio's error of the call), whose exception has the failure as its
    // context, and which the `except KeyboardInterrupt` of PySpark's execution does not catch.
    let fixture = Fixture::start(None).await;
    let recording = Recording::start(Level::Debug);
    let handling = Arc::new(Handling::default());
    spark_connect_core::retries::set_listener(Some(handling.clone()));
    recording.fail_at("Plan compression is disabled");
    let failed = bounded(
        fixture
            .client
            .execute_plan_reattachable_logged(relation_request(), ExecutionLog::ToTable),
    )
    .await
    .err()
    .expect("the record fails the execution");
    assert_eq!(failed.message(), "the handler raised");
    assert!(fixture.state.executed.lock().unwrap().is_empty());
    let lookup = std::mem::take(&mut *handling.handled.lock().unwrap());
    assert_eq!(lookup.len(), 1, "{lookup:?}");
    assert!(
        lookup[0].starts_with("error ") && lookup[0].contains("SQL_CONF_NOT_FOUND"),
        "{lookup:?}"
    );

    fixture.script(vec![], End::Fail(Status::internal("the server failed")));
    handling
        .failing_rpc_record
        .store(true, std::sync::atomic::Ordering::SeqCst);
    let mut stream = bounded(
        fixture
            .client
            .execute_plan_reattachable_logged(relation_request(), ExecutionLog::ToTable),
    )
    .await
    .unwrap();
    let failed = bounded(stream.message())
        .await
        .expect_err("the record fails the stream");
    spark_connect_core::retries::set_listener(None);
    assert_eq!(failed.message(), "the handler of the record raised");
    assert_eq!(
        *handling.handled.lock().unwrap(),
        vec![format!("rpc {:?}", Code::Internal)]
    );
}

#[tokio::test]
async fn a_prepared_plan_and_a_plan_of_the_client_api_are_sent_as_they_are() {
    // As PySpark's `to_proto`, which tries the compression of a plan once, of the lookup of the
    // first plan, before a caller that reads settings of its own sends it; and as its `to_table`
    // and `_analyze`, which send the plan that they are given as it is, of no lookup and no
    // compression.
    let fixture = Fixture::start(Some(("4", "ZSTD"))).await;
    let recording = Recording::start(Level::Debug);
    let incompressible = "select 1";
    let mut prepared = query_request(incompressible).plan.unwrap();
    bounded(fixture.client.prepare_plan(&mut prepared))
        .await
        .unwrap();
    let request = ExecutePlanRequest {
        plan: Some(prepared),
        ..relation_request()
    };
    fixture.script(vec![], End::Complete);
    let mut stream = bounded(
        fixture
            .client
            .execute_plan_reattachable_as_is(request, ExecutionLog::Fetch),
    )
    .await
    .unwrap();
    drain(&mut stream).await;
    let logged: Vec<String> = recording
        .texts()
        .into_iter()
        .map(|(_, text)| text)
        .collect();
    let compressions = logged
        .iter()
        .filter(|text| text.starts_with("Plan compression: "))
        .count();
    assert_eq!(compressions, 1, "{logged:?}");
    assert_eq!(
        logged[2],
        "Plan compression not effective. Using original plan."
    );
    assert_eq!(logged[3], "ExecuteAndFetch");
    drop(recording);

    let fresh = Fixture::start(Some(("4", "ZSTD"))).await;
    let recording = Recording::start(Level::Debug);
    fresh.script(vec![], End::Complete);
    let mut stream = bounded(fresh.client.execute_plan_reattachable_as_is(
        query_request(&"select 1".repeat(64)),
        ExecutionLog::ToTable,
    ))
    .await
    .unwrap();
    drain(&mut stream).await;
    let analysis = AnalyzePlanRequest {
        session_id: SESSION.into(),
        analyze: Some(analyze_plan_request::Analyze::Schema(
            analyze_plan_request::Schema {
                plan: query_request(&"select 1".repeat(64)).plan,
            },
        )),
        ..Default::default()
    };
    bounded(
        fresh
            .client
            .analyze_plan_as_is(analysis, AnalysisLog::Schema),
    )
    .await
    .unwrap();
    assert_eq!(
        fresh
            .state
            .looked_up
            .load(std::sync::atomic::Ordering::SeqCst),
        0
    );
    let sent = fresh.state.executed.lock().unwrap()[0].clone();
    assert!(matches!(
        sent.plan.and_then(|plan| plan.op_type),
        Some(plan::OpType::Root(_))
    ));
    assert_eq!(
        recording.texts(),
        texts(&[
            (
                Level::Debug,
                "Executing plan <spark.connect.Plan, truncated>"
            ),
            (Level::Debug, "ExecuteAndFetch"),
            (
                Level::Debug,
                "ExecuteAndFetchAsIterator. Request: <spark.connect.ExecutePlanRequest>"
            ),
            (
                Level::Debug,
                "ExecuteAndFetchAsIterator. Response received: \
                 <spark.connect.ExecutePlanResponse>"
            ),
            (
                Level::Debug,
                "Schema for plan: <spark.connect.Plan, truncated>"
            ),
        ])
    );
}
