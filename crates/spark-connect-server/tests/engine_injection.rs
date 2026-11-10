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

//! Real gRPC tests of the injected backend boundary. The fake is independent of Spark execution.

use std::collections::HashMap;
use std::io::Cursor;
use std::pin::Pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::task::{Context, Poll};
use std::time::Duration;

use arrow::array::{Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::ipc::reader::StreamReader;
use arrow::record_batch::RecordBatch;
use futures::{Stream, StreamExt};
use prost::Message;
use proto::execute_plan_response::ResponseType;
use proto::spark_connect_service_client::SparkConnectServiceClient;
use spark_connect_proto as proto;
use spark_connect_server::engine::{
    BatchStream, ConfigResult, Engine, EngineSession, PreparedOperation, PreparedQuery,
    SessionContext,
};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tonic::transport::Channel;
use tonic::{Code, Status};
use uuid::Uuid;

const LIVE: i64 = 1_000_000;
const BLOCKED_PREPARE: i64 = 2_000_000;
const MISMATCHED_BATCH: i64 = 3_000_000;
const ANNOTATED_BATCH: i64 = 4_000_000;
const FLOODED_RESULT: i64 = 5_000_000;

#[derive(Default)]
struct Spy {
    factory_calls: Mutex<Vec<(String, String)>>,
    fail_next: AtomicBool,
    sessions_dropped: AtomicUsize,
    prepares: AtomicUsize,
    commands: AtomicUsize,
    analyses: AtomicUsize,
    configs: AtomicUsize,
    executions: AtomicUsize,
    streams_dropped: AtomicUsize,
    batches_polled: AtomicUsize,
    blocked_prepares_dropped: AtomicUsize,
    blocked_analyses_dropped: AtomicUsize,
    blocked_configs_dropped: AtomicUsize,
    metadata_drop_observations: Mutex<Vec<(usize, usize)>>,
    slow_drop: Mutex<Option<(String, Arc<DropGate>)>>,
}

#[derive(Default)]
struct DropGate {
    released: Mutex<bool>,
    condition: Condvar,
    entered: AtomicUsize,
    exited: AtomicBool,
    timed_out: AtomicBool,
}

impl DropGate {
    fn wait(&self) {
        let released = self.released.lock().unwrap();
        self.entered.fetch_add(1, Ordering::SeqCst);
        let (released, timeout) = self
            .condition
            .wait_timeout_while(released, Duration::from_secs(3), |released| !*released)
            .unwrap();
        self.timed_out
            .store(timeout.timed_out() && !*released, Ordering::SeqCst);
        self.exited.store(true, Ordering::SeqCst);
    }

    fn release(&self) {
        *self.released.lock().unwrap() = true;
        self.condition.notify_all();
    }
}

// Even a failing assertion must never leave a worker blocked in the fake Drop.
struct UnblockDrop(Arc<DropGate>);

impl Drop for UnblockDrop {
    fn drop(&mut self) {
        self.0.release();
    }
}

struct FakeEngine(Arc<Spy>);

impl Engine for FakeEngine {
    fn create_session(&self, context: SessionContext) -> Result<Arc<dyn EngineSession>, Status> {
        let drop_gate = self
            .0
            .slow_drop
            .lock()
            .unwrap()
            .as_ref()
            .filter(|(id, _)| id == &context.session_id)
            .map(|(_, gate)| gate.clone());
        let user = context
            .user_context
            .map(|user| user.user_id)
            .unwrap_or_default();
        self.0
            .factory_calls
            .lock()
            .unwrap()
            .push((user, context.session_id));
        if self.0.fail_next.swap(false, Ordering::SeqCst) {
            return Err(Status::unavailable(
                "fake factory is temporarily unavailable",
            ));
        }
        Ok(Arc::new(FakeSession {
            spy: self.0.clone(),
            configs: Mutex::new(HashMap::new()),
            views: Mutex::new(HashMap::new()),
            drop_gate,
        }))
    }
}

struct FakeSession {
    spy: Arc<Spy>,
    configs: Mutex<HashMap<String, String>>,
    views: Mutex<HashMap<String, i64>>,
    drop_gate: Option<Arc<DropGate>>,
}

impl Drop for FakeSession {
    fn drop(&mut self) {
        if let Some(gate) = &self.drop_gate {
            // Keep Drop synchronous, while allowing Tokio to replace this worker
            // so the test measures registry contention rather than timer starvation.
            tokio::task::block_in_place(|| gate.wait());
        }
        self.spy.metadata_drop_observations.lock().unwrap().push((
            self.spy.blocked_analyses_dropped.load(Ordering::SeqCst),
            self.spy.blocked_configs_dropped.load(Ordering::SeqCst),
        ));
        self.spy.sessions_dropped.fetch_add(1, Ordering::SeqCst);
    }
}

struct BlockedPrepare(Arc<Spy>);

impl Drop for BlockedPrepare {
    fn drop(&mut self) {
        self.0
            .blocked_prepares_dropped
            .fetch_add(1, Ordering::SeqCst);
    }
}

enum MetadataCall {
    Analysis,
    Config,
}

struct BlockedMetadata {
    spy: Arc<Spy>,
    call: MetadataCall,
}

impl Drop for BlockedMetadata {
    fn drop(&mut self) {
        let counter = match self.call {
            MetadataCall::Analysis => &self.spy.blocked_analyses_dropped,
            MetadataCall::Config => &self.spy.blocked_configs_dropped,
        };
        counter.fetch_add(1, Ordering::SeqCst);
    }
}

#[tonic::async_trait]
impl EngineSession for FakeSession {
    async fn prepare(&self, op: proto::plan::OpType) -> Result<PreparedOperation, Status> {
        self.spy.prepares.fetch_add(1, Ordering::SeqCst);
        let (value, live, batch_mode) = match op {
            proto::plan::OpType::Root(relation) => match relation.rel_type {
                Some(proto::relation::RelType::Range(range)) => {
                    if range.end == BLOCKED_PREPARE {
                        let _dropped = BlockedPrepare(self.spy.clone());
                        std::future::pending::<()>().await;
                    }
                    if range.end == -1 {
                        return Err(Status::invalid_argument("fake rejected the plan"));
                    }
                    // Deliberately unlike range semantics: this proves the injected
                    // backend, rather than the default planner, executed the request.
                    let batch_mode = match range.end {
                        MISMATCHED_BATCH => BatchMode::WrongType,
                        ANNOTATED_BATCH => BatchMode::DifferentAnnotations,
                        FLOODED_RESULT => BatchMode::Flooded,
                        _ => BatchMode::Declared,
                    };
                    (999, range.end == LIVE, batch_mode)
                }
                Some(proto::relation::RelType::Read(read)) => {
                    let Some(proto::read::ReadType::NamedTable(table)) = read.read_type else {
                        return Err(Status::unimplemented("fake read"));
                    };
                    let value = self
                        .views
                        .lock()
                        .unwrap()
                        .get(&table.unparsed_identifier)
                        .copied()
                        .ok_or_else(|| Status::not_found("fake view not found"))?;
                    (value, false, BatchMode::Declared)
                }
                _ => return Err(Status::unimplemented("fake relation")),
            },
            proto::plan::OpType::Command(command) => {
                self.spy.commands.fetch_add(1, Ordering::SeqCst);
                match command.command_type {
                    Some(proto::command::CommandType::CreateDataframeView(view)) => {
                        let Some(proto::relation::RelType::Range(input)) =
                            view.input.and_then(|input| input.rel_type)
                        else {
                            return Err(Status::invalid_argument("fake view requires range input"));
                        };
                        let mut views = self.views.lock().unwrap();
                        if !view.replace && views.contains_key(&view.name) {
                            return Err(Status::already_exists("fake view already exists"));
                        }
                        views.insert(view.name, input.end);
                        return Ok(PreparedOperation::Command(Vec::new()));
                    }
                    Some(proto::command::CommandType::SqlCommand(command)) => {
                        return Ok(PreparedOperation::Command(vec![
                            ResponseType::SqlCommandResult(
                                proto::execute_plan_response::SqlCommandResult {
                                    relation: command.input,
                                },
                            ),
                        ]));
                    }
                    _ => return Err(Status::unimplemented("fake command")),
                }
            }
            _ => return Err(Status::unimplemented("fake operation")),
        };
        Ok(PreparedOperation::Query(Box::new(FakeQuery {
            spy: self.spy.clone(),
            value,
            live,
            batch_mode,
        })))
    }

    async fn analyze(
        &self,
        input: proto::analyze_plan_request::Analyze,
    ) -> Result<proto::analyze_plan_response::Result, Status> {
        self.spy.analyses.fetch_add(1, Ordering::SeqCst);
        match input {
            proto::analyze_plan_request::Analyze::Schema(_) => {
                Ok(proto::analyze_plan_response::Result::Schema(
                    proto::analyze_plan_response::Schema {
                        schema: Some(proto::DataType {
                            kind: Some(proto::data_type::Kind::Struct(proto::data_type::Struct {
                                fields: vec![proto::data_type::StructField {
                                    name: "x".repeat(3 * 1024 * 1024 + 1),
                                    data_type: Some(proto::DataType {
                                        kind: Some(proto::data_type::Kind::Long(
                                            proto::data_type::Long::default(),
                                        )),
                                    }),
                                    nullable: false,
                                    metadata: None,
                                }],
                                ..Default::default()
                            })),
                        }),
                    },
                ))
            }
            proto::analyze_plan_request::Analyze::IsStreaming(_) => {
                let _dropped = BlockedMetadata {
                    spy: self.spy.clone(),
                    call: MetadataCall::Analysis,
                };
                std::future::pending().await
            }
            proto::analyze_plan_request::Analyze::IsLocal(_) => {
                Ok(proto::analyze_plan_response::Result::IsLocal(
                    proto::analyze_plan_response::IsLocal { is_local: true },
                ))
            }
            _ => Err(Status::unimplemented("fake analysis")),
        }
    }

    async fn config(
        &self,
        input: proto::config_request::operation::OpType,
    ) -> Result<ConfigResult, Status> {
        use proto::config_request::operation::OpType;
        self.spy.configs.fetch_add(1, Ordering::SeqCst);
        if matches!(&input, OpType::Get(get) if get.keys.iter().any(|key| key == "fake.block")) {
            let _dropped = BlockedMetadata {
                spy: self.spy.clone(),
                call: MetadataCall::Config,
            };
            return std::future::pending().await;
        }
        let mut configs = self.configs.lock().unwrap();
        match input {
            OpType::Set(set) => {
                for pair in set.pairs {
                    configs.insert(
                        pair.key,
                        pair.value
                            .ok_or_else(|| Status::invalid_argument("fake config value"))?,
                    );
                }
                Ok(ConfigResult {
                    pairs: Vec::new(),
                    warnings: vec!["handled by fake backend".into()],
                })
            }
            OpType::GetOption(get) => Ok(ConfigResult {
                pairs: get
                    .keys
                    .into_iter()
                    .map(|key| {
                        let value = configs.get(&key).cloned();
                        proto::KeyValue { key, value }
                    })
                    .collect(),
                warnings: Vec::new(),
            }),
            _ => Err(Status::unimplemented("fake config operation")),
        }
    }
}

struct FakeQuery {
    spy: Arc<Spy>,
    value: i64,
    live: bool,
    batch_mode: BatchMode,
}

enum BatchMode {
    Declared,
    WrongType,
    DifferentAnnotations,
    Flooded,
}

#[tonic::async_trait]
impl PreparedQuery for FakeQuery {
    fn schema(&self) -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new(
            "backend_value",
            DataType::Int64,
            false,
        )]))
    }

    async fn execute(self: Box<Self>) -> Result<BatchStream, Status> {
        self.spy.executions.fetch_add(1, Ordering::SeqCst);
        let batch = match self.batch_mode {
            BatchMode::Declared => RecordBatch::try_new(
                self.schema(),
                vec![Arc::new(Int64Array::from(vec![self.value]))],
            ),
            BatchMode::WrongType => RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "backend_value",
                    DataType::Utf8,
                    false,
                )])),
                vec![Arc::new(StringArray::from(vec!["wrong type"]))],
            ),
            BatchMode::DifferentAnnotations => RecordBatch::try_new(
                Arc::new(Schema::new(vec![Field::new(
                    "backend_value",
                    DataType::Int64,
                    true,
                )
                .with_metadata(HashMap::from([(
                    "backend.note".into(),
                    "ignored annotation".into(),
                )]))])),
                vec![Arc::new(Int64Array::from(vec![self.value]))],
            ),
            BatchMode::Flooded => RecordBatch::try_new(
                self.schema(),
                vec![Arc::new(Int64Array::from(vec![self.value; 64 * 1024]))],
            ),
        }
        .unwrap();
        if matches!(self.batch_mode, BatchMode::Flooded) {
            let spy = self.spy.clone();
            return Ok(Box::pin(SpyStream {
                inner: Box::pin(futures::stream::repeat_with(move || {
                    spy.batches_polled.fetch_add(1, Ordering::SeqCst);
                    Ok(batch.clone())
                })),
                spy: self.spy.clone(),
            }));
        }
        let first = futures::stream::iter(vec![Ok(batch)]);
        let inner: BatchStream = if self.live {
            Box::pin(first.chain(futures::stream::pending()))
        } else {
            Box::pin(first)
        };
        Ok(Box::pin(SpyStream {
            inner,
            spy: self.spy.clone(),
        }))
    }
}

struct SpyStream {
    inner: BatchStream,
    spy: Arc<Spy>,
}

impl Stream for SpyStream {
    type Item = Result<RecordBatch, Status>;

    fn poll_next(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.get_mut().inner.as_mut().poll_next(context)
    }
}

impl Drop for SpyStream {
    fn drop(&mut self) {
        self.spy.streams_dropped.fetch_add(1, Ordering::SeqCst);
    }
}

struct Server {
    address: std::net::SocketAddr,
    client: SparkConnectServiceClient<Channel>,
    shutdown: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<()>>,
}

impl Server {
    async fn start(spy: Arc<Spy>) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (shutdown, stopped) = oneshot::channel();
        let task = tokio::spawn(async move {
            spark_connect_server::serve_with_engine(listener, Arc::new(FakeEngine(spy)), async {
                let _ = stopped.await;
            })
            .await
            .unwrap();
        });
        let client = SparkConnectServiceClient::connect(format!("http://{address}"))
            .await
            .unwrap();
        Self {
            address,
            client,
            shutdown: Some(shutdown),
            task: Some(task),
        }
    }

    async fn stop(mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        tokio::time::timeout(Duration::from_secs(5), self.task.take().unwrap())
            .await
            .unwrap()
            .unwrap();
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

fn user(id: &str) -> Option<proto::UserContext> {
    Some(proto::UserContext {
        user_id: id.into(),
        ..Default::default()
    })
}

fn relation(end: i64) -> proto::Relation {
    proto::Relation {
        rel_type: Some(proto::relation::RelType::Range(proto::Range {
            end,
            step: 1,
            ..Default::default()
        })),
        ..Default::default()
    }
}

fn request(session: &str, end: i64) -> proto::ExecutePlanRequest {
    proto::ExecutePlanRequest {
        session_id: session.into(),
        user_context: user("alice"),
        operation_id: Some(Uuid::new_v4().to_string()),
        plan: Some(proto::Plan {
            op_type: Some(proto::plan::OpType::Root(relation(end))),
        }),
        ..Default::default()
    }
}

fn command_request(
    session: &str,
    command_type: proto::command::CommandType,
) -> proto::ExecutePlanRequest {
    let mut request = request(session, 0);
    request.plan = Some(proto::Plan {
        op_type: Some(proto::plan::OpType::Command(proto::Command {
            command_type: Some(command_type),
        })),
    });
    request
}

fn config_request(
    session: &str,
    input: proto::config_request::operation::OpType,
) -> proto::ConfigRequest {
    proto::ConfigRequest {
        session_id: session.into(),
        user_context: user("alice"),
        operation: Some(proto::config_request::Operation {
            op_type: Some(input),
        }),
        ..Default::default()
    }
}

fn get_config(session: &str) -> proto::ConfigRequest {
    config_request(
        session,
        proto::config_request::operation::OpType::GetOption(proto::config_request::GetOption {
            keys: vec!["spark.fake.setting".into()],
        }),
    )
}

fn release(session: &str, operation: &str) -> proto::ReleaseExecuteRequest {
    proto::ReleaseExecuteRequest {
        session_id: session.into(),
        user_context: user("alice"),
        operation_id: operation.into(),
        release: Some(proto::release_execute_request::Release::ReleaseAll(
            proto::release_execute_request::ReleaseAll {},
        )),
        ..Default::default()
    }
}

fn release_session(session: &str) -> proto::ReleaseSessionRequest {
    proto::ReleaseSessionRequest {
        session_id: session.into(),
        user_context: user("alice"),
        ..Default::default()
    }
}

async fn wait_for_count(counter: &AtomicUsize, expected: usize) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while counter.load(Ordering::SeqCst) < expected {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("backend lifecycle event must arrive");
    assert_eq!(counter.load(Ordering::SeqCst), expected);
}

async fn collect(
    stream: &mut tonic::Streaming<proto::ExecutePlanResponse>,
) -> Vec<proto::ExecutePlanResponse> {
    let mut responses = Vec::new();
    while let Some(response) = stream.message().await.unwrap() {
        responses.push(response);
    }
    responses
}

fn values(responses: &[proto::ExecutePlanResponse]) -> Vec<i64> {
    let mut values = Vec::new();
    for response in responses {
        if let Some(ResponseType::ArrowBatch(batch)) = &response.response_type {
            for batch in StreamReader::try_new(Cursor::new(batch.data.clone()), None).unwrap() {
                let batch = batch.unwrap();
                let array = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                values.extend((0..array.len()).map(|index| array.value(index)));
            }
        }
    }
    values
}

async fn assert_cancelled(stream: &mut tonic::Streaming<proto::ExecutePlanResponse>) {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match stream.message().await {
                Err(status) => {
                    assert_eq!(status.code(), Code::Cancelled);
                    break;
                }
                Ok(Some(response)) => assert!(!matches!(
                    response.response_type,
                    Some(ResponseType::ResultComplete(_))
                )),
                Ok(None) => panic!("cancelled fake execution must report cancellation"),
            }
        }
    })
    .await
    .expect("cancelled fake execution must terminate");
}

async fn start_live(
    server: &mut Server,
    request: proto::ExecutePlanRequest,
) -> tonic::Streaming<proto::ExecutePlanResponse> {
    let mut stream = server
        .client
        .execute_plan(request)
        .await
        .unwrap()
        .into_inner();
    assert!(stream.message().await.unwrap().unwrap().schema.is_some());
    assert!(matches!(
        stream.message().await.unwrap().unwrap().response_type,
        Some(ResponseType::ArrowBatch(_))
    ));
    stream
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn injected_backend_owns_query_analysis_config_and_catalog_state() {
    let spy = Arc::new(Spy::default());
    let mut server = Server::start(spy.clone()).await;
    let session = Uuid::new_v4().to_string();
    let mut first = server.client.clone();
    let mut second = server.client.clone();
    let (left, right) = tokio::join!(
        first.config(get_config(&session)),
        second.config(get_config(&session))
    );
    left.unwrap();
    right.unwrap();
    assert_eq!(
        spy.factory_calls.lock().unwrap().len(),
        1,
        "concurrent requests must share one backend session"
    );

    let set = config_request(
        &session,
        proto::config_request::operation::OpType::Set(proto::config_request::Set {
            pairs: vec![proto::KeyValue {
                key: "spark.fake.setting".into(),
                value: Some("backend-owned".into()),
            }],
            silent: Some(false),
        }),
    );
    let configured = server.client.config(set).await.unwrap().into_inner();
    assert_eq!(configured.warnings, vec!["handled by fake backend"]);
    assert_eq!(
        server
            .client
            .config(get_config(&session))
            .await
            .unwrap()
            .into_inner()
            .pairs[0]
            .value
            .as_deref(),
        Some("backend-owned")
    );
    let analysis = server
        .client
        .analyze_plan(proto::AnalyzePlanRequest {
            session_id: session.clone(),
            user_context: user("alice"),
            analyze: Some(proto::analyze_plan_request::Analyze::IsLocal(
                proto::analyze_plan_request::IsLocal {
                    plan: request(&session, 99).plan,
                },
            )),
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    assert!(matches!(
        analysis.result,
        Some(proto::analyze_plan_response::Result::IsLocal(
            proto::analyze_plan_response::IsLocal { is_local: true }
        ))
    ));

    let mut stream = server
        .client
        .execute_plan(request(&session, 99))
        .await
        .unwrap()
        .into_inner();
    let responses = collect(&mut stream).await;
    assert_eq!(values(&responses), vec![999]);
    assert!(matches!(
        responses.last().unwrap().response_type,
        Some(ResponseType::ResultComplete(_))
    ));
    let create_view = command_request(
        &session,
        proto::command::CommandType::CreateDataframeView(proto::CreateDataFrameViewCommand {
            input: Some(relation(77)),
            name: "fake_view".into(),
            is_global: false,
            replace: false,
        }),
    );
    let mut created = server
        .client
        .execute_plan(create_view)
        .await
        .unwrap()
        .into_inner();
    collect(&mut created).await;
    let mut read_view = request(&session, 0);
    read_view.plan = Some(proto::Plan {
        op_type: Some(proto::plan::OpType::Root(proto::Relation {
            rel_type: Some(proto::relation::RelType::Read(proto::Read {
                read_type: Some(proto::read::ReadType::NamedTable(proto::read::NamedTable {
                    unparsed_identifier: "fake_view".into(),
                    ..Default::default()
                })),
                ..Default::default()
            })),
            ..Default::default()
        })),
    });
    let mut stream = server
        .client
        .execute_plan(read_view.clone())
        .await
        .unwrap()
        .into_inner();
    assert_eq!(values(&collect(&mut stream).await), vec![77]);
    let sql = command_request(
        &session,
        proto::command::CommandType::SqlCommand(proto::SqlCommand {
            input: Some(relation(13)),
            ..Default::default()
        }),
    );
    let mut stream = server.client.execute_plan(sql).await.unwrap().into_inner();
    let responses = collect(&mut stream).await;
    assert!(matches!(
        responses[0].response_type,
        Some(ResponseType::SqlCommandResult(_))
    ));

    let mut other_user = get_config(&session);
    other_user.user_context = user("bob");
    assert_eq!(
        server
            .client
            .config(other_user)
            .await
            .unwrap()
            .into_inner()
            .pairs[0]
            .value,
        None
    );
    read_view.user_context = user("bob");
    assert_eq!(
        server
            .client
            .execute_plan(read_view)
            .await
            .unwrap_err()
            .code(),
        Code::NotFound
    );
    let other_session = Uuid::new_v4().to_string();
    assert_eq!(
        server
            .client
            .config(get_config(&other_session))
            .await
            .unwrap()
            .into_inner()
            .pairs[0]
            .value,
        None
    );
    assert_eq!(
        *spy.factory_calls.lock().unwrap(),
        vec![
            ("alice".into(), session.clone()),
            ("bob".into(), session),
            ("alice".into(), other_session),
        ]
    );
    assert_eq!(spy.analyses.load(Ordering::SeqCst), 1);
    assert_eq!(spy.commands.load(Ordering::SeqCst), 2);
    assert_eq!(spy.executions.load(Ordering::SeqCst), 2);
    server.stop().await;
    wait_for_count(&spy.sessions_dropped, 3).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_requests_do_not_open_backends_and_factory_failures_can_retry() {
    let spy = Arc::new(Spy::default());
    let mut server = Server::start(spy.clone()).await;
    let session = Uuid::new_v4().to_string();
    assert_eq!(
        server
            .client
            .execute_plan(request("invalid", 99))
            .await
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );
    let mut invalid = request(&session, 99);
    invalid.operation_id = Some("invalid".into());
    assert_eq!(
        server
            .client
            .execute_plan(invalid)
            .await
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );
    assert_eq!(
        server
            .client
            .release_execute(release(&session, &Uuid::new_v4().to_string()))
            .await
            .unwrap_err()
            .code(),
        Code::NotFound
    );
    assert!(spy.factory_calls.lock().unwrap().is_empty());
    spy.fail_next.store(true, Ordering::SeqCst);
    assert_eq!(
        server
            .client
            .config(get_config(&session))
            .await
            .unwrap_err()
            .code(),
        Code::Unavailable
    );
    server.client.config(get_config(&session)).await.unwrap();
    assert_eq!(spy.factory_calls.lock().unwrap().len(), 2);
    server
        .client
        .release_session(release_session(&session))
        .await
        .unwrap();
    wait_for_count(&spy.sessions_dropped, 1).await;
    server
        .client
        .release_session(release_session(&session))
        .await
        .unwrap();
    assert_eq!(spy.factory_calls.lock().unwrap().len(), 2);
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn native_streams_drop_on_release_disconnect_session_release_and_shutdown() {
    let spy = Arc::new(Spy::default());
    let mut server = Server::start(spy.clone()).await;
    let session = Uuid::new_v4().to_string();
    let live = request(&session, LIVE);
    let operation = live.operation_id.clone().unwrap();
    let mut stream = start_live(&mut server, live.clone()).await;
    let mut duplicate = command_request(
        &session,
        proto::command::CommandType::SqlCommand(proto::SqlCommand {
            input: Some(relation(1)),
            ..Default::default()
        }),
    );
    duplicate.operation_id = Some(operation.clone());
    assert_eq!(
        server
            .client
            .execute_plan(duplicate)
            .await
            .unwrap_err()
            .code(),
        Code::AlreadyExists
    );
    assert_eq!(spy.commands.load(Ordering::SeqCst), 0);
    assert_eq!(spy.prepares.load(Ordering::SeqCst), 1);
    server
        .client
        .release_execute(release(&session, &operation))
        .await
        .unwrap();
    assert_eq!(
        spy.streams_dropped.load(Ordering::SeqCst),
        1,
        "release must wait for backend stream teardown"
    );
    assert_cancelled(&mut stream).await;

    let abandoned = start_live(&mut server, live.clone()).await;
    drop(abandoned);
    wait_for_count(&spy.streams_dropped, 2).await;
    let mut restarted = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match server.client.execute_plan(live.clone()).await {
                Ok(response) => break response.into_inner(),
                Err(status) if status.code() == Code::AlreadyExists => {
                    tokio::time::sleep(Duration::from_millis(10)).await
                }
                Err(status) => panic!("unexpected restart error: {status}"),
            }
        }
    })
    .await
    .expect("disconnected operation must be removed");
    assert!(restarted.message().await.unwrap().unwrap().schema.is_some());
    assert!(matches!(
        restarted.message().await.unwrap().unwrap().response_type,
        Some(ResponseType::ArrowBatch(_))
    ));
    server
        .client
        .release_execute(release(&session, &operation))
        .await
        .unwrap();
    assert_cancelled(&mut restarted).await;
    wait_for_count(&spy.streams_dropped, 3).await;

    let mut live = start_live(&mut server, request(&session, LIVE)).await;
    server
        .client
        .release_session(release_session(&session))
        .await
        .unwrap();
    assert_cancelled(&mut live).await;
    wait_for_count(&spy.sessions_dropped, 1).await;
    assert_eq!(
        server
            .client
            .execute_plan(request(&session, 99))
            .await
            .unwrap_err()
            .code(),
        Code::FailedPrecondition
    );
    let other = Uuid::new_v4().to_string();
    let mut live = start_live(&mut server, request(&other, LIVE)).await;
    server.shutdown.take().unwrap().send(()).unwrap();
    assert_cancelled(&mut live).await;
    server.stop().await;
    wait_for_count(&spy.streams_dropped, 5).await;
    wait_for_count(&spy.sessions_dropped, 2).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn shutdown_releases_backend_session_while_a_result_remains_unread() {
    let spy = Arc::new(Spy::default());
    let mut server = Server::start(spy.clone()).await;
    let (mut client, connection) = h2::client::Builder::new()
        .initial_window_size(1024)
        .initial_connection_window_size(1024)
        .handshake::<_, tonic::codegen::Bytes>(
            tokio::net::TcpStream::connect(server.address)
                .await
                .unwrap(),
        )
        .await
        .unwrap();
    let connection = tokio::spawn(async move {
        let _ = connection.await;
    });
    let request = request(&Uuid::new_v4().to_string(), FLOODED_RESULT).encode_to_vec();
    let mut payload = vec![0];
    payload.extend_from_slice(&(request.len() as u32).to_be_bytes());
    payload.extend_from_slice(&request);
    let request = tonic::codegen::http::Request::builder()
        .method("POST")
        .uri(format!(
            "http://{}/spark.connect.SparkConnectService/ExecutePlan",
            server.address
        ))
        .header("content-type", "application/grpc")
        .body(())
        .unwrap();
    let (response, mut upload) = client.send_request(request, false).unwrap();
    upload.send_data(payload.into(), true).unwrap();
    let response = tokio::time::timeout(Duration::from_secs(5), response)
        .await
        .expect("response headers must arrive")
        .unwrap();
    // Keep the body and HTTP/2 connection alive without reading or releasing
    // capacity. Each 512 KiB batch exceeds the stream's 1 KiB receive window.
    tokio::time::timeout(Duration::from_secs(5), async {
        while spy.batches_polled.load(Ordering::SeqCst) < 4 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("the frontend result queue must fill behind the unread peer");
    assert_eq!(spy.sessions_dropped.load(Ordering::SeqCst), 0);
    server.shutdown.take().unwrap().send(()).unwrap();
    // Shutdown must release the session registry and the producer's session
    // reference before waiting to send a terminal error through the full queue.
    wait_for_count(&spy.streams_dropped, 1).await;
    wait_for_count(&spy.sessions_dropped, 1).await;
    assert!(!response.body().is_end_stream());
    drop(response);
    drop(upload);
    drop(client);
    connection.abort();
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn operation_reservation_precedes_prepare_and_cleans_up_after_cancellation_and_errors() {
    let spy = Arc::new(Spy::default());
    let mut server = Server::start(spy.clone()).await;
    let session = Uuid::new_v4().to_string();
    let blocked = request(&session, BLOCKED_PREPARE);
    let operation = blocked.operation_id.clone().unwrap();
    let mut pending_client = server.client.clone();
    let pending = tokio::spawn(async move { pending_client.execute_plan(blocked).await });
    wait_for_count(&spy.prepares, 1).await;
    let mut duplicate = command_request(
        &session,
        proto::command::CommandType::SqlCommand(proto::SqlCommand {
            input: Some(relation(1)),
            ..Default::default()
        }),
    );
    duplicate.operation_id = Some(operation.clone());
    assert_eq!(
        server
            .client
            .execute_plan(duplicate)
            .await
            .unwrap_err()
            .code(),
        Code::AlreadyExists
    );
    assert_eq!(spy.commands.load(Ordering::SeqCst), 0);
    assert_eq!(spy.prepares.load(Ordering::SeqCst), 1);
    server
        .client
        .release_execute(release(&session, &operation))
        .await
        .unwrap();
    assert_eq!(spy.blocked_prepares_dropped.load(Ordering::SeqCst), 1);
    assert_eq!(
        tokio::time::timeout(Duration::from_secs(5), pending)
            .await
            .unwrap()
            .unwrap()
            .unwrap_err()
            .code(),
        Code::Cancelled
    );

    let mut retry = request(&session, 99);
    retry.operation_id = Some(operation);
    let mut stream = server
        .client
        .execute_plan(retry)
        .await
        .unwrap()
        .into_inner();
    assert_eq!(values(&collect(&mut stream).await), vec![999]);
    let rejected = request(&session, -1);
    let mut retry = request(&session, 99);
    retry.operation_id.clone_from(&rejected.operation_id);
    assert_eq!(
        server
            .client
            .execute_plan(rejected)
            .await
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );
    let mut stream = server
        .client
        .execute_plan(retry)
        .await
        .unwrap()
        .into_inner();
    assert_eq!(values(&collect(&mut stream).await), vec![999]);
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_and_shutdown_cancel_backend_metadata_futures_before_dropping_sessions() {
    let spy = Arc::new(Spy::default());
    let mut server = Server::start(spy.clone()).await;
    let session = Uuid::new_v4().to_string();
    let mut analysis_client = server.client.clone();
    let analysis_session = session.clone();
    let analysis = tokio::spawn(async move {
        analysis_client
            .analyze_plan(proto::AnalyzePlanRequest {
                session_id: analysis_session.clone(),
                user_context: user("alice"),
                analyze: Some(proto::analyze_plan_request::Analyze::IsStreaming(
                    proto::analyze_plan_request::IsStreaming {
                        plan: request(&analysis_session, 99).plan,
                    },
                )),
                ..Default::default()
            })
            .await
    });
    wait_for_count(&spy.analyses, 1).await;
    server
        .client
        .release_session(release_session(&session))
        .await
        .unwrap();
    let status = tokio::time::timeout(Duration::from_secs(5), analysis)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert_eq!(status.code(), Code::Cancelled);
    wait_for_count(&spy.blocked_analyses_dropped, 1).await;
    wait_for_count(&spy.sessions_dropped, 1).await;
    assert_eq!(
        *spy.metadata_drop_observations.lock().unwrap(),
        vec![(1, 0)]
    );

    let session = Uuid::new_v4().to_string();
    let mut config_client = server.client.clone();
    let config = tokio::spawn(async move {
        config_client
            .config(config_request(
                &session,
                proto::config_request::operation::OpType::Get(proto::config_request::Get {
                    keys: vec!["fake.block".into()],
                }),
            ))
            .await
    });
    wait_for_count(&spy.configs, 1).await;
    server.shutdown.take().unwrap().send(()).unwrap();
    let status = tokio::time::timeout(Duration::from_secs(5), config)
        .await
        .unwrap()
        .unwrap()
        .unwrap_err();
    assert_eq!(status.code(), Code::Cancelled);
    wait_for_count(&spy.blocked_configs_dropped, 1).await;
    server.stop().await;
    wait_for_count(&spy.sessions_dropped, 2).await;
    assert_eq!(
        *spy.metadata_drop_observations.lock().unwrap(),
        vec![(1, 0), (1, 1)]
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn slow_backend_drop_does_not_hold_the_global_session_registry_lock() {
    let spy = Arc::new(Spy::default());
    let mut server = Server::start(spy.clone()).await;
    // Channel clones share an HTTP/2 connection. Establish a new TCP connection
    // before blocking Drop so the probe isolates the global registry lock.
    let mut independent = SparkConnectServiceClient::connect(format!("http://{}", server.address))
        .await
        .unwrap();
    let slow_session = Uuid::new_v4().to_string();
    let gate = Arc::new(DropGate::default());
    let unblock = UnblockDrop(gate.clone());
    *spy.slow_drop.lock().unwrap() = Some((slow_session.clone(), gate.clone()));
    server
        .client
        .config(get_config(&slow_session))
        .await
        .unwrap();
    let mut release_client = server.client.clone();
    let released = tokio::spawn(async move {
        release_client
            .release_session(release_session(&slow_session))
            .await
    });
    // Store outcomes without asserting while Drop is blocked. Its own three-second
    // deadline and the RAII unblock guard also cover failures in this test.
    let entered = tokio::time::timeout(Duration::from_secs(2), async {
        while gate.entered.load(Ordering::SeqCst) == 0 {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await;
    let other = tokio::time::timeout(
        Duration::from_secs(1),
        independent.config(get_config(&Uuid::new_v4().to_string())),
    )
    .await;
    let still_blocked = !gate.exited.load(Ordering::SeqCst);
    drop(unblock);
    let released = tokio::time::timeout(Duration::from_secs(5), released).await;
    server.stop().await;

    assert!(entered.is_ok(), "slow Drop must start");
    other
        .expect("another session must remain responsive while Drop is blocked")
        .unwrap();
    assert!(
        still_blocked,
        "the independent request must complete before Drop exits"
    );
    assert!(
        !gate.timed_out.load(Ordering::SeqCst),
        "Drop must be explicitly unblocked"
    );
    released.unwrap().unwrap().unwrap();
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn frontend_rejects_oversized_analysis_and_batch_type_mismatches() {
    let spy = Arc::new(Spy::default());
    let mut server = Server::start(spy.clone()).await;
    let session = Uuid::new_v4().to_string();
    let status = server
        .client
        .analyze_plan(proto::AnalyzePlanRequest {
            session_id: session.clone(),
            user_context: user("alice"),
            analyze: Some(proto::analyze_plan_request::Analyze::Schema(
                proto::analyze_plan_request::Schema {
                    plan: request(&session, 99).plan,
                },
            )),
            ..Default::default()
        })
        .await
        .unwrap_err();
    assert_eq!(status.code(), Code::ResourceExhausted);

    let mut mismatch = server
        .client
        .execute_plan(request(&session, MISMATCHED_BATCH))
        .await
        .unwrap()
        .into_inner();
    let header = mismatch.message().await.unwrap().unwrap();
    assert!(header.schema.is_some());
    assert!(header.response_type.is_none());
    // The very next event must be the error: no inconsistent Arrow payload or
    // successful completion may escape after the declared Int64 schema.
    assert_eq!(mismatch.message().await.unwrap_err().code(), Code::Internal);

    let mut compatible = server
        .client
        .execute_plan(request(&session, ANNOTATED_BATCH))
        .await
        .unwrap()
        .into_inner();
    let responses = collect(&mut compatible).await;
    assert_eq!(values(&responses), vec![999]);
    assert!(matches!(
        responses.last().unwrap().response_type,
        Some(ResponseType::ResultComplete(_))
    ));
    server.stop().await;
}
