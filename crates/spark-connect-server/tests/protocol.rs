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

//! Verify observable Spark Connect wire behavior independently of DataFrame conveniences.

use std::collections::HashSet;
use std::io::Cursor;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

use arrow::array::Int64Array;
use arrow::ipc::reader::StreamReader;
use proto::execute_plan_response::ResponseType;
use proto::spark_connect_service_client::SparkConnectServiceClient;
use spark_connect_proto as proto;
use spark_connect_server::engine::{Engine, EngineSession, SessionContext};

mod support;
use support::{FixtureEngine, LIVE_FIXTURE};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tonic::transport::Channel;
use tonic::Code;
use uuid::Uuid;

struct Server {
    client: SparkConnectServiceClient<Channel>,
    shutdown: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<()>>,
}

impl Server {
    async fn start() -> Self {
        Self::start_with_engine(Arc::new(FixtureEngine::default())).await
    }

    async fn start_with_engine(engine: Arc<dyn Engine>) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (shutdown, stopped) = oneshot::channel();
        let task = tokio::spawn(async move {
            spark_connect_server::serve_with_engine(listener, engine, async {
                let _ = stopped.await;
            })
            .await
            .unwrap();
        });
        let client = SparkConnectServiceClient::connect(format!("http://{address}"))
            .await
            .unwrap();
        Self {
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

#[derive(Default)]
struct CountingEngine {
    sessions_created: AtomicUsize,
}

impl Engine for CountingEngine {
    fn create_session(
        &self,
        context: SessionContext,
    ) -> Result<Arc<dyn EngineSession>, tonic::Status> {
        self.sessions_created.fetch_add(1, Ordering::SeqCst);
        FixtureEngine::default().create_session(context)
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

fn range(session: &str, end: i64) -> proto::ExecutePlanRequest {
    proto::ExecutePlanRequest {
        session_id: session.into(),
        user_context: Some(proto::UserContext::default()),
        operation_id: Some(Uuid::new_v4().to_string()),
        plan: Some(proto::Plan {
            op_type: Some(proto::plan::OpType::Root(proto::Relation {
                rel_type: Some(proto::relation::RelType::Range(proto::Range {
                    start: Some(0),
                    end,
                    step: 1,
                    num_partitions: None,
                })),
                ..Default::default()
            })),
        }),
        ..Default::default()
    }
}

fn release(session: &str, operation: &str) -> proto::ReleaseExecuteRequest {
    proto::ReleaseExecuteRequest {
        session_id: session.into(),
        user_context: Some(proto::UserContext::default()),
        operation_id: operation.into(),
        release: Some(proto::release_execute_request::Release::ReleaseAll(
            proto::release_execute_request::ReleaseAll {},
        )),
        ..Default::default()
    }
}

fn config_request(
    session: &str,
    operation: proto::config_request::operation::OpType,
) -> proto::ConfigRequest {
    proto::ConfigRequest {
        session_id: session.into(),
        user_context: Some(proto::UserContext::default()),
        operation: Some(proto::config_request::Operation {
            op_type: Some(operation),
        }),
        ..Default::default()
    }
}

async fn assert_cancelled(stream: &mut tonic::Streaming<proto::ExecutePlanResponse>) {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match stream.message().await {
                Ok(Some(response)) => assert!(!matches!(
                    response.response_type,
                    Some(ResponseType::ResultComplete(_))
                )),
                Ok(None) => panic!("cancelled execution must report cancellation"),
                Err(status) => {
                    assert_eq!(status.code(), Code::Cancelled);
                    break;
                }
            }
        }
    })
    .await
    .expect("cancelled stream must terminate");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn arrow_stream_preserves_schema_ids_offsets_and_empty_results() {
    let mut server = Server::start().await;
    let session = Uuid::new_v4().to_string();
    let mut server_session = None;
    for end in [0, 25_000] {
        let request = range(&session, end);
        let operation = request.operation_id.clone().unwrap();
        let mut stream = server
            .client
            .execute_plan(request)
            .await
            .unwrap()
            .into_inner();
        let mut response_ids = HashSet::new();
        let (mut rows, mut batches, mut schemas, mut complete) = (0, 0, 0, false);
        while let Some(response) = stream.message().await.unwrap() {
            assert!(!complete, "ResultComplete must be the last response");
            assert_eq!(response.session_id, session);
            assert_eq!(response.operation_id, operation);
            Uuid::parse_str(&response.response_id).unwrap();
            assert!(response_ids.insert(response.response_id));
            Uuid::parse_str(&response.server_side_session_id).unwrap();
            let stable = server_session.get_or_insert(response.server_side_session_id.clone());
            assert_eq!(*stable, response.server_side_session_id);
            if let Some(schema) = response.schema {
                let Some(proto::data_type::Kind::Struct(schema)) = schema.kind else {
                    panic!("struct schema");
                };
                assert_eq!(schema.fields.len(), 1);
                assert_eq!(schema.fields[0].name, "id");
                assert!(!schema.fields[0].nullable);
                assert!(matches!(
                    schema.fields[0].data_type.as_ref().unwrap().kind,
                    Some(proto::data_type::Kind::Long(_))
                ));
                schemas += 1;
            }
            match response.response_type {
                Some(ResponseType::ArrowBatch(batch)) => {
                    assert_eq!(batch.start_offset, Some(rows));
                    let reader = StreamReader::try_new(Cursor::new(batch.data), None).unwrap();
                    let mut batch_rows = 0;
                    for record in reader {
                        let record = record.unwrap();
                        let values = record
                            .column(0)
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .unwrap();
                        for index in 0..values.len() {
                            assert_eq!(values.value(index), rows + batch_rows + index as i64);
                        }
                        batch_rows += record.num_rows() as i64;
                    }
                    assert_eq!(batch_rows, batch.row_count);
                    rows += batch_rows;
                    batches += 1;
                }
                Some(ResponseType::ResultComplete(_)) => complete = true,
                None => {}
                other => panic!("unexpected result: {other:?}"),
            }
        }
        assert_eq!(rows, end);
        assert_eq!(schemas, 1);
        assert!(complete);
        assert!(batches >= 1, "empty execution must still return Arrow data");
        if end > 0 {
            // The fixture supplies multiple physical batches; all must reach the wire.
            assert!(batches > 1, "large result should stream multiple batches");
        }
    }
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn observed_session_mismatch_and_released_sessions_are_rejected() {
    let mut server = Server::start().await;
    let session = Uuid::new_v4().to_string();
    let mut first = server
        .client
        .execute_plan(range(&session, 0))
        .await
        .unwrap()
        .into_inner();
    let original = first
        .message()
        .await
        .unwrap()
        .unwrap()
        .server_side_session_id;
    while first.message().await.unwrap().is_some() {}
    let mut mismatch = range(&session, 1);
    mismatch.client_observed_server_side_session_id = Some(Uuid::new_v4().to_string());
    let error = server.client.execute_plan(mismatch).await.unwrap_err();
    assert_eq!(error.code(), Code::FailedPrecondition);
    let release = proto::ReleaseSessionRequest {
        session_id: session.clone(),
        user_context: Some(proto::UserContext::default()),
        ..Default::default()
    };
    let released = server
        .client
        .release_session(release.clone())
        .await
        .unwrap()
        .into_inner();
    assert_eq!(released.server_side_session_id, original);
    server.client.release_session(release).await.unwrap();
    let error = server
        .client
        .execute_plan(range(&session, 1))
        .await
        .unwrap_err();
    assert_eq!(error.code(), Code::FailedPrecondition);
    // A separate session on the same transport remains usable.
    let mut other = server
        .client
        .execute_plan(range(&Uuid::new_v4().to_string(), 0))
        .await
        .unwrap()
        .into_inner();
    while other.message().await.unwrap().is_some() {}
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_all_cancels_a_pending_backend_stream_and_waits_for_teardown() {
    let engine = Arc::new(FixtureEngine::default());
    let mut server = Server::start_with_engine(engine.clone()).await;
    let session = Uuid::new_v4().to_string();
    let request = range(&session, LIVE_FIXTURE);
    let operation = request.operation_id.clone().unwrap();
    let mut stream = server
        .client
        .execute_plan(request)
        .await
        .unwrap()
        .into_inner();
    let header = stream.message().await.unwrap().unwrap();
    assert!(header.schema.is_some());
    let batch = stream.message().await.unwrap().unwrap();
    assert!(matches!(
        batch.response_type,
        Some(ResponseType::ArrowBatch(_))
    ));
    let release = proto::ReleaseExecuteRequest {
        session_id: session.clone(),
        user_context: Some(proto::UserContext::default()),
        operation_id: operation.clone(),
        client_observed_server_side_session_id: Some(header.server_side_session_id),
        release: Some(proto::release_execute_request::Release::ReleaseAll(
            proto::release_execute_request::ReleaseAll {},
        )),
        ..Default::default()
    };
    let released = tokio::time::timeout(
        Duration::from_secs(3),
        server.client.release_execute(release.clone()),
    )
    .await
    .expect("ReleaseAll must finish without consuming the result")
    .unwrap()
    .into_inner();
    assert_eq!(released.operation_id.as_deref(), Some(operation.as_str()));
    assert_eq!(
        engine.streams_dropped.load(Ordering::SeqCst),
        1,
        "ReleaseAll must wait until the backend stream is dropped"
    );
    let absent = server
        .client
        .release_execute(release)
        .await
        .unwrap()
        .into_inner();
    assert_eq!(
        absent.operation_id, None,
        "completed teardown must remove the operation"
    );
    assert_cancelled(&mut stream).await;
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn arrow_payloads_are_byte_limited_and_oversized_rows_fail() {
    use arrow::array::{LargeStringArray, StringArray, StringViewArray};

    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;

    let session = Uuid::new_v4().to_string();
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("s", DataType::Utf8, false),
    ]));
    // Skewed string sizes make initial byte-ratio slices too large, forcing
    // refinement while preserving every value and row offset.
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(Int64Array::from_iter_values(0..8192)),
            Arc::new(StringArray::from_iter_values(
                (0..8192).map(|id| "x".repeat(if id % 64 == 0 { 200000 } else { 1000 })),
            )),
        ],
    )
    .unwrap();
    let oversized_schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, false)]));
    let oversized = RecordBatch::try_new(
        oversized_schema.clone(),
        vec![Arc::new(StringArray::from(vec!["x".repeat(4000000)]))],
    )
    .unwrap();
    let engine = FixtureEngine::default()
        .with_batches(2, schema, vec![batch])
        .with_batches(3, oversized_schema, vec![oversized]);
    let mut server = Server::start_with_engine(Arc::new(engine)).await;
    let mut stream = server
        .client
        .execute_plan(range(&session, 2))
        .await
        .unwrap()
        .into_inner();
    let (mut rows, mut batches, mut complete) = (0, 0, false);
    while let Some(response) = stream.message().await.unwrap() {
        match response.response_type {
            Some(ResponseType::ArrowBatch(batch)) => {
                assert!(batch.data.len() <= 3 * 1024 * 1024);
                assert_eq!(batch.start_offset, Some(rows));
                let records = StreamReader::try_new(Cursor::new(batch.data), None).unwrap();
                let mut decoded = 0;
                for record in records {
                    let record = record.unwrap();
                    let ids = record
                        .column(0)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap();
                    let strings = record.column(1);
                    for index in 0..record.num_rows() {
                        let id = rows + decoded + index as i64;
                        assert_eq!(ids.value(index), id);
                        let value = match strings.data_type() {
                            arrow::datatypes::DataType::Utf8 => strings
                                .as_any()
                                .downcast_ref::<StringArray>()
                                .unwrap()
                                .value(index),
                            arrow::datatypes::DataType::LargeUtf8 => strings
                                .as_any()
                                .downcast_ref::<LargeStringArray>()
                                .unwrap()
                                .value(index),
                            arrow::datatypes::DataType::Utf8View => strings
                                .as_any()
                                .downcast_ref::<StringViewArray>()
                                .unwrap()
                                .value(index),
                            other => panic!("unexpected string type: {other:?}"),
                        };
                        assert_eq!(value.len(), if id % 64 == 0 { 200000 } else { 1000 });
                        assert!(value.bytes().all(|byte| byte == b'x'));
                    }
                    decoded += record.num_rows() as i64;
                }
                assert_eq!(decoded, batch.row_count);
                rows += decoded;
                batches += 1;
            }
            Some(ResponseType::ResultComplete(_)) => complete = true,
            None => {}
            other => panic!("unexpected result: {other:?}"),
        }
    }
    assert_eq!(rows, 8192);
    assert!(batches > 1);
    assert!(complete);

    let mut oversized = server
        .client
        .execute_plan(range(&session, 3))
        .await
        .unwrap()
        .into_inner();
    assert!(oversized.message().await.unwrap().unwrap().schema.is_some());
    let status = oversized.message().await.unwrap_err();
    assert_eq!(status.code(), Code::ResourceExhausted);
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn release_session_cancels_live_stream_and_live_operation_ids_are_unique() {
    let mut server = Server::start().await;
    let session = Uuid::new_v4().to_string();
    let request = range(&session, LIVE_FIXTURE);
    let mut stream = server
        .client
        .execute_plan(request.clone())
        .await
        .unwrap()
        .into_inner();
    assert!(stream.message().await.unwrap().unwrap().schema.is_some());
    let duplicate = server.client.execute_plan(request).await.unwrap_err();
    assert_eq!(duplicate.code(), Code::AlreadyExists);
    server
        .client
        .release_session(proto::ReleaseSessionRequest {
            session_id: session.clone(),
            user_context: Some(proto::UserContext::default()),
            ..Default::default()
        })
        .await
        .unwrap();
    assert_cancelled(&mut stream).await;
    assert_eq!(
        server
            .client
            .execute_plan(range(&session, 0))
            .await
            .unwrap_err()
            .code(),
        Code::FailedPrecondition
    );
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_requests_do_not_allocate_sessions_and_unknown_release_is_rejected() {
    let engine = Arc::new(CountingEngine::default());
    let mut server = Server::start_with_engine(engine.clone()).await;
    let valid = Uuid::new_v4().to_string();
    assert_eq!(
        server
            .client
            .execute_plan(range("not-a-uuid", 0))
            .await
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );
    let mut invalid_operation = range(&valid, 0);
    invalid_operation.operation_id = Some("not-a-uuid".into());
    assert_eq!(
        server
            .client
            .execute_plan(invalid_operation)
            .await
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );
    for plan in [None, Some(proto::Plan { op_type: None })] {
        let mut request = range(&valid, 0);
        request.plan = plan;
        assert_eq!(
            server
                .client
                .execute_plan(request)
                .await
                .unwrap_err()
                .code(),
            Code::InvalidArgument
        );
    }
    assert_eq!(
        server
            .client
            .analyze_plan(proto::AnalyzePlanRequest {
                session_id: valid.clone(),
                ..Default::default()
            })
            .await
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );
    for operation in [
        None,
        Some(proto::config_request::Operation { op_type: None }),
    ] {
        assert_eq!(
            server
                .client
                .config(proto::ConfigRequest {
                    session_id: valid.clone(),
                    operation,
                    ..Default::default()
                })
                .await
                .unwrap_err()
                .code(),
            Code::InvalidArgument
        );
    }
    // Identifier and envelope validation must precede backend session allocation.
    assert_eq!(engine.sessions_created.load(Ordering::SeqCst), 0);
    assert_eq!(
        server
            .client
            .release_execute(release(&valid, &Uuid::new_v4().to_string()))
            .await
            .unwrap_err()
            .code(),
        Code::NotFound
    );
    assert_eq!(
        server
            .client
            .release_execute(release("not-a-uuid", &valid))
            .await
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );
    assert_eq!(
        server
            .client
            .release_execute(release(&valid, "not-a-uuid"))
            .await
            .unwrap_err()
            .code(),
        Code::InvalidArgument
    );
    let unknown = Uuid::new_v4().to_string();
    assert_eq!(
        server
            .client
            .release_execute(release(&unknown, &valid))
            .await
            .unwrap_err()
            .code(),
        Code::NotFound
    );
    assert_eq!(engine.sessions_created.load(Ordering::SeqCst), 0);
    server
        .client
        .config(config_request(
            &valid,
            proto::config_request::operation::OpType::GetOption(proto::config_request::GetOption {
                keys: vec!["spark.sql.session.timeZone".into()],
            }),
        ))
        .await
        .unwrap();
    assert_eq!(engine.sessions_created.load(Ordering::SeqCst), 1);
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn disconnect_removes_operation_and_shutdown_cancels_live_streams() {
    let mut server = Server::start().await;
    let session = Uuid::new_v4().to_string();
    let request = range(&session, LIVE_FIXTURE);
    let operation = request.operation_id.clone().unwrap();
    let mut abandoned = server
        .client
        .execute_plan(request.clone())
        .await
        .unwrap()
        .into_inner();
    assert!(abandoned.message().await.unwrap().unwrap().schema.is_some());
    drop(abandoned);
    let mut resumed = tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            match server.client.execute_plan(request.clone()).await {
                Ok(response) => break response.into_inner(),
                Err(status) if status.code() == Code::AlreadyExists => {
                    tokio::time::sleep(Duration::from_millis(20)).await;
                }
                Err(status) => panic!("unexpected retry failure: {status}"),
            }
        }
    })
    .await
    .expect("disconnect must remove the operation registry entry");
    assert!(resumed.message().await.unwrap().unwrap().schema.is_some());
    server
        .client
        .release_execute(release(&session, &operation))
        .await
        .unwrap();
    assert_cancelled(&mut resumed).await;

    let mut live = server
        .client
        .execute_plan(range(&session, LIVE_FIXTURE))
        .await
        .unwrap()
        .into_inner();
    assert!(live.message().await.unwrap().unwrap().schema.is_some());
    server.shutdown.take().unwrap().send(()).unwrap();
    assert_cancelled(&mut live).await;
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn backend_status_codes_survive_prepare_and_stream_failures() {
    let engine = FixtureEngine::default()
        .reject(10, Code::Unimplemented)
        .reject(11, Code::InvalidArgument)
        .stream_error(12, Code::ResourceExhausted);
    let mut server = Server::start_with_engine(Arc::new(engine)).await;
    let session = Uuid::new_v4().to_string();
    for (selector, expected) in [(10, Code::Unimplemented), (11, Code::InvalidArgument)] {
        let status = server
            .client
            .execute_plan(range(&session, selector))
            .await
            .unwrap_err();
        assert_eq!(status.code(), expected);
    }
    let mut stream = server
        .client
        .execute_plan(range(&session, 12))
        .await
        .unwrap()
        .into_inner();
    assert!(stream.message().await.unwrap().unwrap().schema.is_some());
    assert_eq!(
        stream.message().await.unwrap_err().code(),
        Code::ResourceExhausted
    );
    // Backend failures must not poison the protocol session or its transport.
    let mut recovered = server
        .client
        .execute_plan(range(&session, 1))
        .await
        .unwrap()
        .into_inner();
    let (mut rows, mut complete) = (0, false);
    while let Some(response) = recovered.message().await.unwrap() {
        match response.response_type {
            Some(ResponseType::ArrowBatch(batch)) => rows += batch.row_count,
            Some(ResponseType::ResultComplete(_)) => complete = true,
            None => {}
            other => panic!("unexpected result: {other:?}"),
        }
    }
    assert_eq!(rows, 1);
    assert!(complete);
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn deeply_nested_requests_hit_the_standard_protobuf_decoder_limit() {
    let mut server = Server::start().await;
    let session = Uuid::new_v4().to_string();
    let mut request = range(&session, 1);
    let Some(proto::plan::OpType::Root(mut relation)) = request.plan.take().unwrap().op_type else {
        unreachable!()
    };
    for _ in 0..60 {
        relation = proto::Relation {
            rel_type: Some(proto::relation::RelType::Filter(Box::new(proto::Filter {
                input: Some(Box::new(relation)),
                condition: Some(proto::Expression {
                    expr_type: Some(proto::expression::ExprType::Literal(
                        proto::expression::Literal {
                            literal_type: Some(proto::expression::literal::LiteralType::Boolean(
                                true,
                            )),
                            ..Default::default()
                        },
                    )),
                    ..Default::default()
                }),
            }))),
            ..Default::default()
        };
    }
    request.plan = Some(proto::Plan {
        op_type: Some(proto::plan::OpType::Root(relation)),
    });
    let status = server.client.execute_plan(request).await.unwrap_err();
    // Each filter introduces nested protobuf messages. The normal decoder rejects
    // this request before the backend can inspect it.
    assert_eq!(status.code(), Code::Internal);
    assert!(status.message().contains("recursion"), "{status}");
    server.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn view_backing_buffers_are_compacted_before_splitting_including_struct_children() {
    use std::sync::Arc;

    use arrow::array::{BinaryArray, BinaryViewArray, StringArray, StringViewArray, StructArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;

    let session = Uuid::new_v4().to_string();
    let text = |id: i64| format!("{id:04}{}", "x".repeat(1020));
    let nested_text = |id: i64| format!("n{id:04}{}", "y".repeat(1019));
    let binary = |id: i64| {
        let mut bytes = id.to_le_bytes().to_vec();
        bytes.extend(std::iter::repeat_n((id % 251) as u8, 1016));
        bytes
    };
    let strings = StringViewArray::from_iter_values((0..4096).map(text));
    let binaries = BinaryViewArray::from_iter_values((0..4096).map(binary));
    let nested_strings = StringViewArray::from_iter_values((0..4096).map(nested_text));
    // A one-row slice of these view arrays still references backing buffers
    // larger than the response limit, although each individual value is 1 KiB.
    for size in [
        strings
            .data_buffers()
            .iter()
            .map(|buffer| buffer.len())
            .sum::<usize>(),
        binaries
            .data_buffers()
            .iter()
            .map(|buffer| buffer.len())
            .sum::<usize>(),
        nested_strings
            .data_buffers()
            .iter()
            .map(|buffer| buffer.len())
            .sum::<usize>(),
    ] {
        assert!(size > 3 * 1024 * 1024);
    }
    let nested_fields: arrow::datatypes::Fields =
        vec![Field::new("text", DataType::Utf8View, false)].into();
    let nested = StructArray::new(nested_fields.clone(), vec![Arc::new(nested_strings)], None);
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("text", DataType::Utf8View, false),
            Field::new("binary", DataType::BinaryView, false),
            Field::new("nested", DataType::Struct(nested_fields), false),
        ])),
        vec![
            Arc::new(Int64Array::from_iter_values(0..4096)),
            Arc::new(strings),
            Arc::new(binaries),
            Arc::new(nested),
        ],
    )
    .unwrap();
    let engine = FixtureEngine::default().with_batches(42, batch.schema(), vec![batch]);
    let mut server = Server::start_with_engine(Arc::new(engine)).await;
    let request = range(&session, 42);
    let mut stream = server
        .client
        .execute_plan(request)
        .await
        .unwrap()
        .into_inner();
    let (mut rows, mut batches, mut complete) = (0, 0, false);
    while let Some(response) = stream.message().await.unwrap() {
        match response.response_type {
            Some(ResponseType::ArrowBatch(batch)) => {
                assert!(batch.data.len() <= 3 * 1024 * 1024);
                assert_eq!(batch.start_offset, Some(rows));
                let records = StreamReader::try_new(Cursor::new(batch.data), None).unwrap();
                let mut decoded = 0;
                for record in records {
                    let record = record.unwrap();
                    let ids = record
                        .column(0)
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .unwrap();
                    assert_eq!(record.column(1).data_type(), &DataType::LargeUtf8);
                    assert_eq!(record.column(2).data_type(), &DataType::LargeBinary);
                    let strings = arrow::compute::cast(record.column(1), &DataType::Utf8).unwrap();
                    let strings = strings.as_any().downcast_ref::<StringArray>().unwrap();
                    let binaries =
                        arrow::compute::cast(record.column(2), &DataType::Binary).unwrap();
                    let binaries = binaries.as_any().downcast_ref::<BinaryArray>().unwrap();
                    let nested = record
                        .column(3)
                        .as_any()
                        .downcast_ref::<StructArray>()
                        .unwrap();
                    assert_eq!(nested.column(0).data_type(), &DataType::LargeUtf8);
                    let nested_strings =
                        arrow::compute::cast(nested.column(0), &DataType::Utf8).unwrap();
                    let nested_strings = nested_strings
                        .as_any()
                        .downcast_ref::<StringArray>()
                        .unwrap();
                    for index in 0..record.num_rows() {
                        let id = rows + decoded + index as i64;
                        assert_eq!(ids.value(index), id);
                        assert_eq!(strings.value(index), text(id));
                        assert_eq!(binaries.value(index), binary(id));
                        assert_eq!(nested_strings.value(index), nested_text(id));
                    }
                    decoded += record.num_rows() as i64;
                }
                assert_eq!(decoded, batch.row_count);
                rows += decoded;
                batches += 1;
            }
            Some(ResponseType::ResultComplete(_)) => complete = true,
            None => {}
            other => panic!("unexpected result: {other:?}"),
        }
    }
    assert_eq!(rows, 4096);
    assert!(batches > 1);
    assert!(complete);
    server.stop().await;
}
