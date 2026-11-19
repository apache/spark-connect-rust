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

use std::collections::{HashMap, VecDeque};
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use arrow::datatypes::{Schema, SchemaRef};
use arrow::ipc::{reader::StreamReader, writer::StreamWriter};
use arrow::record_batch::RecordBatch;
use futures::{FutureExt, Stream, StreamExt};
use prost::Message;
use spark_connect_proto as proto;
use tokio::sync::{mpsc, watch};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use uuid::Uuid;

use crate::engine::{Engine, EngineSession, PreparedOperation, ResultBatch, SessionContext};
use crate::grpc::spark_connect_service_server::SparkConnectService;
use crate::schema::{self, result_schema, MAX_METADATA_RESPONSE_BYTES};

type Output = Pin<Box<dyn Stream<Item = Result<proto::ExecutePlanResponse, Status>> + Send>>;
type SessionKey = (String, String);
const MAX_RELEASED_SESSIONS: usize = 4096;
const RELEASE_WAIT: std::time::Duration = std::time::Duration::from_secs(2);

fn parse_uuid(value: &str, name: &str) -> Result<Uuid, Status> {
    crate::ids::parse_uuid(value)
        .ok_or_else(|| Status::invalid_argument(format!("invalid UUID in {name}")))
}

async fn catch_engine_panic<T: Send>(
    action: impl std::future::Future<Output = Result<T, Status>> + Send,
) -> Result<T, Status> {
    AssertUnwindSafe(action)
        .catch_unwind()
        .await
        .unwrap_or_else(|_| Err(Status::internal("engine request panicked")))
}

struct Session {
    id: String,
    backend: Arc<dyn EngineSession>,
    cancelled: watch::Sender<bool>,
    operations: Mutex<HashMap<String, Arc<Operation>>>,
}

struct Operation {
    cancelled: watch::Sender<bool>,
    done: watch::Sender<bool>,
}

impl Session {
    fn new(backend: Arc<dyn EngineSession>) -> Self {
        Self {
            id: Uuid::new_v4().to_string(),
            backend,
            cancelled: watch::channel(false).0,
            operations: Mutex::new(HashMap::new()),
        }
    }

    async fn while_open<T: Send>(
        &self,
        action: impl std::future::Future<Output = Result<T, Status>> + Send,
    ) -> Result<T, Status> {
        let mut cancelled = self.cancelled.subscribe();
        if *cancelled.borrow() {
            return Err(Status::cancelled("session released"));
        }
        tokio::select! {
            biased;
            _ = cancelled.changed() => Err(Status::cancelled("session released")),
            result = action => result,
        }
    }

    fn operation(
        self: &Arc<Self>,
        id: &str,
    ) -> Result<(Arc<OperationGuard>, watch::Receiver<bool>), Status> {
        // Subscribe before publishing: ReleaseAll can cancel as soon as the
        // operation appears in the registry, before preparation is first polled.
        let (cancelled, cancellation) = watch::channel(false);
        let running = Arc::new(Operation {
            cancelled,
            done: watch::channel(false).0,
        });
        let mut operations = self
            .operations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if operations.contains_key(id) {
            return Err(Status::already_exists("operation is already running"));
        }
        operations.insert(id.to_owned(), running.clone());
        Ok((
            Arc::new(OperationGuard {
                session: self.clone(),
                id: id.to_owned(),
                running,
            }),
            cancellation,
        ))
    }
}

// This guard spans preparation and execution, including eager commands. Drop the
// backend future/stream before the guard so ReleaseAll observes completed teardown.
struct OperationGuard {
    session: Arc<Session>,
    id: String,
    running: Arc<Operation>,
}

impl Drop for OperationGuard {
    fn drop(&mut self) {
        self.session
            .operations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.id);
        self.running.done.send_replace(true);
    }
}

#[derive(Default)]
struct SessionRegistry {
    // None is a recently released-session tombstone, retained until eviction.
    entries: HashMap<SessionKey, Option<Arc<Session>>>,
    // Remember the most recent distinct closed IDs in release order. Repeated
    // releases do not refresh retention. After eviction, an old ID may create a
    // fresh session; active sessions never count toward this finite budget.
    released: VecDeque<SessionKey>,
}

impl SessionRegistry {
    fn release(&mut self, key: SessionKey, allow_reconnect: bool) -> Option<Arc<Session>> {
        // Releasing a missing or already closed session is a no-op. In
        // particular, a later reconnecting release cannot erase a tombstone.
        let session = self.entries.get_mut(&key)?.take()?;
        if allow_reconnect {
            self.entries.remove(&key);
        } else {
            self.released.push_back(key);
            if self.released.len() > MAX_RELEASED_SESSIONS {
                if let Some(oldest) = self.released.pop_front() {
                    self.entries.remove(&oldest);
                }
            }
        }
        Some(session)
    }
}

pub(crate) struct ConnectService {
    engine: Arc<dyn Engine>,
    options: crate::ServerOptions,
    sessions: Mutex<SessionRegistry>,
    closing: std::sync::atomic::AtomicBool,
}

impl ConnectService {
    pub(crate) fn new(engine: Arc<dyn Engine>, options: crate::ServerOptions) -> Self {
        Self {
            engine,
            options,
            sessions: Mutex::new(SessionRegistry::default()),
            closing: std::sync::atomic::AtomicBool::new(false),
        }
    }

    fn key(id: &str, user: Option<&proto::UserContext>) -> SessionKey {
        (
            user.map(|u| u.user_id.clone()).unwrap_or_default(),
            id.into(),
        )
    }

    fn session(
        &self,
        id: &str,
        user: Option<&proto::UserContext>,
        observed: Option<&str>,
    ) -> Result<Arc<Session>, Status> {
        self.find_session(id, user, observed, true)
    }

    fn find_session(
        &self,
        id: &str,
        user: Option<&proto::UserContext>,
        observed: Option<&str>,
        create: bool,
    ) -> Result<Arc<Session>, Status> {
        if create {
            parse_uuid(id, "session_id")?;
        }
        let key = Self::key(id, user);
        let mut sessions = self
            .sessions
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if self.closing.load(std::sync::atomic::Ordering::SeqCst) {
            return Err(Status::unavailable("server is shutting down"));
        }
        let session = match sessions.entries.get(&key) {
            Some(Some(session)) => session.clone(),
            Some(None) => return Err(Status::failed_precondition("session was released")),
            None if !create => return Err(Status::not_found("session does not exist")),
            // An observed ID refers to a previous incarnation; a new session
            // cannot satisfy it. Reject before allocating backend resources.
            None if observed.is_some() => {
                return Err(Status::failed_precondition(
                    "server-side session ID changed",
                ));
            }
            None => {
                let backend = self.engine.create_session(SessionContext {
                    session_id: id.to_owned(),
                    user_context: user.cloned(),
                })?;
                let session = Arc::new(Session::new(backend));
                sessions.entries.insert(key, Some(session.clone()));
                session
            }
        };
        if observed.is_some_and(|id| id != session.id) {
            return Err(Status::failed_precondition(
                "server-side session ID changed",
            ));
        }
        Ok(session)
    }

    pub(crate) fn cancel_all(&self) {
        self.closing
            .store(true, std::sync::atomic::Ordering::SeqCst);
        // Stalled HTTP/2 connections can retain the service after the bounded
        // transport drain. Remove its registry ownership now so backend sessions
        // can close once their cancelled operations finish.
        let sessions = std::mem::take(
            &mut *self
                .sessions
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        for session in sessions.entries.into_values().flatten() {
            session.cancelled.send_replace(true);
        }
    }
}

fn response(
    session_id: &str,
    session: &Session,
    operation_id: &str,
    response_type: Option<proto::execute_plan_response::ResponseType>,
) -> proto::ExecutePlanResponse {
    proto::ExecutePlanResponse {
        session_id: session_id.into(),
        server_side_session_id: session.id.clone(),
        operation_id: operation_id.into(),
        response_id: Uuid::new_v4().to_string(),
        response_type,
        ..Default::default()
    }
}

fn checked_response(
    message: proto::ExecutePlanResponse,
) -> Result<proto::ExecutePlanResponse, Status> {
    if message.encoded_len() > MAX_METADATA_RESPONSE_BYTES {
        return Err(Status::resource_exhausted(
            "schema/command response exceeds the 3 MiB limit",
        ));
    }
    Ok(message)
}

fn complete(id: &str, session: &Session, operation: &str) -> proto::ExecutePlanResponse {
    response(
        id,
        session,
        operation,
        Some(proto::execute_plan_response::ResponseType::ResultComplete(
            proto::execute_plan_response::ResultComplete {},
        )),
    )
}

fn encode_batch(batch: &RecordBatch) -> Result<(Vec<u8>, SchemaRef), Status> {
    use arrow::datatypes::DataType;
    use arrow::record_batch::RecordBatchOptions;

    // Materialize view arrays, including nested children, into the large-offset
    // string/binary representation used for encoded RecordBatch results.
    let source_schema = batch.schema();
    let fields = schema::map_fields(source_schema.fields(), &|field, data_type| {
        let data_type = match data_type {
            DataType::Utf8View => DataType::LargeUtf8,
            DataType::BinaryView => DataType::LargeBinary,
            other => other,
        };
        if data_type == *field.data_type() {
            field.clone()
        } else {
            Arc::new(field.as_ref().clone().with_data_type(data_type))
        }
    });
    let changed = fields
        .iter()
        .zip(source_schema.fields())
        .any(|(target, source)| target.data_type() != source.data_type());
    let materialized = if changed {
        let columns = batch
            .columns()
            .iter()
            .zip(&fields)
            .map(|(column, field)| {
                if column.data_type() == field.data_type() {
                    Ok(column.clone())
                } else {
                    arrow::compute::cast(column.as_ref(), field.data_type())
                }
            })
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| Status::internal(error.to_string()))?;
        let target_schema = Arc::new(Schema::new_with_metadata(
            fields,
            source_schema.metadata().clone(),
        ));
        Some(
            RecordBatch::try_new_with_options(
                target_schema,
                columns,
                &RecordBatchOptions::new().with_row_count(Some(batch.num_rows())),
            )
            .map_err(|error| Status::internal(error.to_string()))?,
        )
    } else {
        None
    };
    let batch = materialized.as_ref().unwrap_or(batch);
    let mut bytes = Vec::new();
    {
        let mut writer = StreamWriter::try_new(&mut bytes, batch.schema().as_ref())
            .map_err(|e| Status::internal(e.to_string()))?;
        writer
            .write(batch)
            .map_err(|e| Status::internal(e.to_string()))?;
        writer
            .finish()
            .map_err(|e| Status::internal(e.to_string()))?;
    }
    Ok((bytes, batch.schema()))
}

// PySpark assembles one table from all IPC batches. Its physical types and
// nullability must agree across batches, while field/schema metadata may differ.
fn check_output_schema(schema: &Schema, expected: &mut Option<Schema>) -> Result<(), Status> {
    use arrow::datatypes::Field;

    let actual = Schema::new(schema::map_fields(schema.fields(), &|field, data_type| {
        Arc::new(Field::new(field.name(), data_type, field.is_nullable()))
    }));
    match expected {
        Some(expected) if *expected != actual => Err(Status::internal(
            "Arrow output schema changed within one result",
        )),
        Some(_) => Ok(()),
        None => {
            *expected = Some(actual);
            Ok(())
        }
    }
}

// Keep backend ownership until the batch is dropped, including on encoder panic.
struct EncodingBatch {
    batch: RecordBatch,
    _guard: Arc<OperationGuard>,
}

impl EncodingBatch {
    fn encode(self) -> Result<(Vec<u8>, SchemaRef), Status> {
        encode_batch(&self.batch)
    }
}

fn arrow_chunk_size(
    options: &[proto::execute_plan_request::RequestOption],
    maximum: usize,
) -> Option<usize> {
    use proto::execute_plan_request::request_option::RequestOption;

    let mut chunking = options
        .iter()
        .filter_map(|option| match &option.request_option {
            Some(RequestOption::ResultChunkingOptions(options)) => Some(options),
            _ => None,
        });
    if maximum == 0
        || !chunking
            .clone()
            .any(|options| options.allow_arrow_batch_chunking)
    {
        return None;
    }
    Some(
        chunking
            .find_map(|options| options.preferred_arrow_chunk_size)
            // Validate i64 directly; Scala's toInt can wrap an invalid preference.
            .and_then(|bytes| usize::try_from(bytes).ok())
            .filter(|bytes| (1024..=maximum).contains(bytes))
            .unwrap_or(maximum),
    )
}

async fn send_batch(
    sender: &mpsc::Sender<Result<proto::ExecutePlanResponse, Status>>,
    id: &str,
    operation: &str,
    batch: ResultBatch,
    batch_schema: &Schema,
    offset: &mut i64,
    output_schema: &mut Option<Schema>,
    guard: &Arc<OperationGuard>,
    chunk_size: Option<usize>,
) -> Result<bool, Status> {
    // Reject unrepresentable counts/offsets before encoding or sending any chunk.
    let row_count = match &batch {
        ResultBatch::Arrow(batch) => i64::try_from(batch.num_rows())
            .map_err(|_| Status::resource_exhausted("result batch row count exceeds i64"))?,
        ResultBatch::Ipc { row_count, .. } if *row_count < 0 => {
            return Err(Status::internal("negative engine IPC row count"));
        }
        ResultBatch::Ipc { row_count, .. } => *row_count,
    };
    let next_offset = offset
        .checked_add(row_count)
        .ok_or_else(|| Status::resource_exhausted("result row offset exceeds i64"))?;
    let data: prost::bytes::Bytes = match batch {
        ResultBatch::Arrow(batch) => {
            // A blocking encoder cannot be cancelled once started. Retain engine
            // ownership until its batch is dropped, even if the caller goes away.
            let encoding = EncodingBatch {
                batch,
                _guard: guard.clone(),
            };
            let (data, schema) = tokio::task::spawn_blocking(move || encoding.encode())
                .await
                .map_err(|_| Status::internal("Arrow result encoding task failed"))??;
            check_output_schema(schema.as_ref(), output_schema)?;
            data.into()
        }
        ResultBatch::Ipc { data, .. } => {
            check_output_schema(batch_schema, output_schema)?;
            data.into()
        }
    };
    let mut payload = proto::execute_plan_response::ArrowBatch {
        row_count,
        start_offset: Some(*offset),
        ..Default::default()
    };
    let Some(chunk_bytes) = chunk_size else {
        // As on the JVM, clients without chunking opt-in receive the original
        // whole IPC batch. Their gRPC receive limit must accommodate it.
        payload.data = data;
        let message = response(
            id,
            &guard.session,
            operation,
            Some(proto::execute_plan_response::ResponseType::ArrowBatch(
                payload,
            )),
        );
        *offset = next_offset;
        return Ok(sender.send(Ok(message)).await.is_ok());
    };
    let chunks = data.len().div_ceil(chunk_bytes).max(1);
    payload.num_chunks_in_batch = Some(
        i64::try_from(chunks)
            .map_err(|_| Status::resource_exhausted("Arrow chunk count exceeds i64"))?,
    );
    for index in 0..chunks {
        let from = index * chunk_bytes;
        let to = from.saturating_add(chunk_bytes).min(data.len());
        // Bytes slices share the original IPC allocation, including compressed
        // buffers. Clients concatenate the chunks before decoding Arrow.
        payload.data = data.slice(from..to);
        payload.chunk_index = Some(index as i64);
        let message = response(
            id,
            &guard.session,
            operation,
            Some(proto::execute_plan_response::ResponseType::ArrowBatch(
                payload.clone(),
            )),
        );
        if sender.send(Ok(message)).await.is_err() {
            return Ok(false);
        }
    }
    *offset = next_offset;
    Ok(true)
}

#[tonic::async_trait]
impl SparkConnectService for ConnectService {
    type ExecutePlanStream = Output;
    type ReattachExecuteStream = Output;

    async fn execute_plan(
        &self,
        request: Request<proto::ExecutePlanRequest>,
    ) -> Result<Response<Output>, Status> {
        let req = request.into_inner();
        let chunk_size = arrow_chunk_size(&req.request_options, self.options.max_arrow_chunk_size);
        let reattachable = req.request_options.iter().any(|option| {
            matches!(
                &option.request_option,
                Some(proto::execute_plan_request::request_option::RequestOption::ReattachOptions(
                    options
                )) if options.reattachable
            )
        });
        let operation = match req.operation_id {
            Some(id) => parse_uuid(&id, "operation_id")?,
            None => Uuid::new_v4(),
        }
        .to_string();
        let op = req
            .plan
            .and_then(|p| p.op_type)
            .ok_or_else(|| Status::invalid_argument("missing plan"))?;
        let session = self.session(
            &req.session_id,
            req.user_context.as_ref(),
            req.client_observed_server_side_session_id.as_deref(),
        )?;
        let mut cancelled = session.cancelled.subscribe();
        if *cancelled.borrow() {
            return Err(Status::cancelled("session released"));
        }
        // Reserve before calling the backend: preparing a command can mutate state.
        let (guard, mut operation_cancelled) = session.operation(&operation)?;
        let prepared = tokio::select! {
            biased;
            _ = cancelled.changed() => return Err(Status::cancelled("session released")),
            _ = operation_cancelled.changed() => return Err(Status::cancelled("operation released")),
            prepared = catch_engine_panic(async { session.backend.prepare(op).await }) => prepared?,
        };
        let query = match prepared {
            PreparedOperation::Query(query) => query,
            PreparedOperation::Command(payloads) => {
                let mut messages = payloads
                    .into_iter()
                    .map(|payload| {
                        let message =
                            response(&req.session_id, &session, &operation, Some(payload));
                        checked_response(message)
                    })
                    .collect::<Result<Vec<_>, Status>>()?;
                if reattachable {
                    messages.push(complete(&req.session_id, &session, &operation));
                }
                return Ok(Response::new(Box::pin(tokio_stream::iter(
                    messages.into_iter().map(Ok),
                ))));
            }
        };
        let arrow_schema = query.schema();
        let spark_schema = result_schema(&arrow_schema)?;
        let expected_batch_schema = schema::comparable_schema(&arrow_schema)?;
        let (sender, receiver) = mpsc::channel(2);
        tokio::spawn(async move {
            let run = async {
                let mut header = response(&req.session_id, &session, &operation, None);
                header.schema = Some(spark_schema);
                let header = checked_response(header)?;
                if sender.send(Ok(header)).await.is_err() {
                    return Ok(());
                }
                let mut batches = query.execute().await?;
                let mut offset = 0;
                let mut output_schema = None;
                let mut sent = false;
                while let Some(batch) = batches.next().await {
                    let batch = batch?;
                    let batch_schema = match &batch {
                        ResultBatch::Arrow(batch) => batch.schema(),
                        ResultBatch::Ipc { data, .. } => {
                            StreamReader::try_new(data.as_slice(), None)
                                .map_err(|e| Status::internal(format!("invalid engine IPC: {e}")))?
                                .schema()
                        }
                    };
                    let actual_schema =
                        schema::comparable_schema(batch_schema.as_ref()).map_err(|error| {
                            Status::internal(format!(
                                "invalid engine batch schema: {}",
                                error.message()
                            ))
                        })?;
                    if actual_schema != expected_batch_schema {
                        return Err(Status::internal(
                            "engine batch schema differs from its declared schema",
                        ));
                    }
                    if !send_batch(
                        &sender,
                        &req.session_id,
                        &operation,
                        batch,
                        batch_schema.as_ref(),
                        &mut offset,
                        &mut output_schema,
                        &guard,
                        chunk_size,
                    )
                    .await?
                    {
                        return Ok(());
                    }
                    sent = true;
                }
                // Teardown can itself panic. Complete only after the backend
                // stream has been dropped within the panic boundary.
                drop(batches);
                if !sent {
                    let batch = RecordBatch::new_empty(arrow_schema);
                    let batch_schema = batch.schema();
                    if !send_batch(
                        &sender,
                        &req.session_id,
                        &operation,
                        batch.into(),
                        batch_schema.as_ref(),
                        &mut offset,
                        &mut output_schema,
                        &guard,
                        chunk_size,
                    )
                    .await?
                    {
                        return Ok(());
                    }
                }
                if reattachable {
                    let _ = sender
                        .send(Ok(complete(&req.session_id, &session, &operation)))
                        .await;
                }
                Ok::<_, Status>(())
            };
            let result = AssertUnwindSafe(async {
                tokio::select! {
                    biased;
                    _ = cancelled.changed() => Err(Status::cancelled("session released")),
                    _ = operation_cancelled.changed() => Err(Status::cancelled("operation released")),
                    _ = sender.closed() => Ok(()),
                    result = run => result,
                }
            })
            .catch_unwind()
            .await
            .unwrap_or_else(|_| Err(Status::internal("engine execution panicked")));
            drop(guard);
            // A slow peer can keep this terminal send pending after cancellation.
            // Backend lifetime must end before waiting for that response slot.
            drop(session);
            if let Err(error) = result {
                let _ = sender.send(Err(error)).await;
            }
        });
        Ok(Response::new(Box::pin(ReceiverStream::new(receiver))))
    }

    async fn analyze_plan(
        &self,
        request: Request<proto::AnalyzePlanRequest>,
    ) -> Result<Response<proto::AnalyzePlanResponse>, Status> {
        let req = request.into_inner();
        let input = req
            .analyze
            .ok_or_else(|| Status::invalid_argument("missing analysis"))?;
        let session = self.session(
            &req.session_id,
            req.user_context.as_ref(),
            req.client_observed_server_side_session_id.as_deref(),
        )?;
        let result = session
            .while_open(catch_engine_panic(async {
                session.backend.analyze(input).await
            }))
            .await?;
        if let proto::analyze_plan_response::Result::Schema(result) = &result {
            if let Some(schema) = &result.schema {
                schema::validate_schema_size(schema)?;
            }
        }
        Ok(Response::new(proto::AnalyzePlanResponse {
            session_id: req.session_id,
            server_side_session_id: session.id.clone(),
            result: Some(result),
        }))
    }

    async fn config(
        &self,
        request: Request<proto::ConfigRequest>,
    ) -> Result<Response<proto::ConfigResponse>, Status> {
        let req = request.into_inner();
        let input = req
            .operation
            .and_then(|operation| operation.op_type)
            .ok_or_else(|| Status::invalid_argument("missing config operation"))?;
        let session = self.session(
            &req.session_id,
            req.user_context.as_ref(),
            req.client_observed_server_side_session_id.as_deref(),
        )?;
        let result = session
            .while_open(catch_engine_panic(async {
                session.backend.config(input).await
            }))
            .await?;
        Ok(Response::new(proto::ConfigResponse {
            session_id: req.session_id,
            server_side_session_id: session.id.clone(),
            pairs: result.pairs,
            warnings: result.warnings,
        }))
    }

    async fn release_execute(
        &self,
        request: Request<proto::ReleaseExecuteRequest>,
    ) -> Result<Response<proto::ReleaseExecuteResponse>, Status> {
        let req = request.into_inner();
        // Spark ignores the observed incarnation here and looks up the supplied
        // operation string as-is, including IDs that do not parse as UUIDs.
        let session = self.find_session(&req.session_id, req.user_context.as_ref(), None, false)?;
        // Completed responses are not retained; release is idempotent for absent operations.
        let running = session
            .operations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&req.operation_id)
            .cloned();
        let operation_id = running.as_ref().map(|_| req.operation_id.clone());
        if matches!(
            req.release,
            Some(proto::release_execute_request::Release::ReleaseAll(_))
        ) {
            if let Some(running) = running {
                let mut done = running.done.subscribe();
                running.cancelled.send_replace(true);
                tokio::time::timeout(RELEASE_WAIT, async {
                    while !*done.borrow_and_update() {
                        if done.changed().await.is_err() {
                            break;
                        }
                    }
                })
                .await
                .map_err(|_| {
                    Status::deadline_exceeded("operation cancellation is still draining")
                })?;
            }
        }
        Ok(Response::new(proto::ReleaseExecuteResponse {
            session_id: req.session_id,
            server_side_session_id: session.id.clone(),
            operation_id,
        }))
    }

    async fn release_session(
        &self,
        request: Request<proto::ReleaseSessionRequest>,
    ) -> Result<Response<proto::ReleaseSessionResponse>, Status> {
        let req = request.into_inner();
        let key = Self::key(&req.session_id, req.user_context.as_ref());
        let session = {
            let mut sessions = self
                .sessions
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            sessions.release(key, req.allow_reconnect)
        };
        let server_id = session.as_ref().map(|s| s.id.clone()).unwrap_or_default();
        if let Some(session) = session {
            session.cancelled.send_replace(true);
        }
        Ok(Response::new(proto::ReleaseSessionResponse {
            session_id: req.session_id,
            server_side_session_id: server_id,
        }))
    }

    async fn reattach_execute(
        &self,
        _: Request<proto::ReattachExecuteRequest>,
    ) -> Result<Response<Output>, Status> {
        Err(Status::unimplemented(
            "execution reattachment is not supported",
        ))
    }
    async fn add_artifacts(
        &self,
        _: Request<tonic::Streaming<proto::AddArtifactsRequest>>,
    ) -> Result<Response<proto::AddArtifactsResponse>, Status> {
        Err(Status::unimplemented("artifact uploads are not supported"))
    }
    async fn artifact_status(
        &self,
        _: Request<proto::ArtifactStatusesRequest>,
    ) -> Result<Response<proto::ArtifactStatusesResponse>, Status> {
        Err(Status::unimplemented("artifact status is not supported"))
    }
    async fn interrupt(
        &self,
        _: Request<proto::InterruptRequest>,
    ) -> Result<Response<proto::InterruptResponse>, Status> {
        Err(Status::unimplemented(
            "operation interruption is not supported",
        ))
    }
    async fn fetch_error_details(
        &self,
        _: Request<proto::FetchErrorDetailsRequest>,
    ) -> Result<Response<proto::FetchErrorDetailsResponse>, Status> {
        Err(Status::unimplemented(
            "enriched error details are not supported",
        ))
    }
    async fn clone_session(
        &self,
        _: Request<proto::CloneSessionRequest>,
    ) -> Result<Response<proto::CloneSessionResponse>, Status> {
        Err(Status::unimplemented(
            "server-side session cloning is not supported",
        ))
    }
    async fn get_status(
        &self,
        _: Request<proto::GetStatusRequest>,
    ) -> Result<Response<proto::GetStatusResponse>, Status> {
        Err(Status::unimplemented("status queries are not supported"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CHUNK_SIZE: usize = 1024 * 1024;

    struct RegistryEngine;
    struct RegistrySession;

    impl Engine for RegistryEngine {
        fn create_session(
            &self,
            _context: SessionContext,
        ) -> Result<Arc<dyn EngineSession>, Status> {
            Ok(Arc::new(RegistrySession))
        }
    }

    #[tonic::async_trait]
    impl EngineSession for RegistrySession {
        async fn prepare(
            &self,
            _operation: proto::plan::OpType,
        ) -> Result<PreparedOperation, Status> {
            Err(Status::unimplemented(
                "registry fixture does not execute queries",
            ))
        }

        async fn analyze(
            &self,
            _input: proto::analyze_plan_request::Analyze,
        ) -> Result<proto::analyze_plan_response::Result, Status> {
            Err(Status::unimplemented(
                "registry fixture does not analyze queries",
            ))
        }

        async fn config(
            &self,
            _input: proto::config_request::operation::OpType,
        ) -> Result<crate::engine::ConfigResult, Status> {
            Err(Status::unimplemented(
                "registry fixture has no configuration",
            ))
        }
    }

    async fn release(service: &ConnectService, id: &str, allow_reconnect: bool) {
        service
            .release_session(Request::new(proto::ReleaseSessionRequest {
                session_id: id.into(),
                allow_reconnect,
                ..Default::default()
            }))
            .await
            .unwrap();
    }

    #[test]
    fn schema_response_limit_includes_protocol_envelope() {
        let session = Session::new(Arc::new(RegistrySession));
        let schema = proto::DataType {
            kind: Some(proto::data_type::Kind::Struct(proto::data_type::Struct {
                fields: vec![proto::data_type::StructField {
                    name: "x".repeat(MAX_METADATA_RESPONSE_BYTES - 64),
                    ..Default::default()
                }],
                ..Default::default()
            })),
        };
        assert!(schema.encoded_len() < MAX_METADATA_RESPONSE_BYTES);
        let mut header = response(
            &Uuid::new_v4().to_string(),
            &session,
            &Uuid::new_v4().to_string(),
            None,
        );
        header.schema = Some(schema);
        assert_eq!(
            checked_response(header).unwrap_err().code(),
            tonic::Code::ResourceExhausted
        );
    }

    #[cfg(target_pointer_width = "64")]
    #[tokio::test]
    async fn unrepresentable_arrow_row_counts_fail_before_encoding() {
        use arrow::array::NullArray;
        use arrow::datatypes::{DataType, Field, Schema};
        use arrow::record_batch::RecordBatchOptions;

        let rows = i64::MAX as usize + 1;
        let empty = RecordBatch::try_new_with_options(
            Arc::new(Schema::empty()),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(rows)),
        )
        .unwrap();
        let nulls = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("nulls", DataType::Null, true)])),
            vec![Arc::new(NullArray::new(rows))],
        )
        .unwrap();
        for batch in [empty, nulls] {
            let session = Arc::new(Session::new(Arc::new(RegistrySession)));
            let (guard, _) = session.operation("row_count").unwrap();
            let (sender, mut receiver) = mpsc::channel(2);
            let mut offset = 7;
            let mut output_schema = None;
            let error = send_batch(
                &sender,
                "session",
                "row_count",
                batch.clone().into(),
                batch.schema().as_ref(),
                &mut offset,
                &mut output_schema,
                &guard,
                Some(CHUNK_SIZE),
            )
            .await
            .unwrap_err();
            assert_eq!(error.code(), tonic::Code::ResourceExhausted);
            assert_eq!(error.message(), "result batch row count exceeds i64");
            assert_eq!(offset, 7);
            assert!(output_schema.is_none());
            assert!(receiver.try_recv().is_err());
        }
    }

    #[tokio::test]
    async fn overflowing_engine_batches_do_not_partially_emit() {
        use arrow::array::LargeStringArray;
        use arrow::datatypes::{DataType, Field, Schema};

        let schema = Arc::new(Schema::new(vec![Field::new(
            "text",
            DataType::LargeUtf8,
            false,
        )]));
        let small = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(LargeStringArray::from(vec![""]))],
        )
        .unwrap();
        let large = "z".repeat(2 * 1024 * 1024);
        let oversized = RecordBatch::try_new(
            schema.clone(),
            vec![Arc::new(LargeStringArray::from(vec![large.as_str(); 2]))],
        )
        .unwrap();
        for ipc in [false, true] {
            let session = Arc::new(Session::new(Arc::new(RegistrySession)));
            let (guard, _) = session.operation("row_offset").unwrap();
            let (sender, mut receiver) = mpsc::channel(4);
            let mut offset = i64::MAX - 2;
            let mut output_schema = None;
            send_batch(
                &sender,
                "session",
                "row_offset",
                small.clone().into(),
                schema.as_ref(),
                &mut offset,
                &mut output_schema,
                &guard,
                Some(CHUNK_SIZE),
            )
            .await
            .unwrap();
            let batch = if ipc {
                let (data, _) = encode_batch(&oversized).unwrap();
                assert!(data.len() > CHUNK_SIZE);
                ResultBatch::Ipc { data, row_count: 2 }
            } else {
                oversized.clone().into()
            };
            let error = send_batch(
                &sender,
                "session",
                "row_offset",
                batch,
                schema.as_ref(),
                &mut offset,
                &mut output_schema,
                &guard,
                Some(CHUNK_SIZE),
            )
            .await
            .unwrap_err();
            assert_eq!(error.code(), tonic::Code::ResourceExhausted);
            assert_eq!(error.message(), "result row offset exceeds i64");
            assert_eq!(offset, i64::MAX - 1);
            let previous = receiver.try_recv().unwrap().unwrap();
            let Some(proto::execute_plan_response::ResponseType::ArrowBatch(previous)) =
                previous.response_type
            else {
                panic!("expected the preceding valid Arrow response");
            };
            assert_eq!(previous.start_offset, Some(i64::MAX - 2));
            assert_eq!(previous.row_count, 1);
            assert!(receiver.try_recv().is_err());
            // A one-row batch still fits exactly at the signed boundary.
            send_batch(
                &sender,
                "session",
                "row_offset",
                small.clone().into(),
                schema.as_ref(),
                &mut offset,
                &mut output_schema,
                &guard,
                Some(CHUNK_SIZE),
            )
            .await
            .unwrap();
            assert_eq!(offset, i64::MAX);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancelled_encoding_wait_retains_operation_until_blocking_task_finishes() {
        let session = Arc::new(Session::new(Arc::new(RegistrySession)));
        let operation = Uuid::new_v4().to_string();
        let (guard, _) = session.operation(&operation).unwrap();
        let mut done = guard.running.done.subscribe();
        let encoding = EncodingBatch {
            batch: RecordBatch::new_empty(Arc::new(arrow::datatypes::Schema::empty())),
            _guard: guard.clone(),
        };
        let (started, ready) = tokio::sync::oneshot::channel();
        let (release, resume) = std::sync::mpsc::channel();
        let waiting = tokio::spawn(async move {
            tokio::task::spawn_blocking(move || {
                let _ = started.send(());
                resume
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
                encoding.encode()
            })
            .await
        });
        tokio::time::timeout(std::time::Duration::from_secs(5), ready)
            .await
            .unwrap()
            .unwrap();
        waiting.abort();
        let _ = waiting.await;
        drop(guard);
        assert!(!*done.borrow());
        assert!(session.operations.lock().unwrap().contains_key(&operation));
        release.send(()).unwrap();
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            while !*done.borrow_and_update() {
                done.changed().await.unwrap();
            }
        })
        .await
        .unwrap();
        assert!(!session.operations.lock().unwrap().contains_key(&operation));
    }

    #[tokio::test]
    async fn released_session_retention_is_bounded_without_evicting_active_sessions() {
        let service =
            ConnectService::new(Arc::new(RegistryEngine), crate::ServerOptions::default());
        let active_id = Uuid::new_v4().to_string();
        let active = service.session(&active_id, None, None).unwrap();

        let oldest = Uuid::new_v4().to_string();
        let previous = service.session(&oldest, None, None).unwrap();
        release(&service, &oldest, false).await;
        assert!(*previous.cancelled.borrow());
        release(&service, &oldest, false).await;
        release(&service, &oldest, true).await;
        assert!(matches!(service.session(&oldest, None, None),
            Err(status) if status.code() == tonic::Code::FailedPrecondition));
        {
            let registry = service.sessions.lock().unwrap();
            assert_eq!(registry.entries.len(), 2);
            assert_eq!(registry.released.len(), 1);
        }
        let mut newest = oldest.clone();
        for _ in 1..MAX_RELEASED_SESSIONS {
            newest = Uuid::new_v4().to_string();
            service.session(&newest, None, None).unwrap();
            release(&service, &newest, false).await;
        }
        release(&service, &oldest, false).await;
        assert!(matches!(
            service.session(&oldest, None, None),
            Err(status) if status.code() == tonic::Code::FailedPrecondition
        ));
        {
            let registry = service.sessions.lock().unwrap();
            assert_eq!(registry.released.len(), MAX_RELEASED_SESSIONS);
            assert_eq!(registry.entries.len(), MAX_RELEASED_SESSIONS + 1);
        }

        // Unknown releases neither allocate tombstones nor evict closed IDs.
        let unknown = Uuid::new_v4().to_string();
        release(&service, &unknown, false).await;
        release(&service, &unknown, true).await;
        assert!(matches!(service.session(&oldest, None, None),
            Err(status) if status.code() == tonic::Code::FailedPrecondition));
        service.session(&unknown, None, None).unwrap();
        release(&service, &unknown, false).await;
        assert!(service.session(&oldest, None, None).is_ok());
        assert!(matches!(
            service.session(&newest, None, None),
            Err(status) if status.code() == tonic::Code::FailedPrecondition
        ));
        assert!(Arc::ptr_eq(
            &active,
            &service.session(&active_id, None, None).unwrap()
        ));
        let registry = service.sessions.lock().unwrap();
        assert_eq!(registry.released.len(), MAX_RELEASED_SESSIONS);
        assert_eq!(registry.entries.len(), MAX_RELEASED_SESSIONS + 2);
    }

    #[tokio::test]
    async fn reconnecting_release_of_live_session_allows_a_new_backend() {
        let service =
            ConnectService::new(Arc::new(RegistryEngine), crate::ServerOptions::default());
        let id = Uuid::new_v4().to_string();
        let previous = service.session(&id, None, None).unwrap();
        release(&service, &id, true).await;
        assert!(*previous.cancelled.borrow());
        let current = service.session(&id, None, None).unwrap();
        assert!(!Arc::ptr_eq(&current, &previous));
    }
}
