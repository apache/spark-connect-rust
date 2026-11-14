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
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use arrow::ipc::writer::StreamWriter;
use arrow::record_batch::RecordBatch;
use futures::{Stream, StreamExt};
use spark_connect_proto as proto;
use tokio::sync::{mpsc, watch};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status};
use uuid::Uuid;

use crate::engine::{Engine, EngineSession, PreparedOperation, SessionContext};
use crate::grpc::spark_connect_service_server::SparkConnectService;
use crate::schema::{self, result_schema};

type Output = Pin<Box<dyn Stream<Item = Result<proto::ExecutePlanResponse, Status>> + Send>>;
type SessionKey = (String, String);
const MAX_ARROW_IPC_BYTES: usize = crate::schema::MAX_RESULT_BYTES;
const MAX_RELEASED_SESSIONS: usize = 4096;

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
    ) -> Result<(OperationGuard, watch::Receiver<bool>), Status> {
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
            OperationGuard {
                session: self.clone(),
                id: id.to_owned(),
                running,
            },
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
    entries: HashMap<SessionKey, Option<Arc<Session>>>,
    // Remember the most recent distinct closed IDs in release order. Repeated
    // releases do not refresh retention. After eviction, an old ID may create a
    // fresh session; active sessions never count toward this finite budget.
    released: VecDeque<SessionKey>,
}

impl SessionRegistry {
    fn release(&mut self, key: SessionKey, allow_reconnect: bool) -> Option<Arc<Session>> {
        if allow_reconnect {
            self.released.retain(|closed| closed != &key);
            return self.entries.remove(&key).flatten();
        }
        let previous = self.entries.insert(key.clone(), None);
        if !matches!(&previous, Some(None)) {
            self.released.push_back(key);
            if self.released.len() > MAX_RELEASED_SESSIONS {
                if let Some(oldest) = self.released.pop_front() {
                    self.entries.remove(&oldest);
                }
            }
        }
        previous.flatten()
    }
}

// None is a recently released-session tombstone, retained by SessionRegistry.
pub(crate) struct ConnectService {
    engine: Arc<dyn Engine>,
    sessions: Mutex<SessionRegistry>,
    closing: std::sync::atomic::AtomicBool,
}

impl ConnectService {
    pub(crate) fn new(engine: Arc<dyn Engine>) -> Self {
        Self {
            engine,
            sessions: Mutex::new(SessionRegistry::default()),
            closing: std::sync::atomic::AtomicBool::new(false),
        }
    }

    fn key(id: &str, user: Option<&proto::UserContext>) -> Result<SessionKey, Status> {
        Uuid::parse_str(id).map_err(|_| Status::invalid_argument("session_id must be a UUID"))?;
        Ok((
            user.map(|u| u.user_id.clone()).unwrap_or_default(),
            id.into(),
        ))
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
        let key = Self::key(id, user)?;
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

fn encode_batch(batch: &RecordBatch) -> Result<Vec<u8>, Status> {
    use arrow::datatypes::{DataType, FieldRef, Schema};
    use arrow::record_batch::RecordBatchOptions;

    fn materialized_field(field: &FieldRef) -> FieldRef {
        let data_type = materialized_type(field.data_type());
        if data_type == *field.data_type() {
            field.clone()
        } else {
            Arc::new(field.as_ref().clone().with_data_type(data_type))
        }
    }

    fn materialized_type(data_type: &DataType) -> DataType {
        match data_type {
            DataType::Utf8View => DataType::LargeUtf8,
            DataType::BinaryView => DataType::LargeBinary,
            DataType::List(field) => DataType::List(materialized_field(field)),
            DataType::LargeList(field) => DataType::LargeList(materialized_field(field)),
            DataType::Struct(fields) => {
                DataType::Struct(fields.iter().map(materialized_field).collect())
            }
            DataType::Map(field, sorted) => DataType::Map(materialized_field(field), *sorted),
            other => other.clone(),
        }
    }

    // IPC writes all backing buffers of view arrays even after a row slice. Materialize
    // their bytes, including nested children, so splitting measures only retained rows.
    // Use 64-bit offsets so materializing a batch cannot overflow 32-bit byte offsets
    // before the result-size check has a chance to split it.
    let source_schema = batch.schema();
    let fields: Vec<_> = source_schema
        .fields()
        .iter()
        .map(materialized_field)
        .collect();
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
    Ok(bytes)
}

async fn send_batch(
    sender: &mpsc::Sender<Result<proto::ExecutePlanResponse, Status>>,
    id: &str,
    session: &Session,
    operation: &str,
    batch: RecordBatch,
    offset: &mut i64,
) -> Result<bool, Status> {
    // Measure encoded IPC, since memory-size estimates miss schema/dictionary overhead.
    let mut pending = vec![batch];
    while let Some(batch) = pending.pop() {
        let data = encode_batch(&batch)?;
        if data.len() > MAX_ARROW_IPC_BYTES {
            if batch.num_rows() <= 1 {
                return Err(Status::resource_exhausted(
                    "one Arrow row/schema exceeds the 3 MiB result limit",
                ));
            }
            // Estimate a row budget with headroom instead of repeatedly halving a
            // wide batch. Recheck every slice: skew and dictionary/schema overhead
            // can make its actual IPC size larger than this estimate.
            let rows_per_batch = ((batch.num_rows() as u128 * MAX_ARROW_IPC_BYTES as u128 * 9
                / (data.len() as u128 * 10)) as usize)
                .clamp(1, batch.num_rows() - 1);
            drop(data);
            for start in (0..batch.num_rows()).step_by(rows_per_batch).rev() {
                pending.push(batch.slice(start, rows_per_batch.min(batch.num_rows() - start)));
            }
            continue;
        }
        let result = response(
            id,
            session,
            operation,
            Some(proto::execute_plan_response::ResponseType::ArrowBatch(
                proto::execute_plan_response::ArrowBatch {
                    row_count: batch.num_rows() as i64,
                    data: data.into(),
                    start_offset: Some(*offset),
                    ..Default::default()
                },
            )),
        );
        *offset = offset
            .checked_add(batch.num_rows() as i64)
            .ok_or_else(|| Status::resource_exhausted("result row offset exceeds i64"))?;
        if sender.send(Ok(result)).await.is_err() {
            return Ok(false);
        }
    }
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
        let operation = req
            .operation_id
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        Uuid::parse_str(&operation)
            .map_err(|_| Status::invalid_argument("operation_id must be a UUID"))?;
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
            prepared = session.backend.prepare(op) => prepared?,
        };
        let query = match prepared {
            PreparedOperation::Query(query) => query,
            PreparedOperation::Command(payloads) => {
                let mut messages = payloads
                    .into_iter()
                    .map(|payload| {
                        Ok(response(
                            &req.session_id,
                            &session,
                            &operation,
                            Some(payload),
                        ))
                    })
                    .collect::<Vec<_>>();
                messages.push(Ok(complete(&req.session_id, &session, &operation)));
                return Ok(Response::new(Box::pin(tokio_stream::iter(messages))));
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
                if sender.send(Ok(header)).await.is_err() {
                    return Ok(());
                }
                let mut batches = query.execute().await?;
                let mut offset = 0;
                let mut sent = false;
                while let Some(batch) = batches.next().await {
                    let batch = batch?;
                    let actual_schema = schema::comparable_schema(batch.schema().as_ref())
                        .map_err(|error| {
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
                        &session,
                        &operation,
                        batch,
                        &mut offset,
                    )
                    .await?
                    {
                        return Ok(());
                    }
                    sent = true;
                }
                if !sent {
                    let batch = RecordBatch::new_empty(arrow_schema);
                    if !send_batch(
                        &sender,
                        &req.session_id,
                        &session,
                        &operation,
                        batch,
                        &mut offset,
                    )
                    .await?
                    {
                        return Ok(());
                    }
                }
                let _ = sender
                    .send(Ok(complete(&req.session_id, &session, &operation)))
                    .await;
                Ok::<_, Status>(())
            };
            let result = tokio::select! {
                biased;
                _ = cancelled.changed() => Err(Status::cancelled("session released")),
                _ = operation_cancelled.changed() => Err(Status::cancelled("operation released")),
                _ = sender.closed() => Ok(()),
                result = run => result,
            };
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
        let result = session.while_open(session.backend.analyze(input)).await?;
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
        let result = session.while_open(session.backend.config(input)).await?;
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
        Uuid::parse_str(&req.operation_id)
            .map_err(|_| Status::invalid_argument("operation_id must be a UUID"))?;
        let session = self.find_session(
            &req.session_id,
            req.user_context.as_ref(),
            req.client_observed_server_side_session_id.as_deref(),
            false,
        )?;
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
                while !*done.borrow_and_update() {
                    if done.changed().await.is_err() {
                        break;
                    }
                }
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
        let key = Self::key(&req.session_id, req.user_context.as_ref())?;
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

    #[tokio::test]
    async fn released_session_retention_is_bounded_without_evicting_active_sessions() {
        let service = ConnectService::new(Arc::new(RegistryEngine));
        let active_id = Uuid::new_v4().to_string();
        let previous = service.session(&active_id, None, None).unwrap();
        release(&service, &active_id, false).await;
        assert!(*previous.cancelled.borrow());
        release(&service, &active_id, true).await;
        let active = service.session(&active_id, None, None).unwrap();
        assert!(!Arc::ptr_eq(&active, &previous));

        // Unknown IDs are remembered too, but repeated release adds no entries.
        let oldest = Uuid::new_v4().to_string();
        release(&service, &oldest, false).await;
        release(&service, &oldest, false).await;
        {
            let registry = service.sessions.lock().unwrap();
            assert_eq!(registry.entries.len(), 2);
            assert_eq!(registry.released.len(), 1);
        }
        let mut newest = oldest.clone();
        for _ in 1..MAX_RELEASED_SESSIONS {
            newest = Uuid::new_v4().to_string();
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

        release(&service, &Uuid::new_v4().to_string(), false).await;
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
}
