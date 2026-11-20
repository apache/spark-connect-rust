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

use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use futures::Stream;
use spark_connect_proto as proto;
use spark_connect_server::engine::{
    BatchStream, ConfigResult, Engine, EngineSession, PreparedOperation, PreparedQuery,
    ResultBatch, SessionContext,
};
use tonic::Status;

#[derive(Default)]
struct Spy {
    streams_started: AtomicUsize,
    streams_dropped: AtomicUsize,
    configs_started: AtomicUsize,
    configs_dropped: AtomicUsize,
    sessions_dropped: AtomicUsize,
}

struct PendingEngine(Arc<Spy>);

impl Engine for PendingEngine {
    fn create_session(&self, _context: SessionContext) -> Result<Arc<dyn EngineSession>, Status> {
        Ok(Arc::new(PendingSession(self.0.clone())))
    }
}

struct PendingSession(Arc<Spy>);

impl Drop for PendingSession {
    fn drop(&mut self) {
        self.0.sessions_dropped.fetch_add(1, Ordering::SeqCst);
    }
}

#[tonic::async_trait]
impl EngineSession for PendingSession {
    async fn prepare(&self, _operation: proto::plan::OpType) -> Result<PreparedOperation, Status> {
        Ok(PreparedOperation::Query(Box::new(PendingQuery(
            self.0.clone(),
        ))))
    }

    async fn analyze(
        &self,
        _input: proto::analyze_plan_request::Analyze,
    ) -> Result<proto::analyze_plan_response::Result, Status> {
        Err(Status::unimplemented("shutdown fixture does not analyze"))
    }

    async fn config(
        &self,
        _input: proto::config_request::operation::OpType,
    ) -> Result<ConfigResult, Status> {
        let _call = PendingConfig(self.0.clone());
        self.0.configs_started.fetch_add(1, Ordering::SeqCst);
        std::future::pending().await
    }
}

struct PendingConfig(Arc<Spy>);

impl Drop for PendingConfig {
    fn drop(&mut self) {
        self.0.configs_dropped.fetch_add(1, Ordering::SeqCst);
    }
}

struct PendingQuery(Arc<Spy>);

#[tonic::async_trait]
impl PreparedQuery for PendingQuery {
    fn schema(&self) -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]))
    }

    async fn execute(self: Box<Self>) -> Result<BatchStream, Status> {
        self.0.streams_started.fetch_add(1, Ordering::SeqCst);
        Ok(Box::pin(PendingResults(self.0.clone())))
    }
}

struct PendingResults(Arc<Spy>);

impl Stream for PendingResults {
    type Item = Result<ResultBatch, Status>;

    fn poll_next(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        Poll::Pending
    }
}

impl Drop for PendingResults {
    fn drop(&mut self) {
        self.0.streams_dropped.fetch_add(1, Ordering::SeqCst);
    }
}

async fn wait_for_count(counter: &AtomicUsize, expected: usize) {
    tokio::time::timeout(Duration::from_secs(5), async {
        while counter.load(Ordering::SeqCst) != expected {
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .expect("backend work must finish without disconnecting its client");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn aborting_serve_cancels_backend_work_before_clients_disconnect() {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let address = listener.local_addr().unwrap();
    let spy = Arc::new(Spy::default());
    let server = tokio::spawn(spark_connect_server::serve_with_engine(
        listener,
        Arc::new(PendingEngine(spy.clone())),
        std::future::pending(),
    ));
    let mut client = proto::spark_connect_service_client::SparkConnectServiceClient::connect(
        format!("http://{address}"),
    )
    .await
    .unwrap();
    let session = uuid::Uuid::new_v4().to_string();
    let mut result = client
        .execute_plan(proto::ExecutePlanRequest {
            session_id: session.clone(),
            operation_id: Some(uuid::Uuid::new_v4().to_string()),
            plan: Some(proto::Plan {
                op_type: Some(proto::plan::OpType::Root(proto::Relation::default())),
            }),
            ..Default::default()
        })
        .await
        .unwrap()
        .into_inner();
    assert!(result.message().await.unwrap().unwrap().schema.is_some());
    wait_for_count(&spy.streams_started, 1).await;

    let mut config_client = client.clone();
    let config = tokio::spawn(async move {
        config_client
            .config(proto::ConfigRequest {
                session_id: session,
                operation: Some(proto::config_request::Operation {
                    op_type: Some(proto::config_request::operation::OpType::GetAll(
                        Default::default(),
                    )),
                }),
                ..Default::default()
            })
            .await
    });
    wait_for_count(&spy.configs_started, 1).await;
    server.abort();
    assert!(server.await.unwrap_err().is_cancelled());

    // Retain both the streamed response and its client connection: cancellation
    // must come from dropping serve_with_engine, not from a peer disconnect.
    wait_for_count(&spy.streams_dropped, 1).await;
    wait_for_count(&spy.configs_dropped, 1).await;
    wait_for_count(&spy.sessions_dropped, 1).await;
    assert!(tokio::time::timeout(Duration::from_secs(5), config)
        .await
        .unwrap()
        .unwrap()
        .is_err());
    drop(result);
    drop(client);
}
