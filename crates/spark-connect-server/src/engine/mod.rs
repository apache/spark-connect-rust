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

//! Native engine boundary for Spark Connect planning and execution.
//!
//! The frontend owns protocol IDs, operation lifetime, framing and backpressure.
//! Engines own their catalog, configuration and native plans. This Rust API is
//! an initial in-process contract for the JNI Spark engine.

use std::pin::Pin;
use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use futures::Stream;
use spark_connect_proto as proto;
use tonic::Status;

/// Convert an engine Arrow schema into the Spark Connect schema representation.
pub use crate::schema::spark_schema;

/// Protocol context for a new backend session.
#[derive(Clone, Debug)]
pub struct SessionContext {
    pub session_id: String,
    /// Client-supplied metadata, not an authenticated identity.
    pub user_context: Option<proto::UserContext>,
}

/// Factory for sessions bound to one native engine.
pub trait Engine: Send + Sync {
    /// Allocate local session state without blocking or performing I/O.
    /// Defer asynchronous initialization to the session methods. The frontend
    /// serializes creation to ensure one backend session per protocol session.
    fn create_session(&self, context: SessionContext) -> Result<Arc<dyn EngineSession>, Status>;
}

/// Backend-owned catalog, configuration and planning state.
///
/// Dropping a method future must cancel its work. Dropping the final session
/// reference releases its resources after in-flight operations have finished.
// async_trait repeats Future's must_use annotation on generated trait methods.
#[allow(clippy::double_must_use)]
#[tonic::async_trait]
pub trait EngineSession: Send + Sync {
    /// Prepare a query, or perform an eager protocol command. The frontend
    /// reserves the operation ID before calling this method.
    /// If cancelled, a command must leave its state change fully committed or
    /// rolled back, with no partially applied changes.
    async fn prepare(&self, operation: proto::plan::OpType) -> Result<PreparedOperation, Status>;

    async fn analyze(
        &self,
        input: proto::analyze_plan_request::Analyze,
    ) -> Result<proto::analyze_plan_response::Result, Status>;

    async fn config(
        &self,
        input: proto::config_request::operation::OpType,
    ) -> Result<ConfigResult, Status>;
}

/// Prepared native query or command-specific response payloads.
pub enum PreparedOperation {
    Query(Box<dyn PreparedQuery>),
    /// The frontend adds envelope IDs and the final ResultComplete message.
    Command(Vec<proto::execute_plan_response::ResponseType>),
}

/// Query execution starts after the frontend validates and sends its schema.
// async_trait repeats Future's must_use annotation on generated trait methods.
#[allow(clippy::double_must_use)]
#[tonic::async_trait]
pub trait PreparedQuery: Send {
    /// The declared Spark schema. Batches must have the same field names and
    /// Spark-visible types; equivalent Arrow representations are accepted.
    /// Batch nullability and metadata may differ and remain in each IPC batch,
    /// while the protocol schema header uses this declaration. The declared
    /// fields must not contain metadata, which Spark schema conversion rejects.
    fn schema(&self) -> SchemaRef;

    /// Dropping this future or its returned stream must cancel backend work;
    /// implementations must not leave detached execution tasks running.
    async fn execute(self: Box<Self>) -> Result<BatchStream, Status>;
}

/// Pull-based results: the frontend polls only as client backpressure allows.
pub type BatchStream = Pin<Box<dyn Stream<Item = Result<RecordBatch, Status>> + Send>>;

/// Engine configuration output without frontend-owned protocol IDs.
#[derive(Default)]
pub struct ConfigResult {
    pub pairs: Vec<proto::KeyValue>,
    pub warnings: Vec<String>,
}
