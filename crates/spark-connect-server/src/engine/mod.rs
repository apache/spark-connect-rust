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
//! an initial in-process contract for an injected execution engine.

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
    /// Allocate local session state without blocking, performing I/O or panicking.
    /// Defer asynchronous initialization to the session methods. The frontend
    /// serializes creation to ensure one backend session per active incarnation.
    /// A reconnecting session release permits a replacement with the same
    /// `session_id` while references to the previous incarnation are still
    /// draining. The engine must keep those incarnations' resources independent.
    fn create_session(&self, context: SessionContext) -> Result<Arc<dyn EngineSession>, Status>;
}

/// Backend-owned catalog, configuration and planning state.
///
/// Dropping a method future must cancel its work. Dropping the final session
/// reference releases its resources after in-flight operations have finished.
/// Resource cleanup must not panic.
/// With unwinding enabled, panics in these methods become `INTERNAL` errors;
/// the frontend keeps the session available for later requests. Implementations
/// must leave their state usable when unwinding.
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
    /// The frontend adds envelope IDs and, for requests that opt into
    /// reattachment, the final `ResultComplete` message.
    /// Eager command effects may already be committed when later frontend
    /// response validation fails, for example when a payload exceeds the size
    /// limit. Such a failure does not roll back effects or make retries safe.
    Command(Vec<proto::execute_plan_response::ResponseType>),
}

/// Query execution starts after the frontend validates and sends its schema.
// async_trait repeats Future's must_use annotation on generated trait methods.
#[allow(clippy::double_must_use)]
#[tonic::async_trait]
pub trait PreparedQuery: Send {
    /// The declared Spark schema. This synchronous accessor must not panic.
    /// Batches must have the same field names and Spark-visible types;
    /// equivalent Arrow representations are accepted.
    /// Batch nullability and metadata may differ from this declaration, but
    /// emitted IPC batches in one result must share physical types, field names
    /// and nullability. Metadata may differ between batches. Record-batch encoding
    /// materializes view arrays to large-offset types; existing IPC retains its
    /// physical types. Mixing incompatible output representations fails before
    /// the changed output schema is sent. Declared fields must not contain
    /// metadata, which Spark schema conversion rejects.
    fn schema(&self) -> SchemaRef;

    /// Dropping this future or its returned stream must cancel backend work;
    /// implementations must not leave detached execution tasks running.
    async fn execute(self: Box<Self>) -> Result<BatchStream, Status>;
}

/// Engine output, optionally already serialized for Spark Connect.
///
/// Engines must bound batch production. Optional byte chunking bounds network
/// payloads, not the complete batch's allocation. When chunking is disabled or
/// the client does not opt in, each batch must fit the client's gRPC receive limit.
pub enum ResultBatch {
    /// Encoded once as a complete IPC stream before optional byte chunking.
    Arrow(RecordBatch),
    /// A complete IPC stream containing its schema, exactly one record batch,
    /// and EOS. The producer must supply that batch's actual nonnegative row count.
    /// The frontend validates the schema and trusts the producer's batch contents
    /// and declared row count. IPC bytes, including compression, are preserved.
    /// Clients that opt in receive byte chunks for reassembly before decoding;
    /// otherwise the whole stream is sent, so producers must respect the client's
    /// gRPC receive limit when selecting batch sizes.
    Ipc { data: Vec<u8>, row_count: i64 },
}

impl From<RecordBatch> for ResultBatch {
    fn from(batch: RecordBatch) -> Self {
        Self::Arrow(batch)
    }
}

/// Pull-based results: the frontend polls only as client backpressure allows.
pub type BatchStream = Pin<Box<dyn Stream<Item = Result<ResultBatch, Status>> + Send>>;

/// Engine configuration output without frontend-owned protocol IDs.
#[derive(Default)]
pub struct ConfigResult {
    pub pairs: Vec<proto::KeyValue>,
    pub warnings: Vec<String>,
}
