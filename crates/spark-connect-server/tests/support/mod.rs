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

//! Explicit backend outputs for frontend protocol tests.
//!
//! Range.end is only a fixture lookup key. This backend neither interprets SQL
//! nor implements Spark operators; each case supplies Arrow batches, a pending
//! stream, or a chosen failure. Real query semantics are tested against the JVM.

use std::collections::HashMap;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};

use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use futures::{Stream, StreamExt};
use spark_connect_proto as proto;
use spark_connect_server::engine::{
    BatchStream, ConfigResult, Engine, EngineSession, PreparedOperation, PreparedQuery,
    SessionContext,
};
use tonic::{Code, Status};

pub const LIVE_FIXTURE: i64 = 1_000_000_000_000;

#[derive(Clone)]
enum Ending {
    Complete,
    Pending,
    Error(Code),
}

#[derive(Clone)]
enum Output {
    Query {
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
        ending: Ending,
    },
    Rejected(Code),
}

pub struct FixtureEngine {
    outputs: HashMap<i64, Output>,
    pub streams_dropped: Arc<AtomicUsize>,
}

impl Default for FixtureEngine {
    fn default() -> Self {
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let batch = |start, end| {
            RecordBatch::try_new(
                schema.clone(),
                vec![Arc::new(Int64Array::from_iter_values(start..end))],
            )
            .unwrap()
        };
        let mut engine = Self {
            outputs: HashMap::new(),
            streams_dropped: Arc::new(AtomicUsize::new(0)),
        };
        engine.outputs.insert(
            0,
            Output::Query {
                schema: schema.clone(),
                batches: Vec::new(),
                ending: Ending::Complete,
            },
        );
        engine.outputs.insert(
            1,
            Output::Query {
                schema: schema.clone(),
                batches: vec![batch(0, 1)],
                ending: Ending::Complete,
            },
        );
        engine.outputs.insert(
            25_000,
            Output::Query {
                schema: schema.clone(),
                batches: vec![batch(0, 8192), batch(8192, 16_384), batch(16_384, 25_000)],
                ending: Ending::Complete,
            },
        );
        engine.outputs.insert(
            LIVE_FIXTURE,
            Output::Query {
                schema: schema.clone(),
                batches: vec![batch(0, 8192)],
                ending: Ending::Pending,
            },
        );
        engine
    }
}

impl FixtureEngine {
    pub fn with_batches(
        mut self,
        selector: i64,
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
    ) -> Self {
        self.outputs.insert(
            selector,
            Output::Query {
                schema,
                batches,
                ending: Ending::Complete,
            },
        );
        self
    }

    pub fn reject(mut self, selector: i64, code: Code) -> Self {
        self.outputs.insert(selector, Output::Rejected(code));
        self
    }

    pub fn stream_error(mut self, selector: i64, code: Code) -> Self {
        self.outputs.insert(
            selector,
            Output::Query {
                schema: Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)])),
                batches: Vec::new(),
                ending: Ending::Error(code),
            },
        );
        self
    }
}

impl Engine for FixtureEngine {
    fn create_session(&self, _context: SessionContext) -> Result<Arc<dyn EngineSession>, Status> {
        Ok(Arc::new(FixtureSession {
            outputs: self.outputs.clone(),
            streams_dropped: self.streams_dropped.clone(),
        }))
    }
}

struct FixtureSession {
    outputs: HashMap<i64, Output>,
    streams_dropped: Arc<AtomicUsize>,
}

#[tonic::async_trait]
impl EngineSession for FixtureSession {
    async fn prepare(&self, operation: proto::plan::OpType) -> Result<PreparedOperation, Status> {
        let proto::plan::OpType::Root(relation) = operation else {
            return Err(Status::unimplemented("fixture accepts root selectors only"));
        };
        let Some(proto::relation::RelType::Range(selector)) = relation.rel_type else {
            return Err(Status::unimplemented("fixture requires a range selector"));
        };
        match self.outputs.get(&selector.end) {
            Some(Output::Query {
                schema,
                batches,
                ending,
            }) => Ok(PreparedOperation::Query(Box::new(FixtureQuery {
                schema: schema.clone(),
                batches: batches.clone(),
                ending: ending.clone(),
                streams_dropped: self.streams_dropped.clone(),
            }))),
            Some(Output::Rejected(code)) => Err(Status::new(*code, "prepared fixture error")),
            None => Err(Status::unimplemented("unknown fixture selector")),
        }
    }

    async fn analyze(
        &self,
        _input: proto::analyze_plan_request::Analyze,
    ) -> Result<proto::analyze_plan_response::Result, Status> {
        Err(Status::unimplemented(
            "fixture does not analyze query semantics",
        ))
    }

    async fn config(
        &self,
        input: proto::config_request::operation::OpType,
    ) -> Result<ConfigResult, Status> {
        let proto::config_request::operation::OpType::GetOption(input) = input else {
            return Err(Status::unimplemented(
                "fixture supports optional config lookup only",
            ));
        };
        Ok(ConfigResult {
            pairs: input
                .keys
                .into_iter()
                .map(|key| proto::KeyValue { key, value: None })
                .collect(),
            warnings: Vec::new(),
        })
    }
}

struct FixtureQuery {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    ending: Ending,
    streams_dropped: Arc<AtomicUsize>,
}

#[tonic::async_trait]
impl PreparedQuery for FixtureQuery {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    async fn execute(self: Box<Self>) -> Result<BatchStream, Status> {
        let first = futures::stream::iter(self.batches.into_iter().map(Ok));
        let inner: BatchStream = match self.ending {
            Ending::Complete => Box::pin(first),
            Ending::Pending => Box::pin(first.chain(futures::stream::pending())),
            Ending::Error(code) => Box::pin(first.chain(futures::stream::once(async move {
                Err(Status::new(code, "streamed fixture error"))
            }))),
        };
        Ok(Box::pin(TrackedStream {
            inner,
            streams_dropped: self.streams_dropped,
        }))
    }
}

struct TrackedStream {
    inner: BatchStream,
    streams_dropped: Arc<AtomicUsize>,
}

impl Stream for TrackedStream {
    type Item = Result<RecordBatch, Status>;

    fn poll_next(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        self.inner.as_mut().poll_next(context)
    }
}

impl Drop for TrackedStream {
    fn drop(&mut self) {
        self.streams_dropped.fetch_add(1, Ordering::SeqCst);
    }
}
