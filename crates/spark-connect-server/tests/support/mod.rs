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

//! Explicit backend outputs for frontend protocol and schema tests.
//!
//! Range.end is only a fixture lookup key. This backend neither interprets SQL
//! nor implements Spark operators; each case supplies batches or a chosen error.
//! Test binaries use different subsets of the fixture constructors.

#![allow(dead_code)]

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::Int64Array;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use spark_connect_proto as proto;
use spark_connect_server::engine::{
    BatchStream, ConfigResult, Engine, EngineSession, PreparedOperation, PreparedQuery,
    ResultBatch, SessionContext,
};
use tonic::{Code, Status};

#[derive(Clone)]
pub enum Batch {
    Arrow(RecordBatch),
    Ipc(Vec<u8>, i64),
}

#[derive(Clone)]
enum Output {
    Query {
        schema: SchemaRef,
        batches: Vec<Batch>,
        error: Option<Code>,
    },
    Rejected(Code),
}

#[derive(Clone)]
pub struct FixtureEngine {
    outputs: HashMap<i64, Output>,
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
        Self {
            outputs: HashMap::new(),
        }
        .with_batches(0, schema.clone(), Vec::new())
        .with_batches(1, schema.clone(), vec![batch(0, 1)])
        .with_batches(
            25_000,
            schema.clone(),
            vec![batch(0, 8192), batch(8192, 16_384), batch(16_384, 25_000)],
        )
    }
}

impl FixtureEngine {
    pub fn with_batches(self, selector: i64, schema: SchemaRef, batches: Vec<RecordBatch>) -> Self {
        self.with_results(
            selector,
            schema,
            batches.into_iter().map(Batch::Arrow).collect(),
        )
    }

    pub fn with_results(mut self, selector: i64, schema: SchemaRef, batches: Vec<Batch>) -> Self {
        self.outputs.insert(
            selector,
            Output::Query {
                schema,
                batches,
                error: None,
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
                error: Some(code),
            },
        );
        self
    }
}

impl Engine for FixtureEngine {
    fn create_session(&self, _: SessionContext) -> Result<Arc<dyn EngineSession>, Status> {
        Ok(Arc::new(self.clone()))
    }
}

#[tonic::async_trait]
impl EngineSession for FixtureEngine {
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
                error,
            }) => Ok(PreparedOperation::Query(Box::new(FixtureQuery {
                schema: schema.clone(),
                batches: batches.clone(),
                error: *error,
            }))),
            Some(Output::Rejected(code)) => Err(Status::new(*code, "prepared fixture error")),
            None => Err(Status::unimplemented("unknown fixture selector")),
        }
    }

    async fn analyze(
        &self,
        _: proto::analyze_plan_request::Analyze,
    ) -> Result<proto::analyze_plan_response::Result, Status> {
        Err(Status::unimplemented(
            "fixture does not analyze query semantics",
        ))
    }

    async fn config(
        &self,
        _: proto::config_request::operation::OpType,
    ) -> Result<ConfigResult, Status> {
        Err(Status::unimplemented(
            "fixture does not configure query semantics",
        ))
    }
}

struct FixtureQuery {
    schema: SchemaRef,
    batches: Vec<Batch>,
    error: Option<Code>,
}

#[tonic::async_trait]
impl PreparedQuery for FixtureQuery {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    async fn execute(self: Box<Self>) -> Result<BatchStream, Status> {
        let batches = self.batches.into_iter().map(|batch| {
            Ok(match batch {
                Batch::Arrow(batch) => ResultBatch::Arrow(batch),
                Batch::Ipc(data, row_count) => ResultBatch::Ipc { data, row_count },
            })
        });
        let failure = self
            .error
            .into_iter()
            .map(|code| Err(Status::new(code, "streamed fixture error")));
        Ok(Box::pin(futures::stream::iter(batches.chain(failure))))
    }
}
