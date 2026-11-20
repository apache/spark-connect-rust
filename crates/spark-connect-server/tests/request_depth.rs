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

//! The frontend decodes its requests as prost does: the limit of the nesting of the messages that
//! the generated bindings share with the client (`spark_connect_proto::ClientCodec`) is of the
//! responses that the client decodes, not of the requests that the server receives.

#[path = "support/requests.rs"]
mod requests;
#[path = "support/server.rs"]
mod server;

use std::sync::Arc;

use requests::{relation, user};
use server::Server;
use spark_connect_proto as proto;
use spark_connect_server::engine::{
    ConfigResult, Engine, EngineSession, PreparedOperation, SessionContext,
};
use tonic::Status;
use uuid::Uuid;

/// An engine whose analysis of the tree string of a plan gives the number of its limits.
struct LimitsEngine;

impl Engine for LimitsEngine {
    fn create_session(&self, _context: SessionContext) -> Result<Arc<dyn EngineSession>, Status> {
        Ok(Arc::new(LimitsSession))
    }
}

struct LimitsSession;

#[tonic::async_trait]
impl EngineSession for LimitsSession {
    async fn prepare(&self, _operation: proto::plan::OpType) -> Result<PreparedOperation, Status> {
        Err(Status::unimplemented("the fixture only analyzes"))
    }

    async fn analyze(
        &self,
        input: proto::analyze_plan_request::Analyze,
    ) -> Result<proto::analyze_plan_response::Result, Status> {
        let proto::analyze_plan_request::Analyze::TreeString(tree) = input else {
            return Err(Status::unimplemented("the fixture only gives tree strings"));
        };
        let Some(proto::plan::OpType::Root(mut root)) = tree.plan.and_then(|plan| plan.op_type)
        else {
            return Err(Status::invalid_argument("no relation"));
        };
        let mut limits = 0;
        while let Some(proto::relation::RelType::Limit(limit)) = root.rel_type {
            root = *limit
                .input
                .ok_or_else(|| Status::invalid_argument("no input"))?;
            limits += 1;
        }
        Ok(proto::analyze_plan_response::Result::TreeString(
            proto::analyze_plan_response::TreeString {
                tree_string: limits.to_string(),
            },
        ))
    }

    async fn config(
        &self,
        _input: proto::config_request::operation::OpType,
    ) -> Result<ConfigResult, Status> {
        Err(Status::unimplemented("the fixture only analyzes"))
    }
}

/// An analysis of the tree string of a plan of `limits` limits of a range: AnalyzePlanRequest (0)
/// -> TreeString (1) -> Plan (2) -> Relation (3) -> Limit (4) -> Relation (5) ..., of the Range of
/// the innermost Relation at level `2 * limits + 4`.
fn tree_string_of_limits(limits: usize) -> proto::AnalyzePlanRequest {
    let mut plan = relation(1);
    for _ in 0..limits {
        plan = proto::Relation {
            rel_type: Some(proto::relation::RelType::Limit(Box::new(proto::Limit {
                input: Some(Box::new(plan)),
                limit: 1,
            }))),
            ..Default::default()
        };
    }
    proto::AnalyzePlanRequest {
        session_id: Uuid::new_v4().to_string(),
        user_context: user("alice"),
        analyze: Some(proto::analyze_plan_request::Analyze::TreeString(
            proto::analyze_plan_request::TreeString {
                plan: Some(proto::Plan {
                    op_type: Some(proto::plan::OpType::Root(plan)),
                }),
                level: None,
            },
        )),
        ..Default::default()
    }
}

#[tokio::test]
async fn a_request_is_decoded_as_prost_decodes_it() {
    // The bindings share the codec of the client, whose limit of 100 levels is of the responses
    // that the client decodes: the frontend decodes its requests as prost does, within prost's
    // limit in a build of prost's default features (as of this crate and of the JNI library), and
    // of any depth in a build that lifts it (of prost's `no-recursion-limit`, as one of the Python
    // binding). So a request of 40 limits (of 84 levels) reaches the engine, and one of 60 (of 124
    // levels, past the limit of the responses) reaches it if prost decodes it, else fails of
    // prost's error.
    use prost::Message;

    let mut server = Server::start_with_engine(Arc::new(LimitsEngine)).await;
    for limits in [40, 60] {
        let request = tree_string_of_limits(limits);
        let decoded = proto::AnalyzePlanRequest::decode(request.encode_to_vec().as_slice());
        let analyzed = server.client.analyze_plan(request).await;
        match decoded {
            Ok(_) => assert!(matches!(
                analyzed.unwrap().into_inner().result,
                Some(proto::analyze_plan_response::Result::TreeString(tree))
                    if tree.tree_string == limits.to_string()
            )),
            Err(error) => {
                assert!(limits > 48, "{limits} limits are within prost's limit");
                let status = analyzed.unwrap_err();
                assert_eq!(status.code(), tonic::Code::Internal);
                assert_eq!(status.message(), error.to_string());
            }
        }
    }
    server.stop().await;
}
