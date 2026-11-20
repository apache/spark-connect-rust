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

//! Shared request envelopes and terminal-stream assertions. Individual test
//! binaries use different subsets of these helpers.

#![allow(dead_code)]

use std::time::Duration;

use proto::execute_plan_response::ResponseType;
use spark_connect_proto as proto;
use tonic::Code;
use uuid::Uuid;

pub fn user(id: &str) -> Option<proto::UserContext> {
    Some(proto::UserContext {
        user_id: id.into(),
        ..Default::default()
    })
}

pub fn relation(end: i64) -> proto::Relation {
    proto::Relation {
        rel_type: Some(proto::relation::RelType::Range(proto::Range {
            end,
            step: 1,
            ..Default::default()
        })),
        ..Default::default()
    }
}

pub fn request(session: &str, end: i64) -> proto::ExecutePlanRequest {
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

pub fn reattachable_request(session: &str, end: i64) -> proto::ExecutePlanRequest {
    let mut request = request(session, end);
    request
        .request_options
        .push(proto::execute_plan_request::RequestOption {
            request_option: Some(
                proto::execute_plan_request::request_option::RequestOption::ReattachOptions(
                    proto::ReattachOptions { reattachable: true },
                ),
            ),
        });
    request
}

pub fn config_request(
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

pub fn release(session: &str, operation: &str) -> proto::ReleaseExecuteRequest {
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

pub fn release_session(session: &str) -> proto::ReleaseSessionRequest {
    proto::ReleaseSessionRequest {
        session_id: session.into(),
        user_context: user("alice"),
        ..Default::default()
    }
}

pub async fn assert_cancelled(stream: &mut tonic::Streaming<proto::ExecutePlanResponse>) {
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
                Ok(None) => panic!("cancelled execution must report cancellation"),
            }
        }
    })
    .await
    .expect("cancelled execution must terminate");
}
