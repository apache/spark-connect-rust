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

//! The hook of the executions of a session, as PySpark's `SparkSession.Hook.on_execute_plan`:
//! called with each `ExecutePlanRequest` before it is sent, of whose request the execution is.
//! These tests run when SPARK_REMOTE is set; the ones without a server are in `session.rs`.

use std::sync::{Arc, Mutex};

use spark_connect::{ExecutePlanHook, SparkSession};
use spark_connect_proto as proto;

fn live() -> Option<SparkSession> {
    let remote = std::env::var("SPARK_REMOTE").ok()?;
    Some(
        SparkSession::builder()
            .remote(&remote)
            .get_or_create()
            .unwrap(),
    )
}

#[test]
fn the_request_of_a_hook_is_sent_with_the_operation_id_of_the_original() {
    let Some(session) = live() else {
        return;
    };
    let replacement = session.range(5).unwrap().to_relation().unwrap();
    let operation_ids: Arc<Mutex<Vec<Option<String>>>> = Arc::default();
    let seen = Arc::clone(&operation_ids);
    let hook: ExecutePlanHook = Arc::new(move |mut request| {
        seen.lock().unwrap().push(request.operation_id.clone());
        request.operation_id = Some("replaced".to_string());
        if let Some(proto::plan::OpType::Root(_)) =
            request.plan.as_ref().and_then(|p| p.op_type.as_ref())
        {
            request.plan = Some(proto::Plan {
                op_type: Some(proto::plan::OpType::Root(replacement.clone())),
            });
        }
        Ok(request)
    });
    session.set_execute_plan_hook(Some(hook));
    let rows = session.range(1).unwrap().collect().unwrap();
    session.set_execute_plan_hook(None);
    assert_eq!(rows.len(), 5);
    let operation_ids = operation_ids.lock().unwrap();
    assert_eq!(operation_ids.len(), 1);
    assert!(operation_ids[0]
        .as_deref()
        .is_some_and(|id| id != "replaced"));
}
