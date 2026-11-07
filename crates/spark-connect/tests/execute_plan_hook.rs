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
//! The tests of a server run when SPARK_REMOTE is set.

use std::sync::{Arc, Mutex};

use spark_connect::{ExecutePlanHook, SparkError, SparkSession};
use spark_connect_proto as proto;

fn offline() -> SparkSession {
    // The channel connects lazily: the hook runs before a request is sent.
    SparkSession::builder()
        .remote("sc://127.0.0.1:1")
        .get_or_create()
        .unwrap()
}

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
fn the_error_of_a_hook_is_the_error_of_the_execution() {
    let session = offline();
    let seen: Arc<Mutex<Vec<proto::ExecutePlanRequest>>> = Arc::default();
    let requests = Arc::clone(&seen);
    let hook: ExecutePlanHook = Arc::new(move |request| {
        requests.lock().unwrap().push(request);
        Err(SparkError::value_msg("the hook failed"))
    });
    session.set_execute_plan_hook(Some(hook));
    let error = session.range(3).unwrap().collect().unwrap_err();
    assert!(error.to_string().contains("the hook failed"), "{error}");
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert_eq!(seen[0].session_id, session.session_id());
    assert!(seen[0].operation_id.is_some());
    assert!(seen[0].plan.is_some());
}

#[test]
fn a_cleared_or_replaced_hook_is_not_called() {
    let session = offline();
    let calls = Arc::new(Mutex::new(0));
    let counted = Arc::clone(&calls);
    let first: ExecutePlanHook = Arc::new(move |_| {
        *counted.lock().unwrap() += 1;
        Err(SparkError::value_msg("the first hook"))
    });
    session.set_execute_plan_hook(Some(first));
    session.set_execute_plan_hook(None);
    let second: ExecutePlanHook = Arc::new(|_| Err(SparkError::value_msg("the second hook")));
    session.set_execute_plan_hook(Some(second));
    let error = session.range(1).unwrap().collect().unwrap_err();
    assert!(error.to_string().contains("the second hook"), "{error}");
    assert_eq!(*calls.lock().unwrap(), 0);
}

#[test]
fn the_request_of_a_hook_is_sent_with_the_operation_id_of_the_original() {
    let Some(session) = live() else {
        return;
    };
    let replacement = session.range(5).unwrap().to_relation();
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
