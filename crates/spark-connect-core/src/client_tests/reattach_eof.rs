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

use super::transport_test_support::{MockServer, Reply};
use crate::RetryPolicy;
use prost::Message;
use spark_connect_proto::{
    execute_plan_response, ExecutePlanRequest, ExecutePlanResponse, ReattachExecuteRequest,
};
use tonic::Status;

fn response(id: &str, complete: bool) -> ExecutePlanResponse {
    ExecutePlanResponse {
        response_id: id.into(),
        response_type: complete.then(|| {
            execute_plan_response::ResponseType::ResultComplete(
                execute_plan_response::ResultComplete {},
            )
        }),
        ..Default::default()
    }
}

fn responses(items: &[ExecutePlanResponse]) -> Reply {
    Ok(items.iter().map(|item| Ok(item.encode_to_vec())).collect())
}

#[tokio::test]
async fn resumes_after_multiple_clean_eofs_until_result_complete() {
    let expected = vec![
        response("first", false),
        response("second", false),
        response("done", true),
    ];
    let server = MockServer::start(vec![
        responses(&expected[..1]),
        responses(&expected[1..2]),
        responses(&expected[2..]),
    ])
    .await;
    let client = server.client(Some("reader")).await;
    let request = ExecutePlanRequest {
        session_id: client.session_id().into(),
        operation_id: Some("operation".into()),
        ..Default::default()
    };
    let mut stream = client
        .execute_plan_reattachable(request.clone())
        .await
        .unwrap();
    let actual = tokio::time::timeout(std::time::Duration::from_secs(5), async {
        let mut actual = Vec::new();
        while let Some(item) = stream.message().await.unwrap() {
            actual.push(item);
        }
        actual
    })
    .await
    .unwrap();
    assert_eq!(actual, expected);
    assert!(stream.message().await.unwrap().is_none());
    let calls = server.requests();
    assert_eq!(calls.len(), 3);
    assert!(calls[0].0.ends_with("/ExecutePlan"));
    for (call, last_id) in calls[1..].iter().zip(["first", "second"]) {
        assert!(call.0.ends_with("/ReattachExecute"));
        let reattach = ReattachExecuteRequest::decode(call.1.as_slice()).unwrap();
        assert_eq!(reattach.last_response_id.as_deref(), Some(last_id));
        assert_eq!(reattach.session_id, request.session_id);
        assert_eq!(reattach.operation_id, "operation");
    }
    server.assert_finished();
}

#[tokio::test]
async fn resumes_an_empty_initial_stream_without_a_response_id() {
    let expected = response("done", true);
    let server = MockServer::start(vec![
        responses(&[]),
        responses(std::slice::from_ref(&expected)),
    ])
    .await;
    let client = server.client(None).await;
    let mut stream = client
        .execute_plan_reattachable(ExecutePlanRequest::default())
        .await
        .unwrap();
    let actual = tokio::time::timeout(std::time::Duration::from_secs(5), stream.message())
        .await
        .unwrap()
        .unwrap();
    assert_eq!(actual, Some(expected));
    assert!(stream.message().await.unwrap().is_none());
    let calls = server.requests();
    let reattach = ReattachExecuteRequest::decode(calls[1].1.as_slice()).unwrap();
    assert_eq!(reattach.last_response_id, None);
    server.assert_finished();
}

#[tokio::test]
async fn result_complete_does_not_reattach() {
    let expected = response("done", true);
    let server = MockServer::start(vec![responses(std::slice::from_ref(&expected))]).await;
    let client = server.client(None).await;
    let mut stream = client
        .execute_plan_reattachable(ExecutePlanRequest::default())
        .await
        .unwrap();
    assert_eq!(stream.message().await.unwrap(), Some(expected));
    assert!(stream.message().await.unwrap().is_none());
    assert_eq!(server.requests().len(), 1);
    server.assert_finished();
}

#[tokio::test]
async fn reattach_retries_transient_rpc_failures_after_clean_eof() {
    let expected = response("done", true);
    let server = MockServer::start(vec![
        responses(&[]),
        Err(Status::unavailable("retry")),
        responses(std::slice::from_ref(&expected)),
    ])
    .await;
    let mut client = server.client(None).await;
    client.set_retry_policy(RetryPolicy {
        initial_backoff_ms: 0,
        ..Default::default()
    });
    let mut stream = client
        .execute_plan_reattachable(ExecutePlanRequest::default())
        .await
        .unwrap();
    assert_eq!(
        tokio::time::timeout(std::time::Duration::from_secs(5), stream.message())
            .await
            .unwrap()
            .unwrap(),
        Some(expected)
    );
    assert_eq!(server.requests().len(), 3);
    server.assert_finished();
}

#[tokio::test]
async fn non_retryable_stream_error_is_returned_without_reattaching() {
    let server = MockServer::start(vec![Ok(vec![Err(Status::invalid_argument("bad plan"))])]).await;
    let client = server.client(None).await;
    let mut stream = client
        .execute_plan_reattachable(ExecutePlanRequest::default())
        .await
        .unwrap();
    assert!(stream
        .message()
        .await
        .unwrap_err()
        .to_string()
        .contains("bad plan"));
    assert_eq!(server.requests().len(), 1);
    server.assert_finished();
}

#[tokio::test]
async fn transient_stream_error_still_reattaches() {
    let expected = response("done", true);
    let server = MockServer::start(vec![
        Ok(vec![Err(Status::unavailable("disconnected"))]),
        responses(std::slice::from_ref(&expected)),
    ])
    .await;
    let client = server.client(None).await;
    let mut stream = client
        .execute_plan_reattachable(ExecutePlanRequest::default())
        .await
        .unwrap();
    assert_eq!(stream.message().await.unwrap(), Some(expected));
    assert_eq!(server.requests().len(), 2);
    server.assert_finished();
}

#[tokio::test]
async fn non_retryable_reattach_error_after_clean_eof_is_returned() {
    let server = MockServer::start(vec![
        responses(&[]),
        Err(Status::not_found("operation expired")),
    ])
    .await;
    let client = server.client(None).await;
    let mut stream = client
        .execute_plan_reattachable(ExecutePlanRequest::default())
        .await
        .unwrap();
    assert!(stream
        .message()
        .await
        .unwrap_err()
        .to_string()
        .contains("operation expired"));
    assert_eq!(server.requests().len(), 2);
    server.assert_finished();
}

#[tokio::test]
async fn reattach_retry_budget_after_clean_eof_is_respected() {
    let server = MockServer::start(vec![
        responses(&[]),
        Err(Status::unavailable("first")),
        Err(Status::unavailable("last")),
    ])
    .await;
    let mut client = server.client(None).await;
    client.set_retry_policy(RetryPolicy {
        max_retries: Some(1),
        initial_backoff_ms: 0,
        ..Default::default()
    });
    let mut stream = client
        .execute_plan_reattachable(ExecutePlanRequest::default())
        .await
        .unwrap();
    assert!(stream
        .message()
        .await
        .unwrap_err()
        .to_string()
        .contains("last"));
    assert_eq!(server.requests().len(), 3);
    server.assert_finished();
}
