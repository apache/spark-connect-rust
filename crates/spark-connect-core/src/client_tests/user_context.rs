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
use super::{ChannelBuilder, ExecutePlanResponseReattachableIterator, SparkConnectClient};
use crate::CHUNK_SIZE;
use prost::Message;
use spark_connect_proto::{
    execute_plan_response, AddArtifactsRequest, AddArtifactsResponse, AnalyzePlanRequest,
    AnalyzePlanResponse, ExecutePlanRequest, ExecutePlanResponse, ReattachExecuteRequest,
    UserContext,
};
use std::io::Write;
use tonic::Status;

fn reply(message: &impl Message) -> Reply {
    Ok(vec![Ok(message.encode_to_vec())])
}

fn completed() -> ExecutePlanResponse {
    ExecutePlanResponse {
        response_id: "done".into(),
        response_type: Some(execute_plan_response::ResponseType::ResultComplete(
            execute_plan_response::ResultComplete {},
        )),
        ..Default::default()
    }
}

#[tokio::test]
async fn typed_plan_requests_fill_missing_identity_and_preserve_explicit_context() {
    let extension = prost_types::Any {
        type_url: "example.org/context".into(),
        value: vec![1, 2, 3],
    };
    for context in [
        None,
        Some(UserContext::default()),
        Some(UserContext {
            user_name: "name".into(),
            extensions: vec![extension.clone()],
            ..Default::default()
        }),
        Some(UserContext {
            user_id: "explicit".into(),
            user_name: "name".into(),
            extensions: vec![extension.clone()],
        }),
    ] {
        let server = MockServer::start(vec![
            reply(&completed()),
            reply(&AnalyzePlanResponse::default()),
            reply(&completed()),
        ])
        .await;
        let client = server.client(Some("configured")).await;
        let mut expected = context.clone().unwrap_or_default();
        if expected.user_id.is_empty() {
            expected.user_id = "configured".into();
        }
        let request = ExecutePlanRequest {
            user_context: context.clone(),
            ..Default::default()
        };
        client
            .execute_plan(request.clone())
            .await
            .unwrap()
            .message()
            .await
            .unwrap();
        client
            .analyze_plan(AnalyzePlanRequest {
                user_context: context,
                ..Default::default()
            })
            .await
            .unwrap();
        client
            .execute_plan_reattachable(request)
            .await
            .unwrap()
            .message()
            .await
            .unwrap();

        let calls = server.requests();
        assert_eq!(calls.len(), 3);
        assert_eq!(
            ExecutePlanRequest::decode(calls[0].1.as_slice())
                .unwrap()
                .user_context,
            Some(expected.clone())
        );
        assert_eq!(
            AnalyzePlanRequest::decode(calls[1].1.as_slice())
                .unwrap()
                .user_context,
            Some(expected.clone())
        );
        assert_eq!(
            ExecutePlanRequest::decode(calls[2].1.as_slice())
                .unwrap()
                .user_context,
            Some(expected)
        );
        server.assert_finished();
    }
}

#[tokio::test]
async fn reattach_and_release_keep_the_identity_from_the_initial_typed_request() {
    let server = MockServer::start(vec![
        Ok(vec![Err(Status::unavailable("stream disconnected"))]),
        reply(&completed()),
    ])
    .await;
    let client = server.client(Some("configured")).await;
    let mut stream = client
        .execute_plan_reattachable(ExecutePlanRequest::default())
        .await
        .unwrap();
    assert_eq!(
        tokio::time::timeout(std::time::Duration::from_secs(5), stream.message())
            .await
            .unwrap()
            .unwrap(),
        Some(completed())
    );
    let calls = server.requests();
    assert!(calls[1].0.ends_with("/ReattachExecute"));
    let execute = ExecutePlanRequest::decode(calls[0].1.as_slice()).unwrap();
    let reattach = ReattachExecuteRequest::decode(calls[1].1.as_slice()).unwrap();
    assert_eq!(execute.user_context.as_ref().unwrap().user_id, "configured");
    assert_eq!(reattach.user_context, execute.user_context);
    let release = stream.iter.create_release_request(true).await;
    assert_eq!(release.user_context, execute.user_context);
    let release = stream.iter.create_release_request(false).await;
    assert_eq!(release.user_context, execute.user_context);
    server.assert_finished();
}

#[tokio::test]
async fn every_artifact_chunk_has_user_context_with_or_without_a_configured_id() {
    let mut file = tempfile::NamedTempFile::new().unwrap();
    file.write_all(&vec![42; CHUNK_SIZE * 2 + 1]).unwrap();
    let path = file.path().to_str().unwrap();
    for user_id in [None, Some("configured")] {
        let server = MockServer::start(vec![
            reply(&AddArtifactsResponse::default()),
            reply(&AddArtifactsResponse::default()),
        ])
        .await;
        let client = server.client(user_id).await;
        client
            .add_artifacts(&[path], false, false, true)
            .await
            .unwrap();
        client
            .add_named_artifact("files/named", path)
            .await
            .unwrap();
        let calls = server.requests();
        assert_eq!(
            calls.len(),
            6,
            "both upload paths must send all three chunks"
        );
        for (method, bytes) in calls {
            assert!(method.ends_with("/AddArtifacts"));
            let request = AddArtifactsRequest::decode(bytes.as_slice()).unwrap();
            assert_eq!(request.session_id, client.session_id());
            assert_eq!(
                request.user_context,
                Some(UserContext {
                    user_id: user_id.unwrap_or_default().into(),
                    ..Default::default()
                })
            );
        }
        server.assert_finished();
    }
}

#[tokio::test]
async fn typed_plan_requests_without_a_configured_id_still_have_context() {
    let server = MockServer::start(vec![
        reply(&completed()),
        reply(&AnalyzePlanResponse::default()),
        reply(&completed()),
    ])
    .await;
    let client = server.client(None).await;
    client
        .execute_plan(ExecutePlanRequest::default())
        .await
        .unwrap()
        .message()
        .await
        .unwrap();
    client
        .analyze_plan(AnalyzePlanRequest::default())
        .await
        .unwrap();
    client
        .execute_plan_reattachable(ExecutePlanRequest::default())
        .await
        .unwrap()
        .message()
        .await
        .unwrap();
    let calls = server.requests();
    assert_eq!(
        ExecutePlanRequest::decode(calls[0].1.as_slice())
            .unwrap()
            .user_context,
        Some(UserContext::default())
    );
    assert_eq!(
        AnalyzePlanRequest::decode(calls[1].1.as_slice())
            .unwrap()
            .user_context,
        Some(UserContext::default())
    );
    assert_eq!(
        ExecutePlanRequest::decode(calls[2].1.as_slice())
            .unwrap()
            .user_context,
        Some(UserContext::default())
    );
    server.assert_finished();
}

#[tokio::test]
async fn raw_requests_remain_byte_exact_including_unknown_fields() {
    let server = MockServer::start(vec![
        reply(&completed()),
        reply(&AnalyzePlanResponse::default()),
        reply(&completed()),
    ])
    .await;
    let client = server.client(Some("configured")).await;
    let mut execute = ExecutePlanRequest::default().encode_to_vec();
    let mut analyze = AnalyzePlanRequest::default().encode_to_vec();
    let mut reattach = ReattachExecuteRequest::default().encode_to_vec();
    // Unknown protobuf field 2047, varint value 1; must survive without re-encoding.
    for request in [&mut execute, &mut analyze, &mut reattach] {
        request.extend_from_slice(&[0xf8, 0x7f, 0x01]);
    }
    client
        .execute_plan_raw(execute.clone())
        .await
        .unwrap()
        .message()
        .await
        .unwrap();
    client.analyze_plan_raw(analyze.clone()).await.unwrap();
    client
        .reattach_execute_raw(reattach.clone())
        .await
        .unwrap()
        .message()
        .await
        .unwrap();
    assert_eq!(
        server
            .requests()
            .into_iter()
            .map(|(_, bytes)| bytes)
            .collect::<Vec<_>>(),
        [execute, analyze, reattach]
    );
    server.assert_finished();
}

#[tokio::test]
async fn test_fill_user_context_missing_id_survives_reattach_and_release() {
    let builder = ChannelBuilder::parse("sc://localhost/;user_id=test_user").unwrap();
    let client = SparkConnectClient::connect(&builder).await.unwrap();
    let expected = Some(UserContext {
        user_id: "test_user".to_string(),
        ..Default::default()
    });

    for user_context in [None, Some(UserContext::default())] {
        let mut request = ExecutePlanRequest {
            user_context,
            ..Default::default()
        };
        client.fill_user_context(&mut request.user_context);
        assert_eq!(request.user_context, expected);

        let iter = ExecutePlanResponseReattachableIterator::new(request);
        assert_eq!(iter.request().user_context, expected);
        assert_eq!(iter.create_reattach_request().await.user_context, expected);
        assert_eq!(
            iter.create_release_request(true).await.user_context,
            expected
        );
        assert_eq!(
            iter.create_release_request(false).await.user_context,
            expected
        );
    }
}

#[tokio::test]
async fn test_fill_user_context_preserves_explicit_id_and_other_fields() {
    let builder = ChannelBuilder::parse("sc://localhost/;user_id=test_user").unwrap();
    let client = SparkConnectClient::connect(&builder).await.unwrap();

    for user_id in ["", "explicit_user"] {
        let mut context = UserContext {
            user_id: user_id.to_string(),
            user_name: "original_name".to_string(),
            extensions: vec![Default::default()],
        };
        context.extensions[0].type_url = "type.googleapis.com/test.Context".to_string();
        context.extensions[0].value = vec![1, 2, 3];
        let mut expected = context.clone();
        expected.user_id = if user_id.is_empty() {
            "test_user"
        } else {
            user_id
        }
        .to_string();
        let mut request = AnalyzePlanRequest {
            user_context: Some(context),
            ..Default::default()
        };

        client.fill_user_context(&mut request.user_context);
        assert_eq!(request.user_context, Some(expected));
    }
}

#[tokio::test]
async fn test_fill_user_context_without_configured_id_keeps_message_present() {
    let builder = ChannelBuilder::parse("sc://localhost").unwrap();
    let client = SparkConnectClient::connect(&builder).await.unwrap();
    assert_eq!(client.user_id(), None);

    for mut context in [
        None,
        Some(UserContext::default()),
        Some(UserContext {
            user_id: "request_user".to_string(),
            user_name: "original_name".to_string(),
            ..Default::default()
        }),
    ] {
        let expected = context.clone().unwrap_or_default();
        client.fill_user_context(&mut context);
        assert_eq!(context, Some(expected));
    }
}
