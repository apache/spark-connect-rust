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

//! The request that removes the relation of a dropped checkpoint, against a loopback gRPC service
//! that answers the checkpoint as a server does and records every request; no Spark server
//! required. The live tests of `e2e_dataframe.rs` check that the server releases the relation.

use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use futures::stream;
use prost::Message;
use spark_connect::{ExecutePlanHook, SparkSession};
use spark_connect_core::bytes_codec::BytesCodec;
use spark_connect_proto as proto;
use spark_connect_proto::execute_plan_request::request_option::RequestOption as RequestOptionKind;
use tokio::net::TcpListener;
use tonic::body::Body;
use tonic::codegen::{http, BoxFuture, Service};
use tonic::server::{NamedService, ServerStreamingService, UnaryService};
use tonic::{Request, Response, Status};

const SERVICE: &str = "/spark.connect.SparkConnectService/";

/// The id of the relation that the service gives a checkpoint.
const RELATION_ID: &str = "checkpoint-relation";

/// What the service received, in order.
#[derive(Debug)]
enum Received {
    Execute(Box<proto::ExecutePlanRequest>),
    Release(proto::ReleaseExecuteRequest),
}

type Log = Arc<Mutex<Vec<Received>>>;

#[derive(Clone)]
struct TestService(Log);

impl NamedService for TestService {
    const NAME: &'static str = "spark.connect.SparkConnectService";
}

impl Service<http::Request<Body>> for TestService {
    type Response = http::Response<Body>;
    type Error = Infallible;
    type Future = BoxFuture<Self::Response, Self::Error>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: http::Request<Body>) -> Self::Future {
        let method = request.uri().path().trim_start_matches(SERVICE).to_string();
        let log = self.0.clone();
        Box::pin(async move {
            let mut grpc = tonic::server::Grpc::new(BytesCodec);
            Ok(match method.as_str() {
                "ExecutePlan" => grpc.server_streaming(Execute(log), request).await,
                "ReleaseExecute" => grpc.unary(ReleaseExecute(log), request).await,
                "Config" => grpc.unary(Config, request).await,
                _ => grpc.unary(Unsupported, request).await,
            })
        })
    }
}

/// Whether `request` asks for a reattachable execution.
fn is_reattachable(request: &proto::ExecutePlanRequest) -> bool {
    request.request_options.iter().any(|option| {
        matches!(
            option.request_option,
            Some(RequestOptionKind::ReattachOptions(proto::ReattachOptions {
                reattachable: true
            }))
        )
    })
}

/// ExecutePlan: records the request, and answers as a server does: a checkpoint with the result
/// of its relation, and a reattachable execution with the ResultComplete that ends it, which a
/// server sends of no other execution.
struct Execute(Log);

impl ServerStreamingService<Vec<u8>> for Execute {
    type Response = Vec<u8>;
    type ResponseStream = stream::Iter<std::vec::IntoIter<Result<Vec<u8>, Status>>>;
    type Future = BoxFuture<Response<Self::ResponseStream>, Status>;

    fn call(&mut self, request: Request<Vec<u8>>) -> Self::Future {
        let request = proto::ExecutePlanRequest::decode(request.into_inner().as_slice()).unwrap();
        let response = |response_type| {
            Ok(proto::ExecutePlanResponse {
                session_id: request.session_id.clone(),
                operation_id: request.operation_id.clone().unwrap_or_default(),
                response_id: "checkpoint".into(),
                response_type: Some(response_type),
                ..Default::default()
            }
            .encode_to_vec())
        };
        let mut frames = Vec::new();
        if let Some(proto::plan::OpType::Command(proto::Command {
            command_type: Some(proto::command::CommandType::CheckpointCommand(_)),
        })) = request.plan.as_ref().and_then(|plan| plan.op_type.as_ref())
        {
            frames.push(response(
                proto::execute_plan_response::ResponseType::CheckpointCommandResult(
                    proto::CheckpointCommandResult {
                        relation: Some(proto::CachedRemoteRelation {
                            relation_id: RELATION_ID.into(),
                        }),
                    },
                ),
            ));
        }
        if is_reattachable(&request) {
            frames.push(response(
                proto::execute_plan_response::ResponseType::ResultComplete(
                    proto::execute_plan_response::ResultComplete {},
                ),
            ));
        }
        self.0
            .lock()
            .unwrap()
            .push(Received::Execute(Box::new(request)));
        Box::pin(async { Ok(Response::new(stream::iter(frames))) })
    }
}

/// ReleaseExecute: records the release.
struct ReleaseExecute(Log);

impl UnaryService<Vec<u8>> for ReleaseExecute {
    type Response = Vec<u8>;
    type Future = BoxFuture<Response<Vec<u8>>, Status>;

    fn call(&mut self, request: Request<Vec<u8>>) -> Self::Future {
        let request =
            proto::ReleaseExecuteRequest::decode(request.into_inner().as_slice()).unwrap();
        let body = proto::ReleaseExecuteResponse {
            session_id: request.session_id.clone(),
            operation_id: Some(request.operation_id.clone()),
            ..Default::default()
        }
        .encode_to_vec();
        self.0.lock().unwrap().push(Received::Release(request));
        Box::pin(async { Ok(Response::new(body)) })
    }
}

/// Config: answers with no pairs, which leaves plan compression off.
struct Config;

impl UnaryService<Vec<u8>> for Config {
    type Response = Vec<u8>;
    type Future = BoxFuture<Response<Vec<u8>>, Status>;

    fn call(&mut self, request: Request<Vec<u8>>) -> Self::Future {
        let session_id = proto::ConfigRequest::decode(request.into_inner().as_slice())
            .unwrap()
            .session_id;
        let body = proto::ConfigResponse {
            session_id,
            ..Default::default()
        }
        .encode_to_vec();
        Box::pin(async { Ok(Response::new(body)) })
    }
}

struct Unsupported;

impl UnaryService<Vec<u8>> for Unsupported {
    type Response = Vec<u8>;
    type Future = BoxFuture<Response<Vec<u8>>, Status>;

    fn call(&mut self, _request: Request<Vec<u8>>) -> Self::Future {
        Box::pin(async { Err(Status::unimplemented("not answered by the test service")) })
    }
}

/// Serves on a thread of its own, since a session blocks the thread that calls it, and returns a
/// session of the service with the log of what it receives.
fn serve() -> (SparkSession, Log) {
    let log = Log::default();
    let service = TestService(log.clone());
    let (sender, receiver) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async move {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            sender.send(listener.local_addr().unwrap()).unwrap();
            let incoming = stream::unfold(listener, |listener| async {
                Some((listener.accept().await.map(|(socket, _)| socket), listener))
            });
            tonic::transport::Server::builder()
                .add_service(service)
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
    });
    let remote = format!("sc://{}", receiver.recv().unwrap());
    let session = SparkSession::builder()
        .remote(&remote)
        .get_or_create()
        .unwrap();
    (session, log)
}

/// The command of `request`, if its plan is one.
fn command_of(request: &proto::ExecutePlanRequest) -> Option<&proto::command::CommandType> {
    match request.plan.as_ref()?.op_type.as_ref()? {
        proto::plan::OpType::Command(command) => command.command_type.as_ref(),
        _ => None,
    }
}

/// The request of the removal of the checkpoint's relation, once the service received it.
fn removal(log: &Log) -> proto::ExecutePlanRequest {
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        let found = log
            .lock()
            .unwrap()
            .iter()
            .find_map(|received| match received {
                Received::Execute(request)
                    if matches!(
                        command_of(request),
                        Some(proto::command::CommandType::RemoveCachedRemoteRelationCommand(_))
                    ) =>
                {
                    Some(request.as_ref().clone())
                }
                _ => None,
            });
        if let Some(request) = found {
            return request;
        }
        assert!(
            Instant::now() < deadline,
            "the checkpoint was never removed"
        );
        std::thread::sleep(Duration::from_millis(10));
    }
}

#[test]
fn a_dropped_checkpoint_is_removed_by_one_execution_of_the_thread_as_pyspark() {
    // As PySpark's `CachedRemoteRelation.__del__`: one ExecutePlan of the removal, through the
    // channel itself, of the request of `_execute_plan_request_with_metadata`. Its only option
    // allows result chunking, so the execution is not reattachable and no ReleaseExecute
    // follows it, it goes to no hook of the executions, and it has the tags and the user
    // context extensions of the thread that drops the DataFrame, the thread of the finalizer.
    let (spark, log) = serve();
    let checkpoint = spark.range(3).unwrap().local_checkpoint().unwrap();
    spark.add_tag("dropper").unwrap();
    let extension = prost_types::Any {
        type_url: "type.googleapis.com/test.dropper".into(),
        value: b"dropper".to_vec(),
    };
    spark.add_threadlocal_user_context_extension(extension.clone());
    let hooked = Arc::new(Mutex::new(0));
    let calls = Arc::clone(&hooked);
    let hook: ExecutePlanHook = Arc::new(move |request| {
        *calls.lock().unwrap() += 1;
        Ok(request)
    });
    spark.set_execute_plan_hook(Some(hook));
    drop(checkpoint);

    let removal = removal(&log);
    assert_eq!(
        command_of(&removal),
        Some(
            &proto::command::CommandType::RemoveCachedRemoteRelationCommand(
                proto::RemoveCachedRemoteRelationCommand {
                    relation: Some(proto::CachedRemoteRelation {
                        relation_id: RELATION_ID.into(),
                    }),
                }
            )
        )
    );
    let chunking = RequestOptionKind::ResultChunkingOptions(proto::ResultChunkingOptions {
        allow_arrow_batch_chunking: true,
        preferred_arrow_chunk_size: None,
    });
    let options: Vec<_> = removal
        .request_options
        .iter()
        .map(|option| option.request_option.clone())
        .collect();
    assert_eq!(options, vec![Some(chunking)]);
    assert_eq!(removal.session_id, spark.session_id());
    assert_eq!(removal.tags, vec!["dropper".to_string()]);
    let context = removal.user_context.as_ref().expect("a user context");
    assert_eq!(context.extensions, vec![extension]);
    assert!(removal.client_type.is_some());
    let operation_id = removal.operation_id.clone().expect("an operation id");
    assert_eq!(*hooked.lock().unwrap(), 0, "the removal went to the hook");

    // The client would release a reattachable execution once it has read its responses.
    std::thread::sleep(Duration::from_millis(500));
    let log = log.lock().unwrap();
    assert!(
        !log.iter().any(|received| matches!(
            received,
            Received::Release(release) if release.operation_id == operation_id
        )),
        "the removal was released: {log:?}"
    );
}
