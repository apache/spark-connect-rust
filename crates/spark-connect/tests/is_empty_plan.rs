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

//! The request of `DataFrame::is_empty`, against a loopback gRPC service that records the
//! bytes of every ExecutePlan request; no Spark server required.

use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use futures::stream;
use prost::Message;
use spark_connect::SparkSession;
use spark_connect_core::bytes_codec::BytesCodec;
use spark_connect_proto as proto;
use tokio::net::TcpListener;
use tonic::body::Body;
use tonic::codegen::{http, BoxFuture, Service};
use tonic::server::{NamedService, ServerStreamingService, UnaryService};
use tonic::{Request, Response, Status};

const SERVICE: &str = "/spark.connect.SparkConnectService/";
type Requests = Arc<Mutex<Vec<Vec<u8>>>>;

#[derive(Clone)]
struct TestService(Requests);

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
        let requests = self.0.clone();
        Box::pin(async move {
            let mut grpc = tonic::server::Grpc::new(BytesCodec);
            Ok(match method.as_str() {
                "ExecutePlan" => grpc.server_streaming(Execute(requests), request).await,
                _ => grpc.unary(Config, request).await,
            })
        })
    }
}

/// ExecutePlan: records the request and answers with an empty, complete result.
struct Execute(Requests);

impl ServerStreamingService<Vec<u8>> for Execute {
    type Response = Vec<u8>;
    type ResponseStream = futures::stream::Iter<std::vec::IntoIter<Result<Vec<u8>, Status>>>;
    type Future = BoxFuture<Response<Self::ResponseStream>, Status>;

    fn call(&mut self, request: Request<Vec<u8>>) -> Self::Future {
        let bytes = request.into_inner();
        let session_id = proto::ExecutePlanRequest::decode(bytes.as_slice())
            .unwrap()
            .session_id;
        self.0.lock().unwrap().push(bytes);
        let done = proto::ExecutePlanResponse {
            session_id,
            response_id: "complete".into(),
            response_type: Some(proto::execute_plan_response::ResponseType::ResultComplete(
                proto::execute_plan_response::ResultComplete {},
            )),
            ..Default::default()
        };
        let frames = stream::iter(vec![Ok(done.encode_to_vec())]);
        Box::pin(async { Ok(Response::new(frames)) })
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

/// Serves on a thread of its own, since a session blocks the thread that calls it.
fn serve() -> (String, Requests) {
    let requests = Requests::default();
    let service = TestService(requests.clone());
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
    (format!("sc://{}", receiver.recv().unwrap()), requests)
}

#[test]
fn is_empty_sends_a_limit_of_one_row_over_an_empty_projection_as_pyspark() {
    let (remote, requests) = serve();
    let session = SparkSession::builder()
        .remote(&remote)
        .get_or_create()
        .unwrap();
    assert!(session.range(3).unwrap().is_empty().unwrap());

    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 1);
    let request = proto::ExecutePlanRequest::decode(requests[0].as_slice()).unwrap();
    let Some(proto::plan::OpType::Root(root)) = request.plan.and_then(|p| p.op_type) else {
        panic!("not a root plan");
    };
    let Some(proto::relation::RelType::Limit(limit)) = root.rel_type else {
        panic!("not a limit: {root:?}");
    };
    assert_eq!(limit.limit, 1);
    let input = limit.input.unwrap();
    let Some(proto::relation::RelType::Project(project)) = input.rel_type else {
        panic!("not a projection: {input:?}");
    };
    assert!(project.expressions.is_empty());
    assert!(matches!(
        project.input.unwrap().rel_type,
        Some(proto::relation::RelType::Range(_))
    ));
}
