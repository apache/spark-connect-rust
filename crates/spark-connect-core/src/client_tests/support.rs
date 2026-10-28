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

//! A scripted gRPC endpoint using in-memory I/O, without a Spark server or TCP port.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use tonic::codegen::{http, Body, BoxFuture, BoxStream, Service, StdError};
use tonic::{Request, Response, Status};

use super::{
    ChannelBuilder, SparkConnectClient, SparkConnectServiceClient, GRPC_MAX_MESSAGE_LENGTH_DEFAULT,
};
use crate::bytes_codec::BytesCodec;

pub type Reply = Result<Vec<Result<Vec<u8>, Status>>, Status>;
type RecordedRequests = Arc<Mutex<Vec<(String, Vec<u8>)>>>;

#[derive(Clone)]
struct RpcService {
    replies: Arc<Mutex<VecDeque<Reply>>>,
    requests: RecordedRequests,
}

impl tonic::server::NamedService for RpcService {
    const NAME: &'static str = "spark.connect.SparkConnectService";
}

impl<B> Service<http::Request<B>> for RpcService
where
    B: Body + Send + 'static,
    B::Error: Into<StdError> + Send + 'static,
{
    type Response = http::Response<tonic::body::Body>;
    type Error = std::convert::Infallible;
    type Future = BoxFuture<Self::Response, Self::Error>;

    fn poll_ready(
        &mut self,
        _: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        std::task::Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: http::Request<B>) -> Self::Future {
        let path = request.uri().path().to_string();
        let method = RpcMethod {
            service: self.clone(),
            path,
        };
        Box::pin(async move {
            let mut grpc = tonic::server::Grpc::new(BytesCodec);
            Ok(if method.path.ends_with("/AddArtifacts") {
                grpc.client_streaming(method, request).await
            } else {
                grpc.server_streaming(method, request).await
            })
        })
    }
}

struct RpcMethod {
    service: RpcService,
    path: String,
}

impl RpcMethod {
    fn record(&self, request: Vec<u8>) {
        self.service
            .requests
            .lock()
            .unwrap()
            .push((self.path.clone(), request));
    }

    fn reply(&self) -> Reply {
        self.service
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| Err(Status::internal("unexpected RPC")))
    }
}

impl tonic::server::ServerStreamingService<Vec<u8>> for RpcMethod {
    type Response = Vec<u8>;
    type ResponseStream = BoxStream<Vec<u8>>;
    type Future = BoxFuture<Response<Self::ResponseStream>, Status>;

    fn call(&mut self, request: Request<Vec<u8>>) -> Self::Future {
        self.record(request.into_inner());
        let reply = self.reply();
        Box::pin(async move {
            let stream: Self::ResponseStream = Box::pin(futures::stream::iter(reply?));
            Ok(Response::new(stream))
        })
    }
}

impl tonic::server::ClientStreamingService<Vec<u8>> for RpcMethod {
    type Response = Vec<u8>;
    type Future = BoxFuture<Response<Vec<u8>>, Status>;

    fn call(&mut self, request: Request<tonic::Streaming<Vec<u8>>>) -> Self::Future {
        let method = RpcMethod {
            service: self.service.clone(),
            path: self.path.clone(),
        };
        Box::pin(async move {
            let mut stream = request.into_inner();
            while let Some(request) = stream.message().await? {
                method.record(request);
            }
            let mut replies = method.reply()?.into_iter();
            Ok(Response::new(replies.next().expect("missing unary reply")?))
        })
    }
}

pub struct MockServer {
    service: RpcService,
    connections: tokio::sync::mpsc::UnboundedSender<tokio::io::DuplexStream>,
    task: tokio::task::JoinHandle<()>,
}

impl MockServer {
    pub async fn start(replies: Vec<Reply>) -> Self {
        use futures::StreamExt;
        let (connections, incoming) = tokio::sync::mpsc::unbounded_channel();
        let incoming = tokio_stream::wrappers::UnboundedReceiverStream::new(incoming)
            .map(Ok::<_, std::io::Error>);
        let service = RpcService {
            replies: Arc::new(Mutex::new(replies.into())),
            requests: Arc::new(Mutex::new(Vec::new())),
        };
        let server_service = service.clone();
        let task = tokio::spawn(async move {
            tonic::transport::Server::builder()
                .add_service(server_service)
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        Self {
            service,
            connections,
            task,
        }
    }

    pub async fn client(&self, user_id: Option<&str>) -> SparkConnectClient {
        let url = match user_id {
            Some(id) => format!("sc://localhost/;user_id={id}"),
            None => "sc://localhost".to_string(),
        };
        let mut client = SparkConnectClient::connect(&ChannelBuilder::parse(&url).unwrap())
            .await
            .unwrap();
        let connections = self.connections.clone();
        let connector = tower::service_fn(move |_| {
            let (client, server) = tokio::io::duplex(64 * 1024);
            let result = connections
                .send(server)
                .map(|_| hyper_util::rt::TokioIo::new(client))
                .map_err(|_| {
                    std::io::Error::new(std::io::ErrorKind::BrokenPipe, "test server stopped")
                });
            std::future::ready(result)
        });
        client.channel = tonic::transport::Endpoint::from_static("http://localhost")
            .connect_with_connector_lazy(connector);
        client.stub = SparkConnectServiceClient::new(client.channel.clone())
            .max_decoding_message_size(GRPC_MAX_MESSAGE_LENGTH_DEFAULT)
            .max_encoding_message_size(GRPC_MAX_MESSAGE_LENGTH_DEFAULT);
        client
    }

    pub fn requests(&self) -> Vec<(String, Vec<u8>)> {
        self.service.requests.lock().unwrap().clone()
    }

    pub fn assert_finished(&self) {
        assert!(
            self.service.replies.lock().unwrap().is_empty(),
            "not all expected RPCs were made"
        );
    }
}

impl Drop for MockServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}
