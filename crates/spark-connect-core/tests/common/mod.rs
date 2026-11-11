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

//! The loopback Spark Connect service of the integration tests of this crate. A test file
//! takes it with `mod common;`, names the RPCs that its service answers and the handler of
//! each, and serves them on a loopback port:
//!
//! ```ignore
//! let server = FakeService::new()
//!     .server_streaming("ExecutePlan", ProstCodec::default, {
//!         let state = state.clone();
//!         move |_| Execute(state.clone())
//!     })
//!     .unary("ReleaseExecute", ProstCodec::default, move |_| ReleaseRpc(state.clone()))
//!     .serve()
//!     .await;
//! let url = format!("sc://{}", server.address);
//! ```
//!
//! The handlers are the `tonic::server` service traits (`UnaryService`,
//! `ServerStreamingService`, `ClientStreamingService`), so that a test keeps what its RPCs do
//! next to its cases; this module has only the routing and the listener.

#![allow(dead_code)] // every test binary compiles this module and uses only a part of it

use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures::stream;
use tokio::net::TcpListener;
use tonic::body::Body;
use tonic::codec::Codec;
use tonic::codegen::{http, BoxFuture, Service};
use tonic::server::{
    ClientStreamingService, Grpc, NamedService, ServerStreamingService, UnaryService,
};
use tonic::transport::Server;

/// The path of an RPC of the service is this prefix followed by the name of its method.
pub const SERVICE: &str = "/spark.connect.SparkConnectService/";

type Reply = BoxFuture<http::Response<Body>, Infallible>;
/// Answers one HTTP request of a method, whose name it is given.
type Handler = Arc<dyn Fn(&str, http::Request<Body>) -> Reply + Send + Sync>;

/// A Spark Connect service that answers the RPCs a test names, each with the handler the test
/// gives for it. A method without a route goes to the fallback, when the test sets one, and is
/// answered as UNIMPLEMENTED otherwise.
#[derive(Clone, Default)]
pub struct FakeService {
    routes: Vec<(&'static str, Handler)>,
    fallback: Option<Handler>,
}

impl FakeService {
    pub fn new() -> Self {
        Self::default()
    }

    /// Answers the unary RPC `method` with the service that `service` makes for each call.
    /// `service` is given the name of the method.
    pub fn unary<C, S>(
        mut self,
        method: &'static str,
        codec: impl Fn() -> C + Send + Sync + 'static,
        service: impl Fn(&str) -> S + Send + Sync + 'static,
    ) -> Self
    where
        C: Codec + Send + 'static,
        S: UnaryService<C::Decode, Response = C::Encode> + Send + 'static,
        S::Future: Send,
    {
        self.routes.push((method, unary_handler(codec, service)));
        self
    }

    /// Answers the server-streaming RPC `method` with the service that `service` makes for
    /// each call. `service` is given the name of the method.
    pub fn server_streaming<C, S>(
        mut self,
        method: &'static str,
        codec: impl Fn() -> C + Send + Sync + 'static,
        service: impl Fn(&str) -> S + Send + Sync + 'static,
    ) -> Self
    where
        C: Codec + Send + 'static,
        S: ServerStreamingService<C::Decode, Response = C::Encode> + Send + 'static,
        S::Future: Send,
        S::ResponseStream: Send + 'static,
    {
        let handler: Handler = Arc::new(move |method, request| {
            let service = service(method);
            let mut grpc = Grpc::new(codec());
            Box::pin(async move { Ok(grpc.server_streaming(service, request).await) })
        });
        self.routes.push((method, handler));
        self
    }

    /// Answers the client-streaming RPC `method` with the service that `service` makes for
    /// each call. `service` is given the name of the method.
    pub fn client_streaming<C, S>(
        mut self,
        method: &'static str,
        codec: impl Fn() -> C + Send + Sync + 'static,
        service: impl Fn(&str) -> S + Send + Sync + 'static,
    ) -> Self
    where
        C: Codec + Send + 'static,
        S: ClientStreamingService<C::Decode, Response = C::Encode> + Send + 'static,
        S::Future: Send,
    {
        let handler: Handler = Arc::new(move |method, request| {
            let service = service(method);
            let mut grpc = Grpc::new(codec());
            Box::pin(async move { Ok(grpc.client_streaming(service, request).await) })
        });
        self.routes.push((method, handler));
        self
    }

    /// Answers every method that has no route as a unary RPC, with the service that `service`
    /// makes for each call. `service` is given the name of the method.
    pub fn fallback_unary<C, S>(
        mut self,
        codec: impl Fn() -> C + Send + Sync + 'static,
        service: impl Fn(&str) -> S + Send + Sync + 'static,
    ) -> Self
    where
        C: Codec + Send + 'static,
        S: UnaryService<C::Decode, Response = C::Encode> + Send + 'static,
        S::Future: Send,
    {
        self.fallback = Some(unary_handler(codec, service));
        self
    }

    /// Serves the service on a loopback port of its own, until the returned server is dropped.
    pub async fn serve(self) -> FakeServer {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let incoming = stream::unfold(listener, |listener| async {
            Some((listener.accept().await.map(|(socket, _)| socket), listener))
        });
        let task = tokio::spawn(async move {
            Server::builder()
                .add_service(self)
                .serve_with_incoming(incoming)
                .await
                .unwrap();
        });
        FakeServer { address, task }
    }
}

fn unary_handler<C, S>(
    codec: impl Fn() -> C + Send + Sync + 'static,
    service: impl Fn(&str) -> S + Send + Sync + 'static,
) -> Handler
where
    C: Codec + Send + 'static,
    S: UnaryService<C::Decode, Response = C::Encode> + Send + 'static,
    S::Future: Send,
{
    Arc::new(move |method, request| {
        let service = service(method);
        let mut grpc = Grpc::new(codec());
        Box::pin(async move { Ok(grpc.unary(service, request).await) })
    })
}

impl NamedService for FakeService {
    const NAME: &'static str = "spark.connect.SparkConnectService";
}

impl Service<http::Request<Body>> for FakeService {
    type Response = http::Response<Body>;
    type Error = Infallible;
    type Future = Reply;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: http::Request<Body>) -> Self::Future {
        let method = request.uri().path().trim_start_matches(SERVICE).to_string();
        let route = self.routes.iter().find(|(name, _)| *name == method);
        match route.map(|(_, handler)| handler).or(self.fallback.as_ref()) {
            Some(handler) => handler(&method, request),
            // UNIMPLEMENTED, as a tonic server answers a method that it does not have.
            None => Box::pin(async {
                Ok(http::Response::builder()
                    .status(200)
                    .header("grpc-status", "12")
                    .header("content-type", "application/grpc")
                    .body(Body::empty())
                    .unwrap())
            }),
        }
    }
}

/// A served [`FakeService`]. Dropping it stops the server.
pub struct FakeServer {
    /// The loopback address that the server listens on.
    pub address: SocketAddr,
    task: tokio::task::JoinHandle<()>,
}

impl Drop for FakeServer {
    fn drop(&mut self) {
        self.task.abort();
    }
}
