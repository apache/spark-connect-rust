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

pub mod engine;
mod ids;
mod schema;
mod service;

/// Generated server bindings for custom embedding. This experimental API follows
/// the protocol schema and may change with the protocol.
pub mod grpc {
    pub use spark_connect_proto::spark_connect_service_server;
}

/// Transport options for the experimental frontend.
/// Start from [`ServerOptions::default()`] and assign the desired fields.
#[derive(Clone, Debug)]
#[non_exhaustive]
pub struct ServerOptions {
    /// Maximum Arrow chunk payload in bytes, excluding the protobuf envelope.
    /// Defaults to 90% of 128 MiB, as in Spark's JVM server. A client can request
    /// a smaller size. Zero disables chunking, even when the client opts in.
    /// Clients must accept this payload plus the protobuf envelope: a 128 MiB
    /// receive limit accommodates the default; tonic's 4 MiB default does not.
    /// Alternatively, clients can request a smaller chunk size.
    /// Whole batches sent without chunking are not bounded by this option.
    pub max_arrow_chunk_size: usize,
}

impl Default for ServerOptions {
    fn default() -> Self {
        Self {
            max_arrow_chunk_size: 128 * 1024 * 1024 * 9 / 10,
        }
    }
}

struct CancelOnDrop(std::sync::Arc<service::ConnectService>);

impl Drop for CancelOnDrop {
    fn drop(&mut self) {
        self.0.cancel_all();
    }
}

fn tcp_incoming(listener: tokio::net::TcpListener) -> tonic::transport::server::TcpIncoming {
    // Builder socket options do not apply to serve_with_incoming_shutdown.
    tonic::transport::server::TcpIncoming::from(listener).with_nodelay(Some(true))
}

/// Serve with an injected execution engine and a fresh session registry.
///
/// The frontend binds each protocol session to the engine session created for it.
/// The caller owns the engine and signals shutdown before releasing its resources.
/// Dropping this future also cancels registered engine work, without waiting for
/// the bounded transport drain used when shutdown is signalled.
pub async fn serve_with_engine(
    listener: tokio::net::TcpListener,
    engine: std::sync::Arc<dyn engine::Engine>,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<(), tonic::transport::Error> {
    serve_with_engine_and_options(listener, engine, ServerOptions::default(), shutdown).await
}

/// Serve with an injected engine and explicit transport options.
///
/// Session lifetime and shutdown follow [`serve_with_engine`]. Producers should
/// bound engine batches independently of this transport chunk size. Clients must
/// allow protobuf overhead in their gRPC receive limits.
pub async fn serve_with_engine_and_options(
    listener: tokio::net::TcpListener,
    engine: std::sync::Arc<dyn engine::Engine>,
    options: ServerOptions,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<(), tonic::transport::Error> {
    let service = std::sync::Arc::new(service::ConnectService::new(engine, options));
    let _cancel_on_drop = CancelOnDrop(service.clone());
    let router = tonic::transport::Server::builder().add_service(
        grpc::spark_connect_service_server::SparkConnectServiceServer::from_arc(service.clone())
            .max_decoding_message_size(128 * 1024 * 1024),
    );
    let (draining, drain_started) = tokio::sync::oneshot::channel();
    let server = router.serve_with_incoming_shutdown(tcp_incoming(listener), async move {
        shutdown.await;
        service.cancel_all();
        let _ = draining.send(());
    });
    tokio::pin!(server);
    tokio::select! {
        result = &mut server => result,
        _ = drain_started => {
            // A peer that stops reading must not prevent local process shutdown.
            tokio::time::timeout(std::time::Duration::from_secs(2), &mut server)
                .await.unwrap_or(Ok(()))
        }
    }
}

#[cfg(test)]
mod tests {
    use futures::StreamExt;

    #[tokio::test]
    async fn accepted_connections_disable_nagle() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let mut incoming = super::tcp_incoming(listener);
        let _client = tokio::net::TcpStream::connect(address).await.unwrap();
        let accepted = incoming.next().await.unwrap().unwrap();
        assert!(accepted.nodelay().unwrap());
    }
}
