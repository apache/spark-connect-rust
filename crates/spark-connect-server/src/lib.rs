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
mod schema;
mod service;

// tonic's generated async-trait methods duplicate Future's must_use annotation.
#[allow(clippy::double_must_use)]
pub mod grpc {
    include!(concat!(
        env!("OUT_DIR"),
        "/spark.connect.SparkConnectService.rs"
    ));
}

/// Serve with an injected Spark JVM engine and a fresh session registry.
///
/// The frontend binds each protocol session to the engine session created for it.
/// The caller owns the JVM and signals shutdown before releasing JNI resources.
pub async fn serve_with_engine(
    listener: tokio::net::TcpListener,
    engine: std::sync::Arc<dyn engine::Engine>,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> Result<(), tonic::transport::Error> {
    let service = std::sync::Arc::new(service::ConnectService::new(engine));
    let router = tonic::transport::Server::builder().add_service(
        grpc::spark_connect_service_server::SparkConnectServiceServer::from_arc(service.clone())
            .max_decoding_message_size(64 * 1024 * 1024),
    );
    let (draining, drain_started) = tokio::sync::oneshot::channel();
    let server = router.serve_with_incoming_shutdown(
        tokio_stream::wrappers::TcpListenerStream::new(listener),
        async move {
            shutdown.await;
            service.cancel_all();
            let _ = draining.send(());
        },
    );
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
