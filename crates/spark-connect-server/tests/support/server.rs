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

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use spark_connect_proto::spark_connect_service_client::SparkConnectServiceClient;
use spark_connect_server::engine::Engine;
use tokio::sync::oneshot;
use tokio::task::JoinHandle;
use tonic::transport::Channel;

pub struct Server {
    // Some test binaries need a second connection to the same listener.
    #[allow(dead_code)]
    pub address: SocketAddr,
    pub client: SparkConnectServiceClient<Channel>,
    pub shutdown: Option<oneshot::Sender<()>>,
    task: Option<JoinHandle<()>>,
}

impl Server {
    pub async fn start_with_engine(engine: Arc<dyn Engine>) -> Self {
        Self::start_with_options(engine, spark_connect_server::ServerOptions::default()).await
    }

    pub async fn start_with_options(
        engine: Arc<dyn Engine>,
        options: spark_connect_server::ServerOptions,
    ) -> Self {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        let (shutdown, stopped) = oneshot::channel();
        let task = tokio::spawn(async move {
            spark_connect_server::serve_with_engine_and_options(listener, engine, options, async {
                let _ = stopped.await;
            })
            .await
            .unwrap();
        });
        let client = SparkConnectServiceClient::connect(format!("http://{address}"))
            .await
            .unwrap();
        Self {
            address,
            client,
            shutdown: Some(shutdown),
            task: Some(task),
        }
    }

    pub async fn stop(mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        tokio::time::timeout(Duration::from_secs(5), self.task.take().unwrap())
            .await
            .unwrap()
            .unwrap();
    }
}

impl Drop for Server {
    fn drop(&mut self) {
        if let Some(shutdown) = self.shutdown.take() {
            let _ = shutdown.send(());
        }
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}
