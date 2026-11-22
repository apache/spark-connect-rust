// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

//! The failure of a call of a client while the client converts another failure (see
//! `SparkConnectClient::inside_error_handling`): a call that a handler of a record of the
//! conversion makes, within it, is not converted, as PySpark's `_handle_error` raises the error
//! of a call as it is while it handles another, and the call of another task of an async caller,
//! which the thread polls between two polls of the conversion, is converted. No Spark server
//! required: a loopback tonic service fails each analysis with an error whose details it holds
//! for a while for one of them.

mod common;

use std::sync::{Arc, Mutex};
use std::time::Duration;

use common::{FakeServer, FakeService};
use prost::Message;
use spark_connect_core::{ChannelBuilder, SparkConnectClient};
use spark_connect_proto::analyze_plan_request::{Analyze, Schema};
use spark_connect_proto::{
    plan, relation, AnalyzePlanRequest, AnalyzePlanResponse, ConfigRequest, ConfigResponse,
    FetchErrorDetailsRequest, FetchErrorDetailsResponse, Plan, Relation, Sql,
};
use tonic::codegen::BoxFuture;
use tonic::server::UnaryService;
use tonic::{Request, Response, Status};
use tonic_prost::ProstCodec;

/// The error ID, and the query, of the failure whose details the server holds.
const HELD: &str = "held";
/// How long the server holds them.
const HOLD: Duration = Duration::from_millis(1000);
/// When the other call starts, after the first one.
const LATER: Duration = Duration::from_millis(200);

/// The error IDs of the details that the server was asked for, in order.
type Fetched = Arc<Mutex<Vec<String>>>;

/// Held by each test, as the listener of the retries is of the process.
static SERIAL: Mutex<()> = Mutex::new(());

/// An error of the server whose details it keeps under `error_id`.
fn error_kept_under(error_id: &str) -> Status {
    #[derive(Clone, PartialEq, prost::Message)]
    struct ErrorInfo {
        #[prost(string, tag = "1")]
        reason: String,
        #[prost(map = "string, string", tag = "3")]
        metadata: std::collections::HashMap<String, String>,
    }
    #[derive(Clone, PartialEq, prost::Message)]
    struct RpcStatus {
        #[prost(int32, tag = "1")]
        code: i32,
        #[prost(string, tag = "2")]
        message: String,
        #[prost(message, repeated, tag = "3")]
        details: Vec<prost_types::Any>,
    }
    let info = ErrorInfo {
        reason: "org.apache.spark.sql.AnalysisException".into(),
        metadata: [("errorId".to_string(), error_id.to_string())].into(),
    };
    let status = RpcStatus {
        code: tonic::Code::Internal as i32,
        message: "boom".into(),
        details: vec![prost_types::Any {
            type_url: "type.googleapis.com/google.rpc.ErrorInfo".into(),
            value: info.encode_to_vec(),
        }],
    };
    Status::with_details(tonic::Code::Internal, "boom", status.encode_to_vec().into())
}

/// Fails each analysis with an error of the server kept under the query of its plan.
struct Analysis;

impl UnaryService<AnalyzePlanRequest> for Analysis {
    type Response = AnalyzePlanResponse;
    type Future = BoxFuture<Response<AnalyzePlanResponse>, Status>;

    fn call(&mut self, request: Request<AnalyzePlanRequest>) -> Self::Future {
        let query = match &request.get_ref().analyze {
            Some(Analyze::Schema(Schema {
                plan:
                    Some(Plan {
                        op_type:
                            Some(plan::OpType::Root(Relation {
                                rel_type: Some(relation::RelType::Sql(sql)),
                                ..
                            })),
                    }),
            })) => sql.query.clone(),
            _ => String::new(),
        };
        Box::pin(async move { Err(error_kept_under(&query)) })
    }
}

/// Gives the details of an error, of none, at once, but those of [`HELD`] after [`HOLD`].
struct Details(Fetched);

impl UnaryService<FetchErrorDetailsRequest> for Details {
    type Response = FetchErrorDetailsResponse;
    type Future = BoxFuture<Response<FetchErrorDetailsResponse>, Status>;

    fn call(&mut self, request: Request<FetchErrorDetailsRequest>) -> Self::Future {
        let error_id = request.get_ref().error_id.clone();
        self.0.lock().unwrap().push(error_id.clone());
        Box::pin(async move {
            if error_id == HELD {
                tokio::time::sleep(HOLD).await;
            }
            Ok(Response::new(FetchErrorDetailsResponse::default()))
        })
    }
}

/// Of no settings, as of the stack trace of an error.
struct Settings;

impl UnaryService<ConfigRequest> for Settings {
    type Response = ConfigResponse;
    type Future = BoxFuture<Response<ConfigResponse>, Status>;

    fn call(&mut self, _request: Request<ConfigRequest>) -> Self::Future {
        Box::pin(async { Ok(Response::new(ConfigResponse::default())) })
    }
}

async fn serve(fetched: Fetched) -> FakeServer {
    FakeService::new()
        .unary("AnalyzePlan", ProstCodec::default, |_| Analysis)
        .unary("FetchErrorDetails", ProstCodec::default, move |_| {
            Details(fetched.clone())
        })
        .unary("Config", ProstCodec::default, |_| Settings)
        .serve()
        .await
}

async fn connect(server: &FakeServer) -> SparkConnectClient {
    let builder = ChannelBuilder::parse(&format!(
        "sc://localhost:{}/;user_id=guard",
        server.address.port()
    ))
    .unwrap();
    SparkConnectClient::connect(&builder).await.unwrap()
}

/// The analysis of the schema of a plan of the query `query`, which the server fails.
fn failing(client: &SparkConnectClient, query: &str) -> AnalyzePlanRequest {
    let relation = Relation {
        rel_type: Some(relation::RelType::Sql(Sql {
            query: query.to_string(),
            ..Default::default()
        })),
        ..Default::default()
    };
    AnalyzePlanRequest {
        session_id: client.session_id().to_string(),
        analyze: Some(Analyze::Schema(Schema {
            plan: Some(Plan {
                op_type: Some(plan::OpType::Root(relation)),
            }),
        })),
        ..Default::default()
    }
}

/// A failure of `first` whose details the server holds, and, [`LATER`], a failure of `second` on
/// the same thread, as two tasks of an async caller: both are converted, the details of `second`
/// asked for while those of `first` are held.
async fn two_failures(first: SparkConnectClient, second: SparkConnectClient, fetched: &Fetched) {
    let held = async {
        let request = failing(&first, HELD);
        first.analyze_plan(request).await.unwrap_err()
    };
    let other = async {
        tokio::time::sleep(LATER).await;
        let request = failing(&second, "other");
        second.analyze_plan(request).await.unwrap_err()
    };
    let (_, _) = tokio::time::timeout(Duration::from_secs(10), async { tokio::join!(held, other) })
        .await
        .expect("the failures waited past the bound");
    // The details of the other failure are asked for while those of the first are held.
    assert_eq!(
        *fetched.lock().unwrap(),
        vec![HELD.to_string(), "other".to_string()]
    );
    fetched.lock().unwrap().clear();
}

#[test]
fn the_failure_of_another_task_of_the_client_is_converted_during_a_conversion() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    // Of a runtime of one thread, whose tasks interleave on it, as of an async caller.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let fetched = Fetched::default();
        let server = serve(fetched.clone()).await;
        let client = connect(&server).await;
        two_failures(client.clone(), client.clone(), &fetched).await;
        // Of the client of another session.
        two_failures(client.clone(), client.with_new_session_id(), &fetched).await;
    });
    // Of a runtime of one worker, whose tasks share it.
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(1)
        .enable_all()
        .build()
        .unwrap();
    runtime.block_on(async {
        let fetched = Fetched::default();
        let server = serve(fetched.clone()).await;
        let client = connect(&server).await;
        let tasks = {
            let fetched = fetched.clone();
            tokio::spawn(async move { two_failures(client.clone(), client, &fetched).await })
        };
        tasks.await.unwrap();
        drop(server);
    });
}

/// A listener of the retries of the process that runs a failing analysis of its client, through
/// the blocking API, at the record of the failure (`GRPC Error received`) of the analysis of
/// [`HELD`], as a handler of the record makes a call.
struct Nested {
    client: Mutex<Option<SparkConnectClient>>,
    fetched: Fetched,
    /// Whether the nested call was converted, of the details of its error asked for.
    converted: Mutex<Option<bool>>,
}

impl spark_connect_core::retries::RetriesListener for Nested {
    fn retries_exceeded(
        &self,
        _exceeded: spark_connect_core::retries::RetriesExceeded,
    ) -> spark_connect_core::error::Result<()> {
        Ok(())
    }

    fn caught(
        &self,
        _error: spark_connect_core::error::SparkError,
    ) -> Option<spark_connect_core::error::SparkError> {
        None
    }

    fn rpc_error(
        &self,
        _error: &spark_connect_core::error::GrpcioError,
    ) -> spark_connect_core::retries::Delivery {
        let client = self.client.lock().unwrap().take()?;
        let request = failing(&client, "nested");
        let _ = spark_connect_core::runtime::block_on(client.analyze_plan(request));
        let converted = self.fetched.lock().unwrap().iter().any(|id| id == "nested");
        *self.converted.lock().unwrap() = Some(converted);
        None
    }
}

#[test]
fn a_call_within_a_conversion_of_its_client_is_not_converted() {
    let _serial = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    let nested = Arc::new(Nested {
        client: Mutex::new(None),
        fetched: Fetched::default(),
        converted: Mutex::new(None),
    });
    spark_connect_core::retries::set_listener(Some(nested.clone()));
    // The server and the channel of the client run on the runtime of the blocking API, which the
    // nested call waits on, as the Python binding's do.
    let server = spark_connect_core::runtime::block_on(serve(nested.fetched.clone()));
    let client = spark_connect_core::runtime::block_on(connect(&server));
    let failure_of_the_held_call = || {
        *nested.client.lock().unwrap() = Some(client.clone());
        let request = failing(&client, HELD);
        async {
            tokio::time::timeout(Duration::from_secs(10), client.analyze_plan(request))
                .await
                .expect("the failure waited past the bound")
        }
    };
    let assert_not_converted = |failed: spark_connect_core::Result<AnalyzePlanResponse>| {
        assert!(failed.is_err());
        assert_eq!(*nested.converted.lock().unwrap(), Some(false));
        assert_eq!(*nested.fetched.lock().unwrap(), vec![HELD.to_string()]);
        nested.fetched.lock().unwrap().clear();
        *nested.converted.lock().unwrap() = None;
    };
    // Of the blocking API, whose nested call runs on the thread that polls the conversion.
    assert_not_converted(spark_connect_core::runtime::block_on(
        failure_of_the_held_call(),
    ));
    // Of a runtime of one thread, whose nested blocking call runs on another thread, for the
    // caller.
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    assert_not_converted(runtime.block_on(failure_of_the_held_call()));
    spark_connect_core::retries::set_listener(None);
    drop(server);
}
