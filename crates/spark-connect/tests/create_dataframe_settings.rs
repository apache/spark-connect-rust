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

//! The settings that `create_dataframe` and `create_dataframe_from_columns` read, and how they
//! send their relation, against a loopback gRPC service that answers Config and ArtifactStatus
//! and records them; no Spark server required. As PySpark's `createDataFrame`
//! (`connect/session.py`), data that is not empty reads the settings of its conversions in one
//! Config get, and a relation of `spark.sql.session.localRelationCacheThreshold` bytes or more,
//! of any size, is uploaded as cached artifacts.

use std::convert::Infallible;
use std::io::Cursor;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use arrow::array::{Array, Int64Array, StringArray};
use arrow::ipc::reader::StreamReader;
use futures::stream;
use prost::Message;
use spark_connect::row::{Row, Value};
use spark_connect::types::{DataType, StructField};
use spark_connect::{DataFrame, SparkSession};
use spark_connect_core::bytes_codec::BytesCodec;
use spark_connect_proto as proto;
use spark_connect_proto::relation::RelType;
use tokio::net::TcpListener;
use tonic::body::Body;
use tonic::codegen::{http, BoxFuture, Service};
use tonic::server::{NamedService, UnaryService};
use tonic::{Request, Response, Status};

const SERVICE: &str = "/spark.connect.SparkConnectService/";

/// The settings of PySpark 4.3.0's createDataFrame, in the order of its Config get, and the
/// defaults of a Spark 4.3.0 server for them, but the cache threshold, which each test sets.
const SETTINGS: [(&str, &str); 13] = [
    ("spark.sql.timestampType", "TIMESTAMP_LTZ"),
    ("spark.sql.session.timeZone", "UTC"),
    ("spark.sql.session.localRelationCacheThreshold", ""),
    ("spark.sql.session.localRelationSizeLimit", "3221225472"),
    ("spark.sql.session.localRelationChunkSizeRows", "10000"),
    ("spark.sql.session.localRelationChunkSizeBytes", "16777216"),
    (
        "spark.sql.session.localRelationBatchOfChunksSizeBytes",
        "1073741824",
    ),
    (
        "spark.sql.execution.pandas.convertToArrowArraySafely",
        "true",
    ),
    ("spark.sql.execution.pandas.inferPandasDictAsMap", "false"),
    ("spark.sql.pyspark.inferNestedDictAsStruct.enabled", "false"),
    (
        "spark.sql.pyspark.legacy.inferArrayTypeFromFirstElement.enabled",
        "false",
    ),
    (
        "spark.sql.pyspark.legacy.inferMapTypeFromFirstPair.enabled",
        "false",
    ),
    ("spark.sql.execution.arrow.useLargeVarTypes", "false"),
];

/// A Spark 4.3.0 server's default of `spark.sql.session.localRelationCacheThreshold`.
const DEFAULT_CACHE_THRESHOLD: &str = "1048576";

/// What the service received, in order: the keys of a Config get, the names of an
/// ArtifactStatus, or the name of another RPC, which the service refuses.
#[derive(Debug, PartialEq)]
enum Received {
    ConfigGet(Vec<String>),
    ArtifactStatus(Vec<String>),
    Other(String),
}

type Log = Arc<Mutex<Vec<Received>>>;

#[derive(Clone)]
struct TestService {
    cache_threshold: &'static str,
    log: Log,
}

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
        let service = self.clone();
        Box::pin(async move {
            let mut grpc = tonic::server::Grpc::new(BytesCodec);
            Ok(match method.as_str() {
                "Config" => grpc.unary(Config(service), request).await,
                "ArtifactStatus" => grpc.unary(ArtifactStatus(service.log), request).await,
                other => {
                    service
                        .log
                        .lock()
                        .unwrap()
                        .push(Received::Other(other.to_string()));
                    grpc.unary(Unsupported, request).await
                }
            })
        })
    }
}

/// Config: a get of the settings is answered with [`SETTINGS`], and of no other key.
struct Config(TestService);

impl UnaryService<Vec<u8>> for Config {
    type Response = Vec<u8>;
    type Future = BoxFuture<Response<Vec<u8>>, Status>;

    fn call(&mut self, request: Request<Vec<u8>>) -> Self::Future {
        let request = proto::ConfigRequest::decode(request.into_inner().as_slice()).unwrap();
        let keys = match request.operation.and_then(|operation| operation.op_type) {
            Some(proto::config_request::operation::OpType::Get(get)) => get.keys,
            other => panic!("a Config request that the tests do not send: {other:?}"),
        };
        self.0
            .log
            .lock()
            .unwrap()
            .push(Received::ConfigGet(keys.clone()));
        let threshold = self.0.cache_threshold;
        let pairs = keys
            .into_iter()
            .filter_map(|key| {
                let (setting, value) = SETTINGS.iter().find(|(setting, _)| *setting == key)?;
                let value = if *setting == "spark.sql.session.localRelationCacheThreshold" {
                    threshold
                } else {
                    value
                };
                Some(proto::KeyValue {
                    key,
                    value: Some(value.to_string()),
                })
            })
            .collect();
        let body = proto::ConfigResponse {
            session_id: request.session_id,
            pairs,
            ..Default::default()
        }
        .encode_to_vec();
        Box::pin(async { Ok(Response::new(body)) })
    }
}

/// ArtifactStatus: every artifact exists already, so that the upload sends no AddArtifacts.
struct ArtifactStatus(Log);

impl UnaryService<Vec<u8>> for ArtifactStatus {
    type Response = Vec<u8>;
    type Future = BoxFuture<Response<Vec<u8>>, Status>;

    fn call(&mut self, request: Request<Vec<u8>>) -> Self::Future {
        let request =
            proto::ArtifactStatusesRequest::decode(request.into_inner().as_slice()).unwrap();
        let mut names = request.names.clone();
        names.sort();
        self.0.lock().unwrap().push(Received::ArtifactStatus(names));
        let body = proto::ArtifactStatusesResponse {
            session_id: request.session_id,
            statuses: request
                .names
                .into_iter()
                .map(|name| {
                    (
                        name,
                        proto::artifact_statuses_response::ArtifactStatus { exists: true },
                    )
                })
                .collect(),
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
fn serve(cache_threshold: &'static str) -> (SparkSession, Log) {
    let log = Log::default();
    let service = TestService {
        cache_threshold,
        log: log.clone(),
    };
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

fn settings_get() -> Received {
    Received::ConfigGet(SETTINGS.iter().map(|(key, _)| key.to_string()).collect())
}

fn field(name: &str, data_type: DataType) -> StructField {
    StructField {
        name: name.to_string(),
        data_type,
        nullable: true,
        metadata: Default::default(),
    }
}

fn schema() -> DataType {
    DataType::Struct {
        fields: vec![
            field("a", DataType::Long),
            field(
                "s",
                DataType::String {
                    collation: "UTF8_BINARY".to_string(),
                },
            ),
        ],
    }
}

/// `count` rows of a long and a short string: 20 of them are a few hundred bytes of Arrow data.
fn rows(count: i64) -> Vec<Row> {
    (0..count)
        .map(|i| {
            Row::new(
                vec!["a".to_string(), "s".to_string()],
                vec![Value::Long(i), Value::String(format!("s{i}"))],
            )
        })
        .collect()
}

fn relation(frame: &DataFrame) -> RelType {
    frame.to_relation().unwrap().rel_type.unwrap()
}

#[test]
fn rows_and_columns_below_64_kib_are_uploaded_at_the_cache_threshold_as_pyspark() {
    // As PySpark's createDataFrame, whose `cache_threshold <= _table.nbytes` applies at any size:
    // with a threshold of 64 bytes, 20 short rows are uploaded, after the settings are read.
    let (spark, log) = serve("64");
    let frame = spark.create_dataframe(rows(20), schema()).unwrap();
    let RelType::ChunkedCachedLocalRelation(cached) = relation(&frame) else {
        panic!("rows of the cache threshold or more are uploaded");
    };
    assert_eq!(cached.data_hashes.len(), 1);
    let schema_hash = cached
        .schema_hash
        .expect("the schema is an artifact of its own");
    let mut names = vec![
        format!("cache/{}", cached.data_hashes[0]),
        format!("cache/{schema_hash}"),
    ];
    names.sort();
    assert_eq!(
        std::mem::take(&mut *log.lock().unwrap()),
        vec![settings_get(), Received::ArtifactStatus(names)]
    );

    let columns: Vec<arrow::array::ArrayRef> = vec![
        Arc::new(Int64Array::from_iter_values(0..20)),
        Arc::new(Int64Array::from_iter_values(20..40)),
    ];
    let longs = DataType::Struct {
        fields: vec![field("a", DataType::Long), field("b", DataType::Long)],
    };
    let frame = spark.create_dataframe_from_columns(columns, longs).unwrap();
    assert!(matches!(
        relation(&frame),
        RelType::ChunkedCachedLocalRelation(_)
    ));
    let log = std::mem::take(&mut *log.lock().unwrap());
    assert_eq!(log.len(), 2, "{log:?}");
    assert_eq!(log[0], settings_get());
    assert!(matches!(log[1], Received::ArtifactStatus(_)), "{log:?}");
}

#[test]
fn rows_below_the_cache_threshold_are_inline_after_the_settings_are_read() {
    // The default threshold keeps small rows inline, in the IPC stream of their batch, as
    // before; the settings are read all the same, as PySpark reads them.
    let (spark, log) = serve(DEFAULT_CACHE_THRESHOLD);
    let frame = spark.create_dataframe(rows(3), schema()).unwrap();
    let RelType::LocalRelation(local) = relation(&frame) else {
        panic!("rows below the cache threshold are inline");
    };
    let reader = StreamReader::try_new(Cursor::new(local.data.unwrap()), None).unwrap();
    let batches: Vec<_> = reader.collect::<Result<_, _>>().unwrap();
    assert_eq!(batches.len(), 1);
    let a = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    let s = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap();
    assert_eq!(a.values().to_vec(), vec![0, 1, 2]);
    assert_eq!(
        (0..s.len()).map(|i| s.value(i)).collect::<Vec<_>>(),
        vec!["s0", "s1", "s2"]
    );
    assert_eq!(
        std::mem::take(&mut *log.lock().unwrap()),
        vec![settings_get()]
    );
}

#[test]
fn no_rows_and_rows_that_do_not_fit_the_schema_send_no_request() {
    // No rows are a relation of the schema alone, as PySpark's empty data is, with no request;
    // rows are converted before the settings are read, so that a row of another width fails
    // without one.
    let (spark, log) = serve("64");
    let frame = spark.create_dataframe(Vec::new(), schema()).unwrap();
    let RelType::LocalRelation(local) = relation(&frame) else {
        panic!("no rows are a local relation of the schema");
    };
    assert!(local.data.is_none());
    let short = vec![Row::new(vec!["a".to_string()], vec![Value::Long(1)])];
    let error = spark
        .create_dataframe(short, schema())
        .err()
        .expect("a row of another width")
        .to_string();
    assert!(
        error.contains("Row 0 has 1 values, but the schema has 2 fields"),
        "{error}"
    );
    assert!(log.lock().unwrap().is_empty(), "{:?}", log.lock().unwrap());
}
