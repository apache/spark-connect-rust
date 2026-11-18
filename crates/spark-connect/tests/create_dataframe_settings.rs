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
    use_large_var_types: &'static str,
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
        let large = self.0.use_large_var_types;
        let pairs = keys
            .into_iter()
            .filter_map(|key| {
                let (setting, value) = SETTINGS.iter().find(|(setting, _)| *setting == key)?;
                let value = match *setting {
                    "spark.sql.session.localRelationCacheThreshold" => threshold,
                    "spark.sql.execution.arrow.useLargeVarTypes" => large,
                    _ => value,
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
    serve_with(cache_threshold, "false")
}

/// [`serve`], of the value `use_large_var_types` of `spark.sql.execution.arrow.useLargeVarTypes`.
fn serve_with(
    cache_threshold: &'static str,
    use_large_var_types: &'static str,
) -> (SparkSession, Log) {
    let log = Log::default();
    let service = TestService {
        cache_threshold,
        use_large_var_types,
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
    // No rows are a relation of the schema alone, as PySpark's empty data is, with no request,
    // and so are columns of no rows; rows are converted before the settings are read, so that a
    // row of another width fails without one.
    let (spark, log) = serve("64");
    let frame = spark.create_dataframe(Vec::new(), schema()).unwrap();
    let RelType::LocalRelation(local) = relation(&frame) else {
        panic!("no rows are a local relation of the schema");
    };
    assert!(local.data.is_none());
    let empty: Vec<arrow::array::ArrayRef> = vec![
        Arc::new(Int64Array::from(Vec::<i64>::new())),
        Arc::new(Int64Array::from(Vec::<i64>::new())),
    ];
    let longs = DataType::Struct {
        fields: vec![field("a", DataType::Long), field("b", DataType::Long)],
    };
    let from_columns = spark
        .create_dataframe_from_columns(empty, longs.clone())
        .unwrap();
    assert_eq!(
        relation(&from_columns),
        relation(&spark.create_dataframe(Vec::new(), longs).unwrap()),
        "columns of no rows are the relation of no rows"
    );
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

/// `count` rows of one string column `s`, each "abcd", whose pyarrow `nbytes` is `8 * count` as
/// strings (`count` 4-byte offsets and 4 bytes each) and `12 * count` as large strings.
fn strings(count: usize) -> (Vec<Row>, DataType) {
    let rows = (0..count)
        .map(|_| Row::new(vec!["s".to_string()], vec![Value::String("abcd".into())]))
        .collect();
    let schema = DataType::Struct {
        fields: vec![field(
            "s",
            DataType::String {
                collation: "UTF8_BINARY".to_string(),
            },
        )],
    };
    (rows, schema)
}

#[test]
fn the_cache_threshold_is_compared_with_the_nbytes_of_pyarrow_as_pyspark() {
    // PySpark's `cache_threshold <= _table.nbytes`, of pyarrow's `nbytes`, which counts `len`
    // offsets of a string array, not the `len + 1` of its buffer: 1000 strings "abcd" are 8000
    // bytes, so a threshold of 8001 keeps them inline and one of 8000 uploads them.
    for (threshold, cached) in [("8001", false), ("8004", false), ("8000", true)] {
        let (spark, _) = serve(threshold);
        let (rows, schema) = strings(1000);
        let frame = spark.create_dataframe(rows, schema).unwrap();
        assert_eq!(
            matches!(relation(&frame), RelType::ChunkedCachedLocalRelation(_)),
            cached,
            "threshold {threshold}"
        );
    }
    // And of large strings under useLargeVarTypes: 12000 bytes.
    for (threshold, cached) in [("12001", false), ("12000", true), ("10000", true)] {
        let (spark, _) = serve_with(threshold, "true");
        let (rows, schema) = strings(1000);
        let frame = spark.create_dataframe(rows, schema).unwrap();
        assert_eq!(
            matches!(relation(&frame), RelType::ChunkedCachedLocalRelation(_)),
            cached,
            "threshold {threshold} of large strings"
        );
    }
}

#[test]
fn rows_and_columns_are_of_the_large_types_of_pyspark_under_use_large_var_types() {
    // As PySpark's `to_arrow_type(..., prefers_large_types=True)` of
    // `spark.sql.execution.arrow.useLargeVarTypes`: strings and binaries, also in arrays, maps
    // and structs, are large, while the binaries of a variant stay binaries.
    use arrow::datatypes::DataType as A;
    let string = || DataType::String {
        collation: "UTF8_BINARY".to_string(),
    };
    let schema = DataType::Struct {
        fields: vec![
            field("s", string()),
            field("b", DataType::Binary),
            field(
                "a",
                DataType::Array {
                    element_type: Box::new(string()),
                    contains_null: true,
                },
            ),
            field(
                "m",
                DataType::Map {
                    key_type: Box::new(string()),
                    value_type: Box::new(DataType::Binary),
                    value_contains_null: true,
                },
            ),
            field(
                "st",
                DataType::Struct {
                    fields: vec![field("x", string()), field("n", DataType::Long)],
                },
            ),
            field("v", DataType::Variant),
            field("l", DataType::Long),
        ],
    };
    let names = ["s", "b", "a", "m", "st", "v", "l"]
        .map(str::to_string)
        .to_vec();
    let rows = vec![
        Row::new(
            names.clone(),
            vec![
                Value::String("x".into()),
                Value::Binary(vec![1, 2]),
                Value::List(vec![Value::String("y".into()), Value::Null]),
                Value::Map(vec![(Value::String("k".into()), Value::Binary(vec![3]))]),
                Value::Struct(
                    Row::new(
                        vec!["x".to_string(), "n".to_string()],
                        vec![Value::String("z".into()), Value::Long(7)],
                    )
                    .into(),
                ),
                Value::Variant {
                    value: vec![12, 1],
                    metadata: vec![1, 0, 0],
                },
                Value::Long(1),
            ],
        ),
        Row::new(names.clone(), vec![Value::Null; 7]),
    ];
    let types = |frame: &DataFrame| {
        let RelType::LocalRelation(local) = relation(frame) else {
            panic!("small rows are inline");
        };
        let reader = StreamReader::try_new(Cursor::new(local.data.unwrap()), None).unwrap();
        let batches: Vec<_> = reader.collect::<Result<_, _>>().unwrap();
        let schema = batches[0].schema();
        let column = batches[0].column(0).clone();
        (
            schema
                .fields()
                .iter()
                .map(|field| field.data_type().clone())
                .collect::<Vec<_>>(),
            column,
        )
    };
    let leaf = |data_type: &A, index: usize| match data_type {
        A::Struct(fields) => fields[index].data_type().clone(),
        A::List(field) => field.data_type().clone(),
        A::Map(entries, _) => match entries.data_type() {
            A::Struct(fields) => fields[index].data_type().clone(),
            other => panic!("map entries of {other:?}"),
        },
        other => panic!("not nested: {other:?}"),
    };

    let (spark, _) = serve_with(DEFAULT_CACHE_THRESHOLD, "true");
    let (large, column) = types(
        &spark
            .create_dataframe(rows.clone(), schema.clone())
            .unwrap(),
    );
    assert_eq!(large[0], A::LargeUtf8);
    assert_eq!(large[1], A::LargeBinary);
    assert_eq!(leaf(&large[2], 0), A::LargeUtf8);
    assert_eq!(
        (leaf(&large[3], 0), leaf(&large[3], 1)),
        (A::LargeUtf8, A::LargeBinary)
    );
    assert_eq!(
        (leaf(&large[4], 0), leaf(&large[4], 1)),
        (A::LargeUtf8, A::Int64)
    );
    assert_eq!(
        (leaf(&large[5], 0), leaf(&large[5], 1)),
        (A::Binary, A::Binary)
    );
    assert_eq!(large[6], A::Int64);
    // The values are kept.
    let strings = column
        .as_any()
        .downcast_ref::<arrow::array::LargeStringArray>()
        .unwrap();
    assert_eq!((strings.value(0), strings.is_null(1)), ("x", true));

    // The columns of the same types, as the rows, and of PySpark's types without the setting.
    let (small, _) = types(
        &serve(DEFAULT_CACHE_THRESHOLD)
            .0
            .create_dataframe(rows, schema)
            .unwrap(),
    );
    assert_eq!((small[0].clone(), small[1].clone()), (A::Utf8, A::Binary));
    assert_eq!(leaf(&small[2], 0), A::Utf8);
    let columns: Vec<arrow::array::ArrayRef> = vec![
        Arc::new(StringArray::from(vec![Some("x"), None])),
        Arc::new(Int64Array::from(vec![1, 2])),
    ];
    let two = DataType::Struct {
        fields: vec![field("s", string()), field("l", DataType::Long)],
    };
    let (spark, _) = serve_with(DEFAULT_CACHE_THRESHOLD, "true");
    let (large, _) = types(&spark.create_dataframe_from_columns(columns, two).unwrap());
    assert_eq!(large, [A::LargeUtf8, A::Int64]);
}

#[test]
fn the_ipc_of_the_binding_is_of_the_large_types_and_its_nbytes_under_use_large_var_types() {
    // The IPC that the binding gives, of its pandas string columns, as `_pandas_string_ipc`
    // writes them, of 4-byte offsets: under useLargeVarTypes it is sent of PySpark's large
    // strings, and inline or uploaded by their `nbytes`, 12000 bytes for 1000 strings "abcd".
    let (_, schema) = strings(1000);
    let ipc = {
        let column: arrow::array::ArrayRef = Arc::new(StringArray::from(vec!["abcd"; 1000]));
        let batch = arrow::record_batch::RecordBatch::try_from_iter([("s", column)]).unwrap();
        let mut writer =
            arrow::ipc::writer::StreamWriter::try_new(Vec::new(), &batch.schema()).unwrap();
        writer.write(&batch).unwrap();
        writer.into_inner().unwrap()
    };
    let (spark, _) = serve_with(DEFAULT_CACHE_THRESHOLD, "true");
    let frame = spark
        .create_dataframe_from_inline_arrow_ipc(ipc.clone(), schema.clone())
        .unwrap();
    let RelType::LocalRelation(local) = relation(&frame) else {
        panic!("1000 strings are below the default threshold");
    };
    let reader = StreamReader::try_new(Cursor::new(local.data.unwrap()), None).unwrap();
    let batches: Vec<_> = reader.collect::<Result<_, _>>().unwrap();
    assert_eq!(
        batches[0].schema().field(0).data_type(),
        &arrow::datatypes::DataType::LargeUtf8
    );
    assert_eq!(batches[0].num_rows(), 1000);
    for (threshold, cached) in [("12001", false), ("12000", true)] {
        let (spark, _) = serve_with(threshold, "true");
        let frame = spark
            .create_dataframe_from_inline_arrow_ipc(ipc.clone(), schema.clone())
            .unwrap();
        assert_eq!(
            matches!(relation(&frame), RelType::ChunkedCachedLocalRelation(_)),
            cached,
            "threshold {threshold}"
        );
    }
    // Without the setting, the IPC is sent as it is, below the threshold.
    let (spark, _) = serve(DEFAULT_CACHE_THRESHOLD);
    let frame = spark
        .create_dataframe_from_inline_arrow_ipc(ipc.clone(), schema)
        .unwrap();
    let RelType::LocalRelation(local) = relation(&frame) else {
        panic!("inline");
    };
    assert_eq!(local.data.unwrap(), ipc);
}
