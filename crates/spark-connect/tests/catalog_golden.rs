//! Golden parity test for the catalog: the calls of the Rust `Catalog` send the plans that the
//! reference PySpark client sends for the same calls.
//!
//! The goldens live in `tests/golden/catalog.jsonl` (captured by
//! `scripts/capture_catalog_golden.py`). A case is the base64-encoded `spark.connect.Plan`s that a
//! call of PySpark's `Catalog` sends, in order, whose plan IDs and origins are cleared, as the
//! `common` of their relations is here. PySpark runs every call at once, also the ones that return
//! lists. Of the Rust calls, those that return a DataFrame are lazy: the plan of the call is the
//! relation of that DataFrame, which it sends when it runs. The other calls run at once, against a
//! loopback service that records the requests it is sent. A case that the golden lacks, or of which
//! the Rust API sends other plans, fails the test. As PySpark's `_execute_and_fetch`, which runs an
//! operation as the execution of its DataFrame, each relation has a plan ID, which the golden
//! clears, and each request the tags of the thread.

use std::collections::{BTreeSet, HashMap};
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::task::{Context, Poll};

use base64::{engine::general_purpose::STANDARD, Engine};
use futures::stream;
use prost::Message;
use spark_connect::catalog::Catalog;
use spark_connect::{DataFrame, DataType, Result, SparkSession, StorageLevel, StorageLevelExt};
use spark_connect_core::bytes_codec::BytesCodec;
use spark_connect_proto as proto;
use tokio::net::TcpListener;
use tonic::body::Body;
use tonic::codegen::{http, BoxFuture, Service};
use tonic::server::{NamedService, ServerStreamingService, UnaryService};
use tonic::{Request, Response, Status};

/// The root relations of the plans of each case of the golden, by name, in the order that the
/// call sent them, of their `common` cleared.
fn load_goldens() -> HashMap<String, Vec<proto::Relation>> {
    // Goldens live at the repo root; tests run with CWD at the crate dir.
    let path = format!(
        "{}/../../tests/golden/catalog.jsonl",
        env!("CARGO_MANIFEST_DIR")
    );
    let text = std::fs::read_to_string(path).expect("golden file catalog.jsonl missing");
    let mut goldens = HashMap::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let case: serde_json::Value = serde_json::from_str(line).unwrap();
        let name = case["name"].as_str().unwrap().to_string();
        let plans = case["b64"]
            .as_array()
            .unwrap_or_else(|| panic!("{name}: no plans"))
            .iter()
            .map(|b64| {
                let bytes = STANDARD.decode(b64.as_str().unwrap()).unwrap();
                match proto::Plan::decode(bytes.as_slice()).unwrap().op_type {
                    Some(proto::plan::OpType::Root(mut root)) => {
                        root.common = None;
                        root
                    }
                    other => panic!("{name}: a plan of no root relation: {other:?}"),
                }
            })
            .collect();
        assert!(
            goldens.insert(name.clone(), plans).is_none(),
            "{name} is in the golden twice"
        );
    }
    goldens
}

// The loopback service.

const SERVICE: &str = "/spark.connect.SparkConnectService/";

/// The execution requests that the service was sent, in order.
type Sent = Arc<Mutex<Vec<proto::ExecutePlanRequest>>>;

/// A Spark Connect service that records the request of each execution and answers it with no rows,
/// but `currentDatabase`, which it answers as a server of the default database does: `listViews`
/// of a pattern reads it.
#[derive(Clone)]
struct Recorder(Sent);

impl NamedService for Recorder {
    const NAME: &'static str = "spark.connect.SparkConnectService";
}

impl Service<http::Request<Body>> for Recorder {
    type Response = http::Response<Body>;
    type Error = Infallible;
    type Future = BoxFuture<Self::Response, Self::Error>;

    fn poll_ready(&mut self, _: &mut Context<'_>) -> Poll<std::result::Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: http::Request<Body>) -> Self::Future {
        let method = request.uri().path().trim_start_matches(SERVICE).to_string();
        let sent = self.0.clone();
        Box::pin(async move {
            let mut grpc = tonic::server::Grpc::new(BytesCodec);
            Ok(match method.as_str() {
                "ExecutePlan" => grpc.server_streaming(Execute(sent), request).await,
                "Config" => grpc.unary(Config, request).await,
                "ReleaseExecute" => grpc.unary(ReleaseExecute, request).await,
                _ => grpc.unary(Unsupported, request).await,
            })
        })
    }
}

/// ExecutePlan: records the request, and completes, after the row of `currentDatabase`.
struct Execute(Sent);

impl ServerStreamingService<Vec<u8>> for Execute {
    type Response = Vec<u8>;
    type ResponseStream = stream::Iter<std::vec::IntoIter<std::result::Result<Vec<u8>, Status>>>;
    type Future = BoxFuture<Response<Self::ResponseStream>, Status>;

    fn call(&mut self, request: Request<Vec<u8>>) -> Self::Future {
        let request = proto::ExecutePlanRequest::decode(request.into_inner().as_slice()).unwrap();
        let response = |response_type| {
            proto::ExecutePlanResponse {
                session_id: request.session_id.clone(),
                operation_id: request.operation_id.clone().unwrap_or_default(),
                response_id: "golden".into(),
                response_type: Some(response_type),
                ..Default::default()
            }
            .encode_to_vec()
        };
        let mut frames = Vec::new();
        if let Some(proto::plan::OpType::Root(root)) =
            request.plan.as_ref().and_then(|plan| plan.op_type.as_ref())
        {
            if is_current_database(root) {
                frames.push(Ok(response(
                    proto::execute_plan_response::ResponseType::ArrowBatch(row_of("default")),
                )));
            }
        }
        self.0.lock().unwrap().push(request.clone());
        frames.push(Ok(response(
            proto::execute_plan_response::ResponseType::ResultComplete(
                proto::execute_plan_response::ResultComplete {},
            ),
        )));
        Box::pin(async { Ok(Response::new(stream::iter(frames))) })
    }
}

fn is_current_database(root: &proto::Relation) -> bool {
    matches!(
        &root.rel_type,
        Some(proto::relation::RelType::Catalog(proto::Catalog {
            cat_type: Some(proto::catalog::CatType::CurrentDatabase(_)),
        }))
    )
}

/// The Arrow batch of one row of the string `value`.
fn row_of(value: &str) -> proto::execute_plan_response::ArrowBatch {
    use arrow::array::StringArray;
    use arrow::datatypes::{DataType as ArrowType, Field, Schema};
    use arrow::ipc::writer::StreamWriter;
    use arrow::record_batch::RecordBatch;

    let schema = Arc::new(Schema::new(vec![Field::new(
        "current_database()",
        ArrowType::Utf8,
        false,
    )]));
    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![Arc::new(StringArray::from(vec![value]))],
    )
    .unwrap();
    let mut data = Vec::new();
    let mut writer = StreamWriter::try_new(&mut data, &schema).unwrap();
    writer.write(&batch).unwrap();
    writer.finish().unwrap();
    drop(writer);
    proto::execute_plan_response::ArrowBatch {
        row_count: 1,
        data: data.into(),
        ..Default::default()
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

/// ReleaseExecute: the client releases an execution after it has read its responses.
struct ReleaseExecute;

impl UnaryService<Vec<u8>> for ReleaseExecute {
    type Response = Vec<u8>;
    type Future = BoxFuture<Response<Vec<u8>>, Status>;

    fn call(&mut self, request: Request<Vec<u8>>) -> Self::Future {
        let request =
            proto::ReleaseExecuteRequest::decode(request.into_inner().as_slice()).unwrap();
        let body = proto::ReleaseExecuteResponse {
            session_id: request.session_id,
            operation_id: Some(request.operation_id),
            ..Default::default()
        }
        .encode_to_vec();
        Box::pin(async { Ok(Response::new(body)) })
    }
}

/// Any other RPC: refused.
struct Unsupported;

impl UnaryService<Vec<u8>> for Unsupported {
    type Response = Vec<u8>;
    type Future = BoxFuture<Response<Vec<u8>>, Status>;

    fn call(&mut self, _request: Request<Vec<u8>>) -> Self::Future {
        Box::pin(async { Err(Status::unimplemented("not answered by the catalog golden")) })
    }
}

/// Serves the recorder on a thread of its own, since a session blocks the thread that calls it.
fn serve() -> (String, Sent) {
    let sent = Sent::default();
    let service = Recorder(sent.clone());
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
    (format!("sc://{}", receiver.recv().unwrap()), sent)
}

// The cases, as the capture calls PySpark's Catalog.

/// A call of the Rust catalog, which gives the DataFrame of a lazy call.
type Call = Box<dyn Fn(&Catalog) -> Option<DataFrame>>;

/// A call that runs at once. What it returns is not part of the golden: the loopback service
/// answers most calls with no rows.
fn eager<T>(call: impl Fn(&Catalog) -> Result<T> + 'static) -> Call {
    Box::new(move |catalog| {
        let _ = call(catalog);
        None
    })
}

/// A call that returns the DataFrame of its catalog relation, which it sends when it runs.
fn lazy(call: impl Fn(&Catalog) -> Result<DataFrame> + 'static) -> Call {
    Box::new(move |catalog| Some(call(catalog).unwrap()))
}

const TABLE: &str = "golden_tbl";
const VIEW: &str = "golden_view";
const FUNCTION: &str = "golden_fn";
const DATABASE: &str = "golden_db";
const PATH: &str = "/tmp/spark-connect-rust-golden-catalog";

/// The schema of the capture's tables, PySpark's `StructType` of a long `a` and a string `b`.
fn schema() -> DataType {
    DataType::from_ddl("a BIGINT, b STRING").unwrap()
}

fn options(pairs: &[(&str, &str)]) -> HashMap<String, String> {
    pairs
        .iter()
        .map(|(key, value)| (key.to_string(), value.to_string()))
        .collect()
}

fn cases() -> Vec<(&'static str, Call)> {
    vec![
        ("current_catalog", eager(|c| c.current_catalog())),
        (
            "set_current_catalog",
            eager(|c| c.set_current_catalog("spark_catalog")),
        ),
        ("list_catalogs", lazy(|c| c.list_catalogs())),
        (
            "list_catalogs_pattern",
            lazy(|c| c.list_catalogs_with_pattern(Some("spark*"))),
        ),
        ("current_database", eager(|c| c.current_database())),
        (
            "set_current_database",
            eager(|c| c.set_current_database("default")),
        ),
        ("list_databases", lazy(|c| c.list_databases())),
        (
            "list_databases_pattern",
            lazy(|c| c.list_databases_with_pattern(Some("def*"))),
        ),
        ("get_database", lazy(|c| c.get_database("default"))),
        ("database_exists", eager(|c| c.database_exists("default"))),
        (
            "create_database",
            eager(|c| c.create_database(DATABASE, true, options(&[("k", "v")]))),
        ),
        (
            "create_database_defaults",
            eager(|c| c.create_database(DATABASE, false, HashMap::new())),
        ),
        (
            "drop_database",
            eager(|c| c.drop_database(DATABASE, true, true)),
        ),
        (
            "drop_database_defaults",
            eager(|c| c.drop_database(DATABASE, false, false)),
        ),
        (
            "create_table",
            eager(|c| {
                c.create_table_with(
                    TABLE,
                    None,
                    Some("parquet"),
                    Some(&schema()),
                    Some("a table"),
                    options(&[("compression", "snappy")]),
                )
            }),
        ),
        (
            "create_table_path",
            eager(|c| c.create_table(TABLE, Some(PATH), None, None)),
        ),
        // PySpark's createExternalTable creates the table as createTable does, of a
        // `description` option bound to createTable's parameter.
        (
            "create_external_table",
            eager(|c| {
                c.create_external_table_with(
                    TABLE,
                    Some(PATH),
                    Some("parquet"),
                    Some(&schema()),
                    options(&[("compression", "snappy")]),
                )
            }),
        ),
        (
            "create_external_table_description",
            eager(|c| {
                c.create_external_table_with(
                    TABLE,
                    None,
                    Some("parquet"),
                    None,
                    options(&[("description", "d"), ("k", "v")]),
                )
            }),
        ),
        ("list_tables", lazy(|c| c.list_tables())),
        (
            "list_tables_db",
            lazy(|c| c.list_tables_in_database("default")),
        ),
        (
            "list_tables_pattern",
            lazy(|c| c.list_tables_with_pattern(None, Some("golden*"))),
        ),
        (
            "list_tables_db_pattern",
            lazy(|c| c.list_tables_with_pattern(Some("default"), Some("golden*"))),
        ),
        ("get_table", lazy(|c| c.get_table(TABLE))),
        ("table_exists", eager(|c| c.table_exists(TABLE))),
        (
            "table_exists_db",
            eager(|c| c.table_exists_with_database(TABLE, Some("default"))),
        ),
        ("list_columns", lazy(|c| c.list_columns(TABLE))),
        (
            "list_columns_db",
            lazy(|c| c.list_columns_with_database(TABLE, Some("default"))),
        ),
        ("list_partitions", lazy(|c| c.list_partitions(TABLE))),
        (
            "get_table_properties",
            eager(|c| c.get_table_properties(TABLE)),
        ),
        (
            "get_create_table_string",
            eager(|c| c.get_create_table_string(TABLE, false)),
        ),
        (
            "get_create_table_string_serde",
            eager(|c| c.get_create_table_string(TABLE, true)),
        ),
        ("analyze_table", eager(|c| c.analyze_table(TABLE, false))),
        (
            "analyze_table_no_scan",
            eager(|c| c.analyze_table(TABLE, true)),
        ),
        ("is_cached", eager(|c| c.is_cached(TABLE))),
        ("cache_table", eager(|c| c.cache_table(TABLE))),
        (
            "cache_table_storage_level",
            eager(|c| c.cache_table_with_storage_level(TABLE, Some(StorageLevel::disk_only()))),
        ),
        ("uncache_table", eager(|c| c.uncache_table(TABLE))),
        ("clear_cache", eager(|c| c.clear_cache())),
        ("refresh_table", eager(|c| c.refresh_table(TABLE))),
        ("refresh_by_path", eager(|c| c.refresh_by_path(PATH))),
        ("recover_partitions", eager(|c| c.recover_partitions(TABLE))),
        ("truncate_table", eager(|c| c.truncate_table(TABLE))),
        ("drop_table", eager(|c| c.drop_table(TABLE, true, true))),
        (
            "drop_table_defaults",
            eager(|c| c.drop_table(TABLE, false, false)),
        ),
        ("list_functions", lazy(|c| c.list_functions())),
        (
            "list_functions_db",
            lazy(|c| c.list_functions_in_database("default")),
        ),
        (
            "list_functions_pattern",
            lazy(|c| c.list_functions_with_pattern(None, Some("golden*"))),
        ),
        (
            "list_functions_db_pattern",
            lazy(|c| c.list_functions_with_pattern(Some("default"), Some("golden*"))),
        ),
        ("get_function", lazy(|c| c.get_function(FUNCTION))),
        ("function_exists", eager(|c| c.function_exists(FUNCTION))),
        (
            "function_exists_db",
            eager(|c| c.function_exists_with_database(FUNCTION, Some("default"))),
        ),
        ("list_views", lazy(|c| c.list_views(None, None))),
        (
            "list_views_db",
            lazy(|c| c.list_views(Some("default"), None)),
        ),
        (
            "list_views_db_pattern",
            lazy(|c| c.list_views(Some("default"), Some("golden*"))),
        ),
        // A pattern of no database: as PySpark, the call runs currentDatabase first, and lists
        // the views of the database it returns.
        (
            "list_views_pattern",
            lazy(|c| c.list_views(None, Some("golden*"))),
        ),
        ("drop_temp_view", eager(|c| c.drop_temp_view(VIEW))),
        (
            "drop_global_temp_view",
            eager(|c| c.drop_global_temp_view(VIEW)),
        ),
        ("drop_view", eager(|c| c.drop_view(VIEW, true))),
        ("drop_view_defaults", eager(|c| c.drop_view(VIEW, false))),
    ]
}

#[test]
fn catalog_calls_send_the_plans_of_official_pyspark() {
    let goldens = load_goldens();
    let cases = cases();
    let names: BTreeSet<&str> = cases.iter().map(|(name, _)| *name).collect();
    assert_eq!(names.len(), cases.len(), "a case is listed twice");
    let golden_names: BTreeSet<&str> = goldens.keys().map(String::as_str).collect();
    assert_eq!(
        names, golden_names,
        "the cases differ from those of the golden"
    );

    let (remote, sent) = serve();
    let spark = SparkSession::builder()
        .remote(&remote)
        .get_or_create()
        .unwrap();
    // The calls run on this thread, whose tags their requests carry.
    spark.add_tag("golden").unwrap();
    let catalog = spark.catalog();
    let mut failures = Vec::new();
    let mut failed_cases = BTreeSet::new();
    for (name, call) in &cases {
        let failures_before = failures.len();
        sent.lock().unwrap().clear();
        let dataframe = call(&catalog);
        let mut plans = Vec::new();
        for request in std::mem::take(&mut *sent.lock().unwrap()) {
            if request.tags != ["golden"] {
                failures.push(format!("{name}: a request of the tags {:?}", request.tags));
            }
            match request.plan.and_then(|plan| plan.op_type) {
                Some(proto::plan::OpType::Root(root)) => plans.push(root),
                other => failures.push(format!("{name}: a plan of no root relation: {other:?}")),
            }
        }
        if let Some(dataframe) = dataframe {
            plans.push(dataframe.to_relation().unwrap());
        }
        for plan in &mut plans {
            // As PySpark's `_create_proto_relation`, which gives every relation a plan ID.
            if plan
                .common
                .as_ref()
                .and_then(|common| common.plan_id)
                .is_none()
            {
                failures.push(format!("{name}: a relation of no plan ID: {plan:?}"));
            }
            plan.common = None;
        }
        let expected = &goldens[*name];
        if &plans != expected {
            failures.push(format!(
                "{name}: MISMATCH\n  expected: {expected:?}\n  actual:   {plans:?}"
            ));
        }
        if failures.len() > failures_before {
            failed_cases.insert(*name);
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} golden catalog cases failed:\n{}",
        failed_cases.len(),
        cases.len(),
        failures.join("\n")
    );
}

/// As PySpark's `createTable`, and `createExternalTable`, which calls it: a schema that is not a
/// struct type raises `NOT_EXPECTED_TYPE` of the name of its class, before any request.
#[test]
fn create_table_of_a_schema_that_is_not_a_struct_fails_as_pyspark() {
    let (remote, sent) = serve();
    let spark = SparkSession::builder()
        .remote(&remote)
        .get_or_create()
        .unwrap();
    let catalog = spark.catalog();
    let array = DataType::Array {
        element_type: Box::new(DataType::Integer),
        contains_null: true,
    };
    for (result, class) in [
        (
            catalog.create_table_with(
                TABLE,
                None,
                None,
                Some(&DataType::Long),
                None,
                HashMap::new(),
            ),
            "LongType",
        ),
        (
            catalog.create_external_table_with(
                TABLE,
                Some(PATH),
                None,
                Some(&array),
                HashMap::new(),
            ),
            "ArrayType",
        ),
    ] {
        let Err(error) = result else {
            panic!("the schema {class} was taken");
        };
        assert_eq!(
            error.kind,
            spark_connect_core::error::SparkErrorKind::TypeError
        );
        assert_eq!(
            error.message(),
            format!("[NOT_EXPECTED_TYPE] Argument `schema` should be struct type, got {class}.")
        );
    }
    assert!(sent.lock().unwrap().is_empty());
}
