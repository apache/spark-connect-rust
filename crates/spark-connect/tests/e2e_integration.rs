//! End-to-end integration tests against a live Spark Connect server.
//!
//! These tests connect to a live Spark Connect server at `sc://localhost:15002`
//! and verify the complete execution path: plan building, execution, Arrow IPC decode, and row collection.
//!
//! Run with: `SPARK_REMOTE=sc://localhost:15002 cargo test --test e2e_integration`
//! Or simply skip with: `cargo test --test e2e_integration` (will skip all tests)

mod common;

use std::time::Duration;

use common::should_run;
use spark_connect::column::{col, lit};
use spark_connect::row::Value;
use spark_connect::session::SparkSession;
use spark_connect::types::DataType;
use spark_connect_core::runtime::block_on;
use spark_connect_core::{ChannelBuilder, SparkConnectClient};
use spark_connect_proto as proto;

fn remote_url() -> String {
    std::env::var("SPARK_REMOTE").unwrap_or_else(|_| "sc://localhost:15002".to_string())
}

async fn connect_client() -> SparkConnectClient {
    let builder = ChannelBuilder::parse(&remote_url()).expect("Failed to parse SPARK_REMOTE");
    SparkConnectClient::connect(&builder)
        .await
        .expect("Failed to connect")
}

/// A request to run `query`, with an operation ID of its own, which it returns as well.
fn sql_execute_request(
    client: &SparkConnectClient,
    query: &str,
) -> (proto::ExecutePlanRequest, String) {
    let operation_id = uuid::Uuid::new_v4().to_string();
    let mut relation = proto::Relation::default();
    relation.rel_type = Some(proto::relation::RelType::Sql(proto::Sql {
        query: query.to_string(),
        ..Default::default()
    }));
    let mut plan = proto::Plan::default();
    plan.op_type = Some(proto::plan::OpType::Root(relation));
    let request = proto::ExecutePlanRequest {
        session_id: client.session_id().to_string(),
        user_context: Some(proto::UserContext::default()),
        operation_id: Some(operation_id.clone()),
        plan: Some(plan),
        ..Default::default()
    };
    (request, operation_id)
}

/// Waits until the server no longer knows `operation_id`, as the client releases executions
/// in the background: `ReattachExecute` of a released execution fails with
/// `OPERATION_NOT_FOUND`, while it replays the responses of a kept one.
async fn assert_released(client: &SparkConnectClient, operation_id: &str) {
    for _ in 0..100 {
        let request = proto::ReattachExecuteRequest {
            session_id: client.session_id().to_string(),
            user_context: Some(proto::UserContext::default()),
            operation_id: operation_id.to_string(),
            ..Default::default()
        };
        let error = match client.reattach_execute(request).await {
            Err(e) => e.to_string(),
            Ok(mut stream) => match stream.message().await {
                Err(status) => status.message().to_string(),
                Ok(_) => String::new(),
            },
        };
        if error.contains("OPERATION_NOT_FOUND") {
            return;
        }
        tokio::time::sleep(Duration::from_millis(100)).await;
    }
    panic!("The server still has operation {operation_id}");
}

#[test]
fn test_range_collect() {
    if !should_run() {
        println!("Skipping test_range_collect - set SPARK_REMOTE to run");
        return;
    }

    let remote_url =
        std::env::var("SPARK_REMOTE").unwrap_or_else(|_| "sc://localhost:15002".to_string());
    let session = SparkSession::builder()
        .remote(&remote_url)
        .get_or_create()
        .expect("Failed to create session");

    let df = session.range(5).expect("Failed to create range DataFrame");
    let rows = df.collect().expect("Failed to collect rows");

    assert_eq!(rows.len(), 5, "Expected 5 rows from range(5)");
    for (i, row) in rows.iter().enumerate() {
        let id_value = row.get(0).expect("Row should have at least one field");
        match id_value {
            Value::Long(id) => {
                assert_eq!(*id, i as i64, "Expected id to be {}", i);
            }
            _ => panic!("Expected Long value, got {:?}", id_value),
        }
    }
}

#[test]
fn test_filter_collect() {
    if !should_run() {
        println!("Skipping test_filter_collect - set SPARK_REMOTE to run");
        return;
    }

    let remote_url =
        std::env::var("SPARK_REMOTE").unwrap_or_else(|_| "sc://localhost:15002".to_string());
    let session = SparkSession::builder()
        .remote(&remote_url)
        .get_or_create()
        .expect("Failed to create session");

    // range(10).filter(id > 7) must return exactly rows 8 and 9 - this exercises
    // the Filter relation end-to-end, not just range.
    let df = session
        .range(10)
        .expect("Failed to create range DataFrame")
        .filter(col("id").gt(lit(7)));

    let rows = df.collect().expect("Failed to collect rows");

    assert_eq!(
        rows.len(),
        2,
        "range(10).filter(id>7) should yield 2 rows (8,9)"
    );
    let ids: Vec<i64> = rows
        .iter()
        .map(|r| match r.get(0).expect("no id") {
            Value::Long(v) => *v,
            other => panic!("expected Long, got {other:?}"),
        })
        .collect();
    assert_eq!(ids, vec![8, 9], "filter should keep only ids > 7");
}

#[test]
fn test_select_alias() {
    if !should_run() {
        println!("Skipping test_select_alias - set SPARK_REMOTE to run");
        return;
    }

    let remote_url =
        std::env::var("SPARK_REMOTE").unwrap_or_else(|_| "sc://localhost:15002".to_string());
    let session = SparkSession::builder()
        .remote(&remote_url)
        .get_or_create()
        .expect("Failed to create session");

    let df = session
        .range(5)
        .expect("Failed to create range DataFrame")
        .select(vec![col("id").alias("x")]);

    let rows = df.collect().expect("Failed to collect rows");

    assert_eq!(rows.len(), 5, "Expected 5 rows");
    let values: Vec<i64> = rows
        .iter()
        .map(|r| match r.get(0).expect("No x value") {
            Value::Long(val) => *val,
            _ => panic!("Expected Long"),
        })
        .collect();
    assert_eq!(values, vec![0, 1, 2, 3, 4], "Expected 0,1,2,3,4");
}

#[test]
fn test_sql_query() {
    if !should_run() {
        println!("Skipping test_sql_query - set SPARK_REMOTE to run");
        return;
    }

    let remote_url =
        std::env::var("SPARK_REMOTE").unwrap_or_else(|_| "sc://localhost:15002".to_string());
    let session = SparkSession::builder()
        .remote(&remote_url)
        .get_or_create()
        .expect("Failed to create session");

    let df = session
        .sql("SELECT 1 AS a, 'x' AS b")
        .expect("Failed to execute SQL");

    let rows = df.collect().expect("Failed to collect rows");

    assert_eq!(rows.len(), 1, "Expected 1 row from SQL query");
    let row = &rows[0];

    // Just verify we got 2 columns and some data
    assert_eq!(row.len(), 2, "Expected 2 columns");

    // Check field 'b' is a string ('x')
    match row.get(1).expect("No b value") {
        Value::String(b) => assert_eq!(b, "x", "Expected b='x'"),
        _ => panic!("Expected String for b"),
    }
}

#[test]
fn test_groupby_count() {
    if !should_run() {
        println!("Skipping test_groupby_count - set SPARK_REMOTE to run");
        return;
    }

    let remote_url =
        std::env::var("SPARK_REMOTE").unwrap_or_else(|_| "sc://localhost:15002".to_string());
    let session = SparkSession::builder()
        .remote(&remote_url)
        .get_or_create()
        .expect("Failed to create session");

    // Group by a static column (all rows in one group) and count
    let df = session
        .range(6)
        .expect("Failed to create range DataFrame")
        .group_by(vec![lit(1).alias("k")])
        .count();

    let rows = df.collect().expect("Failed to collect rows");

    assert_eq!(rows.len(), 1, "Expected 1 group");
    // First column is 'k' (value 1), second is count (should be 6)
    match rows[0].get(1).expect("No count") {
        Value::Long(count) => assert_eq!(*count, 6, "Expected count of 6"),
        _ => panic!("Expected Long for count"),
    }
}

#[test]
fn test_count() {
    if !should_run() {
        println!("Skipping test_count - set SPARK_REMOTE to run");
        return;
    }

    let remote_url =
        std::env::var("SPARK_REMOTE").unwrap_or_else(|_| "sc://localhost:15002".to_string());
    let session = SparkSession::builder()
        .remote(&remote_url)
        .get_or_create()
        .expect("Failed to create session");

    let df = session
        .range(100)
        .expect("Failed to create range DataFrame");
    let count = df.count().expect("Failed to count");

    assert_eq!(count, 100, "Expected count of 100");
}

/// The sync API inside a `current_thread` runtime, as the default `#[tokio::test]` is: it blocks
/// the test thread, and the requests carry the tags and the user context extensions of that
/// thread, as they do on a plain thread.
#[tokio::test]
async fn test_sync_api_inside_a_current_thread_runtime() {
    if !should_run() {
        println!(
            "Skipping test_sync_api_inside_a_current_thread_runtime - set SPARK_REMOTE to run"
        );
        return;
    }

    let session = SparkSession::builder()
        .remote(&remote_url())
        .get_or_create()
        .expect("Failed to create session");

    // The requests as the client sends them: their tags and the types of their extensions.
    type SentRequests = Vec<(Vec<String>, Vec<String>)>;
    let sent: std::sync::Arc<std::sync::Mutex<SentRequests>> = Default::default();
    let recorded = std::sync::Arc::clone(&sent);
    session.set_execute_plan_hook(Some(std::sync::Arc::new(move |request| {
        let extensions = request
            .user_context
            .iter()
            .flat_map(|context| context.extensions.iter().map(|e| e.type_url.clone()))
            .collect();
        recorded
            .lock()
            .unwrap()
            .push((request.tags.clone(), extensions));
        Ok(request)
    })));

    let df = session.range(3).expect("Failed to create range DataFrame");
    assert_eq!(df.count().expect("count"), 3);
    assert_eq!(df.collect().expect("collect").len(), 3);
    let rows = session
        .sql("SELECT 1 AS x")
        .and_then(|df| df.collect())
        .expect("sql");
    assert_eq!(rows.len(), 1);

    // A tag and an extension of the test thread are on the operation that the test thread runs,
    // which a thread that waits for the server to show it interrupts by the tag.
    let tag = "s13-current-thread-tag";
    let extension_type = "type.googleapis.com/s13.current_thread";
    session.add_tag(tag).expect("add_tag");
    assert_eq!(session.get_tags(), vec![tag.to_string()]);
    session.add_threadlocal_user_context_extension(prost_types::Any {
        type_url: extension_type.to_string(),
        value: Vec::new(),
    });
    let running = std::sync::atomic::AtomicBool::new(true);
    let interrupted = std::thread::scope(|scope| {
        let interrupter = scope.spawn(|| {
            let deadline = std::time::Instant::now() + Duration::from_secs(60);
            while running.load(std::sync::atomic::Ordering::SeqCst) {
                let ids = session.interrupt_tag(tag).expect("interrupt_tag");
                if !ids.is_empty() {
                    return ids;
                }
                assert!(
                    std::time::Instant::now() < deadline,
                    "the operation never started"
                );
                std::thread::sleep(Duration::from_millis(200));
            }
            Vec::new()
        });
        let outcome = session
            .sql("SELECT reflect('java.lang.Thread', 'sleep', 60000L)")
            .and_then(|df| df.collect());
        running.store(false, std::sync::atomic::Ordering::SeqCst);
        assert!(outcome.is_err(), "the interrupted operation must fail");
        interrupter.join().expect("interrupter")
    });
    assert_eq!(
        interrupted.len(),
        1,
        "the tag of the test thread is on its operation"
    );
    {
        let sent = sent.lock().unwrap();
        assert!(
            sent.iter().any(|(tags, exts)| tags == &[tag.to_string()]
                && exts.iter().any(|e| e == extension_type)),
            "the operation carries the tag and the extension of the test thread: {sent:?}"
        );
    }
    session.set_execute_plan_hook(None);
    session.remove_tag(tag);
    assert!(session.get_tags().is_empty());

    // stop() releases the session from this thread too; the stopped session then raises
    // NO_ACTIVE_SESSION, and a new session works.
    session.stop().expect("stop");
    assert!(session.is_stopped());
    let err = session
        .range(1)
        .and_then(|df| df.count())
        .expect_err("a stopped session must not run");
    assert_eq!(err.get_condition().as_deref(), Some("NO_ACTIVE_SESSION"));
    let fresh = SparkSession::builder()
        .remote(&remote_url())
        .get_or_create()
        .expect("Failed to create a new session");
    assert_eq!(fresh.range(2).and_then(|df| df.count()).expect("count"), 2);
    fresh.stop().expect("stop");
}

#[test]
fn test_schema() {
    if !should_run() {
        println!("Skipping test_schema - set SPARK_REMOTE to run");
        return;
    }

    let remote_url =
        std::env::var("SPARK_REMOTE").unwrap_or_else(|_| "sc://localhost:15002".to_string());
    let session = SparkSession::builder()
        .remote(&remote_url)
        .get_or_create()
        .expect("Failed to create session");

    let df = session.range(3).expect("Failed to create range DataFrame");
    let schema = df.schema().expect("Failed to get schema");

    match schema {
        DataType::Struct { fields } => {
            assert_eq!(fields.len(), 1, "Expected 1 field");
            assert_eq!(fields[0].name, "id", "Expected field name 'id'");
            assert_eq!(fields[0].data_type, DataType::Long, "Expected Long type");
        }
        _ => panic!("Expected Struct schema"),
    }
}

#[test]
fn test_decimal_round() {
    if !should_run() {
        println!("Skipping test_decimal_round - set SPARK_REMOTE to run");
        return;
    }

    let remote_url =
        std::env::var("SPARK_REMOTE").unwrap_or_else(|_| "sc://localhost:15002".to_string());
    let session = SparkSession::builder()
        .remote(&remote_url)
        .get_or_create()
        .expect("Failed to create session");

    // Test: SELECT ROUND(3.14159, 2) AS v should return 3.14, not 3
    // Using pure SQL first
    let df = session
        .sql("SELECT ROUND(3.14159, 2) AS v")
        .expect("Failed to execute SQL");

    let rows = df.collect().expect("Failed to collect rows");

    assert_eq!(rows.len(), 1, "Expected 1 row");
    let row = &rows[0];

    // A SQL decimal literal (3.14159) is DECIMAL in Spark, so ROUND(.., 2) yields
    // DECIMAL(4,2)=3.14 - not DOUBLE. Accept either, checking the numeric value.
    match row.get(0).expect("No value") {
        Value::Double(d) => {
            assert!((d - 3.14).abs() < 0.01, "Expected ~3.14, got {}", d);
        }
        Value::Decimal { value, .. } => {
            let d: f64 = value.parse().expect("decimal value parses as f64");
            assert!((d - 3.14).abs() < 0.01, "Expected ~3.14, got {}", value);
        }
        other => panic!("Expected Double/Decimal, got {:?}", other),
    }

    // Also test with lit(3.14159) to ensure literal decoding works
    let df2 = session
        .sql("SELECT 3.14159 AS v")
        .expect("Failed to execute SQL");

    let rows2 = df2.collect().expect("Failed to collect rows");
    assert_eq!(rows2.len(), 1, "Expected 1 row");
    let row2 = &rows2[0];

    // The SQL literal 3.14159 decodes as DECIMAL(6,5) in Spark (not DOUBLE).
    match row2.get(0).expect("No value") {
        Value::Double(d) => {
            assert!(
                (d - 3.14159).abs() < 0.00001,
                "Expected ~3.14159, got {}",
                d
            );
        }
        Value::Decimal { value, .. } => {
            let d: f64 = value.parse().expect("decimal value parses as f64");
            assert!(
                (d - 3.14159).abs() < 0.00001,
                "Expected ~3.14159, got {}",
                value
            );
        }
        other => panic!("Expected Double/Decimal, got {:?}", other),
    }
}

#[test]
fn test_string_concat() {
    if !should_run() {
        println!("Skipping test_string_concat - set SPARK_REMOTE to run");
        return;
    }

    let remote_url =
        std::env::var("SPARK_REMOTE").unwrap_or_else(|_| "sc://localhost:15002".to_string());
    let session = SparkSession::builder()
        .remote(&remote_url)
        .get_or_create()
        .expect("Failed to create session");

    // Test: SELECT 'a' AS a, 'b' AS b -> CONCAT(a, b) should return 'ab', not empty
    let df = session
        .sql("SELECT CONCAT('a', 'b') AS result")
        .expect("Failed to execute SQL");

    let rows = df.collect().expect("Failed to collect rows");

    assert_eq!(rows.len(), 1, "Expected 1 row");
    let row = &rows[0];

    match row.get(0).expect("No value") {
        Value::String(s) => {
            assert_eq!(s, "ab", "CONCAT('a', 'b') should return 'ab', got '{}'", s);
        }
        other => panic!("Expected String, got {:?}", other),
    }
}

#[test]
fn test_cache_persist() {
    if !should_run() {
        println!("Skipping test_cache_persist - set SPARK_REMOTE to run");
        return;
    }

    let remote_url =
        std::env::var("SPARK_REMOTE").unwrap_or_else(|_| "sc://localhost:15002".to_string());
    let session = SparkSession::builder()
        .remote(&remote_url)
        .get_or_create()
        .expect("Failed to create session");

    // Verify the cache()/persist() CLIENT API: the AnalyzePlan.Persist round-trips
    // and returns a usable DataFrame. We intentionally do NOT assert on server-side
    // block *materialization* here: on a local single-JVM Spark 4.2.0 server the
    // BlockManager rejects the persisted block with "StorageLevel is null or invalid"
    // for EVERY Spark Connect client (the reference pyspark client included) — it's a
    // server/local-mode quirk, not a client behavior, so asserting it would test the
    // environment rather than this code.
    use spark_connect::StorageLevelExt;

    // cache() = persist(MEMORY_AND_DISK_DESER): must round-trip.
    let cached = session
        .range(5)
        .expect("Failed to create range DataFrame")
        .cache()
        .expect("cache() should round-trip the Persist AnalyzePlan");

    // Its storage level should read back as the level cache() requested.
    let level = cached
        .storage_level()
        .expect("storage_level() should round-trip");
    assert!(
        level.use_memory && level.use_disk,
        "cache() level should be MEMORY_AND_DISK(_DESER); got {level:?}"
    );

    // persist() with an explicit preset: must round-trip too, and unpersist() cleans up.
    let persisted = session
        .range(3)
        .expect("Failed to create range DataFrame")
        .persist(spark_connect::StorageLevel::memory_and_disk())
        .expect("persist(MEMORY_AND_DISK) should round-trip the Persist AnalyzePlan");
    persisted
        .unpersist(true)
        .expect("unpersist() should round-trip");
    // NOTE: we deliberately don't collect() a persisted relation here. On a local
    // single-JVM Spark 4.2.0 server the BlockManager rejects materializing the cached
    // block ("StorageLevel is null or invalid") for EVERY Spark Connect client,
    // reference pyspark included — a server/local-mode quirk that also poisons later
    // execution of the same relation. This test therefore validates the client-side
    // cache/persist/unpersist/storage_level round-trips, which is the client's job.
}

#[test]
fn test_with_watermark() {
    if !should_run() {
        println!("Skipping test_with_watermark - set SPARK_REMOTE to run");
        return;
    }

    let remote_url =
        std::env::var("SPARK_REMOTE").unwrap_or_else(|_| "sc://localhost:15002".to_string());
    let session = SparkSession::builder()
        .remote(&remote_url)
        .get_or_create()
        .expect("Failed to create session");

    // Create a DataFrame with a timestamp and add watermark
    let df = session
        .sql("SELECT CURRENT_TIMESTAMP AS ts, 1 AS value")
        .expect("Failed to execute SQL")
        .with_watermark("ts", "10 seconds");

    // Just verify that the method doesn't fail
    let _schema = df.schema().expect("Failed to get schema");
}

#[test]
fn test_to_arrow_full_path() {
    // Full server -> collect -> Arrow IPC path for DataFrame::to_arrow (the
    // conversion logic itself is unit-tested in dataframe.rs::conversion_tests).
    if !should_run() {
        println!("Skipping test_to_arrow_full_path - set SPARK_REMOTE to run");
        return;
    }
    use arrow::ipc::reader::FileReader;
    use std::io::Cursor;

    let remote_url =
        std::env::var("SPARK_REMOTE").unwrap_or_else(|_| "sc://localhost:15002".to_string());
    let session = SparkSession::builder()
        .remote(&remote_url)
        .get_or_create()
        .expect("Failed to create session");

    let ipc = session
        .range(5)
        .expect("range")
        .to_arrow()
        .expect("to_arrow");
    let reader = FileReader::try_new(Cursor::new(ipc), None).expect("valid Arrow IPC file");
    let rows: usize = reader.map(|b| b.expect("batch").num_rows()).sum();
    assert_eq!(rows, 5, "to_arrow must round-trip all 5 rows of range(5)");
}

/// End-to-end guard for SPARK-59037: a single ~5 MiB string cell yields one Arrow
/// batch, and thus one gRPC response message, larger than tonic's 4 MiB default
/// decode cap. Before the client raised `max_decoding_message_size` to 128 MiB this
/// `collect()` failed with a gRPC "message length too large" decode error (the same
/// batch the reference grpcio client returns fine). It must now succeed and return
/// the whole value intact.
#[test]
fn test_collect_batch_larger_than_grpc_default_cap() {
    if !should_run() {
        println!(
            "Skipping test_collect_batch_larger_than_grpc_default_cap - set SPARK_REMOTE to run"
        );
        return;
    }

    let remote_url =
        std::env::var("SPARK_REMOTE").unwrap_or_else(|_| "sc://localhost:15002".to_string());
    let session = SparkSession::builder()
        .remote(&remote_url)
        .get_or_create()
        .expect("Failed to create session");

    // 5 MiB, comfortably above tonic's 4 MiB default receive cap. A single row cannot be
    // split across Arrow batches, so this forces one oversized gRPC response message.
    let n: usize = 5 * 1024 * 1024;
    let df = session
        .sql(&format!("SELECT repeat('x', {n}) AS s"))
        .expect("Failed to build repeat() DataFrame");
    let rows = df
        .collect()
        .expect("collect() of a >4 MiB batch must succeed with the raised gRPC cap");

    assert_eq!(rows.len(), 1, "Expected exactly one row");
    let s = rows[0]
        .get(0)
        .and_then(|v| v.as_str())
        .expect("Expected a string cell");
    assert_eq!(
        s.len(),
        n,
        "Expected the full {n}-byte string to round-trip through collect()"
    );
}

/// The server ends the response stream of a reattachable execution before the result is
/// complete after `spark.connect.execute.reattachable.senderMaxStreamDuration` or
/// `senderMaxStreamSize`, and the client reads the rest by reattaching. CI's server sets a
/// small `senderMaxStreamSize`, so that this result takes several streams.
#[test]
fn test_collect_reads_a_result_across_stream_ends() {
    if !should_run() {
        println!(
            "Skipping test_collect_reads_a_result_across_stream_ends - set SPARK_REMOTE to run"
        );
        return;
    }
    let session = SparkSession::builder()
        .remote(&remote_url())
        .get_or_create()
        .expect("Failed to create session");

    let n: i64 = 300_000;
    let rows = session
        .range(n)
        .expect("Failed to create range DataFrame")
        .collect()
        .expect("Failed to collect rows");

    assert_eq!(rows.len() as i64, n);
    let sum: i64 = rows
        .iter()
        .map(|row| match row.get(0) {
            Some(Value::Long(id)) => *id,
            other => panic!("Expected Long value, got {:?}", other),
        })
        .sum();
    assert_eq!(sum, n * (n - 1) / 2);
}

/// Once its result is complete, the client releases an execution, so that the server does not
/// keep it, with its request and buffered responses, until it is abandoned.
#[test]
fn test_completed_execution_is_released() {
    if !should_run() {
        println!("Skipping test_completed_execution_is_released - set SPARK_REMOTE to run");
        return;
    }
    block_on(async {
        let client = connect_client().await;
        let (request, operation_id) = sql_execute_request(&client, "SELECT 1");
        let mut stream = client
            .execute_plan_reattachable(request)
            .await
            .expect("Failed to execute");
        while stream.message().await.expect("Failed to read").is_some() {}
        assert_released(&client, &operation_id).await;
    });
}

/// A progress handler that panics during an execution of a local iterator panics in the caller,
/// with prefetching as without, instead of ending the rows early without an error.
#[test]
fn test_a_panicking_progress_handler_reaches_the_caller_of_a_local_iterator() {
    if !should_run() {
        return;
    }
    let spark = SparkSession::builder()
        .remote(&remote_url())
        .get_or_create()
        .expect("session");
    // Updates every 10 ms (2 s by default), so the tasks below report progress.
    spark
        .conf()
        .set("spark.connect.progress.reportInterval", "10ms")
        .expect("conf");
    spark.register_progress_handler(|update| {
        if !update.done {
            panic!("a progress handler panics");
        }
    });
    for prefetch in [true, false] {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            spark
                .sql("SELECT id, reflect('java.lang.Thread', 'sleep', 500L) AS s FROM range(0, 4, 1, 4)")
                .expect("sql")
                .to_local_iterator(prefetch)
                .expect("iterator")
                .collect::<Result<Vec<_>, _>>()
                .map(|rows| rows.len())
        }));
        assert!(
            result.is_err(),
            "prefetch={prefetch}: the panic of the handler must reach the caller, got {result:?}"
        );
    }
    let _ = spark.stop();
}

/// A stream dropped before its result is complete releases its execution as well.
#[test]
fn test_dropped_stream_releases_its_execution() {
    if !should_run() {
        println!("Skipping test_dropped_stream_releases_its_execution - set SPARK_REMOTE to run");
        return;
    }
    block_on(async {
        let client = connect_client().await;
        let (request, operation_id) =
            sql_execute_request(&client, "SELECT id FROM range(100000000)");
        let mut stream = client
            .execute_plan_reattachable(request)
            .await
            .expect("Failed to execute");
        stream.message().await.expect("Failed to read");
        drop(stream);
        assert_released(&client, &operation_id).await;
    });
}

#[test]
fn test_day_time_interval_real_server() {
    if !should_run() {
        return;
    }
    let remote = std::env::var("SPARK_REMOTE").unwrap();
    let spark = SparkSession::builder()
        .remote(&remote)
        .get_or_create()
        .unwrap();
    let micros = 93_784_000_005;
    let frame = spark.sql(
        "WITH t AS (SELECT INTERVAL '1 02:03:04.000005' DAY TO SECOND AS d) \
         SELECT d, -d AS negative, make_dt_interval(1, 2, 3, 4.000005) AS made, \
         named_struct('leaf', d) AS nested, array(d, -d, CAST(NULL AS INTERVAL DAY TO SECOND)) AS items, \
         map('k', d) AS mapping FROM t"
    ).unwrap();
    let rows = frame.collect().unwrap();
    assert_eq!(rows.len(), 1);
    let row = &rows[0];
    assert_eq!(row.get(0), Some(&Value::DayTimeInterval(micros)));
    assert_eq!(row.get(1), Some(&Value::DayTimeInterval(-micros)));
    assert_eq!(row.get(2), Some(&Value::DayTimeInterval(micros)));
    let Some(Value::Struct(nested)) = row.get(3) else {
        panic!("expected struct")
    };
    assert_eq!(nested.values(), &[Value::DayTimeInterval(micros)]);
    assert_eq!(
        row.get(4),
        Some(&Value::List(vec![
            Value::DayTimeInterval(micros),
            Value::DayTimeInterval(-micros),
            Value::Null,
        ]))
    );
    assert_eq!(
        row.get(5),
        Some(&Value::Map(vec![(
            Value::String("k".to_string()),
            Value::DayTimeInterval(micros)
        )]))
    );
    for prefetch in [false, true] {
        let streamed: Vec<_> = frame
            .to_local_iterator(prefetch)
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert_eq!(streamed, rows);
    }
}

#[test]
#[allow(clippy::result_large_err)] // The hook returns the client's large SparkError.
fn test_the_server_runs_the_compressed_plans_of_a_session_as_pyspark_sends_them() {
    if !should_run() {
        println!(
            "Skipping test_the_server_runs_the_compressed_plans_of_a_session_as_pyspark_sends_them \
             - set SPARK_REMOTE to run"
        );
        return;
    }
    use proto::plan::compressed_operation::OpType;

    let session = SparkSession::builder()
        .remote(&remote_url())
        .get_or_create()
        .expect("Failed to create session");
    // Before the first plan of the session, with which the client reads the threshold once.
    session
        .conf()
        .set("spark.connect.session.planCompression.threshold", "1000")
        .expect("Failed to set the threshold");
    // The kinds of the compressed operations of the executions, which a hook sees as sent.
    let compressed: std::sync::Arc<std::sync::Mutex<Vec<i32>>> = Default::default();
    let recorded = std::sync::Arc::clone(&compressed);
    session.set_execute_plan_hook(Some(std::sync::Arc::new(move |request| {
        let operation = request.plan.as_ref().and_then(|plan| plan.op_type.as_ref());
        if let Some(proto::plan::OpType::CompressedOperation(operation)) = operation {
            recorded.lock().unwrap().push(operation.op_type);
        }
        Ok(request)
    })));

    // A string literal of 12000 bytes makes each plan of it larger than the threshold.
    let text = "Apache Spark".repeat(1000);
    // A command, the statement.
    let df = session
        .sql(&format!("SELECT '{text}' AS s"))
        .expect("Failed to run the statement");
    // An analysis of the relation of the statement.
    let schema = df.schema().expect("Failed to analyze");
    assert_eq!(schema.field_names().expect("field names"), ["s"]);
    // A query of it.
    let rows = df.collect().expect("Failed to collect");
    assert_eq!(rows.len(), 1);
    assert_eq!(rows[0].get(0), Some(&Value::String(text.clone())));
    // A command of it, which is not SQL.
    df.create_or_replace_temp_view("compressed_plan_view")
        .expect("Failed to create the view");
    let view = session
        .sql("SELECT * FROM compressed_plan_view WHERE length(s) = 12000")
        .expect("Failed to query the view");
    assert_eq!(view.count().expect("Failed to count the view"), 1);

    let kinds = compressed.lock().unwrap().clone();
    assert!(
        kinds.contains(&(OpType::Command as i32)) && kinds.contains(&(OpType::Relation as i32)),
        "the commands and the query over the threshold are sent compressed: {kinds:?}"
    );
    session.set_execute_plan_hook(None);
    session.stop().expect("stop");
}
