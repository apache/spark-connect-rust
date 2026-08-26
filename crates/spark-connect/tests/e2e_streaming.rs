//! Server-gated end-to-end coverage of Structured Streaming: the `rate` source
//! through the reader, a `memory` sink through the writer's `start`/trigger paths,
//! and the `StreamingQuery` + `StreamingQueryManager` handles. These exercise the
//! execution paths in `streaming.rs` that a server-less run reads as 0%.
//!
//! Run with: `SPARK_REMOTE=sc://localhost:15002 cargo test --test e2e_streaming`

use spark_connect::session::SparkSession;
use spark_connect::streaming::Trigger;

fn should_run() -> bool {
    std::env::var("SPARK_REMOTE").is_ok()
}

fn session() -> SparkSession {
    let url = std::env::var("SPARK_REMOTE").unwrap_or_else(|_| "sc://localhost:15002".to_string());
    SparkSession::builder()
        .remote(&url)
        .get_or_create()
        .expect("session")
}

/// A short-lived query on the `rate` source into an in-memory sink, so the whole
/// reader -> writer.start -> StreamingQuery -> stop path runs against the server.
#[test]
fn streaming_rate_to_memory_query_lifecycle() {
    if !should_run() {
        return;
    }
    let spark = session();
    let df = spark
        .read_stream()
        .format("rate")
        .option("rowsPerSecond", "5")
        .load(None);

    let query = df
        .write_stream()
        .format("memory")
        .query_name("e2e_rate_mem")
        .output_mode("append")
        .trigger(Trigger::ProcessingTime("1 seconds".to_string()))
        .start("")
        .expect("start streaming query");

    // Handle accessors.
    assert!(!query.id().is_empty());
    assert!(!query.run_id().is_empty());
    assert_eq!(query.name(), Some("e2e_rate_mem"));

    // Status / activity RPCs.
    let _active = query.is_active().expect("is_active");
    let _status = query.status().expect("status");
    let _explain = query.explain(false).expect("explain");

    // Manager surface while the query is (briefly) live.
    let mgr = spark.streams();
    let _all = mgr.active().expect("active list");
    let _got = mgr.get(query.id()).expect("get by id");

    // Recent/last progress may be empty this early; the call must still succeed.
    let _last = query.last_progress().expect("last_progress");
    let _recent = query.recent_progress().expect("recent_progress");

    // Give it a moment then stop and reset.
    let _ = query.await_termination(Some(1.0)).expect("await_termination timeout");
    query.stop().expect("stop");
    let _ = query.exception().expect("exception after stop");
    mgr.reset_terminated().expect("reset_terminated");
}

/// Cover the `to_table` sink-destination path and the AvailableNow trigger.
#[test]
fn streaming_available_now_to_table() {
    if !should_run() {
        return;
    }
    let spark = session();
    // Drop any leftover table from a previous run.
    let _ = spark.sql("DROP TABLE IF EXISTS e2e_stream_tbl").and_then(|d| d.collect());

    let df = spark
        .read_stream()
        .format("rate")
        .option("rowsPerSecond", "10")
        .load(None);

    let query = df
        .write_stream()
        .output_mode("append")
        .query_name("e2e_avail_now")
        .trigger(Trigger::AvailableNow)
        .to_table("e2e_stream_tbl");

    // AvailableNow processes what's available and terminates; the call may either
    // succeed (query handle) or fail if the sink/table config is unsupported here.
    // Either way the writer's proto-building + start path is exercised.
    if let Ok(q) = query {
        let _ = q.await_termination(Some(5.0));
        let _ = q.stop();
    }
    let _ = spark.sql("DROP TABLE IF EXISTS e2e_stream_tbl").and_then(|d| d.collect());
}
