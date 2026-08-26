//! Server-gated end-to-end coverage of the DataFrame / StatFunctions / SparkSession
//! methods wired up for pyspark parity — temp views, checkpoints, corr/cov/sampleBy,
//! replace, to, withMetadata, randomSplit, inputFiles, isLocal/isCached, exists/scalar,
//! transpose, dropDuplicates, asTable, and session stop/is_stopped. These execute
//! against a live server, covering `dataframe.rs`/`group.rs`/`session.rs` paths that a
//! server-less run reads as 0%.
//!
//! Run with: `SPARK_REMOTE=sc://localhost:15002 cargo test --test e2e_new_methods`

use std::collections::HashMap;

use spark_connect::column::{col, lit};
use spark_connect::session::SparkSession;
use spark_connect::types::DataType;

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

fn numeric(s: &SparkSession) -> spark_connect::dataframe::DataFrame {
    s.sql("SELECT * FROM VALUES (0,1.0),(1,2.0),(1,3.0),(2,4.0),(3,5.0) AS t(g, v)")
        .expect("numeric df")
}

#[test]
fn temp_views_and_register() {
    if !should_run() {
        return;
    }
    let spark = session();
    let df = spark.range_full(0, 5, 1, None).expect("range");

    df.create_temp_view("e2e_tv").expect("create_temp_view");
    let n = spark
        .sql("SELECT count(*) AS c FROM e2e_tv")
        .expect("sql")
        .collect()
        .expect("collect");
    assert_eq!(n.len(), 1);

    df.create_or_replace_global_temp_view("e2e_gtv")
        .expect("create_or_replace_global_temp_view");
    let _ = spark
        .sql("SELECT count(*) FROM global_temp.e2e_gtv")
        .and_then(|d| d.collect());
    // create_global_temp_view again should error (already exists) — exercise the err path.
    let _ = df.create_global_temp_view("e2e_gtv");

    df.register_temp_table("e2e_rtt").expect("register_temp_table");
    let _ = spark.sql("SELECT * FROM e2e_rtt").and_then(|d| d.collect());
}

#[test]
fn checkpoints_and_input_files_local() {
    if !should_run() {
        return;
    }
    let spark = session();
    let df = spark.range_full(0, 10, 1, None).expect("range");

    let lc = df.local_checkpoint().expect("local_checkpoint");
    assert_eq!(lc.collect().expect("collect lc").len(), 10);

    // inputFiles on a computed relation is empty; is_local is false for a range.
    assert!(df.input_files().expect("input_files").is_empty());
    assert!(!df.is_local());
    let _ = df.is_cached().expect("is_cached");
}

#[test]
fn stat_corr_cov_sample_by() {
    if !should_run() {
        return;
    }
    let spark = session();
    let df = numeric(&spark);

    let _c = df.stat().corr("g", "v").expect("corr");
    let _v = df.stat().cov("g", "v").expect("cov");

    let fractions = vec![
        (lit(0).expression().clone(), 1.0),
        (lit(1).expression().clone(), 0.5),
    ];
    let sampled = df
        .stat()
        .sample_by("g", fractions, Some(42))
        .collect()
        .expect("sample_by collect");
    // Non-deterministic count, but the RPC must succeed and return rows-or-none.
    assert!(sampled.len() <= 5);
}

#[test]
fn replace_to_with_metadata_random_split() {
    if !should_run() {
        return;
    }
    let spark = session();
    let df = numeric(&spark);

    // replace values in the "g" column.
    let replaced = df
        .replace(vec![("0".to_string(), "9".to_string())], Some(vec!["g"]))
        .collect()
        .expect("replace collect");
    assert_eq!(replaced.len(), 5);

    // to(): reconcile to a target schema (g -> string).
    let schema = DataType::from_ddl("g string, v double").expect("ddl");
    let toed = df.to(schema).collect().expect("to collect");
    assert_eq!(toed.len(), 5);

    // withMetadata attaches column metadata (lazy; execute to exercise the plan).
    let mut md = HashMap::new();
    md.insert("comment".to_string(), "the group".to_string());
    let with_md = df.with_metadata("g", md);
    assert_eq!(with_md.collect().expect("with_metadata collect").len(), 5);

    // randomSplit returns multiple DataFrames whose row counts sum to the input.
    let parts = df.random_split(vec![0.5, 0.5], Some(1));
    let total: usize = parts
        .iter()
        .map(|p| p.clone().collect().expect("split collect").len())
        .sum();
    assert_eq!(total, 5);
}

#[test]
fn exists_scalar_transpose_dedup_astable() {
    if !should_run() {
        return;
    }
    let spark = session();

    // scalar()/exists() on a single-value subquery-shaped DataFrame.
    let one = spark.sql("SELECT 1 AS x").expect("sql one");
    assert!(one.exists().expect("exists"));
    let _v = spark
        .sql("SELECT 42 AS x")
        .expect("sql")
        .scalar()
        .expect("scalar");

    // dropDuplicates on a subset.
    let df = numeric(&spark);
    let deduped = df
        .drop_duplicates(Some(vec!["g"]))
        .collect()
        .expect("dedup collect");
    assert!(deduped.len() <= 5);

    // asTable produces a usable relation alias.
    let _ = df.as_table("t_alias");

    // transpose the small df (server-side pivot of rows/cols).
    let _ = spark
        .sql("SELECT * FROM VALUES ('a', 1), ('b', 2) AS t(k, n)")
        .expect("sql")
        .transpose();
}

#[test]
fn session_stop_marks_is_stopped() {
    if !should_run() {
        return;
    }
    // Use a fresh session so stopping it doesn't affect other tests.
    let url = std::env::var("SPARK_REMOTE").unwrap_or_else(|_| "sc://localhost:15002".to_string());
    let spark = SparkSession::builder().remote(&url).get_or_create().expect("session");
    let side = spark.new_session();
    assert!(!side.is_stopped());
    side.stop().expect("stop");
    assert!(side.is_stopped());
}

#[test]
fn with_watermark_builds_plan() {
    if !should_run() {
        return;
    }
    let spark = session();
    // with_watermark is lazy; build a plan on a timestamp column (no execution needed
    // to cover the builder). Use a batch df with a timestamp column.
    let df = spark
        .sql("SELECT CAST(id AS TIMESTAMP) AS ts, id FROM range(3)")
        .expect("ts df");
    let _wm = df.with_watermark("ts", "10 minutes");
    let _ = col("ts"); // keep the import used
}
