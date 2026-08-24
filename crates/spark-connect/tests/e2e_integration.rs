//! End-to-end integration tests against a live Spark Connect server.
//!
//! These tests connect to a live Spark Connect server at `sc://localhost:15002`
//! and verify the complete execution path: plan building, execution, Arrow IPC decode, and row collection.
//!
//! Run with: `SPARK_REMOTE=sc://localhost:15002 cargo test --test e2e_integration`
//! Or simply skip with: `cargo test --test e2e_integration` (will skip all tests)

use spark_connect::column::{col, lit};
use spark_connect::functions;
use spark_connect::row::Value;
use spark_connect::session::SparkSession;
use spark_connect::types::DataType;

fn should_run() -> bool {
    std::env::var("SPARK_REMOTE").is_ok()
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

    match row.get(0).expect("No value") {
        Value::Double(d) => {
            // 3.14159 rounded to 2 decimals should be approximately 3.14
            assert!((d - 3.14).abs() < 0.01, "Expected ~3.14, got {}", d);
        }
        other => panic!("Expected Double, got {:?}", other),
    }

    // Also test with lit(3.14159) to ensure literal decoding works
    let df2 = session
        .sql("SELECT 3.14159 AS v")
        .expect("Failed to execute SQL");

    let rows2 = df2.collect().expect("Failed to collect rows");
    assert_eq!(rows2.len(), 1, "Expected 1 row");
    let row2 = &rows2[0];

    match row2.get(0).expect("No value") {
        Value::Double(d) => {
            // The literal 3.14159 should decode as approximately 3.14159
            assert!(
                (d - 3.14159).abs() < 0.00001,
                "Expected ~3.14159, got {}",
                d
            );
        }
        other => panic!("Expected Double, got {:?}", other),
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

    // Test cache() and persist()
    let df = session
        .range(5)
        .expect("Failed to create range DataFrame")
        .cache();

    let rows = df
        .collect()
        .expect("Failed to collect from cached DataFrame");
    assert_eq!(rows.len(), 5, "Expected 5 rows from cached DataFrame");

    let df2 = session
        .range(3)
        .expect("Failed to create range DataFrame")
        .persist();

    let rows2 = df2
        .collect()
        .expect("Failed to collect from persisted DataFrame");
    assert_eq!(rows2.len(), 3, "Expected 3 rows from persisted DataFrame");
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
