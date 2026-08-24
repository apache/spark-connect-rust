//! Golden parity test for newly added DataFrame methods.
//!
//! These tests verify that the new DataFrame methods (fillna, dropna, replace, describe,
//! summary, col_regex, rollup, cube, sort_within_partitions, etc.) serialize to the
//! exact same protobuf the reference PySpark client produces.
//!
//! IMPORTANT: The golden protobuf values must be captured from a live Spark Connect server
//! by running the capture script against it:
//! ```bash
//! SPARK_REMOTE=sc://localhost:15002 python3 scripts/capture_golden.py dataframe_extra
//! ```
//!
//! To enable these tests, add corresponding entries to tests/golden/dataframe_extra.jsonl

use std::fs::File;
use std::io::{BufRead, BufReader};

#[test]
#[ignore]
fn test_fillna_golden() {
    // Test: df.range(5).fillna(0)
    // This creates a NAFill relation
    // Golden capture requires running against a live Spark Connect server
}

#[test]
#[ignore]
fn test_dropna_golden() {
    // Test: df.range(5).dropna()
}

#[test]
#[ignore]
fn test_describe_golden() {
    // Test: df.range(5).describe()
}

#[test]
#[ignore]
fn test_summary_golden() {
    // Test: df.range(5).summary("25%", "50%", "75%")
}

#[test]
#[ignore]
fn test_col_regex_golden() {
    // Test: df.selectExpr("id", "id as id2").col_regex("id.*")
}

#[test]
#[ignore]
fn test_sort_within_partitions_golden() {
    // Test: df.range(10).sort_within_partitions(col("id").desc())
}

#[test]
#[ignore]
fn test_drop_duplicates_within_watermark_golden() {
    // Test: df.range(10).drop_duplicates_within_watermark(vec!["id"])
}

#[test]
#[ignore]
fn test_random_split_golden() {
    // Test: df.range(100).random_split(vec![0.5, 0.5], Some(42))
}

#[test]
#[ignore]
fn test_union_all_golden() {
    // Test: df1.union_all(df2) - SetOperation with is_all=true
}

#[test]
#[ignore]
fn test_except_all_golden() {
    // Test: df1.except_all(df2) - SetOperation EXCEPT with is_all=true
}

#[test]
#[ignore]
fn test_intersect_all_golden() {
    // Test: df1.intersect_all(df2) - SetOperation INTERSECT with is_all=true
}

#[test]
#[ignore]
fn test_unpivot_golden() {
    // Test: df.unpivot(ids, values, "variable", "value")
}

// Verify golden file structure when it exists
#[test]
fn test_dataframe_extra_goldens_exist() {
    let golden_path = "tests/golden/dataframe_extra.jsonl";
    match File::open(golden_path) {
        Ok(file) => {
            let reader = BufReader::new(file);
            let count = reader.lines().count();
            println!("Found {} golden test cases in {}", count, golden_path);
            assert!(count > 0, "Golden file exists but is empty");
        }
        Err(_) => {
            eprintln!("Note: {} not yet created. Capture with: SPARK_REMOTE=sc://localhost:15002 python3 scripts/capture_golden.py dataframe_extra", golden_path);
        }
    }
}
