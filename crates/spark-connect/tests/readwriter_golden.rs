//! Golden parity test for read operations, and the protos of the read relations and the save
//! enums of the Rust plan.
//!
//! The reads of the Rust API's `DataFrameReader` must serialize to the exact same protobuf that
//! the reference PySpark client produces. The goldens live in `tests/golden/readwriter.jsonl`
//! (captured by `scripts/capture_readwriter_golden.py`, base64-encoded `spark.connect.Plan`), whose
//! plan IDs and origins are cleared, as they are here. Writes are commands, which the integration
//! tests run against a server.

#![allow(dead_code)] // golden-test helpers are copy-shared across suites; each binary uses a subset

mod common;

use std::collections::HashMap;

use spark_connect::plan::LogicalPlan;
use spark_connect::readwriter::{ReadType, SaveMode, TableSaveMethod};
use spark_connect::{DataFrame, SparkSession};
use spark_connect_proto as proto;

/// The read relation of each case of the capture, built by the Rust API's reader as PySpark's
/// reader builds it: byte for byte, but the plan ID and origin of its `common`. Of a reader of no
/// schema, PySpark sends the schema `""`, and its `text` sets `wholetext` to `false`.
#[test]
fn reads_match_official_pyspark() {
    let goldens = common::load_goldens("readwriter.jsonl", |plan: &mut proto::Plan| {
        if let Some(proto::plan::OpType::Root(root)) = &mut plan.op_type {
            root.common = None;
        }
    });
    // The channel connects lazily: building plans needs no server.
    let spark = SparkSession::builder()
        .remote("sc://127.0.0.1:1")
        .get_or_create()
        .unwrap();
    let read = || spark.read();
    let cases: Vec<(&str, DataFrame)> = vec![
        ("read_json", read().json("data/test.json")),
        ("read_parquet", read().parquet("data/test.parquet")),
        ("read_csv", read().csv("data/test.csv")),
        (
            "read_csv_with_option",
            read().option("header", "true").csv("data/test.csv"),
        ),
        ("read_orc", read().orc("data/test.orc")),
        ("read_text", read().text("data/test.txt")),
        ("read_table", read().table("my_table")),
        (
            "read_parquet_with_option",
            read()
                .option("mergeSchema", "true")
                .parquet("data/test.parquet"),
        ),
        (
            "read_json_with_schema",
            read()
                .schema("id INT, name STRING".to_string())
                .json("data/test.json"),
        ),
        (
            "read_csv_with_multiple_options",
            read()
                .option("header", "true")
                .option("sep", ",")
                .csv("data/test.csv"),
        ),
    ];
    assert_eq!(cases.len(), goldens.len(), "a case of each golden");
    for (name, df) in cases {
        let Some(proto::plan::OpType::Root(expected)) = &goldens[name].op_type else {
            panic!("{name}: the golden has no root relation");
        };
        let mut relation = df.to_relation().unwrap();
        relation.common = None;
        assert_eq!(&relation, expected, "{name}");
    }
    // PySpark's `text(paths)` sets its `wholetext=False` over an earlier option, as its
    // `_set_opts` sets every named option that is not None.
    let mut over = read()
        .option("wholetext", "true")
        .text("data/test.txt")
        .to_relation()
        .unwrap();
    over.common = None;
    let Some(proto::plan::OpType::Root(expected)) = &goldens["read_text"].op_type else {
        panic!("read_text: the golden has no root relation");
    };
    assert_eq!(&over, expected, "read_text over an earlier wholetext");
}

#[test]
fn test_read_json() {
    // Test reading JSON format
    let read_type = ReadType::DataSource {
        format: Some("json".to_string()),
        schema: None,
        options: HashMap::new(),
        paths: vec!["data/test.json".to_string()],
        predicates: vec![],
        source_name: None,
    };

    let plan = LogicalPlan::Read {
        read_type,
        is_streaming: false,
    };

    let proto = plan.to_proto();
    assert!(proto.rel_type.is_some());
    if let Some(proto::relation::RelType::Read(read)) = proto.rel_type {
        if let Some(proto::read::ReadType::DataSource(ds)) = read.read_type {
            assert_eq!(ds.format, Some("json".to_string()));
            assert_eq!(ds.paths.len(), 1);
            assert_eq!(ds.paths[0], "data/test.json");
        } else {
            panic!("Expected DataSource");
        }
    } else {
        panic!("Expected Read relation");
    }
}

#[test]
fn test_read_parquet() {
    // Test reading Parquet format
    let read_type = ReadType::DataSource {
        format: Some("parquet".to_string()),
        schema: None,
        options: HashMap::new(),
        paths: vec!["data/test.parquet".to_string()],
        predicates: vec![],
        source_name: None,
    };

    let plan = LogicalPlan::Read {
        read_type,
        is_streaming: false,
    };

    let proto = plan.to_proto();
    if let Some(proto::relation::RelType::Read(read)) = proto.rel_type {
        if let Some(proto::read::ReadType::DataSource(ds)) = read.read_type {
            assert_eq!(ds.format, Some("parquet".to_string()));
            assert_eq!(ds.paths.len(), 1);
        } else {
            panic!("Expected DataSource");
        }
    } else {
        panic!("Expected Read relation");
    }
}

#[test]
fn test_read_csv() {
    // Test reading CSV format
    let read_type = ReadType::DataSource {
        format: Some("csv".to_string()),
        schema: None,
        options: HashMap::new(),
        paths: vec!["data/test.csv".to_string()],
        predicates: vec![],
        source_name: None,
    };

    let plan = LogicalPlan::Read {
        read_type,
        is_streaming: false,
    };

    let proto = plan.to_proto();
    if let Some(proto::relation::RelType::Read(read)) = proto.rel_type {
        if let Some(proto::read::ReadType::DataSource(ds)) = read.read_type {
            assert_eq!(ds.format, Some("csv".to_string()));
        } else {
            panic!("Expected DataSource");
        }
    } else {
        panic!("Expected Read relation");
    }
}

#[test]
fn test_read_orc() {
    // Test reading ORC format
    let read_type = ReadType::DataSource {
        format: Some("orc".to_string()),
        schema: None,
        options: HashMap::new(),
        paths: vec!["data/test.orc".to_string()],
        predicates: vec![],
        source_name: None,
    };

    let plan = LogicalPlan::Read {
        read_type,
        is_streaming: false,
    };

    let proto = plan.to_proto();
    if let Some(proto::relation::RelType::Read(read)) = proto.rel_type {
        if let Some(proto::read::ReadType::DataSource(ds)) = read.read_type {
            assert_eq!(ds.format, Some("orc".to_string()));
        } else {
            panic!("Expected DataSource");
        }
    } else {
        panic!("Expected Read relation");
    }
}

#[test]
fn test_read_text() {
    // Test reading text format
    let read_type = ReadType::DataSource {
        format: Some("text".to_string()),
        schema: None,
        options: HashMap::new(),
        paths: vec!["data/test.txt".to_string()],
        predicates: vec![],
        source_name: None,
    };

    let plan = LogicalPlan::Read {
        read_type,
        is_streaming: false,
    };

    let proto = plan.to_proto();
    if let Some(proto::relation::RelType::Read(read)) = proto.rel_type {
        if let Some(proto::read::ReadType::DataSource(ds)) = read.read_type {
            assert_eq!(ds.format, Some("text".to_string()));
        } else {
            panic!("Expected DataSource");
        }
    } else {
        panic!("Expected Read relation");
    }
}

#[test]
fn test_read_with_options() {
    // Test reading with multiple options
    let mut options = HashMap::new();
    options.insert("header".to_string(), "true".to_string());
    options.insert("sep".to_string(), ",".to_string());

    let read_type = ReadType::DataSource {
        format: Some("csv".to_string()),
        schema: None,
        options,
        paths: vec!["data/test.csv".to_string()],
        predicates: vec![],
        source_name: None,
    };

    let plan = LogicalPlan::Read {
        read_type,
        is_streaming: false,
    };

    let proto = plan.to_proto();
    if let Some(proto::relation::RelType::Read(read)) = proto.rel_type {
        if let Some(proto::read::ReadType::DataSource(ds)) = read.read_type {
            assert_eq!(ds.options.get("header"), Some(&"true".to_string()));
            assert_eq!(ds.options.get("sep"), Some(&",".to_string()));
        } else {
            panic!("Expected DataSource");
        }
    } else {
        panic!("Expected Read relation");
    }
}

#[test]
fn test_read_with_schema() {
    // Test reading with schema specification
    let read_type = ReadType::DataSource {
        format: Some("json".to_string()),
        schema: Some("id INT, name STRING".to_string()),
        options: HashMap::new(),
        paths: vec!["data/test.json".to_string()],
        predicates: vec![],
        source_name: None,
    };

    let plan = LogicalPlan::Read {
        read_type,
        is_streaming: false,
    };

    let proto = plan.to_proto();
    if let Some(proto::relation::RelType::Read(read)) = proto.rel_type {
        if let Some(proto::read::ReadType::DataSource(ds)) = read.read_type {
            assert_eq!(ds.schema, Some("id INT, name STRING".to_string()));
        } else {
            panic!("Expected DataSource");
        }
    } else {
        panic!("Expected Read relation");
    }
}

#[test]
fn test_read_table() {
    // Test reading from a named table
    let read_type = ReadType::NamedTable {
        table_name: "my_table".to_string(),
        options: HashMap::new(),
    };

    let plan = LogicalPlan::Read {
        read_type,
        is_streaming: false,
    };

    let proto = plan.to_proto();
    if let Some(proto::relation::RelType::Read(read)) = proto.rel_type {
        if let Some(proto::read::ReadType::NamedTable(nt)) = read.read_type {
            assert_eq!(nt.unparsed_identifier, "my_table");
        } else {
            panic!("Expected NamedTable");
        }
    } else {
        panic!("Expected Read relation");
    }
}

#[test]
fn test_read_jdbc() {
    // Test reading from JDBC with predicates
    let mut options = HashMap::new();
    options.insert("url".to_string(), "jdbc:mysql://localhost/db".to_string());
    options.insert("dbtable".to_string(), "my_table".to_string());

    let predicates = vec!["id > 100".to_string(), "status = 'active'".to_string()];

    let read_type = ReadType::DataSource {
        format: Some("jdbc".to_string()),
        schema: None,
        options,
        paths: vec![],
        predicates,
        source_name: None,
    };

    let plan = LogicalPlan::Read {
        read_type,
        is_streaming: false,
    };

    let proto = plan.to_proto();
    if let Some(proto::relation::RelType::Read(read)) = proto.rel_type {
        if let Some(proto::read::ReadType::DataSource(ds)) = read.read_type {
            assert_eq!(ds.format, Some("jdbc".to_string()));
            assert_eq!(ds.predicates.len(), 2);
            assert_eq!(ds.predicates[0], "id > 100");
        } else {
            panic!("Expected DataSource");
        }
    } else {
        panic!("Expected Read relation");
    }
}

// Write operations are handled as Commands, not Relations, so they are tested separately
// in the integration tests with the actual server.

#[test]
fn test_save_mode_conversions() {
    // Test SaveMode enum conversions
    assert_eq!(SaveMode::Append.to_proto(), 1i32);
    assert_eq!(SaveMode::Overwrite.to_proto(), 2i32);
    assert_eq!(SaveMode::ErrorIfExists.to_proto(), 3i32);
    assert_eq!(SaveMode::Ignore.to_proto(), 4i32);

    assert_eq!(SaveMode::from_str("append"), Some(SaveMode::Append));
    assert_eq!(SaveMode::from_str("overwrite"), Some(SaveMode::Overwrite));
    assert_eq!(SaveMode::from_str("error"), Some(SaveMode::ErrorIfExists));
    assert_eq!(SaveMode::from_str("ignore"), Some(SaveMode::Ignore));
    // As PySpark Connect's WriteOperation, which raises UNSUPPORTED_OPERATION of any other mode.
    assert_eq!(SaveMode::from_str("errorifexists"), None);
}

#[test]
fn test_table_save_method_conversions() {
    // Test TableSaveMethod enum conversions
    assert_eq!(TableSaveMethod::SaveAsTable.to_proto(), 1i32);
    assert_eq!(TableSaveMethod::InsertInto.to_proto(), 2i32);
}
