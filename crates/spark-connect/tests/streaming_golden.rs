//! Unit tests for Structured Streaming plan construction.
//!
//! These tests verify that streaming DataStreamReader and DataStreamWriter
//! build correct proto structures. They don't require a remote connection.

use std::collections::HashMap;

use spark_connect::plan::LogicalPlan;
use spark_connect::readwriter::ReadType;
use spark_connect::{DataFrame, SparkSession};
use spark_connect_proto as proto;

/// The data source of the streaming read of `df`.
fn stream_data_source(df: &DataFrame) -> proto::read::DataSource {
    match df.to_relation().unwrap().rel_type {
        Some(proto::relation::RelType::Read(read)) => {
            assert!(read.is_streaming);
            match read.read_type {
                Some(proto::read::ReadType::DataSource(data_source)) => data_source,
                other => panic!("not a data source: {other:?}"),
            }
        }
        other => panic!("not a read: {other:?}"),
    }
}

/// The reads of the stream reader, as PySpark 4.3.0's `DataStreamReader` builds them: of a reader
/// of no schema, the schema `""` (its `_schema` is `""`, which its `DataSource` sends), and of
/// `text`, the `wholetext` of its `wholetext=False`, also over an earlier one, which its `option`
/// writes as `str(False)`.
#[test]
fn stream_reads_send_pysparks_schema_and_options() {
    // The channel connects lazily: building plans needs no server.
    let spark = SparkSession::builder()
        .remote("sc://127.0.0.1:1")
        .get_or_create()
        .unwrap();
    let rate = stream_data_source(&spark.read_stream().format("rate").load(None));
    assert_eq!(
        (rate.format.as_deref(), rate.schema.as_deref()),
        (Some("rate"), Some(""))
    );
    assert!(rate.options.is_empty() && rate.paths.is_empty());
    let text = stream_data_source(
        &spark
            .read_stream()
            .option("wholetext", "true")
            .text("data/s"),
    );
    assert_eq!(
        (text.format.as_deref(), text.schema.as_deref()),
        (Some("text"), Some(""))
    );
    let wholetext = HashMap::from([("wholetext".to_string(), "False".to_string())]);
    assert_eq!(
        (text.options, text.paths),
        (wholetext, vec!["data/s".to_string()])
    );
    let csv = stream_data_source(&spark.read_stream().schema("a INT").csv("data/s"));
    assert_eq!(csv.schema.as_deref(), Some("a INT"));
}

/// Verify a basic streaming read plan builds correctly.
#[test]
fn test_stream_read_basic_structure() {
    let plan = LogicalPlan::Read {
        read_type: ReadType::DataSource {
            format: Some("rate".to_string()),
            schema: None,
            options: vec![("rowsPerSecond".to_string(), "10".to_string())]
                .into_iter()
                .collect(),
            paths: vec![],
            predicates: vec![],
            source_name: None,
        },
        is_streaming: true,
    };

    let proto_plan = plan.to_proto();

    // Verify Read relation exists and is marked as streaming
    match &proto_plan.rel_type {
        Some(proto::relation::RelType::Read(read)) => {
            assert!(read.is_streaming, "Read should be marked as streaming");
        }
        _ => panic!("Expected Read relation"),
    }
}

#[test]
fn test_stream_read_json_format() {
    let plan = LogicalPlan::Read {
        read_type: ReadType::DataSource {
            format: Some("json".to_string()),
            schema: None,
            options: Default::default(),
            paths: vec!["s3://bucket/path".to_string()],
            predicates: vec![],
            source_name: None,
        },
        is_streaming: true,
    };

    let proto_plan = plan.to_proto();

    // Verify Read relation has JSON format
    match &proto_plan.rel_type {
        Some(proto::relation::RelType::Read(read)) => {
            assert!(read.is_streaming);
            if let Some(proto::read::ReadType::DataSource(ds)) = &read.read_type {
                assert_eq!(ds.format, Some("json".to_string()));
                assert_eq!(ds.paths.len(), 1);
            } else {
                panic!("Expected DataSource read type");
            }
        }
        _ => panic!("Expected Read relation"),
    }
}

#[test]
fn test_stream_read_with_schema() {
    let plan = LogicalPlan::Read {
        read_type: ReadType::DataSource {
            format: Some("json".to_string()),
            schema: Some("id INT, name STRING, timestamp TIMESTAMP".to_string()),
            options: Default::default(),
            paths: vec!["/data/stream".to_string()],
            predicates: vec![],
            source_name: None,
        },
        is_streaming: true,
    };

    let proto_plan = plan.to_proto();

    // Verify Read relation has schema
    match &proto_plan.rel_type {
        Some(proto::relation::RelType::Read(read)) => {
            if let Some(proto::read::ReadType::DataSource(ds)) = &read.read_type {
                assert_eq!(
                    ds.schema,
                    Some("id INT, name STRING, timestamp TIMESTAMP".to_string())
                );
            }
        }
        _ => panic!("Expected Read relation"),
    }
}

#[test]
fn test_stream_read_named_table() {
    let plan = LogicalPlan::Read {
        read_type: ReadType::NamedTable {
            table_name: "my_table".to_string(),
            options: Default::default(),
        },
        is_streaming: true,
    };

    let proto_plan = plan.to_proto();

    // Verify Read relation is NamedTable and streaming
    match &proto_plan.rel_type {
        Some(proto::relation::RelType::Read(read)) => {
            assert!(read.is_streaming);
            if let Some(proto::read::ReadType::NamedTable(nt)) = &read.read_type {
                assert_eq!(nt.unparsed_identifier, "my_table");
            } else {
                panic!("Expected NamedTable read type");
            }
        }
        _ => panic!("Expected Read relation"),
    }
}

#[test]
fn test_stream_read_with_source_name() {
    let plan = LogicalPlan::Read {
        read_type: ReadType::DataSource {
            format: Some("rate".to_string()),
            schema: None,
            options: Default::default(),
            paths: vec![],
            predicates: vec![],
            source_name: Some("my_rate_source".to_string()),
        },
        is_streaming: true,
    };

    let proto_plan = plan.to_proto();

    // Verify source name is set
    match &proto_plan.rel_type {
        Some(proto::relation::RelType::Read(read)) => {
            if let Some(proto::read::ReadType::DataSource(ds)) = &read.read_type {
                assert_eq!(ds.source_name, Some("my_rate_source".to_string()));
            }
        }
        _ => panic!("Expected Read relation"),
    }
}

#[test]
fn test_stream_read_non_streaming_false() {
    // Verify that non-streaming reads are marked appropriately
    let plan = LogicalPlan::Read {
        read_type: ReadType::DataSource {
            format: Some("parquet".to_string()),
            schema: None,
            options: Default::default(),
            paths: vec!["/path".to_string()],
            predicates: vec![],
            source_name: None,
        },
        is_streaming: false,
    };

    let proto_plan = plan.to_proto();

    // Verify Read relation is NOT marked as streaming
    match &proto_plan.rel_type {
        Some(proto::relation::RelType::Read(read)) => {
            assert!(!read.is_streaming);
        }
        _ => panic!("Expected Read relation"),
    }
}
