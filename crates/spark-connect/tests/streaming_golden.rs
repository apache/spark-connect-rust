//! Unit tests for Structured Streaming plan construction.
//!
//! These tests verify that streaming DataStreamReader and DataStreamWriter
//! build correct proto structures. They don't require a remote connection.

use std::collections::HashMap;

use spark_connect::plan::LogicalPlan;
use spark_connect::readwriter::ReadType;
use spark_connect::{DataFrame, SparkSession};
use spark_connect_core::error::SparkErrorKind;
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
    // As PySpark's `xml(path, rowTag="r")`, which is `load(path=path, format="xml")` after its
    // options: `format: "xml" schema: "" options { rowTag: "r" } paths: "data/s"`, streaming.
    let xml = stream_data_source(&spark.read_stream().option("rowTag", "r").xml("data/s"));
    assert_eq!(
        (xml.format.as_deref(), xml.schema.as_deref()),
        (Some("xml"), Some(""))
    );
    let row_tag = HashMap::from([("rowTag".to_string(), "r".to_string())]);
    assert_eq!(
        (xml.options, xml.paths),
        (row_tag, vec!["data/s".to_string()])
    );
    let xml = stream_data_source(&spark.read_stream().schema("a INT").xml("data/s"));
    assert_eq!(xml.schema.as_deref(), Some("a INT"));
}

/// As PySpark's `DataStreamReader.load`, which raises `VALUE_NOT_NON_EMPTY_STR` of a path that
/// is empty once `strip()` takes Python's whitespace off, U+001C to U+001F too, before any
/// request, and so do its format methods, which are `load(path=path, format=...)`: each request of
/// the DataFrame fails with it, after the error of a name that the reader refused, which PySpark
/// raises at `name`.
#[test]
fn stream_reads_of_an_empty_path_fail_as_pyspark() {
    let spark = SparkSession::builder()
        .remote("sc://127.0.0.1:1")
        .get_or_create()
        .unwrap();
    let reader = || spark.read_stream().format("rate");
    for path in ["", "  ", "\t\n", "\u{1c}", " \u{1f}\u{3000}"] {
        for df in [
            reader().load(Some(path)),
            reader().json(path),
            reader().parquet(path),
            reader().csv(path),
            reader().orc(path),
            reader().text(path),
            reader().xml(path),
        ] {
            let error = df.to_relation().unwrap_err();
            assert_eq!(error.kind, SparkErrorKind::ValueError);
            assert_eq!(
                error.message(),
                format!(
                    "[VALUE_NOT_NON_EMPTY_STR] Value for `path` must be a non-empty string, got \
                     '{path}'."
                )
            );
        }
        let error = reader()
            .name("bad name")
            .load(Some(path))
            .to_relation()
            .unwrap_err();
        assert_eq!(
            error.get_condition().as_deref(),
            Some("INVALID_STREAMING_SOURCE_NAME")
        );
    }
    for path in [" a ", "\u{200b}", "."] {
        let source = stream_data_source(&reader().load(Some(path)));
        assert_eq!(source.paths, vec![path.to_string()]);
        assert_eq!(
            stream_data_source(&reader().json(path)).paths,
            vec![path.to_string()]
        );
    }
    assert!(stream_data_source(&reader().load(None)).paths.is_empty());
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

/// As PySpark's `DataStreamReader.name`, which raises `INVALID_STREAMING_SOURCE_NAME` of a name
/// that `re.match(r"^[a-zA-Z0-9_]+$", source_name)` does not match, before the reader is used: each
/// DataFrame of the reader fails with it, from the first such name on, and a name of a final
/// newline, before which `$` matches, is sent.
#[test]
fn stream_reader_names_are_checked_as_pyspark() {
    let spark = SparkSession::builder()
        .remote("sc://127.0.0.1:1")
        .get_or_create()
        .unwrap();
    for name in ["ok_1", "Ab9", "ok\n"] {
        let df = spark.read_stream().format("rate").name(name).load(None);
        assert_eq!(stream_data_source(&df).source_name.as_deref(), Some(name));
    }
    for name in ["bad name", "a-b", "", "\n", "ok\n\n", "é"] {
        let refused = || spark.read_stream().format("rate").name(name);
        for df in [
            refused().load(None),
            refused().name("ok").load(None),
            refused().table("t"),
            refused().changes("t"),
            refused().json("p"),
        ] {
            let error = df.to_relation().unwrap_err();
            assert_eq!(error.kind, SparkErrorKind::ValueError);
            assert_eq!(
                error.message(),
                format!(
                    "[INVALID_STREAMING_SOURCE_NAME] Invalid streaming source name '{name}'. \
                     Source names must contain only ASCII letters, digits, and underscores."
                )
            );
        }
    }
}

/// As PySpark's `DataFrameReader.changes` and `DataStreamReader.changes`, which raise an
/// `AnalysisException` of `_LEGACY_ERROR_TEMP_1189` for a reader of a schema (`if self._schema:`,
/// so not of the schema `""`): each request of the DataFrame fails with it.
#[test]
fn changes_of_a_reader_of_a_schema_fail_as_pyspark() {
    let spark = SparkSession::builder()
        .remote("sc://127.0.0.1:1")
        .get_or_create()
        .unwrap();
    for df in [
        spark.read().schema("a INT".to_string()).changes("t"),
        spark.read_stream().schema("a INT").changes("t"),
    ] {
        let error = df.to_relation().unwrap_err();
        assert_eq!(error.kind, SparkErrorKind::Analysis);
        assert_eq!(
            error.get_condition().as_deref(),
            Some("_LEGACY_ERROR_TEMP_1189")
        );
        assert_eq!(
            error.message(),
            "[_LEGACY_ERROR_TEMP_1189] User specified schema not supported with `changes`."
        );
        assert_eq!(
            error.get_message_parameters(),
            Some([("operation".to_string(), "changes".to_string())].into())
        );
    }
    for df in [
        spark.read().schema(String::new()).changes("t"),
        spark.read_stream().schema("").changes("t"),
    ] {
        let Some(proto::relation::RelType::RelationChanges(changes)) =
            df.to_relation().unwrap().rel_type
        else {
            panic!("expected the changes of the table");
        };
        assert_eq!(changes.unparsed_identifier, "t");
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
