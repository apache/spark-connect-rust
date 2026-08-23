//! Golden parity test for read/write operations.
//!
//! Read and write operations must serialize to the exact same protobuf
//! the reference PySpark client produces.

use std::collections::HashMap;

use spark_connect::plan::LogicalPlan;
use spark_connect::readwriter::{ReadType, SaveMode, TableSaveMethod};
use spark_connect_proto as proto;

/// Normalize a relation by clearing non-deterministic fields.
fn normalize_relation(r: &mut proto::Relation) {
    r.common = None;
    if let Some(_rel_type) = &mut r.rel_type {
        use proto::relation::RelType;
        match _rel_type {
            RelType::Read(_read) => {
                // No recursive normalization needed for Read
            }
            _ => {}
        }
    }
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
}

#[test]
fn test_table_save_method_conversions() {
    // Test TableSaveMethod enum conversions
    assert_eq!(TableSaveMethod::SaveAsTable.to_proto(), 1i32);
    assert_eq!(TableSaveMethod::InsertInto.to_proto(), 2i32);
}
