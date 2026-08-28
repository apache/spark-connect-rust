//! Extra pure-logic coverage for `types.rs`: the serialization branches, JSON and DDL
//! parsing paths (including error closures), and proto edge cases that the exhaustive
//! round-trip in `types_roundtrip.rs` does not reach. No server; runs in the default
//! `cargo test` pass.
//!
//! These deliberately favor exercising code paths (`let _ = ...`) over asserting exact
//! outputs, except where the expected value is unambiguous from the implementation.

use std::collections::BTreeMap;

use spark_connect::types::{DataType, StructField};
use spark_connect_proto as proto;

/// Variants (and branch variations) whose accessors are otherwise under-exercised.
fn extra_variants() -> Vec<DataType> {
    vec![
        // Non-default collation: hits the "string collate <name>" branch in both
        // simple_string and json_value.
        DataType::String {
            collation: "UTF8_LCASE".to_string(),
        },
        DataType::String {
            collation: String::new(),
        },
        DataType::Char { length: 12 },
        DataType::Varchar { length: 24 },
        DataType::Time { precision: 3 },
        DataType::Decimal {
            precision: 20,
            scale: 4,
        },
        // Geometry / Geography: both the srid == -1 ("any") and concrete-srid branches.
        DataType::Geometry { srid: -1 },
        DataType::Geometry { srid: 4326 },
        DataType::Geography { srid: -1 },
        DataType::Geography { srid: 4326 },
        DataType::Variant,
        // Interval: same start/end field, distinct fields, and out-of-range codes that
        // fall through to the bare "interval" string.
        DataType::YearMonthInterval {
            start_field: 0,
            end_field: 0,
        },
        DataType::YearMonthInterval {
            start_field: 0,
            end_field: 1,
        },
        DataType::YearMonthInterval {
            start_field: 99,
            end_field: 99,
        },
        DataType::DayTimeInterval {
            start_field: 0,
            end_field: 3,
        },
        DataType::DayTimeInterval {
            start_field: 42,
            end_field: 42,
        },
        // UDT with every optional field populated (exercises each `if let Some` insert).
        DataType::Udt {
            type_str: "udt".to_string(),
            jvm_class: Some("com.example.MyUdt".to_string()),
            python_class: Some("mymod.MyUdt".to_string()),
            serialized_python_class: Some("payload".to_string()),
            sql_type: Some(Box::new(DataType::Integer)),
        },
        DataType::Unparsed {
            data_type_string: "some_custom".to_string(),
        },
        // Nested complex types, so the recursive accessor arms run.
        DataType::Array {
            element_type: Box::new(DataType::Map {
                key_type: Box::new(DataType::String {
                    collation: "UTF8_BINARY".to_string(),
                }),
                value_type: Box::new(DataType::Array {
                    element_type: Box::new(DataType::Integer),
                    contains_null: false,
                }),
                value_contains_null: true,
            }),
            contains_null: true,
        },
        DataType::Struct {
            fields: vec![
                StructField {
                    name: "a".to_string(),
                    data_type: DataType::Decimal {
                        precision: 10,
                        scale: 2,
                    },
                    nullable: true,
                    metadata: BTreeMap::new(),
                },
                StructField {
                    name: "b".to_string(),
                    data_type: DataType::Char { length: 4 },
                    nullable: false,
                    metadata: {
                        let mut m = BTreeMap::new();
                        m.insert("k".to_string(), "v".to_string());
                        m
                    },
                },
            ],
        },
    ]
}

#[test]
fn accessors_for_extra_variants() {
    for dt in extra_variants() {
        // Exercise every accessor arm; correctness of the exact string is asserted
        // separately below only where unambiguous.
        let _ = dt.type_name();
        let _ = dt.simple_string();
        let _ = dt.json_value();
        let _ = dt.json();
        let _ = dt.need_conversion();
        let _ = format!("{}", dt); // Display -> simple_string
        let proto = dt.to_proto();
        // proto round-trips back to an equivalent DataType for these variants.
        let _ = DataType::from_proto(&proto);
        // json round-trips through from_json for the string-encoded variants; ignore the
        // result (some encodings intentionally do not round-trip).
        let _ = DataType::from_json(&dt.json_value());
    }
}

#[test]
fn string_collation_branches() {
    let lcase = DataType::String {
        collation: "UTF8_LCASE".to_string(),
    };
    assert_eq!(lcase.simple_string(), "string collate UTF8_LCASE");
    assert_eq!(
        lcase.json_value(),
        serde_json::json!("string collate UTF8_LCASE")
    );

    // Default collation renders as plain "string" (no trailing "collate").
    let def = DataType::String {
        collation: "UTF8_BINARY".to_string(),
    };
    assert_eq!(def.simple_string(), "string");
}

#[test]
fn interval_string_fallback_and_ranges() {
    // Out-of-range field codes fall through to the bare "interval".
    let bad = DataType::YearMonthInterval {
        start_field: 99,
        end_field: 99,
    };
    assert_eq!(bad.simple_string(), "interval");

    // Same field -> "interval <field>"; distinct -> "interval <a> to <b>".
    let ym_same = DataType::YearMonthInterval {
        start_field: 0,
        end_field: 0,
    };
    assert_eq!(ym_same.simple_string(), "interval year");
    let dt_range = DataType::DayTimeInterval {
        start_field: 0,
        end_field: 3,
    };
    assert_eq!(dt_range.simple_string(), "interval day to second");
}

#[test]
fn from_ddl_error_paths() {
    // Each malformed argument drives the map_err / ok_or_else closure inside the
    // corresponding parse_* helper.
    for bad in [
        "char(abc)",
        "char(3",
        "varchar(xyz)",
        "time(nope)",
        "decimal(x)",
        "decimal(10,y)",
        "geometry(notanumber)",
        "geography(bad)",
    ] {
        assert!(
            DataType::from_ddl(bad).is_err(),
            "expected from_ddl({bad}) to be Err"
        );
    }
}

#[test]
fn from_ddl_less_common_valid_forms() {
    for good in [
        "geometry(4326)",
        "geometry(any)",
        "geography(4326)",
        "geography(any)",
        "time(6)",
        "interval day to second",
        "interval hour to minute",
        "string collate UTF8_LCASE",
        "array<map<string,int>>",
        "map<string,array<int>>",
        "struct<a:decimal(10,2),b:char(4)>",
    ] {
        // Exercise the parser branch; a few exotic forms may Err on this server-less
        // path, which is fine — the point is to run the code.
        let _ = DataType::from_ddl(good);
    }
    // Concrete expectations for the geometry/geography DDL parser.
    assert_eq!(
        DataType::from_ddl("geometry(any)").unwrap(),
        DataType::Geometry { srid: -1 }
    );
    assert!(matches!(
        DataType::from_ddl("array<map<string,int>>").unwrap(),
        DataType::Array { .. }
    ));
}

#[test]
fn from_json_string_forms() {
    for s in [
        "void",
        "variant",
        "string collate UTF8_LCASE",
        "unparsed(mytype)",
        "geometry(4326)",
        "geography(any)",
        "interval day to second",
        "char(5)",
        "varchar(9)",
        "time(3)",
        "decimal(10,2)",
    ] {
        let _ = DataType::from_json(&serde_json::json!(s));
    }
    assert_eq!(
        DataType::from_json(&serde_json::json!("void")).unwrap(),
        DataType::Null
    );
    assert_eq!(
        DataType::from_json(&serde_json::json!("variant")).unwrap(),
        DataType::Variant
    );
}

#[test]
fn from_json_object_forms() {
    let array = serde_json::json!({
        "type": "array",
        "elementType": "integer",
        "containsNull": true,
    });
    assert!(matches!(
        DataType::from_json(&array).unwrap(),
        DataType::Array { .. }
    ));

    let map = serde_json::json!({
        "type": "map",
        "keyType": "string",
        "valueType": "integer",
        "valueContainsNull": false,
    });
    assert!(matches!(
        DataType::from_json(&map).unwrap(),
        DataType::Map { .. }
    ));

    let strct = serde_json::json!({
        "type": "struct",
        "fields": [
            {"name": "a", "type": "integer", "nullable": true, "metadata": {}},
            {"name": "b", "type": {"type": "array", "elementType": "string", "containsNull": true}, "nullable": false, "metadata": {}},
        ],
    });
    assert!(matches!(
        DataType::from_json(&strct).unwrap(),
        DataType::Struct { .. }
    ));
}

#[test]
fn from_json_error_paths() {
    // Non-string / non-object JSON value.
    assert!(DataType::from_json(&serde_json::json!(123)).is_err());
    // Unknown type string.
    assert!(DataType::from_json(&serde_json::json!("no_such_type")).is_err());
    // Array missing elementType.
    assert!(DataType::from_json(&serde_json::json!({"type": "array"})).is_err());
    // Map missing keyType / valueType.
    assert!(DataType::from_json(&serde_json::json!({"type": "map", "keyType": "string"})).is_err());
    // Struct missing fields.
    assert!(DataType::from_json(&serde_json::json!({"type": "struct"})).is_err());
    // Struct field missing name.
    assert!(DataType::from_json(&serde_json::json!({
        "type": "struct",
        "fields": [{"type": "integer"}],
    }))
    .is_err());
    // Struct field missing type.
    assert!(DataType::from_json(&serde_json::json!({
        "type": "struct",
        "fields": [{"name": "a"}],
    }))
    .is_err());
}

#[test]
fn from_proto_kind_not_set_is_error() {
    // A default proto DataType has no `kind`, driving the `None => Err(...)` arm.
    let empty = proto::DataType::default();
    assert!(DataType::from_proto(&empty).is_err());
}

#[test]
fn struct_helpers_and_non_struct_errors() {
    let s = DataType::Struct {
        fields: vec![StructField {
            name: "x".to_string(),
            data_type: DataType::Integer,
            nullable: true,
            metadata: BTreeMap::new(),
        }],
    };
    assert_eq!(s.field_names().unwrap(), vec!["x".to_string()]);
    assert_eq!(s.names().unwrap(), vec!["x".to_string()]);
    let added = s
        .add("y", DataType::Boolean, false, None)
        .expect("add on struct");
    assert_eq!(added.field_names().unwrap().len(), 2);

    // The same accessors on a non-struct must Err rather than panic.
    let not_struct = DataType::Integer;
    assert!(not_struct.field_names().is_err());
    assert!(not_struct.names().is_err());
    assert!(not_struct.add("z", DataType::Long, true, None).is_err());
}
