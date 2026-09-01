//! Behavioral tests for DataType string/JSON/DDL rendering - asserting the exact
//! values reference pyspark produces (simpleString / typeName / jsonValue / fromDDL).
//! Pure client logic, no server needed.

use spark_connect::types::DataType;

fn s(dt: &DataType) -> String {
    dt.simple_string()
}

#[test]
fn simple_string_matches_pyspark() {
    let string_t = DataType::String {
        collation: String::new(),
    };
    assert_eq!(s(&DataType::Boolean), "boolean");
    assert_eq!(s(&DataType::Byte), "tinyint");
    assert_eq!(s(&DataType::Short), "smallint");
    assert_eq!(s(&DataType::Integer), "int");
    assert_eq!(s(&DataType::Long), "bigint");
    assert_eq!(s(&DataType::Float), "float");
    assert_eq!(s(&DataType::Double), "double");
    assert_eq!(s(&string_t), "string");
    assert_eq!(s(&DataType::Binary), "binary");
    assert_eq!(s(&DataType::Date), "date");
    assert_eq!(s(&DataType::Timestamp), "timestamp");
    assert_eq!(s(&DataType::TimestampNtz), "timestamp_ntz");
    assert_eq!(
        s(&DataType::Decimal {
            precision: 10,
            scale: 2
        }),
        "decimal(10,2)"
    );
    assert_eq!(
        s(&DataType::Array {
            element_type: Box::new(DataType::Integer),
            contains_null: true
        }),
        "array<int>"
    );
    assert_eq!(
        s(&DataType::Map {
            key_type: Box::new(string_t.clone()),
            value_type: Box::new(DataType::Long),
            value_contains_null: true,
        }),
        "map<string,bigint>"
    );
}

#[test]
fn type_name_matches_pyspark() {
    assert_eq!(DataType::Integer.type_name(), "integer");
    assert_eq!(DataType::Long.type_name(), "long");
    assert_eq!(DataType::Double.type_name(), "double");
    assert_eq!(DataType::Boolean.type_name(), "boolean");
    assert_eq!(DataType::Date.type_name(), "date");
}

#[test]
fn from_ddl_parses_to_correct_types() {
    assert_eq!(DataType::from_ddl("int").unwrap(), DataType::Integer);
    assert_eq!(DataType::from_ddl("bigint").unwrap(), DataType::Long);
    assert_eq!(DataType::from_ddl("double").unwrap(), DataType::Double);
    assert_eq!(DataType::from_ddl("boolean").unwrap(), DataType::Boolean);
    assert_eq!(
        DataType::from_ddl("decimal(10,2)").unwrap(),
        DataType::Decimal {
            precision: 10,
            scale: 2
        }
    );
    match DataType::from_ddl("array<int>").unwrap() {
        DataType::Array { element_type, .. } => assert_eq!(*element_type, DataType::Integer),
        other => panic!("expected Array, got {other:?}"),
    }
    match DataType::from_ddl("map<string,int>").unwrap() {
        DataType::Map {
            key_type,
            value_type,
            ..
        } => {
            assert!(matches!(*key_type, DataType::String { .. }));
            assert_eq!(*value_type, DataType::Integer);
        }
        other => panic!("expected Map, got {other:?}"),
    }
    match DataType::from_ddl("struct<a:int,b:string>").unwrap() {
        DataType::Struct { fields } => {
            assert_eq!(fields.len(), 2);
            assert_eq!(fields[0].name, "a");
            assert_eq!(fields[0].data_type, DataType::Integer);
        }
        other => panic!("expected Struct, got {other:?}"),
    }
    // A top-level "a INT, b STRING" is parsed as a struct.
    match DataType::from_ddl("a INT, b STRING").unwrap() {
        DataType::Struct { fields } => assert_eq!(fields.len(), 2),
        other => panic!("expected Struct, got {other:?}"),
    }
}

#[test]
fn json_value_shape() {
    // Primitive -> a JSON string of the typeName; complex -> an object.
    assert_eq!(DataType::Integer.json_value(), serde_json::json!("integer"));
    let arr = DataType::Array {
        element_type: Box::new(DataType::Integer),
        contains_null: true,
    };
    let j = arr.json_value();
    assert_eq!(j["type"], serde_json::json!("array"));
    assert_eq!(j["elementType"], serde_json::json!("integer"));
    assert_eq!(j["containsNull"], serde_json::json!(true));
}

// Every variant with a representative payload.
fn all_variants() -> Vec<DataType> {
    use spark_connect::types::StructField;
    use std::collections::BTreeMap;
    // Canonical default-collation name: the proto/JSON round-trips the default as
    // "UTF8_BINARY" (empty string is an accepted alias that normalizes to it).
    let str_t = DataType::String {
        collation: "UTF8_BINARY".to_string(),
    };
    let field = |n: &str, dt: DataType| StructField {
        name: n.to_string(),
        data_type: dt,
        nullable: true,
        metadata: BTreeMap::new(),
    };
    vec![
        DataType::Null,
        DataType::Boolean,
        DataType::Byte,
        DataType::Short,
        DataType::Integer,
        DataType::Long,
        DataType::Float,
        DataType::Double,
        DataType::Decimal {
            precision: 10,
            scale: 2,
        },
        str_t.clone(),
        DataType::Char { length: 8 },
        DataType::Varchar { length: 16 },
        DataType::Binary,
        DataType::Date,
        DataType::Timestamp,
        DataType::TimestampNtz,
        DataType::Time { precision: 6 },
        DataType::CalendarInterval,
        DataType::YearMonthInterval {
            start_field: 0,
            end_field: 1,
        },
        DataType::DayTimeInterval {
            start_field: 0,
            end_field: 3,
        },
        DataType::Array {
            element_type: Box::new(DataType::Integer),
            contains_null: true,
        },
        DataType::Map {
            key_type: Box::new(str_t.clone()),
            value_type: Box::new(DataType::Long),
            value_contains_null: true,
        },
        DataType::Struct {
            fields: vec![field("a", DataType::Integer), field("b", str_t.clone())],
        },
        DataType::Variant,
    ]
}

#[test]
fn proto_round_trip_is_lossless() {
    // to_proto -> from_proto must be identity for every variant (codec correctness).
    for dt in all_variants() {
        let back = DataType::from_proto(&dt.to_proto())
            .unwrap_or_else(|e| panic!("from_proto failed for {dt:?}: {e}"));
        assert_eq!(back, dt, "proto round-trip changed {dt:?}");
    }
}

#[test]
fn json_round_trip_is_lossless() {
    // json_value -> from_json must be identity too (the schema JSON codec).
    for dt in all_variants() {
        let back = DataType::from_json(&dt.json_value())
            .unwrap_or_else(|e| panic!("from_json failed for {dt:?}: {e}"));
        assert_eq!(back, dt, "json round-trip changed {dt:?}");
    }
}

#[test]
fn struct_type_helpers() {
    use spark_connect::types::StructField;
    use std::collections::BTreeMap;
    let st = DataType::Struct {
        fields: vec![
            StructField {
                name: "a".into(),
                data_type: DataType::Integer,
                nullable: true,
                metadata: BTreeMap::new(),
            },
            StructField {
                name: "b".into(),
                data_type: DataType::String {
                    collation: String::new(),
                },
                nullable: false,
                metadata: BTreeMap::new(),
            },
        ],
    };
    assert_eq!(st.field_names().unwrap(), vec!["a", "b"]);
    assert_eq!(st.names().unwrap(), vec!["a", "b"]);
    // add() appends a field
    let bigger = st.add("c", DataType::Double, true, None).unwrap();
    assert_eq!(bigger.field_names().unwrap(), vec!["a", "b", "c"]);
    // simpleString of the struct
    assert!(st.simple_string().starts_with("struct<"));
}

#[test]
fn remaining_variant_strings() {
    assert_eq!(DataType::Char { length: 8 }.simple_string(), "char(8)");
    assert_eq!(
        DataType::Varchar { length: 16 }.simple_string(),
        "varchar(16)"
    );
    assert_eq!(DataType::Variant.simple_string(), "variant");
    assert_eq!(DataType::CalendarInterval.simple_string(), "interval");
    // fromDDL for the remaining primitives
    assert_eq!(DataType::from_ddl("tinyint").unwrap(), DataType::Byte);
    assert_eq!(DataType::from_ddl("smallint").unwrap(), DataType::Short);
    assert_eq!(DataType::from_ddl("float").unwrap(), DataType::Float);
    assert_eq!(DataType::from_ddl("date").unwrap(), DataType::Date);
    assert_eq!(
        DataType::from_ddl("timestamp").unwrap(),
        DataType::Timestamp
    );
    assert_eq!(DataType::from_ddl("binary").unwrap(), DataType::Binary);
    assert!(matches!(
        DataType::from_ddl("char(8)").unwrap(),
        DataType::Char { length: 8 }
    ));
    assert!(matches!(
        DataType::from_ddl("varchar(16)").unwrap(),
        DataType::Varchar { length: 16 }
    ));
}

#[test]
fn struct_to_ddl_tree_and_nullable() {
    use spark_connect::types::StructField;
    let st = DataType::Struct {
        fields: vec![
            StructField {
                name: "a".to_string(),
                data_type: DataType::Integer,
                nullable: true,
                metadata: Default::default(),
            },
            StructField {
                name: "b".to_string(),
                data_type: DataType::String {
                    collation: "UTF8_BINARY".to_string(),
                },
                nullable: false,
                metadata: Default::default(),
            },
        ],
    };
    assert_eq!(st.to_ddl().unwrap(), "a int,b string NOT NULL");
    let tree = st.tree_string().unwrap();
    assert!(tree.starts_with("root\n"));
    assert!(tree.contains("a: int (nullable = true)"));
    assert!(tree.contains("b: string (nullable = false)"));
    // toNullable makes every field nullable.
    if let DataType::Struct { fields } = st.to_nullable() {
        assert!(fields.iter().all(|f| f.nullable));
    } else {
        panic!("expected struct");
    }
    // to_ddl / tree_string reject non-structs.
    assert!(DataType::Integer.to_ddl().is_err());
    assert!(DataType::Integer.tree_string().is_err());
}

#[test]
fn tree_string_with_depth_truncates_nested_structs() {
    use spark_connect::types::StructField;
    let nested = DataType::Struct {
        fields: vec![StructField {
            name: "outer".to_string(),
            data_type: DataType::Struct {
                fields: vec![StructField {
                    name: "inner".to_string(),
                    data_type: DataType::Integer,
                    nullable: true,
                    metadata: Default::default(),
                }],
            },
            nullable: true,
            metadata: Default::default(),
        }],
    };
    // Full depth shows the nested field, rendering the struct as the bare type name.
    let full = nested.tree_string().unwrap();
    assert!(full.contains("outer: struct (nullable = true)"));
    assert!(full.contains("inner: int (nullable = true)"));
    // max_depth = 1 prints only the top level; the nested field is omitted.
    let shallow = nested.tree_string_with_depth(1).unwrap();
    assert!(shallow.contains("outer: struct (nullable = true)"));
    assert!(!shallow.contains("inner"));
}

#[test]
fn from_json_str_roundtrips() {
    use spark_connect::types::StructField;
    let dt = DataType::from_json_str("\"integer\"").unwrap();
    assert!(matches!(dt, DataType::Integer));
    let f = StructField::from_json_str(
        "{\"name\":\"a\",\"type\":\"integer\",\"nullable\":true,\"metadata\":{}}",
    )
    .unwrap();
    assert_eq!(f.name, "a");
    assert!(f.nullable);
    assert!(matches!(f.data_type, DataType::Integer));
}
