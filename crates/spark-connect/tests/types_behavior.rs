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
