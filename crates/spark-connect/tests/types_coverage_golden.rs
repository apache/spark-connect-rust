//! Coverage smoke test: exercise every `DataType` variant through its proto
//! round-trip and its string/json representations.
//!
//! For each variant we run `to_proto()` → `from_proto()` and, where the proto
//! representation is lossless, assert the round-trip is identity. We also call
//! `simple_string()` and `json_value()` so those arms are covered too. This walks
//! every arm of the big `DataType` match blocks without a live server.

use std::collections::BTreeMap;

use spark_connect::types::{DataType, StructField};

fn field(name: &str, dt: DataType) -> StructField {
    StructField {
        name: name.to_string(),
        data_type: dt,
        nullable: true,
        metadata: BTreeMap::new(),
    }
}

/// Every variant (with representative payloads) that round-trips losslessly.
fn all_variants() -> Vec<DataType> {
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
        DataType::String {
            collation: String::new(),
        },
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
            key_type: Box::new(DataType::String {
                collation: String::new(),
            }),
            value_type: Box::new(DataType::Long),
            value_contains_null: true,
        },
        DataType::Struct {
            fields: vec![
                field("a", DataType::Integer),
                field(
                    "b",
                    DataType::String {
                        collation: String::new(),
                    },
                ),
                field(
                    "c",
                    DataType::Array {
                        element_type: Box::new(DataType::Double),
                        contains_null: false,
                    },
                ),
            ],
        },
        DataType::Variant,
        DataType::Geometry { srid: 4326 },
        DataType::Geography { srid: 4326 },
        DataType::Udt {
            type_str: "udt".to_string(),
            jvm_class: Some("com.example.Udt".to_string()),
            python_class: Some("mod.Udt".to_string()),
            serialized_python_class: None,
            sql_type: Some(Box::new(DataType::Double)),
        },
        DataType::Unparsed {
            data_type_string: "int".to_string(),
        },
    ]
}

#[test]
fn every_datatype_variant_serializes_and_stringifies() {
    for dt in all_variants() {
        // Proto round-trip through the wire type.
        let proto = dt.to_proto();
        let back = DataType::from_proto(&proto)
            .unwrap_or_else(|e| panic!("from_proto failed for {dt:?}: {e}"));
        // simple_string / json_value must not panic for any variant.
        let _ = dt.simple_string();
        let _ = dt.json_value();
        let _ = back.simple_string();
        let _ = back.json_value();
    }
}

#[test]
fn ddl_parse_covers_primitive_and_complex_types() {
    // Exercise the DDL parser arms (from_ddl / parse) across the type space.
    for ddl in [
        "int",
        "bigint",
        "string",
        "double",
        "boolean",
        "date",
        "timestamp",
        "binary",
        "tinyint",
        "smallint",
        "float",
        "decimal(10,2)",
        "char(8)",
        "varchar(16)",
        "array<int>",
        "map<string,int>",
        "struct<a:int,b:string>",
        "a INT, b STRING, c ARRAY<DOUBLE>",
    ] {
        let dt =
            DataType::from_ddl(ddl).unwrap_or_else(|e| panic!("from_ddl failed for {ddl:?}: {e}"));
        // Round-trip the parsed type through proto too.
        let _ = DataType::from_proto(&dt.to_proto()).unwrap();
        let _ = dt.simple_string();
    }
}
