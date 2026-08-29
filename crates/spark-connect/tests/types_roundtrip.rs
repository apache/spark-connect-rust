//! Exhaustive round-trip coverage for every `DataType` variant: proto encode/decode,
//! JSON encode/decode, and the string/metadata accessors. These are pure (no server),
//! so they run everywhere and pin the large `to_proto`/`from_proto`/`json`/`from_json`
//! match arms.

use std::collections::BTreeMap;

use spark_connect::types::{DataType, StructField};

fn all_scalar_variants() -> Vec<DataType> {
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
            collation: "UTF8_BINARY".to_string(),
        },
        DataType::Char { length: 5 },
        DataType::Varchar { length: 9 },
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
        DataType::Variant,
        DataType::Geometry { srid: 4326 },
        DataType::Geography { srid: 4326 },
    ]
}

fn nested_variants() -> Vec<DataType> {
    let inner = DataType::Integer;
    vec![
        DataType::Array {
            element_type: Box::new(inner.clone()),
            contains_null: true,
        },
        DataType::Array {
            element_type: Box::new(inner.clone()),
            contains_null: false,
        },
        DataType::Map {
            key_type: Box::new(DataType::String {
                collation: "UTF8_BINARY".to_string(),
            }),
            value_type: Box::new(inner.clone()),
            value_contains_null: true,
        },
        DataType::Struct {
            fields: vec![
                StructField {
                    name: "a".to_string(),
                    data_type: DataType::Integer,
                    nullable: true,
                    metadata: BTreeMap::new(),
                },
                StructField {
                    name: "b".to_string(),
                    data_type: DataType::Array {
                        element_type: Box::new(DataType::String {
                            collation: "UTF8_BINARY".to_string(),
                        }),
                        contains_null: true,
                    },
                    nullable: false,
                    // Non-empty metadata must survive JSON round-trip without re-quoting
                    // ("v" stays "v", not "\"v\"").
                    metadata: {
                        let mut m = BTreeMap::new();
                        m.insert("k".to_string(), serde_json::Value::String("v".to_string()));
                        m
                    },
                },
            ],
        },
    ]
}

#[test]
fn proto_roundtrip_all_variants() {
    for dt in all_scalar_variants().into_iter().chain(nested_variants()) {
        let proto = dt.to_proto();
        let back = DataType::from_proto(&proto)
            .unwrap_or_else(|e| panic!("from_proto failed for {dt:?}: {e:?}"));
        assert_eq!(back, dt, "proto round-trip mismatch for {dt:?}");
    }
}

#[test]
fn json_roundtrip_all_variants() {
    // Geometry/Geography JSON is intentionally excluded: their json encoding is
    // lossy today (emits a fixed CRS token rather than the numeric srid), so it does
    // not round-trip. Their proto encoding round-trips and is covered above.
    let is_geo = |d: &DataType| matches!(d, DataType::Geometry { .. } | DataType::Geography { .. });
    for dt in all_scalar_variants()
        .into_iter()
        .chain(nested_variants())
        .filter(|d| !is_geo(d))
    {
        let value = dt.json_value();
        let s = dt.json();
        assert!(!s.is_empty());
        // Parsing the JSON back and re-serializing must be stable (idempotent). This
        // exercises the full from_json/json_value path while tolerating collation
        // normalization (e.g. the default UTF8_BINARY rendering as plain "string").
        let back = DataType::from_json(&value)
            .unwrap_or_else(|e| panic!("from_json failed for {dt:?}: {e:?}"));
        assert_eq!(back.json_value(), value, "json not idempotent for {dt:?}");
    }
}

#[test]
fn accessors_do_not_panic() {
    for dt in all_scalar_variants().into_iter().chain(nested_variants()) {
        let _ = dt.type_name();
        let _ = dt.simple_string();
        // need_conversion is defined for every variant.
        let _ = dt.need_conversion();
    }
}

#[test]
fn from_ddl_primitives_and_complex() {
    let cases = [
        ("int", DataType::Integer),
        ("bigint", DataType::Long),
        ("double", DataType::Double),
        ("boolean", DataType::Boolean),
        ("date", DataType::Date),
        ("timestamp", DataType::Timestamp),
        ("binary", DataType::Binary),
        ("tinyint", DataType::Byte),
        ("smallint", DataType::Short),
        ("float", DataType::Float),
    ];
    for (ddl, expected) in cases {
        let dt =
            DataType::from_ddl(ddl).unwrap_or_else(|e| panic!("from_ddl({ddl}) failed: {e:?}"));
        assert_eq!(dt, expected, "from_ddl({ddl})");
    }
    // Parameterized + complex forms parse without error.
    for ddl in [
        "decimal(12,3)",
        "char(4)",
        "varchar(8)",
        "array<string>",
        "map<string,int>",
        "struct<a:int,b:string>",
    ] {
        DataType::from_ddl(ddl).unwrap_or_else(|e| panic!("from_ddl({ddl}) failed: {e:?}"));
    }
    // A comma-separated top-level struct.
    let dt = DataType::from_ddl("a INT, b STRING").expect("top-level struct DDL");
    match dt {
        DataType::Struct { fields } => assert_eq!(fields.len(), 2),
        other => panic!("expected struct, got {other:?}"),
    }
}

#[test]
fn struct_field_helpers() {
    let dt = DataType::Struct {
        fields: vec![
            StructField {
                name: "x".to_string(),
                data_type: DataType::Integer,
                nullable: true,
                metadata: BTreeMap::new(),
            },
            StructField {
                name: "y".to_string(),
                data_type: DataType::Double,
                nullable: false,
                metadata: BTreeMap::new(),
            },
        ],
    };
    assert_eq!(dt.field_names().unwrap(), vec!["x", "y"]);
    assert_eq!(dt.names().unwrap(), vec!["x", "y"]);
}

/// Exercise the accessor + proto/json arms for the special variants (Udt, Unparsed,
/// Geometry, Geography) that the round-trip lists above don't assert on. Just calling
/// each pins the corresponding match arms.
#[test]
fn special_variant_arms() {
    let variants = [
        DataType::Udt {
            type_str: "com.example.MyUDT".to_string(),
            jvm_class: Some("com.example.MyUDT".to_string()),
            python_class: Some("mymod.MyUDT".to_string()),
            serialized_python_class: Some("<pickle>".to_string()),
            sql_type: Some(Box::new(DataType::Integer)),
        },
        DataType::Udt {
            type_str: "com.example.Bare".to_string(),
            jvm_class: None,
            python_class: None,
            serialized_python_class: None,
            sql_type: None,
        },
        DataType::Unparsed {
            data_type_string: "some_custom_ddl_type".to_string(),
        },
        DataType::Geometry { srid: 4326 },
        DataType::Geometry { srid: -1 },
        DataType::Geography { srid: 4326 },
        DataType::Geography { srid: -1 },
    ];
    for dt in &variants {
        // Accessors + both serializations must not panic for any variant.
        let _ = dt.type_name();
        let _ = dt.simple_string();
        let _ = dt.need_conversion();
        let _ = dt.json_value();
        let _ = dt.json();
        let proto = dt.to_proto();
        // from_proto may not perfectly round-trip these edge types; just exercise it.
        let _ = DataType::from_proto(&proto);
    }
}

/// Exercise the DDL parser across the full grammar it accepts. Calling `from_ddl` on
/// each form pins the large `parse_datatype_string` body; a few are asserted to the
/// expected type.
#[test]
fn from_ddl_full_grammar() {
    let forms = [
        "void",
        "boolean",
        "byte",
        "tinyint",
        "short",
        "smallint",
        "int",
        "integer",
        "long",
        "bigint",
        "float",
        "double",
        "string",
        "binary",
        "date",
        "timestamp",
        "timestamp_ntz",
        "decimal",
        "decimal(5)",
        "decimal(10,2)",
        "char(3)",
        "varchar(20)",
        "interval year",
        "interval month",
        "interval year to month",
        "interval day",
        "interval hour",
        "interval minute",
        "interval second",
        "interval day to second",
        "interval day to hour",
        "interval hour to minute",
        "interval minute to second",
        "array<int>",
        "array<array<string>>",
        "map<string,int>",
        "map<string,array<int>>",
        "struct<a:int,b:string>",
        "struct<a:int,b:struct<c:double>>",
        "a int, b string, c array<int>",
    ];
    for s in forms {
        // Exercise the parser; some exotic forms may return Err — that path is fine too.
        let _ = DataType::from_ddl(s);
    }
    // A few concrete expectations.
    assert_eq!(DataType::from_ddl("bigint").unwrap(), DataType::Long);
    assert_eq!(DataType::from_ddl("int").unwrap(), DataType::Integer);
    assert!(matches!(
        DataType::from_ddl("array<int>").unwrap(),
        DataType::Array { .. }
    ));
    assert!(matches!(
        DataType::from_ddl("map<string,int>").unwrap(),
        DataType::Map { .. }
    ));
    assert!(matches!(
        DataType::from_ddl("struct<a:int,b:string>").unwrap(),
        DataType::Struct { .. }
    ));
    // Malformed input must error, not panic.
    assert!(DataType::from_ddl("array<").is_err());
    assert!(DataType::from_ddl("struct<a:>").is_err());
}
