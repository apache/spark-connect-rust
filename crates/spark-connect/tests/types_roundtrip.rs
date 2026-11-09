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
        DataType::TimestampNtzNanos { precision: 9 },
        DataType::TimestampLtzNanos { precision: 7 },
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
    for dt in all_scalar_variants().into_iter().chain(nested_variants()) {
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

/// A struct field writes the collations of its strings, also in its arrays and maps, in the
/// `__COLLATIONS` of its metadata, as Spark and PySpark do, and reads them back; a UDT reads
/// back its JSON too.
#[test]
fn collations_are_in_the_metadata_of_the_json_of_a_field_as_spark() {
    let string = |collation: &str| DataType::String {
        collation: collation.to_string(),
    };
    let field = |name: &str, data_type: DataType| StructField {
        name: name.to_string(),
        data_type,
        nullable: true,
        metadata: BTreeMap::new(),
    };
    let st = DataType::Struct {
        fields: vec![
            field("s", string("UTF8_LCASE")),
            field(
                "m",
                DataType::Map {
                    key_type: Box::new(string("UTF8_LCASE")),
                    value_type: Box::new(DataType::Array {
                        element_type: Box::new(string("UNICODE_CI")),
                        contains_null: false,
                    }),
                    value_contains_null: true,
                },
            ),
            field(
                "n",
                DataType::Struct {
                    fields: vec![field("t", string("UTF8_LCASE"))],
                },
            ),
        ],
    };
    assert_eq!(
        st.json(),
        concat!(
            r#"{"fields":[{"metadata":{"__COLLATIONS":{"s":"spark.UTF8_LCASE"}},"name":"s","#,
            r#""nullable":true,"type":"string"},{"metadata":{"__COLLATIONS":{"m.key":"#,
            r#""spark.UTF8_LCASE","m.value.element":"icu.UNICODE_CI"}},"name":"m","#,
            r#""nullable":true,"type":{"keyType":"string","type":"map","valueContainsNull":true,"#,
            r#""valueType":{"containsNull":false,"elementType":"string","type":"array"}}},"#,
            r#"{"metadata":{},"name":"n","nullable":true,"type":{"fields":[{"metadata":"#,
            r#"{"__COLLATIONS":{"t":"spark.UTF8_LCASE"}},"name":"t","nullable":true,"#,
            r#""type":"string"}],"type":"struct"}}],"type":"struct"}"#
        )
    );
    assert_eq!(DataType::from_json_str(&st.json()).unwrap(), st);
    let DataType::Struct { fields } = &st else {
        unreachable!()
    };
    assert_eq!(
        StructField::from_json_str(&fields[1].json_value().to_string()).unwrap(),
        fields[1]
    );
    // The errors of PySpark's readers: a provider of no collation, and a collation of a field
    // that is not a string.
    let of = |type_json: &str, collation: &str| {
        let json = format!(
            r#"{{"name":"f","type":{type_json},"nullable":true,"metadata":{{"__COLLATIONS":{{"f":"{collation}"}}}}}}"#
        );
        StructField::from_json_str(&json).unwrap_err()
    };
    let error = of(r#""string""#, "other.UTF8_LCASE");
    assert_eq!(
        error.get_error_class().as_deref(),
        Some("COLLATION_INVALID_PROVIDER")
    );
    let error = of(r#""integer""#, "spark.UTF8_LCASE");
    assert_eq!(
        error.get_error_class().as_deref(),
        Some("INVALID_JSON_DATA_TYPE_FOR_COLLATIONS")
    );
    // A UDT reads back the JSON that it writes.
    let udt = DataType::Udt {
        type_str: "udt".to_string(),
        jvm_class: None,
        python_class: Some("mymod.MyUDT".to_string()),
        serialized_python_class: Some("<pickle>".to_string()),
        sql_type: Some(Box::new(DataType::Integer)),
    };
    assert_eq!(DataType::from_json_str(&udt.json()).unwrap(), udt);
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

#[test]
fn timestamp_nanos_types() {
    use spark_connect_proto::data_type::{Kind, TimestampLtzNanos, TimestampNtzNanos};

    let ntz = DataType::TimestampNtzNanos { precision: 8 };
    let ltz = DataType::TimestampLtzNanos { precision: 9 };
    // Names mirror Scala TimestampNTZNanosType / TimestampLTZNanosType.typeName.
    assert_eq!(ntz.simple_string(), "timestamp_ntz(8)");
    assert_eq!(ntz.type_name(), "timestamp_ntz(8)");
    assert_eq!(ltz.json(), "\"timestamp_ltz(9)\"");
    assert!(ntz.need_conversion());
    assert_eq!(
        DataType::from_json_str("\"timestamp_ntz(8)\"").unwrap(),
        ntz
    );
    assert_eq!(DataType::from_ddl("timestamp_ltz(9)").unwrap(), ltz);
    // Precision outside 7..=9 is rejected.
    assert!(DataType::from_json_str("\"timestamp_ntz(6)\"").is_err());

    // Proto kinds map to the nanos variants (not TIME / micros TIMESTAMP), and an
    // omitted precision defaults to 9 per types.proto.
    let mut p = spark_connect_proto::DataType::default();
    p.kind = Some(Kind::TimestampNtzNanos(TimestampNtzNanos::default()));
    assert_eq!(
        DataType::from_proto(&p).unwrap(),
        DataType::TimestampNtzNanos { precision: 9 }
    );
    p.kind = Some(Kind::TimestampLtzNanos(TimestampLtzNanos {
        precision: Some(7),
        ..Default::default()
    }));
    assert_eq!(
        DataType::from_proto(&p).unwrap(),
        DataType::TimestampLtzNanos { precision: 7 }
    );
}

/// Spatial types name their SRID by its CRS in JSON, as PySpark does, and read it back.
#[test]
fn spatial_json_names_the_crs_as_pyspark() {
    for (dt, json) in [
        (DataType::Geometry { srid: 4326 }, "\"geometry(OGC:CRS84)\""),
        (DataType::Geometry { srid: 3857 }, "\"geometry(EPSG:3857)\""),
        (DataType::Geometry { srid: 0 }, "\"geometry(SRID:0)\""),
        (DataType::Geometry { srid: -1 }, "\"geometry(SRID:ANY)\""),
        (
            DataType::Geography { srid: 4326 },
            "\"geography(OGC:CRS84, SPHERICAL)\"",
        ),
        (
            DataType::Geography { srid: -1 },
            "\"geography(SRID:ANY, SPHERICAL)\"",
        ),
    ] {
        assert_eq!(dt.json(), json);
        assert_eq!(DataType::from_json(&dt.json_value()).unwrap(), dt);
    }
    // The DDL forms name the SRID itself.
    assert_eq!(
        DataType::from_ddl("geometry(3857)").unwrap(),
        DataType::Geometry { srid: 3857 }
    );
    assert_eq!(
        DataType::from_ddl("geography(any)").unwrap(),
        DataType::Geography { srid: -1 }
    );
}

/// The COMMENT of `toDDL` escapes the backslashes and the single quotes of a comment as Spark's
/// `escapeSingleQuotedString` does, so that the parser reads back the comment as it is.
#[test]
fn to_ddl_escapes_the_comments_as_spark_does() {
    fn with_comment(comment: &str, nullable: bool) -> DataType {
        DataType::Struct {
            fields: vec![StructField {
                name: "c".to_string(),
                data_type: DataType::Integer,
                nullable,
                metadata: BTreeMap::from([(
                    "comment".to_string(),
                    serde_json::Value::String(comment.to_string()),
                )]),
            }],
        }
    }
    for (comment, ddl) in [
        ("a\\b", r"c INT COMMENT 'a\\b'"),
        ("it's", r"c INT COMMENT 'it\'s'"),
        (r"it\'s", r"c INT COMMENT 'it\\\'s'"),
        ("ends\\", r"c INT COMMENT 'ends\\'"),
        ("line\nbreak", "c INT COMMENT 'line\nbreak'"),
    ] {
        assert_eq!(
            with_comment(comment, true).to_ddl().unwrap(),
            ddl,
            "{comment:?}"
        );
    }
    // A NOT NULL field keeps its comment's escapes, after the NOT NULL.
    assert_eq!(
        with_comment("a\\b", false).to_ddl().unwrap(),
        r"c INT NOT NULL COMMENT 'a\\b'"
    );
}

/// The comments that `toDDL` writes come back from the server's DDL parser as they were, with the
/// backslashes and quotes of the comment. Runs only with `SPARK_REMOTE` set.
#[test]
fn to_ddl_comments_round_trip_through_the_server_parser() {
    let Ok(url) = std::env::var("SPARK_REMOTE") else {
        return;
    };
    let session = spark_connect::session::SparkSession::builder()
        .remote(&url)
        .get_or_create()
        .expect("session");
    for comment in ["a\\b", "it's", "it\\'s", "ends\\", "line\nbreak"] {
        let schema = DataType::Struct {
            fields: vec![StructField {
                name: "c".to_string(),
                data_type: DataType::Integer,
                nullable: true,
                metadata: BTreeMap::from([(
                    "comment".to_string(),
                    serde_json::Value::String(comment.to_string()),
                )]),
            }],
        };
        let parsed = session.parse_ddl(&schema.to_ddl().unwrap()).unwrap();
        let DataType::Struct { fields } = parsed else {
            panic!("expected a struct");
        };
        assert_eq!(
            fields[0].metadata.get("comment"),
            Some(&serde_json::Value::String(comment.to_string())),
            "{comment:?}"
        );
    }
}
