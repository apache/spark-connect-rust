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

/// `from_ddl` reads the `NOT NULL` and `COMMENT` clauses of a field as Spark's grammar does,
/// `colName dataType [NOT NULL] [COMMENT 'string']` (and `complexColType` for a struct field, whose
/// colon is optional): keywords in any case, the comment's escapes, adjacent string literals, and
/// backquoted names. The expected values are the server's `DataType.fromDDL` results.
#[test]
fn from_ddl_reads_not_null_and_comment_as_spark() {
    fn field(
        name: &str,
        data_type: DataType,
        nullable: bool,
        comment: Option<&str>,
    ) -> StructField {
        let metadata = comment
            .map(|c| {
                BTreeMap::from([(
                    "comment".to_string(),
                    serde_json::Value::String(c.to_string()),
                )])
            })
            .unwrap_or_default();
        StructField {
            name: name.to_string(),
            data_type,
            nullable,
            metadata,
        }
    }
    let struct_of = |fields: Vec<StructField>| DataType::Struct { fields };

    // Top-level form, with and without a comma, in any keyword case.
    assert_eq!(
        DataType::from_ddl("x INT NOT NULL").unwrap(),
        struct_of(vec![field("x", DataType::Integer, false, None)])
    );
    // A name of other letters than ASCII must be back quoted, as Spark's parser requires
    // (INVALID_IDENTIFIER for `이름 INT`).
    assert!(DataType::from_ddl("이름 INT NOT NULL").is_err());
    assert_eq!(
        DataType::from_ddl("`이름` INT NOT NULL, `café` INT COMMENT 'c'").unwrap(),
        struct_of(vec![
            field("이름", DataType::Integer, false, None),
            field("café", DataType::Integer, true, Some("c")),
        ])
    );
    assert_eq!(
        DataType::from_ddl("a INT NOT NULL, b STRING").unwrap(),
        struct_of(vec![
            field("a", DataType::Integer, false, None),
            field(
                "b",
                DataType::String {
                    collation: "UTF8_BINARY".into()
                },
                true,
                None
            ),
        ])
    );
    assert_eq!(
        DataType::from_ddl("a INT, b STRING COMMENT 'x'").unwrap(),
        struct_of(vec![
            field("a", DataType::Integer, true, None),
            field(
                "b",
                DataType::String {
                    collation: "UTF8_BINARY".into()
                },
                true,
                Some("x")
            ),
        ])
    );
    assert_eq!(
        DataType::from_ddl("a INT not null, b string Comment 'hi'").unwrap(),
        struct_of(vec![
            field("a", DataType::Integer, false, None),
            field(
                "b",
                DataType::String {
                    collation: "UTF8_BINARY".into()
                },
                true,
                Some("hi")
            ),
        ])
    );
    assert_eq!(
        DataType::from_ddl("a INT NOT NULL COMMENT 'x', b STRING").unwrap(),
        struct_of(vec![
            field("a", DataType::Integer, false, Some("x")),
            field(
                "b",
                DataType::String {
                    collation: "UTF8_BINARY".into()
                },
                true,
                None
            ),
        ])
    );
    assert_eq!(
        DataType::from_ddl("a INT !NULL").unwrap(),
        struct_of(vec![field("a", DataType::Integer, false, None)])
    );
    assert_eq!(
        DataType::from_ddl("a decimal(10, 2) NOT NULL, b interval day to second").unwrap(),
        struct_of(vec![
            field(
                "a",
                DataType::Decimal {
                    precision: 10,
                    scale: 2
                },
                false,
                None
            ),
            field(
                "b",
                DataType::DayTimeInterval {
                    start_field: 0,
                    end_field: 3
                },
                true,
                None
            ),
        ])
    );

    // Nested form, with the optional colon and the same clauses.
    assert_eq!(
        DataType::from_ddl("struct<a: INT NOT NULL>").unwrap(),
        struct_of(vec![field("a", DataType::Integer, false, None)])
    );
    assert_eq!(
        DataType::from_ddl("STRUCT<a: INT NOT NULL COMMENT 'x', b: STRING>").unwrap(),
        DataType::Struct {
            fields: vec![
                field("a", DataType::Integer, false, Some("x")),
                field(
                    "b",
                    DataType::String {
                        collation: "UTF8_BINARY".into()
                    },
                    true,
                    None
                ),
            ]
        }
    );
    assert_eq!(
        DataType::from_ddl("struct<a int comment 'x,y<z'>").unwrap(),
        struct_of(vec![field("a", DataType::Integer, true, Some("x,y<z"))])
    );
    assert_eq!(
        DataType::from_ddl("array<struct<c: LONG NOT NULL COMMENT 'q'>>").unwrap(),
        DataType::Array {
            element_type: Box::new(struct_of(vec![field(
                "c",
                DataType::Long,
                false,
                Some("q")
            )])),
            contains_null: true,
        }
    );

    // The escapes of the comment, as `unescapeSQLString` reads them.
    for (ddl, comment) in [
        (r"a INT COMMENT 'it\'s'", "it's"),
        (r"a INT COMMENT 'it''s'", "it's"),
        (r"a INT COMMENT 'a\\b'", r"a\b"),
        (r"a INT COMMENT 'a\\'", r"a\"),
        (r"a INT COMMENT '\''", "'"),
        ("a INT COMMENT \"dq\"", "dq"),
        (r"a INT COMMENT 'a' 'b'", "ab"),
        (r"a INT COMMENT ''", ""),
        (r"a INT COMMENT R'raw\x'", r"raw\x"),
        (r"a INT COMMENT 'xA\101\n'", "xAA\n"),
        (r"a INT COMMENT 'x\z\%\_'", r"xz\%\_"),
        ("a INT COMMENT 'line\nbreak, comma'", "line\nbreak, comma"),
    ] {
        let parsed = DataType::from_ddl(ddl).unwrap_or_else(|e| panic!("{ddl}: {e:?}"));
        assert_eq!(
            parsed,
            struct_of(vec![field("a", DataType::Integer, true, Some(comment))]),
            "{ddl}"
        );
    }

    // Backquoted names: the quotes go, the doubled backquote is one.
    assert_eq!(
        DataType::from_ddl("`a b` INT NOT NULL").unwrap(),
        struct_of(vec![field("a b", DataType::Integer, false, None)])
    );
    assert_eq!(
        DataType::from_ddl("`a``b` INT COMMENT 'c'").unwrap(),
        struct_of(vec![field("a`b", DataType::Integer, true, Some("c"))])
    );
    assert_eq!(
        DataType::from_ddl("`a,b` INT, `c:d` STRING").unwrap(),
        struct_of(vec![
            field("a,b", DataType::Integer, true, None),
            field(
                "c:d",
                DataType::String {
                    collation: "UTF8_BINARY".into()
                },
                true,
                None
            ),
        ])
    );
    assert_eq!(
        DataType::from_ddl("struct<`a b`: TIME(3) NOT NULL COMMENT 'x'>").unwrap(),
        struct_of(vec![field(
            "a b",
            DataType::Time { precision: 3 },
            false,
            Some("x")
        )])
    );

    // A clause out of order, repeated, unterminated or without a string is an error.
    for ddl in [
        "a INT COMMENT 'x' NOT NULL",
        "a INT NOT NULL NOT NULL",
        "a INT COMMENT 'x' COMMENT 'y'",
        "a INT NOT",
        "a INT NOTNULL",
        "a INT COMMENT x",
        "a INT COMMENT 'x",
        "a INT COMMENT 'it's'",
        "struct<a: INT COMMENT 'x>",
    ] {
        assert!(DataType::from_ddl(ddl).is_err(), "{ddl} must not parse");
    }
}

/// `to_ddl` writes a non-nullable or commented field, at every nesting level, and `from_ddl` reads
/// that back as the same type.
#[test]
fn to_ddl_of_non_nullable_and_commented_fields_round_trips() {
    let comment = |c: &str| {
        BTreeMap::from([(
            "comment".to_string(),
            serde_json::Value::String(c.to_string()),
        )])
    };
    let schemas = [
        DataType::Struct {
            fields: vec![
                StructField {
                    name: "a".to_string(),
                    data_type: DataType::Integer,
                    nullable: false,
                    metadata: BTreeMap::new(),
                },
                StructField {
                    name: "b".to_string(),
                    data_type: DataType::Double,
                    nullable: true,
                    metadata: comment("it's a \\ comment, <with> (parens)"),
                },
                StructField {
                    name: "c".to_string(),
                    data_type: DataType::Long,
                    nullable: false,
                    metadata: comment("ends with a backslash\\"),
                },
            ],
        },
        DataType::Struct {
            fields: vec![
                StructField {
                    name: "outer".to_string(),
                    data_type: DataType::Struct {
                        fields: vec![StructField {
                            name: "inner".to_string(),
                            data_type: DataType::Struct {
                                fields: vec![StructField {
                                    name: "deep".to_string(),
                                    data_type: DataType::Date,
                                    nullable: false,
                                    metadata: comment("deep \"quoted\" 'comment'\nline"),
                                }],
                            },
                            nullable: false,
                            metadata: comment("middle"),
                        }],
                    },
                    nullable: true,
                    metadata: BTreeMap::new(),
                },
                StructField {
                    name: "arr".to_string(),
                    data_type: DataType::Array {
                        element_type: Box::new(DataType::Struct {
                            fields: vec![StructField {
                                name: "x".to_string(),
                                data_type: DataType::Integer,
                                nullable: false,
                                metadata: comment("in an array"),
                            }],
                        }),
                        contains_null: true,
                    },
                    nullable: false,
                    metadata: BTreeMap::new(),
                },
                StructField {
                    name: "m".to_string(),
                    data_type: DataType::Map {
                        key_type: Box::new(DataType::String {
                            collation: "UTF8_BINARY".to_string(),
                        }),
                        value_type: Box::new(DataType::Struct {
                            fields: vec![StructField {
                                name: "v".to_string(),
                                data_type: DataType::Integer,
                                nullable: false,
                                metadata: BTreeMap::new(),
                            }],
                        }),
                        value_contains_null: true,
                    },
                    nullable: true,
                    metadata: comment("a map"),
                },
                StructField {
                    name: "a b`c".to_string(),
                    data_type: DataType::Integer,
                    nullable: false,
                    metadata: BTreeMap::new(),
                },
            ],
        },
    ];
    for schema in schemas {
        let ddl = schema.to_ddl().unwrap();
        assert_eq!(DataType::from_ddl(&ddl).unwrap(), schema, "{ddl}");
    }
}

/// The type names of `DataType::type_name`, which mirror `DataType.typeName()`: a day-time and a
/// year-month interval are not `interval`, and a UDT is its Python class name in lower case.
#[test]
fn type_name_of_intervals_and_udts_as_pyspark() {
    let day_time = DataType::DayTimeInterval {
        start_field: 0,
        end_field: 3,
    };
    let year_month = DataType::YearMonthInterval {
        start_field: 0,
        end_field: 1,
    };
    assert_eq!(day_time.type_name(), "daytimeinterval");
    assert_eq!(year_month.type_name(), "yearmonthinterval");
    assert_eq!(DataType::CalendarInterval.type_name(), "interval");
    let udt = |python_class: Option<&str>| DataType::Udt {
        type_str: "point".to_string(),
        jvm_class: None,
        python_class: python_class.map(str::to_string),
        serialized_python_class: None,
        sql_type: None,
    };
    assert_eq!(udt(Some("mymod.PointUDT")).type_name(), "pointudt");
    assert_eq!(udt(None).type_name(), "udt");
}

/// `from_ddl` agrees with the server's DDL parser on the `NOT NULL` and `COMMENT` clauses, the
/// nested forms and the backquoted names. Runs only with `SPARK_REMOTE` set.
#[test]
fn from_ddl_not_null_and_comment_agree_with_the_server_parser() {
    let Ok(url) = std::env::var("SPARK_REMOTE") else {
        return;
    };
    let session = spark_connect::session::SparkSession::builder()
        .remote(&url)
        .get_or_create()
        .expect("session");
    for ddl in [
        "x INT NOT NULL",
        "`이름` INT NOT NULL, `café` STRING COMMENT 'c'",
        "a INT NOT NULL, b BIGINT",
        "a INT, b DOUBLE COMMENT 'x'",
        "a INT not null, b int Comment 'hi, there'",
        "a INT NOT NULL COMMENT 'x', b DATE",
        "a INT COMMENT 'it\\'s', b INT COMMENT 'a\\\\b'",
        "a INT COMMENT 'x' ''",
        "`a b` INT NOT NULL",
        "`a``b` INT COMMENT 'c'",
        "struct<a: INT NOT NULL>",
        "STRUCT<a: INT NOT NULL COMMENT 'x', b: STRING>",
        "struct<a: int comment 'x,y<z'>",
        "struct<`a b`: TIME(3) NOT NULL>",
        "struct<a:int,b:array<struct<c: int NOT NULL COMMENT 'q'>>>",
        "array<struct<a: INT COMMENT 'c'>>",
        "map<string, struct<a: int not null>>",
        "a decimal(10, 2) NOT NULL",
        "a interval day to second NOT NULL",
    ] {
        let server = session
            .parse_ddl(ddl)
            .unwrap_or_else(|e| panic!("{ddl}: {e:?}"));
        assert_eq!(DataType::from_ddl(ddl).unwrap(), server, "{ddl}");
    }
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

/// `from_ddl` reads an interval as the server's parser does: `interval`, `interval <field>` and
/// `interval <from> to <to>`, the end field later than the start field within one family. The
/// other forms are errors. A top-level `name:type` names a field, as `name type` does.
#[test]
fn from_ddl_interval_ranges_and_colon_fields_as_spark() {
    let field = |name: &str, data_type: DataType| DataType::Struct {
        fields: vec![StructField {
            name: name.to_string(),
            data_type,
            nullable: true,
            metadata: BTreeMap::new(),
        }],
    };
    let year_month = |start_field, end_field| DataType::YearMonthInterval {
        start_field,
        end_field,
    };
    let day_time = |start_field, end_field| DataType::DayTimeInterval {
        start_field,
        end_field,
    };
    for (ddl, expected) in [
        ("interval", DataType::CalendarInterval),
        ("interval year", year_month(0, 0)),
        ("interval year to month", year_month(0, 1)),
        ("interval day", day_time(0, 0)),
        ("interval day to second", day_time(0, 3)),
        ("interval hour to second", day_time(1, 3)),
        ("INTERVAL  DAY   TO SECOND", day_time(0, 3)),
        ("a:interval day to hour", field("a", day_time(0, 1))),
        ("a:int", field("a", DataType::Integer)),
        (
            "x:array<int>",
            field(
                "x",
                DataType::Array {
                    element_type: Box::new(DataType::Integer),
                    contains_null: true,
                },
            ),
        ),
        ("x:struct<a:int>", field("x", field("a", DataType::Integer))),
        ("int:int", field("int", DataType::Integer)),
        ("`a:b`:int", field("a:b", DataType::Integer)),
    ] {
        assert_eq!(DataType::from_ddl(ddl).unwrap(), expected, "{ddl}");
    }
    let string = DataType::String {
        collation: "UTF8_BINARY".to_string(),
    };
    assert_eq!(
        DataType::from_ddl("a:int, b:string").unwrap(),
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
                    data_type: string,
                    nullable: true,
                    metadata: BTreeMap::new(),
                },
            ],
        }
    );
    let comment = BTreeMap::from([(
        "comment".to_string(),
        serde_json::Value::String("x:y".to_string()),
    )]);
    assert_eq!(
        DataType::from_ddl("a:int COMMENT 'x:y'").unwrap(),
        DataType::Struct {
            fields: vec![StructField {
                name: "a".to_string(),
                data_type: DataType::Integer,
                nullable: true,
                metadata: comment,
            }],
        }
    );
    for ddl in [
        "interval year to second",
        "interval day to second NOT NULL",
        "interval day to second xyz",
        "interval day to second foo",
        "interval day to",
        "interval foo",
        "interval to second",
        "interval day second",
        "interval year month",
        "interval day to second to hour",
        "intervalday",
        "interval year to year",
        "interval day to day",
        "interval hour to day",
        "interval month to year",
        "interval second to second",
        "a interval year to second",
        "a:'x:y' int",
    ] {
        assert!(DataType::from_ddl(ddl).is_err(), "{ddl}");
    }
}

/// The spellings of Spark's grammar for the decimal, real, character, timestamp and time types:
/// `dec` and `numeric` are decimal, a bare `decimal` is decimal(10,0), `real` is float, and the
/// time zone clauses name the timestamp and time types, as the server's parser reads them.
#[test]
fn from_ddl_type_spellings_as_spark() {
    let field = |name: &str, data_type: DataType| DataType::Struct {
        fields: vec![StructField {
            name: name.to_string(),
            data_type,
            nullable: true,
            metadata: BTreeMap::new(),
        }],
    };
    let decimal = |precision, scale| DataType::Decimal { precision, scale };
    for (ddl, expected) in [
        ("decimal", decimal(10, 0)),
        ("decimal(10)", decimal(10, 0)),
        ("decimal (10, 2)", decimal(10, 2)),
        ("dec", decimal(10, 0)),
        ("dec(5,1)", decimal(5, 1)),
        ("dec(5)", decimal(5, 0)),
        ("NUMERIC", decimal(10, 0)),
        ("numeric(10,2)", decimal(10, 2)),
        ("real", DataType::Float),
        ("character(5)", DataType::Char { length: 5 }),
        ("timestamp_ltz", DataType::Timestamp),
        ("timestamp without time zone", DataType::TimestampNtz),
        ("timestamp with local time zone", DataType::Timestamp),
        ("Timestamp  With Local Time Zone", DataType::Timestamp),
        ("timestamp(6)", DataType::Timestamp),
        ("timestamp(6) without time zone", DataType::TimestampNtz),
        ("timestamp_ntz(6)", DataType::TimestampNtz),
        ("timestamp(9)", DataType::TimestampLtzNanos { precision: 9 }),
        (
            "timestamp(9) without time zone",
            DataType::TimestampNtzNanos { precision: 9 },
        ),
        (
            "timestamp_ltz(7)",
            DataType::TimestampLtzNanos { precision: 7 },
        ),
        ("time without time zone", DataType::Time { precision: 6 }),
        ("time(3) without time zone", DataType::Time { precision: 3 }),
        ("a decimal", field("a", decimal(10, 0))),
        ("a: numeric(10,2)", field("a", decimal(10, 2))),
        ("a real", field("a", DataType::Float)),
        (
            "a timestamp without time zone",
            field("a", DataType::TimestampNtz),
        ),
    ] {
        assert_eq!(DataType::from_ddl(ddl).unwrap(), expected, "{ddl}");
    }
    for ddl in [
        "character",
        "char",
        "varchar",
        "timestamp(3)",
        "timestamp with time zone",
        "time with local time zone",
        "timestamp_ltz(9) with local time zone",
        "timestamp(9) with time zone",
        "decimal()",
        "decimal(10,2,3)",
        "numeric(10,2,3)",
    ] {
        assert!(DataType::from_ddl(ddl).is_err(), "{ddl}");
    }
}

/// `from_ddl` agrees with the server's parser on the interval ranges, the `name:type` fields and
/// the type spellings above, valid and invalid. Runs only with `SPARK_REMOTE` set.
#[test]
fn from_ddl_interval_colon_and_spellings_agree_with_the_server_parser() {
    let Ok(url) = std::env::var("SPARK_REMOTE") else {
        return;
    };
    let session = spark_connect::session::SparkSession::builder()
        .remote(&url)
        .get_or_create()
        .expect("session");
    for ddl in [
        "interval",
        "interval year",
        "interval year to month",
        "interval hour to second",
        "INTERVAL  DAY   TO SECOND",
        "a:int",
        "x:array<int>",
        "x:struct<a:int>",
        "int:int",
        "a: int",
        "a:int, b:string",
        "`a:b`:int",
        "a:int COMMENT 'x:y'",
        "x:map<string,int>",
        "a:decimal",
        "decimal",
        "decimal(10)",
        "decimal (10,2)",
        "dec(5,1)",
        "numeric",
        "numeric(10,2)",
        "real",
        "a real",
        "character(5)",
        "timestamp_ltz",
        "timestamp without time zone",
        "timestamp with local time zone",
        "timestamp(6) with local time zone",
        "timestamp(9)",
        "timestamp(9) without time zone",
        "time without time zone",
        "a timestamp without time zone",
        "a: timestamp with local time zone",
    ] {
        let server = session
            .parse_ddl(ddl)
            .unwrap_or_else(|e| panic!("{ddl}: {e:?}"));
        assert_eq!(DataType::from_ddl(ddl).unwrap(), server, "{ddl}");
    }
    for ddl in [
        "interval year to second",
        "interval day to second xyz",
        "interval day to",
        "interval foo",
        "interval hour to day",
        "a interval year to second",
        "a:'x:y' int",
        "character",
        "timestamp(3)",
        "timestamp with time zone",
        "time with local time zone",
        "decimal()",
    ] {
        assert!(session.parse_ddl(ddl).is_err(), "{ddl}");
        assert!(DataType::from_ddl(ddl).is_err(), "{ddl}");
    }
}

/// The collations of DDL, backquoted or qualified as the server reads them, and the names that it
/// refuses, agree with the server's parser; so do the `time` of DDL and a trailing whitespace.
#[test]
fn ddl_collations_and_trailing_whitespace_agree_with_the_server_parser() {
    let Ok(url) = std::env::var("SPARK_REMOTE") else {
        return;
    };
    let session = spark_connect::session::SparkSession::builder()
        .remote(&url)
        .get_or_create()
        .expect("session");
    for ddl in [
        "string collate `UTF8_LCASE`",
        "string collate `system`.`builtin`.`utf8_lcase`",
        "string collate `system`.builtin.`utf8_lcase`",
        "string collate UNICODE ",
        "time",
    ] {
        let server = session
            .parse_ddl(ddl)
            .unwrap_or_else(|e| panic!("{ddl}: {e:?}"));
        assert_eq!(DataType::from_ddl(ddl).unwrap(), server, "{ddl}");
    }
    for ddl in [
        "string collate `system`.`builtin`",
        "string collate `UNI``CODE`",
        "string collate `été`",
        "string collate été",
        "string collate `system.builtin.utf8_lcase`",
        "null",
    ] {
        assert!(session.parse_ddl(ddl).is_err(), "{ddl}");
        assert!(DataType::from_ddl(ddl).is_err(), "{ddl}");
    }
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

/// What one of the two parsers of type strings makes of a string: a type, or an error.
type Parsed = Option<DataType>;

/// `text` as the JSON parser reads it, the string of a type of `DataType::from_json`.
fn parsed_as_json(text: &str) -> Parsed {
    DataType::from_json(&serde_json::Value::String(text.to_string())).ok()
}

/// `text` as the DDL parser reads it. It is read alone and as the element of an array, where it
/// cannot be taken for a `name type` field, and the two have to agree.
fn parsed_as_ddl(text: &str) -> Parsed {
    let alone = DataType::from_ddl(text).ok();
    let element = match DataType::from_ddl(&format!("array<{text}>")) {
        Ok(DataType::Array { element_type, .. }) => Some(*element_type),
        Ok(other) => panic!("array<{text}> is not an array: {other:?}"),
        Err(_) => None,
    };
    assert_eq!(alone, element, "DDL {text:?} alone and in an array");
    alone
}

/// The JSON and the DDL parser share one parser of each atomic and parameterized form, so a form
/// is the same type, or the same error, of both. The exceptions are the differences of the two
/// grammars of PySpark: those of `_parse_datatype_json_value`, which reads JSON on the client,
/// and of the server's parser, which reads DDL. Each row of the tables is fed through both.
#[test]
fn json_and_ddl_parsers_agree_on_the_atomic_and_parameterized_forms() {
    let decimal = |precision, scale| DataType::Decimal { precision, scale };
    let string = |collation: &str| DataType::String {
        collation: collation.to_string(),
    };
    let year_month = |start_field, end_field| DataType::YearMonthInterval {
        start_field,
        end_field,
    };
    let day_time = |start_field, end_field| DataType::DayTimeInterval {
        start_field,
        end_field,
    };

    // The same type of both parsers.
    let same: Vec<(&str, DataType)> = vec![
        ("void", DataType::Null),
        ("boolean", DataType::Boolean),
        ("byte", DataType::Byte),
        ("short", DataType::Short),
        ("integer", DataType::Integer),
        ("long", DataType::Long),
        ("float", DataType::Float),
        ("double", DataType::Double),
        ("string", string("UTF8_BINARY")),
        ("binary", DataType::Binary),
        ("date", DataType::Date),
        ("timestamp", DataType::Timestamp),
        ("timestamp_ntz", DataType::TimestampNtz),
        ("variant", DataType::Variant),
        // The defaults of the bare names.
        ("decimal", decimal(10, 0)),
        ("interval", DataType::CalendarInterval),
        // decimal(p,s), with whitespace around its arguments.
        ("decimal(10,2)", decimal(10, 2)),
        ("decimal( 10 , 2 )", decimal(10, 2)),
        ("decimal(38,18)", decimal(38, 18)),
        // Neither parser checks the range of a precision or a scale, as PySpark's does not: the
        // server refuses the type, and the conversion to Arrow refuses what Arrow cannot hold.
        ("decimal(39,0)", decimal(39, 0)),
        ("decimal(300,0)", decimal(300, 0)),
        ("decimal(5,6)", decimal(5, 6)),
        ("decimal(0,0)", decimal(0, 0)),
        // char(n), varchar(n) and time(p).
        ("char(3)", DataType::Char { length: 3 }),
        ("char( 3 )", DataType::Char { length: 3 }),
        ("char(0)", DataType::Char { length: 0 }),
        ("varchar(20)", DataType::Varchar { length: 20 }),
        ("varchar( 20 )", DataType::Varchar { length: 20 }),
        ("time(0)", DataType::Time { precision: 0 }),
        ("time( 3 )", DataType::Time { precision: 3 }),
        ("time(6)", DataType::Time { precision: 6 }),
        ("time(9)", DataType::Time { precision: 9 }),
        // The nanosecond timestamps.
        (
            "timestamp_ntz(9)",
            DataType::TimestampNtzNanos { precision: 9 },
        ),
        (
            "timestamp_ltz(7)",
            DataType::TimestampLtzNanos { precision: 7 },
        ),
        // Intervals of one field, and of a range whose end is later than its start.
        ("interval year", year_month(0, 0)),
        ("interval month", year_month(1, 1)),
        ("interval year to month", year_month(0, 1)),
        ("interval day", day_time(0, 0)),
        ("interval hour", day_time(1, 1)),
        ("interval minute", day_time(2, 2)),
        ("interval second", day_time(3, 3)),
        ("interval day to hour", day_time(0, 1)),
        ("interval day to second", day_time(0, 3)),
        ("interval hour to minute", day_time(1, 2)),
        ("interval minute to second", day_time(2, 3)),
        // A collation that is written in its canonical form, of any whitespace.
        ("string collate UTF8_BINARY", string("UTF8_BINARY")),
        ("string collate UTF8_LCASE", string("UTF8_LCASE")),
        ("string collate UNICODE_CI", string("UNICODE_CI")),
        ("string  collate\tUNICODE", string("UNICODE")),
        (
            "string collate sr_Cyrl_SRB_CI_AI",
            string("sr_Cyrl_SRB_CI_AI"),
        ),
        // The spatial types, of an SRID or of a CRS.
        ("geometry(4326)", DataType::Geometry { srid: 4326 }),
        ("geometry(any)", DataType::Geometry { srid: -1 }),
        ("geography(4326)", DataType::Geography { srid: 4326 }),
    ];
    for (text, expected) in &same {
        assert_eq!(
            parsed_as_json(text).as_ref(),
            Some(expected),
            "JSON {text:?}"
        );
        assert_eq!(parsed_as_ddl(text).as_ref(), Some(expected), "DDL {text:?}");
    }

    // An error of both parsers.
    let neither = [
        // An argument is of digits alone: no sign, but for the scale of a decimal of JSON.
        "decimal()",
        "decimal(,2)",
        "decimal(a,2)",
        "decimal(10,b)",
        "decimal(10,2,3)",
        "decimal(-5,2)",
        "decimal(+5,2)",
        "decimal(10,+2)",
        "decimal(10,2",
        "decimal(10 2)",
        "decimal(1 0,2)",
        "decimal(99999999999,0)",
        "char()",
        "char(x)",
        "char(-3)",
        "char(+3)",
        "char(3,4)",
        "char(3",
        "varchar()",
        "varchar(-1)",
        "varchar(+1)",
        "time()",
        "time(a)",
        "time(-1)",
        "time(+6)",
        "timestamp_ntz(3)",
        "timestamp_ltz(10)",
        "timestamp_ntz(+9)",
        // An interval is of one family, and of nothing more.
        "interval foo",
        "intervalday",
        "interval year to day",
        "interval day to month",
        "interval day to",
        "interval to second",
        "interval day second",
        "interval day to second to hour",
        "interval day to second xyz",
        // A collated string names one collation, of letters, digits and underscores.
        "string collate",
        "string collate ",
        "string collateUNICODE",
        "string collate UNICODE CI",
        "string collate UNI-CODE",
        "string collate 'UNICODE'",
        "string UNICODE",
        "string collate other.builtin.UNICODE",
        "string collate `UNI``CODE`",
        "string collate `été`",
        "string collate `system`.`builtin`",
        "string collate `system.builtin.utf8_lcase`",
        "string collate `system`.`builtin`.`utf8_lcase`.`x`",
        "char(5) collate UNICODE",
        // Not a type.
        "",
        "foo",
        "decimals",
        "strings",
    ];
    for text in neither {
        assert_eq!(parsed_as_json(text), None, "JSON {text:?}");
        // An empty DDL string alone is the empty schema, so it is read as an element only.
        if text.is_empty() {
            assert!(DataType::from_ddl("array<>").is_err());
        } else {
            assert_eq!(parsed_as_ddl(text), None, "DDL {text:?}");
        }
    }

    // The differences of the two grammars: (the string, of JSON, of DDL).
    let differ: Vec<(&str, Parsed, Parsed)> = vec![
        // JSON names a decimal with its scale (PySpark's `_FIXED_DECIMAL`); Spark's `decimal(p)`
        // has scale 0.
        ("decimal(5)", None, Some(decimal(5, 0))),
        ("decimal( 5 )", None, Some(decimal(5, 0))),
        // The JSON of `DecimalType(10, -2)` has its negative scale; an argument of Spark's
        // grammar is an unsigned integer.
        ("decimal(10,-2)", Some(decimal(10, -2)), None),
        ("decimal(10, -2)", Some(decimal(10, -2)), None),
        // PySpark reads the JSON of an interval of any two fields of a family, which is that
        // of the type of its constructor, as `DayTimeIntervalType(3, 0)`; the end field of
        // Spark's grammar is later than the start field.
        ("interval second to day", Some(day_time(3, 0)), None),
        ("interval hour to day", Some(day_time(1, 0)), None),
        ("interval day to day", Some(day_time(0, 0)), None),
        ("interval month to year", Some(year_month(1, 0)), None),
        ("interval year to year", Some(year_month(0, 0)), None),
        // PySpark keeps the collation of JSON as it is written; the server gives the collation
        // of DDL in its canonical form, and takes it qualified.
        (
            "string collate unicode_ci",
            Some(string("unicode_ci")),
            Some(string("UNICODE_CI")),
        ),
        (
            "string collate utf8_binary",
            Some(string("utf8_binary")),
            Some(string("UTF8_BINARY")),
        ),
        (
            "string collate SR_CYRL_SRB_ci",
            Some(string("SR_CYRL_SRB_ci")),
            Some(string("sr_Cyrl_SRB_CI")),
        ),
        (
            "string collate system.builtin.unicode_ci",
            None,
            Some(string("UNICODE_CI")),
        ),
        (
            "string collate SYSTEM.BUILTIN.UTF8_LCASE",
            None,
            Some(string("UTF8_LCASE")),
        ),
        // DDL takes a name in backquotes, as Spark's grammar does, and reads each part of a
        // qualified name; JSON has no quoting.
        (
            "string collate `UTF8_LCASE`",
            None,
            Some(string("UTF8_LCASE")),
        ),
        (
            "string collate `system`.`builtin`.`utf8_lcase`",
            None,
            Some(string("UTF8_LCASE")),
        ),
        (
            "string collate `system`.builtin.`utf8_lcase`",
            None,
            Some(string("UTF8_LCASE")),
        ),
        ("string collate `UNICODE`", None, Some(string("UNICODE"))),
        // JSON reads a name of letters, digits and underscores, as PySpark's `\w+` does, and a
        // letter of any script is one. DDL's unquoted identifiers are of ASCII letters, digits and
        // underscores only, as Spark's lexer reads them.
        ("string collate été", Some(string("été")), None),
        // JSON is in lower case and of single spaces; the keywords of DDL are of any case and
        // any whitespace.
        ("DECIMAL(10,2)", None, Some(decimal(10, 2))),
        ("decimal (10,2)", None, Some(decimal(10, 2))),
        ("Char(3)", None, Some(DataType::Char { length: 3 })),
        ("TIME(3)", None, Some(DataType::Time { precision: 3 })),
        ("INTERVAL DAY TO SECOND", None, Some(day_time(0, 3))),
        ("interval  day\tto second", None, Some(day_time(0, 3))),
        (
            "STRING COLLATE UNICODE_CI",
            None,
            Some(string("UNICODE_CI")),
        ),
        (" decimal(10,2)", None, Some(decimal(10, 2))),
        // The names of Spark's grammar alone: the server reads `time` as `time(6)`, which JSON
        // names only with its precision, as PySpark's `TimeType.jsonValue` does.
        ("time", None, Some(DataType::Time { precision: 6 })),
        ("tinyint", None, Some(DataType::Byte)),
        ("smallint", None, Some(DataType::Short)),
        ("int", None, Some(DataType::Integer)),
        ("bigint", None, Some(DataType::Long)),
        ("real", None, Some(DataType::Float)),
        ("dec(5,1)", None, Some(decimal(5, 1))),
        ("numeric(5,1)", None, Some(decimal(5, 1))),
        ("character(3)", None, Some(DataType::Char { length: 3 })),
        ("timestamp_ltz", None, Some(DataType::Timestamp)),
        ("timestamp_ntz(6)", None, Some(DataType::TimestampNtz)),
        // The names of JSON alone. The server's JSON parser reads `null` for the schemas written
        // before NullType was named `void` (its `nameToType`, "for backwards compatibility"), and
        // its DDL parser has no such name. PySpark rejects it.
        ("null", Some(DataType::Null), None),
        ("calendarinterval", Some(DataType::CalendarInterval), None),
        (
            "unparsed(x)",
            Some(DataType::Unparsed {
                data_type_string: "x".to_string(),
            }),
            None,
        ),
    ];
    for (text, json, ddl) in &differ {
        assert_ne!(json, ddl, "{text:?} is not a difference");
        assert_eq!(&parsed_as_json(text), json, "JSON {text:?}");
        assert_eq!(&parsed_as_ddl(text), ddl, "DDL {text:?}");
    }

    // Whitespace after a form that PySpark's patterns match as a prefix is not a part of it, and
    // is accepted: the JSON of PySpark's `_parse_datatype_json_value` is `re.match`, and DDL is
    // trimmed. A bare name is matched whole, so whitespace after it is not a type in JSON.
    let trailing: Vec<(&str, DataType)> = vec![
        ("string collate UNICODE ", string("UNICODE")),
        ("string collate UNICODE\t", string("UNICODE")),
        ("interval day ", day_time(0, 0)),
        ("interval day to second ", day_time(0, 3)),
        ("interval year to month ", year_month(0, 1)),
        ("decimal(10,2) ", decimal(10, 2)),
        ("char(3) ", DataType::Char { length: 3 }),
        ("varchar(20) ", DataType::Varchar { length: 20 }),
        ("time(6) ", DataType::Time { precision: 6 }),
    ];
    for (text, expected) in &trailing {
        assert_eq!(
            parsed_as_json(text).as_ref(),
            Some(expected),
            "JSON {text:?}"
        );
        assert_eq!(parsed_as_ddl(text).as_ref(), Some(expected), "DDL {text:?}");
    }
    // A JSON-only trailing whitespace: the negative scale of JSON, and bare names.
    assert_eq!(
        parsed_as_json("decimal(10,-2) "),
        Some(decimal(10, -2)),
        "JSON decimal(10,-2) with trailing whitespace"
    );
    for text in [
        "integer ",
        "void ",
        "decimal ",
        "interval ",
        "string ",
        "time ",
    ] {
        assert_eq!(parsed_as_json(text), None, "JSON {text:?}");
    }

    // The JSON of a type is read back as the type, also of an interval that DDL cannot name.
    for data_type in [
        decimal(10, -2),
        decimal(300, 0),
        day_time(3, 0),
        year_month(1, 0),
        string("unicode_ci"),
    ] {
        assert_eq!(
            DataType::from_json(&data_type.json_value()).unwrap(),
            data_type
        );
    }
}

/// A backquoted collation name in DDL is unquoted, as Spark's parser unquotes it, also in a struct
/// field, where a comment follows the type.
#[test]
fn from_ddl_unquotes_a_backquoted_collation_name_of_a_field() {
    let dt = DataType::from_ddl("a: string collate `UTF8_LCASE` COMMENT 'x'")
        .expect("a backquoted collation name of a field");
    match dt {
        DataType::Struct { fields } => {
            assert_eq!(fields.len(), 1);
            assert_eq!(fields[0].name, "a");
            assert_eq!(
                fields[0].data_type,
                DataType::String {
                    collation: "UTF8_LCASE".to_string()
                }
            );
        }
        other => panic!("expected struct, got {other:?}"),
    }
}
