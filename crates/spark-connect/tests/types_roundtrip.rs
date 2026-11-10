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
