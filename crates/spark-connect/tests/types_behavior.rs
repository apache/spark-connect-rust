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
fn simple_string_of_the_ml_udts_matches_pyspark() {
    // PySpark's VectorUDT and MatrixUDT name their simple strings, as a schema of ML features
    // prints them. Another UDT keeps "udt".
    use spark_connect::types::StructField;
    let udt = |python_class: Option<&str>| DataType::Udt {
        type_str: "udt".to_string(),
        jvm_class: None,
        python_class: python_class.map(str::to_string),
        serialized_python_class: None,
        sql_type: Some(Box::new(DataType::Integer)),
    };
    assert_eq!(s(&udt(Some("pyspark.ml.linalg.VectorUDT"))), "vector");
    assert_eq!(s(&udt(Some("pyspark.ml.linalg.MatrixUDT"))), "matrix");
    assert_eq!(
        s(&udt(Some("pyspark.testing.sqlutils.ExamplePointUDT"))),
        "udt"
    );
    assert_eq!(s(&udt(None)), "udt");
    let schema = DataType::Struct {
        fields: vec![StructField {
            name: "features".to_string(),
            data_type: udt(Some("pyspark.ml.linalg.VectorUDT")),
            nullable: true,
            metadata: Default::default(),
        }],
    };
    assert_eq!(s(&schema), "struct<features:vector>");
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
    assert_eq!(st.to_ddl().unwrap(), "a INT,b STRING NOT NULL");
    let tree = st.tree_string().unwrap();
    assert!(tree.starts_with("root\n"));
    assert!(tree.contains("a: integer (nullable = true)"));
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

/// The expected strings are the output of `StructType.toDDL()` of PySpark 4.3.0 on a server.
#[test]
fn struct_to_ddl_quotes_names_and_writes_sql_types() {
    use spark_connect::types::StructField;
    use std::collections::BTreeMap;

    fn field(name: &str, data_type: DataType, nullable: bool) -> StructField {
        StructField {
            name: name.to_string(),
            data_type,
            nullable,
            metadata: BTreeMap::new(),
        }
    }
    fn commented(name: &str, data_type: DataType, nullable: bool, comment: &str) -> StructField {
        let mut f = field(name, data_type, nullable);
        f.metadata
            .insert("comment".to_string(), serde_json::json!(comment));
        f
    }
    fn ddl(fields: Vec<StructField>) -> String {
        DataType::Struct { fields }.to_ddl().unwrap()
    }
    let int = || DataType::Integer;
    let string = || DataType::String {
        collation: "UTF8_BINARY".to_string(),
    };

    // Names are quoted unless they are `[a-zA-Z_][a-zA-Z0-9_]*`; a backtick is doubled.
    assert_eq!(
        ddl(vec![
            field("my col", int(), true),
            field("a`b", int(), true),
            field("a.b", int(), true),
            field("1abc", int(), true),
            field("123", int(), true),
            field("_x", int(), true),
            field("ünï", int(), true),
        ]),
        "`my col` INT,`a``b` INT,`a.b` INT,`1abc` INT,`123` INT,_x INT,`ünï` INT"
    );
    assert_eq!(
        ddl(vec![commented("x", int(), false, "it's")]),
        "x INT NOT NULL COMMENT 'it\\'s'"
    );
    // Nested struct fields are written as `name: TYPE`, with the same quoting and suffixes.
    assert_eq!(
        ddl(vec![field(
            "s",
            DataType::Struct {
                fields: vec![commented("my col", int(), false, "c")],
            },
            true,
        )]),
        "s STRUCT<`my col`: INT NOT NULL COMMENT 'c'>"
    );
    assert_eq!(
        ddl(vec![field(
            "arr",
            DataType::Array {
                element_type: Box::new(DataType::Struct {
                    fields: vec![field("a b", int(), true)],
                }),
                contains_null: true,
            },
            true,
        )]),
        "arr ARRAY<STRUCT<`a b`: INT>>"
    );
    assert_eq!(
        ddl(vec![field(
            "m",
            DataType::Map {
                key_type: Box::new(string()),
                value_type: Box::new(DataType::Struct {
                    fields: vec![field("k`v", DataType::Long, false)],
                }),
                value_contains_null: true,
            },
            true,
        )]),
        "m MAP<STRING, STRUCT<`k``v`: BIGINT NOT NULL>>"
    );
    assert_eq!(
        ddl(vec![field(
            "a",
            DataType::Array {
                element_type: Box::new(DataType::Struct {
                    fields: vec![commented("q", string(), true, "x'y")],
                }),
                contains_null: true,
            },
            true,
        )]),
        "a ARRAY<STRUCT<q: STRING COMMENT 'x\\'y'>>"
    );
    // Atomic types are written as `DataType.sql`.
    assert_eq!(
        ddl(vec![
            field(
                "d",
                DataType::Decimal {
                    precision: 10,
                    scale: 2
                },
                true
            ),
            field("ntz", DataType::TimestampNtz, true),
            field(
                "ym",
                DataType::YearMonthInterval {
                    start_field: 0,
                    end_field: 1
                },
                true
            ),
            field(
                "dt",
                DataType::DayTimeInterval {
                    start_field: 0,
                    end_field: 3
                },
                true
            ),
            field("ch", DataType::Char { length: 3 }, true),
            field("vc", DataType::Varchar { length: 4 }, true),
            field("tm", DataType::Time { precision: 3 }, true),
            field("bin", DataType::Binary, true),
            field("b", DataType::Boolean, true),
            field("v", DataType::Variant, true),
        ]),
        "d DECIMAL(10,2),ntz TIMESTAMP_NTZ,ym INTERVAL YEAR TO MONTH,dt INTERVAL DAY TO SECOND,\
         ch CHAR(3),vc VARCHAR(4),tm TIME(3),bin BINARY,b BOOLEAN,v VARIANT"
    );
    // A non-default collation is written as `STRING COLLATE <name>`.
    assert_eq!(
        ddl(vec![field(
            "c",
            DataType::String {
                collation: "UTF8_LCASE".to_string(),
            },
            true,
        )]),
        "c STRING COLLATE UTF8_LCASE"
    );
    assert_eq!(
        ddl(vec![field(
            "s",
            DataType::Struct {
                fields: vec![field(
                    "c d",
                    DataType::String {
                        collation: "UNICODE_CI".to_string(),
                    },
                    true,
                )],
            },
            true,
        )]),
        "s STRUCT<`c d`: STRING COLLATE UNICODE_CI>"
    );
}

#[test]
fn from_ddl_canonicalizes_the_collation_name() {
    // The canonical names of the server of PySpark 4.3.0, from `createDataFrame([], "c STRING
    // COLLATE <name>")`. `en_us` is rejected by the server (COLLATION_INVALID_NAME), so `from_ddl`
    // only canonicalizes its case, `en_US`.
    let cases = [
        ("utf8_lcase", "UTF8_LCASE"),
        ("Unicode_Ci_Ai", "UNICODE_CI_AI"),
        ("unicode_rtrim", "UNICODE_RTRIM"),
        ("utf8_lcase_rtrim", "UTF8_LCASE_RTRIM"),
        ("sr_ci_ai", "sr_CI_AI"),
        ("SR_CI_AI", "sr_CI_AI"),
        ("sr_CI_AI", "sr_CI_AI"),
        ("EN", "en"),
        ("de_ci", "de_CI"),
        ("zh_hant_hkg", "zh_Hant_HKG"),
        ("SR_CYRL_SRB_CI_AI", "sr_Cyrl_SRB_CI_AI"),
        ("en_us", "en_US"),
    ];
    for (written, canonical) in cases {
        assert_eq!(
            DataType::from_ddl(&format!("STRING COLLATE {written}")).unwrap(),
            DataType::String {
                collation: canonical.to_string(),
            },
            "STRING COLLATE {written}"
        );
    }
    match DataType::from_ddl("a STRING COLLATE UTF8_LCASE").unwrap() {
        DataType::Struct { fields } => assert_eq!(
            fields[0].data_type,
            DataType::String {
                collation: "UTF8_LCASE".to_string(),
            }
        ),
        other => panic!("expected struct, got {other:?}"),
    }
    assert_eq!(
        DataType::from_ddl("array<string collate UNICODE_CI>").unwrap(),
        DataType::Array {
            element_type: Box::new(DataType::String {
                collation: "UNICODE_CI".to_string(),
            }),
            contains_null: true,
        }
    );
    // The DDL written by `to_ddl` parses back to the same collation.
    let st = DataType::Struct {
        fields: vec![spark_connect::types::StructField {
            name: "a".to_string(),
            data_type: DataType::String {
                collation: "UTF8_LCASE".to_string(),
            },
            nullable: true,
            metadata: Default::default(),
        }],
    };
    assert_eq!(DataType::from_ddl(&st.to_ddl().unwrap()).unwrap(), st);
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
    assert!(full.contains("inner: integer (nullable = true)"));
    // max_depth = 1 prints only the top level; the nested field is omitted.
    let shallow = nested.tree_string_with_depth(1).unwrap();
    assert!(shallow.contains("outer: struct (nullable = true)"));
    assert!(!shallow.contains("inner"));
}

#[test]
fn tree_string_uses_python_udt_class_names() {
    use spark_connect::types::StructField;
    let vector_udt = || DataType::Udt {
        type_str: "udt".to_string(),
        jvm_class: None,
        python_class: Some("pyspark.ml.linalg.VectorUDT".to_string()),
        serialized_python_class: None,
        sql_type: Some(Box::new(DataType::Integer)),
    };
    let schema = DataType::Struct {
        fields: vec![
            StructField {
                name: "features".to_string(),
                data_type: vector_udt(),
                nullable: true,
                metadata: Default::default(),
            },
            StructField {
                name: "array".to_string(),
                data_type: DataType::Array {
                    element_type: Box::new(vector_udt()),
                    contains_null: false,
                },
                nullable: false,
                metadata: Default::default(),
            },
            StructField {
                name: "map".to_string(),
                data_type: DataType::Map {
                    key_type: Box::new(DataType::String {
                        collation: String::new(),
                    }),
                    value_type: Box::new(vector_udt()),
                    value_contains_null: true,
                },
                nullable: true,
                metadata: Default::default(),
            },
        ],
    };

    assert_eq!(
        schema.tree_string().unwrap(),
        concat!(
            "root\n",
            " |-- features: vectorudt (nullable = true)\n",
            " |-- array: array (nullable = false)\n",
            " |    |-- element: vectorudt (containsNull = false)\n",
            " |-- map: map (nullable = true)\n",
            " |    |-- key: string\n",
            " |    |-- value: vectorudt (valueContainsNull = true)\n",
        )
    );
}

#[test]
fn tree_string_keeps_generic_udt_name_without_python_class() {
    use spark_connect::types::StructField;
    let schema = DataType::Struct {
        fields: vec![StructField {
            name: "features".to_string(),
            data_type: DataType::Udt {
                type_str: "udt".to_string(),
                jvm_class: Some("org.apache.spark.ml.linalg.VectorUDT".to_string()),
                python_class: None,
                serialized_python_class: None,
                sql_type: Some(Box::new(DataType::Integer)),
            },
            nullable: true,
            metadata: Default::default(),
        }],
    };

    assert_eq!(
        schema.tree_string().unwrap(),
        "root\n |-- features: udt (nullable = true)\n",
    );
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
