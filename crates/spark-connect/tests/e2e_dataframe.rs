//! Behavioral e2e coverage of DataFrame methods not exercised elsewhere: results are
//! asserted against reference pyspark semantics. Covers the untested reachable
//! surface of dataframe.rs (set ops, dedup, joins, reshape, metadata, JSON, temp
//! views, semantics). Run with SPARK_REMOTE set.

use spark_connect::column::{col, lit};
use spark_connect::dataframe::DataFrame;
use spark_connect::functions as f;
use spark_connect::plan::JoinType;
use spark_connect::session::SparkSession;

fn should_run() -> bool {
    std::env::var("SPARK_REMOTE").is_ok()
}
fn session() -> SparkSession {
    let url = std::env::var("SPARK_REMOTE").unwrap_or_else(|_| "sc://localhost:15002".to_string());
    SparkSession::builder()
        .remote(&url)
        .get_or_create()
        .expect("session")
}
// 4 rows, with a duplicate (2,'b') for dedup/set-op assertions.
fn df4(s: &SparkSession) -> DataFrame {
    s.sql("SELECT * FROM VALUES (1,'a'),(2,'b'),(2,'b'),(3,'c') AS t(id, name)")
        .expect("df4")
}

#[test]
fn dataframe_transforms_and_setops() {
    if !should_run() {
        return;
    }
    let s = session();
    let df = df4(&s);

    // where_ (alias of filter)
    assert_eq!(df.where_(col("id").gt(lit(1))).count().unwrap(), 3);
    // selectExpr
    let se = df.select_expr(vec!["id + 1 AS x"]);
    let xs: Vec<i64> = se
        .collect()
        .unwrap()
        .iter()
        .map(|r| r.get(0).unwrap().as_i64().unwrap())
        .collect();
    assert_eq!(xs, vec![2, 3, 3, 4]);
    // dedup
    assert_eq!(df.drop_duplicates(None).count().unwrap(), 3);
    assert_eq!(df.drop_duplicates(Some(vec!["id"])).count().unwrap(), 3);
    // set ops (all vs distinct)
    let other = s
        .sql("SELECT * FROM VALUES (2,'b'),(9,'z') AS t(id, name)")
        .unwrap();
    assert_eq!(df.union_all(&other).count().unwrap(), 6);
    assert_eq!(df.intersect_all(&other).count().unwrap(), 1); // one (2,'b')
    assert_eq!(df.except_all(&other).count().unwrap(), 3); // 4 - 1 matched
    assert_eq!(df.union_by_name(&other).count().unwrap(), 6);
    // join_using
    let right = s
        .sql("SELECT * FROM VALUES (1,100),(2,200) AS t(id, v)")
        .unwrap();
    assert_eq!(
        df.join_using(&right, vec!["id".to_string()], JoinType::Inner)
            .count()
            .unwrap(),
        3
    );
    // reshape / ordering
    assert_eq!(
        df.sort_within_partitions(vec![col("id").expression().clone()])
            .count()
            .unwrap(),
        4
    );
    assert_eq!(
        df.repartition_by_range(2, vec![col("id").expression().clone()])
            .count()
            .unwrap(),
        4
    );
    assert_eq!(df.to_schema(vec!["id", "name"]).count().unwrap(), 4);
    let m = df.melt(vec!["id"], Some(vec!["name"]), "var", "val");
    assert_eq!(m.count().unwrap(), 4);
}

#[test]
fn dataframe_metadata_json_semantics() {
    if !should_run() {
        return;
    }
    let s = session();
    let df = df4(&s);

    // take(n)
    assert_eq!(df.take(2).unwrap().len(), 2);
    // toJSON: one JSON string per row
    let json = df.to_json().unwrap();
    assert_eq!(json.len(), 4);
    // Each row is a JSON object carrying both columns.
    assert!(json
        .iter()
        .all(|j| j.starts_with('{') && j.contains("id") && j.contains("name")));
    // exists / scalar
    assert!(df.exists().unwrap());
    let one = s.sql("SELECT 42 AS v").unwrap();
    assert_eq!(
        one.scalar().unwrap(),
        Some(spark_connect::row::Value::Integer(42))
    );
    // semantics (server-backed AnalyzePlan): two identical plans are equal
    assert!(df.same_semantics(&df4(&s)).unwrap());
    let _ = df.semantic_hash().unwrap();
    // flags
    assert!(!df.is_streaming());
    let _ = df.is_local();
    // input_files (a sql df reads no files -> empty, but the call path runs)
    let _ = df.input_files().unwrap();
    // observe: attach an observed metric then trigger it
    let observed = df.observe("obs", vec![f::count(lit(1)).expression().clone()]);
    let _ = observed.collect().unwrap();
    // storage level / is_cached around cache
    let cached = df.cache().unwrap();
    let _ = cached.count().unwrap();
    let _ = cached.is_cached().unwrap();
    let _ = cached.storage_level().unwrap();
    // temp views (session + global)
    df.create_or_replace_temp_view("cov_df_view").unwrap();
    assert_eq!(s.table("cov_df_view").unwrap().count().unwrap(), 4);
    let _ = df.create_or_replace_global_temp_view("cov_df_gview");
    // random_split returns the requested number of frames
    let parts = df.random_split(vec![0.5, 0.5], Some(1));
    assert_eq!(parts.len(), 2);
}

#[test]
fn day_time_interval_input_roundtrips() {
    if !should_run() {
        return;
    }
    use spark_connect::row::{Row, Value};
    use spark_connect::types::{DataType, StructField};

    fn field(name: &str, data_type: DataType) -> StructField {
        StructField {
            name: name.into(),
            data_type,
            nullable: true,
            metadata: Default::default(),
        }
    }
    let s = session();
    let interval = DataType::DayTimeInterval {
        start_field: 0,
        end_field: 3,
    };
    let schema = DataType::Struct {
        fields: vec![
            field("interval", interval.clone()),
            field(
                "items",
                DataType::Array {
                    element_type: Box::new(interval.clone()),
                    contains_null: true,
                },
            ),
            field(
                "nested",
                DataType::Struct {
                    fields: vec![field("leaf", interval.clone())],
                },
            ),
            field(
                "mapping",
                DataType::Map {
                    key_type: Box::new(DataType::String {
                        collation: "UTF8_BINARY".into(),
                    }),
                    value_type: Box::new(interval),
                    value_contains_null: true,
                },
            ),
        ],
    };
    let names = vec![
        "interval".into(),
        "items".into(),
        "nested".into(),
        "mapping".into(),
    ];
    let mut rows = vec![Row::new(names.clone(), vec![Value::Null; 4])];
    for micros in [
        i64::MIN,
        i64::MIN + 1,
        -172_800_000_001,
        -1,
        0,
        1,
        i64::MAX - 1,
        i64::MAX,
    ] {
        let value = Value::DayTimeInterval(micros);
        rows.push(Row::new(
            names.clone(),
            vec![
                value.clone(),
                Value::List(vec![Value::Null, value.clone()]),
                Value::Struct(Row::new(vec!["leaf".into()], vec![value.clone()]).into()),
                Value::Map([("k".into(), value), ("null".into(), Value::Null)].into()),
            ],
        ));
    }
    let df = s.create_dataframe(rows.clone(), schema.clone()).unwrap();
    let collected = df.collect().unwrap();
    assert_eq!(collected, rows);
    assert_eq!(
        s.create_dataframe(collected, schema)
            .unwrap()
            .collect()
            .unwrap(),
        rows
    );

    let sql = s
        .sql("SELECT INTERVAL '-2 00:00:00.000001' DAY TO SECOND AS interval")
        .unwrap();
    let collected = sql.collect().unwrap();
    assert_eq!(
        collected[0].get(0),
        Some(&Value::DayTimeInterval(-172_800_000_001))
    );
    assert_eq!(
        s.create_dataframe(collected.clone(), sql.schema().unwrap())
            .unwrap()
            .collect()
            .unwrap(),
        collected
    );
}
