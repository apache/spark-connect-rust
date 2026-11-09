//! Behavioral e2e coverage of DataFrame methods not exercised elsewhere: results are
//! asserted against reference pyspark semantics. Covers the untested reachable
//! surface of dataframe.rs (set ops, dedup, joins, reshape, metadata, JSON, temp
//! views, semantics). Run with SPARK_REMOTE set.

use spark_connect::column::{col, lit, lit_string};
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

/// The column names of `df`, then its rows as display strings, sorted.
fn named_rows(df: &DataFrame) -> (Vec<String>, Vec<Vec<String>>) {
    let columns = df.columns().unwrap();
    let mut rows: Vec<Vec<String>> = df
        .collect()
        .unwrap()
        .iter()
        .map(|row| {
            (0..columns.len())
                .map(|i| row.get(i).map_or_else(String::new, |v| v.to_string()))
                .collect()
        })
        .collect();
    rows.sort();
    (columns, rows)
}

/// `GroupedData.sum`, `avg`, `mean`, `min` and `max` aggregate every numeric column without
/// columns, and reject a given column that is not a numeric one, as PySpark's `_numeric_agg`
/// does, for groupBy, rollup, cube, grouping sets and pivot.
#[test]
fn group_numeric_aggregates_follow_pyspark() {
    use spark_connect::row::Value;
    use spark_connect::SparkErrorKind;
    if !should_run() {
        return;
    }
    let s = session();
    // k: grouping string; i int, l long, d double (NULL in group b), s string, p pivot string.
    let df = s
        .sql(
            "SELECT * FROM VALUES ('a', 1, 10L, 1.5D, 'x', 'p'), ('a', 2, 20L, 2.5D, 'y', 'q'), \
             ('b', 3, 30L, CAST(NULL AS DOUBLE), 'z', 'p') AS t(k, i, l, d, s, p)",
        )
        .unwrap();
    let names = |names: &[&str]| names.iter().map(|n| n.to_string()).collect::<Vec<String>>();
    let rows = |rows: &[&[&str]]| {
        rows.iter()
            .map(|r| r.iter().map(|v| v.to_string()).collect::<Vec<String>>())
            .collect::<Vec<Vec<String>>>()
    };

    // Every numeric column, in schema order, with PySpark's names (`sum(i)`, ...).
    let (cols, got) = named_rows(&df.group_by(vec![col("k")]).sum(vec![]).unwrap());
    assert_eq!(cols, names(&["k", "sum(i)", "sum(l)", "sum(d)"]));
    assert_eq!(
        got,
        rows(&[&["a", "3", "30", "4"], &["b", "3", "30", "NULL"]])
    );
    let (cols, got) = named_rows(&df.group_by(vec![col("k")]).avg(vec![]).unwrap());
    assert_eq!(cols, names(&["k", "avg(i)", "avg(l)", "avg(d)"]));
    assert_eq!(
        got,
        rows(&[&["a", "1.5", "15", "2"], &["b", "3", "30", "NULL"]])
    );
    let (cols, got) = named_rows(&df.group_by(vec![col("k")]).mean(vec![]).unwrap());
    assert_eq!(cols, names(&["k", "avg(i)", "avg(l)", "avg(d)"]));
    assert_eq!(
        got,
        rows(&[&["a", "1.5", "15", "2"], &["b", "3", "30", "NULL"]])
    );
    let (cols, got) = named_rows(&df.group_by(vec![col("k")]).min(vec![]).unwrap());
    assert_eq!(cols, names(&["k", "min(i)", "min(l)", "min(d)"]));
    assert_eq!(
        got,
        rows(&[&["a", "1", "10", "1.5"], &["b", "3", "30", "NULL"]])
    );
    let (cols, got) = named_rows(&df.group_by(vec![col("k")]).max(vec![]).unwrap());
    assert_eq!(cols, names(&["k", "max(i)", "max(l)", "max(d)"]));
    assert_eq!(
        got,
        rows(&[&["a", "2", "20", "2.5"], &["b", "3", "30", "NULL"]])
    );

    // A given numeric column only.
    let (cols, got) = named_rows(&df.group_by(vec![col("k")]).sum(vec!["d", "i"]).unwrap());
    assert_eq!(cols, names(&["k", "sum(d)", "sum(i)"]));
    assert_eq!(got, rows(&[&["a", "4", "3"], &["b", "NULL", "3"]]));

    // Rollup, cube and grouping sets (`GROUPING SETS ((k), ())` is `ROLLUP(k)`).
    let (cols, got) = named_rows(&df.rollup(vec![col("k")]).sum(vec![]).unwrap());
    assert_eq!(cols, names(&["k", "sum(i)", "sum(l)", "sum(d)"]));
    assert_eq!(
        got,
        rows(&[
            &["NULL", "6", "60", "4"],
            &["a", "3", "30", "4"],
            &["b", "3", "30", "NULL"]
        ])
    );
    // With one grouping column, a cube is a rollup.
    assert_eq!(
        named_rows(&df.cube(vec![col("k")]).sum(vec![]).unwrap()),
        named_rows(&df.rollup(vec![col("k")]).sum(vec![]).unwrap())
    );
    let grouping_sets = df
        .grouping_sets(vec![vec![col("k")], vec![]])
        .sum(vec![])
        .unwrap();
    assert_eq!(
        named_rows(&grouping_sets).1,
        named_rows(&df.rollup(vec![col("k")]).sum(vec![]).unwrap()).1
    );

    // Pivot: the numeric columns of each pivot value, or the given one.
    let pivot = |grouped: spark_connect::group::GroupedData| {
        grouped
            .pivot(
                col("p"),
                Some(vec![Value::String("p".into()), Value::String("q".into())]),
            )
            .unwrap()
    };
    let (cols, got) = named_rows(&pivot(df.group_by(vec![col("k")])).sum(vec![]).unwrap());
    assert_eq!(
        cols,
        names(&["k", "p_sum(i)", "p_sum(l)", "p_sum(d)", "q_sum(i)", "q_sum(l)", "q_sum(d)"])
    );
    assert_eq!(
        got,
        rows(&[
            &["a", "1", "10", "1.5", "2", "20", "2.5"],
            &["b", "3", "30", "NULL", "NULL", "NULL", "NULL"]
        ])
    );
    let (cols, got) = named_rows(&pivot(df.group_by(vec![col("k")])).sum(vec!["i"]).unwrap());
    assert_eq!(cols, names(&["k", "p", "q"]));
    assert_eq!(got, rows(&[&["a", "1", "2"], &["b", "3", "NULL"]]));

    // A column that is not numeric, or not in the schema, is NOT_NUMERIC_COLUMNS with the
    // invalid columns in PySpark's order.
    let check = |err: spark_connect::SparkError, invalid: &str| {
        assert_eq!(err.kind, SparkErrorKind::TypeError);
        assert_eq!(err.get_condition().as_deref(), Some("NOT_NUMERIC_COLUMNS"));
        assert_eq!(
            err.params.get("invalid_columns").map(String::as_str),
            Some(invalid)
        );
    };
    check(
        df.group_by(vec![col("k")])
            .sum(vec!["s"])
            .err()
            .expect("a string column"),
        "['s']",
    );
    check(
        df.group_by(vec![col("k")])
            .min(vec!["i", "nope", "s"])
            .err()
            .expect("a missing and a string column"),
        "['nope', 's']",
    );
    check(
        pivot(df.group_by(vec![col("k")]))
            .sum(vec!["s"])
            .err()
            .expect("a string column of a pivot"),
        "['s']",
    );
    check(
        df.rollup(vec![col("k")])
            .avg(vec!["s"])
            .err()
            .expect("a string column of a rollup"),
        "['s']",
    );
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
    // exists / scalar: the EXISTS and scalar subqueries of a DataFrame, as PySpark's.
    let one = s.sql("SELECT 42 AS v").unwrap();
    let row = &s
        .range(1)
        .unwrap()
        .select(vec![df.exists(), one.scalar()])
        .collect()
        .unwrap()[0];
    assert_eq!(row.get(0), Some(&spark_connect::row::Value::Bool(true)));
    assert_eq!(row.get(1), Some(&spark_connect::row::Value::Integer(42)));
    // semantics (server-backed AnalyzePlan): two identical plans are equal
    assert!(df.same_semantics(&df4(&s)).unwrap());
    let _ = df.semantic_hash().unwrap();
    // flags
    assert!(!df.is_streaming().unwrap());
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
    let parts = df.random_split(vec![0.5, 0.5], Some(1)).unwrap();
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
                Value::Map(vec![
                    (Value::String("k".into()), value),
                    (Value::String("null".into()), Value::Null),
                ]),
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

/// The rows of `df`, as (i32, i32) pairs, sorted.
fn int_pairs(df: &DataFrame) -> Vec<(i32, i32)> {
    use spark_connect::row::Value;
    let mut pairs: Vec<(i32, i32)> = df
        .collect()
        .unwrap()
        .iter()
        .map(|row| match (row.get(0), row.get(1)) {
            (Some(Value::Integer(a)), Some(Value::Integer(b))) => (*a, *b),
            other => panic!("unexpected row {other:?}"),
        })
        .collect();
    pairs.sort();
    pairs
}

/// Reads `relation` as a DataFrame, polling until the server no longer has it, as a release
/// sent without waiting takes effect. Returns the condition of the error the read fails with.
fn wait_until_relation_is_released(s: &SparkSession, relation: &spark_connect_proto::Relation) {
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
    loop {
        if let Err(error) = DataFrame::from_sent_relation(s.clone(), relation.clone()).count() {
            assert_eq!(
                error.get_condition().as_deref(),
                Some("CONNECT_INVALID_PLAN.DATAFRAME_NOT_FOUND"),
                "the released relation is not found: {error}"
            );
            return;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "the server still has the relation of the dropped checkpoint"
        );
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}

#[test]
fn a_dropped_checkpoint_releases_its_relation_on_the_server() {
    // As PySpark's CachedRemoteRelation.__del__: the last DataFrame of a checkpoint that is
    // dropped removes the relation on the server, and only then is it gone.
    if !should_run() {
        return;
    }
    let s = session();
    // checkpoint() needs a checkpoint directory, which the test server does not set.
    let checkpoint = s.range(5).unwrap().local_checkpoint().unwrap();
    let relation = checkpoint.to_relation();
    assert_eq!(checkpoint.count().unwrap(), 5);
    drop(checkpoint);
    wait_until_relation_is_released(&s, &relation);
}

#[test]
fn a_checkpoint_stays_while_a_dataframe_derived_from_it_is_alive() {
    // The derived DataFrame keeps the checkpoint alive: it still counts after the checkpoint
    // itself is dropped, and the relation is released only once the derived one is dropped.
    if !should_run() {
        return;
    }
    let s = session();
    let checkpoint = s.range(5).unwrap().local_checkpoint().unwrap();
    let relation = checkpoint.to_relation();
    let derived = checkpoint.filter(col("id").gt(lit(1)));
    drop(checkpoint);
    assert_eq!(derived.count().unwrap(), 3);
    assert_eq!(
        DataFrame::from_sent_relation(s.clone(), relation.clone())
            .count()
            .unwrap(),
        5,
        "the relation is still on the server while the derived DataFrame lives"
    );
    drop(derived);
    wait_until_relation_is_released(&s, &relation);
}

#[test]
fn subqueries_return_the_rows_of_pyspark() {
    // As PySpark's DataFrame.scalar(), DataFrame.exists() and Column.isin(df): scalar,
    // correlated EXISTS and IN subqueries, also nested and in other operators.
    if !should_run() {
        return;
    }
    let s = session();
    let t1 = s
        .sql("SELECT * FROM VALUES (1, 2), (3, 4) AS t1(a, b)")
        .unwrap()
        .alias("t1");
    let t2 = s
        .sql("SELECT * FROM VALUES (1, 5), (5, 6), (3, 7) AS t2(a, c)")
        .unwrap()
        .alias("t2");

    let largest = t2.select(vec![f::max(col("c"))]).scalar();
    let with_largest = t1.select(vec![col("a"), largest.clone().alias("m")]);
    assert_eq!(int_pairs(&with_largest), vec![(1, 7), (3, 7)]);
    // A DataFrame with a subquery is the base, and the subquery, of others.
    assert_eq!(
        int_pairs(&with_largest.filter(col("m").gt(col("a")))),
        vec![(1, 7), (3, 7)]
    );
    assert_eq!(
        int_pairs(&t2.filter(col("a").isin_subquery(&with_largest.select(vec![col("a")])))),
        vec![(1, 5), (3, 7)]
    );

    let correlated = t2.filter(col("t2.a").eq(col("t1.a")));
    let largest_of_a = correlated.select(vec![f::max(col("c"))]).scalar();
    assert_eq!(
        int_pairs(&t1.select(vec![col("a"), largest_of_a.alias("m")])),
        vec![(1, 5), (3, 7)]
    );
    assert_eq!(
        int_pairs(&t1.filter(correlated.exists())),
        vec![(1, 2), (3, 4)]
    );
    assert_eq!(int_pairs(&t1.filter(correlated.exists().not())), vec![]);

    let smallest = t2.select(vec![f::min(col("c"))]).scalar();
    let above_smallest = t2.filter(col("c").gt(smallest)).select(vec![col("a")]);
    assert_eq!(
        int_pairs(&t1.filter(col("a").isin_subquery(&above_smallest))),
        vec![(3, 4)]
    );
    let pairs = s.sql("SELECT 1 AS x, 2 AS y").unwrap();
    assert_eq!(
        int_pairs(&t1.filter(f::r#struct(vec![col("a"), col("b")]).isin_subquery(&pairs))),
        vec![(1, 2)]
    );
}

/// A value over the 128 MiB message limit arrives as the chunks of its Arrow batch, which
/// `collect` joins.
#[test]
fn collect_joins_the_chunks_of_a_value_over_the_message_limit() {
    if !should_run() {
        return;
    }
    let s = session();
    let len: i64 = 129 * 1024 * 1024;
    let df = s
        .range(1)
        .expect("range")
        .select(vec![f::repeat(lit_string("x"), lit(len))]);
    let rows = df.collect().unwrap();
    assert_eq!(rows.len(), 1);
    let value = rows[0].get(0).unwrap().as_str().unwrap();
    assert_eq!(value.len(), len as usize);
    assert!(value.bytes().all(|b| b == b'x'));
}
