//! Behavioral e2e coverage of DataFrame methods not exercised elsewhere: results are
//! asserted against reference pyspark semantics. Covers the untested reachable
//! surface of dataframe.rs (set ops, dedup, joins, reshape, metadata, JSON, temp
//! views, semantics). Run with SPARK_REMOTE set.

mod common;

use common::{session, should_run};
use spark_connect::column::{col, lit};
use spark_connect::dataframe::DataFrame;
use spark_connect::functions as f;
use spark_connect::plan::JoinType;
use spark_connect::session::SparkSession;

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
            .unwrap()
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
    let (cols, got) = named_rows(&df.group_by(vec![col("k")]).sum(Vec::<&str>::new()).unwrap());
    assert_eq!(cols, names(&["k", "sum(i)", "sum(l)", "sum(d)"]));
    assert_eq!(
        got,
        rows(&[&["a", "3", "30", "4"], &["b", "3", "30", "NULL"]])
    );
    let (cols, got) = named_rows(&df.group_by(vec![col("k")]).avg(Vec::<&str>::new()).unwrap());
    assert_eq!(cols, names(&["k", "avg(i)", "avg(l)", "avg(d)"]));
    assert_eq!(
        got,
        rows(&[&["a", "1.5", "15", "2"], &["b", "3", "30", "NULL"]])
    );
    let (cols, got) = named_rows(
        &df.group_by(vec![col("k")])
            .mean(Vec::<&str>::new())
            .unwrap(),
    );
    assert_eq!(cols, names(&["k", "avg(i)", "avg(l)", "avg(d)"]));
    assert_eq!(
        got,
        rows(&[&["a", "1.5", "15", "2"], &["b", "3", "30", "NULL"]])
    );
    let (cols, got) = named_rows(&df.group_by(vec![col("k")]).min(Vec::<&str>::new()).unwrap());
    assert_eq!(cols, names(&["k", "min(i)", "min(l)", "min(d)"]));
    assert_eq!(
        got,
        rows(&[&["a", "1", "10", "1.5"], &["b", "3", "30", "NULL"]])
    );
    let (cols, got) = named_rows(&df.group_by(vec![col("k")]).max(Vec::<&str>::new()).unwrap());
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
    let (cols, got) = named_rows(&df.rollup(vec![col("k")]).sum(Vec::<&str>::new()).unwrap());
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
        named_rows(&df.cube(vec![col("k")]).sum(Vec::<&str>::new()).unwrap()),
        named_rows(&df.rollup(vec![col("k")]).sum(Vec::<&str>::new()).unwrap())
    );
    let grouping_sets = df
        .grouping_sets(vec![vec![col("k")], vec![]])
        .sum(Vec::<&str>::new())
        .unwrap();
    assert_eq!(
        named_rows(&grouping_sets).1,
        named_rows(&df.rollup(vec![col("k")]).sum(Vec::<&str>::new()).unwrap()).1
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
    let (cols, got) = named_rows(
        &pivot(df.group_by(vec![col("k")]))
            .sum(Vec::<&str>::new())
            .unwrap(),
    );
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

/// The numeric aggregates and the list methods take the names of their columns, and the
/// columns of their lists, as any iterable of them, as a Vec does: the server runs each form to
/// the same rows.
#[test]
fn list_methods_take_any_iterable_on_the_server() {
    if !should_run() {
        return;
    }
    let s = session();
    let df = s
        .sql("SELECT * FROM VALUES ('a', 1, 10L, 'x'), ('a', 2, 20L, 'y'), ('b', 3, 30L, 'x') AS t(k, i, l, s)")
        .unwrap();

    let grouped = df.group_by(vec![col("k")]);
    let sums = named_rows(&grouped.sum(vec!["i", "l"]).unwrap());
    assert_eq!(
        sums.1,
        vec![
            vec!["a".to_string(), "3".to_string(), "30".to_string()],
            vec!["b".to_string(), "3".to_string(), "30".to_string()],
        ]
    );
    assert_eq!(named_rows(&grouped.sum(["i", "l"]).unwrap()), sums);
    assert_eq!(
        named_rows(&grouped.sum(["i", "l"].iter().copied()).unwrap()),
        sums
    );
    assert_eq!(
        named_rows(&grouped.sum(["i", "l"].map(String::from)).unwrap()),
        sums
    );

    let dropped = named_rows(&df.drop(vec!["s"]));
    assert_eq!(dropped.0, vec!["k", "i", "l"]);
    assert_eq!(named_rows(&df.drop(["s"])), dropped);
    assert_eq!(named_rows(&df.drop(["s"].map(String::from))), dropped);
    let of_columns = named_rows(&df.drop_columns(vec![col("s")], vec!["i"]));
    assert_eq!(of_columns.0, vec!["k", "l"]);
    assert_eq!(named_rows(&df.drop_columns([col("s")], ["i"])), of_columns);
    assert_eq!(
        named_rows(&df.drop_columns(vec![col("s")].into_iter(), ["i"].into_iter())),
        of_columns
    );

    let with_columns = named_rows(&df.with_columns(vec![("p".to_string(), lit(1))]));
    assert_eq!(with_columns.0, vec!["k", "i", "l", "s", "p"]);
    assert_eq!(named_rows(&df.with_columns([("p", lit(1))])), with_columns);

    let renamed = named_rows(&df.to_df(vec!["a", "b", "c", "d"]));
    assert_eq!(renamed.0, vec!["a", "b", "c", "d"]);
    assert_eq!(named_rows(&df.to_df(["a", "b", "c", "d"])), renamed);
    assert_eq!(
        named_rows(&df.to_df(["a", "b", "c", "d"].iter().copied())),
        renamed
    );
}

/// A pivot on a date, timestamp or decimal column takes typed values. The counts match those
/// of PySpark 4.3.0 pivoting the same column on the same values as strings, which the server
/// casts to the column's type.
#[test]
fn pivot_on_typed_values_matches_pyspark() {
    use spark_connect::row::Value;
    if !should_run() {
        return;
    }
    let s = session();
    let names = |names: &[&str]| names.iter().map(|n| n.to_string()).collect::<Vec<String>>();
    let rows = |rows: &[&[&str]]| {
        rows.iter()
            .map(|r| r.iter().map(|v| v.to_string()).collect::<Vec<String>>())
            .collect::<Vec<Vec<String>>>()
    };
    let counted = |df: &DataFrame, column: &str, value: Value| {
        named_rows(
            &df.group_by(vec![col("k")])
                .pivot(col(column), Some(vec![value]))
                .unwrap()
                .count(),
        )
    };

    let dates = s
        .sql(
            "SELECT * FROM VALUES (1, DATE'2024-01-01'), (1, DATE'2024-01-02'), \
             (2, DATE'2024-01-01') AS t(k, d)",
        )
        .unwrap();
    let (cols, got) = counted(&dates, "d", Value::Date(19723));
    assert_eq!(cols, names(&["k", "2024-01-01"]));
    assert_eq!(got, rows(&[&["1", "1"], &["2", "1"]]));

    // Epoch seconds, so that the instants do not depend on the session time zone.
    let stamps = s
        .sql(
            "SELECT * FROM VALUES (1, CAST(1704067200 AS TIMESTAMP)), \
             (1, CAST(1704153600 AS TIMESTAMP)), (2, CAST(1704067200 AS TIMESTAMP)) AS t(k, ts)",
        )
        .unwrap();
    let (_, got) = counted(&stamps, "ts", Value::Timestamp(1_704_067_200_000_000));
    assert_eq!(got, rows(&[&["1", "1"], &["2", "1"]]));

    let decimals = s
        .sql(
            "SELECT * FROM VALUES (1, CAST(1.50 AS DECIMAL(5,2))), \
             (1, CAST(2.25 AS DECIMAL(5,2))), (2, CAST(1.50 AS DECIMAL(5,2))) AS t(k, dec)",
        )
        .unwrap();
    let (cols, got) = counted(
        &decimals,
        "dec",
        Value::Decimal {
            value: "1.50".to_string(),
            precision: Some(5),
            scale: Some(2),
        },
    );
    assert_eq!(cols, names(&["k", "1.50"]));
    assert_eq!(got, rows(&[&["1", "1"], &["2", "1"]]));
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

/// `observe_with` takes Columns, as `select` does, and the `Vec<Expression>` that it took
/// before: an action sets the same metrics of both in their observations.
#[test]
fn observe_with_takes_columns_and_expressions_alike() {
    if !should_run() {
        return;
    }
    use spark_connect::observation::Observation;
    use spark_connect_proto::expression::literal::LiteralType;

    let df = df4(&session());
    let of_columns = Observation::new("of_columns");
    let of_expressions = Observation::new("of_expressions");
    df.observe_with(
        &of_columns,
        [
            f::count(lit(1)).alias("rows"),
            f::max(col("id")).alias("top"),
        ],
    )
    .unwrap()
    .collect()
    .unwrap();
    df.observe_with(
        &of_expressions,
        vec![
            f::count(lit(1)).alias("rows").expression().clone(),
            f::max(col("id")).alias("top").expression().clone(),
        ],
    )
    .unwrap()
    .collect()
    .unwrap();
    let metrics = |observation: &Observation| -> Vec<(String, Option<LiteralType>)> {
        observation
            .get()
            .unwrap()
            .into_iter()
            .map(|(key, value)| (key, value.literal_type))
            .collect()
    };
    let expected = vec![
        ("rows".to_string(), Some(LiteralType::Long(4))),
        ("top".to_string(), Some(LiteralType::Integer(3))),
    ];
    assert_eq!(metrics(&of_columns), expected);
    assert_eq!(metrics(&of_expressions), expected);
}

/// `GroupedData.pivot` takes explicit values of the types of the Spark literals, dates,
/// timestamps and decimals too: they give the columns of the pivot without values, which PySpark
/// computes for the same data. PySpark itself rejects those types as values (NOT_EXPECTED_TYPE),
/// and a NULL value, as it does; `dropna` rejects a `how` other than "any" or "all"
/// (VALUE_NOT_ALLOWED), as PySpark does.
#[test]
fn pivot_values_of_dates_timestamps_and_decimals_match_discovered_pivots() {
    if !should_run() {
        return;
    }
    use spark_connect::row::Value;
    let s = session();
    let df = s
        .sql(
            "SELECT * FROM VALUES (DATE'2020-01-01', TIMESTAMP'2020-01-01 00:00:00', \
             CAST(1.50 AS DECIMAL(5,2)), 'a', 1), \
             (DATE'2020-01-02', TIMESTAMP'2020-01-02 00:00:00', CAST(2.00 AS DECIMAL(5,2)), 'b', 2) \
             AS t(d, ts, dec, k, v)",
        )
        .unwrap();
    let pivot = |column: &str, values: Option<Vec<Value>>| {
        named_rows(
            &df.group_by(vec![col("k")])
                .pivot(col(column), values)
                .unwrap()
                .sum(vec!["v"])
                .unwrap(),
        )
    };
    let decimal = |value: &str| Value::Decimal {
        value: value.to_string(),
        precision: Some(5),
        scale: Some(2),
    };
    // Days since the epoch of 2020-01-01 and 2020-01-02, and their micros in UTC.
    assert_eq!(
        pivot("d", Some(vec![Value::Date(18262), Value::Date(18263)])),
        pivot("d", None)
    );
    assert_eq!(
        pivot(
            "ts",
            Some(vec![
                Value::Timestamp(1_577_836_800_000_000),
                Value::Timestamp(1_577_923_200_000_000),
            ])
        ),
        pivot("ts", None)
    );
    assert_eq!(
        pivot("dec", Some(vec![decimal("1.50"), decimal("2.00")])),
        pivot("dec", None)
    );
    let err = df
        .group_by(vec![col("k")])
        .pivot(col("k"), Some(vec![Value::Null]))
        .err()
        .expect("an error");
    assert_eq!(err.get_condition().as_deref(), Some("NOT_EXPECTED_TYPE"));
    assert_eq!(
        err.get_message_parameters().unwrap()["arg_type"],
        "NoneType"
    );

    // As PySpark's `dropna(how="bogus")`.
    for how in ["bogus", "ANY"] {
        let err = df.dropna(Some(how), None, None).err().expect("an error");
        assert_eq!(err.get_condition().as_deref(), Some("VALUE_NOT_ALLOWED"));
        assert_eq!(err.get_message_parameters().unwrap()["arg_name"], "how");
        assert!(df.na().drop(Some(how), None, None).is_err());
    }
    assert_eq!(
        df.dropna(Some("all"), None, None).unwrap().count().unwrap(),
        2
    );
}

/// A pivot on the values collected from a float, tinyint, smallint or year-month interval column
/// names its columns as the pivot without values, which the server computes, does: "1.1" for a
/// float, as the Scala client's `lit` of it, and the SQL of an interval.
#[test]
fn pivot_values_of_floats_and_year_month_intervals_match_discovered_pivots() {
    if !should_run() {
        return;
    }
    use spark_connect::row::Value;
    let s = session();
    let df = s
        .sql(
            "SELECT * FROM VALUES (CAST(1.1 AS FLOAT), CAST(3 AS TINYINT), CAST(4 AS SMALLINT), \
             INTERVAL '1-0' YEAR TO MONTH, 'a', 1), \
             (CAST(-2.5 AS FLOAT), CAST(-3 AS TINYINT), CAST(-4 AS SMALLINT), \
             INTERVAL '-0-1' YEAR TO MONTH, 'b', 2) AS t(f, b, sh, ym, k, v)",
        )
        .unwrap();
    for column in ["f", "b", "sh", "ym"] {
        // In the order of the pivot without values, which sorts them.
        let values: Vec<Value> = df
            .select(vec![col(column)])
            .order_by(vec![col(column).expression().clone()])
            .collect()
            .unwrap()
            .iter()
            .map(|row| row.get(0).unwrap().clone())
            .collect();
        let pivot = |values: Option<Vec<Value>>| {
            named_rows(
                &df.group_by(vec![col("k")])
                    .pivot(col(column), values)
                    .unwrap()
                    .sum(vec!["v"])
                    .unwrap()
                    .order_by(vec![col("k").expression().clone()]),
            )
        };
        assert_eq!(pivot(Some(values)), pivot(None), "{column}");
    }
    let intervals: Vec<Value> = df
        .select(vec![col("ym")])
        .order_by(vec![col("v").expression().clone()])
        .collect()
        .unwrap()
        .iter()
        .map(|row| row.get(0).unwrap().clone())
        .collect();
    assert_eq!(
        intervals,
        vec![Value::YearMonthInterval(12), Value::YearMonthInterval(-1)]
    );
    let (columns, _) = named_rows(
        &df.group_by(vec![col("k")])
            .pivot(col("ym"), Some(intervals))
            .unwrap()
            .sum(vec!["v"])
            .unwrap(),
    );
    assert_eq!(
        columns,
        vec![
            "k".to_string(),
            "INTERVAL '1-0' YEAR TO MONTH".to_string(),
            "INTERVAL '-0-1' YEAR TO MONTH".to_string(),
        ]
    );
}

/// Year-month intervals are collected as their months, in nested values too, and local rows of
/// them are sent as Spark's Arrow conversion sends them.
#[test]
fn year_month_interval_values_roundtrip() {
    if !should_run() {
        return;
    }
    use spark_connect::row::{Row, Value};
    use spark_connect::types::{DataType, StructField};
    let s = session();
    let rows = s
        .sql(
            "SELECT INTERVAL '178956970-7' YEAR TO MONTH AS a, INTERVAL '-178956970-8' YEAR TO \
             MONTH AS b, array(INTERVAL '2' YEAR) AS c, CAST(NULL AS INTERVAL MONTH) AS d",
        )
        .unwrap()
        .collect()
        .unwrap();
    assert_eq!(rows[0].get(0), Some(&Value::YearMonthInterval(i32::MAX)));
    assert_eq!(rows[0].get(1), Some(&Value::YearMonthInterval(i32::MIN)));
    assert_eq!(
        rows[0].get(2),
        Some(&Value::List(vec![Value::YearMonthInterval(24)]))
    );
    assert_eq!(rows[0].get(3), Some(&Value::Null));

    let interval = DataType::YearMonthInterval {
        start_field: 0,
        end_field: 1,
    };
    let schema = DataType::Struct {
        fields: vec![StructField {
            name: "i".to_string(),
            data_type: interval,
            nullable: true,
            metadata: Default::default(),
        }],
    };
    let input = vec![
        Row::new(vec!["i".to_string()], vec![Value::YearMonthInterval(-13)]),
        Row::new(vec!["i".to_string()], vec![Value::Null]),
    ];
    let df = s.create_dataframe(input, schema).unwrap();
    let texts: Vec<Value> = df
        .select_expr(vec!["CAST(i AS STRING)"])
        .collect()
        .unwrap()
        .iter()
        .map(|row| row.get(0).unwrap().clone())
        .collect();
    assert_eq!(
        texts,
        vec![
            Value::String("INTERVAL '-1-1' YEAR TO MONTH".to_string()),
            Value::Null
        ]
    );
    let back: Vec<Value> = df
        .collect()
        .unwrap()
        .iter()
        .map(|row| row.get(0).unwrap().clone())
        .collect();
    assert_eq!(back, vec![Value::YearMonthInterval(-13), Value::Null]);
}

/// Collect a slow query and iterate it on demand, on this thread, which `caller` notes: the rows
/// of each.
fn run_slow_queries(
    s: &SparkSession,
    caller: &std::sync::Mutex<Option<std::thread::ThreadId>>,
) -> (usize, usize) {
    *caller.lock().unwrap() = Some(std::thread::current().id());
    let slow = "SELECT id, reflect('java.lang.Thread', 'sleep', 300L) AS s FROM range(0, 4, 1, 4)";
    let collected = s.sql(slow).unwrap().collect().unwrap().len();
    let iterated = s
        .sql(slow)
        .unwrap()
        .to_local_iterator(false)
        .unwrap()
        .count();
    (collected, iterated)
}

/// As PySpark calls the progress handlers on the thread that runs the execution: also inside a
/// `current_thread` runtime, where the responses are awaited on another thread, and on a worker
/// of a multi-thread runtime, each update and the done call run on the caller's thread.
#[test]
fn progress_handlers_run_on_the_calling_thread_inside_a_tokio_runtime() {
    if !should_run() {
        return;
    }
    use std::sync::{Arc, Mutex};
    let s = session().new_session();
    s.conf()
        .set("spark.connect.progress.reportInterval", "100ms")
        .unwrap();
    let caller: Arc<Mutex<Option<std::thread::ThreadId>>> = Arc::default();
    let calls: Arc<Mutex<Vec<(bool, bool)>>> = Arc::default();
    let (expected, seen) = (Arc::clone(&caller), Arc::clone(&calls));
    let id = s.register_progress_handler(move |update| {
        let on_caller = *expected.lock().unwrap() == Some(std::thread::current().id());
        seen.lock().unwrap().push((update.done, on_caller));
    });
    let current = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();
    assert_eq!(
        current.block_on(async { run_slow_queries(&s, &caller) }),
        (4, 4)
    );
    let multi = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();
    let (task_session, task_caller) = (s.clone(), Arc::clone(&caller));
    let counts = multi.block_on(async move {
        tokio::spawn(async move { run_slow_queries(&task_session, &task_caller) })
            .await
            .unwrap()
    });
    assert_eq!(counts, (4, 4));
    s.remove_progress_handler(id);
    let calls = calls.lock().unwrap();
    assert!(calls.iter().any(|(done, _)| !done), "{calls:?}");
    assert!(
        calls.iter().filter(|(done, _)| *done).count() >= 4,
        "{calls:?}"
    );
    assert!(calls.iter().all(|(_, on_caller)| *on_caller), "{calls:?}");
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
    let relation = checkpoint.to_relation().unwrap();
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
    let relation = checkpoint.to_relation().unwrap();
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
    let df = s.range(1).expect("range").select(vec![f::repeat(
        spark_connect::column::lit_string("x"),
        lit(len),
    )]);
    let rows = df.collect().unwrap();
    assert_eq!(rows.len(), 1);
    let value = rows[0].get(0).unwrap().as_str().unwrap();
    assert_eq!(value.len(), len as usize);
    assert!(value.bytes().all(|b| b == b'x'));
}

/// `repartition_by_range` splits the ids into contiguous ranges, as PySpark's
/// `repartitionByRange` does; a hash split of the same column would not.
#[test]
fn repartition_by_range_splits_into_contiguous_ranges() {
    if !should_run() {
        return;
    }
    let s = session();
    let rows = s
        .range(100)
        .unwrap()
        .repartition_by_range(4, vec![col("id").expression().clone()])
        .unwrap()
        .with_column("p", f::spark_partition_id())
        .group_by(vec![col("p")])
        .agg(vec![
            f::min(col("id")).alias("lo").expression().clone(),
            f::max(col("id")).alias("hi").expression().clone(),
            f::count(col("id")).alias("n").expression().clone(),
        ])
        .order_by(vec![col("p").expression().clone()])
        .collect()
        .unwrap();
    let ranges: Vec<(i64, i64, i64)> = rows
        .iter()
        .map(|row| {
            let get = |name: &str| row.get_by_name(name).and_then(|v| v.as_i64()).unwrap();
            (get("lo"), get("hi"), get("n"))
        })
        .collect();
    assert_eq!(
        ranges,
        vec![(0, 24, 25), (25, 49, 25), (50, 74, 25), (75, 99, 25)]
    );
}

#[test]
fn is_empty_is_true_only_without_rows() {
    if !should_run() {
        return;
    }
    let s = session();
    assert!(!s.range(3).unwrap().is_empty().unwrap());
    assert!(s
        .range(3)
        .unwrap()
        .filter(col("id").gt(lit(5)))
        .is_empty()
        .unwrap());
}
