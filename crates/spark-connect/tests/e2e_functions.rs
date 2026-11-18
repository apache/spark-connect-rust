//! Behavioral e2e coverage of representative SQL functions across categories: each is
//! applied against a live server and its result asserted to match reference pyspark.
//! (Full byte-for-byte builder parity lives in functions_golden.rs; this pins runtime
//! behavior and covers the exec path.) Run with SPARK_REMOTE set.

mod common;

use common::{session, should_run};
use spark_connect::column::{col, lit, lit_double, lit_string, Column};
use spark_connect::dataframe::DataFrame;
use spark_connect::functions as f;
use spark_connect::row::Value;
use spark_connect::session::SparkSession;

fn base(s: &SparkSession) -> DataFrame {
    s.sql(
        "SELECT -5 AS neg, 4.0 AS four, 'abc' AS s, array(3,1,2) AS arr, \
         CAST(NULL AS STRING) AS n FROM range(1)",
    )
    .expect("base")
}
fn val(df: &DataFrame, c: Column) -> Value {
    df.select(vec![c]).collect().unwrap()[0]
        .get(0)
        .cloned()
        .unwrap_or(Value::Null)
}

#[test]
fn math_and_string_functions() {
    if !should_run() {
        return;
    }
    let s = session();
    let df = base(&s);
    assert_eq!(val(&df, f::abs(col("neg"))).as_i64(), Some(5));
    assert_eq!(val(&df, f::sqrt(col("four"))).as_f64(), Some(2.0));
    assert_eq!(val(&df, f::ceil(lit_double(1.1))).as_i64(), Some(2));
    assert_eq!(val(&df, f::floor(lit_double(1.9))).as_i64(), Some(1));
    assert_eq!(val(&df, f::round(lit_double(3.7))).as_f64(), Some(4.0));
    assert_eq!(val(&df, f::upper(col("s"))).as_str(), Some("ABC"));
    assert_eq!(val(&df, f::lower(lit_string("ABC"))).as_str(), Some("abc"));
    assert_eq!(val(&df, f::length(col("s"))).as_i64(), Some(3));
    assert_eq!(val(&df, f::reverse(col("s"))).as_str(), Some("cba"));
    // md5("abc") is a stable, known digest.
    assert_eq!(
        val(&df, f::md5(col("s"))).as_str(),
        Some("900150983cd24fb0d6963f7d28e17f72")
    );
}

#[test]
fn conditional_and_collection_functions() {
    if !should_run() {
        return;
    }
    let s = session();
    let df = base(&s);
    // coalesce skips the null and returns the fallback
    assert_eq!(
        val(&df, f::coalesce(vec![col("n"), lit_string("x")])).as_str(),
        Some("x")
    );
    assert_eq!(val(&df, f::greatest(lit(1), lit(5))).as_i64(), Some(5));
    assert_eq!(val(&df, f::least(lit(1), lit(5))).as_i64(), Some(1));
    // array [3,1,2]
    assert_eq!(val(&df, f::size(col("arr"))).as_i64(), Some(3));
    assert_eq!(
        val(&df, f::array_contains(col("arr"), lit(2))).as_bool(),
        Some(true)
    );
    // element_at is 1-indexed -> first element is 3
    assert_eq!(
        val(&df, f::element_at(col("arr"), lit(1))).as_i64(),
        Some(3)
    );
    // sort_array -> [1,2,3]; check the first element after sort
    let sorted = val(&df, f::element_at(f::sort_array(col("arr")), lit(1)));
    assert_eq!(sorted.as_i64(), Some(1));
}

#[test]
fn aggregate_functions() {
    if !should_run() {
        return;
    }
    let s = session();
    // CAST to DOUBLE: SQL `10.0` literals are DECIMAL and sum/avg of decimals return
    // Decimal, so use an explicit double column for the float assertions.
    let df = s
        .sql("SELECT id, CAST(v AS DOUBLE) AS v FROM VALUES (1,10.0),(2,20.0),(2,30.0) AS t(id, v)")
        .expect("agg df");
    // one-group aggregates via a constant grouping
    let agg = df.group_by(vec![lit(1)]).agg(vec![
        f::sum(col("v")).alias("s").expression().clone(),
        f::avg(col("v")).alias("a").expression().clone(),
        f::max(col("v")).alias("mx").expression().clone(),
        f::min(col("v")).alias("mn").expression().clone(),
        f::count(col("id")).alias("c").expression().clone(),
    ]);
    let row = &agg.collect().unwrap()[0];
    // row: [group, s, a, mx, mn, c]
    assert_eq!(row.get_by_name("s").and_then(|v| v.as_f64()), Some(60.0));
    assert_eq!(row.get_by_name("a").and_then(|v| v.as_f64()), Some(20.0));
    assert_eq!(row.get_by_name("mx").and_then(|v| v.as_f64()), Some(30.0));
    assert_eq!(row.get_by_name("mn").and_then(|v| v.as_f64()), Some(10.0));
    assert_eq!(row.get_by_name("c").and_then(|v| v.as_i64()), Some(3));
}

/// As PySpark's `F.reduce`, which sends the function `reduce`: the column of the name and the
/// value of the official client's, with and without a finish lambda.
#[test]
fn reduce_is_named_and_valued_as_pyspark() {
    if !should_run() {
        return;
    }
    let s = session();
    let df = s.sql("SELECT array(1, 2, 3) AS a").unwrap();
    let merge = "lambdafunction((namedlambdavariable() + namedlambdavariable()), \
                 namedlambdavariable(), namedlambdavariable())";
    let reduced = df.select(vec![f::reduce(col("a"), lit(0), |args| {
        args[0].clone() + args[1].clone()
    })]);
    assert_eq!(
        reduced.columns().unwrap(),
        [format!(
            "reduce(a, 0, {merge}, lambdafunction(namedlambdavariable(), namedlambdavariable()))"
        )]
    );
    assert_eq!(val(&reduced, col("*")).as_i64(), Some(6));
    let finished = df.select(vec![f::reduce_with_finish(
        col("a"),
        lit(0),
        |args| args[0].clone() + args[1].clone(),
        |args| args[0].clone() * lit(2),
    )]);
    assert_eq!(
        finished.columns().unwrap(),
        [format!(
            "reduce(a, 0, {merge}, lambdafunction((namedlambdavariable() * 2), \
             namedlambdavariable()))"
        )]
    );
    assert_eq!(val(&finished, col("*")).as_i64(), Some(12));
}

/// As PySpark, whose random functions draw a seed for each call: two `uuid()` and two `rand()`
/// columns differ, as do the `uuid()` of two queries, while a DataFrame keeps its seeds across
/// executions and a given seed gives the same values in every query.
#[test]
fn random_functions_draw_a_seed_for_each_call_as_pyspark() {
    if !should_run() {
        return;
    }
    let s = session();
    let df = s.range(3).unwrap();
    let values = |frame: &DataFrame| -> Vec<String> {
        frame
            .collect()
            .unwrap()
            .iter()
            .map(|row| format!("{:?}", (row.get(0), row.get(1))))
            .collect()
    };
    let uuids = df.select(vec![f::uuid(), f::uuid()]);
    for row in uuids.collect().unwrap() {
        assert_ne!(row.get(0), row.get(1));
    }
    assert_eq!(values(&uuids), values(&uuids));
    let one = df.select(vec![f::uuid(), lit(0)]);
    let other = df.select(vec![f::uuid(), lit(0)]);
    assert_ne!(values(&one), values(&other));
    for row in df.select(vec![f::rand(), f::rand()]).collect().unwrap() {
        assert_ne!(row.get(0), row.get(1));
    }
    let seeded = |seed: i64| df.select(vec![f::rand_with_seed(seed), f::randn_with_seed(seed)]);
    assert_eq!(values(&seeded(7)), values(&seeded(7)));
    assert_ne!(values(&seeded(7)), values(&seeded(8)));
    let shuffled = |seed: i64| {
        s.sql("SELECT sequence(1, 50) AS a")
            .unwrap()
            .select(vec![f::shuffle_with_seed(col("a"), lit(seed)), lit(0)])
    };
    assert_eq!(values(&shuffled(3)), values(&shuffled(3)));
}

#[test]
fn optional_arg_function_variants() {
    if !should_run() {
        return;
    }
    use spark_connect::column::{lit, lit_double, lit_string};
    let s = session();
    let df = s.range(1).unwrap();
    let v = |c: Column| {
        df.select(vec![c]).collect().unwrap()[0]
            .get(0)
            .cloned()
            .unwrap()
    };

    // round/bround with an explicit scale.
    assert_eq!(
        v(f::round_scale(lit_double(1.23456), lit(2))).as_f64(),
        Some(1.23)
    );
    assert_eq!(
        v(f::bround_scale(lit_double(2.5), lit(0))).as_f64(),
        Some(2.0)
    );
    // ceil/floor with scale return DECIMAL; just assert they evaluate non-null.
    assert!(!v(f::ceil_scale(lit_double(2.1), lit(1))).is_null());
    assert!(!v(f::floor_scale(lit_double(2.9), lit(1))).is_null());
    // trim/ltrim/rtrim with a trim string (arg order: trim string first).
    assert_eq!(
        v(f::trim_with(lit_string("xxhixx"), lit_string("x"))).as_str(),
        Some("hi")
    );
    assert_eq!(
        v(f::ltrim_with(lit_string("xxhi"), lit_string("x"))).as_str(),
        Some("hi")
    );
    assert_eq!(
        v(f::rtrim_with(lit_string("hixx"), lit_string("x"))).as_str(),
        Some("hi")
    );
    // to_binary with an explicit format.
    assert!(!v(f::to_binary_format(lit_string("41"), lit_string("hex"))).is_null());
    // approx_count_distinct with a relative-standard-deviation literal.
    assert_eq!(
        v(f::approx_count_distinct_rsd(col("id"), lit_double(0.05))).as_i64(),
        Some(1)
    );
}

/// The forms of the optional arguments of PySpark's functions give PySpark's results: those of
/// the same calls of the reference client, against the same server.
#[test]
fn optional_argument_forms_give_pyspark_results() {
    if !should_run() {
        return;
    }
    use spark_connect::column::lit_boolean;
    let s = session();
    let df = s.range(1).unwrap();
    let v = |c: Column| {
        df.select(vec![c]).collect().unwrap()[0]
            .get(0)
            .cloned()
            .unwrap()
    };
    let arr = || f::array([lit(3), lit(1), lit(2)]);

    // A format, a limit and an order that the base functions send as their defaults or not.
    assert_eq!(
        v(
            f::to_date_with_format(lit_string("2024/01/02"), lit_string("yyyy/MM/dd"))
                .cast_str("string")
        )
        .as_str(),
        Some("2024-01-02")
    );
    assert_eq!(
        v(f::element_at(
            f::split_with_limit(lit_string("a-b-c"), lit_string("-"), lit(2)),
            lit(2)
        ))
        .as_str(),
        Some("b-c")
    );
    assert_eq!(
        v(f::element_at(
            f::sort_array_with_asc(arr(), lit_boolean(false)),
            lit(1)
        ))
        .as_i64(),
        Some(3)
    );
    assert_eq!(
        v(f::element_at(
            f::array_sort_with_comparator(arr(), |args| args[1].clone() - args[0].clone()),
            lit(1)
        ))
        .as_i64(),
        Some(3)
    );
    // log of a base, which PySpark sends as `log` where the one-argument form sends `ln`.
    assert_eq!(
        v(f::log_with_base(lit_double(2.0), lit_double(8.0))).as_f64(),
        Some(3.0)
    );
    // The options reach the parser: single quotes are no JSON without allowSingleQuotes.
    assert!(v(f::from_json_with_options(
        lit_string("{'a': 1}"),
        lit_string("a INT"),
        [("allowSingleQuotes", "false")]
    )
    .get_field("a"))
    .is_null());
    // The trailing defaults that PySpark sends: `x`, `n` and a null after the upper character.
    assert_eq!(
        v(f::mask_with_upper_char(lit_string("AbC1"), lit_string("U"))).as_str(),
        Some("UxUn")
    );
    assert_eq!(
        v(f::element_at(
            f::str_to_map_with_pair_delim_and_key_value_delim(
                lit_string("a=1&b=2"),
                lit_string("&"),
                lit_string("=")
            ),
            lit_string("b")
        ))
        .as_str(),
        Some("2")
    );
    assert_eq!(
        v(f::replace_with(
            lit_string("abc"),
            lit_string("b"),
            lit_string("X")
        ))
        .as_str(),
        Some("aXc")
    );
    assert_eq!(
        v(f::locate_with_pos(
            lit_string("b"),
            lit_string("abcb"),
            lit(3)
        ))
        .as_i64(),
        Some(4)
    );
    assert_eq!(
        v(f::overlay_with_len(
            lit_string("SPARK_SQL"),
            lit_string("CORE"),
            lit(7),
            lit(0)
        ))
        .as_str(),
        Some("SPARK_CORESQL")
    );
    // A part of an interval that is given, with `lit(0)` of the others.
    assert_eq!(
        v(f::make_ym_interval_with_parts(None, Some(lit(14))).cast_str("string")).as_str(),
        Some("INTERVAL '1-2' YEAR TO MONTH")
    );
    // first skips the null when it is told to ignore nulls.
    let values = s
        .sql("SELECT * FROM VALUES (CAST(NULL AS INT)), (1) AS t(x)")
        .unwrap();
    let first = values
        .agg([f::first_with_ignore_nulls(col("x"), lit_boolean(true))])
        .collect()
        .unwrap();
    assert_eq!(first[0].get(0).and_then(Value::as_i64), Some(1));
}
