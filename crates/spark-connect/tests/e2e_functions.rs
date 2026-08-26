//! Behavioral e2e coverage of representative SQL functions across categories: each is
//! applied against a live server and its result asserted to match reference pyspark.
//! (Full byte-for-byte builder parity lives in functions_golden.rs; this pins runtime
//! behavior and covers the exec path.) Run with SPARK_REMOTE set.

use spark_connect::column::{col, lit, lit_double, lit_string, Column};
use spark_connect::dataframe::DataFrame;
use spark_connect::functions as f;
use spark_connect::row::Value;
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
        v(f::round_scale(lit_double(3.14159), lit(2))).as_f64(),
        Some(3.14)
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
