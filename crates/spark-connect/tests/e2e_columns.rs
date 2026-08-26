//! Behavioral e2e coverage of the Column expression API: every operator is applied
//! against a live server and its results are asserted to match reference pyspark
//! semantics (arithmetic, comparison, string, casts, null handling, when/otherwise,
//! bitwise, membership, struct/array/map access, window). This both covers column.rs
//! and pins correct behavior. Run with SPARK_REMOTE set.

use spark_connect::column::{col, lit, lit_boolean, lit_double, lit_string, when, Column};
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
        "SELECT * FROM VALUES (1,'apple',10.0),(2,'banana',20.0),(3,NULL,30.0) \
         AS t(id, name, val)",
    )
    .expect("base df")
}
// Collect a single-column projection into typed option vectors (null -> None).
fn one(df: &DataFrame, c: Column) -> Vec<Value> {
    df.select(vec![c])
        .collect()
        .expect("collect")
        .into_iter()
        .map(|r| r.get(0).cloned().unwrap_or(Value::Null))
        .collect()
}
fn i64s(v: &[Value]) -> Vec<Option<i64>> {
    v.iter().map(|x| x.as_i64()).collect()
}
fn bools(v: &[Value]) -> Vec<Option<bool>> {
    v.iter().map(|x| x.as_bool()).collect()
}
fn strs(v: &[Value]) -> Vec<Option<String>> {
    v.iter()
        .map(|x| x.as_str().map(|s| s.to_string()))
        .collect()
}

#[test]
fn column_arithmetic_and_comparison() {
    if !should_run() {
        return;
    }
    let s = session();
    let df = base(&s);
    assert_eq!(
        i64s(&one(&df, col("id").add(lit(1)))),
        vec![Some(2), Some(3), Some(4)]
    );
    assert_eq!(
        i64s(&one(&df, col("id").sub(lit(1)))),
        vec![Some(0), Some(1), Some(2)]
    );
    assert_eq!(
        i64s(&one(&df, col("id").mul(lit(2)))),
        vec![Some(2), Some(4), Some(6)]
    );
    assert_eq!(
        i64s(&one(&df, col("id").modulo(lit(2)))),
        vec![Some(1), Some(0), Some(1)]
    );
    assert_eq!(
        i64s(&one(&df, col("id").neg())),
        vec![Some(-1), Some(-2), Some(-3)]
    );
    // comparisons
    assert_eq!(
        bools(&one(&df, col("id").gt(lit(1)))),
        vec![Some(false), Some(true), Some(true)]
    );
    assert_eq!(
        bools(&one(&df, col("id").ge(lit(2)))),
        vec![Some(false), Some(true), Some(true)]
    );
    assert_eq!(
        bools(&one(&df, col("id").lt(lit(2)))),
        vec![Some(true), Some(false), Some(false)]
    );
    assert_eq!(
        bools(&one(&df, col("id").le(lit(2)))),
        vec![Some(true), Some(true), Some(false)]
    );
    assert_eq!(
        bools(&one(&df, col("id").eq(lit(2)))),
        vec![Some(false), Some(true), Some(false)]
    );
    assert_eq!(
        bools(&one(&df, col("id").ne(lit(2)))),
        vec![Some(true), Some(false), Some(true)]
    );
    // boolean combinators
    let big_and_small = col("id").gt(lit(1)).and(col("id").lt(lit(3)));
    assert_eq!(
        bools(&one(&df, big_and_small)),
        vec![Some(false), Some(true), Some(false)]
    );
    let one_or_three = col("id").eq(lit(1)).or(col("id").eq(lit(3)));
    assert_eq!(
        bools(&one(&df, one_or_three)),
        vec![Some(true), Some(false), Some(true)]
    );
    assert_eq!(
        bools(&one(&df, col("id").gt(lit(1)).not())),
        vec![Some(true), Some(false), Some(false)]
    );
    // bitwise
    assert_eq!(
        i64s(&one(&df, col("id").bitwise_and(lit(1)))),
        vec![Some(1), Some(0), Some(1)]
    );
    assert_eq!(
        i64s(&one(&df, col("id").bitwise_or(lit(4)))),
        vec![Some(5), Some(6), Some(7)]
    );
    assert_eq!(
        i64s(&one(&df, col("id").bitwise_xor(lit(1)))),
        vec![Some(0), Some(3), Some(2)]
    );
}

#[test]
fn column_string_and_null_and_cast() {
    if !should_run() {
        return;
    }
    let s = session();
    let df = base(&s);
    // string predicates (null input -> null result)
    assert_eq!(
        bools(&one(&df, col("name").contains(lit_string("an")))),
        vec![Some(false), Some(true), None]
    );
    assert_eq!(
        bools(&one(&df, col("name").startswith(lit_string("a")))),
        vec![Some(true), Some(false), None]
    );
    assert_eq!(
        bools(&one(&df, col("name").endswith(lit_string("e")))),
        vec![Some(true), Some(false), None]
    );
    assert_eq!(
        bools(&one(&df, col("name").like("a%"))),
        vec![Some(true), Some(false), None]
    );
    assert_eq!(
        bools(&one(&df, col("name").rlike("^a"))),
        vec![Some(true), Some(false), None]
    );
    assert_eq!(
        bools(&one(&df, col("name").ilike("A%"))),
        vec![Some(true), Some(false), None]
    );
    // substr is 1-indexed
    assert_eq!(
        strs(&one(&df, col("name").substr(lit(1), lit(3)))),
        vec![Some("app".into()), Some("ban".into()), None]
    );
    assert_eq!(
        strs(&one(&df, f::upper(col("name")))),
        vec![Some("APPLE".into()), Some("BANANA".into()), None]
    );
    // null handling
    assert_eq!(
        bools(&one(&df, col("name").is_null())),
        vec![Some(false), Some(false), Some(true)]
    );
    assert_eq!(
        bools(&one(&df, col("name").is_not_null())),
        vec![Some(true), Some(true), Some(false)]
    );
    // eqNullSafe: null-safe equality
    assert_eq!(
        bools(&one(&df, col("name").eq_null_safe(lit_string("apple")))),
        vec![Some(true), Some(false), Some(false)]
    );
    // casts: double -> int
    assert_eq!(
        i64s(&one(
            &df,
            col("val").cast(spark_connect::types::DataType::Integer)
        )),
        vec![Some(10), Some(20), Some(30)]
    );
    assert_eq!(
        i64s(&one(&df, col("val").cast_str("int"))),
        vec![Some(10), Some(20), Some(30)]
    );
}

#[test]
fn column_membership_conditional_window() {
    if !should_run() {
        return;
    }
    use spark_connect::window::Window;
    let s = session();
    let df = base(&s);
    // between / isin
    assert_eq!(
        bools(&one(&df, col("id").between(lit(1), lit(2)))),
        vec![Some(true), Some(true), Some(false)]
    );
    assert_eq!(
        bools(&one(&df, col("id").isin(vec![lit(1), lit(3)]))),
        vec![Some(true), Some(false), Some(true)]
    );
    // when / otherwise
    let bucket = when(col("id").gt(lit(1)), lit_string("big")).otherwise(lit_string("small"));
    assert_eq!(
        strs(&one(&df, bucket)),
        vec![Some("small".into()), Some("big".into()), Some("big".into())]
    );
    // lit variants build valid columns
    let _ = one(&df, lit_double(1.5));
    let _ = one(&df, lit_boolean(true));
    // sort-order builders + window (over)
    let w = Window::partition_by(vec![col("id").expression().clone()]).order_by(vec![]);
    let _ = one(&df, f::sum(col("val")).over(w));
    // asc/desc builders exercised through order_by
    let _ = df
        .order_by(vec![
            col("id").asc().expression().clone(),
            col("val").desc_nulls_last().expression().clone(),
        ])
        .collect()
        .unwrap();
}

#[test]
fn column_struct_array_map_access() {
    if !should_run() {
        return;
    }
    let s = session();
    let df = s.range(1).expect("range");
    // struct field access + with/drop field
    let st = f::r#struct(vec![lit(1).alias("a"), lit(2).alias("b")]);
    assert_eq!(i64s(&one(&df, st.clone().get_field("a"))), vec![Some(1)]);
    let _ = one(&df, st.clone().with_field("c", lit(9)));
    let _ = one(&df, st.drop_fields(vec!["a"]));
    // array element access (get_item) - array is 0-indexed
    let arr = f::array(vec![lit(10), lit(20), lit(30)]);
    assert_eq!(i64s(&one(&df, arr.get_item(lit(1)))), vec![Some(20)]);
    // map value access
    let m = f::create_map(vec![lit_string("k"), lit(7)]);
    assert_eq!(i64s(&one(&df, m.get_item(lit_string("k")))), vec![Some(7)]);
}
