// SPDX-License-Identifier: Apache-2.0
#![allow(dead_code)]
use serde_json::{json, Value as Json};
use spark_connect::column::col;
use spark_connect::expression::Expression;
use spark_connect::ml::*;
use spark_connect::row::{Row, Value};
use spark_connect::{Column, DataFrame, SparkSession};

pub fn fixture() -> Json {
    serde_json::from_str(include_str!("../../../../tests/golden/statistics.json")).unwrap()
}
pub fn dataset(spark: &SparkSession, name: &str) -> DataFrame {
    spark
        .sql(fixture()["datasets"][name].as_str().unwrap())
        .unwrap()
        .with_column(
            "features",
            Column::new(Expression::function(
                "array_to_vector",
                vec![col("features").expression().clone()],
            )),
        )
}
pub fn helper(case: &Json, df: &DataFrame) -> DataFrame {
    let args = case["params"].as_array().unwrap();
    let first = args[0].as_str().unwrap();
    let second = args[1].as_str().unwrap();
    match case["op"].as_str().unwrap() {
        "chi" => ChiSquareTest::test(df, first, second, args[2].as_bool().unwrap()),
        "corr" => Correlation::corr(df, first, second),
        "ks" => KolmogorovSmirnovTest::test(
            df,
            first,
            second,
            args[2..].iter().map(|v| {
                v.as_f64()
                    .unwrap_or_else(|| v.as_str().unwrap().parse().unwrap())
            }),
        ),
        other => panic!("unknown helper: {other}"),
    }
    .unwrap()
}
pub fn summary(case: &Json) -> Column {
    let weight = case["weighted"].as_bool().unwrap().then(|| col("weight"));
    let features = col("features");
    match case["metric"].as_str() {
        None => Summarizer::metrics(
            fixture()["metrics"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_str().unwrap()),
        )
        .summary(features, weight),
        Some("mean") => Summarizer::mean(features, weight),
        Some("sum") => Summarizer::sum(features, weight),
        Some("variance") => Summarizer::variance(features, weight),
        Some("std") => Summarizer::std(features, weight),
        Some("count") => Summarizer::count(features, weight),
        Some("numNonZeros") => Summarizer::num_non_zeros(features, weight),
        Some("max") => Summarizer::max(features, weight),
        Some("min") => Summarizer::min(features, weight),
        Some("normL1") => Summarizer::norm_l1(features, weight),
        Some("normL2") => Summarizer::norm_l2(features, weight),
        other => panic!("unknown metric: {other:?}"),
    }
}
fn normalize(value: &Value) -> Json {
    match value {
        Value::Null => Json::Null,
        Value::Bool(v) => json!(v),
        Value::Byte(v) => json!(v),
        Value::Short(v) => json!(v),
        Value::Integer(v) => json!(v),
        Value::Long(v) => json!(v),
        Value::Double(v) if v.is_nan() => json!({"float": "nan"}),
        Value::Double(v) => json!(v),
        Value::String(v) => json!(v),
        Value::List(v) => v.iter().map(normalize).collect(),
        Value::Struct(v) => v
            .fields()
            .iter()
            .zip(v.values())
            .map(|(k, v)| (k.clone(), normalize(v)))
            .collect(),
        other => panic!("unexpected result: {other:?}"),
    }
}
fn row(row: &Row) -> Json {
    row.fields()
        .iter()
        .zip(row.values())
        .map(|(k, v)| (k.clone(), normalize(v)))
        .collect()
}
fn close(actual: &Json, expected: &Json) {
    match (actual, expected) {
        (Json::Object(a), Json::Object(e)) => {
            assert_eq!(a.keys().collect::<Vec<_>>(), e.keys().collect::<Vec<_>>());
            for (k, v) in e {
                close(&a[k], v);
            }
        }
        (Json::Array(a), Json::Array(e)) => {
            assert_eq!(a.len(), e.len());
            for (a, e) in a.iter().zip(e) {
                close(a, e);
            }
        }
        (Json::Number(a), Json::Number(e)) => {
            let (a, e) = (a.as_f64().unwrap(), e.as_f64().unwrap());
            assert!(
                (a - e).abs() <= 1e-10 + 1e-10 * e.abs(),
                "expected {e}, got {a}"
            );
        }
        _ => assert_eq!(actual, expected),
    }
}
pub fn assert_result(df: &DataFrame, expected: &Json) {
    assert_eq!(json!(df.columns().unwrap()), expected["columns"]);
    let mut actual: Vec<_> = df.collect().unwrap().iter().map(row).collect();
    let mut expected = expected["rows"].as_array().unwrap().clone();
    actual.sort_by_key(Json::to_string);
    expected.sort_by_key(Json::to_string);
    close(&json!(actual), &json!(expected));
}
