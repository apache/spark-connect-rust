// SPDX-License-Identifier: Apache-2.0
//! All standard feature stages against Spark. Set SPARK_REMOTE to run.
mod common;
#[path = "common/feature_cases.rs"]
#[macro_use]
mod feature_cases;

use common::live_session;
use serde_json::{json, Value as Json};
use spark_connect::column::col;
use spark_connect::expression::Expression;
use spark_connect::ml::*;
use spark_connect::row::Value;
use spark_connect::{Column, DataFrame};

/// The output of the case `id`: the transform of `input` by its stage, or by the model that its
/// estimator fits on `input`.
macro_rules! output {
    (
        transformers: [$($tid:literal => $transformer:expr),* $(,)?],
        estimators: [$($eid:literal => $estimator:expr),* $(,)?] $(,)?
    ) => {
        fn output(id: &str, input: &DataFrame) -> DataFrame {
            match id {
                $($tid => $transformer.transform(input).unwrap(),)*
                $($eid => $estimator.fit(input).unwrap().transform(input).unwrap(),)*
                _ => panic!("uncovered feature case {id}"),
            }
        }
    };
}

feature_cases!(output);

fn normalized(v: &Value) -> Json {
    match v {
        Value::Null => Json::Null,
        Value::String(v) => json!(v),
        Value::Bool(v) => json!(v),
        Value::List(v) => v.iter().map(normalized).collect(),
        Value::Struct(row) if row.get_by_name("type").is_some() => {
            // Spark VectorUDT: dense values, or sparse indices and values.
            let values = normalized(row.get_by_name("values").unwrap());
            let kind = row.get_by_name("type").unwrap();
            if matches!(kind, Value::Byte(1)) {
                return values;
            }
            let size = row.get_by_name("size").unwrap().as_i64().unwrap() as usize;
            let mut dense = vec![json!(0.0); size];
            let Value::List(indices) = row.get_by_name("indices").unwrap() else {
                panic!("indices")
            };
            for (index, value) in indices.iter().zip(values.as_array().unwrap()) {
                dense[index.as_i64().unwrap() as usize] = value.clone();
            }
            json!(dense)
        }
        _ => json!(v.as_f64().unwrap()),
    }
}

fn assert_values(actual: &Json, expected: &Json, pca: bool) {
    match expected {
        Json::Array(values) => {
            let actual = actual.as_array().unwrap();
            assert_eq!(actual.len(), values.len());
            for (a, e) in actual.iter().zip(values) {
                assert_values(a, e, pca);
            }
        }
        Json::Number(n) => {
            let (mut a, mut e) = (actual.as_f64().unwrap(), n.as_f64().unwrap());
            if pca {
                a = a.abs();
                e = e.abs();
            }
            assert!((a - e).abs() <= 1e-7 + 1e-5 * e.abs(), "{a} != {e}");
        }
        _ => assert_eq!(actual, expected),
    }
}

fn wire_schema(mut value: Json) -> Json {
    // JVM UDTs arrive by class name; Connect omits their backing SQL schema.
    match &mut value {
        Json::Object(map) => {
            if map.get("type").is_some_and(|v| v == "udt") {
                map.remove("sqlType");
            }
            for v in map.values_mut() {
                *v = wire_schema(v.take());
            }
        }
        Json::Array(values) => {
            for v in values {
                *v = wire_schema(v.take());
            }
        }
        _ => {}
    }
    value
}

#[test]
fn all_feature_stages_match_official_results_and_retain_models() {
    let Some(session) = live_session() else {
        return;
    };
    let golden: Json =
        serde_json::from_str(include_str!("../../../tests/golden/ml_features.json")).unwrap();
    let vector = Column::new(Expression::function(
        "array_to_vector",
        vec![col("xs").expression().clone()],
    ));
    let input = session
        .sql(golden["sql"].as_str().unwrap())
        .unwrap()
        .with_column("features", vector.clone())
        .with_column("rawFeatures", vector)
        .coalesce(1);
    for case in golden["cases"].as_array().unwrap() {
        let id = case["id"].as_str().unwrap();
        eprintln!("feature: {id}");
        // Each builder/model drops inside output(); the lazy plan keeps its owners alive.
        let result = output(id, &input)
            .order_by(vec![col("id").asc().expression().clone()])
            .select(
                case["columns"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| col(v.as_str().unwrap())),
            );
        let actual: Json = result
            .collect()
            .unwrap()
            .iter()
            .map(|row| row.values().iter().map(normalized).collect::<Json>())
            .collect();
        assert_values(&actual, &case["rows"], case["name"] == "PCA");
        assert_eq!(
            wire_schema(result.schema().unwrap().json_value()),
            wire_schema(case["schema"].clone()),
            "{id} schema"
        );
    }
}

#[test]
fn stop_words_remover_loads_the_default_stop_words_of_the_server() {
    let Some(session) = live_session() else {
        return;
    };
    // As PySpark's StopWordsRemover.loadDefaultStopWords, a call of the ML helper.
    let english =
        spark_connect::ml::StopWordsRemover::load_default_stop_words(&session, "english").unwrap();
    assert_eq!(english.len(), 181);
    assert_eq!(&english[..4], ["i", "me", "my", "myself"]);
    let error = spark_connect::ml::StopWordsRemover::load_default_stop_words(&session, "klingon")
        .err()
        .unwrap();
    assert!(
        error
            .to_string()
            .contains("klingon is not in the supported language list"),
        "{error}"
    );
    session.stop().unwrap();
}
