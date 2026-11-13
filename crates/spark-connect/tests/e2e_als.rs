//! Reference-backed ALS fit, factor, recommendation and lazy ownership tests.
#[path = "common/als.rs"]
mod als;
mod common;

use common::live_session;
use serde_json::{json, Value as Json};
use spark_connect::column::col;
use spark_connect::ml::{Estimator, StringIndexer, Transformer, ALS};
use spark_connect::row::Value;
use spark_connect::DataFrame;

fn object(fields: &[String], values: &[Value]) -> Json {
    fields
        .iter()
        .zip(values)
        .map(|(name, value)| (name.clone(), normalize(value)))
        .collect()
}

fn normalize(value: &Value) -> Json {
    match value {
        Value::Null => Json::Null,
        Value::Bool(v) => json!(v),
        Value::Byte(v) => json!(v),
        Value::Short(v) => json!(v),
        Value::Integer(v) => json!(v),
        Value::Long(v) => json!(v),
        Value::Float(v) if v.is_nan() => json!({"float": "nan"}),
        Value::Double(v) if v.is_nan() => json!({"float": "nan"}),
        Value::Float(v) => json!(v),
        Value::Double(v) => json!(v),
        Value::String(v) => json!(v),
        Value::List(v) => v.iter().map(normalize).collect(),
        Value::Struct(v) => object(v.fields(), v.values()),
        other => panic!("unexpected ALS result value: {other:?}"),
    }
}

fn assert_close(actual: &Json, expected: &Json) {
    match (actual, expected) {
        (Json::Object(a), Json::Object(e)) => {
            assert_eq!(a.keys().collect::<Vec<_>>(), e.keys().collect::<Vec<_>>());
            for (key, value) in e {
                assert_close(&a[key], value);
            }
        }
        (Json::Array(a), Json::Array(e)) => {
            assert_eq!(a.len(), e.len());
            for (value, expected) in a.iter().zip(e) {
                assert_close(value, expected);
            }
        }
        (Json::Number(a), Json::Number(e)) if a.is_f64() || e.is_f64() => {
            let (a, e) = (a.as_f64().unwrap(), e.as_f64().unwrap());
            assert!(
                (a - e).abs() <= 2e-5 + 2e-4 * e.abs(),
                "expected {e}, got {a}"
            );
        }
        _ => assert_eq!(actual, expected),
    }
}

fn assert_result(df: DataFrame, expected: &Json) {
    assert_eq!(df.schema().unwrap().json_value(), expected["schema"]);
    let mut rows: Vec<Json> = df
        .collect()
        .unwrap()
        .iter()
        .map(|row| object(row.fields(), row.values()))
        .collect();
    rows.sort_by_key(|row| {
        ["id", "u", "i"]
            .iter()
            .find_map(|key| row[*key].as_i64())
            .unwrap()
    });
    assert_close(&json!(rows), &expected["rows"]);
}

#[test]
fn als_explicit_and_implicit_results_match_official_and_keep_models_alive() {
    let Some(spark) = live_session() else {
        return;
    };
    let fixture = als::fixture();
    let training = spark.sql(fixture["train"].as_str().unwrap()).unwrap();
    let held = spark.sql(fixture["held"].as_str().unwrap()).unwrap();
    let users = spark.sql(fixture["users"].as_str().unwrap()).unwrap();
    let items = spark.sql(fixture["items"].as_str().unwrap()).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let mut model = als::configured(&case["values"]).fit(&training).unwrap();
        assert_eq!(model.rank(), case["rank"].as_i64().unwrap() as i32);
        let mut params: Vec<_> = model.params().to_proto().params.into_keys().collect();
        params.sort();
        assert_eq!(
            params,
            [
                "blockSize",
                "coldStartStrategy",
                "itemCol",
                "predictionCol",
                "userCol"
            ]
        );
        let predictions = model.transform(&held).unwrap();
        let results = [
            ("predictions", predictions),
            ("user_factors", model.user_factors().unwrap()),
            ("item_factors", model.item_factors().unwrap()),
            ("all_users", model.recommend_for_all_users(2).unwrap()),
            ("all_items", model.recommend_for_all_items(2).unwrap()),
            (
                "user_subset",
                model.recommend_for_user_subset(&users, 2).unwrap(),
            ),
            (
                "item_subset",
                model.recommend_for_item_subset(&items, 2).unwrap(),
            ),
        ];
        model = model
            .set_cold_start_strategy("drop")
            .set_prediction_col("filtered");
        let dropped = model.transform(&held).unwrap();
        let other = spark.new_session();
        let foreign = other.range(1).unwrap();
        assert!(model.transform(&foreign).is_err());
        assert!(model.recommend_for_user_subset(&foreign, 2).is_err());
        assert!(model.recommend_for_item_subset(&foreign, 2).is_err());
        other.stop().unwrap();
        drop(model);
        for (name, frame) in results {
            assert_result(frame, &case[name]);
        }
        assert_result(dropped, &case["drop_predictions"]);
    }
    spark.stop().unwrap();
}

#[test]
fn als_recommendations_retain_models_used_to_build_the_subset() {
    let Some(spark) = live_session() else {
        return;
    };
    let fixture = als::fixture();
    let training = spark.sql(fixture["train"].as_str().unwrap()).unwrap();
    let model = als::configured(&fixture["cases"][0]["values"])
        .fit(&training)
        .unwrap();
    let labels = spark
        .sql("SELECT * FROM VALUES ('a'), ('a'), ('b') AS t(label)")
        .unwrap();
    let mut indexer = StringIndexer::new()
        .set_input_col("label")
        .set_output_col("index")
        .fit(&labels)
        .unwrap();
    let subset = indexer
        .transform(&labels)
        .unwrap()
        .with_column("u", col("index").cast_str("int"))
        .select([col("u")]);
    let deferred = model.recommend_for_user_subset(&subset, 2).unwrap();
    let expected = model
        .recommend_for_user_subset(
            &spark.sql("SELECT * FROM VALUES (0), (1) AS t(u)").unwrap(),
            2,
        )
        .unwrap();
    let expected = json!({
        "schema": expected.schema().unwrap().json_value(),
        "rows": expected.order_by(vec![col("u").asc().expression().clone()]).collect().unwrap().iter().map(|r| object(r.fields(), r.values())).collect::<Vec<_>>()
    });
    drop(subset);
    drop(indexer);
    drop(model);
    assert_result(deferred, &expected);
    spark.stop().unwrap();
}

#[test]
fn als_rejects_invalid_parameters_and_fractional_ids() {
    let Some(spark) = live_session() else {
        return;
    };
    let training = spark
        .sql("SELECT * FROM VALUES (0, 0, 1.0D), (1, 1, 2.0D) AS t(user, item, rating)")
        .unwrap();
    assert!(ALS::new().set_rank(0).fit(&training).is_err());
    assert!(ALS::new().set_reg_param(-1.0).fit(&training).is_err());
    assert!(ALS::new()
        .set_cold_start_strategy("invalid")
        .fit(&training)
        .is_err());
    let fractional = spark
        .sql("SELECT 0.5D AS user, 0 AS item, 1.0D AS rating")
        .unwrap();
    assert!(ALS::new().fit(&fractional).is_err());
    spark.stop().unwrap();
}
