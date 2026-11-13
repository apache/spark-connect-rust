// SPDX-License-Identifier: Apache-2.0
//! Frequent-pattern mining on a live server. Set SPARK_REMOTE to run.
mod common;
use common::live_session;
use serde_json::{json, Value as Json};
use spark_connect::ml::{FPGrowth, Model, PrefixSpan, Transformer};
use spark_connect::row::Value;
use spark_connect::DataFrame;

fn value(v: &Value) -> Json {
    match v {
        Value::List(v) => v.iter().map(value).collect(),
        Value::String(v) => json!(v),
        Value::Long(v) => json!(v),
        Value::Integer(v) => json!(v),
        Value::Double(v) => json!(v),
        Value::Null => Json::Null,
        _ => panic!("unexpected mining value: {v:?}"),
    }
}
fn sort(values: &mut Json) {
    values.as_array_mut().unwrap().sort_by_key(Json::to_string);
}
fn normalized(frame: &DataFrame, kind: &str) -> Json {
    let mut rows: Vec<Json> = frame
        .collect()
        .unwrap()
        .iter()
        .map(|r| r.values().iter().map(value).collect())
        .collect();
    for row in &mut rows {
        let row = row.as_array_mut().unwrap();
        match kind {
            "predictions" => sort(row.last_mut().unwrap()),
            "itemsets" => sort(&mut row[0]),
            "rules" => {
                sort(&mut row[0]);
                sort(&mut row[1]);
            }
            _ => {
                for set in row[0].as_array_mut().unwrap() {
                    sort(set);
                }
            }
        }
    }
    rows.sort_by_key(Json::to_string);
    json!(rows)
}
fn assert_values(a: &Json, e: &Json) {
    match e {
        Json::Array(e) => {
            let a = a.as_array().unwrap();
            assert_eq!(a.len(), e.len());
            for (a, e) in a.iter().zip(e) {
                assert_values(a, e);
            }
        }
        Json::Number(e) => assert!(
            (a.as_f64().unwrap() - e.as_f64().unwrap()).abs() < 1e-10,
            "{a} != {e}"
        ),
        _ => assert_eq!(a, e),
    }
}
#[test]
fn fpgrowth_itemsets_rules_predictions_and_model_lifetime() {
    let Some(s) = live_session() else { return };
    let golden: Json =
        serde_json::from_str(include_str!("../../../tests/golden/frequent_patterns.json")).unwrap();
    for case in golden["fpgrowth"].as_array().unwrap() {
        let v = &case["values"];
        let mut model = FPGrowth::new()
            .set_items_col("basket")
            .set_prediction_col("suggested")
            .set_min_support(v["minSupport"].as_f64().unwrap())
            .set_min_confidence(v["minConfidence"].as_f64().unwrap())
            .set_num_partitions(v["numPartitions"].as_i64().unwrap() as i32)
            .fit(&s.sql(case["sql"].as_str().unwrap()).unwrap())
            .unwrap();
        let held = s.sql(case["held"].as_str().unwrap()).unwrap();
        let tables = [
            ("itemsets", model.freq_itemsets().unwrap()),
            ("rules", model.association_rules().unwrap()),
            ("predictions", model.clone_box().transform(&held).unwrap()),
        ];
        model = model.set_min_confidence(0.95).set_prediction_col("changed");
        let changed = model.transform(&held).unwrap();
        drop(model);
        for (kind, table) in tables {
            assert_values(&normalized(&table, kind), &case[kind]["rows"]);
            assert_eq!(table.schema().unwrap().json_value(), case[kind]["schema"]);
        }
        assert_values(
            &normalized(&changed, "predictions"),
            &case["changed_predictions"]["rows"],
        );
        assert_eq!(
            changed.schema().unwrap().json_value(),
            case["changed_predictions"]["schema"]
        );
    }
}
#[test]
fn prefixspan_matches_reference_results() {
    let Some(s) = live_session() else { return };
    let golden: Json =
        serde_json::from_str(include_str!("../../../tests/golden/frequent_patterns.json")).unwrap();
    let input = s.sql(golden["sequences"].as_str().unwrap()).unwrap();
    for case in golden["prefixspan"].as_array().unwrap() {
        let v = &case["values"];
        let stage = PrefixSpan::new()
            .set_sequence_col("events")
            .set_min_support(0.5)
            .set_max_pattern_length(v["maxPatternLength"].as_i64().unwrap() as i32)
            .set_max_local_proj_db_size(v["maxLocalProjDBSize"].as_i64().unwrap());
        let result = stage.find_frequent_sequential_patterns(&input).unwrap();
        drop(stage);
        assert_values(&normalized(&result, "sequences"), &case["rows"]);
        assert_eq!(result.schema().unwrap().json_value(), case["schema"]);
    }
}
