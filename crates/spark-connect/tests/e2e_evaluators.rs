// SPDX-License-Identifier: Apache-2.0
//! All Spark ML evaluator metrics, checked against the reference client on a live server.
mod common;
#[path = "common/evaluators.rs"]
mod evaluators;

use spark_connect::column::col;
use spark_connect::expression::Expression;
use spark_connect::{Column, DataFrame, SparkSession};
use std::collections::HashMap;

fn frames(session: &SparkSession, fixture: &serde_json::Value) -> HashMap<String, DataFrame> {
    fixture["datasets"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, spec)| {
            let mut df = session.sql(spec["sql"].as_str().unwrap()).unwrap();
            for column in spec["vectors"].as_array().unwrap() {
                let name = column.as_str().unwrap();
                df = df.with_column(
                    name,
                    Column::new(Expression::function(
                        "array_to_vector",
                        vec![col(name).expression().clone()],
                    )),
                );
            }
            // One partition, as scripts/capture_evaluators_golden.py captures it: a binned metric
            // of BinaryClassificationMetrics depends on the partitioning, which local[*] takes from
            // the server's cores.
            (name.clone(), df.coalesce(1))
        })
        .collect()
}

#[test]
fn all_evaluator_metrics_match_official_pyspark() {
    let Some(session) = common::live_session() else {
        return;
    };
    let fixture = evaluators::fixture();
    let frames = frames(&session, &fixture);
    for case in fixture["cases"].as_array().unwrap() {
        let evaluator = evaluators::configured(case["class"].as_str().unwrap(), &case["values"]);
        let actual = evaluator
            .evaluate(&frames[case["dataset"].as_str().unwrap()])
            .unwrap();
        let expected = case["value"].as_f64().unwrap();
        assert!(
            (actual - expected).abs() <= 1e-10 + 1e-10 * expected.abs(),
            "{}: expected {expected}, got {actual}",
            case["name"]
        );
        assert_eq!(
            evaluator.is_larger_better(),
            case["larger"].as_bool().unwrap()
        );
    }
    session.stop().unwrap();
}

#[test]
fn evaluators_preserve_sparks_parameter_and_input_validation() {
    let Some(session) = common::live_session() else {
        return;
    };
    let fixture = evaluators::fixture();
    let frames = frames(&session, &fixture);
    for name in fixture["classes"].as_object().unwrap().keys() {
        let case = fixture["cases"]
            .as_array()
            .unwrap()
            .iter()
            .find(|case| case["class"] == *name)
            .unwrap();
        let mut values = case["values"].clone();
        values["metricName"] = serde_json::json!("invalid");
        let error = evaluators::configured(name, &values)
            .evaluate(&frames[case["dataset"].as_str().unwrap()])
            .unwrap_err();
        assert!(error.to_string().contains("metricName"), "{name}: {error}");
    }
    for case in fixture["inputs"].as_array().unwrap() {
        let evaluator = evaluators::configured(case["class"].as_str().unwrap(), &case["values"]);
        let frame = session.sql(case["sql"].as_str().unwrap()).unwrap();
        if case["accepted"].as_bool().unwrap() {
            let actual = evaluator.evaluate(&frame).unwrap();
            assert!((actual - case["result"].as_f64().unwrap()).abs() < 1e-10);
            continue;
        }
        let error = evaluator.evaluate(&frame).unwrap_err();
        assert!(
            error
                .to_string()
                .contains(case["message"].as_str().unwrap()),
            "{}: {error}",
            case["name"]
        );
    }
    session.stop().unwrap();
}
