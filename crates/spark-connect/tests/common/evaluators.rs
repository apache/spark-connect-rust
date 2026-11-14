// SPDX-License-Identifier: Apache-2.0
//! Reference fixture construction shared by Rust evaluator tests.
use serde_json::Value as Json;
use spark_connect::ml::*;

pub fn fixture() -> Json {
    serde_json::from_str(include_str!("../../../../tests/golden/evaluators.json")).unwrap()
}

pub fn configured(name: &str, values: &Json) -> Box<dyn Evaluator> {
    match name {
        "BinaryClassificationEvaluator" => {
            let mut evaluator = BinaryClassificationEvaluator::new();
            for (key, value) in values.as_object().unwrap() {
                evaluator = match key.as_str() {
                    "rawPredictionCol" => evaluator.set_raw_prediction_col(value.as_str().unwrap()),
                    "labelCol" => evaluator.set_label_col(value.as_str().unwrap()),
                    "metricName" => evaluator.set_metric_name(value.as_str().unwrap()),
                    "weightCol" => evaluator.set_weight_col(value.as_str().unwrap()),
                    "numBins" => {
                        evaluator.set_num_bins(value.as_i64().unwrap().try_into().unwrap())
                    }
                    key => panic!("unexpected evaluator parameter: {key}"),
                };
            }
            Box::new(evaluator)
        }
        "RegressionEvaluator" => {
            let mut evaluator = RegressionEvaluator::new();
            for (key, value) in values.as_object().unwrap() {
                evaluator = match key.as_str() {
                    "predictionCol" => evaluator.set_prediction_col(value.as_str().unwrap()),
                    "labelCol" => evaluator.set_label_col(value.as_str().unwrap()),
                    "metricName" => evaluator.set_metric_name(value.as_str().unwrap()),
                    "weightCol" => evaluator.set_weight_col(value.as_str().unwrap()),
                    "throughOrigin" => evaluator.set_through_origin(value.as_bool().unwrap()),
                    key => panic!("unexpected evaluator parameter: {key}"),
                };
            }
            Box::new(evaluator)
        }
        "MulticlassClassificationEvaluator" => {
            let mut evaluator = MulticlassClassificationEvaluator::new();
            for (key, value) in values.as_object().unwrap() {
                evaluator = match key.as_str() {
                    "predictionCol" => evaluator.set_prediction_col(value.as_str().unwrap()),
                    "labelCol" => evaluator.set_label_col(value.as_str().unwrap()),
                    "metricName" => evaluator.set_metric_name(value.as_str().unwrap()),
                    "weightCol" => evaluator.set_weight_col(value.as_str().unwrap()),
                    "metricLabel" => evaluator.set_metric_label(value.as_f64().unwrap()),
                    "beta" => evaluator.set_beta(value.as_f64().unwrap()),
                    "probabilityCol" => evaluator.set_probability_col(value.as_str().unwrap()),
                    "eps" => evaluator.set_eps(value.as_f64().unwrap()),
                    key => panic!("unexpected evaluator parameter: {key}"),
                };
            }
            Box::new(evaluator)
        }
        "MultilabelClassificationEvaluator" => {
            let mut evaluator = MultilabelClassificationEvaluator::new();
            for (key, value) in values.as_object().unwrap() {
                evaluator = match key.as_str() {
                    "predictionCol" => evaluator.set_prediction_col(value.as_str().unwrap()),
                    "labelCol" => evaluator.set_label_col(value.as_str().unwrap()),
                    "metricName" => evaluator.set_metric_name(value.as_str().unwrap()),
                    "metricLabel" => evaluator.set_metric_label(value.as_f64().unwrap()),
                    key => panic!("unexpected evaluator parameter: {key}"),
                };
            }
            Box::new(evaluator)
        }
        "ClusteringEvaluator" => {
            let mut evaluator = ClusteringEvaluator::new();
            for (key, value) in values.as_object().unwrap() {
                evaluator = match key.as_str() {
                    "predictionCol" => evaluator.set_prediction_col(value.as_str().unwrap()),
                    "featuresCol" => evaluator.set_features_col(value.as_str().unwrap()),
                    "metricName" => evaluator.set_metric_name(value.as_str().unwrap()),
                    "distanceMeasure" => evaluator.set_distance_measure(value.as_str().unwrap()),
                    "weightCol" => evaluator.set_weight_col(value.as_str().unwrap()),
                    key => panic!("unexpected evaluator parameter: {key}"),
                };
            }
            Box::new(evaluator)
        }
        "RankingEvaluator" => {
            let mut evaluator = RankingEvaluator::new();
            for (key, value) in values.as_object().unwrap() {
                evaluator = match key.as_str() {
                    "predictionCol" => evaluator.set_prediction_col(value.as_str().unwrap()),
                    "labelCol" => evaluator.set_label_col(value.as_str().unwrap()),
                    "metricName" => evaluator.set_metric_name(value.as_str().unwrap()),
                    "k" => evaluator.set_k(value.as_i64().unwrap().try_into().unwrap()),
                    key => panic!("unexpected evaluator parameter: {key}"),
                };
            }
            Box::new(evaluator)
        }
        name => panic!("unexpected evaluator: {name}"),
    }
}
