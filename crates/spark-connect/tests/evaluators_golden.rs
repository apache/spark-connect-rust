// SPDX-License-Identifier: Apache-2.0
//! Match all evaluator parameters, defaults and metric directions to official PySpark.
#[path = "common/evaluators.rs"]
mod evaluators;
use base64::{engine::general_purpose::STANDARD, Engine};
use prost::Message;
use spark_connect::ml::*;

#[test]
fn evaluator_parameter_protocols_match_official_pyspark() {
    let fixture = evaluators::fixture();
    for (name, info) in fixture["classes"].as_object().unwrap() {
        for case in info["params"].as_array().unwrap() {
            // Python's None and arbitrary-size integers are checked by the binding tests.
            if case["values"]
                .as_object()
                .unwrap()
                .iter()
                .any(|(key, value)| {
                    value.is_null()
                        || (matches!(key.as_str(), "k" | "numBins")
                            && i32::try_from(value.as_i64().unwrap()).is_err())
                })
            {
                continue;
            }
            let bytes = STANDARD
                .decode(case["params_b64"].as_str().unwrap())
                .unwrap();
            let expected = spark_connect_proto::MlParams::decode(bytes.as_slice()).unwrap();
            assert_eq!(
                evaluators::configured(name, &case["values"])
                    .params()
                    .to_proto(),
                expected,
                "{name}"
            );
        }
    }
}

#[test]
fn evaluator_metric_directions_match_official_pyspark() {
    let fixture = evaluators::fixture();
    for (name, info) in fixture["classes"].as_object().unwrap() {
        for case in info["directions"].as_array().unwrap() {
            if case["metric"].is_null() {
                continue;
            }
            let evaluator =
                evaluators::configured(name, &serde_json::json!({"metricName": case["metric"]}));
            assert_eq!(
                evaluator.is_larger_better(),
                case["larger"].as_bool().unwrap(),
                "{name}: {}",
                case["metric"]
            );
        }
    }
}

#[test]
fn evaluator_defaults_match_official_pyspark() {
    let fixture = evaluators::fixture();
    {
        let evaluator = BinaryClassificationEvaluator::default();
        assert!(evaluator.params().to_proto().params.is_empty());
        let defaults = &fixture["classes"]["BinaryClassificationEvaluator"]["defaults"];
        assert_eq!(
            evaluator.raw_prediction_col(),
            defaults["rawPredictionCol"].as_str().unwrap()
        );
        assert_eq!(
            evaluator.label_col(),
            defaults["labelCol"].as_str().unwrap()
        );
        assert_eq!(
            evaluator.metric_name(),
            defaults["metricName"].as_str().unwrap()
        );
        assert_eq!(evaluator.weight_col(), None);
        assert_eq!(
            evaluator.num_bins(),
            defaults["numBins"].as_i64().unwrap() as i32
        );
    }
    {
        let evaluator = RegressionEvaluator::default();
        assert!(evaluator.params().to_proto().params.is_empty());
        let defaults = &fixture["classes"]["RegressionEvaluator"]["defaults"];
        assert_eq!(
            evaluator.prediction_col(),
            defaults["predictionCol"].as_str().unwrap()
        );
        assert_eq!(
            evaluator.label_col(),
            defaults["labelCol"].as_str().unwrap()
        );
        assert_eq!(
            evaluator.metric_name(),
            defaults["metricName"].as_str().unwrap()
        );
        assert_eq!(evaluator.weight_col(), None);
        assert_eq!(
            evaluator.through_origin(),
            defaults["throughOrigin"].as_bool().unwrap()
        );
    }
    {
        let evaluator = MulticlassClassificationEvaluator::default();
        assert!(evaluator.params().to_proto().params.is_empty());
        let defaults = &fixture["classes"]["MulticlassClassificationEvaluator"]["defaults"];
        assert_eq!(
            evaluator.prediction_col(),
            defaults["predictionCol"].as_str().unwrap()
        );
        assert_eq!(
            evaluator.label_col(),
            defaults["labelCol"].as_str().unwrap()
        );
        assert_eq!(
            evaluator.metric_name(),
            defaults["metricName"].as_str().unwrap()
        );
        assert_eq!(evaluator.weight_col(), None);
        assert_eq!(
            evaluator.metric_label(),
            defaults["metricLabel"].as_f64().unwrap()
        );
        assert_eq!(evaluator.beta(), defaults["beta"].as_f64().unwrap());
        assert_eq!(
            evaluator.probability_col(),
            defaults["probabilityCol"].as_str().unwrap()
        );
        assert_eq!(evaluator.eps(), defaults["eps"].as_f64().unwrap());
    }
    {
        let evaluator = MultilabelClassificationEvaluator::default();
        assert!(evaluator.params().to_proto().params.is_empty());
        let defaults = &fixture["classes"]["MultilabelClassificationEvaluator"]["defaults"];
        assert_eq!(
            evaluator.prediction_col(),
            defaults["predictionCol"].as_str().unwrap()
        );
        assert_eq!(
            evaluator.label_col(),
            defaults["labelCol"].as_str().unwrap()
        );
        assert_eq!(
            evaluator.metric_name(),
            defaults["metricName"].as_str().unwrap()
        );
        assert_eq!(
            evaluator.metric_label(),
            defaults["metricLabel"].as_f64().unwrap()
        );
    }
    {
        let evaluator = ClusteringEvaluator::default();
        assert!(evaluator.params().to_proto().params.is_empty());
        let defaults = &fixture["classes"]["ClusteringEvaluator"]["defaults"];
        assert_eq!(
            evaluator.prediction_col(),
            defaults["predictionCol"].as_str().unwrap()
        );
        assert_eq!(
            evaluator.features_col(),
            defaults["featuresCol"].as_str().unwrap()
        );
        assert_eq!(
            evaluator.metric_name(),
            defaults["metricName"].as_str().unwrap()
        );
        assert_eq!(
            evaluator.distance_measure(),
            defaults["distanceMeasure"].as_str().unwrap()
        );
        assert_eq!(evaluator.weight_col(), None);
    }
    {
        let evaluator = RankingEvaluator::default();
        assert!(evaluator.params().to_proto().params.is_empty());
        let defaults = &fixture["classes"]["RankingEvaluator"]["defaults"];
        assert_eq!(
            evaluator.prediction_col(),
            defaults["predictionCol"].as_str().unwrap()
        );
        assert_eq!(
            evaluator.label_col(),
            defaults["labelCol"].as_str().unwrap()
        );
        assert_eq!(
            evaluator.metric_name(),
            defaults["metricName"].as_str().unwrap()
        );
        assert_eq!(evaluator.k(), defaults["k"].as_i64().unwrap() as i32);
    }
}
