//! Match all NaiveBayes parameters and model types to official PySpark.
use base64::{engine::general_purpose::STANDARD, Engine};
use prost::Message;
use spark_connect::ml::{Estimator, NaiveBayes};

#[test]
fn naive_bayes_parameters_match_official_pyspark() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/golden/naive_bayes.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let model_type = case["model_type"].as_str().unwrap();
        let bytes = STANDARD
            .decode(case["params_b64"].as_str().unwrap())
            .unwrap();
        let expected = spark_connect_proto::MlParams::decode(bytes.as_slice()).unwrap();
        let estimator = NaiveBayes::new()
            .set_features_col("input")
            .set_label_col("target")
            .set_prediction_col("class")
            .set_probability_col("probs")
            .set_raw_prediction_col("scores")
            .set_smoothing(0.25)
            .set_model_type(model_type)
            .set_thresholds([0.25, 0.5, 1.0])
            .set_weight_col("weight");
        assert_eq!(estimator.params().to_proto(), expected, "{model_type}");
    }
    assert!(NaiveBayes::default().params().to_proto().params.is_empty());
}
