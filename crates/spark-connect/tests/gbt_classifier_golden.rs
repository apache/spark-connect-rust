//! Match every GBTClassifier parameter and seed boundary to official PySpark.
use base64::{engine::general_purpose::STANDARD, Engine};
use prost::Message;
use spark_connect::ml::{Estimator, GBTClassifier};

#[test]
fn gbt_classifier_parameters_match_official_pyspark() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/golden/gbt_classifier.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let seed = case["seed"].as_i64().unwrap();
        let bytes = STANDARD
            .decode(case["params_b64"].as_str().unwrap())
            .unwrap();
        let expected = spark_connect_proto::MlParams::decode(bytes.as_slice()).unwrap();
        let estimator = GBTClassifier::new()
            .set_features_col("input")
            .set_label_col("target")
            .set_prediction_col("class")
            .set_probability_col("probs")
            .set_raw_prediction_col("margin")
            .set_max_depth(3)
            .set_max_bins(16)
            .set_min_instances_per_node(2)
            .set_min_info_gain(0.01)
            .set_max_memory_in_mb(64)
            .set_cache_node_ids(true)
            .set_checkpoint_interval(-1)
            .set_impurity("variance")
            .set_feature_subset_strategy("all")
            .set_loss_type("logistic")
            .set_max_iter(4)
            .set_step_size(0.2)
            .set_validation_tol(0.05)
            .set_validation_indicator_col("validation")
            .set_seed(seed)
            .set_subsampling_rate(0.75)
            .set_leaf_col("leaves")
            .set_min_weight_fraction_per_node(0.1)
            .set_weight_col("weight")
            .set_thresholds([0.25, 0.75]);
        assert_eq!(estimator.params().to_proto(), expected, "seed {seed}");
    }
    assert!(GBTClassifier::default()
        .params()
        .to_proto()
        .params
        .is_empty());
}
