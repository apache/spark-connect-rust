//! Match every DecisionTreeRegressor parameter and seed boundary to official PySpark.
use base64::{engine::general_purpose::STANDARD, Engine};
use prost::Message;
use spark_connect::ml::{DecisionTreeRegressor, Estimator};

#[test]
fn decision_tree_regressor_parameters_match_official_pyspark() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/golden/decision_tree_regressor.json"
    ))
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let seed = case["seed"].as_i64().unwrap();
        let bytes = STANDARD
            .decode(case["params_b64"].as_str().unwrap())
            .unwrap();
        let expected = spark_connect_proto::MlParams::decode(bytes.as_slice()).unwrap();
        let estimator = DecisionTreeRegressor::new()
            .set_features_col("input")
            .set_label_col("target")
            .set_prediction_col("value")
            .set_max_depth(3)
            .set_max_bins(16)
            .set_min_instances_per_node(2)
            .set_min_info_gain(0.01)
            .set_max_memory_in_mb(64)
            .set_cache_node_ids(true)
            .set_checkpoint_interval(-1)
            .set_impurity("variance")
            .set_seed(seed)
            .set_variance_col("uncertainty")
            .set_leaf_col("leaves")
            .set_min_weight_fraction_per_node(0.1)
            .set_weight_col("weight");
        assert_eq!(estimator.params().to_proto(), expected, "seed {seed}");
    }
    assert!(DecisionTreeRegressor::default()
        .params()
        .to_proto()
        .params
        .is_empty());
}
