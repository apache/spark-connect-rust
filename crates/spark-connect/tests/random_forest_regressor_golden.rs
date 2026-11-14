//! Match every RandomForestRegressor parameter and seed boundary to official PySpark.
use base64::{engine::general_purpose::STANDARD, Engine};
use prost::Message;
use spark_connect::ml::{Estimator, RandomForestRegressor};

#[test]
fn random_forest_regressor_parameters_match_official_pyspark() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/golden/random_forest_regressor.json"
    ))
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let seed = case["seed"].as_i64().unwrap();
        let bytes = STANDARD
            .decode(case["params_b64"].as_str().unwrap())
            .unwrap();
        let expected = spark_connect_proto::MlParams::decode(bytes.as_slice()).unwrap();
        let estimator = RandomForestRegressor::new()
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
            .set_num_trees(7)
            .set_feature_subset_strategy("all")
            .set_seed(seed)
            .set_subsampling_rate(0.75)
            .set_leaf_col("leaves")
            .set_min_weight_fraction_per_node(0.1)
            .set_weight_col("weight")
            .set_bootstrap(false);
        assert_eq!(estimator.params().to_proto(), expected, "seed {seed}");
    }
    assert!(RandomForestRegressor::default()
        .params()
        .to_proto()
        .params
        .is_empty());
}

#[test]
fn random_forest_regressor_getters_track_the_parameters() {
    let mut estimator = RandomForestRegressor::default();
    assert_eq!(estimator.features_col(), "features");
    assert_eq!(estimator.label_col(), "label");
    assert_eq!(estimator.prediction_col(), "prediction");
    assert_eq!(estimator.max_depth(), 5);
    assert_eq!(estimator.max_bins(), 32);
    assert_eq!(estimator.min_instances_per_node(), 1);
    assert_eq!(estimator.min_info_gain(), 0.0);
    assert_eq!(estimator.max_memory_in_mb(), 256);
    assert!(!estimator.cache_node_ids());
    assert_eq!(estimator.checkpoint_interval(), 10);
    assert_eq!(estimator.impurity(), "variance");
    assert_eq!(estimator.num_trees(), 20);
    assert_eq!(estimator.feature_subset_strategy(), "auto");
    // Spark's default seed: String.hashCode of the JVM class name.
    assert_eq!(estimator.seed(), 235498149);
    assert_eq!(estimator.subsampling_rate(), 1.0);
    assert_eq!(estimator.leaf_col(), "");
    assert_eq!(estimator.min_weight_fraction_per_node(), 0.0);
    assert_eq!(estimator.weight_col(), None);
    assert!(estimator.bootstrap());
    let params = std::mem::take(estimator.params_mut())
        .set_param_string("labelCol", "target")
        .set_param_int("numTrees", 7)
        .set_param_double("subsamplingRate", 0.75)
        .set_param_bool("bootstrap", false);
    *estimator.params_mut() = params;
    assert_eq!(estimator.label_col(), "target");
    assert_eq!(estimator.num_trees(), 7);
    assert_eq!(estimator.subsampling_rate(), 0.75);
    assert!(!estimator.bootstrap());
}
