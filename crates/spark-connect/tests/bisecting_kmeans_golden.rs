//! Match all BisectingKMeans parameter encodings and integer seed boundaries to PySpark.
use base64::{engine::general_purpose::STANDARD, Engine};
use prost::Message;
use spark_connect::ml::{BisectingKMeans, Estimator};

#[test]
fn bisecting_kmeans_parameters_match_official_pyspark() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/golden/bisecting_kmeans.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let seed = case["seed"].as_i64().unwrap();
        let bytes = STANDARD
            .decode(case["params_b64"].as_str().unwrap())
            .unwrap();
        let expected = spark_connect_proto::MlParams::decode(bytes.as_slice()).unwrap();
        let estimator = BisectingKMeans::new()
            .set_features_col("input")
            .set_prediction_col("cluster")
            .set_k(3)
            .set_max_iter(50)
            .set_seed(seed)
            .set_min_divisible_cluster_size(0.25)
            .set_distance_measure("cosine")
            .set_weight_col("weight");
        assert_eq!(estimator.params().to_proto(), expected, "seed {seed}");
    }
    assert!(BisectingKMeans::default()
        .params()
        .to_proto()
        .params
        .is_empty());
}
