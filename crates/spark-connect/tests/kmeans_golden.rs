//! Match all KMeans parameter encodings and integer seed boundaries to PySpark.
use base64::{engine::general_purpose::STANDARD, Engine};
use prost::Message;
use spark_connect::ml::{Estimator, KMeans};

#[test]
fn kmeans_parameters_match_official_pyspark() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/golden/kmeans.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let seed = case["seed"].as_i64().unwrap();
        let bytes = STANDARD
            .decode(case["params_b64"].as_str().unwrap())
            .unwrap();
        let expected = spark_connect_proto::MlParams::decode(bytes.as_slice()).unwrap();
        let estimator = KMeans::new()
            .set_features_col("input")
            .set_prediction_col("cluster")
            .set_k(3)
            .set_init_mode("random")
            .set_init_steps(3)
            .set_tol(1e-5)
            .set_max_iter(50)
            .set_seed(seed)
            .set_distance_measure("euclidean")
            .set_weight_col("weight")
            .set_solver("block")
            .set_max_block_size_in_mb(2.0);
        assert_eq!(estimator.params().to_proto(), expected, "seed {seed}");
    }
    assert!(KMeans::default().params().to_proto().params.is_empty());
}
