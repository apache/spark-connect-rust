//! Match all GaussianMixture parameter encodings and integer seed boundaries to PySpark.
use base64::{engine::general_purpose::STANDARD, Engine};
use prost::Message;
use spark_connect::ml::{Estimator, GaussianMixture};

#[test]
fn gaussian_mixture_parameters_match_official_pyspark() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/golden/gaussian_mixture.json")).unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let seed = case["seed"].as_i64().unwrap();
        let bytes = STANDARD
            .decode(case["params_b64"].as_str().unwrap())
            .unwrap();
        let expected = spark_connect_proto::MlParams::decode(bytes.as_slice()).unwrap();
        let estimator = GaussianMixture::new()
            .set_features_col("input")
            .set_prediction_col("cluster")
            .set_k(3)
            .set_max_iter(50)
            .set_seed(seed)
            .set_probability_col("probs")
            .set_tol(1e-5)
            .set_aggregation_depth(3)
            .set_weight_col("weight");
        assert_eq!(estimator.params().to_proto(), expected, "seed {seed}");
    }
    assert!(GaussianMixture::default()
        .params()
        .to_proto()
        .params
        .is_empty());
}
