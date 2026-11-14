//! Match all native LinearRegression parameter encodings to official PySpark 4.3.0.
use base64::{engine::general_purpose::STANDARD, Engine};
use prost::Message;
use spark_connect::ml::{Estimator, LinearRegression};

#[test]
fn linear_regression_parameters_match_official_pyspark() {
    let fixture: serde_json::Value =
        serde_json::from_str(include_str!("../../../tests/golden/linear_regression.json")).unwrap();
    let bytes = STANDARD
        .decode(fixture["params_b64"].as_str().unwrap())
        .unwrap();
    let expected = spark_connect_proto::MlParams::decode(bytes.as_slice()).unwrap();
    let estimator = LinearRegression::new()
        .set_features_col("input")
        .set_label_col("target")
        .set_prediction_col("pred")
        .set_max_iter(50)
        .set_reg_param(0.25)
        .set_elastic_net_param(0.5)
        .set_tol(1e-5)
        .set_fit_intercept(false)
        .set_standardization(false)
        .set_solver("l-bfgs")
        .set_weight_col("weight")
        .set_aggregation_depth(3)
        .set_loss("squaredError")
        .set_epsilon(1.5)
        .set_max_block_size_in_mb(2.0);
    assert_eq!(estimator.params().to_proto(), expected);
    assert!(LinearRegression::default()
        .params()
        .to_proto()
        .params
        .is_empty());
}
