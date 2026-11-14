//! Compare parameter encoding against official PySpark 4.3.0.
use base64::{engine::general_purpose::STANDARD, Engine};
use prost::Message;
use spark_connect::ml::{Estimator, GeneralizedLinearRegression};

#[test]
fn generalized_linear_regression_parameters_match_official_pyspark() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/golden/generalized_linear_regression.json"
    ))
    .unwrap();
    let configured = GeneralizedLinearRegression::new()
        .set_label_col("target")
        .set_features_col("input")
        .set_prediction_col("pred")
        .set_family("tweedie")
        .set_link("log")
        .set_fit_intercept(false)
        .set_max_iter(50)
        .set_tol(1e-08)
        .set_reg_param(0.25)
        .set_weight_col("weight")
        .set_solver("irls")
        .set_link_prediction_col("eta")
        .set_variance_power(1.5)
        .set_link_power(0.0)
        .set_offset_col("offset")
        .set_aggregation_depth(3);
    for (estimator, case) in [GeneralizedLinearRegression::default(), configured]
        .into_iter()
        .zip(fixture["cases"].as_array().unwrap())
    {
        let bytes = STANDARD
            .decode(case["params_b64"].as_str().unwrap())
            .unwrap();
        let expected = spark_connect_proto::MlParams::decode(bytes.as_slice()).unwrap();
        assert_eq!(estimator.params().to_proto(), expected);
    }
}
