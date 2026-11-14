//! Parameter bytes captured from official PySpark, including integer widths and double arrays.
use base64::engine::general_purpose::STANDARD;
use base64::Engine;
use prost::Message;
use spark_connect::ml::{AFTSurvivalRegression, Estimator, FMRegressor, IsotonicRegression};

#[test]
fn remaining_regressors_match_official_parameters() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/golden/remaining_regressors.json"
    ))
    .unwrap();
    let estimators: Vec<Box<dyn Estimator>> = vec![
        Box::new(AFTSurvivalRegression::new()),
        Box::new(
            AFTSurvivalRegression::new()
                .set_features_col("input")
                .set_label_col("duration")
                .set_prediction_col("pred")
                .set_fit_intercept(false)
                .set_max_iter(50)
                .set_tol(1e-08)
                .set_censor_col("observed")
                .set_quantile_probabilities([0.2, 0.5, 0.8])
                .set_quantiles_col("quantiles")
                .set_aggregation_depth(3)
                .set_max_block_size_in_mb(1.0),
        ),
        Box::new(IsotonicRegression::new()),
        Box::new(
            IsotonicRegression::new()
                .set_features_col("input")
                .set_label_col("target")
                .set_prediction_col("pred")
                .set_weight_col("weight")
                .set_isotonic(false)
                .set_feature_index(1),
        ),
        Box::new(FMRegressor::new()),
        Box::new(
            FMRegressor::new()
                .set_features_col("input")
                .set_label_col("target")
                .set_prediction_col("pred")
                .set_factor_size(2)
                .set_fit_intercept(false)
                .set_fit_linear(false)
                .set_reg_param(0.1)
                .set_mini_batch_fraction(0.5)
                .set_init_std(0.2)
                .set_max_iter(50)
                .set_step_size(0.3)
                .set_tol(1e-08)
                .set_solver("gd")
                .set_seed(8589934592),
        ),
    ];
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(estimators.len(), cases.len());
    for (estimator, case) in estimators.into_iter().zip(cases) {
        let bytes = STANDARD
            .decode(case["params_b64"].as_str().unwrap())
            .unwrap();
        let expected = spark_connect_proto::MlParams::decode(bytes.as_slice()).unwrap();
        assert_eq!(estimator.params().to_proto(), expected, "{}", case["class"]);
        assert_eq!(
            estimator.operator().name,
            format!(
                "org.apache.spark.ml.regression.{}",
                case["class"].as_str().unwrap()
            )
        );
    }
}
