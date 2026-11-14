//! Parameter encodings independently captured with official PySpark 4.3.0.
use base64::{engine::general_purpose::STANDARD, Engine};
use prost::Message;
use spark_connect::ml::{
    Estimator, FMClassifier, MlVector, MultilayerPerceptronClassifier, OneVsRest,
};

#[test]
fn remaining_classifier_parameters_match_official_pyspark() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../tests/golden/remaining_classifiers.json"
    ))
    .unwrap();
    let estimators: Vec<Box<dyn Estimator>> = vec![
        Box::new(FMClassifier::new()),
        Box::new(
            FMClassifier::new()
                .set_features_col("input")
                .set_label_col("target")
                .set_prediction_col("pred")
                .set_probability_col("prob")
                .set_raw_prediction_col("scores")
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
                .set_thresholds([1.0, 2.0])
                .set_seed(8589934592),
        ),
        Box::new(MultilayerPerceptronClassifier::new()),
        Box::new(
            MultilayerPerceptronClassifier::new()
                .set_features_col("input")
                .set_label_col("target")
                .set_prediction_col("pred")
                .set_max_iter(50)
                .set_tol(1e-08)
                .set_seed(8589934592)
                .set_layers([2, 3, 2])
                .set_block_size(4)
                .set_step_size(0.1)
                .set_solver("gd")
                .set_initial_weights(MlVector::Dense(vec![
                    0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12,
                    0.13, 0.14, 0.15, 0.16,
                ]))
                .set_probability_col("prob")
                .set_raw_prediction_col("scores")
                .set_thresholds([1.0, 2.0]),
        ),
        Box::new(
            MultilayerPerceptronClassifier::new()
                .set_features_col("input")
                .set_label_col("target")
                .set_prediction_col("pred")
                .set_max_iter(50)
                .set_tol(1e-08)
                .set_seed(8589934592)
                .set_layers([2, 3, 2])
                .set_block_size(4)
                .set_step_size(0.1)
                .set_solver("gd")
                .set_initial_weights(MlVector::Sparse {
                    size: 17,
                    indices: vec![0, 16],
                    values: vec![0.1, -0.2],
                })
                .set_probability_col("prob")
                .set_raw_prediction_col("scores")
                .set_thresholds([1.0, 2.0]),
        ),
        Box::new(OneVsRest::new()),
        Box::new(
            OneVsRest::new()
                .set_features_col("input")
                .set_label_col("target")
                .set_prediction_col("pred")
                .set_raw_prediction_col("scores")
                .set_weight_col("weight")
                .set_parallelism(3),
        ),
    ];
    for (estimator, case) in estimators
        .into_iter()
        .zip(fixture["cases"].as_array().unwrap())
    {
        let bytes = STANDARD
            .decode(case["params_b64"].as_str().unwrap())
            .unwrap();
        let expected = spark_connect_proto::MlParams::decode(bytes.as_slice()).unwrap();
        assert_eq!(estimator.params().to_proto(), expected, "{}", case["class"]);
    }
}
