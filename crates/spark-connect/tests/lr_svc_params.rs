// SPDX-License-Identifier: Apache-2.0
//! The param accessors of LogisticRegression and LinearSVC, against PySpark 4.3.0 run over Spark
//! Connect: the defaults, and the threshold and thresholds of `_LogisticRegressionParams`. The
//! expected values and messages are the reference's, from the official pyspark-client 4.3.0 in
//! Connect mode. No Spark server is needed here.

use spark_connect::ml::{Estimator, LinearSVC, LogisticRegression, Params};

fn error_of<T: std::fmt::Debug>(result: spark_connect_core::error::Result<T>) -> String {
    result.expect_err("an error").to_string()
}

#[test]
fn logistic_regression_defaults_are_pyspark_defaults() {
    // Reference: getThreshold 0.5; getThresholds KeyError; neither param set.
    let lr = LogisticRegression::new();
    assert_eq!(lr.max_iter(), 100);
    assert_eq!(lr.reg_param(), 0.0);
    assert_eq!(lr.tol(), 1e-6);
    assert_eq!(lr.family(), "auto");
    assert_eq!(lr.threshold().unwrap(), 0.5);
    let error = error_of(lr.thresholds());
    assert!(
        error.starts_with("Param(parent='LogisticRegression_")
            && error.contains("name='thresholds', doc=\"Thresholds in multi-class"),
        "{error}"
    );
    assert_eq!(lr.features_col(), "features");
    assert_eq!(lr.probability_col(), "probability");
    assert_eq!(lr.weight_col(), None);
}

#[test]
fn set_threshold_gives_the_thresholds_of_the_two_classes() {
    // Reference: setThreshold(0.3) -> getThreshold 0.3, getThresholds [0.7, 0.3].
    let lr = LogisticRegression::new().set_threshold(0.3);
    assert_eq!(lr.threshold().unwrap(), 0.3);
    assert_eq!(lr.thresholds().unwrap(), vec![1.0 - 0.3, 0.3]);
}

#[test]
fn set_thresholds_gives_the_binary_equivalent_threshold() {
    // Reference: setThresholds([0.4, 0.6]) -> getThreshold 0.6 (1 / (1 + 0.4 / 0.6)).
    let lr = LogisticRegression::new().set_thresholds([0.4, 0.6]);
    assert!((lr.threshold().unwrap() - 0.6).abs() < 1e-12);
    assert_eq!(lr.thresholds().unwrap(), vec![0.4, 0.6]);
}

#[test]
fn setters_do_not_clear_each_other_over_connect() {
    // Reference: over Spark Connect the `clear` of pyspark.ml does nothing, so the threshold
    // stays set, and the thresholds that disagree with it raise the consistency error.
    let lr = LogisticRegression::new()
        .set_thresholds([0.3, 0.7])
        .set_threshold(0.5);
    let error = error_of(lr.threshold());
    assert_eq!(
        error,
        "Logistic Regression getThreshold found inconsistent values for threshold (0.5) and \
         thresholds (equivalent to 0.7)"
    );
    assert_eq!(error_of(lr.thresholds()), error);
}

#[test]
fn thresholds_of_more_than_two_classes_are_not_a_binary_threshold() {
    // Reference: setThresholds of 3 values is accepted; getThreshold raises with two spaces
    // before `thresholds:`; getThresholds returns the three values.
    let lr = LogisticRegression::new().set_thresholds([0.4, 0.5, 0.1]);
    assert_eq!(
        error_of(lr.threshold()),
        "Logistic Regression getThreshold only applies to binary classification, but \
         thresholds has length != 2.  thresholds: [0.4, 0.5, 0.1]"
    );
    assert_eq!(lr.thresholds().unwrap(), vec![0.4, 0.5, 0.1]);
}

#[test]
fn constructor_params_are_checked_for_agreement() {
    // Reference: threshold=0.7 with thresholds=[0.3, 0.7] agrees, so getThreshold is 0.7.
    let mut lr = LogisticRegression::new();
    *Estimator::params_mut(&mut lr) = Params::new()
        .set_param_double("threshold", 0.7)
        .set_param_double_array("thresholds", [0.3, 0.7]);
    assert_eq!(lr.threshold().unwrap(), 0.7);

    // Reference: threshold=0.5 with thresholds=[0.3, 0.7] raises from getThreshold and from
    // getThresholds, with the equivalent of the thresholds formatted as `%g`.
    let mut lr = LogisticRegression::new();
    *Estimator::params_mut(&mut lr) = Params::new()
        .set_param_double("threshold", 0.5)
        .set_param_double_array("thresholds", [0.3, 0.7]);
    let message = "Logistic Regression getThreshold found inconsistent values for threshold (0.5) \
                   and thresholds (equivalent to 0.7)";
    assert_eq!(error_of(lr.threshold()), message);
    assert_eq!(error_of(lr.thresholds()), message);

    // Reference: threshold=0.5 with thresholds=[0.4, 0.5, 0.1] raises the length error with one
    // space before `thresholds:`, since the consistency check runs first.
    let mut lr = LogisticRegression::new();
    *Estimator::params_mut(&mut lr) = Params::new()
        .set_param_double("threshold", 0.5)
        .set_param_double_array("thresholds", [0.4, 0.5, 0.1]);
    assert_eq!(
        error_of(lr.threshold()),
        "Logistic Regression getThreshold only applies to binary classification, but \
         thresholds has length != 2. thresholds: [0.4, 0.5, 0.1]"
    );
}

#[test]
fn linear_svc_defaults_and_plain_threshold_are_pyspark_defaults() {
    let svc = LinearSVC::new();
    assert_eq!(svc.max_iter(), 100);
    assert_eq!(svc.threshold(), 0.0);
    assert_eq!(svc.features_col(), "features");
    assert_eq!(svc.weight_col(), None);
    let svc = svc.set_threshold(0.25);
    assert_eq!(svc.threshold(), 0.25);
}

#[test]
#[allow(deprecated)]
fn logistic_regression_keeps_the_feature_col_names_of_v4_2_0() {
    // `feature_col` and `set_feature_col`, the names of v4.2.0, are aliases of PySpark's names.
    assert_eq!(LogisticRegression::new().feature_col(), "features");
    let lr = LogisticRegression::new().set_feature_col("f");
    assert_eq!(lr.features_col(), "f");
    assert_eq!(lr.feature_col(), "f");
    assert_eq!(
        lr.set_features_col("g").feature_col(),
        "g",
        "the alias reads the param that PySpark's name sets"
    );
}
