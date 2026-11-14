//! The classification classes of the completeness test in `ml_params_table.rs`, by family: each
//! family's classes move from `EXCLUDED` to `CONVERTED` within the family's own block.

use spark_connect::ml::*;

/// Golden classes converted in this area, with the `PARAM_NAMES` of their Rust type.
pub const CONVERTED: &[(&str, &[&str])] = &[
    // LogisticRegression and LinearSVC.
    ("LinearSVC", LinearSVC::PARAM_NAMES),
    ("LinearSVCModel", LinearSVCModel::PARAM_NAMES),
    ("LogisticRegression", LogisticRegression::PARAM_NAMES),
    (
        "LogisticRegressionModel",
        LogisticRegressionModel::PARAM_NAMES,
    ),
    // NaiveBayes, MultilayerPerceptron, FM and OneVsRest.
    ("FMClassificationModel", FMClassificationModel::PARAM_NAMES),
    ("FMClassifier", FMClassifier::PARAM_NAMES),
    (
        "MultilayerPerceptronClassificationModel",
        MultilayerPerceptronClassificationModel::PARAM_NAMES,
    ),
    (
        "MultilayerPerceptronClassifier",
        MultilayerPerceptronClassifier::PARAM_NAMES,
    ),
    ("NaiveBayes", NaiveBayes::PARAM_NAMES),
    ("NaiveBayesModel", NaiveBayesModel::PARAM_NAMES),
    ("OneVsRest", OneVsRest::PARAM_NAMES),
    ("OneVsRestModel", OneVsRestModel::PARAM_NAMES),
    // The decision tree, random forest and gradient-boosted tree classifiers.
    (
        "DecisionTreeClassificationModel",
        DecisionTreeClassificationModel::PARAM_NAMES,
    ),
    (
        "DecisionTreeClassifier",
        DecisionTreeClassifier::PARAM_NAMES,
    ),
    (
        "GBTClassificationModel",
        GBTClassificationModel::PARAM_NAMES,
    ),
    ("GBTClassifier", GBTClassifier::PARAM_NAMES),
    (
        "RandomForestClassificationModel",
        RandomForestClassificationModel::PARAM_NAMES,
    ),
    (
        "RandomForestClassifier",
        RandomForestClassifier::PARAM_NAMES,
    ),
];

/// Golden classes of this area that are not converted, each with its reason.
pub const EXCLUDED: &[(&str, &str)] = &[
    // NaiveBayes, MultilayerPerceptron, FM and OneVsRest.
    // The decision tree, random forest and gradient-boosted tree classifiers.
];
