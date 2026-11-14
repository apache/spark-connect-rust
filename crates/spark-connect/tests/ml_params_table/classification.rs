//! The classification classes of the completeness test in `ml_params_table.rs`.

use spark_connect::ml::*;

/// Golden classes converted in this area, with the `PARAM_NAMES` of their Rust type.
pub const CONVERTED: &[(&str, &[&str])] = &[
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
];

/// Golden classes of this area that are not converted, each with its reason.
pub const EXCLUDED: &[(&str, &str)] = &[
    ("DecisionTreeClassificationModel", "pending: classification"),
    ("DecisionTreeClassifier", "pending: classification"),
    ("GBTClassificationModel", "pending: classification"),
    ("GBTClassifier", "pending: classification"),
    ("LinearSVC", "pending: classification"),
    ("LinearSVCModel", "pending: classification"),
    ("LogisticRegression", "pending: classification"),
    ("LogisticRegressionModel", "pending: classification"),
    ("RandomForestClassificationModel", "pending: classification"),
    ("RandomForestClassifier", "pending: classification"),
];
