//! The classification classes of the completeness test in `ml_params_table.rs`.

use spark_connect::ml::{LinearSVC, LinearSVCModel, LogisticRegression, LogisticRegressionModel};

/// Golden classes converted in this area, with the `PARAM_NAMES` of their Rust type.
pub const CONVERTED: &[(&str, &[&str])] = &[
    ("LinearSVC", LinearSVC::PARAM_NAMES),
    ("LinearSVCModel", LinearSVCModel::PARAM_NAMES),
    ("LogisticRegression", LogisticRegression::PARAM_NAMES),
    (
        "LogisticRegressionModel",
        LogisticRegressionModel::PARAM_NAMES,
    ),
];

/// Golden classes of this area that are not converted, each with its reason.
pub const EXCLUDED: &[(&str, &str)] = &[
    ("DecisionTreeClassificationModel", "pending: classification"),
    ("DecisionTreeClassifier", "pending: classification"),
    ("FMClassificationModel", "pending: classification"),
    ("FMClassifier", "pending: classification"),
    ("GBTClassificationModel", "pending: classification"),
    ("GBTClassifier", "pending: classification"),
    (
        "MultilayerPerceptronClassificationModel",
        "pending: classification",
    ),
    ("MultilayerPerceptronClassifier", "pending: classification"),
    ("NaiveBayes", "pending: classification"),
    ("NaiveBayesModel", "pending: classification"),
    ("OneVsRest", "pending: classification"),
    ("OneVsRestModel", "pending: classification"),
    ("RandomForestClassificationModel", "pending: classification"),
    ("RandomForestClassifier", "pending: classification"),
];
