//! The classification classes of the completeness test in `ml_params_table.rs`.

/// Golden classes converted in this area, with the `PARAM_NAMES` of their Rust type.
pub const CONVERTED: &[(&str, &[&str])] = &[];

/// Golden classes of this area that are not converted, each with its reason.
pub const EXCLUDED: &[(&str, &str)] = &[
    ("DecisionTreeClassificationModel", "pending: classification"),
    ("DecisionTreeClassifier", "pending: classification"),
    ("FMClassificationModel", "pending: classification"),
    ("FMClassifier", "pending: classification"),
    ("GBTClassificationModel", "pending: classification"),
    ("GBTClassifier", "pending: classification"),
    ("LinearSVC", "pending: classification"),
    ("LinearSVCModel", "pending: classification"),
    ("LogisticRegression", "pending: classification"),
    ("LogisticRegressionModel", "pending: classification"),
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
