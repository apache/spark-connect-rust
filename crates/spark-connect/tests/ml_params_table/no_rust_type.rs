//! The no and rust and type classes of the completeness test in `ml_params_table.rs`.

/// Golden classes converted in this area, with the `PARAM_NAMES` of their Rust type.
pub const CONVERTED: &[(&str, &[&str])] = &[];

/// Golden classes of this area that are not converted, each with its reason.
pub const EXCLUDED: &[(&str, &str)] = &[
    (
        "BinaryLogisticRegressionSummary",
        "no Rust type: summary, no params",
    ),
    (
        "BinaryLogisticRegressionTrainingSummary",
        "no Rust type: summary, no params",
    ),
    (
        "BinaryRandomForestClassificationSummary",
        "no Rust type: summary, no params",
    ),
    (
        "BinaryRandomForestClassificationTrainingSummary",
        "no Rust type: summary, no params",
    ),
    ("BisectingKMeansSummary", "no Rust type: summary, no params"),
    (
        "Evaluator",
        "no Rust type: abstract base of the evaluators, no params",
    ),
    (
        "FMClassificationSummary",
        "no Rust type: summary, no params",
    ),
    (
        "FMClassificationTrainingSummary",
        "no Rust type: summary, no params",
    ),
    ("GaussianMixtureSummary", "no Rust type: summary, no params"),
    (
        "GeneralizedLinearRegressionSummary",
        "no Rust type: summary, no params",
    ),
    (
        "GeneralizedLinearRegressionTrainingSummary",
        "no Rust type: summary, no params",
    ),
    ("KMeansSummary", "no Rust type: summary, no params"),
    (
        "LinearRegressionSummary",
        "no Rust type: summary, no params",
    ),
    (
        "LinearRegressionTrainingSummary",
        "no Rust type: summary, no params",
    ),
    ("LinearSVCSummary", "no Rust type: summary, no params"),
    (
        "LinearSVCTrainingSummary",
        "no Rust type: summary, no params",
    ),
    (
        "LogisticRegressionSummary",
        "no Rust type: summary, no params",
    ),
    (
        "LogisticRegressionTrainingSummary",
        "no Rust type: summary, no params",
    ),
    (
        "MultilayerPerceptronClassificationSummary",
        "no Rust type: summary, no params",
    ),
    (
        "MultilayerPerceptronClassificationTrainingSummary",
        "no Rust type: summary, no params",
    ),
    (
        "ParamGridBuilder",
        "a builder of param maps (ml::ParamGridBuilder), not a Params class",
    ),
    (
        "RandomForestClassificationSummary",
        "no Rust type: summary, no params",
    ),
    (
        "RandomForestClassificationTrainingSummary",
        "no Rust type: summary, no params",
    ),
];
