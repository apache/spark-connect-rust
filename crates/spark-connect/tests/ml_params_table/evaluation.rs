//! The evaluation classes of the completeness test in `ml_params_table.rs`.

/// Golden classes converted in this area, with the `PARAM_NAMES` of their Rust type.
pub const CONVERTED: &[(&str, &[&str])] = &[];

/// Golden classes of this area that are not converted, each with its reason.
pub const EXCLUDED: &[(&str, &str)] = &[
    ("BinaryClassificationEvaluator", "pending: evaluation"),
    ("ClusteringEvaluator", "pending: evaluation"),
    ("MulticlassClassificationEvaluator", "pending: evaluation"),
    ("MultilabelClassificationEvaluator", "pending: evaluation"),
    ("RankingEvaluator", "pending: evaluation"),
    ("RegressionEvaluator", "pending: evaluation"),
];
