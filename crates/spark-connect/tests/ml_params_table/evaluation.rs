//! The evaluation classes of the completeness test in `ml_params_table.rs`.
use spark_connect::ml::*;

/// Golden classes converted in this area, with the `PARAM_NAMES` of their Rust type.
pub const CONVERTED: &[(&str, &[&str])] = &[
    (
        "BinaryClassificationEvaluator",
        BinaryClassificationEvaluator::PARAM_NAMES,
    ),
    ("ClusteringEvaluator", ClusteringEvaluator::PARAM_NAMES),
    (
        "MulticlassClassificationEvaluator",
        MulticlassClassificationEvaluator::PARAM_NAMES,
    ),
    (
        "MultilabelClassificationEvaluator",
        MultilabelClassificationEvaluator::PARAM_NAMES,
    ),
    ("RankingEvaluator", RankingEvaluator::PARAM_NAMES),
    ("RegressionEvaluator", RegressionEvaluator::PARAM_NAMES),
];

/// Golden classes of this area that are not converted, each with its reason.
pub const EXCLUDED: &[(&str, &str)] = &[];
