//! The tuning and pipeline classes of the completeness test in `ml_params_table.rs`.

use spark_connect::ml::{
    CrossValidator, CrossValidatorModel, Pipeline, PipelineModel, TrainValidationSplit,
    TrainValidationSplitModel,
};

/// Golden classes converted in this area, with the `PARAM_NAMES` of their Rust type.
pub const CONVERTED: &[(&str, &[&str])] = &[
    ("CrossValidator", CrossValidator::PARAM_NAMES),
    ("CrossValidatorModel", CrossValidatorModel::PARAM_NAMES),
    ("Pipeline", Pipeline::PARAM_NAMES),
    ("PipelineModel", PipelineModel::PARAM_NAMES),
    ("TrainValidationSplit", TrainValidationSplit::PARAM_NAMES),
    (
        "TrainValidationSplitModel",
        TrainValidationSplitModel::PARAM_NAMES,
    ),
];

/// Golden classes of this area that are not converted, each with its reason.
pub const EXCLUDED: &[(&str, &str)] = &[];
