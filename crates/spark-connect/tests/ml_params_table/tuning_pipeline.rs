//! The tuning and pipeline classes of the completeness test in `ml_params_table.rs`.

/// Golden classes converted in this area, with the `PARAM_NAMES` of their Rust type.
pub const CONVERTED: &[(&str, &[&str])] = &[];

/// Golden classes of this area that are not converted, each with its reason.
pub const EXCLUDED: &[(&str, &str)] = &[
    ("CrossValidator", "pending: tuning (Base D)"),
    ("CrossValidatorModel", "pending: tuning (Base D)"),
    ("Pipeline", "pending: pipeline (Base D)"),
    ("TrainValidationSplit", "pending: tuning (Base D)"),
    ("TrainValidationSplitModel", "pending: tuning (Base D)"),
];
