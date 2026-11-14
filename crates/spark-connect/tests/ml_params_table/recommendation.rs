//! The recommendation classes of the completeness test in `ml_params_table.rs`.
use spark_connect::ml::*;

/// Golden classes converted in this area, with the `PARAM_NAMES` of their Rust type.
pub const CONVERTED: &[(&str, &[&str])] = &[
    ("ALS", ALS::PARAM_NAMES),
    ("ALSModel", ALSModel::PARAM_NAMES),
];

/// Golden classes of this area that are not converted, each with its reason.
pub const EXCLUDED: &[(&str, &str)] = &[];
