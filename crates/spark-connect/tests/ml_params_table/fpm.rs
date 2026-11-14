//! The fpm classes of the completeness test in `ml_params_table.rs`.
use spark_connect::ml::*;

/// Golden classes converted in this area, with the `PARAM_NAMES` of their Rust type.
pub const CONVERTED: &[(&str, &[&str])] = &[
    ("FPGrowth", FPGrowth::PARAM_NAMES),
    ("FPGrowthModel", FPGrowthModel::PARAM_NAMES),
    ("PrefixSpan", PrefixSpan::PARAM_NAMES),
];

/// Golden classes of this area that are not converted, each with its reason.
pub const EXCLUDED: &[(&str, &str)] = &[];
