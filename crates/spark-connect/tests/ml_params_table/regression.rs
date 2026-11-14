//! The regression classes of the completeness test in `ml_params_table.rs`.

use spark_connect::ml::*;

/// Golden classes converted in this area, with the `PARAM_NAMES` of their Rust type.
pub const CONVERTED: &[(&str, &[&str])] = &[
    ("AFTSurvivalRegression", AFTSurvivalRegression::PARAM_NAMES),
    (
        "AFTSurvivalRegressionModel",
        AFTSurvivalRegressionModel::PARAM_NAMES,
    ),
    (
        "DecisionTreeRegressionModel",
        DecisionTreeRegressionModel::PARAM_NAMES,
    ),
    ("DecisionTreeRegressor", DecisionTreeRegressor::PARAM_NAMES),
    ("FMRegressionModel", FMRegressionModel::PARAM_NAMES),
    ("FMRegressor", FMRegressor::PARAM_NAMES),
    ("GBTRegressionModel", GBTRegressionModel::PARAM_NAMES),
    ("GBTRegressor", GBTRegressor::PARAM_NAMES),
    (
        "GeneralizedLinearRegression",
        GeneralizedLinearRegression::PARAM_NAMES,
    ),
    (
        "GeneralizedLinearRegressionModel",
        GeneralizedLinearRegressionModel::PARAM_NAMES,
    ),
    ("IsotonicRegression", IsotonicRegression::PARAM_NAMES),
    (
        "IsotonicRegressionModel",
        IsotonicRegressionModel::PARAM_NAMES,
    ),
    ("LinearRegression", LinearRegression::PARAM_NAMES),
    ("LinearRegressionModel", LinearRegressionModel::PARAM_NAMES),
    (
        "RandomForestRegressionModel",
        RandomForestRegressionModel::PARAM_NAMES,
    ),
    ("RandomForestRegressor", RandomForestRegressor::PARAM_NAMES),
];

/// Golden classes of this area that are not converted, each with its reason.
pub const EXCLUDED: &[(&str, &str)] = &[];
