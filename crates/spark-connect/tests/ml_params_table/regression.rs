//! The regression classes of the completeness test in `ml_params_table.rs`.

/// Golden classes converted in this area, with the `PARAM_NAMES` of their Rust type.
pub const CONVERTED: &[(&str, &[&str])] = &[];

/// Golden classes of this area that are not converted, each with its reason.
pub const EXCLUDED: &[(&str, &str)] = &[
    ("AFTSurvivalRegression", "pending: regression"),
    ("AFTSurvivalRegressionModel", "pending: regression"),
    ("DecisionTreeRegressionModel", "pending: regression"),
    ("DecisionTreeRegressor", "pending: regression"),
    ("FMRegressionModel", "pending: regression"),
    ("FMRegressor", "pending: regression"),
    ("GBTRegressionModel", "pending: regression"),
    ("GBTRegressor", "pending: regression"),
    ("GeneralizedLinearRegression", "pending: regression"),
    ("GeneralizedLinearRegressionModel", "pending: regression"),
    ("IsotonicRegression", "pending: regression"),
    ("IsotonicRegressionModel", "pending: regression"),
    ("LinearRegression", "pending: regression"),
    ("LinearRegressionModel", "pending: regression"),
    ("RandomForestRegressionModel", "pending: regression"),
    ("RandomForestRegressor", "pending: regression"),
];
