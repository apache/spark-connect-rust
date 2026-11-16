"""Native regression estimators backed by Spark Connect's JVM ML API."""

from pyspark._pyspark import (
    AFTSurvivalRegression,
    AFTSurvivalRegressionModel,
    DecisionTreeRegressionModel,
    DecisionTreeRegressor,
    FMRegressionModel,
    FMRegressor,
    GBTRegressionModel,
    GBTRegressor,
    GeneralizedLinearRegression,
    GeneralizedLinearRegressionModel,
    GeneralizedLinearRegressionSummary,
    GeneralizedLinearRegressionTrainingSummary,
    IsotonicRegression,
    IsotonicRegressionModel,
    LinearRegression,
    LinearRegressionModel,
    LinearRegressionSummary,
    LinearRegressionTrainingSummary,
    RandomForestRegressionModel,
    RandomForestRegressor,
)

# PySpark's own abstract base classes of the native classes, so that the module has the
# names of pyspark.ml.regression.
from pyspark.ml.regression import (
    RegressionModel,  # noqa: F401
    Regressor,  # noqa: F401
)

__all__ = [
    "AFTSurvivalRegression",
    "AFTSurvivalRegressionModel",
    "DecisionTreeRegressor",
    "DecisionTreeRegressionModel",
    "GBTRegressor",
    "GBTRegressionModel",
    "GeneralizedLinearRegression",
    "GeneralizedLinearRegressionModel",
    "GeneralizedLinearRegressionSummary",
    "GeneralizedLinearRegressionTrainingSummary",
    "IsotonicRegression",
    "IsotonicRegressionModel",
    "LinearRegression",
    "LinearRegressionModel",
    "LinearRegressionSummary",
    "LinearRegressionTrainingSummary",
    "RandomForestRegressor",
    "RandomForestRegressionModel",
    "FMRegressor",
    "FMRegressionModel",
]

# The public classes of the native classes (see pyspark.ml._native_classes).
from pyspark.ml._native_classes import publish as _publish  # noqa: E402

_publish(globals(), "pyspark.ml.regression")
