# SPDX-License-Identifier: Apache-2.0
"""Standard regression APIs, with selected estimators delegated to Rust."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyspark.ml._upstream.regression import *  # noqa: F403
    from pyspark.ml._upstream.regression import RegressionModel as RegressionModel
    from pyspark.ml._upstream.regression import Regressor as Regressor


def _load_regression():
    import sys
    from importlib.machinery import SourceFileLoader
    from pathlib import Path
    from types import SimpleNamespace

    from pyspark import _pyspark
    from pyspark.ml._native_classes import publish

    # Keep upstream globals, annotations, and subclass hooks under the public module name.
    module = sys.modules[__name__]
    source = Path(__file__).with_name("_upstream") / "regression.py"
    SourceFileLoader(__name__, str(source)).exec_module(module)
    reference = SimpleNamespace(**vars(module))
    names = (
        "FMRegressor",
        "FMRegressionModel",
        "AFTSurvivalRegression",
        "AFTSurvivalRegressionModel",
        "GeneralizedLinearRegression",
        "GeneralizedLinearRegressionModel",
        "GeneralizedLinearRegressionSummary",
        "GeneralizedLinearRegressionTrainingSummary",
        "LinearRegression",
        "LinearRegressionModel",
        "LinearRegressionSummary",
        "LinearRegressionTrainingSummary",
    )
    native = {name: getattr(_pyspark, name) for name in names}
    publish(
        native, reference, public_module=__name__,
        eager_params=(
            "FMRegressor", "FMRegressionModel",
            "AFTSurvivalRegression", "AFTSurvivalRegressionModel",
            "LinearRegression", "LinearRegressionModel",
            "GeneralizedLinearRegression", "GeneralizedLinearRegressionModel",
        ),
    )
    vars(module).update(native)


_load_regression()
del _load_regression
