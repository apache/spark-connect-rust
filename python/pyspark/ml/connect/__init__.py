"""
pyspark.ml.connect - Spark-Connect ML API, backed by the Rust _pyspark extension.

Re-exports the estimator/transformer/evaluator/pipeline classes so that both
`from pyspark.ml.connect import StandardScaler` and the submodule paths
(`pyspark.ml.connect.feature`, `.classification`, `.evaluation`) work.
"""
from pyspark.ml.connect.feature import (
    ArrayAssembler,
    StandardScaler,
    MaxAbsScaler,
)
from pyspark.ml.connect.classification import (
    GBTClassifier,
    DecisionTreeClassifier,
    FMClassifier,
    MultilayerPerceptronClassifier,
    OneVsRest,
    LinearSVC,
    LogisticRegression,
    NaiveBayes,
    RandomForestClassifier,
)
from pyspark.ml.connect.regression import (
    GBTRegressor,
    DecisionTreeRegressor,
    AFTSurvivalRegression,
    IsotonicRegression,
    FMRegressor,
    GeneralizedLinearRegression,
    LinearRegression,
    RandomForestRegressor,
)
from pyspark.ml.connect.clustering import (
    LDA, BisectingKMeans, GaussianMixture, KMeans, PowerIterationClustering,
)
from pyspark.ml.connect.evaluation import (
    RegressionEvaluator,
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.connect.pipeline import Pipeline, PipelineModel
from pyspark.ml.connect.tuning import CrossValidator, CrossValidatorModel
from pyspark._pyspark import MLModel

# Base ABCs (vendored from upstream base.py) and the submodules, matching the reference
# `pyspark.ml.connect.__all__`.
from pyspark.ml.connect.base import Estimator, Transformer, Evaluator, Model
from pyspark.ml.connect import feature, evaluation, tuning

# Keep the vendored feature module intact; install the native standard-ML stages here.
from pyspark import _pyspark as _native

for _name in _native._ml_feature_names:
    globals()[_name] = getattr(_native, _name)
    setattr(feature, _name, globals()[_name])

__all__ = [
    # Reference `pyspark.ml.connect` public surface.
    "Estimator",
    "Transformer",
    "Evaluator",
    "Model",
    "feature",
    "evaluation",
    "Pipeline",
    "PipelineModel",
    "tuning",
    # Fork-additional concrete estimators/evaluators (Rust-backed).
    "StandardScaler",
    "MaxAbsScaler",
    "ArrayAssembler",
    "StringIndexer",
    "VectorAssembler",
    "LogisticRegression",
    "NaiveBayes",
    "FMClassifier",
    "MultilayerPerceptronClassifier",
    "OneVsRest",
    "LinearSVC",
    "DecisionTreeClassifier",
    "GBTClassifier",
    "GBTRegressor",
    "RandomForestClassifier",
    "AFTSurvivalRegression",
    "IsotonicRegression",
    "FMRegressor",
    "GeneralizedLinearRegression",
    "LinearRegression",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "KMeans",
    "LDA",
    "PowerIterationClustering",
    "GaussianMixture",
    "BisectingKMeans",
    "RegressionEvaluator",
    "BinaryClassificationEvaluator",
    "MulticlassClassificationEvaluator",
    "CrossValidator",
    "CrossValidatorModel",
    "MLModel",
]

__all__ += [name for name in _native._ml_feature_names if name not in __all__]
