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
    GeneralizedLinearRegression,
    LinearRegression,
    RandomForestRegressor,
)
from pyspark.ml.connect.clustering import BisectingKMeans, GaussianMixture, KMeans
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

# feature.py is the byte-identical upstream module, so the native StringIndexer and VectorAssembler
# of the drop-in are set on it here, where `from pyspark.ml.connect.feature import ...` finds them.
from pyspark._pyspark import StringIndexer, VectorAssembler

feature.StringIndexer = StringIndexer
feature.VectorAssembler = VectorAssembler

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
    "GeneralizedLinearRegression",
    "LinearRegression",
    "DecisionTreeRegressor",
    "RandomForestRegressor",
    "KMeans",
    "GaussianMixture",
    "BisectingKMeans",
    "RegressionEvaluator",
    "BinaryClassificationEvaluator",
    "MulticlassClassificationEvaluator",
    "CrossValidator",
    "CrossValidatorModel",
    "MLModel",
]
