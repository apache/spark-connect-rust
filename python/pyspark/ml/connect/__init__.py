"""
pyspark.ml.connect - Spark-Connect ML API, backed by the Rust _pyspark extension.

Re-exports the estimator/transformer/evaluator/pipeline classes so that both
`from pyspark.ml.connect import StandardScaler` and the submodule paths
(`pyspark.ml.connect.feature`, `.classification`, `.evaluation`) work.
"""
from pyspark.ml.connect.feature import (
    StandardScaler,
    MaxAbsScaler,
    StringIndexer,
    VectorAssembler,
)
from pyspark.ml.connect.classification import LogisticRegression
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
    "StringIndexer",
    "VectorAssembler",
    "LogisticRegression",
    "RegressionEvaluator",
    "BinaryClassificationEvaluator",
    "MulticlassClassificationEvaluator",
    "CrossValidator",
    "CrossValidatorModel",
    "MLModel",
]
