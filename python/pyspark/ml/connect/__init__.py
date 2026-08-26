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
)
from pyspark.ml.connect.pipeline import Pipeline
from pyspark._pyspark import MLModel

__all__ = [
    "StandardScaler",
    "MaxAbsScaler",
    "StringIndexer",
    "VectorAssembler",
    "LogisticRegression",
    "RegressionEvaluator",
    "BinaryClassificationEvaluator",
    "Pipeline",
    "MLModel",
]
