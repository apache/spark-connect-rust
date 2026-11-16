"""Classification estimators (pyspark.ml.connect.classification)."""

from pyspark._pyspark import (
    BinaryLogisticRegressionSummary,
    BinaryLogisticRegressionTrainingSummary,
    BinaryRandomForestClassificationSummary,
    BinaryRandomForestClassificationTrainingSummary,
    DecisionTreeClassificationModel,
    DecisionTreeClassifier,
    FMClassificationModel,
    FMClassificationSummary,
    FMClassificationTrainingSummary,
    FMClassifier,
    GBTClassificationModel,
    GBTClassifier,
    LinearSVC,
    LinearSVCModel,
    LinearSVCSummary,
    LinearSVCTrainingSummary,
    LogisticRegression,
    LogisticRegressionModel,
    LogisticRegressionSummary,
    LogisticRegressionTrainingSummary,
    MultilayerPerceptronClassificationModel,
    MultilayerPerceptronClassificationSummary,
    MultilayerPerceptronClassificationTrainingSummary,
    MultilayerPerceptronClassifier,
    NaiveBayes,
    NaiveBayesModel,
    OneVsRest,
    OneVsRestModel,
    RandomForestClassificationModel,
    RandomForestClassificationSummary,
    RandomForestClassificationTrainingSummary,
    RandomForestClassifier,
)

__all__ = [
    "FMClassificationModel",
    "FMClassificationSummary",
    "FMClassificationTrainingSummary",
    "FMClassifier",
    "MultilayerPerceptronClassificationModel",
    "MultilayerPerceptronClassificationSummary",
    "MultilayerPerceptronClassificationTrainingSummary",
    "MultilayerPerceptronClassifier",
    "OneVsRest",
    "OneVsRestModel",
    "NaiveBayes",
    "NaiveBayesModel",
    "GBTClassifier",
    "GBTClassificationModel",
    "DecisionTreeClassifier",
    "DecisionTreeClassificationModel",
    "LinearSVC",
    "LinearSVCModel",
    "LinearSVCSummary",
    "LinearSVCTrainingSummary",
    "LogisticRegression",
    "LogisticRegressionModel",
    "LogisticRegressionSummary",
    "LogisticRegressionTrainingSummary",
    "BinaryLogisticRegressionSummary",
    "BinaryLogisticRegressionTrainingSummary",
    "RandomForestClassifier",
    "RandomForestClassificationModel",
    "RandomForestClassificationSummary",
    "RandomForestClassificationTrainingSummary",
    "BinaryRandomForestClassificationSummary",
    "BinaryRandomForestClassificationTrainingSummary",
]

# The public classes of the native classes (see pyspark.ml._native_classes).
from pyspark.ml._native_classes import publish as _publish  # noqa: E402

_publish(globals(), "pyspark.ml.classification")

# PySpark's own readers and writers of the Python format of the module, which its `__all__` does
# not name: over the param maps of the public classes, they save and load their objects too.
import pyspark.ml.classification as _pysparks  # noqa: E402

OneVsRestReader = _pysparks.OneVsRestReader
OneVsRestWriter = _pysparks.OneVsRestWriter
OneVsRestModelReader = _pysparks.OneVsRestModelReader
OneVsRestModelWriter = _pysparks.OneVsRestModelWriter
del _pysparks
