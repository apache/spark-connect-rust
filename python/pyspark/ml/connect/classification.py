"""Classification estimators (pyspark.ml.connect.classification)."""

from pyspark._pyspark import (
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

# PySpark's own abstract base classes of the native classes, so that the module has the
# names of pyspark.ml.classification.
from pyspark.ml.classification import (
    BinaryLogisticRegressionSummary,
    BinaryLogisticRegressionTrainingSummary,
    ClassificationModel,  # noqa: F401
    Classifier,  # noqa: F401
    LogisticRegression,
    LogisticRegressionModel,
    LogisticRegressionSummary,
    LogisticRegressionTrainingSummary,
    ProbabilisticClassificationModel,  # noqa: F401
    ProbabilisticClassifier,  # noqa: F401
)

__all__ = [
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
    "DecisionTreeClassifier",
    "DecisionTreeClassificationModel",
    "GBTClassifier",
    "GBTClassificationModel",
    "RandomForestClassifier",
    "RandomForestClassificationModel",
    "RandomForestClassificationSummary",
    "RandomForestClassificationTrainingSummary",
    "BinaryRandomForestClassificationSummary",
    "BinaryRandomForestClassificationTrainingSummary",
    "NaiveBayes",
    "NaiveBayesModel",
    "MultilayerPerceptronClassifier",
    "MultilayerPerceptronClassificationModel",
    "MultilayerPerceptronClassificationSummary",
    "MultilayerPerceptronClassificationTrainingSummary",
    "OneVsRest",
    "OneVsRestModel",
    "FMClassifier",
    "FMClassificationModel",
    "FMClassificationSummary",
    "FMClassificationTrainingSummary",
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
