"""
pyspark.ml.connect - Spark-Connect ML API, backed by the Rust _pyspark extension.

Re-exports the estimator/transformer/evaluator/pipeline classes so that both
`from pyspark.ml.connect import StandardScaler` and the submodule paths
(`pyspark.ml.connect.feature`, `.classification`, `.evaluation`) work.
"""

# As PySpark's package, which imports it and runs it at its import; the drop-in's does not run it
# (see dev/parity/known-differences.md).
from pyspark.sql.connect.utils import check_dependencies  # noqa: F401

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
    LDA,
    BisectingKMeans,
    GaussianMixture,
    KMeans,
    PowerIterationClustering,
)
from pyspark.ml.connect.evaluation import (
    ClusteringEvaluator,
    MultilabelClassificationEvaluator,
    RankingEvaluator,
    RegressionEvaluator,
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
)
from pyspark.ml.connect.recommendation import ALS, ALSModel
from pyspark.ml.connect.pipeline import Pipeline, PipelineModel
from pyspark.ml.connect.tuning import (
    CrossValidator,
    CrossValidatorModel,
    TrainValidationSplit,
    TrainValidationSplitModel,
)
from pyspark.ml.connect.fpm import FPGrowth, FPGrowthModel, PrefixSpan
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

# The names of pyspark.ml.feature, in its order, as the native stages make the module that API.
# They are set here, as the vendored module must stay as Apache Spark's.
feature.__all__ = [
    "Binarizer",
    "BucketedRandomProjectionLSH",
    "BucketedRandomProjectionLSHModel",
    "Bucketizer",
    "ChiSqSelector",
    "ChiSqSelectorModel",
    "CountVectorizer",
    "CountVectorizerModel",
    "DCT",
    "ElementwiseProduct",
    "FeatureHasher",
    "HashingTF",
    "IDF",
    "IDFModel",
    "Imputer",
    "ImputerModel",
    "IndexToString",
    "Interaction",
    "MaxAbsScaler",
    "MaxAbsScalerModel",
    "MinHashLSH",
    "MinHashLSHModel",
    "MinMaxScaler",
    "MinMaxScalerModel",
    "NGram",
    "Normalizer",
    "OneHotEncoder",
    "OneHotEncoderModel",
    "PCA",
    "PCAModel",
    "PolynomialExpansion",
    "QuantileDiscretizer",
    "RobustScaler",
    "RobustScalerModel",
    "RegexTokenizer",
    "RFormula",
    "RFormulaModel",
    "SQLTransformer",
    "StandardScaler",
    "StandardScalerModel",
    "StopWordsRemover",
    "StringIndexer",
    "StringIndexerModel",
    "TargetEncoder",
    "TargetEncoderModel",
    "Tokenizer",
    "UnivariateFeatureSelector",
    "UnivariateFeatureSelectorModel",
    "VarianceThresholdSelector",
    "VarianceThresholdSelectorModel",
    "VectorAssembler",
    "VectorIndexer",
    "VectorIndexerModel",
    "VectorSizeHint",
    "VectorSlicer",
    "Word2Vec",
    "Word2VecModel",
]

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
    "ALS",
    "ALSModel",
    "LDA",
    "PowerIterationClustering",
    "GaussianMixture",
    "BisectingKMeans",
    "ClusteringEvaluator",
    "MultilabelClassificationEvaluator",
    "RankingEvaluator",
    "RegressionEvaluator",
    "BinaryClassificationEvaluator",
    "MulticlassClassificationEvaluator",
    "CrossValidator",
    "CrossValidatorModel",
    "TrainValidationSplit",
    "TrainValidationSplitModel",
    "MLModel",
    "FPGrowth",
    "FPGrowthModel",
    "PrefixSpan",
]

__all__ += [name for name in _native._ml_feature_names if name not in __all__]


def _register_native_classes() -> None:
    """Register the native ML classes as virtual subclasses of PySpark's ML base classes (ABCs), so
    that the `isinstance` checks of pyspark.ml hold of them as of PySpark's own classes: a stage of
    a Pipeline, the estimator and evaluator of a validator, and `Params`. Run at the import of
    pyspark.ml.connect, which the import of any of its modules runs first."""
    from pyspark.ml.base import Estimator as _Estimator
    from pyspark.ml.base import Model as _Model
    from pyspark.ml.base import Transformer as _Transformer
    from pyspark.ml.evaluation import Evaluator as _Evaluator
    from pyspark.ml.param import Params as _Params

    for name in dir(_native):
        cls = getattr(_native, name)
        if not isinstance(cls, type) or not cls.__module__.startswith("pyspark.ml.connect"):
            continue
        if hasattr(cls, "fit"):
            _Estimator.register(cls)
        elif hasattr(cls, "evaluate") and hasattr(cls, "isLargerBetter"):
            _Evaluator.register(cls)
        elif hasattr(cls, "transform"):
            (_Model if name.endswith("Model") else _Transformer).register(cls)
        elif hasattr(cls, "params"):
            # A class of Params that neither fits nor transforms (PrefixSpan).
            _Params.register(cls)

    # The meta-algorithms are PySpark's classes of them too, so that `MetaAlgorithmReadWrite`
    # reads the nested stages of a native Pipeline (`getUidMap` of a validator's estimator) and
    # refuses a nested validator, as of PySpark's.
    from pyspark.ml import pipeline as _pipeline
    from pyspark.ml import tuning as _tuning
    from pyspark.ml import classification as _classification

    # The OneVsRest is one of them too, so that its classifier and its models are the nested
    # stages of a validator, as of PySpark's OneVsRest.
    for native, vendored in [
        (_native.Pipeline, _pipeline.Pipeline),
        (_native.PipelineModel, _pipeline.PipelineModel),
        (_native.CrossValidator, _tuning.CrossValidator),
        (_native.CrossValidatorModel, _tuning.CrossValidatorModel),
        (_native.TrainValidationSplit, _tuning.TrainValidationSplit),
        (_native.TrainValidationSplitModel, _tuning.TrainValidationSplitModel),
        (_native.OneVsRest, _classification.OneVsRest),
        (_native.OneVsRestModel, _classification.OneVsRestModel),
    ]:
        vendored.register(native)


_register_native_classes()


def _publish_native_classes() -> None:
    """Make the public classes of every native ML class (see `_classes`), so that every object that
    the native module makes is of its public class, also of a module that the user has not
    imported (the trees of a GBT classifier are regression models)."""
    import importlib

    from pyspark.ml._native_classes import publish

    # The native stages of the vendored feature module, installed above.
    for name, public in publish(
        {name: getattr(_native, name) for name in _native._ml_feature_names}, "pyspark.ml.feature"
    ).items():
        globals()[name] = public
        setattr(feature, name, public)
    for module in [
        "classification",
        "clustering",
        "evaluation",
        "feature",
        "fpm",
        "pipeline",
        "recommendation",
        "regression",
        "tuning",
    ]:
        importlib.import_module("pyspark.ml.connect." + module)
    importlib.import_module("pyspark.ml.stat")
    # The writers, readers and the iterator of `fitMultiple`, of no module of their own.
    publish(
        {"RemoteMLWriter": _native.RemoteMLWriter, "RemoteMLReader": _native.RemoteMLReader},
        "pyspark.ml.connect.readwrite",
    )
    publish({"_FitMultipleIterator": _native._FitMultipleIterator}, "pyspark.ml.base")


_publish_native_classes()
