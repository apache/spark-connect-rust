# SPDX-License-Identifier: Apache-2.0
"""The pyspark.ml API of the native ML classes that follow it in full, without a server.

- Their members against official PySpark's, of tests/golden/ml_api.json (from
  scripts/capture_ml_api_golden.py): the names of the members and Params, the kinds of the
  members, the parameters of each method and of a keyword-only constructor.
- The cases of PySpark's `Params` API (`_ml_api_cases.params_api_cases`) against
  tests/golden/ml_api_cases.json (from scripts/capture_ml_api_cases_golden.py).

Official PySpark runs these tests too, as the reference of the goldens (see `_native_classes`).
"""

import functools
import importlib
import inspect
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
from _ml_api_cases import CLASSES, params_api_cases
from _native_classes import native_class, native_module

ROOT = Path(__file__).resolve().parents[2]
API = json.loads((ROOT / "tests/golden/ml_api.json").read_text())
CASES = json.loads((ROOT / "tests/golden/ml_api_cases.json").read_text())

# The attributes of the instances of PySpark's classes, of their constructors, by class.
INSTANCE_ATTRIBUTES = {
    "CrossValidatorModel": {"avgMetrics", "bestModel", "stdMetrics", "subModels"},
    "OneVsRestModel": {"models"},
    "PipelineModel": {"stages"},
    "TrainValidationSplitModel": {"bestModel", "subModels", "validationMetrics"},
}

# The classes that follow pyspark.ml in full, by the module of pyspark.ml that has them: one per
# line, so that the classes of each module are added where they sort.
FULL = [
    (f"pyspark.ml.connect.{module}", name, f"pyspark.ml.{module}")
    for module, name in [
        ("classification", "BinaryLogisticRegressionSummary"),
        ("classification", "BinaryLogisticRegressionTrainingSummary"),
        ("classification", "LinearSVC"),
        ("classification", "LinearSVCModel"),
        ("classification", "LinearSVCSummary"),
        ("classification", "LinearSVCTrainingSummary"),
        ("classification", "BinaryRandomForestClassificationSummary"),
        ("classification", "BinaryRandomForestClassificationTrainingSummary"),
        ("classification", "DecisionTreeClassificationModel"),
        ("classification", "DecisionTreeClassifier"),
        ("classification", "FMClassificationModel"),
        ("classification", "FMClassificationSummary"),
        ("classification", "FMClassificationTrainingSummary"),
        ("classification", "FMClassifier"),
        ("classification", "GBTClassificationModel"),
        ("classification", "GBTClassifier"),
        ("classification", "LogisticRegression"),
        ("classification", "LogisticRegressionModel"),
        ("classification", "LogisticRegressionSummary"),
        ("classification", "LogisticRegressionTrainingSummary"),
        ("classification", "MultilayerPerceptronClassificationModel"),
        ("classification", "MultilayerPerceptronClassificationSummary"),
        ("classification", "MultilayerPerceptronClassificationTrainingSummary"),
        ("classification", "MultilayerPerceptronClassifier"),
        ("classification", "NaiveBayes"),
        ("classification", "NaiveBayesModel"),
        ("classification", "OneVsRest"),
        ("classification", "OneVsRestModel"),
        ("classification", "RandomForestClassificationModel"),
        ("classification", "RandomForestClassificationSummary"),
        ("classification", "RandomForestClassificationTrainingSummary"),
        ("classification", "RandomForestClassifier"),
        ("clustering", "BisectingKMeans"),
        ("clustering", "BisectingKMeansModel"),
        ("clustering", "BisectingKMeansSummary"),
        ("clustering", "DistributedLDAModel"),
        ("clustering", "GaussianMixture"),
        ("clustering", "GaussianMixtureModel"),
        ("clustering", "GaussianMixtureSummary"),
        ("clustering", "KMeans"),
        ("clustering", "KMeansModel"),
        ("clustering", "KMeansSummary"),
        ("clustering", "LDA"),
        ("clustering", "LDAModel"),
        ("clustering", "LocalLDAModel"),
        ("clustering", "PowerIterationClustering"),
        ("evaluation", "BinaryClassificationEvaluator"),
        ("evaluation", "ClusteringEvaluator"),
        ("evaluation", "MulticlassClassificationEvaluator"),
        ("evaluation", "MultilabelClassificationEvaluator"),
        ("evaluation", "RankingEvaluator"),
        ("evaluation", "RegressionEvaluator"),
        ("feature", "Binarizer"),
        ("feature", "BucketedRandomProjectionLSH"),
        ("feature", "BucketedRandomProjectionLSHModel"),
        ("feature", "Bucketizer"),
        ("feature", "ChiSqSelector"),
        ("feature", "ChiSqSelectorModel"),
        ("feature", "CountVectorizer"),
        ("feature", "CountVectorizerModel"),
        ("feature", "DCT"),
        ("feature", "ElementwiseProduct"),
        ("feature", "FeatureHasher"),
        ("feature", "HashingTF"),
        ("feature", "IDF"),
        ("feature", "IDFModel"),
        ("feature", "Imputer"),
        ("feature", "ImputerModel"),
        ("feature", "IndexToString"),
        ("feature", "Interaction"),
        ("feature", "MaxAbsScaler"),
        ("feature", "MaxAbsScalerModel"),
        ("feature", "MinHashLSH"),
        ("feature", "MinHashLSHModel"),
        ("feature", "MinMaxScaler"),
        ("feature", "MinMaxScalerModel"),
        ("feature", "NGram"),
        ("feature", "Normalizer"),
        ("feature", "OneHotEncoder"),
        ("feature", "OneHotEncoderModel"),
        ("feature", "PCA"),
        ("feature", "PCAModel"),
        ("feature", "PolynomialExpansion"),
        ("feature", "QuantileDiscretizer"),
        ("feature", "RFormula"),
        ("feature", "RFormulaModel"),
        ("feature", "RegexTokenizer"),
        ("feature", "RobustScaler"),
        ("feature", "RobustScalerModel"),
        ("feature", "SQLTransformer"),
        ("feature", "StandardScaler"),
        ("feature", "StandardScalerModel"),
        ("feature", "StopWordsRemover"),
        ("feature", "StringIndexer"),
        ("feature", "StringIndexerModel"),
        ("feature", "TargetEncoder"),
        ("feature", "TargetEncoderModel"),
        ("feature", "Tokenizer"),
        ("feature", "UnivariateFeatureSelector"),
        ("feature", "UnivariateFeatureSelectorModel"),
        ("feature", "VarianceThresholdSelector"),
        ("feature", "VarianceThresholdSelectorModel"),
        ("feature", "VectorAssembler"),
        ("feature", "VectorIndexer"),
        ("feature", "VectorIndexerModel"),
        ("feature", "VectorSizeHint"),
        ("feature", "VectorSlicer"),
        ("feature", "Word2Vec"),
        ("feature", "Word2VecModel"),
        ("fpm", "FPGrowth"),
        ("fpm", "FPGrowthModel"),
        ("fpm", "PrefixSpan"),
        ("pipeline", "Pipeline"),
        ("pipeline", "PipelineModel"),
        ("recommendation", "ALS"),
        ("recommendation", "ALSModel"),
        ("regression", "AFTSurvivalRegression"),
        ("regression", "AFTSurvivalRegressionModel"),
        ("regression", "DecisionTreeRegressionModel"),
        ("regression", "DecisionTreeRegressor"),
        ("regression", "FMRegressionModel"),
        ("regression", "FMRegressor"),
        ("regression", "GBTRegressionModel"),
        ("regression", "GBTRegressor"),
        ("regression", "GeneralizedLinearRegression"),
        ("regression", "GeneralizedLinearRegressionModel"),
        ("regression", "GeneralizedLinearRegressionSummary"),
        ("regression", "GeneralizedLinearRegressionTrainingSummary"),
        ("regression", "IsotonicRegression"),
        ("regression", "IsotonicRegressionModel"),
        ("regression", "LinearRegression"),
        ("regression", "LinearRegressionModel"),
        ("regression", "LinearRegressionSummary"),
        ("regression", "LinearRegressionTrainingSummary"),
        ("regression", "RandomForestRegressionModel"),
        ("regression", "RandomForestRegressor"),
        ("tuning", "CrossValidator"),
        ("tuning", "CrossValidatorModel"),
        ("tuning", "TrainValidationSplit"),
        ("tuning", "TrainValidationSplitModel"),
    ]
]


# The kinds of the golden that a native class has as another: a `functools.cached_property` is a
# property, and a method of `functools.cache` a method (the cases test that they cache).
_SAME_KINDS = {"cached_property": "property", "_lru_cache_wrapper": "method"}


def _kind(cls, name):
    """The kind of the member `name` of `cls`, as the capture script names it (of `_SAME_KINDS`)."""
    raw = inspect.getattr_static(cls, name)
    if type(raw).__name__ in _SAME_KINDS:
        return _SAME_KINDS[type(raw).__name__]
    if isinstance(raw, (property, type(int.real))) or type(raw).__name__ == "getset_descriptor":
        return "property"
    if isinstance(raw, staticmethod):
        return "staticmethod"
    if isinstance(raw, classmethod) or type(raw).__name__ == "classmethod_descriptor":
        return "classmethod"
    if inspect.isfunction(raw) or type(raw).__name__ == "method_descriptor":
        return "method"
    return type(raw).__name__


def _parameters(func, bound):
    parameters = list(inspect.signature(func).parameters.values())
    if bound:
        parameters = parameters[1:]
    return [
        [p.name, p.kind.name, None if p.default is inspect.Parameter.empty else repr(p.default)]
        for p in parameters
    ]


@pytest.mark.parametrize("spec", FULL, ids=lambda spec: spec[1])
def test_members_are_pyspark_mls(spec):
    cls = native_class(*spec)
    golden = API[spec[1]]
    expected = set(golden["members"]) | {p["name"] for p in golden["params"]}
    # The attributes that PySpark sets of each instance (the uid of `Identifiable`, those of the
    # constructors of the models of the meta-algorithms), which the class of a native object has
    # as properties (the cases read them of the instances).
    actual = {name for name in dir(cls) if not name.startswith("_")} - {"uid"}
    actual -= INSTANCE_ATTRIBUTES.get(spec[1], set())
    assert actual == expected


# The members of PySpark's classes that a class of this client runs as they are, of its MRO:
# PySpark's Python code that reads only the class's own members (RFormula's `__str__`, of
# `getFormula`, `isDefined` and `uid`; OneVsRestModel's `transform`, PySpark's
# `Transformer.transform` of its `copy` and of `OneVsRestModel._transform`, which composes the
# transforms of its `models`).
_PYSPARKS_OWN_CODE = {("RFormula", "__str__"), ("OneVsRestModel", "transform")}


@pytest.mark.parametrize("spec", FULL, ids=lambda spec: spec[1])
def test_public_members_and_dunders_are_the_native_classes(spec):
    # A public member or dunder of PySpark's class that the native class lacks would run PySpark's
    # code on a native object, whose `_call_java` reads PySpark's server-side object, of the
    # `_java_obj` of a model or a summary, and not the native object's own: the `__repr__` of a
    # GeneralizedLinearRegressionTrainingSummary and the `__str__` of an RFormulaModel did, and
    # raised. Each such member is the native class's own, or PySpark's code of its own members.
    cls = native_class(*spec)
    if native_module() is None:
        pytest.skip("of this client's native classes")
    pysparks = {
        k
        for k in cls.__mro__
        if k.__module__.startswith("pyspark.ml.")
        and not k.__module__.startswith("pyspark.ml.connect")
        and not issubclass(k, getattr(native_module(), k.__name__, type(None)))
    }
    descriptors = (property, classmethod, staticmethod, functools.cached_property)
    of_pyspark = []
    for name in dir(cls):
        if name.startswith("_") and not (name.startswith("__") and name.endswith("__")):
            continue
        owner = next(k for k in cls.__mro__ if name in vars(k))
        member = vars(owner)[name]
        is_code = callable(member) or isinstance(member, descriptors)
        if owner in pysparks and is_code and (spec[1], name) not in _PYSPARKS_OWN_CODE:
            of_pyspark.append(f"{name} of {owner.__module__}.{owner.__qualname__}")
    assert of_pyspark == []


@pytest.mark.parametrize("spec", FULL, ids=lambda spec: spec[1])
def test_member_kinds_and_parameters_are_pyspark_mls(spec):
    cls = native_class(*spec)
    for name, member in API[spec[1]]["members"].items():
        kind = _SAME_KINDS.get(member["kind"], member["kind"])
        assert _kind(cls, name) == kind, name
        if "parameters" in member:
            # A method of the class has `self` (a classmethod of the class is bound to it).
            parameters = _parameters(getattr(cls, name), kind == "method")
            assert parameters == member["parameters"], name


@pytest.mark.parametrize("spec", FULL, ids=lambda spec: spec[1])
def test_constructors_are_pyspark_mls(spec):
    cls = native_class(*spec)
    init = API[spec[1]]["init"]
    if not init or any(kind != "KEYWORD_ONLY" for _, kind, _ in init):
        # The constructor of a model or a summary, of its server-side object, is PySpark's own;
        # the native classes make them of a fit, a load or a model.
        pytest.skip("not a public constructor")
    assert _parameters(cls, False) == init


@pytest.mark.parametrize("spec", FULL, ids=lambda spec: spec[1])
def test_params_are_pyspark_mls(spec):
    # The Params of the class (`LinearSVC.maxIter`), which a native class has as PySpark's: of the
    # parent "undefined", of the name, doc and converter of PySpark's class, in its order. Read of
    # the class, as a params walk of an instance would read its cached members, of a session.
    from pyspark.ml.param import Param

    cls = native_class(*spec)
    params = [
        getattr(cls, name)
        for name in sorted(dir(cls))
        if isinstance(getattr(cls, name, None), Param)
    ]
    golden = API[spec[1]]["params"]
    assert [(p.parent, p.name, p.doc, p.typeConverter.__name__) for p in params] == [
        ("undefined", g["name"], g["doc"], g["converter"]) for g in golden
    ]


def test_param_grid_builder_is_pysparks():
    # The ParamGridBuilder of the native validators is PySpark's own class, whose param maps of
    # the Params of any class they take.
    if native_module() is None:
        pytest.skip("of this client's native classes")
    from pyspark.ml import tuning
    from pyspark.ml.connect import tuning as native

    assert native.ParamGridBuilder is tuning.ParamGridBuilder


# PySpark's abstract base classes that the native modules have, as PySpark's modules have them
# outside their __all__ (`from pyspark.ml.classification import Classifier`), by module, each with
# a native class that is a subclass of it.
_PYSPARKS_BASES = {
    "classification": {
        "ClassificationModel": "LogisticRegressionModel",
        "Classifier": "LogisticRegression",
        "ProbabilisticClassificationModel": "LogisticRegressionModel",
        "ProbabilisticClassifier": "LogisticRegression",
    },
    "clustering": {"ClusteringSummary": "KMeansSummary"},
    "evaluation": {"JavaEvaluator": "RegressionEvaluator"},
    "regression": {"RegressionModel": "LinearRegressionModel", "Regressor": "LinearRegression"},
}


@pytest.mark.parametrize(
    "module",
    [
        "classification",
        "clustering",
        "evaluation",
        "feature",
        "fpm",
        "recommendation",
        "regression",
        "tuning",
    ],
)
def test_modules_have_the_names_of_pyspark_mls_modules(module):
    # The __all__ of a native module is PySpark's, in its order, and its abstract base classes are
    # PySpark's own, of which the native classes are subclasses.
    if native_module() is None:
        pytest.skip("of this client's native classes")
    official = importlib.import_module("pyspark.ml." + module)
    native = importlib.import_module("pyspark.ml.connect." + module)
    assert native.__all__ == official.__all__
    for base, subclass in _PYSPARKS_BASES.get(module, {}).items():
        assert getattr(native, base) is getattr(official, base)
        assert issubclass(getattr(native, subclass), getattr(native, base)), base


@pytest.mark.parametrize("name", list(CLASSES))
def test_params_api_cases_are_pyspark_mls(name):
    ours = params_api_cases(name)
    golden = CASES[name]["params_api"]
    assert sorted(ours) == sorted(golden)
    differ = {tag: (ours[tag], golden[tag]) for tag in golden if ours[tag] != golden[tag]}
    assert not differ, json.dumps(differ, indent=1)


_JAVA_MODEL_INIT = [["java_model", "POSITIONAL_OR_KEYWORD", "None"]]


@pytest.mark.parametrize(
    "spec", [s for s in FULL if API[s[1]]["init"] == _JAVA_MODEL_INIT], ids=lambda spec: spec[1]
)
def test_models_construct_of_no_server_object_as_pyspark_mls(spec):
    # As PySpark's `JavaModel(java_model=None)`: a model of its params, none of them set, of a new
    # uid of its class and of no server-side object, whose members of the server raise
    # (test_ml_api_e2e.py). Its constructor is PySpark's, whose classes set the defaults in the
    # order of its MRO, of which `_defaultParamMap` and `extractParamMap()` are: that of an object
    # of PySpark's class.
    cls = native_class(*spec)
    official = getattr(importlib.import_module(spec[2]), spec[1])
    order = [p.name for p in official()._defaultParamMap]
    assert API[spec[1]]["default_instance"]
    for model in (cls(), cls(None), cls(java_model=None)):
        assert type(model) is cls
        assert model.uid.startswith(spec[1] + "_") and len(model.uid) == len(spec[1]) + 13
        assert not any(model.isSet(p) for p in model.extractParamMap())
        assert [p.name for p in model._defaultParamMap] == order
        assert [p.name for p in model.extractParamMap()] == order


@pytest.mark.parametrize(
    "module,name,argument,expected",
    [
        ("tuning", "CrossValidatorModel", None, ["seed", "numFolds", "foldCol"]),
        ("tuning", "TrainValidationSplitModel", None, ["seed", "trainRatio"]),
        ("classification", "OneVsRestModel", [], ["predictionCol", "featuresCol", "labelCol", "rawPredictionCol"]),
    ],
)
def test_meta_models_have_the_defaults_of_their_constructor_in_its_order(module, name, argument, expected):
    # The constructor of PySpark's validator model and OneVsRestModel sets the defaults of its
    # classes in the order of its MRO (`HasSeed`, then the class's), of which its `_defaultParamMap`
    # is, of a model of the user's constructor and of a fit (test_ml_api_e2e.py); the `_resetUid`
    # of a load rebuilds it of the names.
    cls = native_class(f"pyspark.ml.connect.{module}", name, f"pyspark.ml.{module}")
    official = getattr(importlib.import_module(f"pyspark.ml.{module}"), name)
    assert [p.name for p in official(argument)._defaultParamMap] == expected
    assert [p.name for p in cls(argument)._defaultParamMap] == expected
    assert [p.name for p in cls(argument).extractParamMap()] == expected


@pytest.mark.parametrize("spec", FULL, ids=lambda spec: spec[1])
def test_classes_are_of_every_base_of_pyspark_mls(spec):
    # The class is a subclass of every class of the MRO of PySpark's class, as its objects are
    # instances of them: the plain mixins (`MLWritable`, `MLReadable`, `Identifiable`,
    # `HasTrainingSummary`, the summary bases), the Java wrappers, the param mixins and PySpark's
    # class itself. It is also a subclass of this client's class of each base of PySpark's module
    # that is a class of the full API (a training summary of its summary), as PySpark's is.
    cls = native_class(*spec)
    official = getattr(importlib.import_module(spec[2]), spec[1])
    assert [base for base in official.__mro__ if not issubclass(cls, base)] == []
    full = {name for _, name, _ in FULL}
    for base in official.__mro__[1:]:
        if base.__module__ == spec[2] and base.__name__ in full:
            assert issubclass(cls, native_class(spec[0], base.__name__, spec[2])), base


def test_native_classes_are_of_pyspark_mls_bases_on_a_fresh_import():
    # The registration runs at the import of any module of pyspark.ml.connect, before any check of
    # the user's.
    code = """
import importlib
from _native_classes import native_module
if native_module() is None:
    module = importlib.import_module("pyspark.ml.classification")
else:
    module = importlib.import_module("pyspark.ml.connect.classification")
from pyspark.ml.base import Estimator, Model
from pyspark.ml.param import Params
print(isinstance(module.LinearSVC(), Estimator), isinstance(module.LinearSVC(), Params),
      issubclass(module.LinearSVCModel, Model))
"""
    # Of the path of this process (ours or official PySpark) and a directory without a pyspark:
    # the directory of the tests' cwd would put the drop-in first on the path of an official run.
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(p for p in sys.path if p))
    out = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=tempfile.gettempdir(),
    )
    assert out.returncode == 0, out.stderr
    assert out.stdout.split() == ["True", "True", "True"]
