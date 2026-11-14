# SPDX-License-Identifier: Apache-2.0
"""The cases of the pyspark.ml API of the native regression classes that follow it in full:
the estimators LinearRegression, GeneralizedLinearRegression, AFTSurvivalRegression,
IsotonicRegression, FMRegressor, DecisionTreeRegressor, RandomForestRegressor and GBTRegressor,
with their models and summaries.

`scripts/capture_regression_api_cases_golden.py` runs them on official PySpark for
tests/golden/regression_api_cases.json, and test_regression_api_offline.py and
test_regression_api_e2e.py run them on the native classes and compare. Official PySpark runs the
tests too, as the reference (`_native_classes.native_class`). The cases reuse the `Recorder` and
the shared case sets of `_ml_api_cases`, as LinearSVC and KMeans do.
"""

import json
import re
from pathlib import Path

from _ml_api_cases import Recorder, model_cases, params_api_common_cases, persistence_cases
from _native_classes import native_class

# The type of a native class in an error message is its module-qualified name, as PyO3 gives it;
# official PySpark's classes are Python classes, whose message has the bare name.
_QUALIFIED = re.compile(r"'pyspark\.ml\.connect\.\w+\.(\w+)'")


class RegressionRecorder(Recorder):
    """The `Recorder` of the cases, which normalizes the module of a native class in an error, and
    rounds the floats of DataFrames (`rounded`): the predictions and residuals of these models
    differ in their last bits between the servers of platforms."""

    def __init__(self, base=None):
        super().__init__(base, rounded=True)

    def __call__(self, tag, f):
        super().__call__(tag, f)
        ok, message = self.out[tag]
        if ok != "ok":
            self.out[tag] = [ok, _QUALIFIED.sub(r"'\1'", message)]

# The classes of each case set: the estimator, then the model, as (module of the native class,
# name, module of PySpark's).
CLASSES = {
    "LinearRegression": [
        ("pyspark.ml.connect.regression", "LinearRegression", "pyspark.ml.regression"),
        ("pyspark.ml.connect.regression", "LinearRegressionModel", "pyspark.ml.regression"),
    ],
    "GeneralizedLinearRegression": [
        ("pyspark.ml.connect.regression", "GeneralizedLinearRegression", "pyspark.ml.regression"),
        (
            "pyspark.ml.connect.regression",
            "GeneralizedLinearRegressionModel",
            "pyspark.ml.regression",
        ),
    ],
    "AFTSurvivalRegression": [
        ("pyspark.ml.connect.regression", "AFTSurvivalRegression", "pyspark.ml.regression"),
        ("pyspark.ml.connect.regression", "AFTSurvivalRegressionModel", "pyspark.ml.regression"),
    ],
    "IsotonicRegression": [
        ("pyspark.ml.connect.regression", "IsotonicRegression", "pyspark.ml.regression"),
        ("pyspark.ml.connect.regression", "IsotonicRegressionModel", "pyspark.ml.regression"),
    ],
    "FMRegressor": [
        ("pyspark.ml.connect.regression", "FMRegressor", "pyspark.ml.regression"),
        ("pyspark.ml.connect.regression", "FMRegressionModel", "pyspark.ml.regression"),
    ],
    "DecisionTreeRegressor": [
        ("pyspark.ml.connect.regression", "DecisionTreeRegressor", "pyspark.ml.regression"),
        ("pyspark.ml.connect.regression", "DecisionTreeRegressionModel", "pyspark.ml.regression"),
    ],
    "RandomForestRegressor": [
        ("pyspark.ml.connect.regression", "RandomForestRegressor", "pyspark.ml.regression"),
        ("pyspark.ml.connect.regression", "RandomForestRegressionModel", "pyspark.ml.regression"),
    ],
    "GBTRegressor": [
        ("pyspark.ml.connect.regression", "GBTRegressor", "pyspark.ml.regression"),
        ("pyspark.ml.connect.regression", "GBTRegressionModel", "pyspark.ml.regression"),
    ],
}

# The constructor arguments of each estimator in the cases.
KWARGS = {
    "LinearRegression": {"maxIter": 5, "regParam": 0.1},
    "GeneralizedLinearRegression": {"family": "gaussian", "link": "identity", "maxIter": 5},
    "AFTSurvivalRegression": {"maxIter": 5},
    "IsotonicRegression": {"weightCol": "w"},
    "FMRegressor": {"factorSize": 2, "maxIter": 3, "seed": 1},
    "DecisionTreeRegressor": {"maxDepth": 3, "seed": 1},
    "RandomForestRegressor": {"numTrees": 3, "maxDepth": 2, "seed": 7},
    "GBTRegressor": {"maxIter": 3, "maxDepth": 2, "seed": 7},
}

# The estimators that have the params of `params_api_common_cases` (maxIter, tol, weightCol) and
# their own params, with the int, float and column params of the other estimators.
COMMON = {"LinearRegression", "GeneralizedLinearRegression", "FMRegressor"}
OWN = {
    "AFTSurvivalRegression": ("maxIter", "tol", "quantilesCol"),
    "GBTRegressor": ("maxIter", "stepSize", "weightCol"),
    "IsotonicRegression": ("featureIndex", None, "weightCol"),
    "DecisionTreeRegressor": ("maxDepth", "minInfoGain", "weightCol"),
    "RandomForestRegressor": ("numTrees", "subsamplingRate", "weightCol"),
}
# The int param of each estimator that a fit of a param map of the class sets, when it has no maxIter.
CLASS_PARAM = {
    "IsotonicRegression": ("featureIndex", 0),
    "DecisionTreeRegressor": ("maxDepth", 3),
    "RandomForestRegressor": ("maxDepth", 3),
}
SEEDED = {"FMRegressor", "DecisionTreeRegressor", "RandomForestRegressor", "GBTRegressor"}

TRAIN_PROPS = {
    "LinearRegression": [
        "r2", "r2adj", "meanSquaredError", "meanAbsoluteError", "rootMeanSquaredError",
        "explainedVariance", "numInstances", "degreesOfFreedom", "devianceResiduals",
        "coefficientStandardErrors", "tValues", "pValues", "featuresCol", "labelCol",
        "predictionCol", "objectiveHistory", "totalIterations",
    ],
    "GeneralizedLinearRegression": [
        "aic", "degreesOfFreedom", "deviance", "dispersion", "nullDeviance", "numInstances",
        "predictionCol", "rank", "residualDegreeOfFreedom", "residualDegreeOfFreedomNull",
        "coefficientStandardErrors", "numIterations", "pValues", "solver", "tValues",
    ],
}

SUMMARY_PROPS = {
    "LinearRegression": [
        "r2", "r2adj", "meanSquaredError", "meanAbsoluteError", "rootMeanSquaredError",
        "explainedVariance", "numInstances", "degreesOfFreedom", "devianceResiduals",
        "featuresCol", "labelCol", "predictionCol",
    ],
    "GeneralizedLinearRegression": [
        "aic", "degreesOfFreedom", "deviance", "dispersion", "nullDeviance", "numInstances",
        "predictionCol", "rank", "residualDegreeOfFreedom", "residualDegreeOfFreedomNull",
    ],
}


def classes(name):
    """The estimator and model classes of the case set `name`."""
    return [native_class(*spec) for spec in CLASSES[name]]


def _regression_frame(spark):
    from pyspark.ml.linalg import Vectors

    # The label is linear in the features, of a deterministic noise, so that the statistics of a
    # summary (the standard errors, t-values and AIC) are of a fit that is not exact: of an exact
    # fit, they are of rounding errors, which differ between platforms.
    rows = [
        (
            Vectors.dense([float(i), float(i % 3)]),
            float(i * 2 + (i % 3) + 5) + 0.25 * ((i * 7) % 5 - 2),
            float(1 + i % 2),
            1.0 if i % 4 else 0.0,
        )
        for i in range(40)
    ]
    return spark.createDataFrame(rows, ["features", "label", "w", "censor"]).coalesce(1)


def _own_params_cases(rec, est, other, int_param, float_param, col_param):
    """The cases of PySpark's `Params` API of an estimator without maxIter, of `int_param`,
    `float_param` and `col_param` (a column param) of its own."""
    rec("param_names", lambda: [p.name for p in est.params])
    rec("param_docs", lambda: [p.doc for p in est.params])
    rec("param_parents_are_uid", lambda: all(p.parent == est.uid for p in est.params))
    rec("explainParams", lambda: est.explainParams())
    rec("explainParam_str", lambda: est.explainParam(int_param))
    for p in est.params:
        getter = "get" + p.name[0].upper() + p.name[1:]
        rec(getter, lambda getter=getter: getattr(est, getter)())
    rec("isSet_int", lambda: est.isSet(int_param))
    rec("isSet_nope", lambda: est.isSet("nope"))
    rec("isDefined_col", lambda: est.isDefined(col_param))
    rec("hasDefault_col", lambda: est.hasDefault(col_param))
    rec("hasParam_nope", lambda: est.hasParam("nope"))
    rec("hasParam_int", lambda: est.hasParam(1))
    rec("getParam_nope", lambda: est.getParam("nope"))
    rec("getParam_method", lambda: est.getParam("fit"))
    rec("getOrDefault_int", lambda: est.getOrDefault(int_param))
    rec("extract", lambda: sorted((p.name, repr(v)) for p, v in est.extractParamMap().items()))
    rec("repr_is_uid", lambda: repr(est) == est.uid)
    rec("set_bad_type", lambda: est.set(getattr(est, int_param), "x"))
    rec("set_str_param", lambda: est.set(int_param, 3))
    rec("clear", lambda: (est.clear(int_param), est.getOrDefault(int_param))[1])
    rec("isSet_foreign", lambda: est.isSet(other.getParam(int_param)))
    if float_param:
        rec("explainParam_float", lambda: est.explainParam(getattr(est, float_param)))
        rec("hasDefault_float", lambda: est.hasDefault(float_param))
        rec("extract_extra", lambda: sorted(
            (p.name, repr(v))
            for p, v in est.extractParamMap({getattr(est, float_param): 0.5}).items()))
    copy = est.copy({getattr(est, int_param): 10})
    rec("copy_int", lambda: copy.getOrDefault(int_param))
    rec("copy_same_uid", lambda: copy.uid == est.uid)
    rec("orig_int", lambda: est.getOrDefault(int_param))
    rec("copy_bad_extra", lambda: est.copy({"nope": 3}))
    rec("copy_list", lambda: est.copy([1]))


def params_api_cases(name):
    """The cases of PySpark's `Params` API of the estimator of the case set `name`, without a
    server."""
    estimator = classes(name)[0]
    est = estimator(**KWARGS[name])
    rec = RegressionRecorder()
    if name in COMMON:
        params_api_common_cases(rec, est, estimator())
    else:
        int_param, float_param, col_param = OWN[name]
        _own_params_cases(rec, est, estimator(), int_param, float_param, col_param)
    if name in SEEDED:
        rec("seed_is_class_name_hash", lambda: est.getSeed() == hash(name))
        rec("hasDefault_seed", lambda: est.hasDefault("seed"))
        rec("isSet_seed", lambda: est.isSet("seed"))
        rec("explainParam_seed", lambda: est.explainParam("seed"))
        rec("copy_seed", lambda: est.copy().getSeed() == hash(name))
        rec("setSeed", lambda: estimator().setSeed(42).getSeed())
        rec("seed_kwarg", lambda: estimator(seed=7).explainParam("seed"))
    return rec.out


def _ensemble_members(rec, model, df, v):
    rec("getNumTrees", lambda: model.getNumTrees)
    rec("totalNumNodes", lambda: model.totalNumNodes)
    rec("treeWeights", lambda: model.treeWeights)
    rec("trees_len", lambda: len(model.trees))
    rec("trees_is_trees", lambda: model.trees is model.trees)
    rec("copy_trees_is_trees", lambda: model.copy().trees is model.trees)
    rec("trees_types", lambda: sorted({type(t).__name__ for t in model.trees}))
    rec("trees_uids_differ", lambda: len({t.uid for t in model.trees}) == len(model.trees))
    rec("trees_uid_not_ensemble", lambda: all(t.uid != model.uid for t in model.trees))
    rec("trees_depth", lambda: [t.depth for t in model.trees])
    rec("trees_seed_hash", lambda: all(
        t.getSeed() == hash("DecisionTreeRegressionModel") for t in model.trees))
    rec("trees_params", lambda: [sorted(p.name for p in t.params) == sorted(
        p.name for p in model.trees[0].params) for t in model.trees])
    rec("trees_predict", lambda: [t.predict(v) for t in model.trees])
    rec("trees_transform_count", lambda: model.trees[0].transform(df).count())
    rec("toDebugString_head", lambda: model.toDebugString.split("\n")[0])
    rec("toDebugString_lines", lambda: len(model.toDebugString.split("\n")))


def _live_linear_regression(rec, est, model, df):
    from pyspark.ml.linalg import Vectors

    v = Vectors.dense([3.0, 1.0])
    for a in ["coefficients", "intercept", "numFeatures", "scale", "hasSummary"]:
        rec(a, lambda a=a: getattr(model, a))
    rec("predict", lambda: model.predict(v))
    rec("model_getMaxIter", lambda: model.getMaxIter())
    rec("model_isSet_regParam", lambda: model.isSet("regParam"))
    s = model.summary
    rec("summary_is_summary", lambda: model.summary is s)
    rec("copy_summary_is_summary", lambda: model.copy().summary is s)
    for a in SUMMARY_PROPS["LinearRegression"] + ["objectiveHistory", "totalIterations"]:
        rec("summary_" + a, lambda a=a: getattr(s, a))
    rec("summary_predictions", lambda: s.predictions)
    rec("summary_residuals", lambda: s.residuals)
    e = model.evaluate(df)
    rec("evaluate_type", lambda: type(e).__name__)
    rec("evaluate_r2", lambda: e.r2)
    rec("evaluate_numInstances", lambda: e.numInstances)
    rec("evaluate_residuals", lambda: e.residuals)
    rec("evaluate_bad", lambda: model.evaluate(5))
    rec("transform_map", lambda: model.transform(df, {model.predictionCol: "q"}).columns)
    rec("copy_predictionCol", lambda: model.copy().setPredictionCol("p").getPredictionCol())


def _live_glr(rec, est, model, df):
    from pyspark.ml.linalg import Vectors

    v = Vectors.dense([3.0, 1.0])
    for a in ["coefficients", "intercept", "numFeatures", "hasSummary"]:
        rec(a, lambda a=a: getattr(model, a))
    rec("predict", lambda: model.predict(v))
    rec("model_isSet_link", lambda: model.isSet("link"))
    s = model.summary
    rec("summary_is_summary", lambda: model.summary is s)
    rec("copy_summary_is_summary", lambda: model.copy().summary is s)
    for a in SUMMARY_PROPS["GeneralizedLinearRegression"] + [
        "coefficientStandardErrors", "numIterations", "pValues", "solver", "tValues"
    ]:
        rec("summary_" + a, lambda a=a: getattr(s, a))
    rec("summary_predictions", lambda: s.predictions)
    for kind in ["deviance", "pearson", "working", "response"]:
        rec("summary_residuals_" + kind, lambda kind=kind: s.residuals(kind))
    rec("summary_residuals_default", lambda: s.residuals())
    rec("summary_residuals_bad", lambda: s.residuals("nope"))
    e = model.evaluate(df)
    rec("evaluate_type", lambda: type(e).__name__)
    rec("evaluate_deviance", lambda: e.deviance)
    rec("evaluate_residuals", lambda: e.residuals("working"))
    rec("evaluate_bad", lambda: model.evaluate(5))
    rec("transform_map", lambda: model.transform(df, {model.predictionCol: "q"}).columns)


def _live_aft(rec, est, model, df):
    from pyspark.ml.linalg import Vectors

    v = Vectors.dense([3.0, 1.0])
    for a in ["coefficients", "intercept", "numFeatures", "scale"]:
        rec(a, lambda a=a: getattr(model, a))
    rec("predict", lambda: model.predict(v))
    rec("predictQuantiles", lambda: model.predictQuantiles(v))
    rec("model_getQuantilesCol", lambda: model.getQuantilesCol())
    rec("model_getQuantileProbabilities", lambda: model.getQuantileProbabilities())
    rec("copy_quantilesCol", lambda: model.copy().setQuantilesCol("q").getQuantilesCol())
    rec("copy_quantileProbabilities", lambda: model.copy().setQuantileProbabilities(
        [0.5]).getQuantileProbabilities())
    rec("transform_quantiles", lambda: model.copy().setQuantilesCol("q").transform(df).columns)
    rec("predict_bad", lambda: model.predict("x"))


def _live_iso(rec, est, model, df):
    from pyspark.ml.linalg import Vectors

    rec("boundaries", lambda: model.boundaries)
    rec("predictions", lambda: model.predictions)
    rec("numFeatures", lambda: model.numFeatures)
    rec("predict_float", lambda: model.predict(3.0))
    rec("predict_int", lambda: model.predict(3))
    rec("predict_vector", lambda: model.predict(Vectors.dense([3.0, 1.0])))
    rec("model_getFeatureIndex", lambda: model.getFeatureIndex())
    rec("copy_featureIndex", lambda: model.copy().setFeatureIndex(1).getFeatureIndex())
    rec("transform_count", lambda: model.transform(df).count())


def _live_fm(rec, est, model, df):
    from pyspark.ml.linalg import Vectors

    v = Vectors.dense([3.0, 1.0])
    for a in ["factors", "intercept", "linear", "numFeatures"]:
        rec(a, lambda a=a: getattr(model, a))
    rec("predict", lambda: model.predict(v))
    rec("model_getFactorSize", lambda: model.getFactorSize())
    rec("model_getSeed", lambda: model.getSeed())
    rec("transform_count", lambda: model.transform(df).count())
    rec("copy_predictionCol", lambda: model.copy().setPredictionCol("p").getPredictionCol())


def _live_tree(rec, est, model, df):
    from pyspark.ml.linalg import Vectors

    v = Vectors.dense([3.0, 1.0])
    rec("depth", lambda: model.depth)
    rec("numNodes", lambda: model.numNodes)
    rec("model_getMaxDepth", lambda: model.getMaxDepth())
    rec("predict", lambda: model.predict(v))
    rec("predictLeaf", lambda: model.predictLeaf(v))
    rec("featureImportances", lambda: model.featureImportances)
    rec("toDebugString", lambda: model.toDebugString)
    rec("transform_count", lambda: model.transform(df).count())
    rec("copy_predictionCol", lambda: model.copy().setPredictionCol("p").getPredictionCol())


def _live_ensemble(rec, est, model, df):
    from pyspark.ml.linalg import Vectors

    v = Vectors.dense([3.0, 1.0])
    rec("numFeatures", lambda: model.numFeatures)
    rec("predict", lambda: model.predict(v))
    rec("predictLeaf", lambda: model.predictLeaf(v))
    rec("featureImportances", lambda: model.featureImportances)
    rec("toDebugString", lambda: model.toDebugString)
    rec("getNumTrees_again", lambda: model.getNumTrees)
    _ensemble_members(rec, model, df, v)
    rec("transform_count", lambda: model.transform(df).count())
    rec("copy_predictionCol", lambda: model.copy().setPredictionCol("p").getPredictionCol())


def _live_gbt(rec, est, model, df):
    _live_ensemble(rec, est, model, df)
    rec("evaluateEachIteration_squared", lambda: model.evaluateEachIteration(df, "squared"))
    rec("evaluateEachIteration_absolute", lambda: model.evaluateEachIteration(df, "absolute"))
    rec("evaluateEachIteration_bad_loss", lambda: model.evaluateEachIteration(df, "nope"))
    rec("evaluateEachIteration_bad_df", lambda: model.evaluateEachIteration(5, "squared"))


_LIVE = {
    "LinearRegression": _live_linear_regression,
    "GeneralizedLinearRegression": _live_glr,
    "AFTSurvivalRegression": _live_aft,
    "IsotonicRegression": _live_iso,
    "FMRegressor": _live_fm,
    "DecisionTreeRegressor": _live_tree,
    "RandomForestRegressor": _live_ensemble,
    "GBTRegressor": _live_gbt,
}


def live_cases(name, spark, base):
    """The fit, model, summary, evaluate, transform and persistence cases of the case set `name`,
    against a server; `base` is a scratch directory of the server's file system."""
    estimator, model_class = classes(name)
    rec = RegressionRecorder(base)
    df = _regression_frame(spark)
    est = estimator(**KWARGS[name])
    model = est.fit(df)
    class_param, value = CLASS_PARAM.get(name, ("maxIter", 3))
    model_cases(rec, est, model, df, class_param=class_param, value=value)
    _LIVE[name](rec, est, model, df)
    persistence_cases(rec, est, model, model_class, df, base, True)
    # The fits of a param map of the estimator's own int param (as `maxIter` for the others).
    int_param = OWN[name][0] if name in OWN else "maxIter"
    value = 1 if int_param == "featureIndex" else 2  # a valid index of the two features
    rec(
        "fit_map_own",
        lambda: est.fit(df, {getattr(est, int_param): value}).getOrDefault(int_param),
    )
    rec("fit_list_own", lambda: [
        m.getOrDefault(int_param)
        for m in est.fit(df, [{getattr(est, int_param): value}, {}])])
    return rec.out


def golden_path():
    return Path(__file__).resolve().parents[2] / "tests/golden/regression_api_cases.json"


def load_golden():
    return json.loads(golden_path().read_text())

