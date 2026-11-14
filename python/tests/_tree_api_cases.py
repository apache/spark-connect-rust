# SPDX-License-Identifier: Apache-2.0
"""The cases of the pyspark.ml API of the native tree classifiers: DecisionTreeClassifier,
RandomForestClassifier and GBTClassifier, and their models, summaries, trees, persistence and fits.

`scripts/capture_tree_api_cases_golden.py` runs them on official PySpark for
tests/golden/tree_api_cases.json. test_tree_api_offline.py (the Params API, without a server) and
test_tree_api_e2e.py (the models, summaries and persistence, against a live server) compare the
native classes with it. Official PySpark runs the tests too, as the reference
(`_native_classes.native_class`).

Each case records the repr of its value, or the type and the first line of the message of its
error, as `_ml_api_cases.Recorder` does.
"""

from _ml_api_cases import Recorder
from _native_classes import native_class


class _Rounded:
    """The rows of a DataFrame of a summary, with each float (and each value of a vector) rounded
    to 9 digits, as the Recorder rounds a float: the thresholds and probabilities of a curve differ
    in their last bits between platforms. The Recorder records it as it records a DataFrame."""

    def __init__(self, frame):
        def cell(x):
            if isinstance(x, float):
                return round(x, 9)
            if type(x).__name__ == "DenseVector":
                return type(x)([round(float(v), 9) for v in x])
            return x

        self.columns = frame.columns
        self._rows = [tuple(cell(x) for x in row) for row in frame.collect()]

    def collect(self):
        return self._rows


def _rounded_vector(vector):
    """A vector with each value rounded to 9 digits, as the Recorder records a vector of a value
    that differs in its last bits between platforms."""
    if type(vector).__name__ == "DenseVector":
        return type(vector)([round(float(v), 9) for v in vector])
    return vector

# The classes of each case set: (module of the native class, name, module of PySpark's).
_PYSPARK_ML = "pyspark.ml.classification"
CLASSES = {
    "DecisionTree": [
        ("pyspark.ml.connect.classification", "DecisionTreeClassifier", _PYSPARK_ML),
        ("pyspark.ml.connect.classification", "DecisionTreeClassificationModel", _PYSPARK_ML),
    ],
    "RandomForest": [
        ("pyspark.ml.connect.classification", "RandomForestClassifier", _PYSPARK_ML),
        ("pyspark.ml.connect.classification", "RandomForestClassificationModel", _PYSPARK_ML),
    ],
    "GBT": [
        ("pyspark.ml.connect.classification", "GBTClassifier", _PYSPARK_ML),
        ("pyspark.ml.connect.classification", "GBTClassificationModel", _PYSPARK_ML),
    ],
}


def classes(name):
    """The estimator and model classes of the case set `name`."""
    return [native_class(*spec) for spec in CLASSES[name]]


def _getter(name):
    return "get" + name[0].upper() + name[1:]


def params_api_cases(name):
    """The cases of PySpark's `Params` API of the estimator of the case set `name`, without a
    server."""
    estimator = classes(name)[0]
    rec = Recorder()
    est = estimator(maxDepth=4, featuresCol="f")
    other = estimator()
    rec("param_names", lambda: [p.name for p in est.params])
    rec("param_docs", lambda: [p.doc for p in est.params])
    rec("param_parents_are_uid", lambda: all(p.parent == est.uid for p in est.params))
    rec("explainParams", lambda: est.explainParams())
    rec("explainParam_str", lambda: est.explainParam("maxDepth"))
    rec("explainParam_param", lambda: est.explainParam(est.impurity))
    for p in est.params:
        getter = _getter(p.name)
        rec(getter, lambda getter=getter: getattr(est, getter)())
    rec("isSet_maxDepth", lambda: est.isSet("maxDepth"))
    rec("isSet_impurity", lambda: est.isSet(est.impurity))
    rec("isDefined_weightCol", lambda: est.isDefined(est.weightCol))
    rec("hasDefault_weightCol", lambda: est.hasDefault("weightCol"))
    rec("hasDefault_maxDepth", lambda: est.hasDefault(est.maxDepth))
    rec("hasParam_nope", lambda: est.hasParam("nope"))
    rec("hasParam_int", lambda: est.hasParam(1))
    rec("getParam_nope", lambda: est.getParam("nope"))
    rec("getParam_method", lambda: est.getParam("fit"))
    rec("getOrDefault_weightCol", lambda: est.getOrDefault(est.weightCol))
    rec("extract", lambda: sorted((p.name, repr(v)) for p, v in est.extractParamMap().items()))
    rec(
        "extract_extra",
        lambda: sorted(
            (p.name, repr(v)) for p, v in est.extractParamMap({est.weightCol: "w"}).items()
        ),
    )
    copy = est.copy({est.impurity: "entropy"})
    rec("copy_impurity", lambda: copy.getImpurity())
    rec("copy_same_uid", lambda: copy.uid == est.uid)
    rec("orig_impurity", lambda: est.getImpurity())
    rec("copy_explainParams", lambda: copy.explainParams())
    rec("copy_bad_extra", lambda: est.copy({"impurity": "gini"}))
    rec("copy_list", lambda: est.copy([1]))
    rec("set_ok", lambda: (est.set(est.featuresCol, "g"), est.getFeaturesCol())[1])
    rec("set_bad_type", lambda: est.set(est.featuresCol, 3))
    rec("set_str_param", lambda: est.set("featuresCol", "h"))
    rec(
        "clear",
        lambda: (est.clear(est.featuresCol), est.getFeaturesCol(), est.isSet("featuresCol")),
    )
    rec("isSet_foreign", lambda: est.isSet(other.featuresCol))
    rec("repr_is_uid", lambda: repr(est) == est.uid)
    rec("setThresholds", lambda: est.setThresholds([0.3, 0.7]).getThresholds())
    rec("setParams_chain", lambda: est.setParams(featuresCol="i").getFeaturesCol())
    rec("setter_chain", lambda: est.setFeaturesCol("j").getFeaturesCol())
    from pyspark.ml.base import Estimator
    from pyspark.ml.param import Params

    rec("isinstance_estimator", lambda: isinstance(est, Estimator))
    rec("isinstance_params", lambda: isinstance(est, Params))
    return rec.out


def _binary_frame(spark):
    from pyspark.ml.linalg import Vectors

    rows = [(float(x > 0), Vectors.dense([float(x), 1.0])) for x in (-4, -3, -2, -1, 1, 2, 3, 4)]
    return spark.createDataFrame(rows, ["label", "features"]).coalesce(1)


def _multiclass_frame(spark):
    from pyspark.ml.linalg import Vectors

    def label(x):
        return 0.0 if x < -2 else (1.0 if x <= 2 else 2.0)

    rows = [(label(x), Vectors.dense([float(x), 1.0])) for x in range(-6, 7)]
    return spark.createDataFrame(rows, ["label", "features"]).coalesce(1)


def _model_cases(rec, est, model, df):
    """The cases of any fitted tree model: its uid, params, members, predictions and transforms."""
    from pyspark.ml.linalg import Vectors

    v = Vectors.dense([3.0, 1.0])
    rec("model_uid_is_estimator_uid", lambda: model.uid == est.uid)
    rec("model_explainParams", lambda: model.explainParams())
    rec(
        "model_extract",
        lambda: sorted((p.name, repr(v)) for p, v in model.extractParamMap().items()),
    )
    rec("repr_prefix", lambda: repr(model).split(":")[0])
    rec("model_getMaxDepth", lambda: model.getMaxDepth())
    rec("model_isSet_maxDepth", lambda: model.isSet("maxDepth"))
    rec("numClasses", lambda: model.numClasses)
    rec("numFeatures", lambda: model.numFeatures)
    rec("featureImportances", lambda: _rounded_vector(model.featureImportances))
    rec("toDebugString", lambda: model.toDebugString)
    rec("predict", lambda: model.predict(v))
    rec("predictRaw", lambda: _rounded_vector(model.predictRaw(v)))
    rec("predictProbability", lambda: _rounded_vector(model.predictProbability(v)))
    rec("predictLeaf", lambda: _rounded_vector(model.predictLeaf(v)))
    rec("transform", lambda: _Rounded(model.transform(df).select("prediction")))
    rec("transform_map", lambda: model.transform(df, {model.predictionCol: "q"}).columns)
    rec("transform_bad_params", lambda: model.transform(df, 5))
    rec("model_setPredictionCol", lambda: model.copy().setPredictionCol("p").getPredictionCol())
    rec("model_setThresholds", lambda: model.copy().setThresholds([0.4, 0.6]).getThresholds())
    rec("model_setLeafCol", lambda: model.copy().setLeafCol("leaf").getLeafCol())
    rec("model_setFeaturesCol_chain", lambda: model.setFeaturesCol("features") is model)
    rec("copy_uid", lambda: model.copy().uid == model.uid)
    rec("has_summary", lambda: hasattr(model, "summary"))
    rec("has_hasSummary", lambda: hasattr(model, "hasSummary"))
    rec("supportedImpurities", lambda: model.supportedImpurities)
    # The native classes are instances of PySpark's ML base classes, as the vendored ones are.
    from pyspark.ml.base import Model, Transformer
    from pyspark.ml.param import Params

    rec("isinstance_model", lambda: isinstance(model, Model))
    rec("isinstance_transformer", lambda: isinstance(model, Transformer))
    rec("isinstance_params", lambda: isinstance(model, Params))
    rec("isinstance_estimator_is_not_model", lambda: isinstance(est, Model))


def _summary_cases(rec, model, summary, df, binary, tag=""):
    """The cases of a training summary `summary` of `model`: the binary one, or the multiclass one
    (whose case names have the prefix `tag`)."""
    props = [
        "accuracy",
        "labels",
        "precisionByLabel",
        "recallByLabel",
        "weightedPrecision",
        "weightedRecall",
        "labelCol",
        "predictionCol",
        "weightCol",
        "falsePositiveRateByLabel",
        "truePositiveRateByLabel",
        "weightedTruePositiveRate",
        "weightedFalsePositiveRate",
        "objectiveHistory",
        "totalIterations",
    ]
    if binary:
        props += ["areaUnderROC", "scoreCol"]
    for a in props:
        rec(tag + "summary_" + a, lambda a=a: getattr(summary, a))
    rec(tag + "summary_fMeasureByLabel", lambda: summary.fMeasureByLabel())
    rec(tag + "summary_fMeasureByLabel_half", lambda: summary.fMeasureByLabel(0.5))
    rec(tag + "summary_weightedFMeasure_keyword", lambda: summary.weightedFMeasure(beta=0.5))
    rec(tag + "summary_weightedFMeasure", lambda: summary.weightedFMeasure())
    rec(tag + "summary_is_summary", lambda: model.summary is summary)
    rec(tag + "copy_summary_is_summary", lambda: model.copy().summary is summary)
    rec(tag + "summary_df_predictions", lambda: _Rounded(summary.predictions.select("prediction")))
    if binary:
        for a in ["roc", "pr", "fMeasureByThreshold", "precisionByThreshold"]:
            rec(tag + "summary_df_" + a, lambda a=a: _Rounded(getattr(summary, a)))
        rec(tag + "summary_df_recallByThreshold", lambda: _Rounded(summary.recallByThreshold))
    e = model.evaluate(df)
    rec(tag + "evaluate_type", lambda: type(e).__name__)
    rec(tag + "evaluate_accuracy", lambda: e.accuracy)
    rec(tag + "evaluate_labels", lambda: e.labels)
    rec(tag + "evaluate_predictions", lambda: _Rounded(e.predictions.select("prediction")))
    if binary:
        rec(tag + "evaluate_areaUnderROC", lambda: e.areaUnderROC)
        rec(tag + "evaluate_roc", lambda: _Rounded(e.roc))
    rec(tag + "evaluate_bad", lambda: model.evaluate(5))


def _rf_summary_cases(rec, model, df):
    s = model.summary
    rec("hasSummary", lambda: model.hasSummary)
    rec("summary_type", lambda: type(s).__name__)
    _summary_cases(rec, model, s, df, binary=True)


def live_cases(name, spark, base):
    """The model, summary, tree, transform and persistence cases of the case set `name`, against a
    server; `base` is a scratch directory of the server's file system."""
    from pyspark.ml.linalg import Vectors

    estimator, model_class = classes(name)
    rec = Recorder(base)
    df = _binary_frame(spark)
    multi_df = _multiclass_frame(spark)
    if name == "DecisionTree":
        est = estimator(maxDepth=3, seed=42)
    elif name == "RandomForest":
        est = estimator(numTrees=3, maxDepth=3, seed=42)
    else:
        est = estimator(maxIter=3, maxDepth=2, seed=42)
    model = est.fit(df)
    _model_cases(rec, est, model, df)
    v = Vectors.dense([3.0, 1.0])
    if name == "DecisionTree":
        rec("depth", lambda: model.depth)
        rec("numNodes", lambda: model.numNodes)
        rec("summary_absent", lambda: model.summary)
    else:
        rec("getNumTrees", lambda: model.getNumTrees)
        rec("numTrees", lambda: model.numTrees)
        rec("treeWeights", lambda: model.treeWeights)
        rec("totalNumNodes", lambda: model.totalNumNodes)
        rec("trees_count", lambda: len(model.trees))
        rec("trees_types", lambda: [type(t).__name__ for t in model.trees])
        rec("trees_numNodes", lambda: [t.numNodes for t in model.trees])
        rec("trees_depth", lambda: [t.depth for t in model.trees])
        rec("trees_predict", lambda: model.trees[0].predict(v))
        rec("trees_identity", lambda: model.trees is model.trees)
        rec("trees_copy_shared", lambda: model.copy().trees is model.trees)
        rec("supportedFeatureSubsetStrategies", lambda: model.supportedFeatureSubsetStrategies)
    if name == "RandomForest":
        _rf_summary_cases(rec, model, df)
        m3 = estimator(numTrees=3, maxDepth=3, seed=42).fit(multi_df)
        rec("multi_summary_type", lambda: type(m3.summary).__name__)
        rec("multi_summary_has_areaUnderROC", lambda: hasattr(m3.summary, "areaUnderROC"))
        _summary_cases(rec, m3, m3.summary, multi_df, binary=False, tag="multi_")
    if name == "GBT":
        rec("evaluateEachIteration", lambda: model.evaluateEachIteration(df))
        rec("predictLeaf_gbt", lambda: model.predictLeaf(v))
        rec("supportedLossTypes", lambda: model.supportedLossTypes)
    _persistence_cases(rec, est, model, model_class, df, base, name)
    return rec.out


def _persistence_cases(rec, est, model, model_class, df, base, name):
    """The cases of the save and load of the estimator and the model, and the fits of param maps."""
    import shutil

    rec("transform_saved", lambda: _Rounded(model.transform(df).select("prediction")))
    shutil.rmtree(base, ignore_errors=True)
    rec("model_save", lambda: model.write().overwrite().save(base + "/model"))
    rec("model_save_again", lambda: model.save(base + "/model"))
    loaded = model_class.load(base + "/model")
    rec("loaded_uid", lambda: loaded.uid == model.uid)
    rec("loaded_getMaxDepth", lambda: loaded.getMaxDepth())
    rec("loaded_isSet_maxDepth", lambda: loaded.isSet("maxDepth"))
    rec("loaded_explainParams", lambda: loaded.explainParams())
    rec("loaded_transform", lambda: _Rounded(loaded.transform(df).select("prediction")))
    rec("loaded_predict", lambda: loaded.predict(_probe()))
    if name == "RandomForest":
        rec("loaded_hasSummary", lambda: loaded.hasSummary)
        rec("loaded_summary", lambda: loaded.summary)
        rec("loaded_getNumTrees", lambda: loaded.getNumTrees)
        rec("loaded_trees_count", lambda: len(loaded.trees))
        rec("loaded_trees_types", lambda: [type(t).__name__ for t in loaded.trees])
    if name == "GBT":
        rec("loaded_getNumTrees", lambda: loaded.getNumTrees)
        rec("loaded_trees_count", lambda: len(loaded.trees))
    rec("est_save", lambda: est.write().overwrite().save(base + "/est"))
    loaded_est = type(est).load(base + "/est")
    rec("est_loaded_uid", lambda: loaded_est.uid == est.uid)
    rec(
        "est_loaded_params",
        lambda: sorted((p.name, repr(v)) for p, v in loaded_est.extractParamMap().items()),
    )
    rec("est_loaded_isSet_maxDepth", lambda: loaded_est.isSet("maxDepth"))
    rec(
        "fitMultiple",
        lambda: sorted(
            (i, m.getMaxDepth())
            for i, m in est.fitMultiple(df, [{est.maxDepth: 2}, {est.maxDepth: 1}])
        ),
    )
    rec("fit_map", lambda: est.fit(df, {est.maxDepth: 2}).getMaxDepth())
    rec("fit_list", lambda: [m.getMaxDepth() for m in est.fit(df, [{est.maxDepth: 2}, {}])])
    rec("fit_bad_params", lambda: est.fit(df, 5))
    rec("fit_bad_dataset", lambda: est.fit(5))
    shutil.rmtree(base, ignore_errors=True)


def _probe():
    from pyspark.ml.linalg import Vectors

    return Vectors.dense([3.0, 1.0])
