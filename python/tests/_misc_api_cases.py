# SPDX-License-Identifier: Apache-2.0
"""The cases of the pyspark.ml API of NaiveBayes, MultilayerPerceptronClassifier, FMClassifier
and OneVsRest (and the models and summaries of the first three): `scripts/capture_misc_api_cases_
golden.py` runs them on official PySpark for tests/golden/misc_api_cases.json, and
test_misc_api_offline.py and test_misc_api_e2e.py run them on the native classes and compare.
Official PySpark runs the tests too, as the reference (`_native_classes.native_class`).

The Recorder and the normalization are those of `_ml_api_cases`. Each case set has its own
params (a float param for the set and copy cases, a str param for the weight), and the persistence
of OneVsRest is not in it: the server has no OneVsRest writer, which official PySpark does not
need, so its cases are the params, the fits and the transforms only.
"""

import json
import os
import shutil

from _ml_api_cases import Recorder
from _native_classes import native_class

CLASSES = {
    "NaiveBayes": [
        ("pyspark.ml.connect.classification", "NaiveBayes", "pyspark.ml.classification"),
        ("pyspark.ml.connect.classification", "NaiveBayesModel", "pyspark.ml.classification"),
    ],
    "MultilayerPerceptronClassifier": [
        (
            "pyspark.ml.connect.classification",
            "MultilayerPerceptronClassifier",
            "pyspark.ml.classification",
        ),
        (
            "pyspark.ml.connect.classification",
            "MultilayerPerceptronClassificationModel",
            "pyspark.ml.classification",
        ),
    ],
    "FMClassifier": [
        ("pyspark.ml.connect.classification", "FMClassifier", "pyspark.ml.classification"),
        ("pyspark.ml.connect.classification", "FMClassificationModel", "pyspark.ml.classification"),
    ],
    "OneVsRest": [
        ("pyspark.ml.connect.classification", "OneVsRest", "pyspark.ml.classification"),
        ("pyspark.ml.connect.classification", "OneVsRestModel", "pyspark.ml.classification"),
    ],
}

# The estimator of each case set with its params: the float param of the param map cases, the
# value it takes, the str param of the weight cases, and the value that the setter rejects.
SPECS = {
    "NaiveBayes": {"param": "smoothing", "value": 2.5, "weight": "weightCol", "bad": "x"},
    "MultilayerPerceptronClassifier": {"param": "tol", "value": 0.5, "weight": "seed", "bad": "x"},
    "FMClassifier": {"param": "regParam", "value": 0.25, "weight": "solver", "bad": "x"},
    "OneVsRest": {"param": "parallelism", "value": 2, "weight": "weightCol", "bad": "x"},
}


def classes(name):
    """The estimator and model classes of the case set `name`."""
    return [native_class(*spec) for spec in CLASSES[name]]


def _cap(name):
    return name[0].upper() + name[1:]


def _estimator(name):
    """The estimator of the case set `name`, with the params of its offline and live cases."""
    estimator = classes(name)[0]
    if name == "NaiveBayes":
        return estimator(smoothing=1.0, weightCol="weight")
    if name == "MultilayerPerceptronClassifier":
        return estimator(maxIter=5, layers=[2, 4, 2], seed=1)
    if name == "FMClassifier":
        return estimator(maxIter=5, seed=1, factorSize=2)
    linear_svc = native_class(
        "pyspark.ml.connect.classification", "LinearSVC", "pyspark.ml.classification"
    )
    return estimator(classifier=linear_svc(maxIter=5, regParam=0.01))


def params_cases(rec, est, other, spec):
    """The cases of PySpark's `Params` API of the estimator `est`, of its param `spec["param"]`
    (a float) and its weight param (a str), with `other` a default estimator of the same class.

    """
    param, value, weight, bad = spec["param"], spec["value"], spec["weight"], spec["bad"]
    getter = "get" + _cap(param)
    rec("param_names", lambda: [p.name for p in est.params])
    rec("param_docs", lambda: [p.doc for p in est.params])
    rec("param_parents_are_uid", lambda: all(p.parent == est.uid for p in est.params))
    rec("explainParams", lambda: est.explainParams())
    rec("explainParam_str", lambda: est.explainParam(param))
    rec("explainParam_param", lambda: est.explainParam(getattr(est, param)))
    for p in est.params:
        rec("get_" + p.name, lambda p=p: getattr(est, "get" + _cap(p.name))())
    rec("isSet_param", lambda: est.isSet(param))
    rec("isDefined_weight", lambda: est.isDefined(getattr(est, weight)))
    rec("hasDefault_weight", lambda: est.hasDefault(weight))
    rec("hasDefault_param", lambda: est.hasDefault(getattr(est, param)))
    rec("hasParam_param", lambda: est.hasParam(param))
    rec("hasParam_nope", lambda: est.hasParam("nope"))
    rec("hasParam_int", lambda: est.hasParam(1))
    rec("getParam_nope", lambda: est.getParam("nope"))
    rec("getParam_method", lambda: est.getParam("fit"))
    rec("isSet_nope", lambda: est.isSet("nope"))
    rec("getOrDefault_param", lambda: est.getOrDefault(param))
    rec("getOrDefault_weight", lambda: est.getOrDefault(getattr(est, weight)))
    rec("extract", lambda: sorted((p.name, repr(v)) for p, v in est.extractParamMap().items()))
    rec(
        "extract_extra",
        lambda: sorted(
            (p.name, repr(v)) for p, v in est.extractParamMap({getattr(est, param): value}).items()
        ),
    )
    copy = est.copy({getattr(est, param): value})
    rec("copy_value", lambda: getattr(copy, getter)())
    rec("copy_same_uid", lambda: copy.uid == est.uid)
    rec("orig_value", lambda: getattr(est, getter)())
    rec("copy_explainParams", lambda: copy.explainParams())
    rec("copy_bad_extra", lambda: est.copy({param: value}))
    rec("copy_list", lambda: est.copy([1]))
    rec("set_ok", lambda: (est.set(getattr(est, param), value), getattr(est, getter)())[1])
    rec("set_bad_type", lambda: est.set(getattr(est, param), bad))
    rec("set_str_param", lambda: est.set(param, value))
    rec("clear", lambda: (est.clear(getattr(est, param)), est.isSet(param)))
    rec("isSet_foreign", lambda: est.isSet(getattr(other, param)))
    rec("repr_is_uid", lambda: repr(est) == est.uid)


def params_api_cases(name):
    """The cases of PySpark's `Params` API of the estimator of the case set `name`, without a
    server."""
    spec = SPECS[name]
    estimator = classes(name)[0]
    est = _estimator(name)
    rec = Recorder()
    params_cases(rec, est, estimator(), spec)
    if name == "NaiveBayes":
        rec("thresholds_unset", lambda: estimator().getThresholds())
        rec("modelType_set", lambda: estimator().setModelType("bernoulli").getModelType())
    elif name == "MultilayerPerceptronClassifier":
        rec("layers_set", lambda: estimator().setLayers([3, 2]).getLayers())
        rec("seed_set", lambda: estimator().setSeed(7).getSeed())
    elif name == "FMClassifier":
        rec("factorSize_set", lambda: estimator().setFactorSize(3).getFactorSize())
        rec("weightCol_no_setter", lambda: hasattr(estimator(), "setWeightCol"))
    else:
        rec("classifier_none", lambda: estimator().setClassifier(None).isSet("classifier"))
    return rec.out


def _misc_model_cases(rec, est, model):
    """The cases of a fitted model `model` of the estimator `est`: its uid, its class, its params and
    its description. (The shared `model_cases` reads a global that another case set reads, so this
    module has its own.)"""
    from pyspark.ml.base import Model, Transformer

    rec("model_uid_is_estimator_uid", lambda: model.uid == est.uid)
    rec("isinstance_model", lambda: (isinstance(model, Model), isinstance(model, Transformer)))
    rec("model_no_attribute", lambda: model.fooBarBaz)
    rec("model_class", lambda: (type(model).__name__, type(model).__module__.split(".")[-1]))
    rec("model_explainParams", lambda: model.explainParams())
    rec(
        "model_extract",
        lambda: sorted((p.name, repr(v)) for p, v in model.extractParamMap().items()),
    )
    rec("repr_prefix", lambda: repr(model).split(":")[0])


def _binary_frame(spark):
    from pyspark.ml.linalg import Vectors

    rows = [
        (0.0, 0.1, Vectors.dense([0.0, 0.0])),
        (0.0, 0.5, Vectors.dense([0.0, 1.0])),
        (1.0, 1.0, Vectors.dense([1.0, 0.0])),
        (1.0, 0.3, Vectors.dense([1.0, 1.0])),
    ]
    return spark.createDataFrame(rows, ["label", "weight", "features"]).coalesce(1)


def _multi_frame(spark):
    from pyspark.ml.linalg import Vectors

    rows = [
        (0.0, Vectors.dense([0.0, 0.0])),
        (0.0, Vectors.dense([0.1, 0.2])),
        (1.0, Vectors.dense([1.0, 1.0])),
        (1.0, Vectors.dense([1.2, 0.9])),
        (2.0, Vectors.dense([2.0, 0.0])),
        (2.0, Vectors.dense([2.1, 0.2])),
    ]
    return spark.createDataFrame(rows, ["label", "features"]).coalesce(1)


def _persistence_cases(rec, est, model, model_class, df, base, spec, has_summary):
    """The cases of the save and load of `est` and `model`, and the fits of param maps."""
    param = spec["param"]
    getter = "get" + _cap(param)
    rec("transform", lambda: model.transform(df))
    shutil.rmtree(base, ignore_errors=True)
    rec("model_save", lambda: model.write().overwrite().save(base + "/model"))
    rec("model_save_again", lambda: model.save(base + "/model"))
    loaded = model_class.load(base + "/model")
    rec("loaded_uid", lambda: loaded.uid == model.uid)
    if has_summary:
        rec("loaded_hasSummary", lambda: loaded.hasSummary)
        rec("loaded_summary", lambda: loaded.summary())
    rec("loaded_transform", lambda: loaded.transform(df))
    rec("loaded_" + param, lambda: getattr(loaded, getter)())
    rec("est_save", lambda: est.write().overwrite().save(base + "/est"))
    loaded_est = type(est).load(base + "/est")
    rec("est_loaded_uid", lambda: loaded_est.uid == est.uid)
    rec(
        "est_loaded_params",
        lambda: sorted((p.name, repr(v)) for p, v in loaded_est.extractParamMap().items()),
    )
    rec(
        "fitMultiple",
        lambda: sorted(
            (i, getattr(m, getter)())
            for i, m in est.fitMultiple(df, [{getattr(est, param): 2.0}, {}])
        ),
    )
    rec("fit_map", lambda: getattr(est.fit(df, {getattr(est, param): 2.0}), getter)())
    rec("fit_list", lambda: [getattr(m, getter)() for m in est.fit(df, [{}, {}])])
    shutil.rmtree(base, ignore_errors=True)


def _naive_bayes_live(rec, spark, base):
    from pyspark.ml.linalg import Vectors

    estimator, model_class = classes("NaiveBayes")
    df = _binary_frame(spark)
    est = _estimator("NaiveBayes")
    model = est.fit(df)
    _misc_model_cases(rec, est, model)
    for a in ["pi", "theta", "sigma", "numClasses", "numFeatures"]:
        rec(a, lambda a=a: getattr(model, a))
    v = Vectors.dense([1.0, 0.0])
    rec("predict", lambda: model.predict(v))
    rec("predictRaw", lambda: model.predictRaw(v))
    rec("predictProbability", lambda: model.predictProbability(v))
    rec("model_getSmoothing", lambda: model.getSmoothing())
    rec("model_getModelType", lambda: model.getModelType())
    rec("model_isSet_smoothing", lambda: model.isSet("smoothing"))
    rec("model_getThresholds", lambda: model.getThresholds())
    rec("model_hasSummary", lambda: hasattr(model, "hasSummary"))
    rec("transform_df", lambda: model.transform(df))
    rec("copy_smoothing", lambda: model.copy({model.smoothing: 3.0}).getSmoothing())
    rec("model_setThresholds", lambda: model.copy().setThresholds([0.5, 0.5]).getThresholds())
    rec("model_setPredictionCol", lambda: model.copy().setPredictionCol("p").getPredictionCol())
    rec("transform_map", lambda: model.transform(df, {model.predictionCol: "q"}).columns)
    rec("transform_map_empty", lambda: model.transform(df, {}).columns)
    _persistence_cases(rec, est, model, model_class, df, base, SPECS["NaiveBayes"], False)


def _mlp_live(rec, spark, base):
    from pyspark.ml.linalg import Vectors

    estimator, model_class = classes("MultilayerPerceptronClassifier")
    df = _binary_frame(spark).drop("weight")
    est = _estimator("MultilayerPerceptronClassifier")
    model = est.fit(df)
    _misc_model_cases(rec, est, model)
    for a in ["numClasses", "numFeatures", "hasSummary"]:
        rec(a, lambda a=a: getattr(model, a))
    rec("weights_size", lambda: model.weights.size)
    rec("weights", lambda: model.weights)
    v = Vectors.dense([1.0, 0.0])
    rec("predict", lambda: model.predict(v))
    rec("predictRaw", lambda: model.predictRaw(v))
    rec("predictProbability", lambda: model.predictProbability(v))
    rec("model_getLayers", lambda: model.getLayers())
    rec("model_getMaxIter", lambda: model.getMaxIter())
    rec("model_isSet_maxIter", lambda: model.isSet("maxIter"))
    s = model.summary()
    rec("summary_is_summary", lambda: model.summary() is s)
    for a in [
        "accuracy",
        "labels",
        "objectiveHistory",
        "totalIterations",
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
    ]:
        rec("summary_" + a, lambda a=a: getattr(s, a))
    rec("summary_fMeasureByLabel", lambda: s.fMeasureByLabel())
    rec("summary_fMeasureByLabel_half", lambda: s.fMeasureByLabel(0.5))
    rec("summary_weightedFMeasure", lambda: s.weightedFMeasure())
    rec("summary_weightedFMeasure_keyword", lambda: s.weightedFMeasure(beta=0.5))
    rec("summary_df_predictions", lambda: s.predictions.select("prediction"))
    e = model.evaluate(df)
    rec("evaluate_accuracy", lambda: e.accuracy)
    rec("evaluate_labels", lambda: e.labels)
    rec("evaluate_fMeasureByLabel", lambda: e.fMeasureByLabel(2.0))
    rec("evaluate_df_predictions", lambda: e.predictions.select("prediction"))
    rec("evaluate_bad", lambda: model.evaluate(5))
    rec("transform_map", lambda: model.transform(df, {model.predictionCol: "q"}).columns)
    rec("copy_maxIter", lambda: model.copy({model.maxIter: 9}).getMaxIter())
    rec("model_setThresholds", lambda: model.copy().setThresholds([0.5, 0.5]).getThresholds())
    _persistence_cases(
        rec, est, model, model_class, df, base, SPECS["MultilayerPerceptronClassifier"], True
    )


def _fm_live(rec, spark, base):
    from pyspark.ml.linalg import Vectors

    estimator, model_class = classes("FMClassifier")
    df = _binary_frame(spark).drop("weight")
    est = _estimator("FMClassifier")
    model = est.fit(df)
    _misc_model_cases(rec, est, model)
    for a in ["factors", "linear", "intercept", "numClasses", "numFeatures", "hasSummary"]:
        rec(a, lambda a=a: getattr(model, a))
    v = Vectors.dense([1.0, 0.0])
    rec("predict", lambda: model.predict(v))
    rec("predictRaw", lambda: model.predictRaw(v))
    rec("predictProbability", lambda: model.predictProbability(v))
    rec("model_getFactorSize", lambda: model.getFactorSize())
    rec("model_getSolver", lambda: model.getSolver())
    s = model.summary()
    rec("summary_is_summary", lambda: model.summary() is s)
    for a in [
        "accuracy",
        "areaUnderROC",
        "labels",
        "objectiveHistory",
        "totalIterations",
        "precisionByLabel",
        "recallByLabel",
        "weightedPrecision",
        "weightedRecall",
        "labelCol",
        "predictionCol",
        "scoreCol",
        "weightCol",
        "falsePositiveRateByLabel",
        "truePositiveRateByLabel",
        "weightedTruePositiveRate",
        "weightedFalsePositiveRate",
    ]:
        rec("summary_" + a, lambda a=a: getattr(s, a))
    rec("summary_fMeasureByLabel", lambda: s.fMeasureByLabel())
    rec("summary_weightedFMeasure", lambda: s.weightedFMeasure(beta=0.5))
    for a in ["roc", "pr", "predictions", "fMeasureByThreshold", "precisionByThreshold"]:
        rec("summary_df_" + a, lambda a=a: getattr(s, a).count())
    e = model.evaluate(df)
    rec("evaluate_areaUnderROC", lambda: e.areaUnderROC)
    rec("evaluate_accuracy", lambda: e.accuracy)
    rec("evaluate_bad", lambda: model.evaluate(5))
    rec("transform_map", lambda: model.transform(df, {model.predictionCol: "q"}).columns)
    rec("copy_factorSize", lambda: model.copy({model.factorSize: 4}).getFactorSize())
    _persistence_cases(rec, est, model, model_class, df, base, SPECS["FMClassifier"], True)


def _unset_one_vs_rest():
    """The OneVsRest of no classifier, whose fit raises PySpark's KeyError of the classifier."""
    return classes("OneVsRest")[0]()


def _native(name):
    """The native class of PySpark's classification module of the name `name`."""
    return native_class("pyspark.ml.connect.classification", name, "pyspark.ml.classification")


def _summary_of(model):
    """The training summary of a model: a method of most classes, and a property of LR."""
    summary = model.summary
    return summary() if callable(summary) else summary


def _submodel_cases(rec, prefix, model, classifier):
    """The cases of the `models` of a fitted OneVsRestModel `model`, of the classifier `classifier`:
    their classes, uids, label columns, params and summaries."""
    rec(prefix + "models_len", lambda: len(model.models))
    rec(prefix + "models_same_list", lambda: model.models is model.models)
    rec(
        prefix + "models_types",
        lambda: [(type(s).__name__, type(s).__module__.split(".")[-1]) for s in model.models],
    )
    rec(
        prefix + "models_uids_are_classifier",
        lambda: [s.uid == classifier.uid for s in model.models],
    )
    rec(prefix + "models_labelCols", lambda: [s.getLabelCol() for s in model.models])
    rec(prefix + "models_isSet_labelCol", lambda: [s.isSet("labelCol") for s in model.models])
    rec(
        prefix + "models_params",
        lambda: [
            sorted((p.name, repr(v)) for p, v in s.extractParamMap().items()) for s in model.models
        ],
    )
    rec(
        prefix + "models_hasSummary",
        lambda: [getattr(s, "hasSummary", None) for s in model.models],
    )
    rec(
        prefix + "models_summary_accuracy",
        lambda: [
            _summary_of(s).accuracy for s in model.models if getattr(s, "hasSummary", False)
        ],
    )


def _one_vs_rest_live(rec, spark, base):
    df = _multi_frame(spark)
    est = _estimator("OneVsRest")
    rec("getClassifier_type", lambda: type(est.getClassifier()).__name__)
    rec("getClassifier_maxIter", lambda: est.getClassifier().getMaxIter())
    rec("copy_classifier_type", lambda: type(est.copy().getClassifier()).__name__)
    rec("copy_parallelism", lambda: est.copy({est.parallelism: 1}).getParallelism())
    rec("isSet_classifier", lambda: est.isSet("classifier"))
    model = est.fit(df)
    _misc_model_cases(rec, est, model)
    _submodel_cases(rec, "ovr_linear_svc_", model, est.getClassifier())
    for prefix, name, kwargs in [
        ("ovr_nb_", "NaiveBayes", {"smoothing": 1.0}),
        ("ovr_fm_", "FMClassifier", {"maxIter": 5, "seed": 1, "factorSize": 2}),
        (
            "ovr_mlp_",
            "MultilayerPerceptronClassifier",
            {"maxIter": 5, "layers": [2, 4, 2], "seed": 1},
        ),
        ("ovr_lr_", "LogisticRegression", {"maxIter": 5, "regParam": 0.01}),
        ("ovr_dt_", "DecisionTreeClassifier", {"maxDepth": 3, "seed": 1}),
        ("ovr_rf_", "RandomForestClassifier", {"numTrees": 3, "maxDepth": 3, "seed": 1}),
        ("ovr_gbt_", "GBTClassifier", {"maxIter": 3, "maxDepth": 2, "seed": 1}),
    ]:
        classifier = _native(name)(**kwargs)
        fitted = _native("OneVsRest")(classifier=classifier).fit(df)
        _submodel_cases(rec, prefix, fitted, classifier)
    rec("model_prefix", lambda: model.uid.split("_")[0])
    rec(
        "model_extract_classifier",
        lambda: repr(model.getClassifier()) == repr(est.getClassifier()),
    )
    rec("model_getFeaturesCol", lambda: model.getFeaturesCol())
    rec("model_isSet_parallelism", lambda: model.isSet("parallelism"))
    # A copy of the model has copies of its models, each a copy with the extra params
    # (`model.copy(extra)`), of which those of the classifier's uid, the models' own, set theirs.
    classifier = est.getClassifier()
    rec(
        "model_copy_models",
        lambda: [
            (type(s).__name__, s is t, s.uid == t.uid)
            for s, t in zip(model.copy().models, model.models)
        ],
    )
    rec("model_copy_models_same_list", lambda: (lambda c: c.models is c.models)(model.copy()))
    rec(
        "model_copy_classifier_param",
        lambda: [s.getPredictionCol() for s in model.copy({classifier.predictionCol: "x"}).models],
    )
    rec(
        "model_copy_classifier_param_own",
        lambda: model.copy({classifier.predictionCol: "x"}).getPredictionCol(),
    )
    rec(
        "model_copy_own_param",
        lambda: [s.getPredictionCol() for s in model.copy({model.predictionCol: "y"}).models],
    )
    rec(
        "model_copy_own_param_own",
        lambda: model.copy({model.predictionCol: "y"}).getPredictionCol(),
    )
    rec("model_transform_map", lambda: model.transform(df, {model.predictionCol: "q"}).columns)
    rec("transform_columns", lambda: model.transform(df).columns)
    rec("transform", lambda: model.transform(df))
    rec("fit_map", lambda: est.fit(df, {est.parallelism: 1}).transform(df))
    rec(
        "fitMultiple",
        lambda: sorted(
            (i, m.transform(df).count())
            for i, m in est.fitMultiple(df, [{est.parallelism: 2}, {}])
        ),
    )
    rec("fit_list", lambda: [m.transform(df).count() for m in est.fit(df, [{}, {}])])
    rec("fit_without_classifier", lambda: _unset_one_vs_rest().fit(df))
    _one_vs_rest_parity_cases(rec, est, model, df)
    _one_vs_rest_persistence(rec, est, model, df, base)
    _one_vs_rest_plain_classifier_cases(rec, df, base)


def _metadata(path):
    """The metadata of the save at `path`, without its timestamp, and with the class name of its
    uid (the uid itself is random)."""
    import glob

    text = open(glob.glob(path + "/metadata/part-*")[0]).read()
    metadata = json.loads(text)
    metadata.pop("timestamp")
    metadata["uid"] = metadata["uid"].split("_")[0]
    return metadata


def _params_of(obj):
    """The params of `obj` and their values, without the classifier (whose repr has a uid)."""
    return sorted(
        (p.name, repr(v)) for p, v in obj.extractParamMap().items() if p.name != "classifier"
    )


def _weight_warnings(fit):
    """The categories and messages of the warnings about a weightCol that `fit` raises."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit()
    return [
        (w.category.__name__, str(w.message)) for w in caught if "weightCol" in str(w.message)
    ]


def _assigned_copy(model, count):
    """A copy of `model` whose `models` are assigned the first `count` of its models."""
    copied = model.copy()
    copied.models = copied.models[:count]
    return copied


def _assigned_copy_leaves_original(model):
    """The length of the models of `model` after an assignment of the models of its copy."""
    _assigned_copy(model, 2)
    return len(model.models)


def _one_vs_rest_parity_cases(rec, est, model, df):
    """The cases of the OneVsRestModel as PySpark's: its transform, which composes the transforms
    of its models, not its own columns; its constructor, of models; its `models` attribute, which
    is assigned; and the weightCol of the classes, which a classifier that is a HasWeightCol takes
    (FMClassifier does, the MultilayerPerceptronClassifier does not, and warns of it)."""
    from pyspark.sql import functions as F

    classifier = est.getClassifier()
    # The transform reads the featuresCol and the rawPredictionCol of the models, never the model's.
    rec(
        "copy_setFeaturesCol_transform_columns",
        lambda: model.copy().setFeaturesCol("f2").transform(df).columns,
    )
    rec("copy_setFeaturesCol_transform", lambda: model.copy().setFeaturesCol("f2").transform(df))
    rec(
        "copy_setRawPredictionCol_rq_transform_columns",
        lambda: model.copy().setRawPredictionCol("rq").transform(df).columns,
    )
    rec(
        "map_rawPredictionCol_rq_transform_columns",
        lambda: model.transform(df, {model.rawPredictionCol: "rq"}).columns,
    )
    rec(
        "copy_setRawPredictionCol_empty_transform_columns",
        lambda: model.copy().setRawPredictionCol("").transform(df).columns,
    )
    raw_fit = _native("OneVsRest")(classifier=classifier, rawPredictionCol="rp").fit(df)
    rec("estimator_rawPredictionCol_rp_model_raw", lambda: raw_fit.getRawPredictionCol())
    rec(
        "estimator_rawPredictionCol_rp_transform_columns",
        lambda: raw_fit.transform(df).columns,
    )
    # The constructor of the models, as PySpark's: a new uid, the defaults, no set params, and no
    # classifier, by a keyword or a positional argument.
    cls = _native("OneVsRestModel")
    rec("ctor_keyword_uid_prefix", lambda: cls(models=model.models).uid.split("_")[0])
    rec(
        "ctor_keyword_uid_suffix_length",
        lambda: len(cls(models=model.models).uid.split("_")[1]),
    )
    rec(
        "ctor_keyword_isSet",
        lambda: [
            cls(models=model.models).isSet(name)
            for name in ["featuresCol", "labelCol", "predictionCol", "rawPredictionCol"]
        ],
    )
    rec("ctor_keyword_params", lambda: _params_of(cls(models=model.models)))
    rec(
        "ctor_keyword_isSet_classifier",
        lambda: cls(models=model.models).isSet("classifier"),
    )
    rec(
        "ctor_keyword_transform_columns",
        lambda: cls(models=model.models).transform(df).columns,
    )
    rec("ctor_keyword_transform", lambda: cls(models=model.models).transform(df))
    rec("ctor_positional_uid_prefix", lambda: cls(model.models).uid.split("_")[0])
    rec("ctor_positional_models_len", lambda: len(cls(model.models).models))
    rec(
        "ctor_positional_transform_columns",
        lambda: cls(model.models).transform(df).columns,
    )
    rec("ctor_positional_transform", lambda: cls(model.models).transform(df))
    # The `models` attribute is assigned, as PySpark's plain attribute; an assignment to the models
    # of a copy leaves the original's models as they were.
    rec("models_assign_type", lambda: type(_assigned_copy(model, 2).models).__name__)
    rec("models_assign_len", lambda: len(_assigned_copy(model, 2).models))
    rec("models_assign_transform_columns", lambda: _assigned_copy(model, 2).transform(df).columns)
    rec("models_assign_transform", lambda: _assigned_copy(model, 2).transform(df))
    rec("models_assign_copy_leaves_original", lambda: _assigned_copy_leaves_original(model))
    # The weightCol of the classes: a HasWeightCol takes it, and the others warn of it by their str.
    weighted = df.withColumn("w", F.lit(1.0))

    def weighted_fit(classifier):
        return _native("OneVsRest")(classifier=classifier, weightCol="w").fit(weighted)

    mlp = _native("MultilayerPerceptronClassifier")(layers=[2, 4, 2], maxIter=2, seed=1)
    rec("mlp_weight_warnings", lambda: _weight_warnings(lambda: weighted_fit(mlp)))
    fm = _native("FMClassifier")(maxIter=2, seed=1, factorSize=2)
    rec("fm_weight_warnings", lambda: _weight_warnings(lambda: weighted_fit(fm)))
    rec(
        "fm_weight_submodel_weightCols",
        lambda: [
            s.getWeightCol() if s.isDefined(s.weightCol) else None
            for s in weighted_fit(fm).models
        ],
    )


def _one_vs_rest_persistence(rec, est, model, df, base):
    """The cases of the save and load of the OneVsRest `est` and of its fitted `model`, of their
    metadata (as PySpark writes it), of the classifier and the models that they save, and of the
    copies and validators of the classifier's params."""
    shutil.rmtree(base, ignore_errors=True)
    rec("est_save", lambda: est.write().overwrite().save(base + "/ovr"))
    rec("est_dir", lambda: sorted(os.listdir(base + "/ovr")))
    rec("est_metadata", lambda: _metadata(base + "/ovr"))
    rec("est_classifier_metadata", lambda: _metadata(base + "/ovr/classifier")["class"])
    loaded = type(est).load(base + "/ovr")
    rec("est_loaded_uid", lambda: loaded.uid == est.uid)
    rec("est_loaded_classifier", lambda: type(loaded.getClassifier()).__name__)
    rec("est_loaded_classifier_uid", lambda: loaded.getClassifier().uid == est.getClassifier().uid)
    rec("est_loaded_params", lambda: _params_of(loaded))
    rec("est_loaded_isSet_weightCol", lambda: loaded.isSet("weightCol"))
    rec("est_model_save", lambda: model.write().overwrite().save(base + "/ovrm"))
    rec("model_dir", lambda: sorted(os.listdir(base + "/ovrm")))
    rec("model_metadata", lambda: _metadata(base + "/ovrm"))
    loaded_model = type(model).load(base + "/ovrm")
    rec("model_loaded_uid", lambda: loaded_model.uid == model.uid)
    rec("model_loaded_len", lambda: len(loaded_model.models))
    rec(
        "model_loaded_models_types",
        lambda: [
            (type(s).__name__, type(s).__module__.split(".")[-1]) for s in loaded_model.models
        ],
    )
    rec("model_loaded_models_uids", lambda: [s.uid for s in loaded_model.models])
    rec(
        "model_loaded_copy_models",
        lambda: [(type(s).__name__, s.uid) for s in loaded_model.copy().models],
    )
    rec("model_loaded_classifier", lambda: type(loaded_model.getClassifier()).__name__)
    rec("model_loaded_params", lambda: _params_of(loaded_model))
    rec("model_loaded_transform", lambda: loaded_model.transform(df))
    rec("model_loaded_resave", lambda: loaded_model.write().overwrite().save(base + "/ovrm2"))
    rec("model_loaded_resave_metadata", lambda: _metadata(base + "/ovrm2")["numClasses"])
    # The classifier's params, which a copy of the estimator, and so a fit of a param map, sets.
    classifier = est.getClassifier()
    rec(
        "copy_classifier_param",
        lambda: est.copy({classifier.maxIter: 7}).getClassifier().getMaxIter(),
    )
    rec("copy_keeps_classifier", lambda: est.copy().getClassifier().getMaxIter())
    rec(
        "save_without_classifier",
        lambda: _native("OneVsRest")().write().overwrite().save(base + "/none"),
    )
    shutil.rmtree(base, ignore_errors=True)
    _one_vs_rest_validator(rec, df, base)


def _one_vs_rest_plain_classifier_cases(rec, df, base):
    """The cases of a OneVsRest of a classifier that is not a native one, which PySpark fits as any
    classifier: PySpark's own LogisticRegression and MultilayerPerceptronClassifier of
    pyspark.ml.classification, a Python subclass of that LogisticRegression, and objects that are
    no classifier. The model is PySpark's OneVsRestModel of the classifier's own models."""
    import pyspark.ml.classification as classification
    from pyspark.sql import functions as F

    class LogisticRegressionSubclass(classification.LogisticRegression):
        pass

    lr = classification.LogisticRegression(maxIter=5, regParam=0.01)
    est = _native("OneVsRest")(classifier=lr)
    rec("plain_getClassifier_is_classifier", lambda: est.getClassifier() is lr)
    model = est.fit(df)
    rec("plain_model_class", lambda: (type(model).__name__, type(model).__module__))
    _submodel_cases(rec, "plain_lr_", model, lr)
    rec("plain_model_prefix", lambda: model.uid.split("_")[0])
    rec("plain_model_uid_is_estimator", lambda: model.uid == est.uid)
    rec("plain_model_params", lambda: _params_of(model))
    rec("plain_model_classifier_is_classifier", lambda: model.getClassifier() is lr)
    rec("plain_transform_columns", lambda: model.transform(df).columns)
    rec("plain_transform", lambda: model.transform(df))
    rec("plain_fit_map", lambda: est.fit(df, {est.predictionCol: "q"}).transform(df).columns)
    rec("plain_fit_list", lambda: [m.transform(df).count() for m in est.fit(df, [{}, {}])])
    rec(
        "plain_fit_classifier_map",
        lambda: [s.getMaxIter() for s in est.fit(df, {lr.maxIter: 3}).models],
    )
    subclassed = _native("OneVsRest")(classifier=LogisticRegressionSubclass(maxIter=5)).fit(df)
    rec("plain_subclass_models_types", lambda: [type(s).__name__ for s in subclassed.models])
    rec("plain_subclass_transform", lambda: subclassed.transform(df))
    # The weightCol, which PySpark's LogisticRegression takes, and its perceptron warns of.
    weighted = df.withColumn("w", F.lit(1.0))

    def weighted_fit(classifier):
        return _native("OneVsRest")(classifier=classifier, weightCol="w").fit(weighted)

    mlp = classification.MultilayerPerceptronClassifier(layers=[2, 4, 2], maxIter=2, seed=1)
    rec("plain_mlp_weight_warnings", lambda: _weight_warnings(lambda: weighted_fit(mlp)))
    rec(
        "plain_lr_weight_submodel_weightCols",
        lambda: [s.getWeightCol() for s in weighted_fit(lr).models],
    )
    rec("plain_object_fit", lambda: _native("OneVsRest")(classifier=object()).fit(df))
    rec("plain_none_fit", lambda: _native("OneVsRest")(classifier=None).fit(df))
    # The save of the estimator saves the classifier, and that of the model its models.
    shutil.rmtree(base, ignore_errors=True)
    rec("plain_est_save", lambda: est.write().overwrite().save(base + "/plain"))
    rec("plain_est_classifier_metadata", lambda: _metadata(base + "/plain/classifier")["class"])
    rec(
        "plain_est_loaded_classifier",
        lambda: type(type(est).load(base + "/plain").getClassifier()).__name__,
    )
    rec("plain_model_save", lambda: model.write().overwrite().save(base + "/plainm"))
    rec(
        "plain_model_loaded_models_types",
        lambda: [type(s).__name__ for s in type(model).load(base + "/plainm").models],
    )
    shutil.rmtree(base, ignore_errors=True)


def _one_vs_rest_validator(rec, df, base):
    """The cases of a validator over a OneVsRest of a LogisticRegression, whose grid sets the
    classifier's param: its save, load and fit, as PySpark's validator over the same."""
    lr = _native("LogisticRegression")(maxIter=5)
    ovr = _native("OneVsRest")(classifier=lr)
    evaluation = native_class(
        "pyspark.ml.connect.evaluation",
        "MulticlassClassificationEvaluator",
        "pyspark.ml.evaluation",
    )
    tuning = native_class("pyspark.ml.connect.tuning", "CrossValidator", "pyspark.ml.tuning")
    from pyspark.ml.tuning import ParamGridBuilder

    grid = ParamGridBuilder().addGrid(lr.regParam, [0.0, 0.5]).build()
    cv = tuning(
        estimator=ovr,
        estimatorParamMaps=grid,
        evaluator=evaluation(metricName="accuracy"),
        numFolds=2,
        seed=7,
    )
    rec("cv_save", lambda: cv.write().overwrite().save(base + "/cv"))
    rec(
        "cv_metadata_parents",
        lambda: [
            m[0]["parent"] == lr.uid
            for m in _metadata(base + "/cv")["paramMap"]["estimatorParamMaps"]
        ],
    )
    loaded = type(cv).load(base + "/cv")
    rec(
        "cv_loaded_grid",
        lambda: [
            sorted((p.name, repr(v)) for p, v in m.items())
            for m in loaded.getEstimatorParamMaps()
        ],
    )
    rec("cv_loaded_estimator", lambda: type(loaded.getEstimator()).__name__)
    model = cv.fit(df)
    rec("cv_avgMetrics", lambda: model.avgMetrics)
    rec("cv_model_save", lambda: model.write().overwrite().save(base + "/cvm"))
    loaded_model = type(model).load(base + "/cvm")
    rec("cv_model_loaded_avgMetrics", lambda: loaded_model.avgMetrics)
    rec("cv_model_loaded_transform", lambda: loaded_model.transform(df))
    shutil.rmtree(base, ignore_errors=True)


def live_cases(name, spark, base):
    """The model, summary, transform and fit cases of the case set `name`, against a server;
    `base` is a scratch directory of the server's file system."""
    rec = Recorder(base)
    if name == "NaiveBayes":
        _naive_bayes_live(rec, spark, base)
    elif name == "MultilayerPerceptronClassifier":
        _mlp_live(rec, spark, base)
    elif name == "FMClassifier":
        _fm_live(rec, spark, base)
    else:
        _one_vs_rest_live(rec, spark, base)
    shutil.rmtree(base, ignore_errors=True)
    return rec.out
