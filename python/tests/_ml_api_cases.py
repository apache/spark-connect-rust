# SPDX-License-Identifier: Apache-2.0
"""The cases of the pyspark.ml API of the native ML classes that follow it in full, of LinearSVC
and KMeans: `scripts/capture_ml_api_cases_golden.py` runs them on official PySpark for
tests/golden/ml_api_cases.json, and test_ml_api_offline.py and test_ml_api_e2e.py run them on the
native classes and compare. Official PySpark runs the tests too, as the reference
(`_native_classes.native_class`). The cases of other classes can reuse `Recorder`,
`params_api_common_cases` and `persistence_cases` of their own module and golden.

Each case records the repr of its value, or the type and the first line of the message of its
error, with what differs between runs normalized: the random part of a uid, the hash of the name of
a class of pyspark.ml (a default seed, which differs between Python processes) and the scratch
path of a save.
"""

import json
import re
import shutil
from pathlib import Path

from _native_classes import native_class

# The classes of each case set: (module of the native class, name, module of PySpark's).
CLASSES = {
    "LinearSVC": [
        ("pyspark.ml.connect.classification", "LinearSVC", "pyspark.ml.classification"),
        ("pyspark.ml.connect.classification", "LinearSVCModel", "pyspark.ml.classification"),
    ],
    "KMeans": [
        ("pyspark.ml.connect.clustering", "KMeans", "pyspark.ml.clustering"),
        ("pyspark.ml.connect.clustering", "KMeansModel", "pyspark.ml.clustering"),
    ],
    "Pipeline": [
        ("pyspark.ml.connect.pipeline", "Pipeline", "pyspark.ml.pipeline"),
        ("pyspark.ml.connect.pipeline", "PipelineModel", "pyspark.ml.pipeline"),
    ],
}

# The names of the classes of pyspark.ml, whose hash is a default seed (the longest first).
_CLASS_NAMES = sorted(
    json.loads((Path(__file__).resolve().parents[2] / "tests/golden/ml_api.json").read_text()),
    key=len,
    reverse=True,
)

_UID = re.compile(r"\b([A-Z][A-Za-z]*)_[0-9a-f]{12}\b")


def classes(name):
    """The estimator and model classes of the case set `name`."""
    return [native_class(*spec) for spec in CLASSES[name]]


def _normalize(text, base=None):
    text = _UID.sub(r"\1_<uid>", text)
    for name in _CLASS_NAMES:
        text = text.replace(str(hash(name)), f"<hash {name}>")
    if base:
        text = text.replace(base, "<path>")
    return text


def _value(v):
    if hasattr(v, "collect") and hasattr(v, "columns"):
        return {"columns": v.columns, "rows": sorted(repr(tuple(r)) for r in v.collect())}
    if isinstance(v, list):
        return [_value(x) for x in v]
    if isinstance(v, float):
        return round(v, 9)
    return repr(v)


class Recorder:
    def __init__(self, base=None):
        self.out = {}
        self.base = base

    def __call__(self, tag, f):
        try:
            value = _value(f())
            self.out[tag] = ["ok", json.loads(_normalize(json.dumps(value), self.base))]
        except Exception as e:  # noqa: BLE001 - every error is a case
            message = _normalize(str(e).split("\n")[0], self.base)[:200]
            self.out[tag] = [type(e).__name__, message]


def params_api_common_cases(rec, est, other):
    """The cases of PySpark's `Params` API of an estimator `est` of maxIter, tol and weightCol."""
    from pyspark.ml.base import Estimator
    from pyspark.ml.param import Params
    from pyspark.ml.tuning import ParamGridBuilder

    cls = type(est)
    rec("isinstance_estimator", lambda: isinstance(est, Estimator))
    rec("isinstance_params", lambda: isinstance(est, Params))
    # The Param of the class: of the parent "undefined", the same object on every access.
    rec(
        "class_param",
        lambda: (
            type(cls.maxIter).__name__,
            cls.maxIter.parent,
            cls.maxIter.name,
            cls.maxIter is cls.maxIter,
            cls.maxIter == est.maxIter,
        ),
    )
    rec(
        "grid_of_class_param",
        lambda: [
            sorted((p.parent, p.name, v) for p, v in m.items())
            for m in ParamGridBuilder().addGrid(cls.maxIter, [1, 2]).build()
        ],
    )
    rec("copy_class_param", lambda: est.copy({cls.maxIter: 3}).getMaxIter())
    # The messages of CPython name the classes as PySpark's.
    rec("no_attribute", lambda: est.fooBarBaz)
    rec("writer_class", lambda: (type(est.write()).__name__, type(est.write()).__module__))
    rec("reader_class", lambda: (type(cls.read()).__name__, type(cls.read()).__module__))
    rec("writer_no_attribute", lambda: est.write().fooBarBaz)
    writer = est.write()
    rec(
        "writer_should_overwrite",
        lambda: (writer.shouldOverwrite, setattr(writer, "shouldOverwrite", True))
        and writer.shouldOverwrite,
    )
    rec(
        "writer_option_map",
        lambda: (writer.option("Key", 1), writer.optionMap.__setitem__("x", "y"))
        and sorted(writer.optionMap.items()),
    )
    rec("param_names", lambda: [p.name for p in est.params])
    rec("param_docs", lambda: [p.doc for p in est.params])
    rec("param_parents_are_uid", lambda: all(p.parent == est.uid for p in est.params))
    rec("explainParams", lambda: est.explainParams())
    rec("explainParam_str", lambda: est.explainParam("maxIter"))
    rec("explainParam_param", lambda: est.explainParam(est.tol))
    for p in est.params:
        getter = "get" + p.name[0].upper() + p.name[1:]
        rec(getter, lambda getter=getter: getattr(est, getter)())
    rec("isSet_maxIter", lambda: est.isSet(est.maxIter))
    rec("isSet_tol", lambda: est.isSet("tol"))
    rec("isDefined_weightCol", lambda: est.isDefined(est.weightCol))
    rec("hasDefault_weightCol", lambda: est.hasDefault("weightCol"))
    rec("hasDefault_tol", lambda: est.hasDefault(est.tol))
    rec("hasParam_tol", lambda: est.hasParam("tol"))
    rec("hasParam_nope", lambda: est.hasParam("nope"))
    rec("hasParam_int", lambda: est.hasParam(1))
    rec("getParam_nope", lambda: est.getParam("nope"))
    rec("getParam_method", lambda: est.getParam("fit"))
    rec("isSet_nope", lambda: est.isSet("nope"))
    rec("getOrDefault_maxIter", lambda: est.getOrDefault("maxIter"))
    rec("getOrDefault_weightCol", lambda: est.getOrDefault(est.weightCol))
    rec("extract", lambda: sorted((p.name, repr(v)) for p, v in est.extractParamMap().items()))
    rec(
        "extract_extra",
        lambda: sorted((p.name, repr(v)) for p, v in est.extractParamMap({est.tol: 0.5}).items()),
    )
    copy = est.copy({est.maxIter: 10})
    rec("copy_maxIter", lambda: copy.getMaxIter())
    rec("copy_same_uid", lambda: copy.uid == est.uid)
    rec("orig_maxIter", lambda: est.getMaxIter())
    rec("copy_explainParams", lambda: copy.explainParams())
    rec("copy_bad_extra", lambda: est.copy({"maxIter": 3}))
    rec("copy_list", lambda: est.copy([1]))
    rec("set_ok", lambda: (est.set(est.maxIter, 7), est.getMaxIter())[1])
    rec("set_bad_type", lambda: est.set(est.maxIter, "x"))
    rec("set_str_param", lambda: est.set("maxIter", 3))
    rec("clear", lambda: (est.clear(est.maxIter), est.getMaxIter(), est.isSet("maxIter")))
    rec("isSet_foreign", lambda: est.isSet(other.maxIter))
    rec("resolve_int", lambda: est.isSet(5))
    rec("repr_is_uid", lambda: repr(est) == est.uid)
    rec("big_int", lambda: (est.setMaxIter(2**70), est.getMaxIter())[1])
    rec("big_int_explain", lambda: est.explainParam("maxIter"))


def _stages():
    """A KMeans and a LinearSVC, the stages of the cases of the Pipeline."""
    kmeans = classes("KMeans")[0](k=2, predictionCol="cluster")
    svc = classes("LinearSVC")[0](maxIter=5)
    return kmeans, svc


def _pipeline_params_api_cases(rec):
    """The cases of PySpark's `Params` API of the Pipeline and the PipelineModel, as PySpark's
    Python classes have it, without a server."""
    from pyspark.ml.base import Estimator, Model

    pipeline_class, model_class = classes("Pipeline")
    kmeans, svc = _stages()
    stages = [kmeans, svc]
    pipeline = pipeline_class(stages=stages)
    rec(
        "isinstance",
        lambda: (isinstance(pipeline, Estimator), isinstance(model_class([]), Model)),
    )
    rec("param_names", lambda: [p.name for p in pipeline.params])
    rec("explainParams_unset", lambda: pipeline_class().explainParams())
    rec("explainParams", lambda: pipeline.explainParams())
    rec("getStages_is_stages", lambda: pipeline.getStages() is stages)
    rec("getStages_unset", lambda: pipeline_class().getStages())
    rec(
        "isSet",
        lambda: (
            pipeline.isSet("stages"),
            pipeline_class().isSet("stages"),
            pipeline.hasDefault("stages"),
        ),
    )
    rec("extract", lambda: sorted((p.name, repr(v)) for p, v in pipeline.extractParamMap().items()))
    rec("positional", lambda: pipeline_class(stages))
    rec("unexpected_keyword", lambda: pipeline_class(foo=1))
    rec("setStages_none", lambda: pipeline_class().setStages(None).getStages())
    rec("setParams", lambda: pipeline_class().setParams(stages=[svc]).getStages()[0] is svc)
    copy = pipeline.copy({svc.maxIter: 7})
    rec(
        "copy",
        lambda: (
            copy.uid == pipeline.uid,
            [s.uid == t.uid for s, t in zip(copy.getStages(), stages)],
            copy.getStages()[0] is kmeans,
            copy.getStages()[1].getMaxIter(),
            svc.getMaxIter(),
        ),
    )
    rec("copy_unset", lambda: pipeline_class().copy())
    rec("copy_bad_extra", lambda: pipeline.copy({"x": 1}))
    cleared = pipeline_class(stages=[svc])
    rec("clear", lambda: (cleared.clear(cleared.stages), cleared.isSet("stages"))[1])
    rec("fit_bad_stage", lambda: pipeline_class(stages=[1]).fit("not a DataFrame"))
    rec("fit_bad_params", lambda: pipeline.fit("not a DataFrame", 5))
    rec(
        "class_param",
        lambda: (
            type(pipeline_class.stages).__name__,
            pipeline_class.stages.parent,
            pipeline_class.stages.name,
        ),
    )
    rec("repr_is_uid", lambda: repr(pipeline) == pipeline.uid)
    rec("no_attribute", lambda: pipeline.fooBarBaz)
    rec("write_class", lambda: type(pipeline.write()).__name__)
    model = model_class([kmeans, svc])
    rec("model_stages", lambda: model.stages[0] is kmeans)
    rec("model_params", lambda: (model.params, model.explainParams(), model.extractParamMap()))
    rec(
        "model_copy",
        lambda: (lambda c: (c.uid != model.uid, c.stages[0] is kmeans, len(c.stages)))(
            model.copy()
        ),
    )
    rec("model_transform_bad_params", lambda: model.transform("not a DataFrame", 5))
    rec("model_uid_prefix", lambda: model.uid.split("_")[0])
    rec("model_no_attribute", lambda: model.fooBarBaz)
    rec("model_set_stages", lambda: (setattr(model, "stages", [svc]), len(model.stages))[1])


def params_api_cases(name):
    """The cases of PySpark's `Params` API of the estimator of the case set `name`, without a
    server."""
    if name == "Pipeline":
        rec = Recorder()
        _pipeline_params_api_cases(rec)
        return rec.out
    estimator = classes(name)[0]
    rec = Recorder()
    if name == "LinearSVC":
        est = estimator(maxIter=5, regParam=0.1)
    else:
        est = estimator(maxIter=5, k=3)
        rec("seed_is_class_name_hash", lambda: est.getSeed() == hash(name))
        rec("hasDefault_seed", lambda: est.hasDefault("seed"))
        rec("isSet_seed", lambda: est.isSet("seed"))
        rec("explainParam_seed", lambda: est.explainParam("seed"))
        rec("copy_seed", lambda: est.copy().getSeed() == hash(name))
        rec("setSeed", lambda: estimator().setSeed(42).getSeed())
        rec("seed_kwarg", lambda: estimator(seed=7).explainParam("seed"))
    params_api_common_cases(rec, est, estimator())
    return rec.out


def _linear_svc_frame(spark):
    from pyspark.ml.linalg import Vectors

    rows = [(float(x > 0), Vectors.dense([float(x), 1.0])) for x in (-4, -3, -2, -1, 1, 2, 3, 4)]
    return spark.createDataFrame(rows, ["label", "features"]).coalesce(1)


def _kmeans_frame(spark):
    from pyspark.ml.linalg import Vectors

    points = [(0.0, 0.0), (0.1, 0.2), (0.2, 0.1), (9.0, 9.0), (9.1, 9.2), (9.2, 9.1)]
    rows = [(Vectors.dense([x, y]),) for x, y in points]
    return spark.createDataFrame(rows, ["features"]).coalesce(1)


def _linear_svc_live(rec, model, df):
    from pyspark.ml.linalg import Vectors

    for a in ["coefficients", "intercept", "numClasses", "numFeatures", "hasSummary"]:
        rec(a, lambda a=a: getattr(model, a))
    v = Vectors.dense([3.0, 1.0])
    rec("predict", lambda: model.predict(v))
    rec("predictRaw", lambda: model.predictRaw(v))
    rec("model_getMaxIter", lambda: model.getMaxIter())
    rec("model_isSet_maxIter", lambda: model.isSet("maxIter"))
    rec("model_isSet_tol", lambda: model.isSet("tol"))
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
    rec("summary_fMeasureByLabel_half", lambda: s.fMeasureByLabel(0.5))
    # The arguments are sent as given: an int as an int, a bool as a boolean (which the server's
    # method does not take), and by keyword.
    rec("summary_fMeasureByLabel_int", lambda: s.fMeasureByLabel(1))
    rec("summary_fMeasureByLabel_bool", lambda: s.fMeasureByLabel(True))
    rec("summary_weightedFMeasure_keyword", lambda: s.weightedFMeasure(beta=0.5))
    rec("summary_weightedFMeasure", lambda: s.weightedFMeasure())
    for a in [
        "roc",
        "pr",
        "predictions",
        "fMeasureByThreshold",
        "precisionByThreshold",
        "recallByThreshold",
    ]:
        rec("summary_df_" + a, lambda a=a: getattr(s, a))
    e = model.evaluate(df)
    rec("evaluate_accuracy", lambda: e.accuracy)
    rec("evaluate_areaUnderROC", lambda: e.areaUnderROC)
    rec("evaluate_roc", lambda: e.roc)
    rec("evaluate_bad", lambda: model.evaluate(5))
    rec("transform_map", lambda: model.transform(df, {model.threshold: 100.0}).select("prediction"))
    rec("copy_threshold", lambda: model.copy({model.threshold: 100.0}).getThreshold())
    rec("model_setThreshold", lambda: model.copy().setThreshold(0.5).getThreshold())


def _kmeans_live(rec, model, df):
    from pyspark.ml.linalg import Vectors

    near, far = Vectors.dense([0.0, 0.1]), Vectors.dense([9.0, 9.0])
    rec("model_seed_is_estimator_hash", lambda: model.getSeed() == hash("KMeans"))
    rec("model_hasDefault_seed", lambda: model.hasDefault("seed"))
    rec("model_isSet_seed", lambda: model.isSet("seed"))
    rec("model_explainParam_seed", lambda: model.explainParam("seed"))
    rec("model_copy_seed", lambda: model.copy().getSeed() == hash("KMeans"))
    rec("model_getK", lambda: model.getK())
    rec("model_isSet_k", lambda: model.isSet("k"))
    rec("clusterCenters", lambda: sorted(c.round(9).tolist() for c in model.clusterCenters()))
    rec("clusterCenters_type", lambda: type(model.clusterCenters()[0]).__name__)
    for a in ["numFeatures", "hasSummary"]:
        rec(a, lambda a=a: getattr(model, a))
    rec("predict_apart", lambda: model.predict(near) != model.predict(far))
    rec("predict_type", lambda: type(model.predict(far)).__name__)
    s = model.summary
    rec("summary_is_summary", lambda: model.summary is s)
    rec("copy_summary_is_summary", lambda: model.copy().summary is s)
    for a in ["k", "clusterSizes", "trainingCost", "numIter", "featuresCol", "predictionCol"]:
        rec("summary_" + a, lambda a=a: getattr(s, a))
    rec("summary_df_predictions", lambda: s.predictions.select("features"))
    rec("summary_df_cluster_count", lambda: s.cluster.count())
    rec("model_setPredictionCol", lambda: model.copy().setPredictionCol("p").getPredictionCol())
    rec("transform_map", lambda: model.transform(df, {model.predictionCol: "q"}).columns)


def model_cases(rec, est, model, df, class_param="maxIter", value=3):
    """The cases of any fitted model `model` of the estimator `est` of the DataFrame `df`: its uid,
    its params and its description. A fit of a map of the Param `class_param` of the class (of
    the parent "undefined") to `value` ignores it, as a Param of another parent."""
    from pyspark.ml.base import Model, Transformer

    rec("model_uid_is_estimator_uid", lambda: model.uid == est.uid)
    rec("isinstance_model", lambda: (isinstance(model, Model), isinstance(model, Transformer)))
    rec("model_no_attribute", lambda: model.fooBarBaz)
    rec("model_class", lambda: (type(model).__name__, type(model).__module__.split(".")[-1]))
    getter = "get" + class_param[0].upper() + class_param[1:]
    rec(
        "fit_class_param",
        lambda: getattr(est.fit(df, {getattr(type(est), class_param): value}), getter)(),
    )
    rec("model_explainParams", lambda: model.explainParams())
    rec(
        "model_extract",
        lambda: sorted((p.name, repr(v)) for p, v in model.extractParamMap().items()),
    )
    rec("repr_prefix", lambda: repr(model).split(":")[0])


def persistence_cases(rec, est, model, model_class, df, base, summary_property=True):
    """The cases of any estimator `est` of maxIter and its fitted model `model` (of the class
    `model_class`): transform, save and load of both, and the fits of param maps. `base` is a
    scratch directory of the server's file system; `summary_property` is whether `summary` of the
    model is a property (else a method, as LinearSVCModel's)."""
    rec("transform", lambda: model.transform(df))
    shutil.rmtree(base, ignore_errors=True)
    rec("model_save", lambda: model.write().overwrite().save(base + "/model"))
    rec("model_save_again", lambda: model.save(base + "/model"))
    loaded = model_class.load(base + "/model")
    rec("loaded_uid", lambda: loaded.uid == model.uid)
    if hasattr(model, "hasSummary"):
        rec("loaded_hasSummary", lambda: loaded.hasSummary)
        rec("loaded_summary", lambda: loaded.summary if summary_property else loaded.summary())
    rec("loaded_getMaxIter", lambda: loaded.getMaxIter())
    rec("loaded_isSet_maxIter", lambda: loaded.isSet("maxIter"))
    rec("loaded_explainParams", lambda: loaded.explainParams())
    rec("loaded_transform", lambda: loaded.transform(df))
    rec("est_save", lambda: est.write().overwrite().save(base + "/est"))
    loaded_est = type(est).load(base + "/est")
    rec("est_loaded_uid", lambda: loaded_est.uid == est.uid)
    rec(
        "est_loaded_params",
        lambda: sorted((p.name, repr(v)) for p, v in loaded_est.extractParamMap().items()),
    )
    rec("est_loaded_isSet_maxIter", lambda: loaded_est.isSet("maxIter"))
    rec(
        "fitMultiple",
        lambda: sorted(
            (i, m.getMaxIter())
            for i, m in est.fitMultiple(df, [{est.maxIter: 2}, {est.maxIter: 3}])
        ),
    )
    rec("fit_map", lambda: est.fit(df, {est.maxIter: 2}).getMaxIter())
    rec("fit_list", lambda: [m.getMaxIter() for m in est.fit(df, [{est.maxIter: 2}, {}])])
    shutil.rmtree(base, ignore_errors=True)


def _metadata(path):
    """The metadata that a save wrote at `path` (of the server's file system, which is this one's),
    without its timestamp."""
    import glob

    metadata = json.loads(open(glob.glob(path + "/metadata/part-*")[0]).read())
    metadata.pop("timestamp")
    return metadata


def _pipeline_live_cases(rec, spark, base):
    """The fit, transform and persistence cases of the Pipeline and the PipelineModel."""
    import os

    pipeline_class, model_class = classes("Pipeline")
    df = _linear_svc_frame(spark)
    kmeans, svc = _stages()
    pipeline = pipeline_class(stages=[kmeans, svc])
    model = pipeline.fit(df)
    rec("fit_stages", lambda: [type(s).__name__ for s in model.stages])
    rec("fit_new_uid", lambda: model.uid != pipeline.uid)
    rec("fit_stage_uids", lambda: [s.uid for s in model.stages] == [kmeans.uid, svc.uid])
    rec("transform", lambda: model.transform(df).select("cluster", "prediction"))
    rec(
        "transform_map",
        lambda: model.transform(df, {model.stages[1].threshold: 100.0}).select("prediction"),
    )
    rec("fit_map", lambda: pipeline.fit(df, {svc.maxIter: 2}).stages[1].getMaxIter())
    rec(
        "fit_list",
        lambda: [m.stages[1].getMaxIter() for m in pipeline.fit(df, [{svc.maxIter: 2}, {}])],
    )
    rec(
        "fitMultiple",
        lambda: sorted(
            (i, m.stages[1].getMaxIter())
            for i, m in pipeline.fitMultiple(df, [{svc.maxIter: 2}, {svc.maxIter: 3}])
        ),
    )
    # A stage after the last estimator is kept as it is.
    trailing = pipeline_class(stages=[svc, model.stages[0]])
    rec(
        "fit_trailing_transformer",
        lambda: (lambda m: (type(m.stages[1]).__name__, m.stages[1] is model.stages[0]))(
            trailing.fit(df)
        ),
    )
    shutil.rmtree(base, ignore_errors=True)
    rec("save", lambda: pipeline.write().overwrite().save(base + "/pipeline"))
    rec("save_again", lambda: pipeline.save(base + "/pipeline"))
    rec("metadata", lambda: _metadata(base + "/pipeline"))
    rec(
        "stage_dirs",
        lambda: sorted(d.split("_", 1) for d in os.listdir(base + "/pipeline/stages")),
    )
    rec("stage_metadata", lambda: _metadata(base + "/pipeline/stages/1_" + svc.uid))
    # A directory of another object has no stageUids in its params.
    rec("load_not_a_pipeline", lambda: pipeline_class.load(base + "/pipeline/stages/1_" + svc.uid))
    rec(
        "model_load_not_a_pipeline",
        lambda: model_class.load(base + "/pipeline/stages/1_" + svc.uid),
    )
    loaded = pipeline_class.load(base + "/pipeline")
    rec(
        "load",
        lambda: (
            loaded.uid == pipeline.uid,
            [type(s).__name__ for s in loaded.getStages()],
            [s.uid for s in loaded.getStages()] == [kmeans.uid, svc.uid],
            loaded.getStages()[1].getMaxIter(),
        ),
    )
    rec("model_save", lambda: model.write().overwrite().save(base + "/model"))
    rec("model_metadata", lambda: _metadata(base + "/model"))
    loaded_model = model_class.load(base + "/model")
    rec(
        "model_load",
        lambda: (loaded_model.uid == model.uid, [type(s).__name__ for s in loaded_model.stages]),
    )
    rec("model_load_transform", lambda: loaded_model.transform(df).select("cluster", "prediction"))
    shutil.rmtree(base, ignore_errors=True)


def live_cases(name, spark, base):
    """The model, summary, transform and persistence cases of the case set `name`, against a
    server; `base` is a scratch directory of the server's file system."""
    if name == "Pipeline":
        rec = Recorder(base)
        _pipeline_live_cases(rec, spark, base)
        return rec.out
    estimator, model_class = classes(name)
    rec = Recorder(base)
    if name == "LinearSVC":
        df = _linear_svc_frame(spark)
        est = estimator(maxIter=5, regParam=0.01)
    else:
        df = _kmeans_frame(spark)
        est = estimator(k=2, maxIter=10)
    model = est.fit(df)
    model_cases(rec, est, model, df)
    if name == "LinearSVC":
        _linear_svc_live(rec, model, df)
    else:
        _kmeans_live(rec, model, df)
    persistence_cases(rec, est, model, model_class, df, base, name != "LinearSVC")
    return rec.out
