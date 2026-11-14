# SPDX-License-Identifier: Apache-2.0
"""The cases of the pyspark.ml.evaluation API of the six native evaluators: PySpark's Params API of
each, its `evaluate` with and without a param map, `isLargerBetter`, and its persistence.

`scripts/capture_evaluation_api_cases_golden.py` runs them on official PySpark for
tests/golden/evaluation_api_cases.json, and test_evaluation_api_offline.py and
test_evaluation_api_e2e.py run them on the native classes and compare. Official PySpark runs the
tests too, as the reference (`_native_classes.native_class`). The cases reuse `Recorder` of
`_ml_api_cases`. The Params API cases are those of `params_api_common_cases` in the shape that
evaluators have: an evaluator has no maxIter, tol or fit, so they are written here.
"""

import shutil

from _ml_api_cases import Recorder
from _native_classes import native_class
from pyspark.ml.linalg import Vectors

MODULE = "pyspark.ml.connect.evaluation"
REFERENCE = "pyspark.ml.evaluation"

# Each evaluator: `metric` is the param of the cases of its metric, `alt_metric` is another metric of
# its frame, `alt` is the param map of the cases of another evaluator of the same frame (of the
# param names), and `int_param` is an int param (its big int case).
EVALUATORS = {
    "BinaryClassificationEvaluator": {
        "metric": "metricName",
        "alt_metric": "areaUnderPR",
        "alt": {"metricName": "areaUnderPR"},
        "int_param": "numBins",
    },
    "RegressionEvaluator": {
        "metric": "metricName",
        "alt_metric": "r2",
        "alt": {"metricName": "r2"},
        "int_param": None,
    },
    "MulticlassClassificationEvaluator": {
        "metric": "metricName",
        "alt_metric": "logLoss",
        "alt": {"metricName": "logLoss"},
        "int_param": None,
    },
    "MultilabelClassificationEvaluator": {
        "metric": "metricName",
        "alt_metric": "hammingLoss",
        "alt": {"metricName": "hammingLoss"},
        "int_param": None,
    },
    "ClusteringEvaluator": {
        "metric": "metricName",
        "alt_metric": "silhouette",
        "alt": {"distanceMeasure": "cosine"},
        "int_param": None,
    },
    "RankingEvaluator": {
        "metric": "metricName",
        "alt_metric": "precisionAtK",
        "alt": {"metricName": "precisionAtK", "k": 2},
        "int_param": "k",
    },
}


def _cls(name):
    return native_class(MODULE, name, REFERENCE)


def _alt(evaluator, name):
    """The param map of the alternative metric of `evaluator`, with its Param keys."""
    return {getattr(evaluator, key): value for key, value in EVALUATORS[name]["alt"].items()}


def _getter(param):
    return "get" + param[0].upper() + param[1:]


def params_api_cases(name):
    """The cases of PySpark's `Params` API of the evaluator `name`, and of its `evaluate` calls that
    are refused before any request, without a server."""
    cls = _cls(name)
    spec = EVALUATORS[name]
    metric = spec["metric"]
    ev = cls()
    other = cls()
    rec = Recorder()
    rec("isinstance_evaluator", lambda: _is_evaluator(ev))
    rec("isinstance_params", lambda: _is_params(ev))
    # The Param of the class: of the parent "undefined", the same object on every access.
    rec(
        "class_param",
        lambda: (
            type(getattr(cls, metric)).__name__,
            getattr(cls, metric).parent,
            getattr(cls, metric).name,
            getattr(cls, metric) is getattr(cls, metric),
            getattr(cls, metric) == getattr(ev, metric),
        ),
    )
    rec("copy_class_param", lambda: ev.copy({getattr(cls, metric): "x"}).getOrDefault(metric))
    rec("no_attribute", lambda: ev.fooBarBaz)
    rec("writer_class", lambda: (type(ev.write()).__name__, type(ev.write()).__module__))
    rec("reader_class", lambda: (type(cls.read()).__name__, type(cls.read()).__module__))
    rec("writer_no_attribute", lambda: ev.write().fooBarBaz)
    writer = ev.write()
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
    rec("param_names", lambda: [p.name for p in ev.params])
    rec("param_docs", lambda: [p.doc for p in ev.params])
    rec("param_parents_are_uid", lambda: all(p.parent == ev.uid for p in ev.params))
    rec("explainParams", lambda: ev.explainParams())
    rec("explainParam_str", lambda: ev.explainParam(metric))
    rec("explainParam_param", lambda: ev.explainParam(getattr(ev, metric)))
    for p in ev.params:
        rec(_getter(p.name), lambda p=p: getattr(ev, _getter(p.name))())
    rec("isSet_metric_before", lambda: ev.isSet(metric))
    rec("isDefined_metric", lambda: ev.isDefined(getattr(ev, metric)))
    rec("hasDefault_metric", lambda: ev.hasDefault(metric))
    rec("hasParam_metric", lambda: ev.hasParam(metric))
    rec("hasParam_nope", lambda: ev.hasParam("nope"))
    rec("hasParam_int", lambda: ev.hasParam(1))
    rec("getParam_nope", lambda: ev.getParam("nope"))
    rec("getParam_method", lambda: ev.getParam("evaluate"))
    rec("isSet_nope", lambda: ev.isSet("nope"))
    rec("getOrDefault_metric", lambda: ev.getOrDefault(metric))
    rec("extract", lambda: sorted((p.name, repr(v)) for p, v in ev.extractParamMap().items()))
    rec(
        "extract_extra",
        lambda: sorted(
            (p.name, repr(v)) for p, v in ev.extractParamMap({getattr(ev, metric): "q"}).items()
        ),
    )
    copy = ev.copy(_alt(ev, name))
    rec("copy_metric", lambda: copy.getOrDefault(metric))
    rec("copy_same_uid", lambda: copy.uid == ev.uid)
    rec("orig_metric", lambda: ev.getOrDefault(metric))
    rec("copy_explainParams", lambda: copy.explainParams())
    rec("copy_bad_extra", lambda: ev.copy({metric: "x"}))
    rec("copy_list", lambda: ev.copy([1]))
    rec("set_ok", lambda: (ev.set(getattr(ev, metric), spec["alt_metric"]), ev.getOrDefault(metric))[1])
    rec("set_bad_type", lambda: ev.set(getattr(ev, metric), 3.5))
    rec("set_str_param", lambda: ev.set(metric, "areaUnderROC"))
    rec("clear", lambda: (ev.clear(getattr(ev, metric)), ev.isSet(metric)))
    rec("isSet_foreign", lambda: ev.isSet(getattr(other, metric)))
    rec("resolve_int", lambda: ev.isSet(5))
    rec("repr_is_uid", lambda: repr(ev) == ev.uid)
    int_param = spec["int_param"]
    if int_param:
        rec("big_int", lambda: (getattr(ev, "set" + _cap(int_param))(2**70), ev.getOrDefault(int_param))[1])
        rec("big_int_explain", lambda: ev.explainParam(int_param))
    # The metric's direction, of the metric of the evaluator and of a copy of another one.
    rec("isLargerBetter", lambda: ev.isLargerBetter())
    rec("isLargerBetter_alt", lambda: ev.copy(_alt(ev, name)).isLargerBetter())
    rec("setParams_forces_keywords", lambda: ev.setParams(1))
    rec(
        "setParams_returns_self",
        lambda: ev.copy().setParams(**_alt_names(name)).getOrDefault(metric),
    )
    rec("setParams_unknown", lambda: ev.setParams(nope=1))
    # The evaluate of a value that is not a DataFrame, and of a param map that is not a dict: both
    # raise before any request is sent.
    rec("evaluate_not_a_dataframe", lambda: ev.evaluate(5))
    rec("evaluate_list_map", lambda: ev.evaluate(5, [1]))
    rec("evaluate_str_map", lambda: ev.evaluate(5, "x"))
    rec("evaluate_map_of_a_non_dataframe", lambda: ev.evaluate(5, _alt(ev, name)))
    return rec.out


def _alt_names(name):
    """The alternative params of `name`, by name, for a setParams."""
    return dict(EVALUATORS[name]["alt"])


def _cap(param):
    return param[0].upper() + param[1:]


def _is_evaluator(obj):
    from pyspark.ml.evaluation import Evaluator

    return isinstance(obj, Evaluator)


def _is_params(obj):
    from pyspark.ml.param import Params

    return isinstance(obj, Params)


def _frame(spark, name):
    """The frame of the evaluator `name`: its columns are those of its default params."""
    if name == "BinaryClassificationEvaluator":
        rows = [
            (Vectors.dense([1.0 - s, s]), label, weight)
            for (s, label, weight) in [
                (0.1, 0.0, 1.0),
                (0.4, 1.0, 2.0),
                (0.35, 0.0, 1.0),
                (0.8, 1.0, 3.0),
                (0.4, 0.0, 0.5),
                (0.9, 0.0, 0.0),
                (0.7, 1.0, 1.0),
                (0.2, 1.0, 1.0),
            ]
        ]
        return spark.createDataFrame(rows, ["rawPrediction", "label", "weight"]).coalesce(1)
    if name == "RegressionEvaluator":
        rows = [
            (1.0, 1.2, 1.0),
            (2.0, 1.8, 2.0),
            (4.0, 3.5, 1.0),
            (3.0, 3.3, 3.0),
            (1.0, 0.4, 0.5),
        ]
        return spark.createDataFrame(rows, ["label", "prediction", "weight"]).coalesce(1)
    if name == "MulticlassClassificationEvaluator":
        rows = [
            (0.0, 0.0, Vectors.dense([0.8, 0.1, 0.1]), 1.0),
            (1.0, 0.0, Vectors.dense([0.6, 0.3, 0.1]), 2.0),
            (2.0, 2.0, Vectors.dense([0.0, 0.2, 0.8]), 1.0),
            (2.0, 1.0, Vectors.dense([0.0, 1.0, 0.0]), 0.5),
            (1.0, 1.0, Vectors.dense([0.1, 0.8, 0.1]), 3.0),
            (0.0, 2.0, Vectors.dense([0.1, 0.1, 0.8]), 1.0),
        ]
        frame = spark.createDataFrame(rows, ["label", "prediction", "probability", "weight"])
        return frame.coalesce(1)
    if name == "MultilabelClassificationEvaluator":
        rows = [
            ([0.0, 2.0], [0.0, 1.0]),
            ([1.0], [1.0, 2.0]),
            ([0.0, 1.0, 2.0], [0.0, 1.0]),
            ([2.0], []),
            ([0.0], [0.0]),
        ]
        return spark.createDataFrame(rows, "label array<double>, prediction array<double>").coalesce(1)
    if name == "ClusteringEvaluator":
        rows = [
            (Vectors.dense([1.0, 0.0]), 0.0),
            (Vectors.dense([0.8, 0.2]), 0.0),
            (Vectors.dense([0.0, 1.0]), 1.0),
            (Vectors.dense([0.1, 0.9]), 1.0),
            (Vectors.dense([0.5, 0.6]), 1.0),
        ]
        return spark.createDataFrame(rows, ["features", "prediction"]).coalesce(1)
    rows = [
        ([1.0, 2.0, 3.0], [1.0, 6.0, 2.0, 7.0, 3.0]),
        ([2.0, 4.0], [4.0, 1.0, 2.0]),
        ([3.0], [1.0, 2.0]),
        ([], [1.0, 2.0]),
        ([1.0], []),
    ]
    return spark.createDataFrame(rows, "label array<double>, prediction array<double>").coalesce(1)


def live_cases(name, spark, base):
    """The evaluate, isLargerBetter and persistence cases of the evaluator `name`, against a server;
    `base` is a scratch directory of the server's file system."""
    cls = _cls(name)
    spec = EVALUATORS[name]
    metric = spec["metric"]
    rec = Recorder(base)
    df = _frame(spark, name)
    ev = cls()
    rec("evaluate", lambda: ev.evaluate(df))
    rec("evaluate_none_map", lambda: ev.evaluate(df, None))
    rec("evaluate_empty_map", lambda: ev.evaluate(df, {}))
    # A param map is applied to a copy: the evaluator keeps its own metric.
    rec("evaluate_map", lambda: ev.evaluate(df, _alt(ev, name)))
    rec("evaluate_after_map", lambda: (ev.getOrDefault(metric), ev.evaluate(df))[1])
    rec("evaluate_map_isLargerBetter", lambda: ev.isLargerBetter())
    rec("evaluate_list_map", lambda: ev.evaluate(df, [_alt(ev, name)]))
    rec("evaluate_str_map", lambda: ev.evaluate(df, "x"))
    rec("evaluate_string_key_map", lambda: ev.evaluate(df, dict(spec["alt"])))
    rec("evaluate_invalid_metric_map", lambda: ev.evaluate(df, {getattr(ev, metric): "invalid"}))
    rec("evaluate_invalid_metric", lambda: cls(**{metric: "invalid"}).evaluate(df))
    # A setter changes the next evaluation.
    setter = "set" + _cap(metric)
    rec("setter_then_evaluate", lambda: getattr(cls(), setter)(spec["alt_metric"]).evaluate(df))
    rec(
        "setter_then_isLargerBetter",
        lambda: getattr(cls(), setter)(spec["alt_metric"]).isLargerBetter(),
    )
    rec("setParams_then_evaluate", lambda: cls().setParams(**_alt_names(name)).evaluate(df))
    rec("isLargerBetter", lambda: ev.isLargerBetter())
    # The persistence: the uid and the params of a saved evaluator, as the load of it.
    path = base + "/evaluator"
    rec("save", lambda: ev.write().overwrite().save(path))
    rec("save_again", lambda: ev.save(path))
    loaded = cls.load(path)
    rec("loaded_uid", lambda: loaded.uid == ev.uid)
    rec("loaded_params", lambda: sorted((p.name, repr(v)) for p, v in loaded.extractParamMap().items()))
    rec("loaded_isSet_metric", lambda: loaded.isSet(metric))
    rec("loaded_evaluate", lambda: loaded.evaluate(df))
    rec("loaded_isLargerBetter", lambda: loaded.isLargerBetter())
    rec("read_load_uid", lambda: cls.read().load(path).uid == ev.uid)
    alt = cls(**_alt_names(name))
    alt_path = base + "/alt"
    rec("alt_save", lambda: alt.write().overwrite().save(alt_path))
    loaded_alt = cls.load(alt_path)
    rec("loaded_alt_uid", lambda: loaded_alt.uid == alt.uid)
    rec("loaded_alt_params", lambda: sorted((p.name, repr(v)) for p, v in loaded_alt.extractParamMap().items()))
    rec("loaded_alt_evaluate", lambda: loaded_alt.evaluate(df))
    rec("loaded_alt_isLargerBetter", lambda: loaded_alt.isLargerBetter())
    rec("loaded_alt_copy_uid", lambda: loaded_alt.copy().uid == alt.uid)
    shutil.rmtree(base, ignore_errors=True)
    return rec.out


CLASSES = list(EVALUATORS)
