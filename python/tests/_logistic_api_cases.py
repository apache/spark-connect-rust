# SPDX-License-Identifier: Apache-2.0
"""The cases of the pyspark.ml API of LogisticRegression, its model and its summaries, of the
binary and the multinomial model: `scripts/capture_logistic_api_cases_golden.py` runs them on
official PySpark for tests/golden/logistic_api_cases.json, and test_logistic_api_offline.py and
test_logistic_api_e2e.py run them on the native classes and compare. Official PySpark runs the
tests too, as the reference (`_native_classes.native_class`).

The cases reuse the `Recorder`, the common cases of the Params API, and the model and persistence
cases of `_ml_api_cases`. The thresholds of LogisticRegression, which PySpark checks in Python, are
the cases of `threshold_cases`, for the estimator and the model alike.
"""

from _ml_api_cases import Recorder, model_cases, params_api_common_cases, persistence_cases
from _native_classes import native_class

# The estimator and model classes of the case set: (module of the native class, name, module of
# PySpark's).
CLASSES = {
    "LogisticRegression": [
        ("pyspark.ml.connect.classification", "LogisticRegression", "pyspark.ml.classification"),
        (
            "pyspark.ml.connect.classification",
            "LogisticRegressionModel",
            "pyspark.ml.classification",
        ),
    ],
}


def classes(name):
    """The estimator and model classes of the case set `name`."""
    return [native_class(*spec) for spec in CLASSES[name]]


def threshold_cases(rec, cls):
    """The thresholds of `cls`, an estimator or a model: the errors and values of PySpark's
    `_LogisticRegressionParams`, whose setters do not clear the other param over Spark Connect."""
    rec("threshold_default", lambda: cls().getThreshold())
    rec("thresholds_default", lambda: cls().getThresholds())
    rec("threshold_only_getThresholds", lambda: cls(threshold=0.3).getThresholds())
    rec("thresholds_only_getThreshold", lambda: cls(thresholds=[0.4, 0.6]).getThreshold())
    rec("thresholds_only_getThresholds", lambda: cls(thresholds=[0.4, 0.6]).getThresholds())
    rec("both_inconsistent_ctor", lambda: cls(threshold=0.3, thresholds=[0.5, 0.5]))
    rec("both_consistent_ctor", lambda: cls(threshold=0.25, thresholds=[3, 1]).getThreshold())
    rec("set_threshold_keeps_thresholds", lambda: cls(thresholds=[0.4, 0.6]).setThreshold(0.7))
    rec(
        "set_threshold_keeps_thresholds_get",
        lambda: cls(thresholds=[0.4, 0.6]).setThreshold(0.7).getThreshold(),
    )
    rec(
        "set_thresholds_keeps_threshold_get",
        lambda: cls(threshold=0.3).setThresholds([0.4, 0.6]).getThreshold(),
    )
    rec(
        "set_thresholds_then_threshold_get",
        lambda: cls(threshold=0.6).setThresholds([0.4, 0.6]).getThresholds(),
    )
    rec(
        "set_threshold_consistent_get",
        lambda: cls(thresholds=[0.4, 0.6]).setThreshold(0.6).getThreshold(),
    )
    rec("thresholds_of_three_get", lambda: cls().setThresholds([0.4, 0.5, 0.1]).getThreshold())
    rec("thresholds_of_three_getall", lambda: cls().setThresholds([0.4, 0.5, 0.1]).getThresholds())
    rec("thresholds_none_get", lambda: cls(thresholds=None).getThreshold())
    rec("thresholds_none_getall", lambda: cls(thresholds=None).getThresholds())
    rec("thresholds_zero_second", lambda: cls(thresholds=[0.5, 0.0]).getThreshold())
    rec("thresholds_opposite_get", lambda: cls(thresholds=[-1.0, 1.0]).getThreshold())
    rec("thresholds_empty_get", lambda: cls(thresholds=[]).getThreshold())
    rec("threshold_str", lambda: cls().setThreshold("x"))
    rec("thresholds_str", lambda: cls().setThresholds("x"))
    rec("threshold_int_get", lambda: cls().setThreshold(1).getThreshold())
    rec("threshold_param_name", lambda: cls().threshold.name)
    rec("thresholds_param_name", lambda: cls().thresholds.name)
    rec(
        "setParams_inconsistent",
        lambda: cls().setParams(threshold=0.1, thresholds=[0.5, 0.5]),
    )
    rec("setParams_both_then_get", lambda: cls().setParams(thresholds=[0.2, 0.8]).getThreshold())


def params_api_cases(name="LogisticRegression"):
    """The cases of PySpark's `Params` API of the estimator of the case set `name`, and of its
    thresholds, without a server."""
    estimator = classes(name)[0]
    rec = Recorder()
    est = estimator(maxIter=5, regParam=0.1)
    params_api_common_cases(rec, est, estimator())
    threshold_cases(rec, estimator)
    return rec.out


def _binary_frame(spark):
    from pyspark.ml.linalg import Vectors

    rows = [(float(x > 0), Vectors.dense([float(x), 1.0])) for x in (-4, -3, -2, -1, 1, 2, 3, 4)]
    return spark.createDataFrame(rows, ["label", "features"]).coalesce(1)


def _multinomial_frame(spark):
    from pyspark.ml.linalg import Vectors

    points = [
        (0.0, (0.0, 0.0)),
        (0.0, (0.2, 0.1)),
        (0.0, (0.1, 0.3)),
        (1.0, (4.0, 0.0)),
        (1.0, (4.2, 0.1)),
        (1.0, (4.1, 0.3)),
        (2.0, (0.0, 4.0)),
        (2.0, (0.2, 4.1)),
        (2.0, (0.1, 4.3)),
    ]
    rows = [(label, Vectors.dense(list(xy))) for label, xy in points]
    return spark.createDataFrame(rows, ["label", "features"]).coalesce(1)


def _binary_live(rec, model, df):
    from pyspark.ml.linalg import Vectors

    for a in ["coefficients", "intercept", "coefficientMatrix", "interceptVector"]:
        rec(a, lambda a=a: getattr(model, a))
    for a in ["numClasses", "numFeatures", "hasSummary"]:
        rec(a, lambda a=a: getattr(model, a))
    v = Vectors.dense([3.0, 1.0])
    rec("predict", lambda: model.predict(v))
    rec("predictRaw", lambda: model.predictRaw(v))
    rec("predictProbability", lambda: model.predictProbability(v))
    rec("model_getThreshold", lambda: model.getThreshold())
    rec("model_getThresholds", lambda: model.getThresholds())
    rec("model_getMaxIter", lambda: model.getMaxIter())
    rec("model_isSet_maxIter", lambda: model.isSet("maxIter"))
    rec("model_isSet_threshold", lambda: model.isSet("threshold"))
    rec("model_setThresholds_get", lambda: model.copy().setThresholds([0.2, 0.8]).getThreshold())
    rec("model_setThreshold_getThresholds", lambda: model.copy().setThreshold(0.25).getThresholds())
    rec("model_setProbabilityCol", lambda: model.copy().setProbabilityCol("p").getProbabilityCol())
    rec("model_threshold_attr", lambda: model.threshold.name)
    s = model.summary
    rec("summary_class", lambda: type(s).__name__)
    rec("summary_is_summary", lambda: model.summary is s)
    rec("copy_summary_is_summary", lambda: model.copy().summary is s)
    for a in [
        "accuracy",
        "areaUnderROC",
        "falsePositiveRateByLabel",
        "featuresCol",
        "labelCol",
        "labels",
        "objectiveHistory",
        "precisionByLabel",
        "predictionCol",
        "probabilityCol",
        "recallByLabel",
        "scoreCol",
        "totalIterations",
        "truePositiveRateByLabel",
        "weightCol",
        "weightedFalsePositiveRate",
        "weightedPrecision",
        "weightedRecall",
        "weightedTruePositiveRate",
    ]:
        rec("summary_" + a, lambda a=a: getattr(s, a))
    rec("summary_fMeasureByLabel", lambda: s.fMeasureByLabel())
    rec("summary_fMeasureByLabel_half", lambda: s.fMeasureByLabel(0.5))
    rec("summary_fMeasureByLabel_keyword", lambda: s.fMeasureByLabel(beta=2.0))
    rec("summary_weightedFMeasure", lambda: s.weightedFMeasure())
    rec("summary_weightedFMeasure_keyword", lambda: s.weightedFMeasure(beta=0.5))
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
    rec("evaluate_class", lambda: type(e).__name__)
    rec("evaluate_accuracy", lambda: e.accuracy)
    rec("evaluate_areaUnderROC", lambda: e.areaUnderROC)
    rec("evaluate_roc", lambda: e.roc)
    rec("evaluate_bad", lambda: model.evaluate(5))
    rec("transform_map", lambda: model.transform(df, {model.threshold: 0.9}).select("prediction"))
    rec("copy_threshold", lambda: model.copy({model.threshold: 0.9}).getThreshold())


def _multinomial_live(rec, model, df):
    rec("multi_numClasses", lambda: model.numClasses)
    rec("multi_hasSummary", lambda: model.hasSummary)
    rec("multi_coefficientMatrix_shape", lambda: model.coefficientMatrix.numRows)
    rec("multi_intercept", lambda: model.intercept)
    rec("multi_coefficients", lambda: model.coefficients)
    rec("multi_interceptVector_size", lambda: model.interceptVector.size)
    s = model.summary
    rec("multi_summary_class", lambda: type(s).__name__)
    for a in ["accuracy", "labels", "objectiveHistory", "totalIterations", "weightedRecall"]:
        rec("multi_summary_" + a, lambda a=a: getattr(s, a))
    rec("multi_summary_areaUnderROC", lambda: s.areaUnderROC)
    rec("multi_summary_fMeasureByLabel", lambda: s.fMeasureByLabel(0.5))
    rec("multi_summary_predictions", lambda: s.predictions.select("prediction"))
    e = model.evaluate(df)
    rec("multi_evaluate_class", lambda: type(e).__name__)
    rec("multi_evaluate_accuracy", lambda: e.accuracy)
    rec("multi_evaluate_fMeasureByLabel", lambda: e.fMeasureByLabel())


def live_cases(name, spark, base):
    """The cases of the binary and the multinomial model of the case set `name`, its summaries,
    evaluations, thresholds and persistence, against a server; `base` is a scratch directory of the
    server's file system."""
    estimator, model_class = classes(name)
    rec = Recorder(base)
    df = _binary_frame(spark)
    est = estimator(maxIter=5, regParam=0.01)
    model = est.fit(df)
    model_cases(rec, est, model, df)
    _binary_live(rec, model, df)
    # A fit with thresholds sends them, and the model's training summary is of its class.
    with_thresholds = estimator(maxIter=5, thresholds=[0.4, 0.6]).fit(df)
    rec("thresholds_fit_getThreshold", lambda: with_thresholds.getThreshold())
    rec("thresholds_fit_getThresholds", lambda: with_thresholds.getThresholds())
    rec("thresholds_fit_summary_class", lambda: type(with_thresholds.summary).__name__)
    persistence_cases(rec, est, model, model_class, df, base, True)
    multi_df = _multinomial_frame(spark)
    multi = estimator(maxIter=10, regParam=0.01).fit(multi_df)
    _multinomial_live(rec, multi, multi_df)
    return rec.out
