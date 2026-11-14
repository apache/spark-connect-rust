# SPDX-License-Identifier: Apache-2.0
"""The cases of the pyspark.ml API of the native frequent-pattern and recommendation classes that
follow it in full: FPGrowth and its model, PrefixSpan, and ALS and its model.

`scripts/capture_fpm_recommendation_api_cases_golden.py` runs them on official PySpark for
tests/golden/fpm_recommendation_api_cases.json, and test_fpm_recommendation_api_offline.py and
test_fpm_recommendation_api_e2e.py run them on the native classes and compare. Official PySpark
runs the tests too, as the reference (`_native_classes.native_class`).

The recorder and the normalization are those of `_ml_api_cases`. Its shared helpers
(`params_api_common_cases`, `model_cases`, `persistence_cases`) take the params of LinearSVC and
KMeans (`maxIter`, `tol`, `weightCol`, `getMaxIter` of the model), which these classes do not have,
so the same cases are written here over the params of each class (`PARAMS`).
"""

import shutil

from _ml_api_cases import Recorder
from _native_classes import native_class

# The classes of each case set: (module of the native class, name, module of PySpark's).
CLASSES = {
    "FPGrowth": [
        ("pyspark.ml.connect.fpm", "FPGrowth", "pyspark.ml.fpm"),
        ("pyspark.ml.connect.fpm", "FPGrowthModel", "pyspark.ml.fpm"),
    ],
    "PrefixSpan": [
        ("pyspark.ml.connect.fpm", "PrefixSpan", "pyspark.ml.fpm"),
    ],
    "ALS": [
        ("pyspark.ml.connect.recommendation", "ALS", "pyspark.ml.recommendation"),
        ("pyspark.ml.connect.recommendation", "ALSModel", "pyspark.ml.recommendation"),
    ],
}

# The params that the cases of each estimator use: an int, a float and a string param, by name.
PARAMS = {
    "FPGrowth": ("numPartitions", "minSupport", "itemsCol"),
    "PrefixSpan": ("maxPatternLength", "minSupport", "sequenceCol"),
    "ALS": ("maxIter", "regParam", "userCol"),
}

# The params of the models (of their estimators' defaults, by copy), by case set.
MODEL_PARAMS = {
    "FPGrowth": ("itemsCol", "minConfidence", "minSupport", "numPartitions", "predictionCol"),
    "ALS": ("blockSize", "coldStartStrategy", "itemCol", "predictionCol", "userCol"),
}

def classes(name):
    """The estimator and model classes of the case set `name`."""
    return [native_class(*spec) for spec in CLASSES[name]]


def _cap(name):
    return name[0].upper() + name[1:]


def _getter(name):
    return "get" + _cap(name)


def _params_cases(rec, est, other, int_p, float_p, str_p):
    """The cases of PySpark's `Params` API of the estimator `est`, over its int, float and string
    params, of the estimator `other` of the same class."""
    from pyspark.ml.base import Estimator
    from pyspark.ml.param import Params
    from pyspark.ml.tuning import ParamGridBuilder

    cls = type(est)
    rec("isinstance_estimator", lambda: isinstance(est, Estimator))
    rec("isinstance_params", lambda: isinstance(est, Params))
    rec(
        "class_param",
        lambda: (
            type(getattr(cls, int_p)).__name__,
            getattr(cls, int_p).parent,
            getattr(cls, int_p).name,
            getattr(cls, int_p) is getattr(cls, int_p),
            getattr(cls, int_p) == getattr(est, int_p),
        ),
    )
    rec(
        "grid_of_class_param",
        lambda: [
            sorted((p.parent, p.name, v) for p, v in m.items())
            for m in ParamGridBuilder().addGrid(getattr(cls, int_p), [1, 2]).build()
        ],
    )
    rec("copy_class_param", lambda: getattr(est.copy({getattr(cls, int_p): 3}), _getter(int_p))())
    rec("no_attribute", lambda: est.fooBarBaz)
    rec("param_names", lambda: [p.name for p in est.params])
    rec("param_docs", lambda: [p.doc for p in est.params])
    rec("param_parents_are_uid", lambda: all(p.parent == est.uid for p in est.params))
    rec("explainParams", lambda: est.explainParams())
    rec("explainParam_str", lambda: est.explainParam(int_p))
    rec("explainParam_param", lambda: est.explainParam(getattr(est, float_p)))
    for p in est.params:
        rec(_getter(p.name), lambda p=p: getattr(est, _getter(p.name))())
    rec("isSet_int", lambda: est.isSet(int_p))
    rec("isSet_float", lambda: est.isSet(float_p))
    rec("isDefined_str", lambda: est.isDefined(getattr(est, str_p)))
    rec("hasDefault_str", lambda: est.hasDefault(str_p))
    rec("hasDefault_float", lambda: est.hasDefault(getattr(est, float_p)))
    rec("hasParam_float", lambda: est.hasParam(float_p))
    rec("hasParam_nope", lambda: est.hasParam("nope"))
    rec("hasParam_int", lambda: est.hasParam(1))
    rec("getParam_nope", lambda: est.getParam("nope"))
    rec("getParam_method", lambda: est.getParam("fit"))
    rec("isSet_nope", lambda: est.isSet("nope"))
    rec("getOrDefault_int", lambda: est.getOrDefault(int_p))
    rec("getOrDefault_str", lambda: est.getOrDefault(getattr(est, str_p)))
    rec("extract", lambda: sorted((p.name, repr(v)) for p, v in est.extractParamMap().items()))
    rec(
        "extract_extra",
        lambda: sorted(
            (p.name, repr(v)) for p, v in est.extractParamMap({getattr(est, float_p): 0.5}).items()
        ),
    )
    copy = est.copy({getattr(est, int_p): 10})
    rec("copy_int", lambda: getattr(copy, _getter(int_p))())
    rec("copy_same_uid", lambda: copy.uid == est.uid)
    rec("orig_int", lambda: getattr(est, _getter(int_p))())
    rec("copy_explainParams", lambda: copy.explainParams())
    rec("copy_bad_extra", lambda: est.copy({int_p: 3}))
    rec("copy_list", lambda: est.copy([1]))
    rec("set_ok", lambda: (est.set(getattr(est, int_p), 7), getattr(est, _getter(int_p))())[1])
    rec("set_bad_type", lambda: est.set(getattr(est, int_p), "x"))
    rec("set_str_param", lambda: est.set(int_p, 3))
    rec("clear", lambda: (est.clear(getattr(est, int_p)), est.isSet(int_p)))
    rec("isSet_foreign", lambda: est.isSet(getattr(other, int_p)))
    rec("resolve_int", lambda: est.isSet(5))
    rec("repr_is_uid", lambda: repr(est) == est.uid)
    rec("big_int", lambda: (est.set(getattr(est, int_p), 2**70), est.isSet(int_p))[1])


def params_api_cases(name):
    """The cases of PySpark's `Params` API of the estimator of the case set `name`, without a
    server."""
    estimator = classes(name)[0]
    int_p, float_p, str_p = PARAMS[name]
    rec = Recorder()
    if name == "FPGrowth":
        est = estimator(minSupport=0.25, numPartitions=3)
    elif name == "PrefixSpan":
        est = estimator(minSupport=0.25, maxPatternLength=3)
    else:
        est = estimator(maxIter=5, regParam=0.25)
    _params_cases(rec, est, estimator(), int_p, float_p, str_p)
    if name == "ALS":
        rec("seed_is_class_name_hash", lambda: est.getSeed() == hash(name))
        rec("hasDefault_seed", lambda: est.hasDefault("seed"))
        rec("isSet_seed", lambda: est.isSet("seed"))
        rec("explainParam_seed", lambda: est.explainParam("seed"))
        rec("copy_seed", lambda: est.copy().getSeed() == hash(name))
        rec("setSeed", lambda: estimator().setSeed(42).getSeed())
        rec("seed_kwarg", lambda: estimator(seed=7).explainParam("seed"))
    return rec.out


def _fpgrowth_frame(spark):
    rows = [
        (["milk", "bread"],),
        (["bread", "butter"],),
        (["milk", "bread", "butter"],),
        (["beer"],),
        (["milk", "bread"],),
    ]
    return spark.createDataFrame(rows, ["items"]).coalesce(1)


def _als_frame(spark):
    rows = [
        (0, 0, 4.0),
        (0, 1, 2.0),
        (1, 1, 3.0),
        (1, 2, 5.0),
        (2, 0, 1.0),
        (2, 2, 4.0),
    ]
    return spark.createDataFrame(rows, ["user", "item", "rating"]).coalesce(1)


def _prefix_span_frame(spark):
    rows = [
        ([[1, 2], [3]],),
        ([[1], [3, 2], [1, 2]],),
        ([[1, 2], [5]],),
        ([[6]],),
    ]
    return spark.createDataFrame(rows, ["sequence"]).coalesce(1)


def _model_common(rec, est, model, model_params):
    """The cases of any fitted model of the estimator `est` that has the params `model_params`:
    its uid, its params and its description, as `model_cases` has them."""
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
    for name in model_params:
        rec("model_" + _getter(name), lambda name=name: getattr(model, _getter(name))())
        rec("model_isSet_" + name, lambda name=name: model.isSet(name))
        rec("model_hasDefault_" + name, lambda name=name: model.hasDefault(name))


def _model_setters(rec, model, setters):
    """Each setter of the model returns the object that it is called on, and sets the value."""
    for setter, value in setters:
        rec(
            setter + "_returns_self",
            lambda s=setter, v=value: (lambda c: getattr(c, s)(v) is c)(model.copy()),
        )


def _attr(obj, name):
    """The value of the member `name` of `obj`: a getter's call, or a property."""
    value = getattr(obj, name)
    return value() if callable(value) else value


def _fit_multiple(rec, est, df, name, values, member):
    """The fits of `est` of the param `name`, one per value, by the member `member` of each model."""
    rec(
        "fitMultiple",
        lambda: sorted(
            (i, _attr(m, member))
            for i, m in est.fitMultiple(df, [{getattr(est, name): v} for v in values])
        ),
    )


def live_cases(name, spark, base):
    """The cases of the fits, models, transforms and persistence of the case set `name`, against a
    server; `base` is a scratch directory of the server's file system."""
    rec = Recorder(base)
    if name == "FPGrowth":
        estimator, model_class = classes(name)
        df = _fpgrowth_frame(spark)
        est = estimator(minSupport=0.4, minConfidence=0.5, itemsCol="items")
        model = est.fit(df)
        _model_common(rec, est, model, MODEL_PARAMS[name])
        rec("freqItemsets", lambda: model.freqItemsets)
        rec("associationRules", lambda: model.associationRules)
        rec("freqItemsets_count", lambda: model.freqItemsets.count())
        rec("associationRules_count", lambda: model.associationRules.count())
        _model_setters(
            rec,
            model,
            [("setItemsCol", "basket"), ("setMinConfidence", 0.9), ("setPredictionCol", "next")],
        )
        rec("transform", lambda: model.transform(df))
        rec("transform_map", lambda: model.transform(df, {model.predictionCol: "q"}).columns)
        rec("transform_bad", lambda: model.transform(5))
        rec("fit_bad", lambda: est.fit(5))
        # A DataFrame of another session: the transform is lazy, and fails when it is analyzed.
        other = spark.newSession()
        held = other.createDataFrame([(["milk"],)], ["items"])
        rec("other_session_transform_columns", lambda: _condition(lambda: model.transform(held).columns))
        rec("other_session_transform_collect", lambda: _condition(lambda: model.transform(held).collect()))
        other.stop()
        _fit_multiple(rec, est, df, "minSupport", [0.4, 0.6], "getMinSupport")
        rec("fit_map", lambda: _attr(est.fit(df, {est.minSupport: 0.6}), "getMinSupport"))
        _persistence_fpm(rec, est, model, model_class, df, base)
    elif name == "PrefixSpan":
        (estimator,) = classes(name)
        df = _prefix_span_frame(spark)
        est = estimator(minSupport=0.5, maxPatternLength=2)
        rec("findFrequentSequentialPatterns", lambda: est.findFrequentSequentialPatterns(df))
        rec(
            "findFrequentSequentialPatterns_count",
            lambda: est.findFrequentSequentialPatterns(df).count(),
        )
        rec(
            "findFrequentSequentialPatterns_setters",
            lambda: est.setMinSupport(0.25).setSequenceCol("sequence").getMinSupport(),
        )
    else:
        estimator, model_class = classes(name)
        df = _als_frame(spark)
        est = estimator(
            rank=2, maxIter=3, seed=0, userCol="user", itemCol="item", ratingCol="rating"
        )
        model = est.fit(df)
        _model_common(rec, est, model, MODEL_PARAMS[name])
        rec("rank", lambda: model.rank)
        rec("userFactors_columns", lambda: model.userFactors.columns)
        rec("itemFactors_columns", lambda: model.itemFactors.columns)
        rec("userFactors_count", lambda: model.userFactors.count())
        rec("itemFactors_count", lambda: model.itemFactors.count())
        rec("recommendForAllUsers", lambda: model.recommendForAllUsers(2))
        rec("recommendForAllItems", lambda: model.recommendForAllItems(2))
        users = spark.createDataFrame([(0,), (2,)], ["user"])
        items = spark.createDataFrame([(1,), (2,)], ["item"])
        rec("recommendForUserSubset", lambda: model.recommendForUserSubset(users, 2))
        rec("recommendForItemSubset", lambda: model.recommendForItemSubset(items, 2))
        rec("recommendForAllUsers_column", lambda: model.recommendForAllUsers(_lit(spark, 2)))
        rec("recommendForAllUsers_list", lambda: model.recommendForAllUsers([]))
        rec("recommendForUserSubset_object", lambda: model.recommendForUserSubset(object(), 2))
        _model_setters(
            rec,
            model,
            [
                ("setUserCol", "user"),
                ("setItemCol", "item"),
                ("setPredictionCol", "p"),
                ("setColdStartStrategy", "drop"),
                ("setBlockSize", 16),
            ],
        )
        rec("transform", lambda: model.transform(df))
        rec("transform_map", lambda: model.transform(df, {model.predictionCol: "q"}).columns)
        rec("transform_bad", lambda: model.transform(5))
        rec("fit_bad", lambda: est.fit(5))
        rec("setNumBlocks", lambda: estimator().setNumBlocks(3).getNumItemBlocks())
        # Another session's DataFrames: the transform fails when it is analyzed, and the subsets
        # recommend for them, as PySpark does.
        other = spark.newSession()
        other_df = other.createDataFrame([(0, 0, 4.0)], ["user", "item", "rating"])
        other_users = other.createDataFrame([(0,), (2,)], ["user"])
        other_items = other.createDataFrame([(1,), (2,)], ["item"])
        rec("other_session_transform_columns", lambda: _condition(lambda: model.transform(other_df).columns))
        rec("other_session_transform_collect", lambda: _condition(lambda: model.transform(other_df).collect()))
        rec("other_session_recommendForUserSubset", lambda: model.recommendForUserSubset(other_users, 2))
        rec("other_session_recommendForItemSubset", lambda: model.recommendForItemSubset(other_items, 2))
        other.stop()
        _fit_multiple(rec, est, df, "rank", [2, 3], "rank")
        rec(
            "fit_map",
            lambda: _attr(model.copy({model.coldStartStrategy: "drop"}), "getColdStartStrategy"),
        )
        _persistence_als(rec, est, model, model_class, df, base)
    return rec.out


def _condition(run):
    """The value of `run()`, or the error condition of the Spark error that it raises."""
    try:
        return run()
    except Exception as e:  # noqa: BLE001 - the condition of a Spark error is the case
        condition = getattr(e, "getCondition", lambda: None)()
        if condition is None:
            raise
        return "condition " + condition


def _lit(spark, value):
    from pyspark.sql import functions as F

    return F.lit(value)


def _persistence_fpm(rec, est, model, model_class, df, base):
    """The save and load of FPGrowth and its model: the loaded objects give their params and
    their members."""
    shutil.rmtree(base, ignore_errors=True)
    rec("model_save", lambda: model.write().overwrite().save(base + "/model"))
    loaded = model_class.load(base + "/model")
    rec("loaded_uid", lambda: loaded.uid == model.uid)
    rec("loaded_minSupport", lambda: loaded.getMinSupport())
    rec("loaded_isSet_minSupport", lambda: loaded.isSet("minSupport"))
    rec("loaded_transform", lambda: loaded.transform(df))
    rec("loaded_freqItemsets_count", lambda: loaded.freqItemsets.count())
    rec("est_save", lambda: est.write().overwrite().save(base + "/est"))
    loaded_est = type(est).load(base + "/est")
    rec("est_loaded_uid", lambda: loaded_est.uid == est.uid)
    rec(
        "est_loaded_params",
        lambda: sorted((p.name, repr(v)) for p, v in loaded_est.extractParamMap().items()),
    )
    shutil.rmtree(base, ignore_errors=True)


def _persistence_als(rec, est, model, model_class, df, base):
    """The save and load of ALS and its model: the loaded objects give their params and rank."""
    shutil.rmtree(base, ignore_errors=True)
    rec("model_save", lambda: model.write().overwrite().save(base + "/model"))
    rec("model_save_again", lambda: model.save(base + "/model"))
    loaded = model_class.load(base + "/model")
    rec("loaded_uid", lambda: loaded.uid == model.uid)
    rec("loaded_rank", lambda: loaded.rank)
    rec("loaded_userCol", lambda: loaded.getUserCol())
    rec("loaded_isSet_userCol", lambda: loaded.isSet("userCol"))
    rec("loaded_explainParams", lambda: loaded.explainParams())
    rec("loaded_transform", lambda: loaded.transform(df))
    rec("loaded_userFactors_count", lambda: loaded.userFactors.count())
    rec("est_save", lambda: est.write().overwrite().save(base + "/est"))
    loaded_est = type(est).load(base + "/est")
    rec("est_loaded_uid", lambda: loaded_est.uid == est.uid)
    rec("est_loaded_rank", lambda: loaded_est.getRank())
    rec(
        "est_loaded_params",
        lambda: sorted((p.name, repr(v)) for p, v in loaded_est.extractParamMap().items()),
    )
    rec("est_loaded_isSet_rank", lambda: loaded_est.isSet("rank"))
    shutil.rmtree(base, ignore_errors=True)
