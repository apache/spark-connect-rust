# SPDX-License-Identifier: Apache-2.0
"""The classes of the objects that the native classification classes make, against a live Connect
server. Every object that this client makes in Rust, of a fit, a fit of param maps, fitMultiple, a
copy, a load, a training summary, an evaluation, the trees of an ensemble, the models of a
OneVsRest and the models of a validator, is of the public class of pyspark.ml.connect, and an
instance of every class of the MRO of PySpark's class, as PySpark's object is of its class.

The meta-algorithms of pyspark.ml (Pipeline, OneVsRest, CrossValidator, TrainValidationSplit) fit,
transform and save the native classifiers. Their saves load in pyspark.ml, of PySpark's classes,
which it takes of the class names of the metadata, and in this client, of the public classes; what
this client's meta-algorithms save loads in pyspark.ml too, of the same uids and predictions.
Official PySpark runs the tests too, as the reference, of its classes
(`_native_classes.native_class`).
"""

import importlib
import os
import tempfile

import pytest
from _native_classes import native_class

CLASSIFICATION = "classification"

# The estimators, of their model classes, their keyword arguments, and the classes of the training
# summary and of an evaluation of their binary model, of the models that have them.
ESTIMATORS = {
    "LogisticRegression": (
        "LogisticRegressionModel",
        {"maxIter": 5},
        ("BinaryLogisticRegressionTrainingSummary", "BinaryLogisticRegressionSummary"),
    ),
    "LinearSVC": (
        "LinearSVCModel",
        {"maxIter": 5, "regParam": 0.01},
        ("LinearSVCTrainingSummary", "LinearSVCSummary"),
    ),
    "DecisionTreeClassifier": (
        "DecisionTreeClassificationModel",
        {"maxDepth": 3, "seed": 1},
        None,
    ),
    "RandomForestClassifier": (
        "RandomForestClassificationModel",
        {"numTrees": 3, "maxDepth": 3, "seed": 1},
        (
            "BinaryRandomForestClassificationTrainingSummary",
            "BinaryRandomForestClassificationSummary",
        ),
    ),
    "GBTClassifier": ("GBTClassificationModel", {"maxIter": 3, "maxDepth": 2, "seed": 1}, None),
    "NaiveBayes": ("NaiveBayesModel", {"smoothing": 1.0}, None),
    "MultilayerPerceptronClassifier": (
        "MultilayerPerceptronClassificationModel",
        {"maxIter": 5, "layers": [2, 4, 2], "seed": 1},
        (
            "MultilayerPerceptronClassificationTrainingSummary",
            "MultilayerPerceptronClassificationSummary",
        ),
    ),
    "FMClassifier": (
        "FMClassificationModel",
        {"maxIter": 5, "seed": 1, "factorSize": 2},
        ("FMClassificationTrainingSummary", "FMClassificationSummary"),
    ),
}

# The classes of the training summary and of an evaluation of a multiclass model, which differ
# from those of its binary model.
MULTICLASS_SUMMARIES = {
    "LogisticRegression": ("LogisticRegressionTrainingSummary", "LogisticRegressionSummary"),
    "RandomForestClassifier": (
        "RandomForestClassificationTrainingSummary",
        "RandomForestClassificationSummary",
    ),
}

# The module and the class of the trees of each ensemble: a GBTClassificationModel's trees are
# regression trees, as in PySpark.
ENSEMBLES = {
    "RandomForestClassifier": (CLASSIFICATION, "DecisionTreeClassificationModel"),
    "GBTClassifier": ("regression", "DecisionTreeRegressionModel"),
}

VALIDATORS = {"CrossValidator": {"numFolds": 2}, "TrainValidationSplit": {"trainRatio": 0.75}}


def cls(module, name):
    """The class `name` of pyspark.ml.connect.<module>: the native public class in this client,
    PySpark's class of pyspark.ml.<module> in official PySpark."""
    return native_class(f"pyspark.ml.connect.{module}", name, f"pyspark.ml.{module}")


def pyspark_class(module, name):
    """PySpark's class `name` of pyspark.ml.<module>, the vendored one in this client."""
    return getattr(importlib.import_module(f"pyspark.ml.{module}"), name)


def check(obj, module, name):
    """`obj` is of the class `name` of pyspark.ml.connect.<module> (`cls`), and an instance of every
    class of the MRO of PySpark's class of the name, PySpark's class itself included."""
    assert type(obj) is cls(module, name), (type(obj), name)
    assert [c for c in pyspark_class(module, name).__mro__ if not isinstance(obj, c)] == []
    return obj


def summary_of(model):
    """The training summary of a model: a property of some classes, a method of the others."""
    summary = model.summary
    return summary() if callable(summary) else summary


def predictions(model, df):
    return [row.prediction for row in model.transform(df).select("prediction").collect()]


def uids(objects):
    return [(o.uid, type(o).__name__) for o in objects]


@pytest.fixture(scope="module")
def session():
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live ML class tests")
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.remote(remote).getOrCreate()
    yield spark
    spark.stop()


def _frame(spark, labels):
    """A frame of 12 rows of `labels` classes, of nonnegative features, as NaiveBayes takes."""
    from pyspark.ml.linalg import Vectors

    rows = [
        (float(i % labels), Vectors.dense([float(i % labels) + 0.1 * (i % 4), float(i % 3)]))
        for i in range(12)
    ]
    return spark.createDataFrame(rows, ["label", "features"]).coalesce(1)


@pytest.fixture(scope="module")
def binary(session):
    return _frame(session, 2)


@pytest.fixture(scope="module")
def multiclass(session):
    return _frame(session, 3)


@pytest.fixture
def scratch():
    with tempfile.TemporaryDirectory() as path:
        yield path


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_models_and_estimators_made_in_rust_are_of_their_public_classes(name, binary, scratch):
    model_name, kwargs, summaries = ESTIMATORS[name]
    model_class = cls(CLASSIFICATION, model_name)
    est = check(cls(CLASSIFICATION, name)(**kwargs), CLASSIFICATION, name)
    check(est.copy(), CLASSIFICATION, name)
    model = check(est.fit(binary), CLASSIFICATION, model_name)
    check(model.copy(), CLASSIFICATION, model_name)
    check(model.copy({model.predictionCol: "p"}), CLASSIFICATION, model_name)
    check(est.fit(binary, {est.predictionCol: "p"}), CLASSIFICATION, model_name)
    maps = [{}, {est.predictionCol: "p"}]
    for fitted in est.fit(binary, maps):
        check(fitted, CLASSIFICATION, model_name)
    for _, fitted in est.fitMultiple(binary, maps):
        check(fitted, CLASSIFICATION, model_name)

    model.write().overwrite().save(scratch + "/model")
    loaded = check(model_class.load(scratch + "/model"), CLASSIFICATION, model_name)
    check(loaded.copy(), CLASSIFICATION, model_name)
    model.save(scratch + "/model-saved")
    check(model_class.read().load(scratch + "/model-saved"), CLASSIFICATION, model_name)
    est.write().overwrite().save(scratch + "/estimator")
    check(cls(CLASSIFICATION, name).load(scratch + "/estimator"), CLASSIFICATION, name)

    if summaries is not None:
        training, evaluated = summaries
        check(summary_of(model), CLASSIFICATION, training)
        check(model.evaluate(binary), CLASSIFICATION, evaluated)
        check(loaded.evaluate(binary), CLASSIFICATION, evaluated)


@pytest.mark.parametrize("name", list(MULTICLASS_SUMMARIES))
def test_the_summaries_of_a_multiclass_model_are_of_their_public_classes(name, multiclass):
    _, kwargs, _ = ESTIMATORS[name]
    model = cls(CLASSIFICATION, name)(**kwargs).fit(multiclass)
    training, evaluated = MULTICLASS_SUMMARIES[name]
    check(summary_of(model), CLASSIFICATION, training)
    check(model.evaluate(multiclass), CLASSIFICATION, evaluated)


@pytest.mark.parametrize("name", list(ENSEMBLES))
def test_the_trees_of_an_ensemble_are_of_their_public_classes(name, binary, scratch):
    model_name, kwargs, _ = ESTIMATORS[name]
    module, tree = ENSEMBLES[name]
    model = cls(CLASSIFICATION, name)(**kwargs).fit(binary)
    model.write().overwrite().save(scratch + "/model")
    loaded = cls(CLASSIFICATION, model_name).load(scratch + "/model")
    for each in (model, model.copy(), loaded):
        assert each.trees
        for t in each.trees:
            check(t, module, tree)


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_the_models_of_a_one_vs_rest_are_of_their_public_classes(name, multiclass, scratch):
    model_name, kwargs, _ = ESTIMATORS[name]
    ovr = cls(CLASSIFICATION, "OneVsRest")(classifier=cls(CLASSIFICATION, name)(**kwargs))
    check(ovr, CLASSIFICATION, "OneVsRest")
    check(ovr.getClassifier(), CLASSIFICATION, name)
    check(ovr.copy().getClassifier(), CLASSIFICATION, name)
    model = check(ovr.fit(multiclass), CLASSIFICATION, "OneVsRestModel")
    check(model.getClassifier(), CLASSIFICATION, name)

    ovr.write().overwrite().save(scratch + "/estimator")
    loaded_ovr = cls(CLASSIFICATION, "OneVsRest").load(scratch + "/estimator")
    check(loaded_ovr, CLASSIFICATION, "OneVsRest")
    check(loaded_ovr.getClassifier(), CLASSIFICATION, name)
    model.write().overwrite().save(scratch + "/model")
    loaded = cls(CLASSIFICATION, "OneVsRestModel").load(scratch + "/model")
    check(loaded, CLASSIFICATION, "OneVsRestModel")
    check(loaded.getClassifier(), CLASSIFICATION, name)
    for each in (model, model.copy(), loaded):
        assert len(each.models) == 3
        for submodel in each.models:
            check(submodel, CLASSIFICATION, model_name)


def test_the_fits_of_param_maps_of_a_one_vs_rest_are_of_their_public_classes(multiclass):
    lr = cls(CLASSIFICATION, "LogisticRegression")(maxIter=5)
    ovr = cls(CLASSIFICATION, "OneVsRest")(classifier=lr)
    maps = [{}, {ovr.predictionCol: "p"}]
    fits = [ovr.fit(multiclass, maps[1])] + ovr.fit(multiclass, maps)
    fits += [fitted for _, fitted in ovr.fitMultiple(multiclass, maps)]
    for model in fits:
        check(model, CLASSIFICATION, "OneVsRestModel")
        for submodel in model.models:
            check(submodel, CLASSIFICATION, "LogisticRegressionModel")


@pytest.mark.parametrize("validator", list(VALIDATORS))
def test_the_models_of_a_validator_are_of_their_public_classes(validator, binary, scratch):
    from pyspark.ml.tuning import ParamGridBuilder

    lr = cls(CLASSIFICATION, "LogisticRegression")(maxIter=5)
    evaluator = cls("evaluation", "BinaryClassificationEvaluator")()
    # The Params of the estimator's own: a validator's save requires that each Param of the maps
    # is the estimator's, which a class Param (of the parent "undefined") is not.
    grid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.5]).build()
    est = cls("tuning", validator)(
        estimator=lr,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        seed=3,
        collectSubModels=True,
        **VALIDATORS[validator],
    )
    check(est, "tuning", validator)
    check(est.getEstimator(), CLASSIFICATION, "LogisticRegression")
    check(est.copy().getEstimator(), CLASSIFICATION, "LogisticRegression")
    model = check(est.fit(binary), "tuning", validator + "Model")
    check(est.fit(binary, {est.seed: 4}).bestModel, CLASSIFICATION, "LogisticRegressionModel")

    est.write().overwrite().save(scratch + "/estimator")
    loaded_est = cls("tuning", validator).load(scratch + "/estimator")
    check(loaded_est, "tuning", validator)
    check(loaded_est.getEstimator(), CLASSIFICATION, "LogisticRegression")
    check(loaded_est.getEvaluator(), "evaluation", "BinaryClassificationEvaluator")
    model.write().option("persistSubModels", "true").overwrite().save(scratch + "/model")
    loaded = cls("tuning", validator + "Model").load(scratch + "/model")
    check(loaded, "tuning", validator + "Model")
    for each in (model, model.copy(), loaded):
        check(each.bestModel, CLASSIFICATION, "LogisticRegressionModel")
        submodels = each.subModels
        if validator == "CrossValidator":
            submodels = [submodel for fold in submodels for submodel in fold]
        assert len(submodels) == len(grid) * VALIDATORS[validator].get("numFolds", 1)
        for submodel in submodels:
            check(submodel, CLASSIFICATION, "LogisticRegressionModel")


def _check_loads(saved, path, meta_module, meta_name, members, df):
    """`saved`, a fitted meta-algorithm saved at `path`, loads in pyspark.ml of PySpark's classes,
    and in this client of the public classes, of the uids and the predictions of `saved`. The meta
    model is the class `meta_name` of the module `meta_module`; `members` is the function of a
    meta model of its submodels, and the (module, class name) of each."""
    get, specs = members
    by_pyspark = pyspark_class(meta_module, meta_name).load(path)
    assert type(by_pyspark) is pyspark_class(meta_module, meta_name)
    assert [type(o) for o in get(by_pyspark)] == [pyspark_class(*spec) for spec in specs]
    by_ours = check(cls(meta_module, meta_name).load(path), meta_module, meta_name)
    assert len(get(by_ours)) == len(specs)
    for o, (module, name) in zip(get(by_ours), specs):
        check(o, module, name)
    for loaded in (by_pyspark, by_ours):
        assert loaded.uid == saved.uid
        assert uids(get(loaded)) == uids(get(saved))
        assert predictions(loaded, df) == predictions(saved, df)


def test_the_pipelines_fit_save_and_load_native_classifiers_both_ways(binary, scratch):
    def stages():
        return [
            cls(CLASSIFICATION, "DecisionTreeClassifier")(
                maxDepth=3,
                seed=1,
                predictionCol="tree",
                rawPredictionCol="tree_raw",
                probabilityCol="tree_probability",
            ),
            cls(CLASSIFICATION, "LogisticRegression")(maxIter=5),
        ]

    names = ["DecisionTreeClassifier", "LogisticRegression"]
    specs = [(CLASSIFICATION, ESTIMATORS[name][0]) for name in names]
    members = (lambda m: m.stages, specs)
    # PySpark's Pipeline and this client's.
    for meta in (pyspark_class("pipeline", "Pipeline"), cls("pipeline", "Pipeline")):
        model = meta(stages=stages()).fit(binary)
        for stage, (module, name) in zip(model.stages, specs):
            check(stage, module, name)
        path = f"{scratch}/{meta.__module__}"
        model.write().overwrite().save(path)
        _check_loads(model, path, "pipeline", "PipelineModel", members, binary)

        estimator = meta(stages=stages())
        estimator.write().overwrite().save(path + "-estimator")
        by_pyspark = pyspark_class("pipeline", "Pipeline").load(path + "-estimator")
        stage_types = [type(s) for s in by_pyspark.getStages()]
        assert stage_types == [pyspark_class(CLASSIFICATION, name) for name in names]
        by_ours = cls("pipeline", "Pipeline").load(path + "-estimator")
        for stage, name in zip(by_ours.getStages(), names):
            check(stage, CLASSIFICATION, name)
        for loaded in (by_pyspark, by_ours):
            assert uids(loaded.getStages()) == uids(estimator.getStages())


def test_the_one_vs_rests_fit_save_and_load_native_classifiers_both_ways(multiclass, scratch):
    members = (lambda m: m.models, [(CLASSIFICATION, "LogisticRegressionModel")] * 3)
    # PySpark's OneVsRest and this client's.
    for meta in (pyspark_class(CLASSIFICATION, "OneVsRest"), cls(CLASSIFICATION, "OneVsRest")):
        est = meta(classifier=cls(CLASSIFICATION, "LogisticRegression")(maxIter=5))
        model = est.fit(multiclass)
        for submodel in model.models:
            check(submodel, CLASSIFICATION, "LogisticRegressionModel")
        path = f"{scratch}/{meta.__module__}"
        model.write().overwrite().save(path)
        _check_loads(model, path, CLASSIFICATION, "OneVsRestModel", members, multiclass)

        est.write().overwrite().save(path + "-estimator")
        by_pyspark = pyspark_class(CLASSIFICATION, "OneVsRest").load(path + "-estimator")
        lr_class = pyspark_class(CLASSIFICATION, "LogisticRegression")
        assert type(by_pyspark.getClassifier()) is lr_class
        by_ours = cls(CLASSIFICATION, "OneVsRest").load(path + "-estimator")
        check(by_ours.getClassifier(), CLASSIFICATION, "LogisticRegression")
        for loaded in (by_pyspark, by_ours):
            assert loaded.uid == est.uid
            assert loaded.getClassifier().uid == est.getClassifier().uid


@pytest.mark.parametrize("validator", list(VALIDATORS))
def test_the_validators_fit_save_and_load_a_native_classifier_both_ways(
    validator, binary, scratch
):
    from pyspark.ml.tuning import ParamGridBuilder

    members = (lambda m: [m.bestModel], [(CLASSIFICATION, "LogisticRegressionModel")])
    # PySpark's validator and this client's.
    for meta in (pyspark_class("tuning", validator), cls("tuning", validator)):
        lr = cls(CLASSIFICATION, "LogisticRegression")(maxIter=5)
        grid = ParamGridBuilder().addGrid(lr.regParam, [0.01, 0.5]).build()
        evaluator = cls("evaluation", "BinaryClassificationEvaluator")()
        est = meta(
            estimator=lr,
            estimatorParamMaps=grid,
            evaluator=evaluator,
            seed=3,
            **VALIDATORS[validator],
        )
        model = est.fit(binary)
        check(model.bestModel, CLASSIFICATION, "LogisticRegressionModel")
        path = f"{scratch}/{meta.__module__}"
        model.write().overwrite().save(path)
        _check_loads(model, path, "tuning", validator + "Model", members, binary)

        est.write().overwrite().save(path + "-estimator")
        by_pyspark = pyspark_class("tuning", validator).load(path + "-estimator")
        lr_class = pyspark_class(CLASSIFICATION, "LogisticRegression")
        assert type(by_pyspark.getEstimator()) is lr_class
        by_ours = cls("tuning", validator).load(path + "-estimator")
        check(by_ours.getEstimator(), CLASSIFICATION, "LogisticRegression")
        check(by_ours.getEvaluator(), "evaluation", "BinaryClassificationEvaluator")
        for loaded in (by_pyspark, by_ours):
            assert loaded.uid == est.uid
            assert loaded.getEstimator().uid == lr.uid
            assert loaded.getEvaluator().uid == evaluator.uid
