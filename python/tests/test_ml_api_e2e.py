# SPDX-License-Identifier: Apache-2.0
"""The pyspark.ml API of the native ML classes that follow it in full, against a live Connect
server: the cases of their models, summaries, transforms and persistence
(`_ml_api_cases.live_cases`) against tests/golden/ml_api_cases.json, from official PySpark (see
scripts/capture_ml_api_cases_golden.py). Official PySpark runs them too, as the reference.
"""

import json
import os
import tempfile
from pathlib import Path

import _ml_unbound_cases
import pytest
from _ml_api_cases import CLASSES, live_cases
from _native_classes import native_class
from pyspark.sql import SparkSession

CASES = json.loads(
    (Path(__file__).resolve().parents[2] / "tests/golden/ml_api_cases.json").read_text()
)


@pytest.fixture(scope="module")
def session():
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live ML API tests")
    spark = SparkSession.builder.remote(remote).getOrCreate()
    yield spark
    spark.stop()


@pytest.mark.parametrize("name", list(CLASSES))
def test_live_cases_are_pyspark_mls(name, session):
    # The saves go to a scratch directory of the local file system, of the server that the tests
    # run against on the same machine, as in CI.
    ours = live_cases(name, session, tempfile.mkdtemp(prefix="ml-api-cases-"))
    golden = CASES[name]["live"]
    assert sorted(ours) == sorted(golden)
    differ = {tag: (ours[tag], golden[tag]) for tag in golden if ours[tag] != golden[tag]}
    assert not differ, json.dumps(differ, indent=1)


UNBOUND = json.loads(_ml_unbound_cases.GOLDEN.read_text())


@pytest.mark.parametrize("name", _ml_unbound_cases.MODELS)
def test_models_of_no_server_object_are_pyspark_mls(name, session):
    # A model of no server-side object, as PySpark's `JavaModel(java_model=None)` makes one: of its
    # params and their defaults, and of the AssertionError of its members of the server, a
    # transform, a save and the `params` walk of a model of cached members
    # (tests/golden/ml_unbound_models.json).
    assert _ml_unbound_cases.outcomes(session, name) == UNBOUND["models"][name]


def test_objects_of_this_client_are_of_the_classes_of_pyspark_mls_modules(session):
    # Every object that this client makes, as PySpark's do, is of the class of its module, of every
    # base of PySpark's class (test_ml_api_offline.py): of a fit (of a param map, of fitMultiple),
    # a copy, a load, a summary, the trees of an ensemble (regression models of a classifier), the
    # models of a OneVsRest, the best model of a validator, toLocal and the stages of a pipeline.
    # A Pipeline of pyspark.ml saves native stages, which are its MLWritable stages.
    from pyspark.ml import Pipeline as PySparkPipeline
    from pyspark.ml import PipelineModel as PySparkPipelineModel
    from pyspark.ml.linalg import Vectors
    from pyspark.ml.tuning import ParamGridBuilder

    def cls(module, name):
        return native_class(f"pyspark.ml.connect.{module}", name, f"pyspark.ml.{module}")

    def check(obj, module):
        assert type(obj) is cls(module, type(obj).__name__), (type(obj), module)
        return obj

    rows = [(float(i % 2), Vectors.dense([float(i), float(i % 3)])) for i in range(12)]
    df = session.createDataFrame(rows, ["label", "features"]).coalesce(1)
    path = tempfile.mkdtemp(prefix="ml-public-classes-")
    lr = cls("classification", "LogisticRegression")(maxIter=3)
    model = check(lr.fit(df), "classification")
    check(model.summary, "classification")
    check(model.evaluate(df), "classification")
    check(model.copy(), "classification")
    check(lr.fit(df, {lr.maxIter: 2}), "classification")
    for _, fitted in lr.fitMultiple(df, [{lr.maxIter: 2}]):
        check(fitted, "classification")
    model.write().overwrite().save(path + "/lr")
    check(cls("classification", "LogisticRegressionModel").load(path + "/lr"), "classification")
    lr.write().overwrite().save(path + "/lr-estimator")
    check(cls("classification", "LogisticRegression").load(path + "/lr-estimator"), "classification")
    gbt = check(cls("classification", "GBTClassifier")(maxIter=2).fit(df), "classification")
    for tree in gbt.trees:
        check(tree, "regression")
    forest = check(cls("classification", "RandomForestClassifier")(numTrees=2).fit(df), "classification")
    for tree in forest.trees:
        check(tree, "classification")
    check(forest.summary, "classification")
    ovr = cls("classification", "OneVsRest")(classifier=cls("classification", "LogisticRegression")(maxIter=2))
    for submodel in check(ovr.fit(df), "classification").models:
        check(submodel, "classification")
    check(check(cls("regression", "LinearRegression")(maxIter=2).fit(df), "regression").summary, "regression")
    check(check(cls("clustering", "KMeans")(k=2, maxIter=2).fit(df), "clustering").summary, "clustering")
    lda = check(cls("clustering", "LDA")(k=2, maxIter=2, optimizer="em").fit(df), "clustering")
    check(lda.toLocal(), "clustering")
    evaluator = cls("evaluation", "BinaryClassificationEvaluator")()
    grid = ParamGridBuilder().addGrid(lr.maxIter, [2]).build()
    # A seed of folds of both labels: of a random seed (PySpark's of no seed differs between
    # processes), a fold of 12 rows of one label fits a model of one class, whose rawPrediction the
    # binary evaluator refuses (seed 2 does).
    validator = cls("tuning", "CrossValidator")(
        estimator=lr, estimatorParamMaps=grid, evaluator=evaluator, numFolds=2, seed=1
    )
    check(check(validator.fit(df), "tuning").bestModel, "classification")
    scaler = cls("feature", "StandardScaler")(inputCol="features", outputCol="scaled")
    stages = [scaler, cls("classification", "LogisticRegression")(maxIter=2)]
    pipeline = check(cls("pipeline", "Pipeline")(stages=stages).fit(df), "pipeline")
    pipeline.write().overwrite().save(path + "/pipeline")
    loaded = check(cls("pipeline", "PipelineModel").load(path + "/pipeline"), "pipeline")
    for stage, module in zip(loaded.stages, ["feature", "classification"]):
        check(stage, module)
    ours = PySparkPipeline(stages=[cls("classification", "LogisticRegression")(maxIter=2)]).fit(df)
    ours.write().overwrite().save(path + "/pyspark-pipeline")
    assert PySparkPipelineModel.load(path + "/pyspark-pipeline").transform(df).count() == 12


def test_the_param_maps_of_the_model_of_a_validator_are_in_pysparks_order(session):
    # The model of a validator's fit is of the constructor of PySpark's class
    # (`CrossValidatorModel(bestModel, ...)`), whose defaults are in the order of its MRO
    # (test_ml_api_offline.py), and then of the `_copyValues` of the validator, which sets its
    # params in the order of their names.
    from pyspark.ml.linalg import Vectors
    from pyspark.ml.tuning import ParamGridBuilder

    def cls(module, name):
        return native_class(f"pyspark.ml.connect.{module}", name, f"pyspark.ml.{module}")

    rows = [(float(i % 2), Vectors.dense([float(i), float(i % 3)])) for i in range(12)]
    df = session.createDataFrame(rows, ["label", "features"]).coalesce(1)
    lr = cls("classification", "LogisticRegression")(maxIter=2)
    grid = ParamGridBuilder().addGrid(lr.regParam, [0.1]).build()
    evaluator = cls("evaluation", "BinaryClassificationEvaluator")()
    shared = {"estimator": lr, "estimatorParamMaps": grid, "evaluator": evaluator, "seed": 1}
    for name, extra, defaults, given in [
        ("CrossValidator", {"numFolds": 2}, ["seed", "numFolds", "foldCol"], ["numFolds", "seed"]),
        ("TrainValidationSplit", {}, ["seed", "trainRatio"], ["seed"]),
    ]:
        model = cls("tuning", name)(**shared, **extra).fit(df)
        assert [p.name for p in model._defaultParamMap] == defaults, name
        assert [p.name for p in model._paramMap] == ["estimator", "estimatorParamMaps", "evaluator", *given], name
        assert [p.name for p in model.extractParamMap()][: len(defaults)] == defaults, name


def test_the_param_maps_and_the_metadata_of_a_one_vs_rest_are_in_pysparks_order(session):
    # A OneVsRest's model of a fit and of `OneVsRestModel(models)` has the defaults of the
    # constructor in its order (test_ml_offline.py). A load is of `_resetUid`, of the defaults in
    # the order of their names, and sets the classifier and then the params of the metadata in its
    # order; a save writes the metadata of `extractParamMap()` and `_defaultParamMap` in their
    # order, so of the constructor's of a OneVsRest of its constructor, and of the names of a load
    # or a copy.
    import glob

    from pyspark.ml.linalg import Vectors

    def cls(name):
        return native_class("pyspark.ml.connect.classification", name, "pyspark.ml.classification")

    def maps(obj):
        return [p.name for p in obj._defaultParamMap], [p.name for p in obj._paramMap]

    def metadata(path):
        (part,) = glob.glob(os.path.join(path, "metadata", "part-*"))
        with open(part) as file:
            saved = json.load(file)
        return list(saved["paramMap"]), list(saved["defaultParamMap"])

    rows = [(float(i % 3), Vectors.dense([float(i % 3) + 0.1 * i, float(i % 2)]), 1.0) for i in range(12)]
    df = session.createDataFrame(rows, ["label", "features", "w"]).coalesce(1)
    path = tempfile.mkdtemp(prefix="ml-one-vs-rest-order-")
    estimator = cls("OneVsRest")(classifier=cls("LogisticRegression")(maxIter=2), weightCol="w")
    model = estimator.fit(df)
    constructor = ["parallelism", "predictionCol", "featuresCol", "labelCol", "rawPredictionCol"]
    names = sorted(constructor)
    model_constructor, model_names = constructor[1:], sorted(constructor[1:])
    assert maps(model) == (model_constructor, ["classifier", "weightCol"])
    for obj, at, (params, defaults) in [
        (estimator, "/estimator", (constructor + ["weightCol"], constructor)),
        (model, "/model", (model_constructor + ["weightCol"], model_constructor)),
    ]:
        obj.write().overwrite().save(path + at)
        assert metadata(path + at) == (params, defaults)
        loaded = type(obj).load(path + at)
        assert maps(loaded) == (names if obj is estimator else model_names, ["classifier", *params])
        loaded.write().overwrite().save(path + at + "-again")
        resaved = names if obj is estimator else model_names
        assert metadata(path + at + "-again") == (resaved + ["weightCol"], resaved)
        obj.copy().write().overwrite().save(path + at + "-copy")
        assert metadata(path + at + "-copy") == (resaved + ["weightCol"], resaved)


def test_a_subclass_reads_the_jvm_class_of_its_python_name(session):
    # PySpark's RemoteMLReader names the JVM class of a load of the Python class (its module, of
    # org.apache.spark for pyspark, and its name), so a subclass of the user's reads a class that
    # the server does not have, and raises its UNSUPPORTED_EXCEPTION; the class's own load, and a
    # Pipeline's (of its Python format, which loads a Pipeline), read their own.
    def cls(module, name):
        return native_class(f"pyspark.ml.connect.{module}", name, f"pyspark.ml.{module}")

    LR = cls("classification", "LogisticRegression")
    Pipeline = cls("pipeline", "Pipeline")
    Binarizer = cls("feature", "Binarizer")
    MyLR = type("MyLR", (LR,), {})
    MyPipeline = type("MyPipeline", (Pipeline,), {})
    path = tempfile.mkdtemp(prefix="ml-subclass-load-")
    LR(maxIter=3).save(path + "/lr")
    Pipeline(stages=[Binarizer(threshold=0.5, inputCol="a", outputCol="b")]).save(path + "/pipeline")
    with pytest.raises(Exception, match=r"UNSUPPORTED_EXCEPTION.*Unsupported read for .*\.MyLR"):
        MyLR.load(path + "/lr")
    assert type(LR.load(path + "/lr")) is LR and LR.load(path + "/lr").getMaxIter() == 3
    assert type(MyPipeline.load(path + "/pipeline")) is Pipeline


def test_a_writer_reads_the_params_of_its_object_at_its_save(session):
    # As PySpark's `RemoteMLWriter.save`, which reads the params of its `_instance` at the save:
    # a param set after `write()` is saved, of an estimator, a model and a Pipeline's stage.
    def cls(module, name):
        return native_class(f"pyspark.ml.connect.{module}", name, f"pyspark.ml.{module}")

    LR = cls("classification", "LogisticRegression")
    Pipeline = cls("pipeline", "Pipeline")
    Binarizer = cls("feature", "Binarizer")
    path = tempfile.mkdtemp(prefix="ml-writer-at-save-")
    lr = LR(maxIter=3)
    writer = lr.write()
    assert writer._instance is lr
    lr.setMaxIter(7)
    writer.save(path + "/lr")
    assert LR.load(path + "/lr").getMaxIter() == 7
    stage = Binarizer(threshold=0.1, inputCol="a", outputCol="b")
    pipeline = Pipeline(stages=[stage])
    writer = pipeline.write()
    assert writer._instance is pipeline
    stage.setThreshold(0.9)
    writer.save(path + "/pipeline")
    assert Pipeline.load(path + "/pipeline").getStages()[0].getThreshold() == 0.9


def test_a_writer_and_a_reader_save_and_load_of_their_attributes_as_pysparks(session):
    # As PySpark's `RemoteMLWriter.save` and `RemoteMLReader.load`, which read the writer's
    # `_instance`, `shouldOverwrite` and `optionMap` and the reader's `_clazz` at the call, of a
    # copy and a pickle of them too: a `Write` takes an int as its bool of the overwrite, and a save
    # of PySpark's Python format sends the value as it is to the ML helper's `handleOverwrite`,
    # whose `(String, int)` the server does not have.
    import copy
    import pickle

    def cls(module, name):
        return native_class(f"pyspark.ml.connect.{module}", name, f"pyspark.ml.{module}")

    LR = cls("classification", "LogisticRegression")
    Pipeline = cls("pipeline", "Pipeline")
    Binarizer = cls("feature", "Binarizer")
    path = tempfile.mkdtemp(prefix="ml-writer-attributes-")
    writer = LR(maxIter=3).write()
    writer._instance = LR(maxIter=9)
    writer.save(path + "/lr")
    assert LR.load(path + "/lr").getMaxIter() == 9
    writer.shouldOverwrite = 0
    with pytest.raises(Exception, match="already exists"):
        writer.save(path + "/lr")
    writer.shouldOverwrite, writer._instance = 1, LR(maxIter=4)
    copy.copy(writer).save(path + "/lr")
    assert LR.load(path + "/lr").getMaxIter() == 4
    pickle.loads(pickle.dumps(writer)).save(path + "/lr")
    type(writer)(LR(maxIter=5)).overwrite().save(path + "/lr")
    assert pickle.loads(pickle.dumps(LR.read())).load(path + "/lr").getMaxIter() == 5
    pipeline = Pipeline(stages=[Binarizer(threshold=0.5, inputCol="a", outputCol="b")])
    pipeline.save(path + "/pipeline")
    reader = LR.read()
    reader._clazz = Pipeline
    assert type(reader.load(path + "/pipeline")) is Pipeline
    writer = pipeline.write()
    writer.shouldOverwrite = 1
    with pytest.raises(Exception, match=r"handleOverwrite\(java\.lang\.String, int\)"):
        writer.save(path + "/pipeline")
