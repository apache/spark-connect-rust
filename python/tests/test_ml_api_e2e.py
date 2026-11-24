# SPDX-License-Identifier: Apache-2.0
"""The pyspark.ml API of the native ML classes that follow it in full, against a live Connect
server: the cases of their models, summaries, transforms and persistence
(`_ml_api_cases.live_cases`) against tests/golden/ml_api_cases.json, from official PySpark (see
scripts/capture_ml_api_cases_golden.py). Official PySpark runs them too, as the reference.
"""

import json
import os
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
def test_live_cases_are_pyspark_mls(name, session, tmp_path):
    # The saves go to a scratch directory of the local file system, of the server that the tests
    # run against on the same machine, as in CI.
    ours = live_cases(name, session, str(tmp_path))
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


def test_a_model_of_a_remote_model_ref_is_of_its_object_and_releases_it_as_pyspark_mls(session):
    # A model class constructs of the RemoteModelRef of a server-side object, as PySpark's
    # `JavaModel(java_model)` over Spark Connect (`LinearSVCModel(RemoteModelRef(id))`): of the
    # members, transforms and saves of the object, and of its lifetime in the ML cache. Its
    # collection releases a reference of the RemoteModelRef, once its transforms are gone, which
    # deletes the object at the last (CONNECT_ML.CACHE_INVALID of the next use of the id), and a
    # reference of one more (`add_ref`) keeps it.
    import gc

    from pyspark.errors.exceptions.connect import SparkException
    from pyspark.ml.linalg import Vectors
    from pyspark.ml.util import RemoteModelRef

    def cls(module, name):
        return native_class(f"pyspark.ml.connect.{module}", name, f"pyspark.ml.{module}")

    rows = [(float(i % 2), Vectors.dense([float(i), float(i % 3)])) for i in range(12)]
    df = session.createDataFrame(rows, ["label", "features"]).coalesce(1)
    for estimator, model_class, member in [
        (cls("classification", "LinearSVC")(maxIter=3), cls("classification", "LinearSVCModel"), "coefficients"),
        (cls("regression", "LinearRegression")(maxIter=3), cls("regression", "LinearRegressionModel"), "intercept"),
        (cls("clustering", "KMeans")(k=2, maxIter=2, seed=1), cls("clustering", "KMeansModel"), "clusterCenters"),
        (cls("feature", "StringIndexer")(inputCol="label", outputCol="i"), cls("feature", "StringIndexerModel"), "labelsArray"),
    ]:
        fitted = estimator.fit(df)
        reference = RemoteModelRef(fitted._java_obj.ref_id)
        # One more reference, which this test keeps: the object outlives the model.
        reference.add_ref()
        model = model_class(reference)
        assert type(model) is model_class and model._java_obj is reference
        assert model.uid != fitted.uid and model.uid.startswith(model_class.__name__ + "_")

        def value(obj, member=member):
            got = getattr(obj, member)
            got = got() if callable(got) else got
            return [list(item) for item in got] if isinstance(got, list) else got

        assert value(model) == value(fitted)
        assert model.transform(df).collect() == fitted.transform(df).collect()
        del model
        gc.collect()
        assert reference._ref_count == 1 and value(fitted) is not None
    # A reference of no more: the object goes with the last of the model and its transforms.
    fitted = cls("classification", "LinearSVC")(maxIter=3).fit(df)
    reference = RemoteModelRef(fitted._java_obj.ref_id)
    model = cls("classification", "LinearSVCModel")(reference)
    transformed = model.transform(df)
    del model
    gc.collect()
    assert reference._ref_count == 1 and transformed.count() == 12
    del transformed
    gc.collect()
    assert reference._ref_count == 0
    with pytest.raises(SparkException, match="CONNECT_ML.CACHE_INVALID"):
        fitted.numFeatures  # noqa: B018


def test_a_model_that_a_finalizer_saves_from_a_collected_cycle_lost_its_object_as_pyspark_mls(
    session, tmp_path
):
    # PySpark's `JavaWrapper.__del__` releases the reference of a model (and of its training
    # summary) of its `RemoteModelRef` when Python finalizes the object, also in a reference cycle
    # that the collector finalizes before it breaks it, and the last release deletes the object: a
    # model that a finalizer of the cycle saves then raises CONNECT_ML.CACHE_INVALID at its
    # members, its summary's and its transforms, of a fit (of a training summary, in a cycle
    # through a function of a UDF of its training dataset, and of none) and of a load. The objects
    # of the cycle that the collector does not finalize (a model that a variable keeps, whose copy
    # the finalizer saves) keep working.
    import gc
    import time

    from pyspark.errors.exceptions.connect import SparkException
    from pyspark.ml.linalg import Vectors
    from pyspark.sql import functions as F

    def cls(module, name):
        return native_class(f"pyspark.ml.connect.{module}", name, f"pyspark.ml.{module}")

    rows = [(float(i % 2), Vectors.dense([float(i), float(i % 3)]), "ab"[i % 2]) for i in range(12)]
    df = session.createDataFrame(rows, ["label", "features", "s"]).coalesce(1)
    saved = []

    class Resurrect:
        def f(self, x):
            return 1.0

        def __del__(self):
            saved.append(self.model)

    def lost(use):
        # The CONNECT_ML.CACHE_INVALID of `use`, of a deletion that this client sends without
        # waiting for it.
        deadline = time.monotonic() + 10
        while True:
            try:
                use()
            except SparkException as e:
                assert "CONNECT_ML.CACHE_INVALID" in str(e)
                return True
            if time.monotonic() > deadline:
                return False
            time.sleep(0.2)

    path = str(tmp_path / "model")
    cls("regression", "LinearRegression")(maxIter=2).fit(df).write().overwrite().save(path)
    shapes = {
        "summary": lambda job: cls("regression", "LinearRegression")(maxIter=2).fit(
            df.withColumn("w", F.udf(job.f, "double")("label"))
        ),
        "no summary": lambda job: cls("feature", "StringIndexer")(inputCol="s", outputCol="i").fit(df),
        "load": lambda job: cls("regression", "LinearRegressionModel").load(path),
    }
    for shape, make in shapes.items():
        job = Resurrect()
        model = make(job)
        job.model = model
        model.job = job
        if shape == "summary":
            summary = model.summary
            summary.model = model
            del summary
        del model, job
        gc.collect()
        assert len(saved) == 1, shape
        (model,) = saved
        assert lost(lambda: model.transform(df).count()), shape
        if shape == "summary":
            assert model.hasSummary
            assert lost(lambda: model.summary.r2), shape
        del saved[:], model
        gc.collect()
    # A copy that the finalizer saves, of a model that lives.
    model = cls("regression", "LinearRegression")(maxIter=2).fit(df)
    job = Resurrect()
    job.model = model.copy()
    job.model.job = job
    del job
    gc.collect()
    assert len(saved) == 1
    assert saved[0].transform(df).count() == 12 and model.transform(df).count() == 12
    del saved[:]


def test_objects_of_this_client_are_of_the_classes_of_pyspark_mls_modules(session, tmp_path):
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
    path = str(tmp_path)
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


def test_the_param_maps_and_the_metadata_of_a_one_vs_rest_are_in_pysparks_order(session, tmp_path):
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
    path = str(tmp_path)
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


def test_a_subclass_reads_the_jvm_class_of_its_python_name(session, tmp_path):
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
    path = str(tmp_path)
    LR(maxIter=3).save(path + "/lr")
    Pipeline(stages=[Binarizer(threshold=0.5, inputCol="a", outputCol="b")]).save(path + "/pipeline")
    with pytest.raises(Exception, match=r"UNSUPPORTED_EXCEPTION.*Unsupported read for .*\.MyLR"):
        MyLR.load(path + "/lr")
    assert type(LR.load(path + "/lr")) is LR and LR.load(path + "/lr").getMaxIter() == 3
    assert type(MyPipeline.load(path + "/pipeline")) is Pipeline


def test_a_writer_reads_the_params_of_its_object_at_its_save(session, tmp_path):
    # As PySpark's `RemoteMLWriter.save`, which reads the params of its `_instance` at the save:
    # a param set after `write()` is saved, of an estimator, a model and a Pipeline's stage.
    def cls(module, name):
        return native_class(f"pyspark.ml.connect.{module}", name, f"pyspark.ml.{module}")

    LR = cls("classification", "LogisticRegression")
    Pipeline = cls("pipeline", "Pipeline")
    Binarizer = cls("feature", "Binarizer")
    path = str(tmp_path)
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


def test_a_writer_and_a_reader_save_and_load_of_their_attributes_as_pysparks(session, tmp_path):
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
    path = str(tmp_path)
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


def test_models_in_cycles_with_their_summaries_are_freed_and_deleted_as_pyspark_mls(session):
    # A model and its training summary in a reference cycle (`summary.model = model`), of no UDF,
    # are freed by the garbage collector, fit after fit, and the objects of the models leave the
    # ML cache of the server, as PySpark's, which hold them as Python attributes: of
    # LinearRegression, LogisticRegression, LinearSVC and KMeans.
    import gc
    import time
    import weakref

    from pyspark.ml.linalg import Vectors

    def cls(module, name):
        return native_class(f"pyspark.ml.connect.{module}", name, f"pyspark.ml.{module}")

    def cached():
        return {json.loads(entry)["id"] for entry in session.client._get_ml_cache_info()}

    rows = [(float(i % 2), Vectors.dense([float(i), float(i % 3)])) for i in range(12)]
    df = session.createDataFrame(rows, ["label", "features"]).coalesce(1)
    for make in [
        lambda: cls("regression", "LinearRegression")(maxIter=2),
        lambda: cls("classification", "LogisticRegression")(maxIter=2),
        lambda: cls("classification", "LinearSVC")(maxIter=2),
        lambda: cls("clustering", "KMeans")(k=2, maxIter=2, seed=1),
    ]:
        kept, ids = [], set()
        # Of no collection until the cycles are made, which keep the objects in the ML cache.
        gc.disable()
        try:
            for _ in range(10):
                model = make().fit(df)
                ids.add(model._java_obj.ref_id)
                summary = model.summary() if callable(model.summary) else model.summary
                summary.model = model
                kept.append((weakref.ref(model), weakref.ref(summary)))
                del model, summary
            assert ids <= cached()
        finally:
            gc.enable()
        gc.collect()
        assert [(m(), s()) for m, s in kept] == [(None, None)] * 10
        # The deletion is sent without waiting for the server.
        for _ in range(100):
            if not ids & cached():
                break
            time.sleep(0.1)
        assert not ids & cached()
