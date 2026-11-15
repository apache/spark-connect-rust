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
    validator = cls("tuning", "CrossValidator")(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator, numFolds=2)
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
