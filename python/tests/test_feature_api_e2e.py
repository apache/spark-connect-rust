"""The transforms, fits, models and persistence of the native feature classes, against a live
Connect server: their cases (`_feature_api_cases.live_cases`) against
tests/golden/feature_api_cases.json, from official PySpark (see
scripts/capture_feature_api_cases_golden.py). Official PySpark runs them too, as the reference.
"""

import ast
import json
import math
import os
import shutil
from pathlib import Path

import pytest
from _feature_api_cases import (
    ESTIMATORS,
    MODELS,
    TRANSFORMERS,
    _make,
    _map_param,
    classes,
    fit_frame,
    live_cases,
)
from pyspark.sql import SparkSession

NAMES = TRANSFORMERS + ESTIMATORS + MODELS

CASES = json.loads(
    (Path(__file__).resolve().parents[2] / "tests/golden/feature_api_cases.json").read_text()
)


def _parsed(value):
    # A row of a DataFrame is recorded as the repr of its tuple, and a DataFrame as the repr of its
    # dict of columns and rows.
    if isinstance(value, str) and value[:1] in "({" and value[-1:] in ")}":
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


def _close(ours, golden):
    """Whether a recorded value matches the golden's: a float within the tolerance of the e2e ALS
    tests, since the server computes single-precision results (Word2Vec vectors and similarities,
    scaled values) with last bits that differ between the server of the capture and that of the
    test; anything else exactly."""
    ours, golden = _parsed(ours), _parsed(golden)
    if isinstance(ours, float) and isinstance(golden, float):
        if math.isnan(ours) or math.isnan(golden):
            return math.isnan(ours) and math.isnan(golden)
        return abs(ours - golden) <= 2e-5 + 2e-4 * abs(golden)
    if isinstance(ours, (list, tuple)) and isinstance(golden, (list, tuple)):
        return len(ours) == len(golden) and all(_close(a, b) for a, b in zip(ours, golden))
    if isinstance(ours, dict) and isinstance(golden, dict):
        return ours.keys() == golden.keys() and all(_close(ours[k], golden[k]) for k in golden)
    return ours == golden


@pytest.fixture(scope="module")
def session():
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live feature API tests")
    spark = SparkSession.builder.remote(remote).getOrCreate()
    yield spark
    spark.stop()


@pytest.mark.parametrize("name", NAMES)
def test_live_cases_are_pyspark_mls(name, session, tmp_path):
    # The saves go to a scratch directory of the local file system, of the server that the tests
    # run against on the same machine, as in CI.
    ours = live_cases(name, session, str(tmp_path))
    golden = CASES[name]["live"]
    assert sorted(ours) == sorted(golden)
    differ = {tag: (ours[tag], golden[tag]) for tag in golden if not _close(ours[tag], golden[tag])}
    assert not differ, json.dumps(differ, indent=1)


def _of_its_class(obj):
    # The object is of the class of its module (of PySpark's on official PySpark), which is a
    # subclass of every class of the MRO of PySpark's class (test_ml_api_offline.py).
    assert type(obj) is classes(type(obj).__name__), type(obj)
    return obj


def _extra(stage):
    # A param map of a param of the stage of a value that it has, for a copy with extra params.
    param = next(p for p in stage.params if stage.isDefined(p))
    return {param: stage.getOrDefault(param)}


@pytest.mark.parametrize("name", TRANSFORMERS + ESTIMATORS)
def test_objects_of_this_client_are_of_the_classes_of_pyspark_mls(name, session, tmp_path):
    # Every object of a feature class that this client makes is of the class of its module, as
    # PySpark's: a stage, its copy (of extra params too) and its load (of its class and of read()),
    # and of an estimator, its fit (of a param map, of a list of them and of fitMultiple), the copy
    # and the load of its model, and the Bucketizer of a QuantileDiscretizer.
    base = str(tmp_path)
    try:
        stage = _of_its_class(_make(name))
        _of_its_class(stage.copy())
        _of_its_class(stage.copy(_extra(stage)))
        stage.write().overwrite().save(base + "/stage")
        _of_its_class(classes(name).load(base + "/stage"))
        _of_its_class(classes(name).read().load(base + "/stage"))
        if name not in ESTIMATORS:
            return
        df = fit_frame(session)
        model = _of_its_class(stage.fit(df))
        _of_its_class(stage.fit(df, {getattr(classes(name), _map_param(stage)): "mapped"}))
        for fitted in stage.fit(df, [{}]):
            _of_its_class(fitted)
        for _, fitted in stage.fitMultiple(df, [{}]):
            _of_its_class(fitted)
        _of_its_class(model.copy())
        _of_its_class(model.copy(_extra(model)))
        model.write().overwrite().save(base + "/model")
        _of_its_class(type(model).load(base + "/model"))
        _of_its_class(type(model).read().load(base + "/model"))
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_models_of_the_class_methods_are_of_the_classes_of_pyspark_mls(session):
    # The models that the class methods make of given values, as PySpark's.
    vocabulary = classes("CountVectorizerModel").from_vocabulary(["a", "b"], inputCol="tokens")
    _of_its_class(vocabulary)
    _of_its_class(classes("StringIndexerModel").from_labels(["a", "b"], inputCol="s"))
    indexer = classes("StringIndexerModel").from_arrays_of_labels(
        [["a"], ["b"]], inputCols=["s", "t"]
    )
    _of_its_class(indexer)


def test_a_pipeline_of_pyspark_ml_saves_and_loads_the_feature_stages(session, tmp_path):
    # pyspark.ml's own Pipeline over the feature stages: its fit makes their models, of the classes
    # of their module, and its save writes each stage with its own writer, as they are MLWritable.
    # Its load reads them as PySpark's classes of pyspark.ml.feature, which transform as the fitted
    # pipeline does.
    from pyspark.ml import Pipeline, PipelineModel

    df = session.createDataFrame(
        [("a b c", "x", 1.0), ("b c", "y", 2.0), ("a a d", "x", 3.0)], ["s", "cat", "n"]
    ).coalesce(1)
    stages = [
        classes("Tokenizer")(inputCol="s", outputCol="words"),
        classes("HashingTF")(inputCol="words", outputCol="tf", numFeatures=16),
        classes("IDF")(inputCol="tf", outputCol="idf"),
        classes("StringIndexer")(inputCol="cat", outputCol="index"),
        classes("VectorAssembler")(inputCols=["idf", "index", "n"], outputCol="features"),
        classes("StandardScaler")(inputCol="features", outputCol="scaled"),
    ]
    model = Pipeline(stages=stages).fit(df)
    for stage in model.stages:
        _of_its_class(stage)
    base = str(tmp_path)
    try:
        Pipeline(stages=stages).write().overwrite().save(base + "/pipeline")
        assert [type(s).__name__ for s in Pipeline.load(base + "/pipeline").getStages()] == [
            "Tokenizer", "HashingTF", "IDF", "StringIndexer", "VectorAssembler", "StandardScaler"
        ]
        model.write().overwrite().save(base + "/model")
        loaded = PipelineModel.load(base + "/model")
        assert [type(s).__module__ for s in loaded.stages] == ["pyspark.ml.feature"] * 6
        assert sorted(map(str, loaded.transform(df).select("scaled").collect())) == sorted(
            map(str, model.transform(df).select("scaled").collect())
        )
    finally:
        shutil.rmtree(base, ignore_errors=True)
