# SPDX-License-Identifier: Apache-2.0
"""The pyspark.ml API of the native frequent-pattern and recommendation classes that follow it in
full, against a live Connect server: the cases of their fits, models, transforms, recommendations
and persistence (`_fpm_recommendation_api_cases.live_cases`) against
tests/golden/fpm_recommendation_api_cases.json, from official PySpark (see
scripts/capture_fpm_recommendation_api_cases_golden.py). Official PySpark runs them too, as the
reference.
"""

import ast
import json
import math
import os
import re
import shutil
import tempfile
from pathlib import Path

import pytest
from _fpm_recommendation_api_cases import (
    CLASSES,
    _als_frame,
    _fpgrowth_frame,
    classes,
    live_cases,
)
from pyspark.sql import SparkSession

CASES = json.loads(
    (Path(__file__).resolve().parents[2] / "tests/golden/fpm_recommendation_api_cases.json").read_text()
)


def _parsed(value):
    # A row of a DataFrame is recorded as the repr of its tuple.
    if isinstance(value, str) and value.startswith("(") and value.endswith(")"):
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value
    return value


# A decimal number in a recorded text, such as a rating in the repr of a nested Row.
_DECIMAL = re.compile(r"-?\d+\.\d+(?:[eE][-+]?\d+)?")


def _close_float(ours, golden):
    if math.isnan(ours) or math.isnan(golden):
        return math.isnan(ours) and math.isnan(golden)
    return abs(ours - golden) <= 2e-5 + 2e-4 * abs(golden)


def _close_text(ours, golden):
    # A text that is not a literal, such as a row with a nested Row: the same text around its
    # decimal numbers, which are close.
    numbers = _DECIMAL.findall(ours), _DECIMAL.findall(golden)
    return _DECIMAL.sub("#", ours) == _DECIMAL.sub("#", golden) and all(
        _close_float(float(a), float(b)) for a, b in zip(*numbers)
    )


def _close(ours, golden):
    """Whether a recorded value matches the golden's: a float within the tolerance of the e2e ALS
    tests, since the server computes ALS factors and predictions in single precision, whose last
    bits differ between the server of the capture and that of the test; anything else exactly."""
    ours, golden = _parsed(ours), _parsed(golden)
    if isinstance(ours, float) and isinstance(golden, float):
        return _close_float(ours, golden)
    if isinstance(ours, str) and isinstance(golden, str):
        return _close_text(ours, golden)
    if isinstance(ours, (list, tuple)) and isinstance(golden, (list, tuple)):
        return len(ours) == len(golden) and all(_close(a, b) for a, b in zip(ours, golden))
    if isinstance(ours, dict) and isinstance(golden, dict):
        return ours.keys() == golden.keys() and all(_close(ours[k], golden[k]) for k in golden)
    return ours == golden


@pytest.fixture(scope="module")
def session():
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live fpm and recommendation API tests")
    spark = SparkSession.builder.remote(remote).getOrCreate()
    yield spark
    spark.stop()


@pytest.mark.parametrize("name", list(CLASSES))
def test_live_cases_are_pyspark_mls(name, session):
    # The saves go to a scratch directory of the local file system, of the server that the tests
    # run against on the same machine, as in CI.
    ours = live_cases(name, session, tempfile.mkdtemp(prefix="fpm-recommendation-api-cases-"))
    golden = CASES[name]["live"]
    assert sorted(ours) == sorted(golden)
    differ = {tag: (ours[tag], golden[tag]) for tag in golden if not _close(ours[tag], golden[tag])}
    assert not differ, json.dumps(differ, indent=1)


def _class_of(name):
    # The class `name` of the case sets: this client's, or PySpark's on official PySpark.
    for set_name, specs in CLASSES.items():
        for spec, cls in zip(specs, classes(set_name)):
            if spec[1] == name:
                return cls
    raise KeyError(name)


def _of_its_class(obj):
    # The object is of the class of its module (of PySpark's on official PySpark), which is a
    # subclass of every class of the MRO of PySpark's class (test_ml_api_offline.py).
    assert type(obj) is _class_of(type(obj).__name__), type(obj)
    return obj


def test_objects_of_this_client_are_of_the_classes_of_pyspark_mls(session):
    # Every FPGrowth, PrefixSpan and ALS object that this client makes is of the class of its
    # module, as PySpark's: an estimator, its copy and its load, its fit (of a param map, of a list
    # of them and of fitMultiple), and the copy and the load of its model.
    base = tempfile.mkdtemp(prefix="fpm-recommendation-api-classes-")
    try:
        _of_its_class(_of_its_class(_class_of("PrefixSpan")(minSupport=0.5)).copy())
        for name, df, kwargs in (
            ("FPGrowth", _fpgrowth_frame(session), {"minSupport": 0.4}),
            ("ALS", _als_frame(session), {"rank": 2, "maxIter": 2, "seed": 1}),
        ):
            estimator = _of_its_class(_class_of(name)(**kwargs))
            _of_its_class(estimator.copy())
            estimator.write().overwrite().save(base + "/estimator")
            _of_its_class(_class_of(name).load(base + "/estimator"))
            model = _of_its_class(estimator.fit(df))
            _of_its_class(estimator.fit(df, {estimator.predictionCol: "mapped"}))
            for fitted in estimator.fit(df, [{}]):
                _of_its_class(fitted)
            for _, fitted in estimator.fitMultiple(df, [{}]):
                _of_its_class(fitted)
            _of_its_class(model.copy())
            model.write().overwrite().save(base + "/model")
            _of_its_class(type(model).load(base + "/model"))
    finally:
        shutil.rmtree(base, ignore_errors=True)


def test_the_validators_of_pyspark_ml_save_and_load_an_als_and_its_evaluator(session):
    # pyspark.ml's own validators over an ALS and a RegressionEvaluator: their fit makes the best
    # model, of the class of its module, and their saves write the estimator and the evaluator with
    # their own writers, as they are MLWritable. Their loads read them as PySpark's classes, and the
    # loaded model transforms.
    from _native_classes import native_class
    from pyspark.ml.tuning import (
        CrossValidator,
        CrossValidatorModel,
        ParamGridBuilder,
        TrainValidationSplit,
        TrainValidationSplitModel,
    )

    evaluator = native_class(
        "pyspark.ml.connect.evaluation", "RegressionEvaluator", "pyspark.ml.evaluation"
    )(labelCol="rating")
    rows = [(user, item, float((user * 3 + item) % 5 + 1)) for user in range(4) for item in range(4)]
    df = session.createDataFrame(rows, ["user", "item", "rating"]).coalesce(1)
    als = _class_of("ALS")(rank=2, maxIter=2, seed=1, coldStartStrategy="drop")
    grid = ParamGridBuilder().addGrid(als.regParam, [0.1, 0.5]).build()
    base = tempfile.mkdtemp(prefix="fpm-recommendation-api-validators-")
    try:
        for validator, model_class, kwargs in (
            (CrossValidator, CrossValidatorModel, {"numFolds": 2}),
            (TrainValidationSplit, TrainValidationSplitModel, {"trainRatio": 0.75}),
        ):
            estimator = validator(
                estimator=als, estimatorParamMaps=grid, evaluator=evaluator, seed=1, **kwargs
            )
            fitted = estimator.fit(df)
            _of_its_class(fitted.bestModel)
            estimator.write().overwrite().save(base + "/validator")
            loaded = validator.load(base + "/validator")
            assert type(loaded.getEstimator()).__module__ == "pyspark.ml.recommendation"
            assert type(loaded.getEvaluator()).__module__ == "pyspark.ml.evaluation"
            fitted.write().overwrite().save(base + "/model")
            assert model_class.load(base + "/model").transform(df).count() == 16
    finally:
        shutil.rmtree(base, ignore_errors=True)
