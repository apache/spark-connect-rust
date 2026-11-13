#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Regression behavior tests, runnable unchanged with native or official PySpark."""

import gc
import json
import os
from pathlib import Path

import pytest
from _native_classes import native_class
from pyspark.errors import PySparkException
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


@pytest.fixture
def session():
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live tests")
    spark = SparkSession.builder.remote(remote).getOrCreate()
    try:
        yield spark
    finally:
        spark.stop()


def _regressor(name):
    return native_class("pyspark.ml.connect.regression", name, "pyspark.ml.regression")


def _estimator(name, options, setters):
    cls = _regressor(name)
    if not setters:
        return cls(**options)
    estimator = cls()
    for key, value in options.items():
        assert getattr(estimator, "set" + key[0].upper() + key[1:])(value) is estimator
    return estimator


def _fixture():
    return json.loads((Path(__file__).resolve().parents[2] / "tests/golden/remaining_regressors.json").read_text())


@pytest.mark.parametrize("case_index", range(4))
@pytest.mark.parametrize("setters", [False, True], ids=["constructor", "setters"])
def test_aft_matches_official_censoring_quantiles_and_intercept(session, case_index, setters):
    fixture = _fixture()
    case = fixture["aft"][case_index]
    rows = [(i, y, c if case["censored"] else 1.0, Vectors.dense(x)) for i, (y, c, x) in enumerate(fixture["aft_rows"])]
    train = session.createDataFrame(rows, ["id", "duration", "observed", "input"])
    estimator = _estimator("AFTSurvivalRegression", case["options"], setters)
    model = estimator.fit(train)
    lazy = model.transform(train).orderBy("id").select("pred", "quantiles")
    # A lazy transform must own its cached model after the Python handle is gone.
    del model
    gc.collect()
    result = lazy.collect()
    assert [r.pred for r in result] == pytest.approx(case["predictions"], rel=1e-6)
    for row, expected in zip(result, case["quantiles"]):
        assert list(row.quantiles) == pytest.approx(expected, rel=1e-6)
        assert 0 < row.quantiles[0] < row.quantiles[1] < row.quantiles[2]
    # Scaling every survival time scales predictions, including censored observations.
    if case["options"]["fitIntercept"]:
        scaled = estimator.fit(train.withColumn("duration", F.col("duration") * 2.0))
        predictions = scaled.transform(train).orderBy("id").select("pred").collect()
        assert [r.pred for r in predictions] == pytest.approx([x * 2.0 for x in case["predictions"]], rel=2e-5)


@pytest.mark.parametrize("features", ["scalar", "dense", "sparse"])
@pytest.mark.parametrize("isotonic", [True, False])
@pytest.mark.parametrize("setters", [False, True], ids=["constructor", "setters"])
def test_isotonic_weights_feature_index_and_interpolation(session, features, isotonic, setters):
    def feature(x):
        if features == "scalar":
            return float(x)
        if features == "dense":
            return Vectors.dense([99.0, x])
        return Vectors.sparse(2, [0, 1], [99.0, float(x)])

    sign = 1.0 if isotonic else -1.0
    rows = [(feature(x), sign * y, w) for x, y, w in [(0, 0.0, 1.0), (1, 3.0, 1.0), (2, 1.0, 3.0), (3, 4.0, 1.0)]]
    train = session.createDataFrame(rows, ["input", "target", "weight"])
    options = dict(featuresCol="input", labelCol="target", predictionCol="pred", weightCol="weight", isotonic=isotonic, featureIndex=0 if features == "scalar" else 1)
    estimator = _estimator("IsotonicRegression", options, setters)
    held = session.createDataFrame([(i, feature(x)) for i, x in enumerate([-1.0, 0.5, 1.5, 2.5, 4.0])], ["id", "input"])
    model = estimator.fit(train)
    lazy = model.transform(held).orderBy("id")
    del model
    gc.collect()
    assert [r.pred for r in lazy.collect()] == pytest.approx([sign*x for x in [0.0, 0.75, 1.5, 2.75, 4.0]])
    # Without weightCol, the pooled block is 2.0 instead of the weighted 1.5.
    options.pop("weightCol")
    unweighted = _regressor("IsotonicRegression")(**options).fit(train).transform(held).orderBy("id")
    assert [r.pred for r in unweighted.collect()] == pytest.approx([sign*x for x in [0.0, 1.0, 2.0, 3.0, 4.0]])


@pytest.mark.parametrize("solver", ["adamW", "gd"])
@pytest.mark.parametrize("setters", [False, True], ids=["constructor", "setters"])
def test_fm_regressor_learns_interactions_with_both_solvers(session, solver, setters):
    rows = [(Vectors.dense([x, y]), x*y) for x in [-1.0, 0.0, 1.0] for y in [-1.0, 0.0, 1.0]]
    train = session.createDataFrame(rows, ["input", "target"])
    options = dict(featuresCol="input", labelCol="target", predictionCol="pred", factorSize=2, fitIntercept=False, fitLinear=False, regParam=0.0, miniBatchFraction=1.0, initStd=0.1, maxIter=300, stepSize=1.0 if solver == "gd" else 0.1, tol=1e-8, solver=solver, seed=42)
    estimator = _estimator("FMRegressor", options, setters)
    held = session.createDataFrame([(0, Vectors.dense([0.5, -2.0])), (1, Vectors.sparse(2, [0, 1], [-0.5, -1.0])), (2, Vectors.dense([0.0, 1.5]))], ["id", "input"])
    first = estimator.fit(train)
    second = estimator.fit(train.withColumn("target", -F.col("target")))
    lazy = first.transform(held).orderBy("id")
    other = second.transform(held).orderBy("id")
    del first, second
    gc.collect()
    assert [r.pred for r in lazy.collect()] == pytest.approx([-1.0, 0.5, 0.0], abs=0.025)
    assert [r.pred for r in other.collect()] == pytest.approx([1.0, -0.5, 0.0], abs=0.025)


def test_fm_regressor_fits_intercept_and_linear_terms(session):
    train = session.createDataFrame([(Vectors.dense([x]), 2*x+1) for x in [-1.0, -0.5, 0.0, 0.5, 1.0]], ["features", "label"])
    model = _regressor("FMRegressor")(factorSize=2, maxIter=300, stepSize=0.1, tol=1e-8, seed=42).fit(train)
    values = [r.prediction for r in model.transform(train).collect()]
    assert values == pytest.approx([-1.0, 0.0, 1.0, 2.0, 3.0], abs=0.025)


@pytest.mark.parametrize("name,options", [
    ("AFTSurvivalRegression", {"quantileProbabilities": [0.0, 1.0]}),
    ("AFTSurvivalRegression", {"aggregationDepth": 1}),
    ("AFTSurvivalRegression", {"maxBlockSizeInMB": -1.0}),
    ("IsotonicRegression", {"featureIndex": -1}),
    ("FMRegressor", {"factorSize": 0}),
    ("FMRegressor", {"solver": "invalid"}),
    ("FMRegressor", {"miniBatchFraction": 0.0}),
    ("FMRegressor", {"regParam": -1.0}),
])
def test_regressors_reject_invalid_parameters(session, name, options):
    train = session.createDataFrame([(Vectors.dense([0.0]), 1.0, 1.0), (Vectors.dense([1.0]), 2.0, 0.0)], ["features", "label", "censor"])
    estimator = _regressor(name)(**options)
    with pytest.raises((ValueError, RuntimeError, PySparkException)):
        estimator.fit(train)


@pytest.mark.parametrize("column,value", [("label", 0.0), ("label", -1.0), ("censor", 2.0)])
def test_aft_rejects_nonpositive_survival_times_and_invalid_censor(session, column, value):
    train = session.createDataFrame([(Vectors.dense([0.0]), 1.0, 1.0), (Vectors.dense([1.0]), 2.0, 0.0)], ["features", "label", "censor"])
    estimator = _regressor("AFTSurvivalRegression")()
    with pytest.raises((ValueError, RuntimeError, PySparkException)):
        estimator.fit(train.withColumn(column, F.lit(value)))
