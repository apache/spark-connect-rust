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
"""GeneralizedLinearRegression behaviors, also run with official PySpark."""

import gc
import math
import os

import pytest
from _native_classes import native_class
from pyspark.ml.connect.functions import array_to_vector
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


def _estimator():
    return native_class("pyspark.ml.connect.regression", "GeneralizedLinearRegression", "pyspark.ml.regression")


def _vectors(session, sql):
    return session.sql(sql).withColumn("features", array_to_vector("xs"))


@pytest.mark.parametrize(
    "family,link,power",
    [
        ("gaussian", None, None),
        ("binomial", None, None),
        ("poisson", None, None),
        ("gamma", None, None),
        ("tweedie", None, None),
        ("tweedie", None, 0.0),
        ("gaussian", "log", None),
    ],
)
def test_generalized_linear_regression_families_links_and_offsets(session, family, link, power):
    def response(eta):
        if family == "binomial":
            return 1.0 / (1.0 + math.exp(-eta))
        if family == "gamma":
            return 1.0 / eta
        if family == "tweedie" and power is None:
            return eta**-2
        if family == "poisson" or link == "log" or power == 0.0:
            return math.exp(eta)
        return eta

    rows = []
    for i in range(12):
        x, offset = i / 5.0, (i % 3) / 10.0
        rows.append((Vectors.dense([x]), response(1.0 + 0.3 * x + offset), offset))
    training = session.createDataFrame(rows, ["input", "target", "offset"])
    options = dict(
        featuresCol="input",
        labelCol="target",
        predictionCol="pred",
        family=family,
        offsetCol="offset",
        linkPredictionCol="eta",
        maxIter=100,
        tol=1e-10,
    )
    if family == "tweedie":
        options["variancePower"] = 1.5
    if link is not None:
        options["link"] = link
    if power is not None:
        options["linkPower"] = power
    model = _estimator()(**options).fit(training)
    held_out = session.createDataFrame(
        [
            (0, Vectors.dense([0.7]), 0.15),
            (1, Vectors.sparse(1, {0: 1.3}), 0.25),
            (2, Vectors.sparse(1, {}), 0.0),
        ],
        ["id", "input", "offset"],
    )
    actual = model.transform(held_out).orderBy("id").select("pred", "eta").collect()
    for row, eta in zip(actual, [1.36, 1.64, 1.0]):
        assert row[0] == pytest.approx(response(eta), rel=1e-5, abs=1e-6)
        assert row[1] == pytest.approx(eta, abs=1e-5)


@pytest.mark.parametrize("use_setters", [False, True], ids=["constructor", "setters"])
def test_generalized_linear_regression_weights_and_independent_lazy_models(session, use_setters):
    frame = _vectors(
        session,
        "SELECT array(CAST(x AS DOUBLE)) xs, CAST(y AS DOUBLE) label, CAST(w AS DOUBLE) weight "
        "FROM VALUES (-3,-3,1),(-2,-1,1),(-1,1,1),(1,5,1),(2,7,1),(3,9,1),(100,-1000,0) AS t(x,y,w)",
    )
    options = dict(
        weightCol="weight",
        predictionCol="pred",
        family="gaussian",
        link="identity",
        fitIntercept=True,
        maxIter=50,
        regParam=0.0,
        tol=1e-9,
        solver="irls",
        aggregationDepth=3,
    )
    estimator = _estimator()() if use_setters else _estimator()(**options)
    if use_setters:
        for name, value in options.items():
            assert getattr(estimator, "set" + name[0].upper() + name[1:])(value) is estimator
    first = estimator.fit(frame)
    second = estimator.fit(frame.withColumn("label", F.lit(10.0) - F.col("label")))
    held_out = _vectors(
        session, "SELECT id,array(x) xs FROM VALUES (0,-1.5D),(1,0.5D),(2,4.5D) AS t(id,x)"
    )
    lazy = first.transform(held_out).orderBy("id").select("pred")
    estimator.setPredictionCol("changed")
    del first, estimator
    gc.collect()
    assert [r[0] for r in lazy.collect()] == pytest.approx([0.0, 4.0, 12.0], abs=1e-6)
    assert [
        r[0] for r in second.transform(held_out).orderBy("id").select("pred").collect()
    ] == pytest.approx([10.0, 6.0, -2.0], abs=1e-6)


def test_generalized_linear_regression_defaults_no_intercept_and_regularization(session):
    frame = _vectors(
        session,
        "SELECT array(CAST(x AS DOUBLE)) xs, CAST(2*x AS DOUBLE) label FROM VALUES (-3),(-2),(-1),(1),(2),(3) AS t(x)",
    )
    point = _vectors(session, "SELECT array(2.0D) xs")
    default = _estimator()().fit(frame).transform(point)
    assert default.columns == ["xs", "features", "prediction"]
    assert default.select("prediction").collect()[0][0] == pytest.approx(4.0, abs=1e-6)
    regularized = _estimator()(fitIntercept=False, regParam=2.0).fit(frame)
    value = regularized.transform(point).select("prediction").collect()[0][0]
    assert 0.0 < value < 3.9


@pytest.mark.parametrize(
    "case",
    [
        "family",
        "link",
        "solver",
        "iterations",
        "tolerance",
        "regularization",
        "aggregation",
        "variance",
        "label",
        "weight",
        "nonvector",
        "empty",
    ],
)
def test_generalized_linear_regression_rejects_invalid_params_and_data(session, case):
    frame = _vectors(
        session,
        "SELECT array(CAST(x AS DOUBLE)) xs, CAST(x AS DOUBLE) label FROM VALUES (1),(2),(3),(4) AS t(x)",
    )
    options = {
        "family": dict(family="unknown"),
        "link": dict(family="poisson", link="logit"),
        "solver": dict(solver="unknown"),
        "iterations": dict(maxIter=-1),
        "tolerance": dict(tol=-1.0),
        "regularization": dict(regParam=-1.0),
        "aggregation": dict(aggregationDepth=1),
        "variance": dict(family="tweedie", variancePower=0.5),
    }.get(case, {})
    if case == "label":
        options["family"] = "poisson"
        frame = frame.withColumn("label", F.lit(-1.0))
    elif case == "weight":
        options["weightCol"] = "weight"
        frame = frame.withColumn("weight", F.lit(-1.0))
    elif case == "nonvector":
        frame = frame.withColumn("features", F.lit(1.0))
    elif case == "empty":
        frame = frame.limit(0)
    with pytest.raises(Exception):
        _estimator()(**options).fit(frame)
