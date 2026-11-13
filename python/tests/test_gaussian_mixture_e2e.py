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
"""Native GaussianMixture behaviors, also run with official PySpark."""

import gc
import math
import os

import pytest
from _native_classes import native_class
from pyspark.ml.connect.functions import array_to_vector, vector_to_array
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
    return native_class("pyspark.ml.connect.clustering", "GaussianMixture", "pyspark.ml.clustering")


def _vectors(session, sql, name="features"):
    return session.sql(sql).withColumn(name, array_to_vector("xs"))


def _training(session, shift=0, name="features"):
    return _vectors(
        session,
        f"SELECT array(CASE WHEN id<40 THEN -5.0D ELSE 5.0D END + "
        f"{shift} + sin(id)*0.5D) xs, 1.0D weight FROM range(80)",
        name,
    )


def _outputs(model, frame, prediction="prediction", probability="probability"):
    return (
        model.transform(frame)
        .orderBy("id")
        .select(prediction, vector_to_array(probability).alias("p"))
        .collect()
    )


def _assert_partition(values, expected):
    assert len(values) == len(expected)
    for i in range(len(values)):
        for j in range(len(values)):
            assert (values[i] == values[j]) == (expected[i] == expected[j]), values


@pytest.mark.parametrize("use_setters", [False, True], ids=["constructor", "setters"])
def test_gaussian_mixture_learns_dense_sparse_inputs_and_retains_lazy_predictions(
    session, use_setters
):
    GaussianMixture = _estimator()
    training = _training(session, name="input")
    held_out = session.createDataFrame(
        [
            (0, Vectors.dense([-5.0])),
            (1, Vectors.sparse(1, {0: -4.5})),
            (2, Vectors.dense([4.5])),
            (3, Vectors.sparse(1, {0: 5.0})),
        ],
        ["id", "input"],
    )
    values = dict(
        featuresCol="input",
        predictionCol="cluster",
        k=2,
        probabilityCol="probs",
        tol=1e-5,
        maxIter=100,
        seed=42,
        aggregationDepth=3,
        weightCol="weight",
    )
    if use_setters:
        estimator = GaussianMixture()
        for name, value in values.items():
            assert getattr(estimator, "set" + name[0].upper() + name[1:])(value) is estimator
    else:
        estimator = GaussianMixture(**values)
    model = estimator.fit(training)
    lazy = (
        model.transform(held_out)
        .orderBy("id")
        .select("cluster", vector_to_array("probs").alias("p"))
    )
    estimator.setPredictionCol("changed")
    del model, estimator
    gc.collect()
    rows = lazy.collect()
    _assert_partition([r[0] for r in rows], [0, 0, 1, 1])
    for row in rows:
        assert len(row.p) == 2 and sum(row.p) == pytest.approx(1.0)
        assert all(math.isfinite(x) and 0 <= x <= 1 for x in row.p)
        assert max(range(2), key=row.p.__getitem__) == row[0]


def test_gaussian_mixture_multivariate_probabilities(session):
    training = _vectors(
        session,
        "SELECT array(CASE WHEN id<40 THEN -5.0D ELSE 5.0D END + sin(id)*0.5D, "
        "CASE WHEN id<40 THEN -3.0D ELSE 3.0D END + cos(id)*0.7D) xs FROM range(80)",
    )
    points = [[-5.0, -3.0], [-4.5, -3.5], [5.0, 3.0], [4.5, 3.5]]
    held = session.createDataFrame(
        [
            (i, Vectors.sparse(2, {j: v for j, v in enumerate(x)}))
            if i % 2
            else (i, Vectors.dense(x))
            for i, x in enumerate(points)
        ],
        ["id", "features"],
    )
    rows = _outputs(_estimator()(seed=42, tol=1e-5).fit(training), held)
    _assert_partition([r[0] for r in rows], [0, 0, 1, 1])
    for row in rows:
        assert len(row.p) == 2 and sum(row.p) == pytest.approx(1.0)
        assert all(math.isfinite(p) and 0 <= p <= 1 for p in row.p)
        assert max(range(2), key=row.p.__getitem__) == row[0]


def test_gaussian_mixture_repeated_fits_retain_independent_models(session):
    estimator = _estimator()(k=2, seed=42, tol=1e-5)
    first = estimator.fit(_training(session))
    second = estimator.fit(_training(session, shift=10))
    held = _vectors(
        session,
        "SELECT id,array(x) xs FROM VALUES (0,-5.0D),(1,4.5D),(2,5.5D),(3,15.0D) AS t(id,x)",
    )
    _assert_partition([r[0] for r in _outputs(first, held)], [0, 1, 1, 1])
    _assert_partition([r[0] for r in _outputs(second, held)], [0, 0, 0, 1])


def test_gaussian_mixture_three_components_and_seed_reproducibility(session):
    training = _vectors(
        session,
        "SELECT array(CASE WHEN id<40 THEN -8.0D WHEN id<80 THEN 0.0D ELSE 8.0D END + sin(id)*0.5D) xs FROM range(120)",
    )
    held = _vectors(
        session,
        "SELECT id,array(x) xs FROM VALUES (0,-8.0D),(1,-7.5D),(2,-0.2D),(3,0.2D),(4,7.5D),(5,8.0D) AS t(id,x)",
    )
    estimator = _estimator()(k=3, seed=2**33, tol=1e-6, maxIter=100)
    first = _outputs(estimator.fit(training), held)
    second = _outputs(estimator.fit(training), held)
    _assert_partition([r[0] for r in first], [0, 0, 1, 1, 2, 2])
    assert [r[0] for r in first] == [r[0] for r in second]
    for a, b in zip(first, second):
        assert a.p == pytest.approx(b.p)
        assert len(a.p) == 3 and sum(a.p) == pytest.approx(1.0)


def test_gaussian_mixture_instance_weights_change_component_probabilities(session):
    training = _vectors(
        session,
        "SELECT array(x) xs,CASE WHEN x<0 THEN 9.0D ELSE 1.0D END weight "
        "FROM VALUES (-5.0D),(-4.5D),(-4.0D),(-3.5D),(-3.0D),(3.0D),(3.5D),(4.0D),(4.5D),(5.0D) AS t(x)",
    )
    held = _vectors(
        session, "SELECT id,array(x) xs FROM VALUES (0,-4.0D),(1,0.0D),(2,4.0D) AS t(id,x)"
    )
    plain = _outputs(_estimator()(k=2, seed=42, tol=1e-8).fit(training), held)
    weighted = _outputs(
        _estimator()(k=2, seed=42, tol=1e-8, weightCol="weight").fit(training), held
    )
    _assert_partition([plain[0][0], plain[2][0]], [0, 1])
    _assert_partition([weighted[0][0], weighted[2][0]], [0, 1])
    assert plain[1].p[plain[0][0]] == pytest.approx(0.5, abs=0.002)
    assert weighted[1].p[weighted[0][0]] == pytest.approx(0.9, abs=0.002)


def test_gaussian_mixture_can_disable_probability_output(session):
    model = _estimator()(k=2, seed=42, probabilityCol="", predictionCol="cluster").fit(
        _training(session)
    )
    transformed = model.transform(_training(session))
    assert "probability" not in transformed.columns
    assert len(transformed.select("cluster").collect()) == 80


@pytest.mark.parametrize(
    "case", ["k", "iterations", "tol", "aggregation", "weight", "nonvector", "empty"]
)
def test_gaussian_mixture_rejects_invalid_parameters_and_data(session, case):
    estimator = _estimator()(k=2, seed=42)
    frame = _training(session)
    if case == "k":
        estimator.setK(1)
    elif case == "iterations":
        estimator.setMaxIter(-1)
    elif case == "tol":
        estimator.setTol(-1.0)
    elif case == "aggregation":
        estimator.setAggregationDepth(1)
    elif case == "weight":
        estimator.setWeightCol("weight")
        frame = frame.withColumn("weight", F.lit(-1.0))
    elif case == "nonvector":
        frame = frame.withColumn("features", F.lit(1.0))
    else:
        frame = frame.filter("false")
    with pytest.raises(Exception):
        estimator.fit(frame).transform(frame).collect()
