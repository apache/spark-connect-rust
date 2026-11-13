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
"""Native BisectingKMeans behaviors, also run with official PySpark."""

import gc
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
    return native_class("pyspark.ml.connect.clustering", "BisectingKMeans", "pyspark.ml.clustering")


def _vectors(session, sql, name="features"):
    return session.sql(sql).withColumn(name, array_to_vector("xs"))


def _predictions(model, frame, name="prediction"):
    return [r[0] for r in model.transform(frame).orderBy("id").select(name).collect()]


def _assert_partition(actual, expected):
    assert len(actual) == len(expected)
    for i in range(len(actual)):
        for j in range(len(actual)):
            assert (actual[i] == actual[j]) == (expected[i] == expected[j]), actual


def _quadrants(session):
    return _vectors(
        session,
        "SELECT id, array(x,y) xs FROM VALUES "
        "(0,9.0D,20.0D),(1,11.0D,20.0D),(2,9.0D,40.0D),(3,11.0D,40.0D),"
        "(4,29.0D,20.0D),(5,31.0D,20.0D),(6,29.0D,40.0D),(7,31.0D,40.0D) AS t(id,x,y)",
    )


@pytest.mark.parametrize("use_setters", [False, True], ids=["constructor", "setters"])
def test_bisecting_kmeans_learns_independent_models_and_retains_lazy_predictions(
    session, use_setters
):
    BisectingKMeans = _estimator()
    training = _vectors(
        session,
        "SELECT array(CAST(x AS DOUBLE)) xs, 1.0D weight "
        "FROM VALUES (10),(11),(12),(28),(29),(30) AS t(x)",
        "input",
    )
    shifted = _vectors(
        session,
        "SELECT array(CAST(x AS DOUBLE)) xs, 1.0D weight "
        "FROM VALUES (30),(31),(32),(48),(49),(50) AS t(x)",
        "input",
    )
    points = [11.0, 19.0, 29.0, 41.0, 49.0]
    held_out = session.createDataFrame(
        [
            (i, Vectors.sparse(1, {0: x})) if i % 2 else (i, Vectors.dense([x]))
            for i, x in enumerate(points)
        ],
        ["id", "input"],
    )
    values = dict(
        featuresCol="input",
        predictionCol="cluster",
        maxIter=30,
        seed=42,
        k=2,
        minDivisibleClusterSize=1.0,
        distanceMeasure="euclidean",
        weightCol="weight",
    )
    if use_setters:
        estimator = BisectingKMeans()
        for name, value in values.items():
            assert getattr(estimator, "set" + name[0].upper() + name[1:])(value) is estimator
    else:
        estimator = BisectingKMeans(**values)
    first = estimator.fit(training)
    second = estimator.fit(shifted)
    lazy = first.transform(held_out).orderBy("id").select("cluster")
    estimator.setPredictionCol("changed")
    del first, estimator
    gc.collect()
    _assert_partition([r[0] for r in lazy.collect()], [0, 0, 1, 1, 1])
    _assert_partition(_predictions(second, held_out, "cluster"), [0, 0, 0, 1, 1])


@pytest.mark.parametrize("size,clusters", [(1.0, 4), (0.75, 2), (9.0, 1)])
def test_bisecting_kmeans_divisibility_and_default_k_control_leaf_count(session, size, clusters):
    frame = _quadrants(session)
    options = dict(seed=42)
    if size != 1.0:
        options.update(k=4, minDivisibleClusterSize=size)
    values = _predictions(_estimator()(**options).fit(frame), frame)
    assert len(set(values)) == clusters


def test_bisecting_kmeans_cosine_distance_ignores_vector_magnitude(session):
    training = _vectors(
        session,
        "SELECT array(x,y) xs FROM VALUES "
        "(1.0D,0.0D),(5.0D,0.1D),(0.0D,1.0D),(0.1D,5.0D),"
        "(-1.0D,-1.0D),(-5.0D,-4.0D) AS t(x,y)",
    )
    held_out = session.createDataFrame(
        [
            (i, Vectors.sparse(2, {j: v for j, v in enumerate(x) if v}))
            for i, x in enumerate(
                [
                    [100.0, 0.0],
                    [0.01, 0.0],
                    [0.0, 100.0],
                    [0.0, 0.01],
                    [-100.0, -100.0],
                    [-0.01, -0.01],
                ]
            )
        ],
        ["id", "features"],
    )
    model = _estimator()(k=3, seed=42, distanceMeasure="cosine").fit(training)
    _assert_partition(_predictions(model, held_out), [0, 0, 1, 1, 2, 2])


def test_bisecting_kmeans_instance_weights_move_a_cluster_boundary(session):
    frame = _vectors(
        session,
        "SELECT array(x) xs, weight FROM VALUES "
        "(-10.0D,1.0D),(-9.0D,1.0D),(-8.0D,1.0D),(2.0D,100.0D),"
        "(3.0D,1.0D),(4.0D,1.0D),(10.0D,1.0D),(11.0D,1.0D) AS t(x,weight)",
    )
    held_out = _vectors(
        session,
        "SELECT id, array(x) xs FROM VALUES (0,-9.0D),(1,-2.0D),(2,2.0D),(3,11.0D) AS t(id,x)",
    )
    plain = _estimator()(k=2, seed=42).fit(frame)
    weighted = _estimator()(k=2, seed=42, weightCol="weight").fit(frame)
    _assert_partition(_predictions(plain, held_out), [0, 0, 1, 1])
    _assert_partition(_predictions(weighted, held_out), [0, 1, 1, 1])


def test_bisecting_kmeans_seeded_fits_are_reproducible(session):
    frame = _quadrants(session)
    estimator = _estimator()(k=3, seed=2**33)
    assert _predictions(estimator.fit(frame), frame) == _predictions(estimator.fit(frame), frame)


@pytest.mark.parametrize(
    "case", ["k", "iterations", "size", "distance", "weight", "nonvector", "zero", "empty"]
)
def test_bisecting_kmeans_rejects_invalid_parameters_and_data(session, case):
    estimator = _estimator()(k=2, seed=42)
    frame = _quadrants(session)
    if case == "k":
        estimator.setK(1)
    elif case == "iterations":
        estimator.setMaxIter(-1)
    elif case == "size":
        estimator.setMinDivisibleClusterSize(-1.0)
    elif case == "distance":
        estimator.setDistanceMeasure("unknown")
    elif case == "weight":
        estimator.setWeightCol("weight")
        frame = frame.withColumn("weight", F.lit(-1.0))
    elif case == "nonvector":
        frame = frame.withColumn("features", F.lit(1.0))
    elif case == "zero":
        estimator.setDistanceMeasure("cosine")
        frame = frame.withColumn("features", array_to_vector(F.array(F.lit(0.0), F.lit(0.0))))
    else:
        frame = frame.filter("false")
    with pytest.raises(Exception):
        estimator.fit(frame).transform(frame).collect()
