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
"""Native GBTRegressor behaviors; these same cases run with official PySpark."""

import gc
import math
import os

import pytest
from _native_classes import native_class
from pyspark.ml.connect.functions import array_to_vector, vector_to_array
from pyspark.sql import SparkSession
from pyspark.sql import functions as F


@pytest.fixture
def session():
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live-server tests")
    spark = SparkSession.builder.remote(remote).getOrCreate()
    try:
        yield spark
    finally:
        spark.stop()


def _regressor():
    return native_class("pyspark.ml.connect.regression", "GBTRegressor", "pyspark.ml.regression")


def _vectors(session, sql, name="features"):
    return session.sql(sql).withColumn(name, array_to_vector("xs"))


def _predictions(model, frame, name="prediction"):
    return [r[0] for r in model.transform(frame).orderBy("id").select(name).collect()]


@pytest.mark.parametrize("use_setters", [False, True], ids=["constructor", "setters"])
def test_gbt_regressor_learns_nonlinear_values_and_retains_lazy_predictions(session, use_setters):
    GBTRegressor = _regressor()
    training = _vectors(
        session,
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS target, 1.0D AS weight "
        "FROM VALUES (-3,-4.0),(-2,-4.0),(-1,5.5),(0,5.5),(1,5.5),(2,-4.0),(3,-4.0) AS t(x,y)",
        "input",
    )
    held_out = _vectors(
        session,
        "SELECT id, array(CAST(x AS DOUBLE)) AS xs "
        "FROM VALUES (0,-2.5),(1,-0.5),(2,0.5),(3,2.5) AS t(id,x)",
        "input",
    )
    options = dict(
        featuresCol="input",
        labelCol="target",
        predictionCol="value",
        maxDepth=3,
        maxBins=16,
        minInstancesPerNode=1,
        minInfoGain=0.0,
        maxMemoryInMB=64,
        cacheNodeIds=True,
        subsamplingRate=1.0,
        checkpointInterval=-1,
        lossType="squared",
        maxIter=3,
        stepSize=0.2,
        seed=42,
        impurity="variance",
        featureSubsetStrategy="all",
        leafCol="leaves",
        minWeightFractionPerNode=0.0,
        weightCol="weight",
    )
    if use_setters:
        estimator = GBTRegressor(validationTol=0.05)
        for name, value in options.items():
            assert getattr(estimator, "set" + name[0].upper() + name[1:])(value) is estimator
    else:
        estimator = GBTRegressor(**options, validationTol=0.05)
    first = estimator.fit(training)
    second = estimator.fit(training.withColumn("target", F.col("target") + 10.0))
    expected = [-4.0, 5.5, 5.5, -4.0]
    assert _predictions(first, held_out, "value") == expected
    assert _predictions(second, held_out, "value") == [6.0, 15.5, 15.5, 6.0]
    assert _predictions(estimator.fit(training), held_out, "value") == expected
    leaves = first.transform(held_out).select(vector_to_array("leaves").alias("leaf")).collect()
    assert all(len(r.leaf) == 3 and all(x >= 0 for x in r.leaf) for r in leaves)
    delayed = first.transform(held_out).select("id", "value")
    del first
    gc.collect()
    assert [r.value for r in delayed.orderBy("id").collect()] == expected


def test_gbt_regressor_instance_weights_change_the_predicted_mean(session):
    GBTRegressor = _regressor()
    training = _vectors(
        session,
        "SELECT 0 AS id, array(0.0D) AS xs, CAST(y AS DOUBLE) AS label, CAST(w AS DOUBLE) AS weight "
        "FROM VALUES (-2,1),(-2,1),(-2,1),(10,9) AS t(y,w)",
    )
    plain = GBTRegressor(seed=42, maxDepth=0, maxIter=1).fit(training)
    weighted = GBTRegressor(seed=42, maxDepth=0, maxIter=1, weightCol="weight").fit(training)
    assert _predictions(plain, training) == [1.0] * 4
    assert _predictions(weighted, training) == [7.0] * 4


def test_gbt_regressor_supports_squared_and_absolute_loss(session):
    GBTRegressor = _regressor()
    training = _vectors(
        session,
        "SELECT id, array(0.0D) AS xs, CAST(y AS DOUBLE) AS label "
        "FROM VALUES (0,0),(1,0),(2,0),(3,10) AS t(id,y)",
    )
    options = dict(seed=42, maxDepth=0, maxIter=3, stepSize=0.5)
    squared = GBTRegressor(**options, lossType="squared").fit(training)
    absolute = GBTRegressor(**options, lossType="absolute").fit(training)
    assert _predictions(squared, training) == pytest.approx([2.5] * 4)
    assert _predictions(absolute, training) == pytest.approx([2.0] * 4)


def test_gbt_regressor_iterations_and_learning_rate_improve_squared_loss(session):
    GBTRegressor = _regressor()
    training = _vectors(
        session,
        "SELECT id, array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS label "
        "FROM VALUES (0,-2,0),(1,-1,0),(2,1,2),(3,2,4) AS t(id,x,y)",
    )

    def error(iterations, step):
        model = GBTRegressor(seed=42, maxDepth=1, maxIter=iterations, stepSize=step).fit(training)
        return sum(
            (prediction - label) ** 2
            for prediction, label in zip(_predictions(model, training), [0.0, 0.0, 2.0, 4.0])
        )

    assert error(1, 0.1) > error(3, 0.1) > error(3, 0.5)


def test_gbt_regressor_validation_rows_are_held_out_and_stop_boosting_early(session):
    GBTRegressor = _regressor()
    training = _vectors(
        session,
        "SELECT id, array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS label, validation "
        "FROM VALUES (0,-3,-2,false),(1,-2,-2,false),(2,2,3,false),(3,3,3,false),"
        "(4,-3,3,true),(5,-2,3,true),(6,2,-2,true),(7,3,-2,true) AS t(id,x,y,validation)",
    )
    options = dict(seed=42, maxDepth=1, maxIter=6, stepSize=0.2, leafCol="leaves")
    model = GBTRegressor(**options, validationIndicatorCol="validation", validationTol=0.01).fit(
        training
    )
    reference = GBTRegressor(seed=42, maxDepth=1, maxIter=1).fit(
        training.filter(~F.col("validation"))
    )
    assert _predictions(model, training) == _predictions(reference, training)
    leaves = model.transform(training).select(vector_to_array("leaves").alias("leaf")).collect()
    assert all(len(r.leaf) == 1 for r in leaves)
    unvalidated = GBTRegressor(**options).fit(training.filter(~F.col("validation")))
    assert all(
        len(r.leaf) == 6
        for r in unvalidated.transform(training)
        .select(vector_to_array("leaves").alias("leaf"))
        .collect()
    )


def test_gbt_regressor_seeded_subsampling_and_feature_sampling_are_reproducible(session):
    GBTRegressor = _regressor()
    training = _vectors(
        session,
        "SELECT id, array(CAST(id AS DOUBLE),CAST(id % 3 AS DOUBLE)) AS xs, "
        "CAST(CASE WHEN id < 20 THEN -3 ELSE 8 END AS DOUBLE) AS label FROM range(40)",
    )
    estimator = GBTRegressor(
        seed=2**33,
        maxDepth=2,
        maxIter=5,
        stepSize=0.2,
        subsamplingRate=0.8,
        featureSubsetStrategy="sqrt",
    )
    first = _predictions(estimator.fit(training), training)
    assert first == pytest.approx(_predictions(estimator.fit(training), training))
    assert all(math.isfinite(x) for x in first)
    assert first[0] < first[-1]


def test_gbt_regressor_rejects_invalid_parameters_and_training_data(session):
    GBTRegressor = _regressor()
    training = _vectors(
        session,
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS label, 1.0D AS weight "
        "FROM VALUES (-1,-2.5),(1,3.5) AS t(x,y)",
    )
    for options in [
        dict(maxIter=0),
        dict(stepSize=0.0),
        dict(stepSize=1.1),
        dict(validationTol=-0.1),
        dict(lossType="logistic"),
        dict(maxDepth=31),
        dict(maxBins=1),
        dict(minInstancesPerNode=0),
        dict(minInfoGain=-1.0),
        dict(maxMemoryInMB=-1),
        dict(checkpointInterval=0),
        dict(featureSubsetStrategy="invalid"),
        dict(subsamplingRate=0.0),
        dict(minWeightFractionPerNode=0.5),
    ]:
        with pytest.raises(Exception):
            GBTRegressor(**options).fit(training)
    for frame in [
        training.drop("label"),
        training.drop("features"),
        training.withColumn("label", F.lit(float("nan"))),
        training.withColumn("features", F.lit(1.0)),
    ]:
        with pytest.raises(Exception):
            GBTRegressor(maxIter=1).fit(frame)
    with pytest.raises(Exception):
        GBTRegressor(weightCol="weight").fit(training.withColumn("weight", F.lit(-1.0)))
    with pytest.raises(Exception):
        GBTRegressor(validationIndicatorCol="missing").fit(training)
