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
"""Native NaiveBayes behaviors; the same cases run with official PySpark."""

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
        pytest.skip("SPARK_REMOTE is required for live-server tests")
    spark = SparkSession.builder.remote(remote).getOrCreate()
    try:
        yield spark
    finally:
        spark.stop()


def _classifier():
    return native_class("pyspark.ml.connect.classification", "NaiveBayes", "pyspark.ml.classification")


def _vectors(session, sql, name="features"):
    return session.sql(sql).withColumn(name, array_to_vector("xs"))


def _weighted_training(session):
    return _vectors(
        session,
        "SELECT array(1.0D,1.0D) xs, label, weight FROM VALUES "
        "(0.0D,1.0D),(1.0D,9.0D) AS t(label,weight)",
    )


def _outputs(model, data, prediction="prediction", probability="probability", raw="rawPrediction"):
    return (
        model.transform(data)
        .select(
            prediction, vector_to_array(probability).alias("p"), vector_to_array(raw).alias("r")
        )
        .collect()
    )


@pytest.mark.parametrize("model_type", ["multinomial", "bernoulli", "complement", "gaussian"])
@pytest.mark.parametrize("use_setters", [False, True], ids=["constructor", "setters"])
def test_naive_bayes_learns_multiclass_dense_and_sparse_vectors(session, model_type, use_setters):
    NaiveBayes = _classifier()
    if model_type == "gaussian":
        sql = (
            "SELECT label, array(x,y) xs, 1.0D weight FROM VALUES "
            + "(0.0D,-3.0D,-2.0D),(0.0D,-2.0D,-3.0D),(1.0D,2.0D,3.0D),"
            + "(1.0D,3.0D,2.0D),(2.0D,7.0D,8.0D),(2.0D,8.0D,7.0D) AS t(label,x,y)"
        )
        points = [[-2.5, -2.5], [2.5, 2.5], [7.5, 7.5]]
    else:
        values = (
            "(0.0D,1.0D,0.0D,0.0D),(0.0D,1.0D,1.0D,0.0D),(1.0D,0.0D,1.0D,0.0D),"
            + "(1.0D,0.0D,1.0D,1.0D),(2.0D,0.0D,0.0D,1.0D),(2.0D,1.0D,0.0D,1.0D)"
            if model_type == "bernoulli"
            else "(0.0D,4.0D,1.0D,0.0D),(0.0D,5.0D,0.0D,1.0D),(1.0D,0.0D,4.0D,1.0D),"
            + "(1.0D,1.0D,5.0D,0.0D),(2.0D,1.0D,0.0D,4.0D),(2.0D,0.0D,1.0D,5.0D)"
        )
        sql = (
            "SELECT label, array(x,y,z) xs, 1.0D weight FROM VALUES "
            + values
            + " AS t(label,x,y,z)"
        )
        v = 1.0 if model_type == "bernoulli" else 4.0
        points = [[v, 0.0, 0.0], [0.0, v, 0.0], [0.0, 0.0, v]]
    training = _vectors(session, sql, "input").withColumnRenamed("label", "target")
    held_out = session.createDataFrame(
        [
            (i, Vectors.sparse(len(x), {j: v for j, v in enumerate(x) if v}))
            if i % 2
            else (i, Vectors.dense(x))
            for i, x in enumerate(points)
        ],
        ["id", "input"],
    )
    values = dict(
        featuresCol="input",
        labelCol="target",
        predictionCol="class",
        probabilityCol="probs",
        rawPredictionCol="scores",
        smoothing=0.5,
        modelType=model_type,
        thresholds=[1.0, 1.0, 1.0],
        weightCol="weight",
    )
    if use_setters:
        estimator = NaiveBayes()
        for name, value in values.items():
            assert getattr(estimator, "set" + name[0].upper() + name[1:])(value) is estimator
    else:
        estimator = NaiveBayes(**values)
    model = estimator.fit(training)
    lazy = (
        model.transform(held_out)
        .orderBy("id")
        .select("class", vector_to_array("probs").alias("p"), vector_to_array("scores").alias("r"))
    )
    del model, estimator
    gc.collect()
    rows = lazy.collect()
    assert [r[0] for r in rows] == [0.0, 1.0, 2.0]
    for row in rows:
        assert len(row.p) == len(row.r) == 3
        assert all(math.isfinite(x) and 0.0 <= x <= 1.0 for x in row.p)
        assert all(math.isfinite(x) for x in row.r)
        assert sum(row.p) == pytest.approx(1.0)
        assert max(range(3), key=row.p.__getitem__) == row[0]


@pytest.mark.parametrize("smoothing", [0.0, 1.0, 10.0])
def test_naive_bayes_weights_and_smoothing_change_class_priors(session, smoothing):
    NaiveBayes = _classifier()
    training = _weighted_training(session)
    plain = NaiveBayes(smoothing=smoothing).fit(training)
    weighted = NaiveBayes(smoothing=smoothing, weightCol="weight").fit(training)
    assert _outputs(plain, training)[0].p == pytest.approx([0.5, 0.5])
    row = _outputs(weighted, training)[0]
    prior = (1.0 + smoothing) / (10.0 + 2 * smoothing)
    assert row.p == pytest.approx([prior, 1.0 - prior])
    assert row[0] == 1.0
    assert row.r == pytest.approx(
        [math.log(prior) - 2 * math.log(2), math.log(1 - prior) - 2 * math.log(2)]
    )


def test_naive_bayes_thresholds_change_prediction_without_changing_probabilities(session):
    NaiveBayes = _classifier()
    training = _weighted_training(session)
    normal = _outputs(NaiveBayes(weightCol="weight").fit(training), training)[0]
    adjusted = _outputs(
        NaiveBayes(weightCol="weight", thresholds=[0.1, 1.0]).fit(training), training
    )[0]
    assert normal[0] == 1.0 and adjusted[0] == 0.0
    assert adjusted.p == pytest.approx(normal.p)
    assert adjusted.r == pytest.approx(normal.r)


def test_naive_bayes_can_disable_vector_outputs(session):
    training = _weighted_training(session)
    transformed = (
        _classifier()(
            weightCol="weight", predictionCol="class", probabilityCol="", rawPredictionCol=""
        )
        .fit(training)
        .transform(training)
    )
    assert "probability" not in transformed.columns and "rawPrediction" not in transformed.columns
    assert [r[0] for r in transformed.select("class").collect()] == [1.0, 1.0]


def test_naive_bayes_repeated_fits_keep_independent_models(session):
    training = _weighted_training(session)
    estimator = _classifier()(weightCol="weight")
    first = estimator.fit(training)
    second = estimator.fit(training.withColumn("label", F.lit(1.0) - F.col("label")))
    estimator.setModelType("bernoulli")
    assert _outputs(first, training)[0][0] == 1.0
    assert _outputs(second, training)[0][0] == 0.0


@pytest.mark.parametrize(
    "case", ["smoothing", "model_type", "thresholds", "negative", "binary", "weight"]
)
def test_naive_bayes_rejects_invalid_parameters_and_training_data(session, case):
    NaiveBayes = _classifier()
    training = _weighted_training(session)
    estimator = NaiveBayes()
    if case == "smoothing":
        estimator.setSmoothing(-1.0)
    elif case == "model_type":
        estimator.setModelType("unknown")
    elif case == "thresholds":
        estimator.setThresholds([1.0])
    elif case == "negative":
        training = training.withColumn(
            "features", array_to_vector(F.array(F.lit(-1.0), F.lit(1.0)))
        )
    elif case == "binary":
        estimator.setModelType("bernoulli")
        training = training.withColumn("features", array_to_vector(F.array(F.lit(2.0), F.lit(1.0))))
    else:
        estimator.setWeightCol("weight")
        training = training.withColumn("weight", F.lit(-1.0))
    with pytest.raises(Exception):
        estimator.fit(training).transform(training).collect()
