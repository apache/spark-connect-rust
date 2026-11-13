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
"""Native GBTClassifier behaviors; these same cases run with official PySpark."""

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


def _classifier():
    return native_class("pyspark.ml.connect.classification", "GBTClassifier", "pyspark.ml.classification")


def _vectors(session, sql, name="features"):
    return session.sql(sql).withColumn(name, array_to_vector("xs"))


def _predictions(model, frame, name="prediction"):
    return [r[0] for r in model.transform(frame).orderBy("id").select(name).collect()]


@pytest.mark.parametrize("use_setters", [False, True], ids=["constructor", "setters"])
def test_gbt_classifier_learns_nonlinear_labels_and_retains_lazy_predictions(session, use_setters):
    GBTClassifier = _classifier()
    training = _vectors(session,
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS target, 1.0D AS weight "
        "FROM VALUES (-3,0),(-2,0),(-1,1),(0,1),(1,1),(2,0),(3,0) AS t(x,y)", "input")
    held_out = _vectors(session,
        "SELECT id, array(CAST(x AS DOUBLE)) AS xs "
        "FROM VALUES (0,-2.5),(1,-0.5),(2,0.5),(3,2.5) AS t(id,x)", "input")
    options = dict(featuresCol="input", labelCol="target", predictionCol="class", maxDepth=3,
                   maxBins=16, minInstancesPerNode=1, minInfoGain=0.0, maxMemoryInMB=64,
                   cacheNodeIds=True, checkpointInterval=-1, lossType="logistic", maxIter=3,
                   stepSize=0.2, seed=42, subsamplingRate=1.0, impurity="variance",
                   featureSubsetStrategy="all", leafCol="leaves", minWeightFractionPerNode=0.0,
                   weightCol="weight")
    if use_setters:
        estimator = GBTClassifier(validationTol=0.02)
        for name, value in options.items():
            assert getattr(estimator, "set" + name[0].upper() + name[1:])(value) is estimator
    else:
        estimator = GBTClassifier(**options, validationTol=0.02)
    assert estimator.setProbabilityCol("probs") is estimator
    assert estimator.setRawPredictionCol("margin") is estimator
    first = estimator.fit(training)
    second = estimator.fit(training.withColumn("target", 1.0 - F.col("target")))
    expected = [0.0,1.0,1.0,0.0]
    assert _predictions(first, held_out, "class") == expected
    assert _predictions(second, held_out, "class") == [1.0,0.0,0.0,1.0]
    assert _predictions(estimator.fit(training), held_out, "class") == expected
    rows = first.transform(held_out).orderBy("id").select(
        vector_to_array("probs").alias("p"), vector_to_array("margin").alias("raw"),
        vector_to_array("leaves").alias("leaf")).collect()
    for cls,row in zip(expected,rows):
        assert len(row.p) == len(row.raw) == 2
        assert sum(row.p) == pytest.approx(1.0)
        assert row.raw[0] == pytest.approx(-row.raw[1])
        assert row.p[1] == pytest.approx(1.0 / (1.0 + math.exp(2.0 * row.raw[0])))
        assert row.p[int(cls)] > 0.5
        assert len(row.leaf) == 3 and all(x >= 0 for x in row.leaf)
    delayed = first.transform(held_out).select("id", "class")
    del first
    gc.collect()
    assert [r["class"] for r in delayed.orderBy("id").collect()] == expected


def test_gbt_classifier_weights_and_thresholds_change_the_predicted_class(session):
    GBTClassifier = _classifier()
    training = _vectors(session,
        "SELECT array(0.0D) AS xs, CAST(y AS DOUBLE) AS label, CAST(w AS DOUBLE) AS weight "
        "FROM VALUES (0,1),(0,1),(0,1),(1,9) AS t(y,w)")
    test = _vectors(session, "SELECT 0 AS id, array(0.0D) AS xs")
    options = dict(maxDepth=0, maxIter=1, seed=42)
    plain = GBTClassifier(**options).fit(training)
    weighted = GBTClassifier(**options, weightCol="weight").fit(training)
    thresholded = GBTClassifier(**options).setThresholds([1.0,0.1]).fit(training)
    assert _predictions(plain, test) == [0.0]
    assert _predictions(weighted, test) == [1.0]
    assert _predictions(thresholded, test) == [1.0]
    for model, raw, p1 in [(plain,[0.5,-0.5],1.0/(1.0+math.e)),
                           (weighted,[-0.5,0.5],1.0/(1.0+math.exp(-1.0))),
                           (thresholded,[0.5,-0.5],1.0/(1.0+math.e))]:
        row = model.transform(test).select(vector_to_array("rawPrediction").alias("raw"),
                                          vector_to_array("probability").alias("p")).first()
        assert row.raw == pytest.approx(raw)
        assert row.p == pytest.approx([1.0-p1,p1])


def test_gbt_classifier_iterations_and_learning_rate_change_probabilities(session):
    GBTClassifier = _classifier()
    training = _vectors(session,
        "SELECT id, array(CAST(id AS DOUBLE)) AS xs, CAST(id >= 4 AS DOUBLE) AS label FROM range(8)")
    def confidence(iterations, step):
        model = GBTClassifier(seed=42, maxDepth=1, maxIter=iterations, stepSize=step).fit(training)
        return model.transform(training).orderBy("id").select(vector_to_array("probability").alias("p")).first().p[0]
    initial = confidence(1,0.1)
    slow = confidence(3,0.1)
    fast = confidence(3,0.5)
    assert 0.5 < initial < slow < fast < 1.0


def test_gbt_classifier_validation_rows_are_held_out_and_stop_boosting_early(session):
    GBTClassifier = _classifier()
    training = _vectors(session,
        "SELECT id, array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS label, validation "
        "FROM VALUES (0,-3,0,false),(1,-2,0,false),(2,2,1,false),(3,3,1,false),"
        "(4,-3,1,true),(5,-2,1,true),(6,2,0,true),(7,3,0,true) AS t(id,x,y,validation)")
    options = dict(seed=42, maxDepth=1, maxIter=6, stepSize=0.2, leafCol="leaves")
    model = GBTClassifier(**options, validationIndicatorCol="validation", validationTol=0.01).fit(training)
    reference = GBTClassifier(seed=42, maxDepth=1, maxIter=1).fit(training.filter(~F.col("validation")))
    assert _predictions(model, training) == _predictions(reference, training)
    rows = model.transform(training).select(vector_to_array("leaves").alias("leaf")).collect()
    assert all(len(r.leaf) == 1 for r in rows)
    unvalidated = GBTClassifier(**options).fit(training.filter(~F.col("validation")))
    assert all(len(r.leaf) == 6 for r in unvalidated.transform(training).select(
        vector_to_array("leaves").alias("leaf")).collect())


def test_gbt_classifier_seeded_subsampling_and_feature_sampling_are_reproducible(session):
    GBTClassifier = _classifier()
    training = _vectors(session,
        "SELECT id, array(CAST(id AS DOUBLE),CAST(id % 3 AS DOUBLE)) AS xs, "
        "CAST(id >= 20 AS DOUBLE) AS label FROM range(40)")
    estimator = GBTClassifier(seed=2**33, maxDepth=2, maxIter=5, stepSize=0.2,
                              subsamplingRate=0.8, featureSubsetStrategy="sqrt")
    first = _predictions(estimator.fit(training), training)
    assert first == _predictions(estimator.fit(training), training)
    assert first[:10] == [0.0]*10 and first[-10:] == [1.0]*10


def test_gbt_classifier_rejects_invalid_parameters_and_training_data(session):
    GBTClassifier = _classifier()
    training = _vectors(session,
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS label, 1.0D AS weight "
        "FROM VALUES (-1,0),(1,1) AS t(x,y)")
    for options in [dict(maxIter=0),dict(stepSize=0.0),dict(stepSize=1.1),dict(validationTol=-0.1),
                    dict(lossType="squared"),dict(maxDepth=31),dict(maxBins=1),
                    dict(minInstancesPerNode=0),dict(minInfoGain=-1.0),dict(maxMemoryInMB=-1),
                    dict(checkpointInterval=0),dict(featureSubsetStrategy="invalid"),
                    dict(subsamplingRate=0.0),dict(minWeightFractionPerNode=0.5)]:
        with pytest.raises(Exception):
            GBTClassifier(**options).fit(training)
    with pytest.raises(Exception):
        GBTClassifier().setThresholds([1.0,1.0,1.0]).fit(training)
    for frame in [training.drop("label"),training.drop("features"),
                  training.withColumn("label",F.lit(2.0)),training.withColumn("label",F.lit(-1.0)),
                  training.withColumn("label",F.lit(0.5)),training.withColumn("features",F.lit(1.0))]:
        with pytest.raises(Exception):
            GBTClassifier(maxIter=1).fit(frame)
    with pytest.raises(Exception):
        GBTClassifier(weightCol="weight").fit(training.withColumn("weight",F.lit(-1.0)))
    with pytest.raises(Exception):
        GBTClassifier(validationIndicatorCol="missing").fit(training)
