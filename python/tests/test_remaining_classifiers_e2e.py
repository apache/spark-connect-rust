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
"""Native FMClassifier, MultilayerPerceptronClassifier and OneVsRest behaviors.

Run unchanged against official PySpark to check the numerical expectations.
"""

import gc
import math
import os

import pytest
from _native_classes import native_class
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


def _classifier(name):
    return native_class("pyspark.ml.connect.classification", name, "pyspark.ml.classification")


def _triangles(session, names=("features", "label")):
    rows = []
    for label, point in enumerate([[-2.0, -2.0], [2.0, -2.0], [0.0, 2.0]]):
        for delta in [-0.1, 0.0, 0.1]:
            rows.append((Vectors.dense([point[0] + delta, point[1]]), float(label)))
    return session.createDataFrame(rows, list(names))


def _classes(frame, column="prediction"):
    return [r[0] for r in frame.select(column).collect()]


@pytest.mark.parametrize("use_setters", [False, True], ids=["constructor", "setters"])
def test_fm_classifier_learns_interactions_and_keeps_lazy_models(session, use_setters):
    rows = [(Vectors.dense([x, y]), float(x * y > 0)) for x in [-1.0, 1.0] for y in [-1.0, 1.0]]
    train = session.createDataFrame(rows, ["input", "target"])
    options = dict(
        featuresCol="input",
        labelCol="target",
        predictionCol="pred",
        probabilityCol="prob",
        rawPredictionCol="scores",
        factorSize=2,
        fitIntercept=True,
        fitLinear=False,
        regParam=0.0,
        miniBatchFraction=1.0,
        initStd=0.1,
        maxIter=200,
        stepSize=0.1,
        tol=1e-8,
        solver="adamW",
        thresholds=[1.0, 1.0],
        seed=42,
    )
    estimator = (
        _classifier("FMClassifier")() if use_setters else _classifier("FMClassifier")(**options)
    )
    if use_setters:
        for name, value in options.items():
            assert getattr(estimator, "set" + name[0].upper() + name[1:])(value) is estimator
    first = estimator.fit(train)
    second = estimator.fit(train.withColumn("target", F.lit(1.0) - F.col("target")))
    held = session.createDataFrame(
        [(Vectors.sparse(2, {0: x, 1: y}),) for x in [-1.0, 1.0] for y in [-1.0, 1.0]], ["input"]
    )
    lazy = first.transform(held)
    del first, estimator
    gc.collect()
    result = lazy.select("pred", "prob", "scores").collect()
    assert [r[0] for r in result] == [1.0, 0.0, 0.0, 1.0]
    for row in result:
        assert len(row[1]) == len(row[2]) == 2
        assert sum(row[1]) == pytest.approx(1.0, abs=1e-8)
        assert all(math.isfinite(float(x)) for x in row[2])
    assert _classes(second.transform(held), "pred") == [0.0, 1.0, 1.0, 0.0]


def test_fm_classifier_gradient_descent_and_thresholds(session):
    train = session.createDataFrame(
        [(Vectors.dense([x]), float(x > 0)) for x in [-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]],
        ["features", "label"],
    )
    model = _classifier("FMClassifier")(
        solver="gd", stepSize=0.1, maxIter=200, seed=17, thresholds=[1.0, 4.0]
    ).fit(train)
    for row in model.transform(train).select("prediction", "probability").collect():
        expected = 0.0 if row[1][0] >= row[1][1] / 4.0 else 1.0
        assert row[0] == expected


@pytest.mark.parametrize("weights", ["dense", "sparse"])
def test_multilayer_perceptron_multiclass_and_initial_weights(session, weights):
    train = _triangles(session, ("input", "target"))
    initial = (
        Vectors.dense([0.01 * i for i in range(9)])
        if weights == "dense"
        else Vectors.sparse(9, {0: 0.1, 8: -0.1})
    )
    estimator = _classifier("MultilayerPerceptronClassifier")(
        featuresCol="input",
        labelCol="target",
        predictionCol="pred",
        probabilityCol="prob",
        rawPredictionCol="scores",
        layers=[2, 3],
        initialWeights=initial,
        seed=2**33,
        maxIter=100,
        tol=1e-8,
        blockSize=3,
        stepSize=0.1,
        solver="l-bfgs",
    )
    model = estimator.fit(train)
    lazy = model.transform(train)
    del model
    gc.collect()
    result = lazy.select("target", "pred", "prob", "scores").collect()
    assert [r[1] for r in result] == [r[0] for r in result]
    for row in result:
        assert len(row[2]) == len(row[3]) == 3
        assert sum(row[2]) == pytest.approx(1.0, abs=1e-8)


def test_multilayer_perceptron_hidden_layers_setters_and_gradient_descent(session):
    train = session.createDataFrame(
        [(Vectors.dense([x, y]), float(x != y)) for x in [0.0, 1.0] for y in [0.0, 1.0]],
        ["features", "label"],
    )
    estimator = (
        _classifier("MultilayerPerceptronClassifier")()
        .setLayers([2, 3, 2])
        .setMaxIter(300)
        .setSeed(42)
        .setBlockSize(4)
        .setTol(1e-9)
    )
    assert _classes(estimator.fit(train).transform(train)) == [0.0, 1.0, 1.0, 0.0]
    simple = _triangles(session)
    gd = (
        _classifier("MultilayerPerceptronClassifier")(
            layers=[2, 3], solver="gd", stepSize=0.1, maxIter=100, seed=42
        )
        .setThresholds([1.0, 1.0, 1.0])
        .fit(simple)
    )
    assert _classes(gd.transform(simple)) == [0.0] * 3 + [1.0] * 3 + [2.0] * 3


@pytest.mark.parametrize("parallelism", [1, 3])
def test_one_vs_rest_custom_columns_parallel_fit_and_lazy_independent_models(session, parallelism):
    base = _classifier("LinearSVC")(maxIter=100, tol=1e-8, rawPredictionCol="scores")
    estimator = _classifier("OneVsRest")(
        classifier=base,
        featuresCol="input",
        labelCol="target",
        predictionCol="pred",
        rawPredictionCol="scores",
        parallelism=parallelism,
    )
    train = _triangles(session, ("input", "target"))
    first = estimator.fit(train)
    second = estimator.fit(train.withColumn("target", (F.col("target") + 1.0) % 3.0))
    lazy = first.transform(train)
    del first, estimator, base
    gc.collect()
    result = lazy.select("target", "pred", "scores").collect()
    assert [r[1] for r in result] == [r[0] for r in result]
    for row in result:
        assert len(row[2]) == 3
        assert row[1] == float(max(range(3), key=lambda i: row[2][i]))
    assert _classes(second.transform(train), "pred") == [1.0] * 3 + [2.0] * 3 + [0.0] * 3


@pytest.mark.parametrize(
    "name",
    [
        "LogisticRegression",
        "LinearSVC",
        "DecisionTreeClassifier",
        "RandomForestClassifier",
        "GBTClassifier",
        "NaiveBayes",
        "FMClassifier",
        "MultilayerPerceptronClassifier",
    ],
)
def test_one_vs_rest_accepts_each_native_binary_classifier(session, name):
    options = {
        "LogisticRegression": dict(maxIter=30),
        "LinearSVC": dict(maxIter=50),
        "RandomForestClassifier": dict(numTrees=10, seed=42),
        "GBTClassifier": dict(maxIter=5, seed=42),
        "FMClassifier": dict(maxIter=150, stepSize=0.1, seed=42),
        "MultilayerPerceptronClassifier": dict(layers=[3, 2], maxIter=100, seed=42),
    }.get(name, {})
    rows = []
    for label in range(3):
        for scale in [1.0, 1.5, 2.0]:
            rows.append((Vectors.sparse(3, {label: scale}), float(label)))
    train = session.createDataFrame(rows, ["features", "label"])
    classifier = _classifier(name)(**options)
    estimator = _classifier("OneVsRest")().setClassifier(classifier).setParallelism(2)
    result = estimator.fit(train).transform(train)
    assert _classes(result) == [0.0] * 3 + [1.0] * 3 + [2.0] * 3
    assert "probability" not in result.columns


def test_one_vs_rest_weights_and_mutated_base_classifier(session):
    train = _triangles(session).withColumn("weight", F.lit(1.0))
    bad = train.withColumn("label", (F.col("label") + 1.0) % 3.0).withColumn("weight", F.lit(0.0))
    base = _classifier("LinearSVC")(maxIter=1)
    estimator = _classifier("OneVsRest")(classifier=base, weightCol="weight")
    base.setMaxIter(100)
    result = estimator.fit(train.union(bad)).transform(train)
    assert _classes(result) == [0.0] * 3 + [1.0] * 3 + [2.0] * 3
    base.setMaxIter(-1)
    with pytest.raises(Exception):
        estimator.fit(train)


def test_one_vs_rest_ignores_unsupported_weights_with_warning(session):
    train = _triangles(session).withColumn("weight", F.lit(-1.0))
    base = _classifier("MultilayerPerceptronClassifier")(layers=[2, 2], maxIter=50, seed=42)
    with pytest.warns(UserWarning, match="weightCol is ignored"):
        model = _classifier("OneVsRest")(classifier=base, weightCol="weight").fit(train)
    assert _classes(model.transform(train)) == [0.0] * 3 + [1.0] * 3 + [2.0] * 3


@pytest.mark.parametrize(
    "name,options",
    [
        ("FMClassifier", dict(factorSize=0)),
        ("FMClassifier", dict(solver="unknown")),
        ("FMClassifier", dict(miniBatchFraction=0.0)),
        ("FMClassifier", dict(maxIter=-1)),
        ("MultilayerPerceptronClassifier", dict(layers=[2])),
        ("MultilayerPerceptronClassifier", dict(layers=[2, 3], solver="unknown")),
        ("MultilayerPerceptronClassifier", dict(layers=[2, 3], initialWeights=[0.0])),
        ("OneVsRest", dict()),
        ("OneVsRest", dict(parallelism=0)),
    ],
)
def test_remaining_classifiers_reject_invalid_configuration(session, name, options):
    if name == "OneVsRest" and options:
        options = dict(options, classifier=_classifier("LinearSVC")())
    train = _triangles(session)
    if name == "FMClassifier":
        train = train.filter(F.col("label") < 2.0)
    with pytest.raises(Exception):
        _classifier(name)(**options).fit(train)


@pytest.mark.parametrize("name", ["FMClassifier", "MultilayerPerceptronClassifier", "OneVsRest"])
def test_remaining_classifiers_reject_nonvector_and_empty_data(session, name):
    options = {
        "MultilayerPerceptronClassifier": dict(layers=[2, 3]),
        "OneVsRest": dict(classifier=_classifier("LinearSVC")()),
    }.get(name, {})
    train = _triangles(session)
    with pytest.raises(Exception):
        _classifier(name)(**options).fit(train.withColumn("features", F.lit(1.0)))
    empty = train.limit(0)
    if name == "MultilayerPerceptronClassifier":
        assert _classifier(name)(**options).fit(empty).transform(empty).count() == 0
    else:
        with pytest.raises(Exception):
            _classifier(name)(**options).fit(empty)


def test_one_vs_rest_first_tie_and_nan_scores_follow_pyspark(session):
    train = _triangles(session)
    base = _classifier("LinearSVC")(maxIter=0, fitIntercept=False)
    model = _classifier("OneVsRest")(classifier=base).fit(train)
    points = session.createDataFrame(
        [(Vectors.dense([0.0, 0.0]),), (Vectors.dense([float("nan"), 0.0]),)],
        ["features"],
    )
    assert _classes(model.transform(points)) == [0.0, -1.0]


@pytest.mark.parametrize(
    "parameter", ["featuresCol", "labelCol", "predictionCol", "rawPredictionCol", "parallelism"]
)
def test_one_vs_rest_rejects_null_required_parameters(session, parameter):
    estimator = _classifier("OneVsRest")(classifier=_classifier("LinearSVC")(), **{parameter: None})
    frame = _triangles(session)
    with pytest.raises(Exception):
        estimator.fit(frame).transform(frame).collect()
