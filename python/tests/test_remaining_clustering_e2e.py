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
"""Remaining native clustering algorithms, with an explicit official-reference mode."""

import gc
import json
import math
import os
from pathlib import Path

import pytest
from pyspark.errors import PySparkException
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

GOLDEN = json.loads((Path(__file__).parents[2] / "tests/golden/remaining_clustering.json").read_text())
REFERENCE = os.environ.get("CLUSTERING_REFERENCE") == "1"


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


def _algorithm(name):
    if REFERENCE:
        from pyspark.ml import clustering
        return getattr(clustering, name)
    from pyspark import _pyspark
    from pyspark.ml.connect import clustering
    cls = getattr(clustering, name)
    assert cls is getattr(_pyspark, name), "The native tests must exercise the Rust bindings"
    return cls


def _configured(name, values, setters):
    cls = _algorithm(name)
    if not setters:
        return cls(**values)
    instance = cls()
    for key, value in values.items():
        assert getattr(instance, "set" + key[0].upper() + key[1:])(value) is instance
    return instance


def _documents(session, rows, arrays=False):
    def vector(i, values):
        if arrays:
            return values
        if i % 2:
            return Vectors.sparse(len(values), {j: x for j, x in enumerate(values) if x})
        return Vectors.dense(values)
    return session.createDataFrame([(i, vector(i, x)) for i, x in enumerate(rows)], ["id", "input"]).coalesce(1)


def _topics(model, frame):
    return model.transform(frame).orderBy("id").select(vector_to_array("topics").alias("p"))


@pytest.mark.parametrize("case", GOLDEN["lda"], ids=lambda c: c["values"]["optimizer"])
@pytest.mark.parametrize("setters", [False, True], ids=["constructor", "setters"])
def test_lda_matches_official_dense_sparse_inference_and_retains_model(session, case, setters):
    train = _documents(session, GOLDEN["documents"])
    held = _documents(session, GOLDEN["held_out"])
    estimator = _configured("LDA", case["values"], setters)
    model = estimator.fit(train)
    predictions = _topics(model, held)
    estimator.setTopicDistributionCol("changed")
    del model, estimator
    gc.collect()
    rows = predictions.collect()
    assert len(rows) == len(case["topics"])
    for row, expected in zip(rows, case["topics"]):
        assert list(row.p) == pytest.approx(expected, rel=1e-6, abs=1e-8)
        assert all(math.isfinite(x) and x >= 0 for x in row.p)
    assert list(rows[-1].p) == [0.0, 0.0]
    for row in rows[:-1]:
        assert sum(row.p) == pytest.approx(1.0)
    labels = [max(range(2), key=row.p.__getitem__) for row in rows[:-1]]
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[0] != labels[2]


@pytest.mark.parametrize("optimizer", ["online", "em"])
def test_lda_repeated_fits_keep_independent_topic_counts(session, optimizer):
    train = _documents(session, GOLDEN["documents"])
    estimator = _algorithm("LDA")(featuresCol="input", topicDistributionCol="topics", k=2,
                                  maxIter=3, seed=2**33, optimizer=optimizer,
                                  subsamplingRate=1.0, checkpointInterval=-1)
    first = estimator.fit(train)
    second = estimator.setK(3).fit(train)
    assert all(len(r.p) == 2 for r in _topics(first, train).collect())
    assert all(len(r.p) == 3 for r in _topics(second, train).collect())


@pytest.mark.skipif(REFERENCE, reason="Numeric array features are a native client extension")
def test_lda_native_array_features_are_sent_as_they_are_as_pyspark_ml(session):
    # As pyspark.ml's LDA, the array features of createDataFrame (whose elements may be null) are
    # sent as they are, and the JVM, which takes a vector or an array of non-null elements, rejects
    # them, where the deprecated pyspark.ml.connect converted them.
    train = _documents(session, GOLDEN["documents"], arrays=True)
    with pytest.raises(PySparkException, match="must be of type equal to one of"):
        _algorithm("LDA")(**GOLDEN["lda"][0]["values"]).fit(train)


@pytest.mark.parametrize("options", [{"k": 1}, {"maxIter": -1}, {"optimizer": "unknown"}, {"learningDecay": 0.0}])
def test_lda_rejects_invalid_configuration(session, options):
    with pytest.raises(PySparkException):
        _algorithm("LDA")(featuresCol="input", **options).fit(_documents(session, GOLDEN["documents"]))


@pytest.mark.parametrize("kind", ["empty", "nonvector"])
def test_lda_rejects_invalid_training_data(session, kind):
    train = _documents(session, GOLDEN["documents"])
    if kind == "empty":
        train = train.limit(0)
    else:
        train = train.withColumn("input", F.lit(1.0))
    with pytest.raises(PySparkException):
        _algorithm("LDA")(featuresCol="input", k=2, maxIter=2).fit(train)


def _graph(session):
    return session.createDataFrame([tuple(x) for x in GOLDEN["edges"]], ["source", "target", "affinity"]).repartition(1)


def _assert_partition(actual, expected):
    assert [r.id for r in actual] == [r[0] for r in expected]
    for i in range(len(actual)):
        for j in range(len(actual)):
            assert (actual[i].cluster == actual[j].cluster) == (expected[i][1] == expected[j][1])


@pytest.mark.parametrize("case", GOLDEN["pic"], ids=lambda c: c["values"]["initMode"] + ("-weighted" if "weightCol" in c["values"] else "-unweighted"))
@pytest.mark.parametrize("setters", [False, True], ids=["constructor", "setters"])
def test_pic_matches_official_assignments_and_snapshots_parameters(session, case, setters):
    algorithm = _configured("PowerIterationClustering", case["values"], setters)
    result = algorithm.assignClusters(_graph(session))
    algorithm.setK(1)  # An invalid later mutation cannot change the already-built relation.
    del algorithm
    gc.collect()
    assert [(f.name, f.dataType.simpleString()) for f in result.schema.fields] == [("id", "bigint"), ("cluster", "int")]
    _assert_partition(result.orderBy("id").collect(), case["assignments"])


def test_pic_default_columns_and_empty_weight_column(session):
    graph = _graph(session).selectExpr("source AS src", "target AS dst", "affinity AS weight")
    cls = _algorithm("PowerIterationClustering")
    default = cls().assignClusters(graph).orderBy("id").collect()
    empty_weight = cls(weightCol="").assignClusters(graph).orderBy("id").collect()
    _assert_partition(default, [[r.id, r.cluster] for r in empty_weight])
    assert [r.id for r in default] == list(range(6))


@pytest.mark.parametrize("options", [{"k": 1}, {"maxIter": -1}, {"initMode": "unknown"}, {"weightCol": "missing"}])
def test_pic_rejects_invalid_configuration(session, options):
    values = dict(srcCol="source", dstCol="target", **options)
    with pytest.raises(PySparkException):
        _algorithm("PowerIterationClustering")(**values).assignClusters(_graph(session)).collect()


def test_pic_rejects_negative_edge_weights(session):
    graph = _graph(session).withColumn("affinity", -F.col("affinity"))
    with pytest.raises(PySparkException):
        _algorithm("PowerIterationClustering")(srcCol="source", dstCol="target", weightCol="affinity").assignClusters(graph).collect()


@pytest.mark.skipif(REFERENCE, reason="Official PIC loses the cached input model after GC; native plans retain it")
def test_pic_retains_cached_models_in_its_input_plan(session):
    lda = _algorithm("LDA")(**GOLDEN["lda"][0]["values"]).fit(_documents(session, GOLDEN["documents"]))
    from pyspark.ml.functions import array_to_vector
    graph = _graph(session).withColumn("input", array_to_vector(F.array(*[F.lit(x) for x in GOLDEN["documents"][0]])))
    transformed = lda.transform(graph).select("source", "target", "affinity")
    case = GOLDEN["pic"][-1]
    result = _algorithm("PowerIterationClustering")(**case["values"]).assignClusters(transformed)
    del lda, graph, transformed
    gc.collect()
    _assert_partition(result.orderBy("id").collect(), case["assignments"])
