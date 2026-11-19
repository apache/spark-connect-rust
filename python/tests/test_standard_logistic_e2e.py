# SPDX-License-Identifier: Apache-2.0
"""Standard LogisticRegression imports must execute the Rust ML implementation."""

import gc
import os

import pytest
from pyspark.ml.classification import (
    BinaryLogisticRegressionTrainingSummary,
    LogisticRegression,
    LogisticRegressionModel,
)
from pyspark.ml.linalg import Vectors
from pyspark.sql import SparkSession


@pytest.fixture(scope="module")
def session():
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for standard LogisticRegression delegation tests")
    spark = SparkSession.builder.remote(remote).getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def frame(session):
    return session.createDataFrame(
        [(float(i >= 2), Vectors.dense([float(i)])) for i in range(4)],
        ["label", "features"],
    )


@pytest.fixture
def native_operations_only(monkeypatch):
    # Rust transport alone would still permit Python to build fit/fetch/read/write requests.
    # Disable those builders and the transform relation so this test proves native ML routing.
    import pyspark.ml.connect.proto as proto
    import pyspark.ml.connect.serialize as serialize
    from pyspark.sql.connect.client import SparkConnectClient

    def python_ml_request(*args, **kwargs):
        pytest.fail("Standard LogisticRegression used a Python ML request builder")

    monkeypatch.setattr(serialize, "serialize_ml_params", python_ml_request)
    monkeypatch.setattr(proto.TransformerRelation, "plan", python_ml_request)
    monkeypatch.setattr(SparkConnectClient, "execute_command", python_ml_request)


def test_fit_transform_fetch_and_persistence(frame, tmp_path, native_operations_only):
    lr = LogisticRegression(maxIter=20, regParam=0.1)
    model = lr.fit(frame)
    assert type(model) is LogisticRegressionModel
    assert model.uid == lr.uid and model.getMaxIter() == 20
    assert model.numClasses == 2 and model.numFeatures == 1
    assert model.coefficients[0] > 0
    summary = model.summary
    assert type(summary) is BinaryLogisticRegressionTrainingSummary
    assert summary.accuracy == pytest.approx(1.0)
    assert model.evaluate(frame).accuracy == pytest.approx(1.0)
    assert model.predict(Vectors.dense([3.0])) == 1.0
    rows = model.transform(frame).orderBy("label").select("prediction").collect()
    assert [row.prediction for row in rows] == [0.0, 0.0, 1.0, 1.0]

    # A call-local override fits a copy, preserving the estimator and its Python parameter map.
    assert lr.fit(frame, {lr.maxIter: 2}).getMaxIter() == 2
    assert lr.getMaxIter() == 20

    model_path = str(tmp_path / "model")
    model.write().overwrite().save(model_path)
    loaded = LogisticRegressionModel.load(model_path)
    assert type(loaded) is LogisticRegressionModel and loaded.uid == model.uid
    assert list(loaded.coefficients) == pytest.approx(list(model.coefficients))
    assert loaded.transform(frame).select("prediction").collect() == (
        model.transform(frame).select("prediction").collect()
    )

    # The writer reads Python params at save time, even if created before a setter call.
    estimator_path = str(tmp_path / "estimator")
    writer = lr.write()
    lr.setMaxIter(7)
    writer.save(estimator_path)
    restored = LogisticRegression.load(estimator_path)
    assert type(restored) is LogisticRegression and restored.getMaxIter() == 7


def test_lazy_transform_retains_model_after_python_reference_is_dropped(
    frame, native_operations_only,
):
    model = LogisticRegression(maxIter=20, regParam=0.1).fit(frame)
    predictions = model.transform(frame).select("prediction")
    del model
    gc.collect()
    assert sorted(row.prediction for row in predictions.collect()) == [0.0, 0.0, 1.0, 1.0]


def test_python_fit_override_can_delegate_to_rust(frame, native_operations_only):
    calls = []

    class RecordingLR(LogisticRegression):
        def _fit(self, dataset):
            calls.append(self.getMaxIter())
            return super()._fit(dataset)

    lr = RecordingLR(maxIter=20, regParam=0.1)
    assert type(lr.fit(frame, {lr.maxIter: 2})) is LogisticRegressionModel
    assert calls == [2] and lr.getMaxIter() == 20
