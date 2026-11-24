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
    from pyspark import _pyspark

    assert model.setPredictionCol("eager_prediction") is model
    assert _pyspark._ml_params(model)[0]["predictionCol"] == "eager_prediction"
    assert len(model.transform(frame).select("eager_prediction").collect()) == 4
    model.setPredictionCol("prediction")
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


def test_standard_pipeline_and_tuning_use_native_logistic(session, tmp_path):
    from pyspark import _pyspark
    from pyspark.ml import Pipeline, PipelineModel
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

    data = session.createDataFrame(
        [(float(i % 2), float(i % 2), (i // 2) % 2) for i in range(40)],
        ["label", "x", "fold"],
    )
    lr = LogisticRegression(maxIter=3)
    pipeline = Pipeline(stages=[VectorAssembler(inputCols=["x"], outputCol="features"), lr])
    validator = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.2]).build(),
        evaluator=BinaryClassificationEvaluator(),
        numFolds=2,
        foldCol="fold",
        parallelism=1,
    )
    model = validator.fit(data)
    assert isinstance(model.bestModel.stages[-1], _pyspark.LogisticRegressionModel)
    predictions = model.transform(data).select("prediction").collect()
    assert sorted(row.prediction for row in predictions) == [0.0] * 20 + [1.0] * 20
    path = str(tmp_path / "pipeline")
    model.bestModel.write().overwrite().save(path)
    restored = PipelineModel.load(path)
    assert isinstance(restored.stages[-1], _pyspark.LogisticRegressionModel)
    restored_predictions = restored.transform(data).select("prediction").collect()
    assert sorted(row.prediction for row in restored_predictions) == [0.0] * 20 + [1.0] * 20


def test_live_standard_logistic_without_optional_python_dependencies(session, tmp_path):
    from _standard_logistic_cases import without_optional_dependencies

    without_optional_dependencies(f"""
import gc
import os
from pyspark import _pyspark
from pyspark.sql import SparkSession
from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel
from pyspark.ml.linalg import Vectors

spark = SparkSession.builder.remote(os.environ["SPARK_REMOTE"]).getOrCreate()
try:
    data = spark.createDataFrame(
        [(float(i >= 2), Vectors.dense([float(i)])) for i in range(4)],
        ["label", "features"],
    )
    estimator = LogisticRegression(maxIter=10, regParam=0.1)
    model = estimator.fit(data)
    assert isinstance(model, _pyspark.LogisticRegressionModel)
    assert model.summary.accuracy > 0.9
    assert model.evaluate(data).accuracy > 0.9
    path = {str(tmp_path / "dependency-free-model")!r}
    model.write().overwrite().save(path)
    restored = LogisticRegressionModel.load(path)
    predictions = restored.transform(data).select("prediction")
    del restored
    gc.collect()
    assert sorted(row.prediction for row in predictions.collect()) == [0.0, 0.0, 1.0, 1.0]
finally:
    spark.stop()
""")
