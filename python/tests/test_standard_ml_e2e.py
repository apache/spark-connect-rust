# SPDX-License-Identifier: Apache-2.0
"""Migrated standard estimators must execute the Rust ML implementation."""

import gc
import json
import os
import sys

import pytest
from pyspark.ml.classification import LinearSVC, LogisticRegression
from pyspark.ml.linalg import Vectors
from pyspark.ml.regression import GeneralizedLinearRegression, LinearRegression
from pyspark.sql import SparkSession

_REGRESSORS = (LinearRegression, GeneralizedLinearRegression)


@pytest.fixture(
    params=[LogisticRegression, LinearSVC, LinearRegression, GeneralizedLinearRegression],
    ids=lambda cls: cls.__name__,
)
def estimator_cls(request):
    return request.param


@pytest.fixture(scope="module")
def session():
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for standard ML delegation tests")
    spark = SparkSession.builder.remote(remote).getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture
def frame(session, estimator_cls):
    if estimator_cls in _REGRESSORS:
        return session.createDataFrame(
            [(2.0 * i + 1.0, Vectors.dense([float(i)])) for i in range(8)],
            ["label", "features"],
        )
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
        pytest.fail("Standard estimator_cls used a Python ML request builder")

    monkeypatch.setattr(serialize, "serialize_ml_params", python_ml_request)
    monkeypatch.setattr(proto.TransformerRelation, "plan", python_ml_request)
    monkeypatch.setattr(SparkConnectClient, "execute_command", python_ml_request)


def test_fit_transform_fetch_and_persistence(frame, tmp_path, native_operations_only, estimator_cls):
    estimator = estimator_cls(maxIter=20, regParam=0.0 if estimator_cls in _REGRESSORS else 0.1)
    model = estimator.fit(frame)
    assert type(model) is getattr(sys.modules[estimator_cls.__module__], estimator_cls.__name__ + "Model")
    assert model.uid == estimator.uid and model.getMaxIter() == 20
    assert model.numFeatures == 1
    if estimator_cls not in _REGRESSORS:
        assert model.numClasses == 2
    from pyspark import _pyspark

    assert model.setPredictionCol("eager_prediction") is model
    assert _pyspark._ml_params(model)[0]["predictionCol"] == "eager_prediction"
    assert len(model.transform(frame).select("eager_prediction").collect()) == frame.count()
    model.setPredictionCol("prediction")
    assert model.coefficients[0] > 0
    summary = model.summary() if estimator_cls is LinearSVC else model.summary
    prefix = "BinaryLogisticRegression" if estimator_cls is LogisticRegression else estimator_cls.__name__
    public = sys.modules[estimator_cls.__module__]
    assert type(summary) is getattr(public, prefix + "TrainingSummary")
    evaluation = model.evaluate(frame)
    assert type(evaluation) is getattr(public, prefix + "Summary")
    if estimator_cls in _REGRESSORS:
        assert list(model.coefficients) == pytest.approx([2.0])
        assert model.intercept == pytest.approx(1.0)
        if estimator_cls is GeneralizedLinearRegression:
            assert summary.deviance == pytest.approx(0.0, abs=1e-10)
            assert evaluation.deviance == pytest.approx(0.0, abs=1e-10)
            assert summary.nullDeviance > 0
            assert summary.solver == "irls" and summary.numIterations >= 0
            residuals = summary.residuals("response")
        else:
            assert summary.rootMeanSquaredError == pytest.approx(0.0, abs=1e-10)
            assert evaluation.r2 == pytest.approx(1.0)
            residuals = summary.residuals
        assert summary.predictions.count() == frame.count()
        assert residuals.count() == frame.count()
        assert len(summary.coefficientStandardErrors) == len(summary.tValues) == len(summary.pValues) == 2
        assert model.predict(Vectors.dense([3.0])) == pytest.approx(7.0)
    else:
        assert summary.accuracy == pytest.approx(1.0)
        assert evaluation.accuracy == pytest.approx(1.0)
        assert model.predict(Vectors.dense([3.0])) == 1.0

    rows = model.transform(frame).orderBy("label").select("prediction").collect()
    expected = [2.0 * i + 1.0 for i in range(8)] if estimator_cls in _REGRESSORS else [0.0, 0.0, 1.0, 1.0]
    assert [row.prediction for row in rows] == pytest.approx(expected)

    # A call-local override fits a copy, preserving the estimator and its Python parameter map.
    assert estimator.fit(frame, {estimator.maxIter: 2}).getMaxIter() == 2
    assert estimator.getMaxIter() == 20

    model_path = str(tmp_path / "model")
    model.write().overwrite().save(model_path)
    loaded = type(model).load(model_path)
    assert type(loaded) is type(model) and loaded.uid == model.uid
    assert list(loaded.coefficients) == pytest.approx(list(model.coefficients))
    assert loaded.transform(frame).select("prediction").collect() == (
        model.transform(frame).select("prediction").collect()
    )

    # The writer reads Python params at save time, even if created before a setter call.
    estimator_path = str(tmp_path / "estimator")
    writer = estimator.write()
    estimator.setMaxIter(7)
    writer.save(estimator_path)
    restored = estimator_cls.load(estimator_path)
    assert type(restored) is estimator_cls and restored.getMaxIter() == 7


def test_lazy_transform_retains_model_after_python_reference_is_dropped(
    frame, native_operations_only, estimator_cls,
):
    model = estimator_cls(maxIter=20, regParam=0.0 if estimator_cls in _REGRESSORS else 0.1).fit(frame)
    predictions = model.transform(frame).select("prediction")
    del model
    gc.collect()
    expected = [2.0 * i + 1.0 for i in range(8)] if estimator_cls in _REGRESSORS else [0.0, 0.0, 1.0, 1.0]
    assert sorted(row.prediction for row in predictions.collect()) == pytest.approx(expected)


@pytest.mark.parametrize("holder", ["copy", "shallow", "summary"])
def test_deleting_native_model_wrappers_updates_counts_and_cache(frame, estimator_cls, holder):
    model = estimator_cls(maxIter=3, regParam=0.1).fit(frame)
    reference = model._java_obj
    assert reference._ref_count == 2
    if holder == "copy":
        kept = model.copy()
        assert kept._java_obj is reference and reference._ref_count == 3
        remaining = 2
    elif holder == "shallow":
        import copy

        kept = copy.copy(model)
        assert kept._java_obj is reference and reference._ref_count == 2
        remaining = 2
    else:
        kept = model.summary() if estimator_cls is LinearSVC else model.summary
        remaining = 1
    del model
    assert reference._ref_count == remaining
    client = frame.sparkSession.client
    assert reference.ref_id in {json.loads(item)["id"] for item in client._get_ml_cache_info()}
    del kept
    assert reference._ref_count == 0
    # No polling or forced collection: the next query must observe the finalizer's deletion.
    assert reference.ref_id not in {json.loads(item)["id"] for item in client._get_ml_cache_info()}


def test_python_fit_override_can_delegate_to_rust(frame, native_operations_only, estimator_cls):
    calls = []

    class RecordingEstimator(estimator_cls):
        def _fit(self, dataset):
            calls.append(self.getMaxIter())
            return super()._fit(dataset)

    estimator = RecordingEstimator(maxIter=20, regParam=0.1)
    assert type(estimator.fit(frame, {estimator.maxIter: 2})) is getattr(
        sys.modules[estimator_cls.__module__], estimator_cls.__name__ + "Model",
    )
    assert calls == [2] and estimator.getMaxIter() == 20


def test_standard_pipeline_and_tuning_use_native_estimator(session, tmp_path, estimator_cls):
    from pyspark import _pyspark
    from pyspark.ml import Pipeline, PipelineModel
    from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

    data = session.createDataFrame(
        [(float(i % 2), float(i % 2), (i // 2) % 2) for i in range(40)],
        ["label", "x", "fold"],
    )
    estimator = estimator_cls(maxIter=3)
    pipeline = Pipeline(stages=[VectorAssembler(inputCols=["x"], outputCol="features"), estimator])
    validator = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=ParamGridBuilder().addGrid(
            estimator.regParam, [0.0, 0.1] if estimator_cls in _REGRESSORS else [0.1, 0.2],
        ).build(),
        evaluator=RegressionEvaluator() if estimator_cls in _REGRESSORS else BinaryClassificationEvaluator(),
        numFolds=2,
        foldCol="fold",
        parallelism=1,
    )
    model = validator.fit(data)
    assert isinstance(model.bestModel.stages[-1], getattr(_pyspark, estimator_cls.__name__ + "Model"))
    predictions = model.transform(data).select("prediction").collect()
    assert sorted(row.prediction for row in predictions) == pytest.approx([0.0] * 20 + [1.0] * 20)
    path = str(tmp_path / "pipeline")
    model.bestModel.write().overwrite().save(path)
    restored = PipelineModel.load(path)
    assert isinstance(restored.stages[-1], getattr(_pyspark, estimator_cls.__name__ + "Model"))
    restored_predictions = restored.transform(data).select("prediction").collect()
    assert sorted(row.prediction for row in restored_predictions) == pytest.approx([0.0] * 20 + [1.0] * 20)


def test_live_standard_estimator_without_optional_python_dependencies(session, tmp_path, estimator_cls):
    from _standard_ml_cases import without_optional_dependencies

    regression = estimator_cls in _REGRESSORS
    generalized = estimator_cls is GeneralizedLinearRegression
    name = estimator_cls.__name__
    without_optional_dependencies(f"""
import gc
import os
from pyspark import _pyspark
from pyspark.sql import SparkSession
from {estimator_cls.__module__} import {name}, {name}Model
from pyspark.ml.linalg import Vectors

spark = SparkSession.builder.remote(os.environ["SPARK_REMOTE"]).getOrCreate()
try:
    regression = {regression!r}
    data = spark.createDataFrame(
        [(2.0 * i + 1.0 if regression else float(i >= 2), Vectors.dense([float(i)])) for i in range(4)],
        ["label", "features"],
    )
    estimator = {name}(maxIter=10, regParam=0.0 if regression else 0.1)
    model = estimator.fit(data)
    assert isinstance(model, _pyspark.{name}Model)
    summary = model.summary
    summary = summary() if callable(summary) else summary
    evaluation = model.evaluate(data)
    if {generalized!r}:
        assert summary.deviance < 1e-10
        assert evaluation.deviance < 1e-10
    elif regression:
        assert summary.rootMeanSquaredError < 1e-10
        assert evaluation.rootMeanSquaredError < 1e-10
    else:
        assert summary.accuracy > 0.9
        assert evaluation.accuracy > 0.9
    path = {str(tmp_path / "dependency-free-model")!r}
    model.write().overwrite().save(path)
    restored = {name}Model.load(path)
    predictions = restored.transform(data).select("prediction")
    del restored
    gc.collect()
    expected = [1.0, 3.0, 5.0, 7.0] if regression else [0.0, 0.0, 1.0, 1.0]
    actual = sorted(row.prediction for row in predictions.collect())
    assert len(actual) == len(expected)
    assert all(abs(a - e) < 1e-10 for a, e in zip(actual, expected))
finally:
    spark.stop()
""")
