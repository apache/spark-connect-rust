# SPDX-License-Identifier: Apache-2.0
"""Python subclasses of the native ML classes against a live Connect server: PySpark's `fit`,
`transform` and `evaluate` run the subclass's `_fit`, `_transform` and `_evaluate`, also of a param
map, of a list of maps, of `fitMultiple`, and in the copies that a Pipeline and a CrossValidator
fit. Official PySpark runs them too, as the reference (`_native_classes.native_class`)."""

import os

import pytest
from _native_classes import native_class


def _cls(module, name):
    return native_class(f"pyspark.ml.connect.{module}", name, f"pyspark.ml.{module}")


@pytest.fixture(scope="module")
def session():
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live ML subclass tests")
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.remote(remote).getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture(scope="module")
def frame(session):
    from pyspark.ml.linalg import Vectors

    rows = [(Vectors.dense([float(i), float(i % 3)]), float(i % 2), float(i)) for i in range(12)]
    return session.createDataFrame(rows, ["features", "label", "x"]).coalesce(1)


def _counting(base, primitive):
    """A subclass of `base` whose `primitive` (`_fit`, ...) counts its calls, of every copy."""
    calls = []

    def hook(self, dataset):
        calls.append(self.uid)
        return getattr(super(cls, self), primitive)(dataset)

    cls = type("Counting" + base.__name__, (base,), {primitive: hook})
    return cls, calls


def test_fit_runs_the_fit_of_a_subclass(frame):
    LR = _cls("classification", "LogisticRegression")
    Counting, calls = _counting(LR, "_fit")
    estimator = Counting(maxIter=2)
    model = estimator.fit(frame)
    assert calls == [estimator.uid] and type(model).__name__ == "LogisticRegressionModel"
    assert estimator.fit(frame, {estimator.maxIter: 3}).getMaxIter() == 3
    models = estimator.fit(frame, [{}, {estimator.maxIter: 1}])
    assert [m.getMaxIter() for m in models] == [2, 1]
    fitted = sorted(i for i, _ in estimator.fitMultiple(frame, [{}, {}]))
    assert fitted == [0, 1]
    # A fit of a map or of a list fits copies, of the subclass and its uid.
    assert calls == [estimator.uid] * 6


def test_transform_and_evaluate_run_the_primitives_of_a_subclass(frame):
    Binarizer = _cls("feature", "Binarizer")
    Evaluator = _cls("evaluation", "RegressionEvaluator")
    Transforming, transforms = _counting(Binarizer, "_transform")
    Evaluating, evaluations = _counting(Evaluator, "_evaluate")
    binarizer = Transforming(threshold=4.5, inputCol="x", outputCol="b")
    assert binarizer.transform(frame).columns == ["features", "label", "x", "b"]
    assert binarizer.transform(frame, {binarizer.outputCol: "c"}).columns[-1] == "c"
    assert transforms == [binarizer.uid] * 2
    evaluator = Evaluating(predictionCol="x", labelCol="label")
    rmse = evaluator.evaluate(frame)
    assert rmse > 0 and evaluator.evaluate(frame, {evaluator.metricName: "mae"}) > 0
    assert evaluations == [evaluator.uid] * 2


def test_the_copies_of_a_pipeline_and_a_validator_fit_the_subclass(frame):
    LR = _cls("classification", "LogisticRegression")
    Pipeline = _cls("pipeline", "Pipeline")
    CrossValidator = _cls("tuning", "CrossValidator")
    Evaluator = _cls("evaluation", "BinaryClassificationEvaluator")
    from pyspark.ml.tuning import ParamGridBuilder

    Counting, calls = _counting(LR, "_fit")
    estimator = Counting(maxIter=2)
    Pipeline(stages=[estimator]).fit(frame)
    assert calls == [estimator.uid]
    grid = ParamGridBuilder().addGrid(estimator.maxIter, [1, 2]).build()
    validator = CrossValidator(
        estimator=estimator, estimatorParamMaps=grid, evaluator=Evaluator(), numFolds=2, seed=1
    )
    model = validator.fit(frame)
    # Two folds of two maps, and the fit of the best map on the whole dataset.
    assert len(calls) == 1 + 2 * 2 + 1 and set(calls) == {estimator.uid}
    assert type(model.bestModel).__name__ == "LogisticRegressionModel"


def test_copy_deepcopy_and_pickle_of_a_model_and_its_summary(frame):
    # A model of a server-side object copies (sharing the object), and is not copied deep nor
    # pickled, as PySpark's of the lock of its `RemoteModelRef`; so is its summary, which PySpark's
    # does not pickle of its DataFrames' session.
    import copy
    import pickle

    LR = _cls("classification", "LogisticRegression")
    Pipeline = _cls("pipeline", "Pipeline")
    Binarizer = _cls("feature", "Binarizer")
    model = LR(maxIter=2).fit(frame)
    model.tag = [1]
    shallow = copy.copy(model)
    assert type(shallow) is type(model) and shallow.uid == model.uid and shallow.tag is model.tag
    assert shallow.numFeatures == 2 and shallow.transform(frame).count() == 12
    for call in (lambda: copy.deepcopy(model), lambda: pickle.dumps(model)):
        with pytest.raises(TypeError, match=r"^cannot pickle '_thread.lock' object$"):
            call()
    summary = model.summary
    assert copy.copy(summary).totalIterations == summary.totalIterations
    with pytest.raises(TypeError, match=r"^cannot pickle '_thread.lock' object$"):
        copy.deepcopy(summary)
    with pytest.raises(AttributeError, match="Can't pickle local object"):
        pickle.dumps(summary)
    # A PipelineModel of transformers copies, copies deep and pickles, of its stages.
    fitted = Pipeline(stages=[Binarizer(threshold=4.5, inputCol="x", outputCol="b")]).fit(frame)
    for copied in (copy.copy(fitted), copy.deepcopy(fitted), pickle.loads(pickle.dumps(fitted))):
        assert len(copied.stages) == 1 and copied.uid == fitted.uid
        assert copied.transform(frame).columns[-1] == "b"
