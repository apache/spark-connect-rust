# SPDX-License-Identifier: Apache-2.0
"""The classes of the objects that the native regression and clustering classes make, against a
live Connect server. Every object that this client makes in Rust, of a fit (of a param map too), a
fit of a list of param maps, fitMultiple, a load (of its class and of read()), a copy (of extra
params too), a summary, an evaluation, DistributedLDAModel.toLocal and the trees of an ensemble, is
of the public class of pyspark.ml.connect, as PySpark's is of its class. The
vendored pyspark.ml.Pipeline, CrossValidator and TrainValidationSplit fit, save and load the
native stages. Official PySpark runs the tests too, as the reference, of its classes
(`_native_classes.native_class`).
"""

import os
import tempfile

import pytest
from _clustering_api_cases import DOCUMENTS, POINTS, _vectors
from _native_classes import native_class
from _regression_api_cases import KWARGS, _regression_frame

REGRESSION = "pyspark.ml.connect.regression", "pyspark.ml.regression"
CLUSTERING = "pyspark.ml.connect.clustering", "pyspark.ml.clustering"
EVALUATION = "pyspark.ml.connect.evaluation", "pyspark.ml.evaluation"

# The estimators of each module, of their model classes and keyword arguments, and the frame that
# they fit.
ESTIMATORS = {
    "LinearRegression": (REGRESSION, "LinearRegressionModel", KWARGS["LinearRegression"], "rows"),
    "GeneralizedLinearRegression": (
        REGRESSION,
        "GeneralizedLinearRegressionModel",
        KWARGS["GeneralizedLinearRegression"],
        "rows",
    ),
    "AFTSurvivalRegression": (
        REGRESSION,
        "AFTSurvivalRegressionModel",
        KWARGS["AFTSurvivalRegression"],
        "rows",
    ),
    "IsotonicRegression": (
        REGRESSION,
        "IsotonicRegressionModel",
        KWARGS["IsotonicRegression"],
        "rows",
    ),
    "FMRegressor": (REGRESSION, "FMRegressionModel", KWARGS["FMRegressor"], "rows"),
    "DecisionTreeRegressor": (
        REGRESSION,
        "DecisionTreeRegressionModel",
        KWARGS["DecisionTreeRegressor"],
        "rows",
    ),
    "RandomForestRegressor": (
        REGRESSION,
        "RandomForestRegressionModel",
        KWARGS["RandomForestRegressor"],
        "rows",
    ),
    "GBTRegressor": (REGRESSION, "GBTRegressionModel", KWARGS["GBTRegressor"], "rows"),
    "KMeans": (CLUSTERING, "KMeansModel", {"k": 3, "seed": 5}, "points"),
    "BisectingKMeans": (CLUSTERING, "BisectingKMeansModel", {"k": 3, "seed": 5}, "points"),
    "GaussianMixture": (
        CLUSTERING,
        "GaussianMixtureModel",
        {"k": 2, "seed": 7, "maxIter": 5},
        "points",
    ),
    "LDA": (
        CLUSTERING,
        "DistributedLDAModel",
        {"k": 2, "optimizer": "em", "maxIter": 3, "seed": 42},
        "documents",
    ),
}

# The summary classes of a fit's `summary`, and of an `evaluate`, of the models that have them.
SUMMARIES = {
    "LinearRegression": ("LinearRegressionTrainingSummary", "LinearRegressionSummary"),
    "GeneralizedLinearRegression": (
        "GeneralizedLinearRegressionTrainingSummary",
        "GeneralizedLinearRegressionSummary",
    ),
    "KMeans": ("KMeansSummary", None),
    "BisectingKMeans": ("BisectingKMeansSummary", None),
    "GaussianMixture": ("GaussianMixtureSummary", None),
}


def cls(modules, name):
    native, reference = modules
    return native_class(native, name, reference)


def official(module, name):
    """PySpark's class `name` of `module`: the vendored class in this client, the JVM-backed one
    in official PySpark."""
    import importlib

    return getattr(importlib.import_module(module), name)


def _extra(stage):
    # A param map of a param of the stage of a value that it has, for a copy with extra params and
    # a fit of a param map, as the feature tests make it.
    param = next(p for p in stage.params if stage.isDefined(p))
    return {param: stage.getOrDefault(param)}


@pytest.fixture(scope="module")
def session():
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live ML class tests")
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.remote(remote).getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture(scope="module")
def frames(session):
    return {
        "rows": _regression_frame(session),
        "points": _vectors(session, [[p] for p in POINTS], ["features"]),
        "documents": _vectors(session, [[d] for d in DOCUMENTS], ["features"]),
    }


@pytest.fixture
def scratch():
    with tempfile.TemporaryDirectory() as path:
        yield path


@pytest.mark.parametrize("name", list(ESTIMATORS))
def test_models_and_estimators_made_in_rust_are_of_their_public_classes(
    name, session, frames, scratch
):
    modules, model_name, kwargs, frame = ESTIMATORS[name]
    estimator_class, model_class = cls(modules, name), cls(modules, model_name)
    est = estimator_class(**kwargs)
    df = frames[frame]
    model = est.fit(df)
    assert type(model) is model_class
    assert type(model.copy()) is model_class
    assert type(model.copy(_extra(model))) is model_class
    assert type(est.copy()) is estimator_class
    assert type(est.copy(_extra(est))) is estimator_class
    assert type(est.fit(df, _extra(est))) is model_class
    (listed,) = est.fit(df, [{}])
    assert type(listed) is model_class
    index, fitted = next(est.fitMultiple(df, [{}]))
    assert index == 0 and type(fitted) is model_class

    model.write().overwrite().save(scratch + "/model")
    loaded = model_class.load(scratch + "/model")
    assert type(loaded) is model_class
    assert type(loaded.copy()) is model_class
    assert type(model_class.read().load(scratch + "/model")) is model_class
    est.write().overwrite().save(scratch + "/estimator")
    assert type(estimator_class.load(scratch + "/estimator")) is estimator_class
    assert type(estimator_class.read().load(scratch + "/estimator")) is estimator_class

    if name in SUMMARIES:
        training, evaluated = SUMMARIES[name]
        assert type(model.summary) is cls(modules, training)
        if evaluated is not None:
            assert type(model.evaluate(df)) is cls(modules, evaluated)


@pytest.mark.parametrize("name", ["RandomForestRegressor", "GBTRegressor"])
def test_the_trees_of_an_ensemble_are_of_the_public_tree_class(name, session, frames, scratch):
    modules, model_name, kwargs, frame = ESTIMATORS[name]
    model = cls(modules, name)(**kwargs).fit(frames[frame])
    tree_class = cls(REGRESSION, "DecisionTreeRegressionModel")
    assert model.trees and all(type(tree) is tree_class for tree in model.trees)
    model.write().overwrite().save(scratch + "/model")
    loaded = cls(modules, model_name).load(scratch + "/model")
    assert loaded.trees and all(type(tree) is tree_class for tree in loaded.trees)


def test_the_lda_models_are_of_their_public_classes(session, frames, scratch):
    lda, base = cls(CLUSTERING, "LDA"), cls(CLUSTERING, "LDAModel")
    local_class = cls(CLUSTERING, "LocalLDAModel")
    distributed_class = cls(CLUSTERING, "DistributedLDAModel")
    df = frames["documents"]
    distributed = lda(k=2, optimizer="em", maxIter=3, seed=42).fit(df)
    local = lda(k=2, optimizer="online", maxIter=3, seed=42).fit(df)
    assert type(distributed) is distributed_class and type(local) is local_class
    assert isinstance(distributed, base) and isinstance(local, base)
    assert type(distributed.toLocal()) is local_class
    assert isinstance(distributed.toLocal(), base)
    local.write().overwrite().save(scratch + "/local")
    assert type(local_class.load(scratch + "/local")) is local_class
    assert type(local_class.read().load(scratch + "/local")) is local_class


def test_power_iteration_clustering_loads_as_its_public_class(session, scratch):
    pic_class = cls(CLUSTERING, "PowerIterationClustering")
    pic = pic_class(k=2, maxIter=5)
    pic.write().overwrite().save(scratch + "/pic")
    assert type(pic_class.load(scratch + "/pic")) is pic_class
    assert type(pic_class.read().load(scratch + "/pic")) is pic_class
    assert type(pic.copy()) is pic_class
    assert type(pic.copy(_extra(pic))) is pic_class


def test_the_vendored_pipeline_fits_saves_and_loads_native_stages(session, frames, scratch):
    from pyspark.ml import Pipeline, PipelineModel

    regression = cls(REGRESSION, "LinearRegression")(**KWARGS["LinearRegression"])
    kmeans = cls(CLUSTERING, "KMeans")(k=2, seed=5, predictionCol="cluster")
    df = frames["rows"]
    model = Pipeline(stages=[regression, kmeans]).fit(df)
    assert [type(stage) for stage in model.stages] == [
        cls(REGRESSION, "LinearRegressionModel"),
        cls(CLUSTERING, "KMeansModel"),
    ]
    assert {"prediction", "cluster"} <= set(model.transform(df).columns)
    model.write().overwrite().save(scratch + "/pipeline")
    loaded = PipelineModel.load(scratch + "/pipeline")
    # The vendored loader makes each stage of PySpark's class of the stage's JVM class
    # (DefaultParamsReader), as official PySpark's does: of pyspark.ml.<module>, which the public
    # class of the native stage subclasses.
    assert [type(stage) for stage in loaded.stages] == [
        official("pyspark.ml.regression", "LinearRegressionModel"),
        official("pyspark.ml.clustering", "KMeansModel"),
    ]
    assert [stage.uid for stage in loaded.stages] == [stage.uid for stage in model.stages]
    assert {"prediction", "cluster"} <= set(loaded.transform(df).columns)


@pytest.mark.parametrize("validator", ["CrossValidator", "TrainValidationSplit"])
def test_the_vendored_validators_fit_save_and_load_a_native_estimator(
    validator, session, frames, scratch
):
    import pyspark.ml.tuning as tuning
    from pyspark.ml.tuning import ParamGridBuilder

    regression_class = cls(REGRESSION, "LinearRegression")
    model_class = cls(REGRESSION, "LinearRegressionModel")
    evaluator = cls(EVALUATION, "RegressionEvaluator")()
    estimator = regression_class(maxIter=5)
    # The Params of the estimator's own: a validator's save requires that each Param of the maps
    # is the estimator's, which a class Param (of the parent "undefined") is not.
    grid = ParamGridBuilder().addGrid(estimator.regParam, [0.01, 0.5]).build()
    validator_class = getattr(tuning, validator)
    extra = {"numFolds": 2} if validator == "CrossValidator" else {"trainRatio": 0.75}
    model = validator_class(
        estimator=estimator,
        estimatorParamMaps=grid,
        evaluator=evaluator,
        seed=3,
        **extra,
    ).fit(frames["rows"])
    assert type(model.bestModel) is model_class
    model.write().overwrite().save(scratch + "/validator")
    loaded = getattr(tuning, validator + "Model").load(scratch + "/validator")
    # Of PySpark's class of its JVM class, as the vendored loader makes it (see the Pipeline's).
    assert type(loaded.bestModel) is official("pyspark.ml.regression", "LinearRegressionModel")
    assert loaded.bestModel.uid == model.bestModel.uid
    assert "prediction" in loaded.transform(frames["rows"]).columns
