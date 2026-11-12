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
"""
pyspark.ml on this client against a live Spark Connect server: the estimators, models,
transformers, evaluators, tuning, statistics and persistence of the vendored pyspark.ml, over the
Rust transport. The expected values are those of the reference client; the tests pass with it as
well.
"""

import gc
import os

import pytest
from pyspark.loose_version import LooseVersion

# The generated pyspark.sql.connect.proto, of Apache Spark v4.3.0, needs protobuf >= 6.33.5.
_protobuf = pytest.importorskip("google.protobuf")
if LooseVersion(_protobuf.__version__) < LooseVersion("6.33.5"):
    pytest.skip("pyspark.sql.connect.proto needs protobuf >= 6.33.5", allow_module_level=True)

from pyspark.sql import SparkSession  # noqa: E402


@pytest.fixture
def session():
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live-server tests")
    session = SparkSession.builder.remote(remote).getOrCreate()
    try:
        yield session
    finally:
        session.stop()


def _labeled(session):
    from pyspark.ml.linalg import Vectors

    rows = [(i, float(i >= 4), Vectors.dense([float(i), 1.0])) for i in range(8)]
    return session.createDataFrame(rows, ["id", "label", "features"])


def test_ml_connect_random_forest_null_bootstrap_fits_as_pyspark(session):
    # As PySpark, a None param is sent as a null: the server reads a null bootstrap as false, so
    # the forest of bootstrap=None is the forest of bootstrap=False, and a null seed fails.
    from pyspark.ml.connect.classification import RandomForestClassifier

    df = _labeled(session)

    def probabilities(estimator):
        return [
            row.probability.toArray().tolist()
            for row in estimator.fit(df).transform(df).orderBy("id").collect()
        ]

    assert probabilities(RandomForestClassifier(bootstrap=None, numTrees=5, seed=42)) == (
        probabilities(RandomForestClassifier(bootstrap=False, numTrees=5, seed=42))
    )
    with pytest.raises(Exception, match="NullPointerException"):
        RandomForestClassifier(numTrees=5, seed=None).fit(df)


def test_ml_connect_scalers_fit_and_transform_as_pyspark(session):
    # pyspark.ml.connect's StandardScaler, MaxAbsScaler and StringIndexer: the server fits each one
    # and caches the model, and the transform of the model gives the values of PySpark 4.3.0.
    from pyspark.ml.connect.feature import MaxAbsScaler, StandardScaler, StringIndexer
    from pyspark.ml.connect.functions import array_to_vector

    rows = [(1, [2.0, 4.0], "a"), (2, [4.0, 2.0], "b"), (3, [6.0, 6.0], "a")]
    df = session.createDataFrame(rows, "x int, f array<double>, s string")
    df = df.withColumn("v", array_to_vector("f"))

    def rows(frame, column):
        return [row[column] for row in frame.orderBy("x").collect()]

    scaled = StandardScaler(inputCol="v", outputCol="o").fit(df).transform(df)
    values = [v.toArray().tolist() for v in rows(scaled, "o")]
    assert values == [[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]]

    maxabs = MaxAbsScaler(inputCol="v", outputCol="o").fit(df).transform(df)
    values = [v.toArray().tolist() for v in rows(maxabs, "o")]
    assert values[0] == pytest.approx([1 / 3, 2 / 3]) and values[2] == [1.0, 1.0]

    indexed = StringIndexer(inputCol="s", outputCol="o").fit(df).transform(df)
    assert rows(indexed, "o") == [0.0, 1.0, 0.0]


def test_ml_classifiers_fit_transform_and_summarize_as_pyspark(session):
    from pyspark.ml.classification import LogisticRegression, LogisticRegressionModel

    df = _labeled(session)
    model = LogisticRegression(maxIter=50, regParam=0.01).fit(df)
    assert isinstance(model, LogisticRegressionModel)
    assert model.numClasses == 2 and model.numFeatures == 2
    assert model.getMaxIter() == 50 and model.coefficients[0] > 0
    predictions = model.transform(df)
    assert {"rawPrediction", "probability", "prediction"} <= set(predictions.columns)
    assert [r.prediction for r in predictions.orderBy("id").select("prediction").collect()] == [
        0.0
    ] * 4 + [1.0] * 4
    assert model.summary.accuracy == 1.0
    assert model.summary.totalIterations > 0
    assert model.summary.roc.columns == ["FPR", "TPR"]
    assert model.evaluate(df).accuracy == 1.0
    assert model.predict(df.first().features) == 0.0


def test_ml_pipelines_save_and_load_as_pyspark(session, tmp_path):
    from pyspark.ml import Pipeline, PipelineModel
    from pyspark.ml.classification import DecisionTreeClassifier
    from pyspark.ml.feature import StringIndexer, VectorAssembler

    df = session.createDataFrame(
        [(i, "a" if i % 2 else "b", float(i), float(i % 2)) for i in range(10)],
        ["id", "c", "x", "label"],
    )
    pipeline = Pipeline(
        stages=[
            StringIndexer(inputCol="c", outputCol="ci"),
            VectorAssembler(inputCols=["ci", "x"], outputCol="features"),
            DecisionTreeClassifier(maxDepth=3, seed=1),
        ]
    )
    model = pipeline.fit(df)
    expected = [r.prediction for r in model.transform(df).orderBy("id").collect()]
    assert expected == [float(i % 2) for i in range(10)]

    path = str(tmp_path / "pipeline")
    model.write().overwrite().save(path)
    loaded = PipelineModel.load(path)
    assert [type(stage).__name__ for stage in loaded.stages] == [
        "StringIndexerModel",
        "VectorAssembler",
        "DecisionTreeClassificationModel",
    ]
    assert [r.prediction for r in loaded.transform(df).orderBy("id").collect()] == expected
    # Of a tie of frequencies, the labels are in alphabetical order.
    assert loaded.stages[0].labels == ["a", "b"]

    estimator_path = str(tmp_path / "estimator")
    pipeline.write().overwrite().save(estimator_path)
    assert len(Pipeline.load(estimator_path).getStages()) == 3


def test_ml_tuning_caches_its_dataset_of_a_dfs_path_as_pyspark(session, tmp_path, monkeypatch):
    # As official PySpark 4.3.0: of SPARKML_TEMP_DFS_PATH, CrossValidator caches its dataset in
    # a directory there, of the session of the dataset, DataFrame._session. And the resources of
    # the cluster are those of the client of the session, SparkSession._client.
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

    monkeypatch.setenv("SPARKML_TEMP_DFS_PATH", str(tmp_path))
    df = _labeled(session)
    lr = LogisticRegression(maxIter=5)
    cv = CrossValidator(
        estimator=lr,
        estimatorParamMaps=ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01]).build(),
        evaluator=BinaryClassificationEvaluator(),
        numFolds=2,
        seed=1,
    )
    assert len(cv.fit(df).avgMetrics) == 2
    assert isinstance(session._client._resources(), dict)


def test_session_clients_release_and_close_as_pyspark(session):
    # As PySpark Connect's SparkConnectClient.release_session and close: the session is released
    # on the server, and the client is closed, of a session that is then stopped.
    other = SparkSession.builder.remote(os.environ["SPARK_REMOTE"]).create()
    client = other.client
    assert not client.is_closed
    assert other.range(1).count() == 1
    client.release_session()
    client.close()
    assert client.is_closed and other.is_stopped


def test_models_whose_summary_keeps_their_transform_are_released_as_pyspark(session):
    # As PySpark: the summary of a KMeans, BisectingKMeans or GaussianMixture model keeps
    # model.transform(dataset), whose plan keeps the model. Python's garbage collector frees that
    # cycle, and the model is released on the server, once nothing else uses the model; a
    # DataFrame that still uses it keeps it.
    import gc
    import time

    from pyspark.ml.clustering import BisectingKMeans, GaussianMixture, KMeans
    from pyspark.ml.linalg import Vectors

    def cached(model):
        return model._java_obj.ref_id in "\n".join(session.client._get_ml_cache_info())

    def released(ref_id):
        for _ in range(50):
            gc.collect()
            if ref_id not in "\n".join(session.client._get_ml_cache_info()):
                return True
            time.sleep(0.1)
        return False

    data = session.createDataFrame(
        [(Vectors.dense(x, y),) for x, y in [(0, 0), (0.1, 0.2), (9, 9), (9.1, 9.2)]],
        ["features"])
    for estimator in [KMeans(k=2, seed=1), BisectingKMeans(k=2, seed=1),
                      GaussianMixture(k=2, seed=1, maxIter=3)]:
        model = estimator.fit(data)
        ref_id = model._java_obj.ref_id
        assert cached(model) and model.summary.k == 2
        del model
        assert released(ref_id), type(estimator).__name__

    model = KMeans(k=2, seed=1).fit(data)
    ref_id = model._java_obj.ref_id
    transformed = model.transform(data)
    derived = model.summary.predictions.select("prediction")
    del model
    gc.collect()
    assert ref_id in "\n".join(session.client._get_ml_cache_info())
    assert len(transformed.collect()) == 4 and len(derived.collect()) == 4
    del transformed
    gc.collect()
    assert len(derived.collect()) == 4
    del derived
    assert released(ref_id)


def _native_estimators():
    # Every native pyspark.ml.connect estimator, with the class of the model that the server
    # caches for its fit, and data that it fits.
    from pyspark.ml.connect import (
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        GBTClassifier,
        KMeans,
        LinearRegression,
        LinearSVC,
        LogisticRegression,
        MaxAbsScaler,
        RandomForestClassifier,
        RandomForestRegressor,
        StandardScaler,
        StringIndexer,
    )

    ml = "org.apache.spark.ml."
    return [
        (GBTClassifier(maxIter=2), ml + "classification.GBTClassificationModel"),
        (DecisionTreeRegressor(), ml + "regression.DecisionTreeRegressionModel"),
        (DecisionTreeClassifier(), ml + "classification.DecisionTreeClassificationModel"),
        (KMeans(k=2, seed=1), ml + "clustering.KMeansModel"),
        (LinearSVC(maxIter=2), ml + "classification.LinearSVCModel"),
        (LinearRegression(maxIter=2), ml + "regression.LinearRegressionModel"),
        (LogisticRegression(maxIter=2), ml + "classification.LogisticRegressionModel"),
        (RandomForestClassifier(numTrees=2), ml + "classification.RandomForestClassificationModel"),
        (RandomForestRegressor(numTrees=2), ml + "regression.RandomForestRegressionModel"),
        (StandardScaler(inputCol="features", outputCol="o"), ml + "feature.StandardScalerModel"),
        (MaxAbsScaler(inputCol="features", outputCol="o"), ml + "feature.MaxAbsScalerModel"),
        (StringIndexer(inputCol="s", outputCol="o"), ml + "feature.StringIndexerModel"),
    ]


def _native_training(session):
    from pyspark.ml.linalg import Vectors

    rows = [(float(i % 2), Vectors.dense([float(i), float(i % 3)]), "ab"[i % 2]) for i in range(8)]
    return session.createDataFrame(rows, ["label", "features", "s"])


def test_native_models_are_released_once_their_last_owner_is_gone_as_pyspark(session):
    # As pyspark.ml's RemoteModelRef: the server caches the model of a fit of every native
    # pyspark.ml.connect estimator while the model, or a DataFrame that it transformed, is
    # alive, and deletes it once the last of them is gone.
    import time

    def cached(model_class):
        infos = session.client._get_ml_cache_info()
        return sum(f'"class":"{model_class}"' in info for info in infos)

    def wait_for(model_class, expected):
        # The release is sent without waiting for the server, so poll for it to land.
        deadline = time.time() + 30
        while cached(model_class) != expected and time.time() < deadline:
            gc.collect()
            time.sleep(0.05)
        return cached(model_class)

    data = _native_training(session)
    for estimator, model_class in _native_estimators():
        assert cached(model_class) == 0, model_class
        model = estimator.fit(data)
        assert cached(model_class) == 1, model_class
        del model
        assert wait_for(model_class, 0) == 0, model_class

        model = estimator.fit(data)
        lazy = model.transform(data)
        del model
        gc.collect()
        assert cached(model_class) == 1, model_class
        assert len(lazy.collect()) == 8
        del lazy
        assert wait_for(model_class, 0) == 0, model_class


def test_native_models_transform_a_dataframe_of_another_session_as_pyspark(session):
    # As PySpark: the transform of a DataFrame of another session is a DataFrame, whose
    # analysis and execution fail on the server of that session, which caches no such model,
    # with CONNECT_ML.CACHE_INVALID.
    from pyspark.errors.exceptions.connect import SparkException

    data = _native_training(session)
    other = session.newSession()
    try:
        held_out = _native_training(other)
        for estimator, model_class in _native_estimators():
            transformed = estimator.fit(data).transform(held_out)
            for run in (lambda: transformed.columns, transformed.collect):
                with pytest.raises(SparkException) as error:
                    run()
                assert error.value.getCondition() == "CONNECT_ML.CACHE_INVALID", model_class
    finally:
        other.stop()


def test_the_transform_of_a_deep_dataframe_of_another_session_counts_its_levels_as_pyspark(
    session,
):
    # As PySpark, which serializes the plan of the transform with the plan of its input in one
    # recursion: the transform of a DataFrame of another session counts the levels of that
    # DataFrame, so that levels over it past the bound raise RecursionError before any request,
    # where the input counted as one level.
    from pyspark.ml.connect import KMeans

    model = KMeans(k=2, seed=1).fit(_native_training(session))
    other = session.newSession()
    try:
        deep = _native_training(other)
        for _ in range(700):
            deep = deep.select("*")
        transformed = model.transform(deep)
        for _ in range(100):
            transformed = transformed.select("*")
        with pytest.raises(RecursionError):
            transformed.columns
    finally:
        other.stop()


def test_ml_feature_transformers_as_pyspark(session):
    from pyspark.ml.feature import IDF, HashingTF, MinMaxScaler, Tokenizer

    text = session.createDataFrame([(0, "a b c"), (1, "a d")], ["id", "text"])
    words = Tokenizer(inputCol="text", outputCol="words").transform(text)
    assert [r.words for r in words.orderBy("id").collect()] == [["a", "b", "c"], ["a", "d"]]
    tf = HashingTF(inputCol="words", outputCol="tf", numFeatures=16).transform(words)
    idf = IDF(inputCol="tf", outputCol="idf").fit(tf)
    assert len(idf.idf) == 16 and idf.numDocs == 2
    assert idf.transform(tf).select("idf").first().idf.size == 16

    scaler = MinMaxScaler(inputCol="features", outputCol="scaled").fit(_labeled(session))
    assert list(scaler.originalMin) == [0.0, 1.0] and list(scaler.originalMax) == [7.0, 1.0]


def test_ml_connect_vector_assembler_and_evaluator_as_pyspark(session):
    # pyspark.ml.connect's VectorAssembler sends inputCols as a list of strings, and the native
    # BinaryClassificationEvaluator scores its probabilityCol (default probability), which the
    # JVM evaluator reads as its rawPredictionCol, with the values of PySpark 4.3.0.
    from pyspark.ml.connect.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.connect.feature import VectorAssembler
    from pyspark.ml.functions import array_to_vector
    from pyspark.sql import functions as F

    df = session.createDataFrame([(1.0, 2.0, 3), (4.0, 5.0, 6)], "x double, y double, z int")
    assembled = VectorAssembler(inputCols=["x", "y"], outputCol="v").transform(df)
    assert [r.v.toArray().tolist() for r in assembled.orderBy("x").collect()] == [
        [1.0, 2.0],
        [4.0, 5.0],
    ]

    scores = [(0.0, 0.1), (0.0, 0.4), (1.0, 0.35), (1.0, 0.8)]
    raw = session.createDataFrame(scores, "label double, score double").select(
        "label",
        array_to_vector(F.array(1.0 - F.col("score"), F.col("score"))).alias("probability"),
    )
    assert BinaryClassificationEvaluator().evaluate(raw) == 0.75
    evaluator = BinaryClassificationEvaluator(probabilityCol="probability", labelCol="label")
    assert evaluator.evaluate(raw) == 0.75
    pr = BinaryClassificationEvaluator(metricName="areaUnderPR", probabilityCol="probability")
    assert pr.evaluate(raw) == pytest.approx(0.7916666666666666)


def test_ml_connect_binary_evaluator_scores_the_probability_of_pyspark_docstring(session):
    # The example of PySpark's BinaryClassificationEvaluator: probabilities that are not vectors,
    # and the default probabilityCol, which the evaluator reads with no other option. The metric
    # is that of the server's evaluator: its areaUnderPR is the trapezoids of Spark's thresholds,
    # 0.9027777777777777, where PySpark's torcheval metric of the docstring gives 0.9166666865.
    from pyspark.ml.connect.evaluation import BinaryClassificationEvaluator

    dataset = session.createDataFrame(
        [(1, 0.6), (0, 0.55), (0, 0.1), (1, 0.6), (1, 0.4)],
        schema=["label", "probability"],
    )
    pr = BinaryClassificationEvaluator(metricName="areaUnderPR")
    assert pr.evaluate(dataset) == pytest.approx(0.9027777777777777, abs=1e-9)
    assert BinaryClassificationEvaluator().evaluate(dataset) == pytest.approx(5 / 6, abs=1e-9)


def test_ml_connect_setters_change_the_next_evaluate_and_fit_as_pyspark(session):
    # As PySpark's Params._set: a setter of pyspark.ml.connect sets the param of the evaluator or
    # estimator itself and returns it, so the next evaluate or fit uses the value of a setter whose
    # result is dropped, with the values of PySpark 4.3.0.
    from pyspark.ml.connect.classification import LogisticRegression
    from pyspark.ml.connect.evaluation import (
        BinaryClassificationEvaluator,
        MulticlassClassificationEvaluator,
        RegressionEvaluator,
    )
    from pyspark.ml.connect.pipeline import Pipeline
    from pyspark.ml.linalg import Vectors

    scored = session.createDataFrame(
        [
            (Vectors.dense([0.9, 0.1]), 0.0, 0.0),
            (Vectors.dense([0.6, 0.4]), 0.0, 1.0),
            (Vectors.dense([0.65, 0.35]), 1.0, 1.0),
            (Vectors.dense([0.2, 0.8]), 1.0, 1.0),
        ],
        ["score", "label", "pred"],
    )
    regression = RegressionEvaluator(labelCol="label", predictionCol="pred")
    assert regression.setMetricName("r2") is regression
    assert regression.evaluate(scored) == pytest.approx(0.0, abs=1e-9)  # The rmse is 0.5.
    multiclass = MulticlassClassificationEvaluator(predictionCol="pred")
    assert multiclass.setMetricName("accuracy") is multiclass
    assert multiclass.evaluate(scored) == 0.75  # The f1 is 0.733.
    binary = BinaryClassificationEvaluator()
    assert binary.setProbabilityCol("score") is binary  # No probability column is there.
    assert binary.evaluate(scored) == 0.75
    assert binary.setMetricName("areaUnderPR") is binary
    assert binary.evaluate(scored) == pytest.approx(0.7916666666666666)

    features = [[0.0, 1.0], [1.0, 0.0], [2.0, 1.5], [0.5, 0.2]]
    labels = [0.0, 1.0, 1.0, 0.0]
    labeled = session.createDataFrame(
        [(i, y, Vectors.dense(f)) for i, (y, f) in enumerate(zip(labels, features))],
        ["id", "y", "features"],
    )
    estimator = LogisticRegression()
    assert estimator.setLabelCol("y") is estimator  # No label column is there.
    predictions = estimator.fit(labeled).transform(labeled).orderBy("id").collect()
    assert [r.prediction for r in predictions] == labels

    # The native Pipeline of stage names is not implemented: its fit raises NOT_IMPLEMENTED, as
    # the server cannot fit the stages, and the setter still sets them.
    from pyspark.errors import PySparkNotImplementedError

    pipeline = Pipeline()
    assert pipeline.setStages(["a", "b"]) is pipeline
    with pytest.raises(PySparkNotImplementedError) as error:
        pipeline.fit(labeled)
    assert error.value.getCondition() == "NOT_IMPLEMENTED"
    assert error.value.getMessageParameters() == {"feature": "Pipeline.fit"}


def test_ml_statistics_as_pyspark(session):
    from pyspark.ml.stat import ChiSquareTest, Correlation, Summarizer

    df = _labeled(session)
    summary = df.select(Summarizer.mean(df.features), Summarizer.count(df.features))
    assert summary.columns == ["mean(features)", "count(features)"]
    row = summary.first()
    assert list(row["mean(features)"]) == [3.5, 1.0] and row["count(features)"] == 8
    metrics = df.select(Summarizer.metrics("min", "max").summary(df.features).alias("s")).first()
    assert list(metrics.s.min) == [0.0, 1.0] and list(metrics.s.max) == [7.0, 1.0]
    assert ChiSquareTest.test(df, "features", "label").columns == [
        "pValues",
        "degreesOfFreedom",
        "statistics",
    ]
    assert Correlation.corr(df, "features").columns == ["pearson(features)"]


def test_ml_evaluators_and_tuning_as_pyspark(session):
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

    df = _labeled(session)
    lr = LogisticRegression(maxIter=20)
    evaluator = BinaryClassificationEvaluator()
    assert evaluator.evaluate(lr.fit(df).transform(df)) == 1.0
    grid = ParamGridBuilder().addGrid(lr.regParam, [0.1, 0.01]).build()
    cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator, numFolds=2, seed=1)
    cv_model = cv.fit(df)
    assert len(cv_model.avgMetrics) == 2
    assert cv_model.bestModel.transform(df).count() == 8


def test_connect_plans_run_on_the_server_as_pyspark(session):
    # As official PySpark 4.3.0: DataFrames of the Read, Range and SQL plans of
    # pyspark.sql.connect.plan, as pyspark.testing.connectutils builds them.
    from pyspark.sql import functions as F
    from pyspark.sql.connect.dataframe import DataFrame
    from pyspark.sql.connect.plan import SQL, Range, Read, SubqueryAlias

    assert [r.id for r in DataFrame(Range(0, 6, 2, 2), session).collect()] == [0, 2, 4]
    assert DataFrame(SQL("SELECT ? AS a", args=[F.lit(5)]), session).first().a == 5
    assert DataFrame(SQL("SELECT :b AS b", named_args={"b": F.lit("x")}), session).first().b == "x"
    views = [SubqueryAlias(Range(0, 3, 1), "v")]
    assert DataFrame(SQL("SELECT sum(id) AS s FROM v", views=views), session).first().s == 3
    session.range(4).createOrReplaceTempView("ml_e2e_plans")
    try:
        assert DataFrame(Read("ml_e2e_plans"), session).count() == 4
    finally:
        session.catalog.dropTempView("ml_e2e_plans")


def test_ml_transformed_dataframes_keep_their_models_as_pyspark(session):
    # As official PySpark 4.3.0: a DataFrame of a transform keeps the model alive, and the
    # model is deleted from the cache of the server once nothing keeps it.
    from pyspark.ml.feature import StringIndexer

    df = session.createDataFrame([(0, "x"), (1, "y"), (2, "x")], ["id", "c"])

    def fit_transform(df):
        model = StringIndexer(inputCol="c", outputCol="ci").fit(df)
        return model._java_obj.ref_id, model.transform(df)

    ref_id, out = fit_transform(df)
    gc.collect()
    assert sorted(r.ci for r in out.filter("c = 'x'").collect()) == [0.0, 0.0]
    del out
    gc.collect()
    # Deleted already: the server deletes no IDs of it.
    assert session.client._delete_ml_cache([ref_id]) == [""]

    model = StringIndexer(inputCol="c", outputCol="ci").fit(df)
    assert session.client._delete_ml_cache([model._java_obj.ref_id]) == [model._java_obj.ref_id]
    with pytest.raises(Exception) as error:
        model.transform(df).count()
    assert error.value.getCondition() == "CONNECT_ML.CACHE_INVALID"


def test_ml_deep_dataframes_fit_and_transform_as_pyspark(session):
    # As PySpark: a DataFrame of 120 chained withColumn calls, nested deeper than a protobuf
    # decoder parses, is fitted and transformed, and so is OneVsRest over 12 classes, whose
    # transform adds a layer per class. Their predictions are those of the shallow DataFrames.
    # The chains are over a repartition of the local data: over a local relation, the optimizer
    # folds one projection per iteration, which a server in testing mode (SPARK_TESTING, as in
    # CI) fails past its 100 iterations, for PySpark too.
    from pyspark.ml.classification import LogisticRegression, OneVsRest
    from pyspark.ml.linalg import Vectors
    from pyspark.sql import functions as F

    rows = [(float(i % 2), Vectors.dense([float(i % 7), float(i % 5)])) for i in range(40)]
    base = session.createDataFrame(rows, ["label", "features"]).repartition(1)
    deep = base
    for i in range(120):
        deep = deep.withColumn(f"c{i}", F.lit(i))
    model = LogisticRegression(maxIter=2).fit(deep)
    assert model.coefficients == LogisticRegression(maxIter=2).fit(base).coefficients
    predictions = sorted(r.prediction for r in model.transform(deep).collect())
    assert len(predictions) == 40
    assert predictions == sorted(r.prediction for r in model.transform(base).collect())

    rows12 = [(float(i % 12), Vectors.dense([float(i % 7), float(i % 5)])) for i in range(60)]
    base12 = session.createDataFrame(rows12, ["label", "features"]).repartition(1)
    deep12 = base12
    for i in range(120):
        deep12 = deep12.withColumn(f"c{i}", F.lit(i))
    ovr = OneVsRest(classifier=LogisticRegression(maxIter=2)).fit(base12)
    shallow = sorted(r.prediction for r in ovr.transform(base12).collect())
    assert sorted(r.prediction for r in ovr.transform(deep12).collect()) == shallow
    assert len(shallow) == 60


def test_ml_dataframes_of_plans_as_pyspark(session):
    # As PySpark Connect's DataFrame(plan, session): a DataFrame of the relation of a plan,
    # which further transformations take as input.
    from pyspark.sql.connect.dataframe import DataFrame
    from pyspark.sql.connect.plan import LogicalPlan

    class SQLPlan(LogicalPlan):
        def __init__(self, query):
            super().__init__(None)
            self._query = query

        def plan(self, session):
            plan = self._create_proto_relation()
            plan.sql.query = self._query
            return plan

    df = DataFrame(SQLPlan("SELECT id AS a FROM range(3)"), session)
    assert df.columns == ["a"]
    assert [r.a for r in df.filter("a > 0").orderBy("a").collect()] == [1, 2]
    other = session.range(2)
    assert df.join(other, df["a"] == other["id"]).count() == 2


def test_session_hooks_rewrite_the_requests_sent_to_the_server_as_pyspark(session):
    # As official PySpark 4.3.0's SparkSession.Hook: the server runs the request that the hooks
    # of the session return, as a message of pyspark.sql.connect.proto.
    import pyspark.sql.connect.proto as pb2
    from pyspark.sql.connect.session import SparkSession as ConnectSparkSession

    # The placeholder of a plan is valid while its DataFrame lives, so the DataFrame is kept.
    source = session.range(5)
    replacement = source._plan.plan(session.client)
    calls = []

    class Replace(ConnectSparkSession.Hook):
        def on_execute_plan(self, request):
            assert isinstance(request, pb2.ExecutePlanRequest)
            calls.append(request.operation_id)
            if request.plan.HasField("root"):
                request.plan.root.CopyFrom(replacement)
            return request

    hooked = (
        ConnectSparkSession.builder.remote(os.environ["SPARK_REMOTE"])
        ._registerHook(lambda s: Replace())
        .create()
    )
    try:
        assert [r.id for r in hooked.range(1).collect()] == [0, 1, 2, 3, 4]
        assert len(calls) == 1 and calls[0]
    finally:
        hooked.stop()
