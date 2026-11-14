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
from _native_classes import native_class
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
    from pyspark.ml.connect.feature import MaxAbsScaler, StandardScaler, StringIndexer
    from pyspark.ml.functions import array_to_vector
    from pyspark.ml.linalg import Vector

    data = [(1, [2.0, 4.0], "a"), (2, [4.0, 2.0], "b"), (3, [6.0, 6.0], "a")]
    df = session.createDataFrame(data, "x int, f array<double>, s string").withColumn("f", array_to_vector("f"))

    def rows(frame, column):
        return [list(row[column]) if isinstance(row[column], Vector) else row[column] for row in frame.orderBy("x").collect()]

    scaled = StandardScaler(inputCol="f", outputCol="o", withMean=True).fit(df).transform(df)
    assert scaled.schema["o"].dataType.simpleString() == "vector"
    for got, want in zip(rows(scaled, "o"), [[-1., 0.], [0., -1.], [1., 1.]]):
        assert got == pytest.approx(want, rel=1e-12, abs=1e-12)

    maxabs = MaxAbsScaler(inputCol="f", outputCol="o").fit(df).transform(df)
    assert maxabs.schema["o"].dataType.simpleString() == "vector"
    values = rows(maxabs, "o")
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


def test_ml_connect_fit_of_two_param_maps_returns_two_models_as_pyspark(session):
    # As pyspark.ml's Estimator.fit: a list of two param maps fits the estimator once per map, and
    # returns the two models in a list, each of which transforms the dataset.
    LogisticRegression = native_class(
        "pyspark.ml.connect", "LogisticRegression", "pyspark.ml.classification"
    )

    df = _labeled(session)
    models = LogisticRegression(maxIter=5).fit(df, [{}, {}])
    assert isinstance(models, list) and len(models) == 2
    for model in models:
        assert model.transform(df).count() == 8


def _native_estimators():
    # Every native pyspark.ml.connect estimator, with the class of the model that the server
    # caches for its fit, and data that it fits.
    from pyspark.ml.connect import (
        AFTSurvivalRegression,
        BisectingKMeans,
        DecisionTreeClassifier,
        DecisionTreeRegressor,
        FMClassifier,
        FMRegressor,
        GaussianMixture,
        GBTClassifier,
        GBTRegressor,
        GeneralizedLinearRegression,
        IsotonicRegression,
        KMeans,
        LinearRegression,
        LinearSVC,
        LogisticRegression,
        MultilayerPerceptronClassifier,
        NaiveBayes,
        RandomForestClassifier,
        RandomForestRegressor,
        StringIndexer,
    )

    ml = "org.apache.spark.ml."
    return [
        (AFTSurvivalRegression(labelCol="duration", maxIter=2), ml + "regression.AFTSurvivalRegressionModel"),
        (IsotonicRegression(), ml + "regression.IsotonicRegressionModel"),
        (FMRegressor(maxIter=2, seed=42), ml + "regression.FMRegressionModel"),
        (GaussianMixture(k=2, seed=1, maxIter=2), ml + "clustering.GaussianMixtureModel"),
        (BisectingKMeans(k=2, seed=1), ml + "clustering.BisectingKMeansModel"),
        (NaiveBayes(), ml + "classification.NaiveBayesModel"),
        (GBTRegressor(maxIter=2), ml + "regression.GBTRegressionModel"),
        (GBTClassifier(maxIter=2), ml + "classification.GBTClassificationModel"),
        (DecisionTreeRegressor(), ml + "regression.DecisionTreeRegressionModel"),
        (DecisionTreeClassifier(), ml + "classification.DecisionTreeClassificationModel"),
        (KMeans(k=2, seed=1), ml + "clustering.KMeansModel"),
        (FMClassifier(maxIter=2, seed=42), ml + "classification.FMClassificationModel"),
        (MultilayerPerceptronClassifier(layers=[2,2],maxIter=2,seed=42), ml + "classification.MultilayerPerceptronClassificationModel"),
        (LinearSVC(maxIter=2), ml + "classification.LinearSVCModel"),
        (GeneralizedLinearRegression(maxIter=2), ml + "regression.GeneralizedLinearRegressionModel"),
        (LinearRegression(maxIter=2), ml + "regression.LinearRegressionModel"),
        (LogisticRegression(maxIter=2), ml + "classification.LogisticRegressionModel"),
        (RandomForestClassifier(numTrees=2), ml + "classification.RandomForestClassificationModel"),
        (RandomForestRegressor(numTrees=2), ml + "regression.RandomForestRegressionModel"),
        (StringIndexer(inputCol="s", outputCol="o"), ml + "feature.StringIndexerModel"),
    ]


def _native_training(session):
    from pyspark.ml.linalg import Vectors

    rows = [(float(i % 2), Vectors.dense([float(i), float(i % 3)]), "ab"[i % 2], float(i + 1), 1.0) for i in range(8)]
    return session.createDataFrame(rows, ["label", "features", "s", "duration", "censor"])


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
    # Evaluate the vector column with standard Spark ML binary metrics.
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
        array_to_vector(F.array(1.0 - F.col("score"), F.col("score"))).alias("rawPrediction"),
    )
    assert BinaryClassificationEvaluator().evaluate(raw) == 0.75
    evaluator = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label")
    assert evaluator.evaluate(raw) == 0.75
    pr = BinaryClassificationEvaluator(metricName="areaUnderPR", rawPredictionCol="rawPrediction")
    assert pr.evaluate(raw) == pytest.approx(0.7916666666666666)


def test_ml_connect_binary_evaluator_scores_the_probability_of_pyspark_docstring(session):
    # Standard Spark ML accepts double scores in rawPredictionCol.
    from pyspark.ml.connect.evaluation import BinaryClassificationEvaluator

    dataset = session.createDataFrame(
        [(1, 0.6), (0, 0.55), (0, 0.1), (1, 0.6), (1, 0.4)],
        schema=["label", "rawPrediction"],
    )
    pr = BinaryClassificationEvaluator(metricName="areaUnderPR")
    assert pr.evaluate(dataset) == pytest.approx(0.9027777777777777, abs=1e-9)
    assert BinaryClassificationEvaluator().evaluate(dataset) == pytest.approx(5 / 6, abs=1e-9)


def test_ml_connect_evaluators_take_spark_columns_and_reject_pandas(session):
    # Prepare the double predictions and vector scores required by standard Spark ML.
    from pyspark.ml.connect.evaluation import (
        BinaryClassificationEvaluator,
        MulticlassClassificationEvaluator,
        RegressionEvaluator,
    )
    from pyspark.ml.functions import array_to_vector
    from pyspark.sql import functions as F

    multiclass = session.createDataFrame(
        [(1, 1), (1, 1), (2, 3), (0, 0), (0, 1), (3, 1), (3, 3), (2, 2), (1, 0), (2, 2)],
        schema=["label", "prediction"],
    )
    multiclass = multiclass.withColumn("prediction", F.col("prediction").cast("double"))
    accuracy = MulticlassClassificationEvaluator(metricName="accuracy")
    assert accuracy.evaluate(multiclass) == pytest.approx(0.6)
    with pytest.raises(TypeError, match="requires a Spark DataFrame"):
        accuracy.evaluate(dataset=multiclass.toPandas())
    docstring = session.createDataFrame(
        [(1, 1), (0, 0), (2, 2), (1, 0), (2, 1)], schema=["label", "prediction"])
    docstring = docstring.withColumn("prediction", F.col("prediction").cast("double"))
    assert MulticlassClassificationEvaluator(metricName="accuracy").evaluate(docstring) == pytest.approx(0.6)

    regression = session.createDataFrame(
        [(1.0, 2), (-1.0, -1), (2.0, 3)], schema=["label", "prediction"])
    regression = regression.withColumn("prediction", F.col("prediction").cast("double"))
    mse = RegressionEvaluator(metricName="mse")
    assert mse.evaluate(regression) == pytest.approx(2 / 3)
    with pytest.raises(TypeError, match="requires a Spark DataFrame"):
        mse.evaluate(regression.toPandas())

    binary = session.createDataFrame(
        [
            (1, 0.2, [0.8, 0.2]),
            (0, 0.6, [0.4, 0.6]),
            (1, 0.8, [0.2, 0.8]),
            (1, 0.7, [0.3, 0.7]),
            (0, 0.4, [0.6, 0.4]),
            (0, 0.3, [0.7, 0.3]),
        ],
        schema=["label", "prob", "prob2"],
    ).withColumn("prob2", array_to_vector("prob2"))
    for metric in ("areaUnderROC", "areaUnderPR"):
        expected = BinaryClassificationEvaluator(
            metricName=metric, rawPredictionCol="prob").evaluate(binary)
        for column in ("prob2", "prob"):
            evaluator = BinaryClassificationEvaluator(metricName=metric, rawPredictionCol=column)
            assert evaluator.evaluate(binary) == pytest.approx(expected), (metric, column)
            with pytest.raises(TypeError, match="requires a Spark DataFrame"):
                evaluator.evaluate(binary.toPandas())
    auroc = BinaryClassificationEvaluator(rawPredictionCol="prob2").evaluate(binary)
    assert auroc == pytest.approx(2 / 3)
    # Another object raises the AttributeError of its sparkSession, which pyspark.ml's
    # try_remote_evaluate reads first, as PySpark 4.3.0 does.
    with pytest.raises(AttributeError, match="'list' object has no attribute 'sparkSession'"):
        accuracy.evaluate([(1, 1)])


def test_native_ml_accepts_explicitly_converted_pandas(session):
    import pandas as pd
    import pyspark.ml.connect as connect
    from pyspark.ml.functions import array_to_vector
    from pyspark.sql import DataFrame

    # Exercise the native bindings, independently of the deprecated Python ML implementations.
    assert connect.KMeans is __import__("pyspark._pyspark", fromlist=["KMeans"]).KMeans
    pdf = pd.DataFrame({"features": [[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0]],
                        "label": [0.0, 1.0, 1.0, 0.0]})
    # The estimators take the vectors of pyspark.ml, so the arrays of the converted frame are
    # converted to vectors explicitly too.
    df = session.createDataFrame(pdf).withColumn("features", array_to_vector("features"))
    for estimator in [connect.LogisticRegression(maxIter=5), connect.KMeans(k=2, seed=1)]:
        with pytest.raises(TypeError, match="requires a Spark DataFrame"):
            estimator.fit(pdf)
        model = estimator.fit(df)
        with pytest.raises(TypeError, match="requires a Spark DataFrame"):
            model.transform(pdf)
        transformed = model.transform(df)
        assert isinstance(transformed, DataFrame)
        assert transformed.count() == len(pdf)
        assert set(transformed.toPandas()["prediction"]) <= {0, 1}

    assembler = connect.VectorAssembler(inputCols=["label"], outputCol="assembled")
    with pytest.raises(TypeError, match="requires a Spark DataFrame"):
        assembler.transform(pdf)
    assert [row.assembled.toArray().tolist() for row in assembler.transform(df).collect()] == [
        [0.0], [1.0], [1.0], [0.0]
    ]
    scored = pd.DataFrame({"label": [1.0, 2.0, 3.0], "prediction": [1.0, 2.5, 3.0]})
    evaluator = connect.RegressionEvaluator()
    with pytest.raises(TypeError, match="requires a Spark DataFrame"):
        evaluator.evaluate(scored)
    assert evaluator.evaluate(session.createDataFrame(scored)) == pytest.approx(0.28867513459481287)


def _vector_rows(session, labels, label_type=None):
    # The features of the example of the deprecated LogisticRegression of pyspark.ml.connect, as
    # the vectors that pyspark.ml takes, with the labels `labels`, of the type `label_type` unless
    # they are doubles.
    from pyspark.ml.linalg import Vectors, VectorUDT
    from pyspark.sql import types as T

    features = [[1.0, 2.0], [2.0, -1.0], [-3.0, -2.0], [-1.0, -2.0]]
    schema = T.StructType([
        T.StructField("features", VectorUDT()),
        T.StructField("label", label_type or T.DoubleType()),
    ])
    rows = [(Vectors.dense(f), label) for f, label in zip(features, labels)]
    return session.createDataFrame(rows, schema)


def test_ml_connect_column_params_are_read_by_their_exact_names_as_pyspark(session):
    # As pyspark.ml, which sends the column params as they are given: the JVM reads a column of a
    # fit by its exact name, so that a featuresCol or a labelCol of another case than the schema
    # raises its FIELD_NOT_FOUND, as does the transform of a KMeans model, while the transform of
    # a LogisticRegression model resolves its featuresCol, of the values of PySpark 4.3.0. The
    # test passes with the reference client too.
    from pyspark.errors import IllegalArgumentException

    KMeans = native_class("pyspark.ml.connect.clustering", "KMeans", "pyspark.ml.clustering")
    LogisticRegression = native_class(
        "pyspark.ml.connect.classification", "LogisticRegression", "pyspark.ml.classification"
    )

    def field_not_found(run, field, fields):
        with pytest.raises(IllegalArgumentException) as error:
            run()
        assert error.value.getCondition() == "FIELD_NOT_FOUND"
        assert str(error.value).splitlines()[0] == (
            f"[FIELD_NOT_FOUND] No such struct field `{field}` in {fields}. SQLSTATE: 42704"
        )

    train = _vector_rows(session, [1.0, 1.0, 0.0, 0.0])
    upper = train.withColumnRenamed("features", "Features")
    field_not_found(
        lambda: KMeans(k=2, seed=1, featuresCol="FEATURES").fit(train),
        "FEATURES", "`features`, `label`",
    )
    clusters = KMeans(k=2, seed=1).fit(train)
    field_not_found(lambda: clusters.transform(upper).columns, "features", "`Features`, `label`")
    field_not_found(
        lambda: LogisticRegression(maxIter=5, featuresCol="Features").fit(train),
        "Features", "`features`, `label`",
    )
    field_not_found(
        lambda: LogisticRegression(maxIter=5, labelCol="LABEL").fit(train),
        "LABEL", "`features`, `label`",
    )
    field_not_found(
        lambda: LogisticRegression(maxIter=5).fit(upper), "features", "`Features`, `label`"
    )
    model = LogisticRegression(maxIter=5).fit(train)
    transformed = model.transform(upper)
    assert transformed.columns == ["Features", "label", "rawPrediction", "probability", "prediction"]
    assert [row.prediction for row in transformed.collect()] == [1.0, 1.0, 0.0, 0.0]


def test_ml_connect_evaluators_read_columns_by_their_exact_names_as_pyspark(session):
    # As pyspark.ml's evaluators, which send their column params as they are given: a native
    # evaluate of the predictionCol PREDICTION, which the schema spells prediction, raises the
    # JVM's FIELD_NOT_FOUND of PREDICTION, and one of the exact name evaluates. The estimators and
    # models send their column params as they are given too: see
    # test_ml_connect_column_params_are_read_by_their_exact_names_as_pyspark.
    from pyspark.ml.connect.evaluation import RegressionEvaluator

    scored = session.sql(
        "select label AS LABEL, cast(label AS double) AS prediction "
        "from values (1.0D), (2.0D) as t(label)"
    )
    with pytest.raises(Exception, match="PREDICTION"):
        RegressionEvaluator(labelCol="LABEL", predictionCol="PREDICTION").evaluate(scored)
    metric = RegressionEvaluator(labelCol="LABEL", predictionCol="prediction").evaluate(scored)
    assert metric == pytest.approx(0.0)


def test_ml_connect_estimators_and_scalers_take_vectors_not_arrays_as_pyspark(session):
    # As pyspark.ml's, the native estimators and scalers read the vectors of their featuresCol or
    # inputCol: the JVM raises its error of an array column, in a fit and when the transform of a
    # model fitted on vectors is analyzed, so that arrays are converted with array_to_vector. The
    # scalers output vectors, the input of the next estimator: StandardScaler, of withMean=False and
    # withStd=True by default, scales without centering. The values are those of PySpark 4.3.0,
    # and the test passes with the reference client too.
    from pyspark.errors import IllegalArgumentException
    from pyspark.ml.functions import array_to_vector

    LogisticRegression = native_class(
        "pyspark.ml.connect.classification", "LogisticRegression", "pyspark.ml.classification"
    )
    KMeans = native_class("pyspark.ml.connect.clustering", "KMeans", "pyspark.ml.clustering")
    MaxAbsScaler = native_class("pyspark.ml.connect.feature", "MaxAbsScaler", "pyspark.ml.feature")
    StandardScaler = native_class(
        "pyspark.ml.connect.feature", "StandardScaler", "pyspark.ml.feature"
    )

    features = [[1.0, 2.0], [2.0, -1.0], [-3.0, -2.0], [-1.0, -2.0]]
    arrays = session.createDataFrame(
        [(f, label) for f, label in zip(features, [1, 1, 0, 0])], schema=["features", "label"]
    )
    vectors = arrays.withColumn("features", array_to_vector("features"))
    vector = (
        "requirement failed: Column features must be of type class "
        "org.apache.spark.ml.linalg.VectorUDT:struct<type:tinyint,size:int,indices:array<int>,"
        "values:array<double>> but was actually class org.apache.spark.sql.types.ArrayType:"
        "array<double>."
    )
    # KMeans also takes an array whose elements cannot be null, which those of createDataFrame can.
    compatible = (
        "requirement failed: Column features must be of type equal to one of the following "
        "types: [struct<type:tinyint,size:int,indices:array<int>,values:array<double>>, "
        "array<double>, array<float>] but was actually of type array<double>."
    )
    max_abs = MaxAbsScaler(inputCol="features", outputCol="scaled")
    standard = StandardScaler(inputCol="features", outputCol="scaled")
    for estimator, message in [
        (LogisticRegression(maxIter=20), vector),
        (KMeans(k=2, seed=1), compatible),
        (max_abs, vector),
        (standard, vector),
    ]:
        for run in (
            lambda: estimator.fit(arrays),
            lambda: estimator.fit(vectors).transform(arrays).columns,
        ):
            with pytest.raises(IllegalArgumentException) as error:
                run()
            assert str(error.value).splitlines()[0] == message, estimator

    def scaled(scaler):
        transformed = scaler.fit(vectors).transform(vectors)
        assert transformed.schema["scaled"].dataType.simpleString() == "vector"
        return transformed, [row.scaled.toArray().tolist() for row in transformed.collect()]

    max_abs_scaled, values = scaled(max_abs)
    for got, want in zip(values, [[1 / 3, 1.0], [2 / 3, -0.5], [-1.0, -1.0], [-1 / 3, -1.0]]):
        assert got == pytest.approx(want, rel=1e-12)
    _, values = scaled(standard)
    std = [(59 / 12) ** 0.5, (43 / 12) ** 0.5]  # The sample standard deviations of the features.
    for got, row in zip(values, features):
        assert got == pytest.approx([x / s for x, s in zip(row, std)], rel=1e-12)
    assert values[0] == pytest.approx([0.4509876168016973, 1.0565410875907486], rel=1e-12)

    model = LogisticRegression(featuresCol="scaled", maxIter=20).fit(max_abs_scaled)
    predictions = [row.prediction for row in model.transform(max_abs_scaled).collect()]
    assert predictions == [1.0, 1.0, 0.0, 0.0]
    clusters = KMeans(featuresCol="scaled", k=2, seed=1).fit(max_abs_scaled)
    clustered = clusters.transform(max_abs_scaled)
    assert clustered.schema["scaled"].dataType.simpleString() == "vector"
    assert len({row.prediction for row in clustered.collect()}) == 2


def test_ml_connect_logistic_regression_labels_are_checked_by_the_jvm_as_pyspark(session):
    # As pyspark.ml, which sends the fit to the JVM, which checks the labels: the client makes none
    # of the checks of the deprecated _fit of pyspark.ml.connect, and raises none of its
    # ValueErrors. The JVM fits one class or a gap in the labels (numClasses is the maximum label
    # and 1), and raises its errors of a negative, a fractional or a null label, a string label
    # and no rows, of the values of PySpark 4.3.0. The test passes with the reference client too.
    from pyspark.errors import IllegalArgumentException, SparkRuntimeException
    from pyspark.errors.exceptions.connect import SparkException
    from pyspark.sql import types as T

    LogisticRegression = native_class(
        "pyspark.ml.connect.classification", "LogisticRegression", "pyspark.ml.classification"
    )

    def fitted(labels):
        # The predictions of the model of labels, and the sizes of its probability and raw
        # prediction vectors.
        df = _vector_rows(session, labels)
        rows = LogisticRegression(maxIter=5).fit(df).transform(df).collect()
        return [(r.prediction, len(r.probability), len(r.rawPrediction)) for r in rows]

    assert fitted([1.0, 1.0, 1.0, 1.0]) == [(1.0, 2, 2)] * 4
    assert fitted([0.0, 0.0, 0.0, 0.0]) == [(0.0, 2, 2)] * 4
    assert fitted([0.0, 0.0, 2.0, 2.0]) == [(0.0, 3, 3)] * 2 + [(2.0, 3, 3)] * 2
    assert fitted([1.0, 1.0, 2.0, 2.0]) == [(1.0, 3, 3)] * 2 + [(2.0, 3, 3)] * 2
    for labels, message in [
        ([-1.0, -1.0, 1.0, 1.0], "Labels MUST be in [0, 100), but got -1.0"),
        ([0.5, 0.5, 1.0, 1.0], "Labels MUST be Integers, but got 0.5"),
        ([0.0, None, 1.0, 1.0], "Labels MUST NOT be Null or NaN"),
    ]:
        with pytest.raises(SparkRuntimeException) as error:
            LogisticRegression(maxIter=5).fit(_vector_rows(session, labels))
        assert error.value.getCondition() == "USER_RAISED_EXCEPTION", labels
        assert str(error.value).splitlines()[0] == (
            f"[USER_RAISED_EXCEPTION] {message} SQLSTATE: P0001"
        ), labels
    strings = _vector_rows(session, ["a", "a", "b", "b"], T.StringType())
    with pytest.raises(IllegalArgumentException) as error:
        LogisticRegression(maxIter=5).fit(strings)
    assert str(error.value).splitlines()[0] == "requirement failed"
    with pytest.raises(SparkException) as error:
        LogisticRegression(maxIter=5).fit(_vector_rows(session, []))
    assert str(error.value).splitlines()[0] == "ML algorithm was given empty dataset."


def test_ml_connect_logistic_regression_outputs_the_columns_of_pyspark(session):
    # As pyspark.ml's, the native LogisticRegression outputs the columns of its JVM model, of the
    # metadata of the JVM, in this order: a rawPrediction and a probability vector, each unless its
    # column param is empty, and a double prediction that is not null. The JVM raises its error of
    # an output column that the DataFrame has, in a fit as in a transform, and replaces a column of
    # another case. The values are those of PySpark 4.3.0, and the test passes with the reference
    # client too.
    from pyspark.errors import IllegalArgumentException
    from pyspark.ml.linalg import DenseVector
    from pyspark.sql import functions as F
    from pyspark.sql import types as T

    LogisticRegression = native_class(
        "pyspark.ml.connect.classification", "LogisticRegression", "pyspark.ml.classification"
    )

    def fields(df):
        return [(f.name, f.dataType.simpleString(), f.nullable, f.metadata) for f in df.schema]

    labeled = _vector_rows(session, [1, 1, 0, 0], T.LongType())
    inputs = [("features", "vector", True, {}), ("label", "bigint", True, {})]
    vectors = {"ml_attr": {"num_attrs": 2}}
    nominal = {"ml_attr": {"num_vals": 2, "type": "nominal"}}
    for params, outputs in [
        ({}, [
            ("rawPrediction", "vector", True, vectors),
            ("probability", "vector", True, vectors),
            ("prediction", "double", False, nominal),
        ]),
        ({"predictionCol": "p", "probabilityCol": "q", "rawPredictionCol": "r"}, [
            ("r", "vector", True, vectors),
            ("q", "vector", True, vectors),
            ("p", "double", False, nominal),
        ]),
        ({"probabilityCol": ""}, [
            ("rawPrediction", "vector", True, vectors),
            ("prediction", "double", False, nominal),
        ]),
        ({"probabilityCol": "", "rawPredictionCol": ""}, [
            ("prediction", "double", False, nominal),
        ]),
    ]:
        transformed = LogisticRegression(maxIter=5, **params).fit(labeled).transform(labeled)
        assert fields(transformed) == inputs + outputs, params
        rows = transformed.collect()
        assert [row[-1] for row in rows] == [1.0, 1.0, 0.0, 0.0], params
        assert all(type(row[-1]) is float for row in rows), params

    model = LogisticRegression(maxIter=5).fit(labeled)
    row = model.transform(labeled).first()
    assert isinstance(row.rawPrediction, DenseVector) and isinstance(row.probability, DenseVector)
    assert sum(row.probability) == pytest.approx(1.0)
    for column in ("rawPrediction", "probability", "prediction"):
        extra = labeled.withColumn(column, F.lit(0))
        for run in (
            lambda: model.transform(extra).columns,
            lambda: LogisticRegression(maxIter=5).fit(extra),
        ):
            with pytest.raises(IllegalArgumentException) as error:
                run()
            message = f"requirement failed: Column {column} already exists."
            assert str(error.value).splitlines()[0] == message, column
    assert model.transform(labeled.withColumn("Prediction", F.lit(0))).columns == [
        "features", "label", "prediction", "rawPrediction", "probability"
    ]


def test_ml_connect_empty_pandas_transforms(session):
    import pandas as pd
    from pyspark.ml.connect.classification import LogisticRegression

    model = LogisticRegression(maxIter=5).fit(_labeled(session))
    with pytest.raises(TypeError, match="requires a Spark DataFrame"):
        model.transform(pd.DataFrame({"features": pd.Series([], dtype=object)}))


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
    assert binary.setRawPredictionCol("score") is binary  # Score an explicitly named vector column.
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

    # The setter of a Pipeline sets the stages, and its fit, as `Pipeline._fit`, raises PySpark's
    # TypeError of a stage that is neither an estimator nor a transformer, before any fit.
    pipeline = Pipeline()
    assert pipeline.setStages(["a", "b"]) is pipeline
    with pytest.raises(TypeError) as error:
        pipeline.fit(labeled)
    assert str(error.value) == "Cannot recognize a pipeline stage of type <class 'str'>."


def test_ml_connect_set_params_fits_the_model_of_the_constructor_as_pyspark(session):
    # As PySpark's setParams, which sets the params that it is given: an estimator whose params
    # are set by setParams fits the model of the constructor with the same params, which gives
    # the same predictions on the server, for a classifier and a clusterer.
    from pyspark.ml.connect.classification import LinearSVC
    from pyspark.ml.connect.clustering import KMeans

    labeled = _labeled(session)

    by_setter = LinearSVC().setParams(maxIter=7, regParam=0.1, threshold=0.5)
    by_constructor = LinearSVC(maxIter=7, regParam=0.1, threshold=0.5)
    setter_rows = by_setter.fit(labeled).transform(labeled).orderBy("id").collect()
    constructor_rows = by_constructor.fit(labeled).transform(labeled).orderBy("id").collect()
    assert [(r.prediction, list(r.rawPrediction)) for r in setter_rows] == [
        (r.prediction, list(r.rawPrediction)) for r in constructor_rows
    ]

    clusters = KMeans().setParams(k=2, seed=7, maxIter=5)
    by_constructor = KMeans(k=2, seed=7, maxIter=5)
    setter_clusters = clusters.fit(labeled).transform(labeled).orderBy("id").collect()
    constructor_clusters = by_constructor.fit(labeled).transform(labeled).orderBy("id").collect()
    assert [r.prediction for r in setter_clusters] == [
        r.prediction for r in constructor_clusters
    ]


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
