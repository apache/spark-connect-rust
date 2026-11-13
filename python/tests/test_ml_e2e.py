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
    with pytest.raises(AttributeError, match="'list' object has no attribute 'select'"):
        accuracy.evaluate([(1, 1)])


def test_native_ml_accepts_explicitly_converted_pandas(session):
    import pandas as pd
    import pyspark.ml.connect as connect
    from pyspark.sql import DataFrame

    # Exercise the native bindings, independently of the deprecated Python ML implementations.
    assert connect.KMeans is __import__("pyspark._pyspark", fromlist=["KMeans"]).KMeans
    pdf = pd.DataFrame({"features": [[0.0, 1.0], [1.0, 0.0], [2.0, 2.0], [3.0, 1.0]],
                        "label": [0.0, 1.0, 1.0, 0.0]})
    df = session.createDataFrame(pdf)
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

def test_ml_connect_columns_spelled_in_another_case_than_the_schema_as_pyspark(session):
    # As Spark resolves a column name regardless of case: a native fit, transform and evaluate
    # whose params spell a column FEATURES, LABEL or PREDICTION, which the schema spells in lower
    # case, read the column as the schema spells it, as the JVM reads the params it is given.
    from pyspark.ml.connect import KMeans
    from pyspark.ml.connect.classification import LogisticRegression
    from pyspark.ml.connect.evaluation import RegressionEvaluator

    train = session.sql(
        "select * from values (array(0.0D, 1.0D), 0.0D), (array(1.0D, 0.0D), 1.0D), "
        "(array(2.0D, 2.0D), 1.0D), (array(3.0D, 1.0D), 0.0D) as t(features, label)"
    )
    model = KMeans(k=2, seed=1, featuresCol="FEATURES").fit(train)
    out = model.transform(train.selectExpr("features AS FEATURES", "label")).toPandas()
    assert list(out.columns) == ["FEATURES", "label", "prediction"]
    assert set(out["prediction"]) <= {0, 1}

    fitted = LogisticRegression(featuresCol="FEATURES", labelCol="LABEL", maxIter=5).fit(
        train.selectExpr("features AS FEATURES", "label AS LABEL")
    )
    out = fitted.transform(train.selectExpr("features AS FEATURES", "label")).toPandas()
    assert list(out.columns) == ["FEATURES", "label", "prediction", "probability"]

    # Standard evaluators require exact schema field names, as official PySpark does.
    scored = session.sql(
        "select label AS LABEL, cast(label AS double) AS prediction "
        "from values (1.0D), (2.0D) as t(label)"
    )
    with pytest.raises(Exception, match="PREDICTION"):
        RegressionEvaluator(labelCol="LABEL", predictionCol="PREDICTION").evaluate(scored)
    metric = RegressionEvaluator(labelCol="LABEL", predictionCol="prediction").evaluate(scored)
    assert metric == pytest.approx(0.0)


def test_ml_connect_estimators_accept_arrays_and_native_scaler_vectors(session):
    from pyspark.ml.connect.classification import LogisticRegression
    from pyspark.ml.connect.clustering import KMeans
    from pyspark.ml.connect.feature import MaxAbsScaler
    from pyspark.ml.functions import array_to_vector

    # LogisticRegression is Spark's JVM estimator here, without the torch params of PySpark's
    # (learningRate), and the labels are ints, as in PySpark's example, whose columns its model
    # outputs: a bigint prediction, and then the array<double> of the probabilities.
    labeled = session.createDataFrame(
        [([1.0, 2.0], 1), ([2.0, -1.0], 1), ([-3.0, -2.0], 0), ([-1.0, -2.0], 0)],
        schema=["features", "label"],
    )
    model = LogisticRegression(maxIter=20).fit(labeled)
    transformed = model.transform(labeled)
    assert transformed.schema.simpleString() == (
        "struct<features:array<double>,label:bigint,prediction:bigint,probability:array<double>>"
    )
    rows = transformed.collect()
    assert [(row.features, row.prediction) for row in rows] == [
        ([1.0, 2.0], 1),
        ([2.0, -1.0], 1),
        ([-3.0, -2.0], 0),
        ([-1.0, -2.0], 0),
    ]
    assert all(type(row.prediction) is int for row in rows)
    for row in rows:
        assert sum(row.probability) == pytest.approx(1.0)
        assert row.probability.index(max(row.probability)) == row.prediction

    # A native scaler output can feed the next estimator.
    scaler = MaxAbsScaler(inputCol="features", outputCol="scaled")
    vectors = labeled.withColumn("features", array_to_vector("features"))
    scaled = scaler.fit(vectors).transform(vectors)
    model = LogisticRegression(featuresCol="scaled", maxIter=20).fit(scaled)
    assert [row.prediction for row in model.transform(scaled).collect()] == [1, 1, 0, 0]
    clusters = KMeans(featuresCol="scaled", k=2, seed=1).fit(scaled).transform(scaled)
    assert clusters.schema["scaled"].dataType.simpleString() == "vector"
    assert len({row.prediction for row in clusters.collect()}) == 2


def test_ml_connect_logistic_regression_checks_its_labels_as_pyspark(session):
    # As PySpark's LogisticRegression._fit, of the distinct labels of its query, which skips a
    # null label: at least two, each an integer in [0, numClasses). The ValueErrors are those of
    # PySpark 4.3.0, raised before the fit, which the JVM estimator would otherwise fit or reject.
    from pyspark.ml.connect.classification import LogisticRegression

    features = [[1.0, 2.0], [2.0, -1.0], [-3.0, -2.0], [-1.0, -2.0]]

    def labeled(labels, label_type="double"):
        rows = [(f, label) for f, label in zip(features, labels)]
        return session.createDataFrame(rows, f"features array<double>, label {label_type}")

    too_few = "Training dataset distinct labels must >= 2."
    not_index = "Training labels must be integers in [0, numClasses)."
    for labels, message in [
        ([1.0, 1.0, 1.0, 1.0], too_few),
        ([0.0, 0.0, 0.0, 0.0], too_few),
        ([0.0, 0.0, 2.0, 2.0], not_index),
        ([1.0, 1.0, 2.0, 2.0], not_index),
        ([-1.0, -1.0, 1.0, 1.0], not_index),
        ([0.5, 0.5, 1.0, 1.0], not_index),
    ]:
        with pytest.raises(ValueError) as error:
            LogisticRegression(maxIter=5).fit(labeled(labels))
        assert str(error.value) == message, labels
    with pytest.raises(ValueError) as error:
        LogisticRegression(maxIter=5).fit(labeled(["a", "a", "b", "b"], "string"))
    assert str(error.value) == not_index
    # The labels of a DataFrame of no rows have no distinct labels.
    with pytest.raises(ValueError) as error:
        LogisticRegression(maxIter=5).fit(labeled([], "double"))
    assert str(error.value) == too_few


def test_ml_connect_logistic_regression_outputs_the_columns_of_pyspark(session):
    # The schemas of the transforms of PySpark 4.3.0's LogisticRegression and scalers of
    # pyspark.ml.connect, of their pandas_udf: the columns of their _output_columns, nullable and
    # of nullable elements, with no metadata, a probability only of a probabilityCol, and no raw
    # prediction. As PySpark's, a transform raises its ValueError of an output column that the
    # DataFrame has, of its exact name, while a fit reads only its input columns, so that the
    # output columns of another transform (a rawPrediction, say) are there.
    from pyspark.ml.connect.classification import LogisticRegression
    from pyspark.sql import functions as F

    def field(name, data_type):
        return {"metadata": {}, "name": name, "nullable": True, "type": data_type}

    doubles = {"containsNull": True, "elementType": "double", "type": "array"}
    inputs = [field("features", doubles), field("label", "long")]
    labeled = session.createDataFrame(
        [([1.0, 2.0], 1), ([2.0, -1.0], 1), ([-3.0, -2.0], 0), ([-1.0, -2.0], 0)],
        schema=["features", "label"],
    )
    for estimator, outputs in [
        (LogisticRegression(maxIter=5), [field("prediction", "long"), field("probability", doubles)]),
        (LogisticRegression(maxIter=5, predictionCol="p", probabilityCol="q"),
         [field("p", "long"), field("q", doubles)]),
        (LogisticRegression(maxIter=5, probabilityCol=""), [field("prediction", "long")]),
    ]:
        transformed = estimator.fit(labeled).transform(labeled)
        assert transformed.schema.jsonValue() == {"fields": inputs + outputs, "type": "struct"}
        assert len(transformed.collect()) == 4

    message = "Transformers does not allow output column having the same name with existing columns."
    model = LogisticRegression(maxIter=5).fit(labeled)
    for column in ("prediction", "probability"):
        with pytest.raises(ValueError) as error:
            model.transform(labeled.withColumn(column, F.lit(0)))
        assert str(error.value) == message
    for column in ("rawPrediction", "probability", "prediction"):
        extra = labeled.withColumn(column, F.lit(0))
        if column == "rawPrediction":
            assert model.transform(extra).columns == [
                "features", "label", "rawPrediction", "prediction", "probability"
            ]
        fitted = LogisticRegression(maxIter=5).fit(extra)
        assert [row.prediction for row in fitted.transform(labeled).collect()] == [1, 1, 0, 0]


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
