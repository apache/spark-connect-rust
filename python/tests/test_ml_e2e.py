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
