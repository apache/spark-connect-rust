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
