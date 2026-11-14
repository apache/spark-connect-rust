# SPDX-License-Identifier: Apache-2.0
"""Native ALS recommendation, with an explicit official PySpark reference mode."""

import gc
import os

import pytest
from _als import GOLDEN, assert_frame, configured
from pyspark.errors import PySparkException
from pyspark.sql import SparkSession

REFERENCE = os.environ.get("ALS_REFERENCE") == "1"


def algorithm(name):
    if REFERENCE:
        from pyspark.ml import recommendation
    else:
        from pyspark import _pyspark
        from pyspark.ml.connect import recommendation

        assert getattr(recommendation, name) is getattr(_pyspark, name)
    return getattr(recommendation, name)


@pytest.fixture
def session():
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live tests")
    spark = SparkSession.builder.remote(remote).getOrCreate()
    yield spark
    spark.stop()


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["name"])
@pytest.mark.parametrize("setters", [False, True])
def test_als_predictions_factors_and_recommendations_match_official(session, case, setters):
    stage = configured(algorithm("ALS"), case["values"], setters)
    assert stage.setNumBlocks(1) is stage
    model = stage.fit(session.sql(GOLDEN["train"]).coalesce(1))
    assert isinstance(model, algorithm("ALSModel"))
    assert model.rank == case["rank"]
    tables = {
        "predictions": model.transform(session.sql(GOLDEN["held"])),
        "user_factors": model.userFactors,
        "item_factors": model.itemFactors,
        "all_users": model.recommendForAllUsers(2),
        "all_items": model.recommendForAllItems(2),
        "user_subset": model.recommendForUserSubset(session.sql(GOLDEN["users"]), 2),
        "item_subset": model.recommendForItemSubset(session.sql(GOLDEN["items"]), 2),
    }
    assert model.setColdStartStrategy("drop") is model
    assert model.setPredictionCol("filtered") is model
    tables["drop_predictions"] = model.transform(session.sql(GOLDEN["held"]))
    del stage, model
    gc.collect()
    for key, frame in tables.items():
        assert_frame(frame, case[key])


@pytest.mark.parametrize(
    "values",
    [dict(rank=0), dict(regParam=-0.1), dict(numUserBlocks=0), dict(coldStartStrategy="invalid")],
)
def test_als_invalid_parameters_are_rejected_by_spark(session, values):
    params = dict(GOLDEN["cases"][0]["values"], **values)
    with pytest.raises(PySparkException):
        configured(algorithm("ALS"), params).fit(session.sql(GOLDEN["train"]))


def test_als_rejects_fractional_ids(session):
    values = GOLDEN["cases"][0]["values"]
    with pytest.raises(PySparkException):
        configured(algorithm("ALS"), values).fit(session.sql("SELECT 0.5D AS u, 1 AS i, 3.0D AS r"))


def test_als_retains_model_used_by_a_recommendation_subset(session):
    from pyspark.sql import functions as F

    if REFERENCE:
        from pyspark.ml.feature import StringIndexer
    else:
        from pyspark.ml.connect.feature import StringIndexer
    values = GOLDEN["cases"][0]["values"]
    model = configured(algorithm("ALS"), values).fit(session.sql(GOLDEN["train"]).coalesce(1))
    data = session.sql("SELECT * FROM VALUES ('a'),('c') AS t(category)")
    indexer = StringIndexer(inputCol="category", outputCol="u", stringOrderType="alphabetAsc").fit(
        session.sql("SELECT * FROM VALUES ('a'),('b'),('c') AS t(category)")
    )
    subset = indexer.transform(data).select(F.col("u").cast("int").alias("u"))
    result = model.recommendForUserSubset(subset, 2)
    if not REFERENCE:
        del subset, indexer, model
        gc.collect()
    assert_frame(result, GOLDEN["cases"][0]["user_subset"])


def test_als_other_session_fails_transform_at_collect_and_recommends_as_pyspark(session):
    # As PySpark: the transform of a DataFrame of another session fails at collect, with
    # CONNECT_ML.CACHE_INVALID, and the subsets of another session's DataFrames recommend.
    from pyspark.errors.exceptions.connect import SparkException

    model = configured(algorithm("ALS"), GOLDEN["cases"][0]["values"]).fit(
        session.sql(GOLDEN["train"]).coalesce(1)
    )
    other = session.newSession()
    try:
        transformed = model.transform(other.sql(GOLDEN["held"]))
        with pytest.raises(SparkException) as error:
            transformed.collect()
        assert error.value.getCondition() == "CONNECT_ML.CACHE_INVALID"
        assert model.recommendForUserSubset(other.sql(GOLDEN["users"]), 2).count() > 0
        assert model.recommendForItemSubset(other.sql(GOLDEN["items"]), 2).count() > 0
    finally:
        other.stop()
