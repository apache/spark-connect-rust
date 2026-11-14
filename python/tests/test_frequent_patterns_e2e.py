# SPDX-License-Identifier: Apache-2.0
"""Native frequent-pattern mining, with explicit official-reference mode."""

import gc
import os

import pytest
from _frequent_patterns import GOLDEN, assert_values, configured, normalized
from pyspark.errors import PySparkException
from pyspark.errors.exceptions.connect import SparkException
from pyspark.sql import SparkSession

REFERENCE = os.environ.get("FPM_REFERENCE") == "1"


def algorithm(name):
    if REFERENCE:
        from pyspark.ml import fpm
    else:
        from pyspark import _pyspark
        from pyspark.ml.connect import fpm

        assert getattr(fpm, name) is getattr(_pyspark, name)
    return getattr(fpm, name)


@pytest.fixture
def session():
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live tests")
    spark = SparkSession.builder.remote(remote).getOrCreate()
    yield spark
    spark.stop()


@pytest.mark.parametrize("case", GOLDEN["fpgrowth"], ids=lambda c: c["id"])
@pytest.mark.parametrize("setters", [False, True])
def test_fpgrowth_mines_and_predicts_like_official_and_retains_results(session, case, setters):
    estimator = configured(algorithm("FPGrowth"), case["values"], setters)
    model = estimator.fit(session.sql(case["sql"]))
    assert isinstance(model, algorithm("FPGrowthModel"))
    held = session.sql(case["held"])
    tables = {
        "itemsets": model.freqItemsets,
        "rules": model.associationRules,
        "predictions": model.transform(held),
    }
    assert model.setMinConfidence(0.95) is model
    assert model.setPredictionCol("changed") is model
    tables["changed_predictions"] = model.transform(held)
    del model, estimator
    gc.collect()
    for name, table in tables.items():
        assert_values(
            normalized(table, "predictions" if name == "changed_predictions" else name),
            case[name]["rows"],
        )
        assert table.schema.jsonValue() == case[name]["schema"]


@pytest.mark.parametrize("case", GOLDEN["prefixspan"])
@pytest.mark.parametrize("setters", [False, True])
def test_prefixspan_matches_official_patterns_and_projection_limits(session, case, setters):
    stage = configured(algorithm("PrefixSpan"), case["values"], setters)
    patterns = stage.findFrequentSequentialPatterns(session.sql(GOLDEN["sequences"]))
    stage.setMaxPatternLength(99)
    del stage
    assert_values(normalized(patterns, "sequences"), case["rows"])
    assert patterns.schema.jsonValue() == case["schema"]


@pytest.mark.parametrize(
    "name,values",
    [
        ("FPGrowth", dict(minSupport=-0.1)),
        ("FPGrowth", dict(minConfidence=1.1)),
        ("FPGrowth", dict(numPartitions=0)),
        ("PrefixSpan", dict(minSupport=-0.1)),
        ("PrefixSpan", dict(maxPatternLength=0)),
    ],
)
def test_spark_rejects_invalid_mining_parameters(session, name, values):
    with pytest.raises(PySparkException):
        if name == "FPGrowth":
            algorithm(name)(itemsCol="basket", **values).fit(
                session.sql(GOLDEN["fpgrowth"][0]["sql"])
            )
        else:
            algorithm(name)(sequenceCol="events", **values).findFrequentSequentialPatterns(
                session.sql(GOLDEN["sequences"])
            ).collect()


def test_duplicate_items_are_rejected(session):
    with pytest.raises(PySparkException):
        algorithm("FPGrowth")().fit(session.sql("SELECT array('a','a') AS items"))


def test_prefixspan_composes_with_upstream_fitted_model(session):
    from pyspark.sql import functions as F

    case = GOLDEN["fpgrowth"][0]
    model = algorithm("FPGrowth")(**case["values"]).fit(session.sql(case["sql"]))
    predicted = model.transform(session.sql(case["held"]))
    sequences = predicted.select(F.array("suggested").alias("sequence"))
    result = algorithm("PrefixSpan")(
        minSupport=0.1, maxPatternLength=2
    ).findFrequentSequentialPatterns(sequences)
    # Official helper relations do not retain upstream models; retain them in reference mode.
    if not REFERENCE:
        del sequences, predicted, model
        gc.collect()
    assert result.count() > 0


def test_model_transform_of_another_session_fails_at_collect_as_pyspark(session):
    # As PySpark: the transform of a DataFrame of another session is a DataFrame, whose analysis
    # and execution fail on the server of that session, which caches no such model.
    case = GOLDEN["fpgrowth"][0]
    model = algorithm("FPGrowth")(**case["values"]).fit(session.sql(case["sql"]))
    other = session.newSession()
    try:
        transformed = model.transform(other.sql(case["held"]))
        with pytest.raises(SparkException) as error:
            transformed.collect()
        assert error.value.getCondition() == "CONNECT_ML.CACHE_INVALID"
    finally:
        other.stop()


def test_prefixspan_reports_the_server_limit_for_long_helper_arguments(session):
    with pytest.raises(PySparkException, match="LONG not supported"):
        algorithm("PrefixSpan")(
            sequenceCol="events", maxLocalProjDBSize=2**31
        ).findFrequentSequentialPatterns(session.sql(GOLDEN["sequences"])).collect()
