# SPDX-License-Identifier: Apache-2.0
"""Every standard evaluator metric, checked against official PySpark 4.3.0."""

import os

import pytest
from _evaluators import FIXTURE, dataset
from pyspark.sql import SparkSession


@pytest.fixture(scope="module")
def session():
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live evaluator tests")
    spark = SparkSession.builder.remote(remote).getOrCreate()
    yield spark
    spark.stop()


@pytest.fixture(scope="module")
def frames(session):
    frames = {key: dataset(session, spec).cache() for key, spec in FIXTURE["datasets"].items()}
    yield frames
    for frame in frames.values():
        frame.unpersist()


def evaluator_class(name):
    if os.environ.get("EVALUATORS_REFERENCE") == "1":
        from pyspark.ml import evaluation
    else:
        import pyspark._pyspark as native
        from pyspark.ml.connect import evaluation

        assert getattr(evaluation, name) is getattr(native, name)
    return getattr(evaluation, name)


@pytest.mark.parametrize("case", FIXTURE["cases"], ids=lambda case: case["name"])
def test_every_metric_matches_official(case, frames):
    evaluator = evaluator_class(case["class"])(**case["values"])
    assert evaluator.evaluate(frames[case["dataset"]]) == pytest.approx(
        case["value"], rel=1e-10, abs=1e-10
    )
    assert evaluator.isLargerBetter() is case["larger"]


@pytest.mark.parametrize("name", FIXTURE["classes"])
def test_invalid_metric_is_validated_by_spark(name, frames):
    case = next(case for case in FIXTURE["cases"] if case["class"] == name)
    evaluator = evaluator_class(name)(**dict(case["values"], metricName="invalid"))
    with pytest.raises(Exception, match="metricName"):
        evaluator.evaluate(frames[case["dataset"]])


@pytest.mark.parametrize(
    "name,param,value",
    [
        ("ClusteringEvaluator", "distanceMeasure", "invalid"),
        ("RankingEvaluator", "k", 0),
        ("MulticlassClassificationEvaluator", "eps", 0.5),
        ("MulticlassClassificationEvaluator", "beta", 0.0),
        ("MultilabelClassificationEvaluator", "metricLabel", -1.0),
        ("BinaryClassificationEvaluator", "numBins", -1),
    ],
)
def test_invalid_parameters_are_validated_by_spark(name, param, value, frames):
    case = next(case for case in FIXTURE["cases"] if case["class"] == name)
    evaluator = evaluator_class(name)(**dict(case["values"], **{param: value}))
    with pytest.raises(Exception, match=param):
        evaluator.evaluate(frames[case["dataset"]])


@pytest.mark.parametrize("name", FIXTURE["classes"])
def test_named_setter_changes_the_next_evaluation(name, frames):
    cases = [case for case in FIXTURE["cases"] if case["class"] == name]
    first = cases[0]
    second = next(case for case in cases[1:] if set(first["values"]) <= set(case["values"]))
    evaluator = evaluator_class(name)(**first["values"])
    assert evaluator.evaluate(frames[first["dataset"]]) == pytest.approx(first["value"])
    for key, value in second["values"].items():
        assert getattr(evaluator, "set" + key[0].upper() + key[1:])(value) is evaluator
    assert evaluator.evaluate(frames[second["dataset"]]) == pytest.approx(second["value"])


@pytest.mark.parametrize("case", FIXTURE["inputs"], ids=lambda case: case["name"])
def test_standard_input_types_and_column_names(case, session):
    evaluator = evaluator_class(case["class"])(**case["values"])
    frame = session.sql(case["sql"])
    if case["accepted"]:
        assert evaluator.evaluate(frame) == pytest.approx(case["result"])
    else:
        with pytest.raises(Exception) as error:
            evaluator.evaluate(frame)
        assert case["message"] in str(error.value)
        if case["condition"]:
            assert error.value.getCondition() == case["condition"]
