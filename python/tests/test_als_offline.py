# SPDX-License-Identifier: Apache-2.0
"""Native ALS parameter, model-relation and dataset-contract checks."""

import base64
import inspect

import pytest
from _als import GOLDEN, configured
from _mock_connect_server import mock_connect_session
from pyspark import _pyspark
from pyspark.ml.connect import recommendation
from pyspark.sql import SparkSession
from pyspark.sql.connect import proto as pb2
from test_ml_offline import _commands_but_deletes, _operator_info, _relation, _requests_but_deletes

MODEL_PARAMS = {"blockSize", "coldStartStrategy", "itemCol", "predictionCol", "userCol"}


@pytest.fixture(scope="module")
def server():
    with mock_connect_session() as pair:
        yield pair


@pytest.fixture
def session(server):
    _, service = server
    service.reset()
    spark = SparkSession.builder.remote(service.remote).getOrCreate()
    yield spark, service
    service.forget_session(spark.session_id)
    spark.stop()


def fitted(spark, service, values=None):
    service.execute_payload = _operator_info("als-model")
    return configured(recommendation.ALS, values or GOLDEN["cases"][0]["values"]).fit(
        spark.range(1)
    )


def test_native_exports_and_constructor_signature():
    import pyspark.ml.connect as connect

    assert recommendation.ALS is connect.ALS is _pyspark.ALS
    assert recommendation.ALSModel is connect.ALSModel is _pyspark.ALSModel
    params = inspect.signature(recommendation.ALS).parameters
    assert list(params) == list(GOLDEN["defaults"])
    for key, value in GOLDEN["defaults"].items():
        assert params[key].default == value
        assert params[key].kind == inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError, match="keyword arguments"):
        recommendation.ALS(2)
    with pytest.raises(TypeError, match="unexpected keyword"):
        recommendation.ALS(predictionCol="score")


@pytest.mark.parametrize("case", GOLDEN["params"])
@pytest.mark.parametrize("setters", [False, True])
def test_als_fit_parameters_and_filtered_model_transform_match_reference(session, case, setters):
    spark, service = session
    stage = configured(recommendation.ALS, case["values"], setters)
    service.execute_payload = _operator_info("als-model")
    sent = len(service.execute_requests)
    model = stage.fit(spark.range(1))
    expected = pb2.MlParams.FromString(base64.b64decode(case["params_b64"]))
    fit = _commands_but_deletes(service, sent)[0].ml_command.fit
    assert fit.estimator.name == "org.apache.spark.ml.recommendation.ALS"
    assert fit.params == expected
    assert isinstance(model, recommendation.ALSModel)
    relation = _relation(model.transform(spark.range(1))).ml_relation.transform
    assert relation.obj_ref.id == "als-model"
    assert dict(relation.params.params) == {
        k: v for k, v in expected.params.items() if k in MODEL_PARAMS
    }


def test_num_blocks_sets_both_partition_parameters(session):
    spark, service = session
    stage = recommendation.ALS()
    assert stage.setNumBlocks(3) is stage
    service.execute_payload = _operator_info("blocks-model")
    sent = len(service.execute_requests)
    stage.fit(spark.range(1))
    params = _commands_but_deletes(service, sent)[0].ml_command.fit.params.params
    assert dict(params) == {
        "numUserBlocks": pb2.Expression.Literal(integer=3),
        "numItemBlocks": pb2.Expression.Literal(integer=3),
    }


@pytest.mark.parametrize("case", GOLDEN["fetches"])
def test_recommendation_count_literals_match_reference(session, case):
    spark, service = session
    model = fitted(spark, service)
    sent = _requests_but_deletes(service)
    fetch = _relation(model.recommendForAllUsers(numItems=case["count"])).ml_relation.fetch
    assert fetch.obj_ref.id == "als-model"
    assert fetch.methods[0].method == case["method"]
    assert fetch.methods[0].args[0].param.SerializeToString() == base64.b64decode(case["arg_b64"])
    assert _requests_but_deletes(service) == sent


def test_factor_properties_rank_and_recommendation_methods_are_lazy(session):
    spark, service = session
    model = fitted(spark, service)
    frame = spark.range(1)
    sent = _requests_but_deletes(service)
    assert model.rank == 2
    for attribute in ["userFactors", "itemFactors"]:
        fetch = _relation(getattr(model, attribute)).ml_relation.fetch
        assert fetch.obj_ref.id == "als-model"
        assert fetch.methods[0].method == attribute
        assert not fetch.methods[0].args
    for name, kwarg, subset in [
        ("recommendForAllUsers", "numItems", False),
        ("recommendForAllItems", "numUsers", False),
        ("recommendForUserSubset", "numItems", True),
        ("recommendForItemSubset", "numUsers", True),
    ]:
        kwargs = {kwarg: 2}
        if subset:
            kwargs["dataset"] = frame
        fetch = _relation(getattr(model, name)(**kwargs)).ml_relation.fetch
        assert fetch.methods[0].method == name
        assert fetch.obj_ref.id == "als-model"
        assert fetch.methods[0].args[-1].param == pb2.Expression.Literal(integer=2)
        if subset:
            assert fetch.methods[0].args[0].input == _relation(frame)
    assert _requests_but_deletes(service) == sent


@pytest.mark.parametrize("param", list(GOLDEN["defaults"]) + ["predictionCol"])
def test_als_setters_validate_types(param):
    stage = recommendation.ALS()
    with pytest.raises(TypeError, match="Invalid param value"):
        getattr(stage, "set" + param[0].upper() + param[1:])(object())


def test_pandas_is_rejected_before_requests_in_fit_transform_and_subsets(session):
    import pandas as pd

    spark, service = session
    model = fitted(spark, service)
    operations = [
        recommendation.ALS(rank=2**70).fit,
        model.transform,
        lambda x: model.recommendForUserSubset(x, object()),
        lambda x: model.recommendForItemSubset(x, object()),
    ]
    for operation in operations:
        for pdf in [pd.DataFrame(), pd.DataFrame({"u": [0]})]:
            sent = _requests_but_deletes(service)
            with pytest.raises(TypeError, match="requires a Spark DataFrame"):
                operation(pdf)
            assert _requests_but_deletes(service) == sent


def test_unsendable_recommendation_counts_fail_before_requests(session):
    from pyspark.errors import PySparkTypeError, PySparkValueError

    spark, service = session
    model = fitted(spark, service)
    for value, error in [
        (2**70, PySparkValueError),
        ([], PySparkTypeError),
        (object(), PySparkTypeError),
    ]:
        sent = _requests_but_deletes(service)
        with pytest.raises(error):
            model.recommendForAllUsers(value)
        assert _requests_but_deletes(service) == sent


def test_model_setters_snapshot_transform_parameters(session):
    spark, service = session
    model = fitted(spark, service)
    before = model.transform(spark.range(1))
    for name, value in [
        ("setUserCol", "user"),
        ("setItemCol", "item"),
        ("setPredictionCol", "p"),
        ("setColdStartStrategy", "drop"),
        ("setBlockSize", 16),
    ]:
        assert getattr(model, name)(value) is model
    after = _relation(model.transform(spark.range(1))).ml_relation.transform.params.params
    assert after["userCol"].string == "user" and after["itemCol"].string == "item"
    assert after["predictionCol"].string == "p" and after["coldStartStrategy"].string == "drop"
    assert after["blockSize"].integer == 16
    assert _relation(before).ml_relation.transform.params.params["predictionCol"].string == "score"
