# SPDX-License-Identifier: Apache-2.0
"""Feature bindings: official signatures, parameter wire types and dataset contracts."""
import base64
import inspect

import pytest
from _ml_features import GOLDEN, assert_values, configured
from _mock_connect_server import mock_connect_session
from pyspark import _pyspark
from pyspark.ml.connect import feature
from pyspark.sql import SparkSession
from pyspark.sql.connect import proto as pb2
from test_ml_offline import (
    _commands_but_deletes,
    _operator_info,
    _relation,
    _requests_but_deletes,
)
from test_ml_offline import (
    _ml_mock_server as _ml_mock_server,  # noqa: F401
)


@pytest.fixture(scope="module")
def _feature_mock_server():
    with mock_connect_session() as pair:
        yield pair


@pytest.fixture
def ml_mock(_feature_mock_server):
    _, service = _feature_mock_server
    service.reset()
    spark = SparkSession.builder.remote(service.remote).getOrCreate()
    yield spark, service
    service.forget_session(spark.session_id)
    spark.stop()

def wire(stage, name, spark, service):
    frame = spark.range(1)
    if GOLDEN["api"][name]["estimator"]:
        service.execute_payload = _operator_info("feature-model")
        sent = len(service.execute_requests)
        model = stage.fit(frame)
        params = _commands_but_deletes(service, sent)[0].ml_command.fit.params
        transform = _relation(model.transform(frame)).ml_relation.transform
        assert transform.obj_ref.id == "feature-model"
        accepted = GOLDEN["api"][name]["model_params"]
        expected = {k: v.SerializeToString() for k, v in params.params.items() if k in accepted}
        assert {k: v.SerializeToString() for k, v in transform.params.params.items()} == expected
        return params
    return _relation(stage.transform(frame)).ml_relation.transform.params


@pytest.mark.parametrize("name", GOLDEN["api"])
def test_native_exports_and_official_constructor_signature(name):
    import pyspark.ml.connect as connect
    cls = getattr(feature, name)
    assert cls is getattr(_pyspark, name) is getattr(connect, name)
    expected = GOLDEN["api"][name]["defaults"]
    parameters = inspect.signature(cls).parameters
    assert list(parameters) == list(expected)
    for key, default in expected.items():
        assert_values(parameters[key].default, default)
        assert parameters[key].kind == inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError, match="keyword arguments"):
        cls(1)
    with pytest.raises(TypeError, match="unexpected keyword"):
        cls(unknown=1)
    # setParams, as pyspark.ml's: keyword-only, with the params and defaults of the constructor.
    assert str(inspect.signature(cls.setParams)) == "(self, " + str(inspect.signature(cls))[1:]
    with pytest.raises(TypeError, match="Method setParams forces keyword arguments."):
        cls().setParams(1)
    with pytest.raises(TypeError, match=f"{name}.setParams\\(\\) got an unexpected keyword"):
        cls().setParams(unknown=1)


@pytest.mark.parametrize("case", GOLDEN["params"], ids=lambda c: c["name"])
@pytest.mark.parametrize("via", ["constructor", "setters", "setParams"])
def test_feature_parameters_match_official_wire_and_filter_fitted_model(ml_mock, case, via):
    spark, service = ml_mock
    stage = configured(
        getattr(feature, case["name"]),
        case["values"],
        setters=via == "setters",
        set_params=via == "setParams",
    )
    actual = wire(stage, case["name"], spark, service)
    expected = pb2.MlParams.FromString(base64.b64decode(case["params_b64"]))
    # IEEE NaN is not equal to itself; compare serialized literals.
    assert {k: v.SerializeToString() for k, v in actual.params.items()} == {k: v.SerializeToString() for k, v in expected.params.items()}


@pytest.mark.parametrize("name", GOLDEN["api"])
def test_feature_stages_take_the_params_argument_of_pyspark_ml(ml_mock, name):
    # As pyspark.ml's fit and transform: None and an empty dict change nothing, and any other
    # value raises a TypeError before any request (the native stages take no param map).
    spark, service = ml_mock
    stage = getattr(feature, name)()
    estimator = GOLDEN["api"][name]["estimator"]
    method = stage.fit if estimator else stage.transform
    frame = spark.range(1)
    for value in [{"x": 1}, [{}], ("x",), "x"]:
        sent = _requests_but_deletes(service)
        with pytest.raises(TypeError):
            method(frame, value)
        assert _requests_but_deletes(service) == sent
    if estimator:
        service.execute_payload = _operator_info("params-argument-model")
        for value in [None, {}]:
            sent = len(service.execute_requests)
            stage.fit(frame, value)
            assert len(_commands_but_deletes(service, sent)) == 1
    else:
        base = _relation(stage.transform(frame)).ml_relation.transform.params
        for value in [None, {}]:
            assert _relation(stage.transform(frame, value)).ml_relation.transform.params == base


@pytest.mark.parametrize("name", GOLDEN["api"])
def test_feature_stages_and_fitted_models_reject_pandas_before_requests(ml_mock, name):
    import pandas as pd
    spark, service = ml_mock
    stage = getattr(feature, name)()
    estimator = GOLDEN["api"][name]["estimator"]
    method = stage.fit if estimator else stage.transform
    for pdf in [pd.DataFrame(), pd.DataFrame({"features": [[1., 2.]]})]:
        sent = _requests_but_deletes(service)
        with pytest.raises(TypeError, match="requires a Spark DataFrame"):
            method(pdf)
        assert _requests_but_deletes(service) == sent
    if estimator:
        service.execute_payload = _operator_info("pandas-rejection-model")
        model = stage.fit(spark.range(1))
        sent = _requests_but_deletes(service)
        with pytest.raises(TypeError, match="requires a Spark DataFrame"):
            model.transform(pd.DataFrame())
        assert _requests_but_deletes(service) == sent


@pytest.mark.parametrize("values", [[], [[], [0., 1.]]])
def test_bucketizer_rejects_uninferrable_nested_splits_without_request(ml_mock, values):
    from pyspark.errors import PySparkTypeError
    spark, service = ml_mock
    stage = feature.Bucketizer(splitsArray=values)
    sent = _requests_but_deletes(service)
    with pytest.raises(PySparkTypeError, match="CANNOT_INFER_ARRAY_ELEMENT_TYPE"):
        stage.transform(spark.range(1))
    assert _requests_but_deletes(service) == sent


def test_bucketizer_later_empty_array_has_inferred_type(ml_mock):
    spark, service = ml_mock
    params = wire(feature.Bucketizer(splitsArray=[[0., 1.], []]), "Bucketizer", spark, service)
    assert len(params.params["splitsArray"].array.elements[1].array.elements) == 0


def test_regex_identity_is_deferred_and_retains_python_value(ml_mock):
    from pyspark.errors import PySparkTypeError
    spark, service = ml_mock
    for value, expected in [(True, pb2.Expression.Literal(boolean=True)), (1, pb2.Expression.Literal(integer=1)), ("false", pb2.Expression.Literal(string="false"))]:
        assert wire(feature.RegexTokenizer(gaps=value), "RegexTokenizer", spark, service).params["gaps"] == expected
    values = []
    stage = feature.RegexTokenizer(gaps=values)
    with pytest.raises(PySparkTypeError, match="CANNOT_INFER_ARRAY_ELEMENT_TYPE"):
        stage.transform(spark.range(1))
    values.append(True)
    assert wire(stage, "RegexTokenizer", spark, service).params["gaps"].array.elements[0].boolean
    stage.setGaps(object())
    with pytest.raises(PySparkTypeError, match="UNSUPPORTED_DATA_TYPE"):
        stage.transform(spark.range(1))
    stage.setGaps(False)
    assert wire(stage, "RegexTokenizer", spark, service).params["gaps"] == pb2.Expression.Literal(boolean=False)


@pytest.mark.parametrize("name,param", [(name, param) for name, api in GOLDEN["api"].items() for param, converter in api["params"].items() if converter != "identity"])
def test_feature_setters_reject_invalid_types_without_mutating(name, param):
    stage = getattr(feature, name)()
    setter = getattr(stage, "set" + param[0].upper() + param[1:])
    with pytest.raises(TypeError, match="Invalid param value"):
        setter(object())



def test_elementwise_product_sparse_vector_parameter(ml_mock):
    from pyspark.ml.connect.serialize import serialize_param
    from pyspark.ml.linalg import Vectors
    spark, service = ml_mock
    vector = Vectors.sparse(4, {1: 2., 3: -1.})
    stage = feature.ElementwiseProduct(scalingVec=vector)
    expected = serialize_param(vector, spark.client)
    assert wire(stage, "ElementwiseProduct", spark, service).params["scalingVec"].SerializeToString() == expected.SerializeToString()
