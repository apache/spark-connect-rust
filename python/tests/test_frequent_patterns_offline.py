# SPDX-License-Identifier: Apache-2.0
"""Native frequent-pattern signatures, wire protocol and input rejection."""

import base64
import inspect

import pytest
from _frequent_patterns import GOLDEN, configured
from _mock_connect_server import mock_connect_session
from pyspark import _pyspark
from pyspark.ml.connect import fpm
from pyspark.sql import SparkSession
from pyspark.sql.connect import proto as pb2
from test_ml_offline import _commands_but_deletes, _operator_info, _relation, _requests_but_deletes


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


@pytest.mark.parametrize("name", ["FPGrowth", "PrefixSpan"])
def test_native_exports_and_constructor_signatures(name):
    import pyspark.ml.connect as connect

    cls = getattr(fpm, name)
    assert cls is getattr(_pyspark, name) is getattr(connect, name)
    parameters = inspect.signature(cls).parameters
    assert list(parameters) == list(GOLDEN["defaults"][name])
    for key, value in GOLDEN["defaults"][name].items():
        assert parameters[key].default == value
        assert parameters[key].kind == inspect.Parameter.KEYWORD_ONLY
    with pytest.raises(TypeError, match="keyword arguments"):
        cls(1)
    with pytest.raises(TypeError, match="unexpected keyword"):
        cls(unknown=1)


@pytest.mark.parametrize("case", GOLDEN["params"])
@pytest.mark.parametrize("setters", [False, True])
def test_parameters_and_helper_arguments_match_official(session, case, setters):
    spark, service = session
    stage = configured(getattr(fpm, case["name"]), case["values"], setters)
    expected = pb2.MlParams.FromString(base64.b64decode(case["params_b64"]))
    frame = spark.range(1)
    if case["name"] == "FPGrowth":
        service.execute_payload = _operator_info("mining-model")
        sent = len(service.execute_requests)
        model = stage.fit(frame)
        assert isinstance(model, fpm.FPGrowthModel)
        fit = _commands_but_deletes(service, sent)[0].ml_command.fit
        assert fit.estimator.name == "org.apache.spark.ml.fpm.FPGrowth"
        assert fit.params == expected
        assert fit.dataset == _relation(frame)
        transformed = _relation(model.transform(frame)).ml_relation.transform
        assert transformed.params == expected
        assert transformed.obj_ref.id == "mining-model"
        for attribute in ["freqItemsets", "associationRules"]:
            sent = _requests_but_deletes(service)
            fetch = _relation(getattr(model, attribute)).ml_relation.fetch
            assert fetch.obj_ref.id == "mining-model"
            assert len(fetch.methods) == 1
            assert fetch.methods[0].method == attribute
            assert not fetch.methods[0].args
            assert _requests_but_deletes(service) == sent
    else:
        sent = _requests_but_deletes(service)
        result = stage.findFrequentSequentialPatterns(frame)
        fetch = _relation(result).ml_relation.fetch
        assert fetch.obj_ref.id == "______ML_CONNECT_HELPER______"
        assert len(fetch.methods) == 1
        method = fetch.methods[0]
        assert method.method == case["helper"]
        assert method.args[0].input == _relation(frame)
        assert [arg.param.SerializeToString() for arg in method.args[1:]] == [
            base64.b64decode(v) for v in case["args_b64"]
        ]
        assert _requests_but_deletes(service) == sent


@pytest.mark.parametrize(
    "name,params", [(name, list(values)) for name, values in GOLDEN["defaults"].items()]
)
def test_setters_validate_types(name, params):
    stage = getattr(fpm, name)()
    for param in params:
        with pytest.raises(TypeError, match="Invalid param value"):
            getattr(stage, "set" + param[0].upper() + param[1:])(object())


def test_rejects_pandas_before_validation_or_requests(session):
    import pandas as pd
    from pyspark.errors import PySparkTypeError

    spark, service = session
    service.execute_payload = _operator_info("pandas-model")
    model = fpm.FPGrowth().fit(spark.range(1))
    patterns = fpm.PrefixSpan(maxLocalProjDBSize=2**70).findFrequentSequentialPatterns
    for frame in [pd.DataFrame(), pd.DataFrame({"items": [["a"]]})]:
        for method in [fpm.FPGrowth(numPartitions=2**70).fit, model.transform]:
            sent = _requests_but_deletes(service)
            with pytest.raises(TypeError, match="requires a Spark DataFrame"):
                method(frame)
            assert _requests_but_deletes(service) == sent
        # The helper call sends its dataset first, which pyspark.ml's serialization converts to a
        # literal before the params: a pandas DataFrame raises the UNSUPPORTED_DATA_TYPE of that
        # conversion, as pyspark.ml's does.
        sent = _requests_but_deletes(service)
        with pytest.raises(PySparkTypeError) as error:
            patterns(frame)
        assert error.value.getCondition() == "UNSUPPORTED_DATA_TYPE"
        assert error.value.getMessageParameters() == {"data_type": "DataFrame"}
        assert _requests_but_deletes(service) == sent


@pytest.mark.parametrize(
    "name,param", [("FPGrowth", "numPartitions"), ("PrefixSpan", "maxLocalProjDBSize")]
)
def test_oversized_integers_fail_before_requests_and_setters_recover(session, name, param):
    from pyspark.errors import PySparkValueError

    spark, service = session
    stage = getattr(fpm, name)(**{param: 2**70})
    operation = stage.fit if name == "FPGrowth" else stage.findFrequentSequentialPatterns
    sent = _requests_but_deletes(service)
    with pytest.raises(PySparkValueError, match="VALUE_NOT_BETWEEN"):
        operation(spark.range(1))
    assert _requests_but_deletes(service) == sent
    getattr(stage, "set" + param[0].upper() + param[1:])(1)
    service.execute_payload = _operator_info("recovered-model")
    operation(spark.range(1))


def test_fitted_model_named_setters_snapshot_transform_parameters(session):
    spark, service = session
    service.execute_payload = _operator_info("model-setters")
    model = fpm.FPGrowth(itemsCol="items", minConfidence=0.6).fit(spark.range(1))
    before = model.transform(spark.range(1))
    assert model.setItemsCol("basket") is model
    assert model.setPredictionCol("next") is model
    assert model.setMinConfidence(0.9) is model
    after = _relation(model.transform(spark.range(1))).ml_relation.transform.params.params
    assert after["itemsCol"].string == "basket" and after["predictionCol"].string == "next"
    assert after["minConfidence"].double == 0.9
    assert _relation(before).ml_relation.transform.params.params["minConfidence"].double == 0.6
