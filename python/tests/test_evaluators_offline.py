# SPDX-License-Identifier: Apache-2.0
"""Native evaluator wire, constructor, direction and input validation parity."""

import base64
import inspect

import pytest
from _evaluators import FIXTURE
from pyspark.ml.connect import evaluation
from pyspark.sql.connect import proto as pb2
from test_ml_offline import (
    _ml_mock_server as _ml_mock_server,
)
from test_ml_offline import (
    _native_request,
    _requests_but_deletes,
)
from test_ml_offline import (
    ml_mock as _ml_fixture,
)

ml_mock = _ml_fixture

PARAMS = [(name, case) for name, info in FIXTURE["classes"].items() for case in info["params"]]
DIRECTIONS = [
    (name, case) for name, info in FIXTURE["classes"].items() for case in info["directions"]
]


@pytest.mark.parametrize("name,case", PARAMS)
@pytest.mark.parametrize("use_setters", [False, True])
def test_evaluator_parameters_match_official_pyspark(ml_mock, name, case, use_setters):
    spark, service = ml_mock
    cls = getattr(evaluation, name)
    obj = cls() if use_setters else cls(**case["values"])
    if use_setters:
        for key, value in case["values"].items():
            assert getattr(obj, "set" + key[0].upper() + key[1:])(value=value) is obj
    operator, params = _native_request(spark, service, obj, [])
    expected = pb2.MlParams.FromString(base64.b64decode(case["params_b64"]))
    assert params == expected
    assert operator.name == "org.apache.spark.ml.evaluation." + name
    assert operator.type == pb2.MlOperator.OPERATOR_TYPE_EVALUATOR
    assert obj.uid == operator.uid


@pytest.mark.parametrize("name,case", DIRECTIONS)
def test_metric_direction_matches_official_without_a_session(name, case):
    cls = getattr(evaluation, name)
    obj = cls(metricName=case["metric"])
    assert obj.isLargerBetter() is case["larger"]
    other = cls()
    assert other.setMetricName(case["metric"]) is other
    assert other.isLargerBetter() is case["larger"]


@pytest.mark.parametrize("name", FIXTURE["classes"])
def test_evaluator_exports_signature_and_pandas_guard(ml_mock, name):
    import pandas as pd
    import pyspark._pyspark as native
    import pyspark.ml.connect as root

    cls = getattr(evaluation, name)
    assert cls is getattr(native, name) is getattr(root, name)
    assert str(inspect.signature(cls)) == FIXTURE["classes"][name]["signature"]
    _, service = ml_mock
    before = _requests_but_deletes(service)
    with pytest.raises(TypeError, match="requires a Spark DataFrame"):
        cls().evaluate(pd.DataFrame({"label": [1.0], "prediction": [1.0]}))
    assert _requests_but_deletes(service) == before
    with pytest.raises(TypeError, match="forces keyword arguments"):
        cls("prediction")
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        cls(unknown=1)


@pytest.mark.parametrize(
    "name,param", [("BinaryClassificationEvaluator", "numBins"), ("RankingEvaluator", "k")]
)
def test_pandas_rejection_precedes_unsendable_integer_validation(ml_mock, name, param):
    import pandas as pd

    _, service = ml_mock
    before = _requests_but_deletes(service)
    obj = getattr(evaluation, name)(**{param: 2**80})
    with pytest.raises(TypeError, match="requires a Spark DataFrame"):
        obj.evaluate(pd.DataFrame())
    assert _requests_but_deletes(service) == before
