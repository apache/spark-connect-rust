# SPDX-License-Identifier: Apache-2.0
"""Native statistics API, reference protobufs and validation without a Spark server."""

import base64
import inspect
from _statistics import FIXTURE, api, helper, summary

import pytest
from _mock_connect_server import mock_connect_session
from pyspark import _pyspark
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.connect import proto as pb2
from test_ml_offline import _relation, _requests_but_deletes


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


def clean(message):
    for field, value in list(message.ListFields()):
        if field.name in {"origin", "common"}:
            message.ClearField(field.name)
        elif field.message_type:
            if field.is_repeated:
                for child in value:
                    clean(child)
            else:
                clean(value)
    return message


def test_native_exports_and_reference_signatures():
    import pyspark.ml.connect as connect

    stat = api()
    for name in stat.__all__:
        assert getattr(stat, name) is getattr(connect, name) is getattr(_pyspark, name)
    for key, expected in FIXTURE["signatures"].items():
        cls, method = key.split(".")
        target = getattr(stat, cls) if method == "__init__" else getattr(getattr(stat, cls), method)
        actual = [
            {
                "name": p.name,
                "kind": p.kind.name,
                **({"default": p.default} if p.default is not inspect.Parameter.empty else {}),
            }
            for p in inspect.signature(target).parameters.values()
            if p.name != "self"
        ]
        assert actual == expected, key


@pytest.mark.parametrize("case", FIXTURE["helpers"])
def test_helper_protocol_matches_reference_and_is_lazy(session, case):
    spark, service = session
    frame = spark.range(4)
    sent = _requests_but_deletes(service)
    fetch = _relation(helper(case, frame)).ml_relation.fetch
    assert fetch.obj_ref.id == "______ML_CONNECT_HELPER______"
    assert len(fetch.methods) == 1
    method = fetch.methods[0]
    assert method.method == case["method"]
    assert method.args[0].input == _relation(frame)
    assert [arg.param for arg in method.args[1:]] == [
        pb2.Expression.Literal.FromString(base64.b64decode(v)) for v in case["args_b64"]
    ]
    assert _requests_but_deletes(service) == sent


@pytest.mark.parametrize("case", FIXTURE["summaries"])
def test_all_summary_expressions_match_reference(case):
    actual = summary(case)._expr.to_plan(None)
    expected = pb2.Expression.FromString(base64.b64decode(case["expr_b64"]))
    assert clean(actual) == clean(expected)


def test_pandas_rejected_before_argument_conversion_or_rpc(session):
    import pandas as pd

    _, service = session
    stat = api()
    for frame in [pd.DataFrame(), pd.DataFrame({"features": [[1.0, 2.0]]})]:
        for operation in [
            lambda: stat.ChiSquareTest.test(frame, object(), object()),
            lambda: stat.Correlation.corr(frame, object()),
            lambda: stat.KolmogorovSmirnovTest.test(frame, object(), object(), object()),
        ]:
            sent = _requests_but_deletes(service)
            with pytest.raises(TypeError, match="requires a Spark DataFrame"):
                operation()
            assert _requests_but_deletes(service) == sent


@pytest.mark.parametrize("metric", FIXTURE["metrics"])
@pytest.mark.parametrize(
    "features,weight", [("features", None), (None, None), (1, None), (F.col("features"), "weight")]
)
def test_static_metrics_require_columns(metric, features, weight):
    with pytest.raises(TypeError, match="featureCol and weightCol should be a Column"):
        getattr(api().Summarizer, metric)(features, weight)


@pytest.mark.parametrize("features,weight", [("features", None), (F.col("features"), "weight")])
def test_builder_requires_columns(features, weight):
    with pytest.raises(AssertionError):
        api().Summarizer.metrics("mean").summary(features, weight)


def test_builder_defers_metric_validation_to_spark():
    for metrics in [[], ["invalid"], ["mean", "mean"], [1], [None]]:
        expression = (
            api().Summarizer.metrics(*metrics).summary(F.col("features"))._expr.to_plan(None)
        )
        assert expression.unresolved_function.function_name == "aggregate_metrics"


def test_ks_float_conversion_errors_are_local(session):
    spark, service = session
    sent = _requests_but_deletes(service)
    with pytest.raises(ValueError, match="could not convert string to float"):
        api().KolmogorovSmirnovTest.test(spark.range(1), "id", "norm", "invalid")
    assert _requests_but_deletes(service) == sent


@pytest.mark.parametrize("sparse", [False, True])
def test_gaussian_is_a_mutable_holder_preserving_object_identity(sparse):
    from pyspark.ml.linalg import DenseMatrix, SparseMatrix, Vectors

    mean = Vectors.sparse(2, {1: 2.0}) if sparse else Vectors.dense([1.0, 2.0])
    cov = (
        SparseMatrix(2, 2, [0, 1, 2], [0, 1], [1.0, 2.0])
        if sparse
        else DenseMatrix(2, 2, [1.0, 0.0, 0.0, 2.0])
    )
    value = api().MultivariateGaussian(mean, cov)
    assert value.mean is mean and value.cov is cov
    replacement = object()
    value.mean, value.cov = replacement, None
    assert value.mean is replacement and value.cov is None
    value.note = "mutable"
    assert value.note == "mutable"
