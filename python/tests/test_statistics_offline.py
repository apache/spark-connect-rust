# SPDX-License-Identifier: Apache-2.0
"""Native statistics API, reference protobufs and validation without a Spark server."""

import base64
import inspect

import pytest
from _mock_connect_server import mock_connect_session
from _statistics_cases import FIXTURE, REFERENCE, api, helper, summary
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
    import pyspark.ml.stat as public

    stat = api()
    for name in stat.__all__:
        assert getattr(stat, name) is getattr(public, name)
        if name != "MultivariateGaussian":
            assert getattr(stat, name) is getattr(_pyspark, name)
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
    # As pyspark.ml.stat's, whose helper call serializes the dataset first, as a literal unless it
    # is a DataFrame: a pandas DataFrame raises the UNSUPPORTED_DATA_TYPE of that conversion,
    # before the other arguments, except the params of KolmogorovSmirnovTest, which PySpark
    # converts with float() before the call.
    import pandas as pd
    from pyspark.errors import PySparkTypeError

    _, service = session
    stat = api()
    for frame in [pd.DataFrame(), pd.DataFrame({"features": [[1.0, 2.0]]})]:
        for operation in [
            lambda: stat.ChiSquareTest.test(frame, object(), object()),
            lambda: stat.Correlation.corr(frame, object()),
            lambda: stat.KolmogorovSmirnovTest.test(frame, object(), object(), 1.0),
        ]:
            sent = _requests_but_deletes(service)
            with pytest.raises(PySparkTypeError) as error:
                operation()
            assert error.value.getCondition() == "UNSUPPORTED_DATA_TYPE"
            assert error.value.getMessageParameters() == {"data_type": "DataFrame"}
            assert _requests_but_deletes(service) == sent
        with pytest.raises(TypeError) as error:
            stat.KolmogorovSmirnovTest.test(frame, object(), object(), object())
        assert str(error.value) == "float() argument must be a string or a real number, not 'object'"
        assert _requests_but_deletes(service) == sent


def test_helper_arguments_are_serialized_as_pyspark(session):
    # As PySpark's serialization of the arguments of the helper calls, in their order: an argument
    # that cannot be a literal raises the error of its conversion, and a literal of another type
    # than PySpark's, which PySpark sends for the server to reject, raises a TypeError here, before
    # any request, as of official pyspark-client 4.3.0.
    from pyspark.errors import PySparkTypeError

    spark, service = session
    stat = api()
    df = spark.range(1)
    sent = _requests_but_deletes(service)
    unconvertible = [
        (lambda: stat.ChiSquareTest.test(df, object(), "label"), "UNSUPPORTED_DATA_TYPE"),
        (lambda: stat.ChiSquareTest.test(df, "features", object()), "UNSUPPORTED_DATA_TYPE"),
        (lambda: stat.Correlation.corr(df, object()), "UNSUPPORTED_DATA_TYPE"),
        (lambda: stat.KolmogorovSmirnovTest.test(df, object(), "norm"), "UNSUPPORTED_DATA_TYPE"),
        (lambda: stat.KolmogorovSmirnovTest.test(df, "id", {"a": 1}), "UNSUPPORTED_DATA_TYPE"),
        (lambda: stat.ChiSquareTest.test([], "features", "label"),
         "CANNOT_INFER_ARRAY_ELEMENT_TYPE"),
    ]
    for operation, condition in unconvertible:
        with pytest.raises(PySparkTypeError) as error:
            operation()
        assert error.value.getCondition() == condition
    assert _requests_but_deletes(service) == sent
    if REFERENCE:
        # PySpark sends these, for the server to reject when the result is analyzed.
        return
    refused = [
        (lambda: stat.ChiSquareTest.test(df, 1, "label"), "test() takes a str featuresCol, not 'int'"),
        (lambda: stat.Correlation.corr(df, None), "corr() takes a str column, not 'NoneType'"),
        (lambda: stat.KolmogorovSmirnovTest.test(df, "id", 1), "test() takes a str distName, not 'int'"),
        (lambda: stat.ChiSquareTest.test(5, "features", "label"),
         "test() takes a Spark DataFrame, not 'int'"),
    ]
    for operation, message in refused:
        with pytest.raises(TypeError) as error:
            operation()
        assert str(error.value) == message
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


def test_builder_converts_metrics_with_lit_as_pyspark():
    # As pyspark.ml.stat's SummaryBuilder.summary, which converts each metric with F.lit: a list or
    # a NumPy array is the array of lit, not a typed literal, so that an empty list is an empty
    # array. The expressions are those of official pyspark-client 4.3.0.
    import numpy as np

    expected = {
        "list": ([["mean"]], "GkkKEWFnZ3JlZ2F0ZV9tZXRyaWNzEh4aHAoFYXJyYXkSExoRCgVhcnJheRIICgZqBG1lYW4SBxIFCgFmGAASCwoJWQAAAAAAAPA/"),
        "ndarray": ([np.array(["mean"])], "GmUKEWFnZ3JlZ2F0ZV9tZXRyaWNzEjoaOAoFYXJyYXkSLzotChMaEQoFYXJyYXkSCAoGagRtZWFuEhaiARMKD1oNEgtVVEY4X0JJTkFSWRABEgcSBQoBZhgAEgsKCVkAAAAAAADwPw=="),
        "nested_list": ([[1, 2]], "GksKEWFnZ3JlZ2F0ZV9tZXRyaWNzEiAaHgoFYXJyYXkSFRoTCgVhcnJheRIECgIwARIECgIwAhIHEgUKAWYYABILCglZAAAAAAAA8D8="),
        "empty_list": ([[]], "Gj8KEWFnZ3JlZ2F0ZV9tZXRyaWNzEhQaEgoFYXJyYXkSCRoHCgVhcnJheRIHEgUKAWYYABILCglZAAAAAAAA8D8="),
        "column": ([F.lit("mean")], "Gj4KEWFnZ3JlZ2F0ZV9tZXRyaWNzEhMaEQoFYXJyYXkSCAoGagRtZWFuEgcSBQoBZhgAEgsKCVkAAAAAAADwPw=="),
    }
    for name, (metrics, encoded) in expected.items():
        actual = api().Summarizer.metrics(*metrics).summary(F.col("f"))._expr.to_plan(None)
        reference = pb2.Expression.FromString(base64.b64decode(encoded))
        assert clean(actual) == clean(reference), name


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


def test_gaussian_is_pysparks_plain_class_and_pickles_as_it():
    # PySpark's holder is a plain class of pyspark.ml.stat, whose instances pickle by its module
    # path, as into the Python workers of a server.
    import pickle

    from pyspark.ml.linalg import DenseMatrix, Vectors

    cls = api().MultivariateGaussian
    assert (cls.__module__, cls.__qualname__) == ("pyspark.ml.stat", "MultivariateGaussian")
    value = cls(Vectors.dense([1.0]), DenseMatrix(1, 1, [2.0]))
    assert sorted(vars(value)) == ["cov", "mean"]
    copy = pickle.loads(pickle.dumps(value))
    assert type(copy) is cls
    assert (copy.mean, copy.cov) == (value.mean, value.cov)
