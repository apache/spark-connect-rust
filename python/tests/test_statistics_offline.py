# SPDX-License-Identifier: Apache-2.0
"""Native statistics API, reference protobufs and validation without a Spark server."""

import base64
import inspect

import pytest
from _mock_connect_server import mock_connect_session
from _statistics_cases import (
    ARGUMENTS,
    CLASSES,
    FIXTURE,
    api,
    argument_case_id,
    call_with_arguments,
    first_line,
    helper,
    summary,
)
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
    for name in CLASSES:
        assert getattr(stat, name) is getattr(public, name)
        if name != "MultivariateGaussian":
            assert issubclass(getattr(stat, name), getattr(_pyspark, name))
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


def test_the_module_has_pysparks_names_and_no_all():
    # PySpark's module has no `__all__`, so `from pyspark.ml.stat import *` imports its public
    # names: its classes and the names of its imports, of the objects of those modules.
    import pyspark.ml.stat as public
    from pyspark.ml.wrapper import JavaWrapper
    from pyspark.sql.column import Column

    assert not hasattr(public, "__all__")
    namespace = {}
    exec("from pyspark.ml.stat import *", namespace)  # noqa: S102
    assert sorted(name for name in namespace if name != "__builtins__") == [
        "ArrayType", "ChiSquareTest", "Column", "Correlation", "DataFrame", "DoubleType",
        "JavaWrapper", "KolmogorovSmirnovTest", "Matrix", "MultivariateGaussian", "Optional",
        "Summarizer", "SummaryBuilder", "TYPE_CHECKING", "Tuple", "Vector",
        "invoke_helper_relation", "is_remote", "lit", "since", "sys",
    ]
    assert namespace["JavaWrapper"] is JavaWrapper and namespace["Column"] is Column
    assert all(namespace[name] is getattr(public, name) for name in CLASSES)


def test_the_statistics_classes_construct_as_pysparks_plain_classes(session):
    # PySpark's ChiSquareTest, Correlation, KolmogorovSmirnovTest and Summarizer are plain classes
    # of static methods: an object, and one of a subclass, are made as `object.__new__` makes
    # them, of no arguments unless the subclass has an `__init__` of its own, they take
    # attributes and weak references, they pickle and copy, and their methods are those of the
    # class.
    import copy
    import pickle
    import weakref

    spark, _ = session
    stat = api()
    for name in ["ChiSquareTest", "Correlation", "KolmogorovSmirnovTest", "Summarizer"]:
        cls = getattr(stat, name)
        assert type(cls()) is cls and str(inspect.signature(cls)) == "()"
        for call in (lambda: cls(1), lambda: cls(a=1)):
            with pytest.raises(TypeError, match=f"^{name}\\(\\) takes no arguments$"):
                call()
        Sub = type("Sub", (cls,), {})
        assert type(Sub()) is Sub
        with pytest.raises(TypeError, match="^Sub\\(\\) takes no arguments$"):
            Sub(1)
        WithInit = type("WithInit", (cls,), {"__init__": lambda self, x: setattr(self, "x", x)})
        assert WithInit(1).x == 1
        obj = cls()
        obj.tag = [1]
        assert weakref.ref(obj)() is obj
        for copied in (copy.copy(obj), copy.deepcopy(obj), pickle.loads(pickle.dumps(obj))):
            assert type(copied) is cls and vars(copied) == {"tag": [1]}
    frame = spark.range(4)
    instance = stat.ChiSquareTest().test(frame, "features", "label")
    expected = stat.ChiSquareTest.test(frame, "features", "label")
    assert _relation(instance).ml_relation == _relation(expected).ml_relation
    assert str(stat.Summarizer().mean(F.col("f"))) == str(stat.Summarizer.mean(F.col("f")))


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


@pytest.mark.parametrize("case", ARGUMENTS["cases"], ids=argument_case_id)
def test_helper_arguments_are_serialized_as_pyspark(session, case):
    # As official PySpark serializes the arguments of the helper calls, in their order: an argument
    # that cannot be a literal raises the error of its conversion, before any request, and a
    # literal of another type than the helper takes is sent as PySpark sends it, in a lazy
    # DataFrame, for the server to reject when it is collected (test_statistics_e2e.py).
    spark, service = session
    frame = spark.range(1)
    sent = _requests_but_deletes(service)
    if "error" in case:
        with pytest.raises(Exception) as error:
            call_with_arguments(case, frame)
        assert type(error.value).__name__ == case["error"]["class"]
        assert first_line(error.value) == case["error"]["message"]
    else:
        fetch = _relation(call_with_arguments(case, frame)).ml_relation.fetch
        assert fetch.obj_ref.id == "______ML_CONNECT_HELPER______"
        (method,) = fetch.methods
        assert method.method == case["method"]
        assert len(method.args) == len(case["sent"])
        for arg, expected in zip(method.args, case["sent"]):
            if expected == {"input": True}:
                assert arg.input == _relation(frame)
            else:
                literal = base64.b64decode(expected["literal"])
                assert arg.param == pb2.Expression.Literal.FromString(literal)
    assert _requests_but_deletes(service) == sent


def test_a_typed_pair_argument_is_sent_as_the_literal_of_its_type(session):
    # PySpark's serialize sends a (value, DataType) pair as LiteralExpression(value, dataType), the
    # literal of that type, in the lazy DataFrame of the helper call, and a value that is not of the
    # type raises the bare AssertionError of that constructor, before any request
    # (tests/golden/ml_fetch_pairs.json has more pairs of official PySpark).
    from pyspark.sql.types import IntegerType, StringType

    spark, service = session
    stat = api()
    frame = spark.range(1)
    sent = _requests_but_deletes(service)
    for operation, index, literal in [
        (
            lambda: stat.ChiSquareTest.test((1, IntegerType()), "features", "label"),
            0,
            pb2.Expression.Literal(integer=1),
        ),
        (
            lambda: stat.ChiSquareTest.test(frame, ("features", StringType()), "label"),
            1,
            pb2.Expression.Literal(string="features"),
        ),
        (
            lambda: stat.Correlation.corr(frame, "features", ("pearson", StringType())),
            2,
            pb2.Expression.Literal(string="pearson"),
        ),
    ]:
        (method,) = _relation(operation()).ml_relation.fetch.methods
        assert method.args[index].param == literal
    with pytest.raises(AssertionError):
        stat.ChiSquareTest.test(frame, ("x", IntegerType()), "label")
    assert _requests_but_deletes(service) == sent


@pytest.mark.parametrize("metric", FIXTURE["metrics"])
@pytest.mark.parametrize(
    "features,weight", [("features", None), (None, None), (1, None), (F.col("features"), "weight")]
)
def test_static_metrics_require_columns(metric, features, weight):
    with pytest.raises(TypeError, match="featureCol and weightCol should be a Column"):
        getattr(api().Summarizer, metric)(features, weight)


def test_a_summary_builder_constructs_and_summarizes_as_pysparks():
    # PySpark's SummaryBuilder(jSummaryBuilder), a JavaWrapper, keeps no Java object over Spark
    # Connect: its `summary` reads the `_metrics` of its attributes, which `Summarizer.metrics` sets
    # with a `_java_obj` of None, so that a builder of its constructor has none (an AttributeError),
    # and those of a subclass and the copies of a builder summarize of theirs.
    import copy
    import pickle

    from pyspark.ml.wrapper import JavaWrapper

    stat = api()
    SummaryBuilder = stat.SummaryBuilder

    def expression(column):
        return clean(column._expr.to_plan(None))

    assert str(inspect.signature(SummaryBuilder)) == "(jSummaryBuilder: 'JavaObject')"
    builder = SummaryBuilder(None)
    assert isinstance(builder, JavaWrapper) and vars(builder) == {}
    assert vars(SummaryBuilder(jSummaryBuilder=None)) == {}
    for call, message in [
        (SummaryBuilder, "missing 1 required positional argument: 'jSummaryBuilder'"),
        (lambda: SummaryBuilder(None, 1), "takes 2 positional arguments but 3 were given"),
        (lambda: SummaryBuilder(java_obj=None), "got an unexpected keyword argument 'java_obj'"),
    ]:
        with pytest.raises(TypeError, match=f"^SummaryBuilder.__init__\\(\\) {message}$"):
            call()
    with pytest.raises(AttributeError, match="^'SummaryBuilder' object has no attribute '_metrics'$"):
        builder.summary(F.col("f"))
    metrics = stat.Summarizer.metrics("mean", "count")
    assert type(metrics) is SummaryBuilder
    assert vars(metrics) == {"_metrics": ["mean", "count"], "_java_obj": None}
    expected = expression(stat.Summarizer.metrics("max").summary(F.col("f")))
    builder._metrics = ["max"]
    assert expression(builder.summary(F.col("f"))) == expected
    for copied in (copy.copy(metrics), copy.deepcopy(metrics), pickle.loads(pickle.dumps(metrics))):
        assert type(copied) is SummaryBuilder and vars(copied) == vars(metrics)

    class Builder(SummaryBuilder):
        def __init__(self):
            super().__init__(None)
            self._metrics = ["max"]

    assert expression(Builder().summary(F.col("f"))) == expected
    with pytest.raises(AttributeError, match="^'Sub' object has no attribute '_metrics'$"):
        type("Sub", (SummaryBuilder,), {})(None).summary(F.col("f"))


def test_the_classes_and_static_methods_have_pysparks_docstrings_signatures_and_annotations():
    # PySpark's classes are documented, and their static methods are functions of PySpark's
    # docstrings (of `since` of Summarizer's metrics), signatures and annotations, which `inspect`
    # and `help` give, also Summarizer's `_check_param` and `_get_single_metric`, PySpark's code
    # over Spark Connect. (The drop-in's `Column`, of the annotations, is the class of
    # `pyspark.sql.column.Column`, which is `pyspark.sql.connect.column.Column`.)
    stat = api()
    column = f"{stat.Column.__module__}.{stat.Column.__qualname__}"
    frame = "pyspark.sql.dataframe.DataFrame"
    docs = {
        "ChiSquareTest": (
            "Conduct Pearson's independence test for every feature against the label. For each "
            "feature,"
        ),
        "Correlation": (
            "Compute the correlation matrix for the input dataset of Vectors using the specified "
            "method."
        ),
        "KolmogorovSmirnovTest": (
            "Conduct the two-sided Kolmogorov Smirnov (KS) test for data sampled from a continuous"
        ),
        "Summarizer": "Tools for vectorized statistics on MLlib Vectors.",
    }
    for name, first in docs.items():
        assert inspect.getdoc(getattr(stat, name)).splitlines()[0] == first, name
    signatures = {
        stat.ChiSquareTest.test: (
            f"(dataset: {frame}, featuresCol: str, labelCol: str, flatten: bool = False) "
            f"-> {frame}",
            "Perform a Pearson's independence test using dataset.",
        ),
        stat.Correlation.corr: (
            f"(dataset: {frame}, column: str, method: str = 'pearson') -> {frame}",
            "Compute the correlation matrix with specified method using dataset.",
        ),
        stat.KolmogorovSmirnovTest.test: (
            f"(dataset: {frame}, sampleCol: str, distName: str, *params: float) -> {frame}",
            "Conduct a one-sample, two-sided Kolmogorov-Smirnov test for probability distribution",
        ),
        stat.Summarizer.metrics: (
            "(*metrics: str) -> 'SummaryBuilder'",
            "Given a list of metrics, provides a builder that it turns computes metrics from a "
            "column.",
        ),
    }
    for metric, version in [
        ("mean", "2.4.0"), ("sum", "3.0.0"), ("variance", "2.4.0"), ("std", "3.0.0"),
        ("count", "2.4.0"), ("numNonZeros", "2.4.0"), ("max", "2.4.0"), ("min", "2.4.0"),
        ("normL1", "2.4.0"), ("normL2", "2.4.0"),
    ]:
        function = getattr(stat.Summarizer, metric)
        summary = "numNonZero" if metric == "numNonZeros" else metric
        signatures[function] = (
            f"(col: {column}, weightCol: Optional[{column}] = None) -> {column}",
            f"return a column of {summary} summary",
        )
        assert inspect.getdoc(function).splitlines()[-1] == f".. versionadded:: {version}", metric
    for function, (signature, first) in signatures.items():
        assert str(inspect.signature(function)) == signature, function
        assert inspect.getdoc(function).splitlines()[0] == first, function
        assert set(function.__annotations__) == set(inspect.signature(function).parameters) | {
            "return"
        }, function
    assert str(inspect.signature(stat.Summarizer._check_param)) == (
        f"(featuresCol: {column}, weightCol: Optional[{column}]) -> Tuple[{column}, {column}]"
    )
    assert str(inspect.signature(stat.Summarizer._get_single_metric)) == (
        f"(col: {column}, weightCol: Optional[{column}], metric: str) -> {column}"
    )
    # PySpark's `_check_param` and `_get_single_metric`, of the column of the metric's method.
    features, weight = F.col("f"), F.col("w")
    assert stat.Summarizer._check_param(features, weight) == (features, weight)
    with pytest.raises(TypeError, match="^featureCol and weightCol should be a Column$"):
        stat.Summarizer._check_param("f", None)
    single = stat.Summarizer._get_single_metric(features, None, "mean")
    assert str(single) == str(stat.Summarizer.mean(features))


def test_the_methods_are_of_pysparks_module_and_summary_binds_as_its_function():
    # PySpark's static methods are functions of pyspark.ml.stat (`Summarizer.mean.__module__`), and
    # `SummaryBuilder.summary` (stat.py:504) is a function of its module, of PySpark's parameters
    # and docstring, of which Python binds a call as it binds PySpark's: its messages count `self`.
    import sys

    stat = api()
    for name in ["ChiSquareTest", "Correlation", "KolmogorovSmirnovTest", "Summarizer"]:
        cls = getattr(stat, name)
        statics = [n for n, v in vars(cls).items() if isinstance(v, staticmethod)]
        assert statics, name
        for static in statics:
            function = getattr(cls, static)
            assert function.__module__ == "pyspark.ml.stat", (name, static)
            assert function.__qualname__ == f"{name}.{static}", (name, static)
    summary = stat.SummaryBuilder.summary
    assert inspect.isfunction(summary) and summary.__module__ == "pyspark.ml.stat"
    assert summary.__qualname__ == "SummaryBuilder.summary"
    assert inspect.getdoc(summary).splitlines()[:2] == [
        "Returns an aggregate object that contains the summary of the column with the requested",
        "metrics.",
    ]
    assert list(inspect.signature(summary).parameters) == ["self", "featuresCol", "weightCol"]
    builder = stat.Summarizer.metrics("mean")
    expected = clean(builder.summary(F.col("f"), F.col("w"))._expr.to_plan(None))
    keywords = builder.summary(featuresCol=F.col("f"), weightCol=F.col("w"))
    assert clean(keywords._expr.to_plan(None)) == expected
    cases = [
        (lambda: builder.summary(), "missing 1 required positional argument: 'featuresCol'"),
        (
            lambda: builder.summary(F.col("f"), F.col("w"), 1),
            "takes from 2 to 3 positional arguments but 4 were given",
        ),
        (lambda: builder.summary(F.col("f"), x=1), "got an unexpected keyword argument 'x'"),
    ]
    if sys.version_info < (3, 10):
        return
    for call, message in cases:
        with pytest.raises(TypeError) as error:
            call()
        assert str(error.value) == f"SummaryBuilder.summary() {message}"


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
