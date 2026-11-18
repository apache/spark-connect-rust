"""Offline tests for the arguments of a UDF, a pandas UDF and a UDTF call, and for the name of a
UDTF (no Spark Connect server required, a mock one is used).

As PySpark 4.3.0's Spark Connect client: an argument is a Column, a column name (str), or for a
UDTF also the TableArg of a DataFrame's `asTable()`; any other value, a literal included, fails
at the call with an AssertionError that has no message. A UDTF has no `name`, only `_name`.
"""

import copy

import pytest
from _mock_connect_server import mock_connect_session


@pytest.fixture(scope="module")
def udf_session():
    pytest.importorskip("grpc")
    with mock_connect_session() as (spark, service):
        yield spark, service


def _udfs():
    from pyspark.sql import types as T
    from pyspark.sql.functions import udf, udtf

    def plus_one(x):
        return x + 1

    @udtf(returnType="x int")
    class Echo:
        def eval(self, a):
            yield (a,)

    return udf(plus_one, T.IntegerType()), Echo


@pytest.mark.parametrize("value", [3, 2.5, None, [1]])
def test_udf_literal_argument_is_an_assertion_error_at_the_call(udf_session, value):
    # As PySpark's to_expr: a literal is a ColumnReference of it, whose assert fails.
    udf, _ = _udfs()
    with pytest.raises(AssertionError) as error:
        udf(value)
    assert str(error.value) == ""


def test_udf_keyword_literal_argument_is_an_assertion_error_at_the_call(udf_session):
    udf, _ = _udfs()
    with pytest.raises(AssertionError) as error:
        udf(x=3)
    assert str(error.value) == ""


def test_udf_dataframe_argument_is_an_assertion_error(udf_session):
    spark, _ = udf_session
    udf, _ = _udfs()
    with pytest.raises(AssertionError):
        udf(spark.range(2))


def test_udtf_literal_argument_is_an_assertion_error_at_the_call(udf_session):
    _, Echo = _udfs()
    for value in [3, 2.5, None, [1]]:
        with pytest.raises(AssertionError) as error:
            Echo(value)
        assert str(error.value) == ""
    with pytest.raises(AssertionError):
        Echo(a=3)


def test_udtf_table_argument_is_accepted_and_udf_one_is_an_assertion_error(udf_session):
    from pyspark.sql import DataFrame

    spark, _ = udf_session
    udf, Echo = _udfs()
    assert isinstance(Echo(spark.range(2).asTable()), DataFrame)
    with pytest.raises(AssertionError):
        udf(spark.range(2).asTable())


def test_udf_column_and_column_name_arguments_are_accepted(udf_session):
    from pyspark.sql import Column, DataFrame
    from pyspark.sql import functions as F

    udf, Echo = _udfs()
    assert isinstance(udf(F.col("id")), Column)
    assert isinstance(udf("id"), Column)
    assert isinstance(udf(x="id"), Column)
    assert isinstance(udf(), Column)
    assert isinstance(Echo("id"), DataFrame)
    assert isinstance(Echo(F.col("id")), DataFrame)
    assert isinstance(Echo(a=F.lit(3)), DataFrame)


def test_udf_names_and_keywords_that_utf8_cannot_encode_fail_when_the_plan_is_built(udf_session):
    # As PySpark's UDF and UDTF calls, whose ColumnReference of a column name and
    # NamedArgumentExpression of a keyword set the name and the key in protobuf string fields
    # when a request builds the plan: a str that UTF-8 cannot encode makes the Column or the
    # DataFrame, whose requests fail with the UnicodeEncodeError of its encoding, and are not sent.
    from pyspark.sql import functions as F

    spark, service = udf_session
    udf, Echo = _udfs()
    bad = "a\ud800b"
    message = "'utf-8' codec can't encode character '\\ud800' in position 1: surrogates not allowed"
    sent = len(service.execute_requests), len(service.analyze_requests)
    for plan in (
        spark.range(1).select(udf(bad)),
        spark.range(1).select(udf(**{bad: F.col("id")})),
        Echo(bad),
        Echo(a=bad),
        Echo(**{bad: F.lit(1)}),
    ):
        for request in (lambda: plan.schema, plan.collect):
            with pytest.raises(UnicodeEncodeError) as error:
                request()
            assert str(error.value) == message
    assert (len(service.execute_requests), len(service.analyze_requests)) == sent


def test_udtf_has_no_name_until_it_is_set(udf_session):
    from pyspark.sql.connect.udtf import UserDefinedTableFunction, udtf

    @udtf(returnType="x int")
    class Named:
        def eval(self, a):
            yield (a,)

    assert not hasattr(Named, "name")
    with pytest.raises(AttributeError):
        Named.name
    with pytest.raises(AttributeError):
        copy.deepcopy(Named).name
    assert Named._name == "Named"
    assert "name" not in vars(Named)

    Named.name = "renamed"
    assert Named.name == "renamed"
    assert Named._name == "Named"
    assert copy.deepcopy(Named).name == "renamed"
    assert isinstance(Named, UserDefinedTableFunction)


def test_udtf_name_setter_and_pickling_are_unchanged(udf_session):
    import cloudpickle
    from pyspark.sql.connect.udtf import udtf

    @udtf(returnType="x int")
    class Pickled:
        def eval(self, a):
            yield (a,)

    restored = cloudpickle.loads(cloudpickle.dumps(Pickled))
    assert restored._name == "Pickled"
    assert not hasattr(restored, "name")
    restored.name = "other"
    assert restored.name == "other" and restored._name == "Pickled"


def test_the_classic_names_are_pysparks_classic_classes(udf_session):
    # As PySpark's, see docs/compatibility.md: the UDFs and the UDTFs of udf() and udtf() are of
    # PySpark Connect's classes, and the classic names of pyspark.sql.udf and pyspark.sql.udtf
    # are PySpark's classic classes, which are not their bases.
    import pyspark.sql.udf as classic_udf
    import pyspark.sql.udtf as classic_udtf
    from pyspark.sql.connect.udf import UserDefinedFunction
    from pyspark.sql.connect.udtf import UserDefinedTableFunction
    from pyspark.sql.functions import udf, udtf

    @udtf(returnType="x int")
    class Classic:
        def eval(self, a):
            yield (a,)

    assert type(Classic) is UserDefinedTableFunction
    assert not isinstance(Classic, classic_udtf.UserDefinedTableFunction)
    # A UDF of udf() is its function wrapper, which keeps the UserDefinedFunction it wraps.
    assert type(udf(lambda x: x)._unwrapped) is UserDefinedFunction
    assert not isinstance(udf(lambda x: x)._unwrapped, classic_udf.UserDefinedFunction)


def test_udf_and_udtf_raise_an_error_of_the_session_other_than_no_session_as_pyspark(
    udf_session, monkeypatch
):
    # As PySpark Connect's udf() and udtf() without useArrow, which read the Arrow configuration
    # of the active session: without a session they make a UDF and a UDTF that are not
    # Arrow-optimized, and another error of the session is raised.
    from pyspark.errors import PySparkRuntimeError
    from pyspark.sql import SparkSession
    from pyspark.sql.functions import udf, udtf

    class Echo:
        def eval(self, a):
            yield (a,)

    def raising(condition):
        def active():
            raise PySparkRuntimeError(errorClass=condition, messageParameters={})

        return staticmethod(active)

    monkeypatch.setattr(SparkSession, "active", raising("NO_ACTIVE_OR_DEFAULT_SESSION"))
    assert udf(lambda x: x).evalType == 100  # SQL_BATCHED_UDF
    assert udtf(Echo, returnType="a int").evalType == 300  # SQL_TABLE_UDF
    monkeypatch.setattr(SparkSession, "active", raising("SESSION_OR_CONTEXT_NOT_EXISTS"))
    for make in (lambda: udf(lambda x: x), lambda: udtf(Echo, returnType="a int")):
        with pytest.raises(PySparkRuntimeError) as error:
            make()
        assert error.value.getCondition() == "SESSION_OR_CONTEXT_NOT_EXISTS"


def test_a_udtf_of_no_return_type_sends_no_return_type_as_pyspark(udf_session):
    # As PySpark Connect's PythonUDTF.to_plan: the PythonUDTF of a UDTF of no return type, whose
    # analyze gives the schema, has no return_type, and the command of the handler.
    from pyspark.serializers import CloudPickleSerializer
    from pyspark.sql import types as T
    from pyspark.sql.connect.proto import base_pb2
    from pyspark.sql.functions import lit, udtf
    from pyspark.sql.udtf import AnalyzeResult

    _, service = udf_session

    class Dynamic:
        @staticmethod
        def analyze(a):
            return AnalyzeResult(T.StructType().add("a", "int"))

        def eval(self, a):
            yield (a,)

    udtf(Dynamic, useArrow=False)(lit(1)).explain()
    request = base_pb2.AnalyzePlanRequest.FromString(service.analyze_requests[-1])
    python_udtf = request.explain.plan.root.common_inline_user_defined_table_function.python_udtf
    assert not python_udtf.HasField("return_type")
    assert CloudPickleSerializer().loads(python_udtf.command).__name__ == "Dynamic"
