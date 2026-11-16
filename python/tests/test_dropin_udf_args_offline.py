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


def test_udtf_has_no_name_until_it_is_set(udf_session):
    from pyspark.sql.udtf import UserDefinedTableFunction, udtf

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
    from pyspark.sql.udtf import udtf

    @udtf(returnType="x int")
    class Pickled:
        def eval(self, a):
            yield (a,)

    restored = cloudpickle.loads(cloudpickle.dumps(Pickled))
    assert restored._name == "Pickled"
    assert not hasattr(restored, "name")
    restored.name = "other"
    assert restored.name == "other" and restored._name == "Pickled"


def test_classic_names_are_the_connect_classes(udf_session):
    # Deliberate, see docs/compatibility.md: the classic names are the drop-in's own classes.
    from pyspark.sql.udf import UserDefinedFunction, udf
    from pyspark.sql.udtf import UserDefinedTableFunction, udtf

    @udtf(returnType="x int")
    class Classic:
        def eval(self, a):
            yield (a,)

    assert type(Classic) is UserDefinedTableFunction
    # A UDF of udf() is its function wrapper, which keeps the UserDefinedFunction it wraps.
    assert type(udf(lambda x: x)._unwrapped) is UserDefinedFunction
