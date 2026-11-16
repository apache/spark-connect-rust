# SPDX-License-Identifier: Apache-2.0
"""The classes of the objects of a session as official PySpark 4.3.0's, without a server: of the
modules of PySpark Connect, `pyspark.sql.connect.*`, which `repr`, `type(x).__module__` and pickle
show. The expected names are official PySpark's, of the same objects."""

import pickle

import pytest
from _mock_connect_server import mock_connect_session
from pyspark.sql import Window
from pyspark.sql import functions as F

# The module and the qualified name of the class of each object, in official PySpark.
EXPECTED = {
    "spark": "pyspark.sql.connect.session.SparkSession",
    "conf": "pyspark.sql.connect.conf.RuntimeConf",
    "catalog": "pyspark.sql.connect.catalog.Catalog",
    "read": "pyspark.sql.connect.readwriter.DataFrameReader",
    "readStream": "pyspark.sql.connect.streaming.readwriter.DataStreamReader",
    "streams": "pyspark.sql.connect.streaming.query.StreamingQueryManager",
    "udf": "pyspark.sql.connect.udf.UDFRegistration",
    "udtf": "pyspark.sql.connect.udtf.UDTFRegistration",
    "tvf": "pyspark.sql.connect.tvf.TableValuedFunction",
    "client": "pyspark.sql.connect.client.core.SparkConnectClient",
    "df": "pyspark.sql.connect.dataframe.DataFrame",
    "write": "pyspark.sql.connect.readwriter.DataFrameWriter",
    "writeTo": "pyspark.sql.connect.readwriter.DataFrameWriterV2",
    "writeStream": "pyspark.sql.connect.streaming.readwriter.DataStreamWriter",
    "mergeInto": "pyspark.sql.connect.merge.MergeIntoWriter",
    "whenMatched": "pyspark.sql.connect.merge.MergeIntoWriter.WhenMatched",
    "whenNotMatched": "pyspark.sql.connect.merge.MergeIntoWriter.WhenNotMatched",
    "whenNotMatchedBySource": "pyspark.sql.connect.merge.MergeIntoWriter.WhenNotMatchedBySource",
    "na": "pyspark.sql.connect.dataframe.DataFrameNaFunctions",
    "stat": "pyspark.sql.connect.dataframe.DataFrameStatFunctions",
    "groupBy": "pyspark.sql.connect.group.GroupedData",
    "column": "pyspark.sql.connect.column.Column",
    "window": "pyspark.sql.connect.window.WindowSpec",
    "asTable": "pyspark.sql.connect.table_arg.TableArg",
}


def _objects(spark):
    df = spark.range(1)
    merge = df.mergeInto("t", F.lit(True))
    return {
        "spark": spark,
        "conf": spark.conf,
        "catalog": spark.catalog,
        "read": spark.read,
        "readStream": spark.readStream,
        "streams": spark.streams,
        "udf": spark.udf,
        "udtf": spark.udtf,
        "tvf": spark.tvf,
        "client": spark.client,
        "df": df,
        "write": df.write,
        "writeTo": df.writeTo("t"),
        "writeStream": df.writeStream,
        "mergeInto": merge,
        "whenMatched": merge.whenMatched(),
        "whenNotMatched": merge.whenNotMatched(),
        "whenNotMatchedBySource": merge.whenNotMatchedBySource(),
        "na": df.na,
        "stat": df.stat,
        "groupBy": df.groupBy(),
        "column": F.col("id"),
        "window": Window.partitionBy("id"),
        "asTable": df.asTable(),
    }


@pytest.fixture(scope="module")
def objects():
    with mock_connect_session() as (spark, _service):
        yield _objects(spark)


@pytest.mark.parametrize("name", list(EXPECTED))
def test_the_class_of_an_object_of_a_session_is_pysparks(objects, name):
    cls = type(objects[name])
    assert f"{cls.__module__}.{cls.__qualname__}" == EXPECTED[name]
    # Pickled by reference to that name, which names the same class.
    assert pickle.loads(pickle.dumps(cls)) is cls


def test_the_repr_of_a_session_names_its_connect_class(objects):
    assert repr(objects["spark"]).startswith("<pyspark.sql.connect.session.SparkSession object at ")


def test_the_streaming_query_class_is_pysparks():
    from pyspark.sql.connect.streaming.query import StreamingQuery
    from pyspark.sql.streaming import StreamingQuery as ClassicName

    assert (StreamingQuery.__module__, StreamingQuery.__qualname__) == (
        "pyspark.sql.connect.streaming.query",
        "StreamingQuery",
    )
    assert ClassicName is StreamingQuery
    assert pickle.loads(pickle.dumps(StreamingQuery)) is StreamingQuery


def test_the_repr_of_a_table_argument_is_pysparks(objects):
    # Of no `__repr__` of its own, as PySpark's TableArg.
    assert repr(objects["asTable"]).startswith(
        "<pyspark.sql.connect.table_arg.TableArg object at 0x"
    )


def test_cpythons_messages_name_a_class_by_its_bare_name_as_pysparks():
    # The C name of a class, which these messages give, is its `__name__`, as of PySpark's
    # Python classes, not the dotted path of its module.
    from pyspark.sql.types import IntegerType

    with pytest.raises(TypeError) as raised:
        True ^ F.col("a")
    assert str(raised.value) == "unsupported operand type(s) for ^: 'bool' and 'Column'"
    with pytest.raises(AttributeError) as raised:
        IntegerType().foo
    assert str(raised.value) == "'IntegerType' object has no attribute 'foo'"
    # The module and the qualified name stay those of PySpark Connect, in the repr and the pickle.
    cls = type(F.col("a"))
    assert repr(cls) == "<class 'pyspark.sql.connect.column.Column'>"
    assert pickle.loads(pickle.dumps(cls)) is cls


def test_every_native_class_has_its_bare_name_as_its_c_name():
    # `object.__new__(cls)` refuses a native class, in a message of its C name. It makes an
    # object of a Python class of the module, such as VariantVal, whose C name is its name.
    import pyspark._pyspark as native

    checked = 0
    for cls in vars(native).values():
        if not isinstance(cls, type) or issubclass(cls, BaseException):
            continue
        try:
            object.__new__(cls)
        except TypeError as error:
            assert f"{cls.__module__}.{cls.__name__}" not in str(error), str(error)
            checked += 1
    assert checked > 50
