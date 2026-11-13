#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Drop-in behaviors that need a live Spark Connect server. The official tests run the
reference pyspark over our transport rather than these Rust-backed classes, so the expected
values here are those of the reference client; the tests pass with it as well.
"""

import os
import time
from collections import namedtuple
from datetime import date, datetime
from decimal import Decimal

import pyarrow as pa
import pytest
from _native_classes import native_class

# Keep interval round trips in the live suite selected by CI.
from interval_input_cases import IntervalInputLiveTests  # noqa: F401
from pyspark.errors import AnalysisException, PySparkValueError
from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    MapType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)


@pytest.fixture
def session():
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live-server tests")
    session = SparkSession.builder.remote(remote).getOrCreate()
    try:
        yield session
    finally:
        session.stop()


@pytest.mark.parametrize("prefetch", [False, True])
def test_to_local_iterator_raises_the_error_of_the_execution_as_pyspark(session, prefetch):
    # As official PySpark 4.3.0: the error of a query that fails while it is iterated is
    # raised by the iteration, which then ends, rather than ending it as if complete.
    df = session.sql("SELECT assert_true(id < 0) AS x FROM range(3)")
    rows = df.toLocalIterator(prefetch)
    with pytest.raises(Exception) as error:
        next(rows)
    assert error.value.getCondition() == "USER_RAISED_EXCEPTION"
    assert list(rows) == []


def test_pandas_api_takes_its_index_and_plot_is_of_pyspark(session):
    # As PySpark's DataFrame.pandas_api(index_col): a pandas-on-Spark DataFrame of the index
    # of the columns of index_col; and DataFrame.plot, the PySparkPlotAccessor of the DataFrame.
    sdf = session.createDataFrame([("a", 1, 2.0), ("b", 2, 3.0)], ["k", "i", "f"])
    psdf = sdf.pandas_api(index_col="k")
    assert psdf.index.name == "k" and list(psdf.columns) == ["i", "f"]
    assert psdf.sort_index().to_pandas()["i"].tolist() == [1, 2]
    assert sdf.pandas_api().index.name is None
    pytest.importorskip("plotly")
    figure = sdf.plot.line(x="k", y=["i", "f"])
    assert type(figure).__name__ == "Figure" and len(figure.data) == 2


def test_udtfs_of_ddl_return_types_are_unparsed_as_pyspark(session):
    # As PySpark Connect's: the returnType of a UDTF of a DDL string is an UnparsedDataType,
    # which the server parses.
    from pyspark.sql.connect.types import UnparsedDataType
    from pyspark.sql.functions import lit, udtf

    @udtf(returnType="x int, y string")
    class Pairs:
        def eval(self, n: int):
            for i in range(n):
                yield i, str(i)

    assert isinstance(Pairs.returnType, UnparsedDataType)
    assert Pairs.returnType.data_type_string == "x int, y string"
    assert [tuple(row) for row in Pairs(lit(2)).collect()] == [(0, "0"), (1, "1")]




def test_list_values_of_column_methods_are_evaluated_by_the_server_as_pyspark(session):
    # As PySpark's LiteralExpression of a list: each element is of the type that the first one
    # gives the list, when the type accepts it, so the server evaluates the array of the value.
    import numpy as np

    df = session.range(1)
    when = F.when(F.lit(False), F.array(F.lit(0.0).cast("float")))
    assert df.select(when.otherwise([np.float32(1.5), 2.0]).alias("r")).collect() == [
        Row(r=[1.5, 2.0])]
    when = F.when(F.lit(False), F.array(F.lit(0.0)))
    assert df.select(when.otherwise([1.5, np.float32(2.0)]).alias("r")).collect() == [
        Row(r=[1.5, 2.0])]
    when = F.when(F.lit(False), F.array(F.lit(date(2000, 1, 1))))
    assert df.select(when.otherwise([date(2020, 1, 1), datetime(2020, 1, 2, 3)]).alias("r")
                     ).collect() == [Row(r=[date(2020, 1, 1), date(2020, 1, 2)])]


def test_udf_and_udtf_calls_take_named_arguments_as_pyspark(session):
    # As PySpark's: the keyword arguments of a UDF or UDTF call are its named arguments, and
    # a bare string is a column name.
    import pandas as pd

    @F.udf("int")
    def plus(a, b):
        return a + 10 * b

    @F.pandas_udf("long")
    def pandas_plus(a: pd.Series, b: pd.Series) -> pd.Series:
        return a + 10 * b

    df = session.range(1)
    for fn in (plus, pandas_plus):
        assert df.select(fn(b=F.lit(2), a=F.lit(1)).alias("r")).collect() == [Row(r=21)]
        assert df.select(fn(F.lit(1), b=F.lit(2)).alias("r")).collect() == [Row(r=21)]

    @F.udtf(returnType="a int, b int")
    class Two:
        def eval(self, a, b):
            yield a, b

    assert Two(b=F.lit(2), a=F.lit(1)).collect() == [Row(a=1, b=2)]
    assert Two(F.lit(1), b=F.lit(2)).collect() == [Row(a=1, b=2)]
    with pytest.raises(AnalysisException) as error:
        Two("x", F.lit(2)).collect()
    assert error.value.getCondition() == "UNRESOLVED_COLUMN.WITHOUT_SUGGESTION"


def test_conf_warns_the_warnings_of_the_server_and_takes_its_values_as_pyspark(session):
    # As PySpark's: set and unset warn the warnings of the server, the server rejects a set of
    # None, a set of a float raises a TypeError, and get checks its key and default.
    import warnings

    from pyspark.errors import IllegalArgumentException, PySparkTypeError

    key = "spark.sql.execution.arrow.enabled"
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        session.conf.set(key, False)
        session.conf.unset(key)
    warning = f"The SQL config '{key}' is NOT supported in Spark Connect"
    assert [(w.category, str(w.message)) for w in caught] == [(UserWarning, warning)] * 2
    with pytest.raises(IllegalArgumentException, match="value cannot be null"):
        session.conf.set("spark.sql.test.conf.of.none", None)
    with pytest.raises(TypeError):
        session.conf.set("spark.sql.test.conf.of.a.float", 1.5)
    assert session.conf.get("spark.sql.test.conf.of.none", None) is None
    for args in [(1,), ("spark.sql.test.conf.of.none", 1)]:
        with pytest.raises(PySparkTypeError) as e:
            session.conf.get(*args)
        assert e.value.getCondition() == "NOT_EXPECTED_TYPE"


def test_conf_get_with_a_none_default(session):
    key = "spark.connect.rust.test.unset"
    assert session.conf.get(key, None) is None
    assert session.conf.get(key, "x") == "x"
    with pytest.raises(Exception, match="SQL_CONF_NOT_FOUND"):
        session.conf.get(key)


def test_client_reads_configurations_as_pyspark(session):
    # As PySpark Connect's SparkConnectClient, of which pandas-on-Spark reads its ANSI mode: the
    # values of keys, by keys, and of keys of defaults, which a key that is not set takes.
    from pyspark.errors import SparkNoSuchElementException

    client = session.client
    key, unset = "spark.connect.rust.test.client.set", "spark.connect.rust.test.client.unset"
    session.conf.set(key, "value")
    try:
        assert client.get_config_with_defaults(
            (key, None), (unset, "default"), (unset + ".none", None)
        ) == ("value", "default", None)
        assert client.get_configs(key) == ("value",)
        assert client.get_config_dict(key) == {key: "value"}
    finally:
        session.conf.unset(key)
    with pytest.raises(SparkNoSuchElementException) as error:
        client.get_configs(unset)
    assert error.value.getCondition() == "SQL_CONF_NOT_FOUND"
    # pandas-on-Spark reads its ANSI mode so to show a Series.
    pd = pytest.importorskip("pandas")
    import pyspark.pandas as ps

    assert repr(ps.Series([1, 2, 3], name="x")) == repr(pd.Series([1, 2, 3], name="x"))


def test_numpy_scalar_literals_have_the_types_of_their_dtypes_as_pyspark(session):
    np = pytest.importorskip("numpy")
    from pyspark.sql import functions as F

    # As PySpark: a NumPy scalar's literal is typed by its dtype, as the server reports it.
    df = session.range(1).select(F.lit(np.int64(5)), F.lit(np.float32(1.5)), F.lit(np.int8(5)))
    assert [f.dataType.simpleString() for f in df.schema.fields] == ["bigint", "float", "tinyint"]


def test_user_id_of_the_connection_string_reaches_every_request_as_pyspark():
    # The server keys a session by the user ID in each request. As in PySpark, the
    # queries carry the user ID of the connection string, as the configs do, so a config
    # set with spark.conf.set applies to the queries and the other way around.
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live-server tests")
    remote += ";user_id=user-context-probe" if ";" in remote else "/;user_id=user-context-probe"
    spark = SparkSession.builder.remote(remote).create()
    try:
        spark.conf.set("spark.sql.shuffle.partitions", "7")
        assert spark.sql("SET spark.sql.shuffle.partitions").collect()[0][1] == "7"
        spark.sql("SET spark.sql.shuffle.partitions=9").collect()
        assert spark.conf.get("spark.sql.shuffle.partitions") == "9"
    finally:
        spark.stop()


def test_get_or_create_returns_the_session_with_the_options_as_pyspark(session):
    # As PySpark's builder: getOrCreate() returns the active session, and sets the options
    # of the builder on it; one that fails, of a static configuration, is ignored.
    key = "spark.sql.shuffle.partitions"
    previous = session.conf.get(key)
    try:
        got = (
            SparkSession.builder.remote(os.environ["SPARK_REMOTE"])
            .config(key, "5")
            .config("spark.sql.warehouse.dir", "/not/set")
            .getOrCreate()
        )
        assert got is session
        assert session.conf.get(key) == "5"
        assert session.conf.get("spark.sql.warehouse.dir") != "/not/set"
    finally:
        session.conf.set(key, previous)


def test_nested_duplicate_field_names_of_local_rows_reach_the_server(session):
    # As PySpark: the server reads the Arrow data of nested fields with deduplicated names,
    # and the schema of the relation keeps the names, so the rows come back as they were.
    R = Row("a", "a")
    df = session.createDataFrame([(R(1, 2),)])
    assert df.schema.simpleString() == "struct<_1:struct<a:bigint,a:bigint>>"
    assert tuple(df.collect()[0]._1) == (1, 2)
    df = session.createDataFrame([((1, 2),)], "s struct<a:bigint,a:bigint>")
    assert tuple(df.collect()[0].s) == (1, 2)
    df = session.createDataFrame([([(1, 2)],)], "s array<struct<a:bigint,a:bigint>>")
    assert [tuple(item) for item in df.collect()[0].s] == [(1, 2)]


def test_repr_shows_the_rows_with_eager_evaluation_as_pyspark(session):
    df = session.range(3)
    assert repr(df) == "DataFrame[id: bigint]"
    keys = ["spark.sql.repl.eagerEval.enabled", "spark.sql.repl.eagerEval.maxNumRows",
            "spark.sql.repl.eagerEval.truncate"]
    try:
        session.conf.set(keys[0], "true")
        session.conf.set(keys[1], "2")
        assert repr(df) == "+---+\n| id|\n+---+\n|  0|\n|  1|\n+---+\nonly showing top 2 rows"
        session.conf.set(keys[2], "3")
        assert repr(session.createDataFrame([("abcdef",)], "s string")) == (
            "+---+\n|  s|\n+---+\n|abc|\n+---+\n")
    finally:
        for key in keys:
            session.conf.unset(key)
    assert repr(df) == "DataFrame[id: bigint]"


def test_server_session_id_is_observed_and_a_changed_one_closes_the_session_as_pyspark():
    # SparkConnectClient._server_session_id: None before the first response, then the server's
    # ID; a changed one makes the next response close the client, as a server-side change does.
    import uuid

    from pyspark.errors.exceptions.connect import SparkConnectException
    from pyspark.sql.connect.session import SparkSession as ConnectSession

    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live-server tests")
    session = ConnectSession(connection=remote)
    try:
        assert session._client._server_session_id is None
        assert session.range(3).count() == 3
        assert session._client._server_session_id
        assert not session.is_stopped

        session._client._server_session_id = str(uuid.uuid4())
        with pytest.raises(SparkConnectException) as error:
            session.range(3).collect()
        # The close by the session change makes the fetch of the error's details raise this.
        assert "NO_ACTIVE_SESSION" in str(error.value)
        assert session.is_stopped
    finally:
        session.release_session_on_close = False
        session.stop()


def test_udtfs_and_udfs_of_a_dataframe_or_the_session_raise_no_active_session_as_pyspark(session):
    # As PySpark Connect's: a UDTF or a UDF that captures a DataFrame or the session pickles it
    # of its session, which raises NO_ACTIVE_SESSION on the Python worker of the server, where
    # no session is active.
    from pyspark.errors import PythonException

    df = session.range(10)

    @F.udtf(returnType="x: int")
    class Collect:
        def eval(self):
            df.collect()
            yield (1,)

    with pytest.raises(PythonException, match="NO_ACTIVE_SESSION"):
        Collect().collect()

    @F.udf("int")
    def of_session(x):
        return len(session.version) + x

    with pytest.raises(PythonException, match="NO_ACTIVE_SESSION"):
        session.range(1).select(of_session("id")).collect()




def test_reflected_and_or_name_the_column_first_as_pyspark(session):
    # As PySpark's Column: True & col and True | col name (col AND true) and (col OR true).
    df = session.range(2).select((F.col("id") > 0).alias("b"))
    assert df.select(True & F.col("b"), True | F.col("b")).columns == [
        "(b AND true)",
        "(b OR true)",
    ]


def test_renames_are_applied_in_their_order_as_pyspark(session):
    # As PySpark's withColumnsRenamed of a dict, whose renames the server applies in their order.
    df = session.range(10)
    assert df.withColumnsRenamed({"id": "a", "a": "b"}).columns == ["b"]
    assert df.withColumnsRenamed({"a": "b", "id": "a"}).columns == ["a"]
    renames = {"id": "c1", "c1": "c2", "c2": "c3", "c3": "c4", "c4": "c5", "c5": "c6"}
    assert df.withColumnsRenamed(renames).columns == ["c6"]




def test_join_as_of_and_merge_asof_give_the_rows_of_pyspark(session):
    # As PySpark: _joinAsOf matches each row of the left DataFrame with the nearest row of the
    # right one on the as-of column, as the direction, exact matches and tolerance say, and
    # pandas-on-Spark's merge_asof, which calls it, gives the same rows.
    import math

    pytest.importorskip("pandas")
    import pyspark.pandas as ps
    from pyspark.sql import functions as F

    def rows(values):
        return sorted(
            tuple("nan" if isinstance(v, float) and math.isnan(v) else v for v in row)
            for row in values
        )

    left = session.createDataFrame(
        [(1, 1.0, "a"), (3, 3.0, "c"), (5, 5.0, "b")], "t int, x double, k string")
    right = session.createDataFrame(
        [(2, 2.0, "x"), (3, 3.5, "c"), (4, 4.0, "y")], "t int, y double, k string")
    joined = rows(left._joinAsOf(right, "t", "t", how="left").collect())
    assert joined == [
        (1, 1.0, "a", None, None, None), (3, 3.0, "c", 3, 3.5, "c"), (5, 5.0, "b", 4, 4.0, "y")]
    assert rows(left._joinAsOf(right, "t", "t", how="left", allowExactMatches=False).collect()) == [
        (1, 1.0, "a", None, None, None), (3, 3.0, "c", 2, 2.0, "x"), (5, 5.0, "b", 4, 4.0, "y")]
    assert rows(left._joinAsOf(right, "t", "t", how="left", direction="forward").collect()) == [
        (1, 1.0, "a", 2, 2.0, "x"), (3, 3.0, "c", 3, 3.5, "c"), (5, 5.0, "b", None, None, None)]
    assert rows(left._joinAsOf(right, "t", "t", how="left", direction="nearest").collect()) == [
        (1, 1.0, "a", 2, 2.0, "x"), (3, 3.0, "c", 3, 3.5, "c"), (5, 5.0, "b", 4, 4.0, "y")]
    assert rows(left._joinAsOf(
        right, "t", "t", how="left", tolerance=F.lit(1)).collect()) == [
        (1, 1.0, "a", None, None, None), (3, 3.0, "c", 3, 3.5, "c"), (5, 5.0, "b", 4, 4.0, "y")]
    assert rows(left._joinAsOf(right, "t", "t", on=["k"], how="left").collect()) == [
        (1, 1.0, "a", None, None, None),
        (3, 3.0, "c", 3, 3.5, "c"),
        (5, 5.0, "b", None, None, None),
    ]
    assert rows(left._joinAsOf(right, "t", "t").collect()) == [
        (3, 3.0, "c", 3, 3.5, "c"), (5, 5.0, "b", 4, 4.0, "y")]

    # pandas-on-Spark's ANSI check is off for the merges, as in the tests of PySpark.
    key = "spark.sql.ansi.enabled"
    session.conf.set(key, "false")
    try:
        def merged(**kwargs):
            frame = ps.merge_asof(ps.DataFrame(left), ps.DataFrame(right), **kwargs)
            return rows(frame.to_pandas().values.tolist())

        assert merged(on="t") == [
            (1, 1.0, "a", "nan", None), (3, 3.0, "c", 3.5, "c"), (5, 5.0, "b", 4.0, "y")]
        assert merged(on="t", by="k", direction="forward") == [
            (1, 1.0, "a", "nan"), (3, 3.0, "c", 3.5), (5, 5.0, "b", "nan")]
        assert merged(on="t", tolerance=1, allow_exact_matches=False) == [
            (1, 1.0, "a", "nan", None), (3, 3.0, "c", "nan", None), (5, 5.0, "b", "nan", None)]
        assert merged(left_on="t", right_on="t", left_by="k", right_by="k") == [
            (1, 1.0, "a", "nan"), (3, 3.0, "c", 3.5), (5, 5.0, "b", "nan")]
    finally:
        session.conf.unset(key)


def test_join_with_an_unsupported_join_type_raises_as_pyspark(session):
    # As PySpark: the join type is checked when the join is built, which raises the
    # AnalysisException of UNSUPPORTED_JOIN_TYPE, naming the type as it is normalized.
    df = session.range(3)
    with pytest.raises(AnalysisException) as error:
        df.join(df, "id", "foo").collect()
    assert error.value.getCondition() == "UNSUPPORTED_JOIN_TYPE"
    assert error.value.getMessageParameters()["typ"] == "foo"
    assert sorted(row.id for row in df.join(df, "id", "left_semi").collect()) == [0, 1, 2]


def test_streaming_query_status_and_progress_are_as_pyspark(session):
    # As PySpark: the status is a dict of message, isDataAvailable and isTriggerActive, and the
    # progress is a StreamingQueryProgress, a dict, of the query.
    from pyspark.sql.streaming.listener import StreamingQueryProgress

    query = (
        session.readStream.format("rate")
        .load()
        .writeStream.format("memory")
        .queryName("fug_status_query")
        .start()
    )
    try:
        # Wait, bounded, for the first micro-batch, rather than processAllAvailable.
        deadline = time.monotonic() + 60
        while query.lastProgress is None and time.monotonic() < deadline:
            time.sleep(0.5)
        status = query.status
        assert isinstance(status, dict)
        assert list(status) == ["message", "isDataAvailable", "isTriggerActive"]
        assert isinstance(status["message"], str)
        assert isinstance(status["isDataAvailable"], bool)
        assert isinstance(status["isTriggerActive"], bool)
        last = query.lastProgress
        assert isinstance(last, StreamingQueryProgress)
        assert last["name"] == "fug_status_query"
        recent = query.recentProgress
        assert recent and all(isinstance(p, StreamingQueryProgress) for p in recent)
    finally:
        query.stop()


def test_replace_and_unpivot_take_their_arguments_as_pyspark(session):
    # As PySpark's: replace sends each value as a literal of its type, and None as a null, over
    # a subset of a column name or a list or tuple of them. unpivot and melt take a column or a
    # column name, or a list or tuple of them.
    df = session.createDataFrame(
        [(1, "a", 1.5, True, "1"), (2, "b", None, False, "2")],
        "id int, k string, v double, b boolean, s string",
    )

    def rows(*changes):
        base = [dict(id=1, k="a", v=1.5, b=True, s="1"), dict(id=2, k="b", v=None, b=False, s="2")]
        return [Row(**{**row, **change}) for row, change in zip(base, changes)]

    assert df.replace("a", None, "k").collect() == rows({"k": None}, {})
    # A string replaces in the string columns, and a number in the numeric ones.
    assert df.replace("1", "one").collect() == rows({"s": "one"}, {})
    assert df.replace(1, 10, ("id",)).collect() == rows({"id": 10}, {})
    assert df.replace([1.5, 2], [9.5, 20]).collect() == rows({"v": 9.5}, {"id": 20})
    assert df.replace(True, False).collect() == rows({"b": False}, {})
    assert df.na.replace({"a": "x", "b": "y"}, subset="k").collect() == rows({"k": "x"}, {"k": "y"})

    wide = session.createDataFrame([(1, 11, 12)], "id int, x int, y int")
    expected = [Row(id=1, var="x", val=11), Row(id=1, var="y", val=12)]
    assert wide.unpivot("id", ["x", "y"], "var", "val").collect() == expected
    assert wide.unpivot(F.col("id"), ("x", F.col("y")), "var", "val").collect() == expected
    assert wide.melt(["id"], None, "var", "val").collect() == expected
    assert wide.melt("id", "x", "var", "val").collect() == expected[:1]


def _run_in_time_zone(code, zone):
    # Runs `code` in a new interpreter whose time zone is `zone`: the local time zone of the
    # client, which the test process (UTC) does not have.
    import subprocess
    import sys

    env = dict(os.environ, TZ=zone)
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_nanosecond_timestamps_are_not_collected_as_pyspark(session):
    # As PySpark's conversion of the schema from its proto: a TIMESTAMP_NTZ(p) or TIMESTAMP_LTZ(p)
    # with p > 6, also nested, raises UNSUPPORTED_OPERATION from every collecting API, toPandas
    # and toArrow, and from every schema surface (the schema, columns, dtypes, printSchema, repr
    # and json, and the DDL strings of the types), while count, show and the microsecond
    # timestamps keep working.
    import datetime
    import re

    from pyspark.errors import PySparkValueError
    from pyspark.sql.functions import from_json, lit
    from pyspark.sql.types import DataType, StructType

    def message(proto_text):
        return r"\[UNSUPPORTED_OPERATION\] data type " + re.escape(proto_text) + r" is not supported\."

    ntz9 = "timestamp_ntz_nanos {\n  precision: 9\n}\n"
    ltz7 = "timestamp_ltz_nanos {\n  precision: 7\n}\n"
    ltz9 = "timestamp_ltz_nanos {\n  precision: 9\n}\n"
    cases = [
        ("SELECT CAST('2020-01-01 00:00:00.123456789' AS TIMESTAMP_NTZ(9)) c", ntz9),
        ("SELECT CAST('2020-01-01 00:00:00.1234567' AS TIMESTAMP_LTZ(7)) c", ltz7),
        ("SELECT named_struct('a', CAST('2020-01-01 00:00:00.123456789' AS TIMESTAMP_NTZ(9))) c",
         ntz9),
        ("SELECT array(CAST('2020-01-01 00:00:00.123456789' AS TIMESTAMP_LTZ(9))) c", ltz9),
        ("SELECT map('k', CAST('2020-01-01 00:00:00.123456789' AS TIMESTAMP_NTZ(9))) c", ntz9),
    ]
    # The types need spark.sql.timestampNanosTypes.enabled, which is on by default only on a
    # server under SPARK_TESTING, as CI's servers are: turn it on, and back to the server's
    # default after, so that the test runs against any server.
    session.conf.set("spark.sql.timestampNanosTypes.enabled", "true")
    try:
        for query, proto_text in cases:
            df = session.sql(query)
            for collect in (
                lambda d: d.collect(),
                lambda d: d.first(),
                lambda d: d.head(),
                lambda d: d.head(1),
                lambda d: d.take(1),
                lambda d: d.tail(1),
                lambda d: list(d.toLocalIterator()),
                lambda d: d.toPandas(),
                lambda d: d.toArrow(),
            ):
                with pytest.raises(PySparkValueError, match=message(proto_text)):
                    collect(df)
            # The schema surfaces raise too: PySpark converts a schema from its proto, so the
            # schema of the frame, its columns and types, and the schema of a DDL string do not.
            for surface in (
                lambda d: d.schema,
                lambda d: d.columns,
                lambda d: d.dtypes,
                lambda d: d.printSchema(),
                lambda d: repr(d),
                lambda d: d.schema.json(),
            ):
                with pytest.raises(PySparkValueError, match=message(proto_text)):
                    surface(df)
        # The DDL strings of the types, as the server parses them.
        for surface in (
            lambda: session.createDataFrame([(1,)], "a timestamp_ntz(9)"),
            lambda: DataType.fromDDL("timestamp_ntz(9)"),
            lambda: StructType.fromDDL("a struct<x: timestamp_ntz(9)>"),
            lambda: session.range(1).select(from_json(lit('{"a": 1}'), "a timestamp_ntz(9)")).schema,
        ):
            with pytest.raises(PySparkValueError, match=message(ntz9)):
                surface()
        # Spark Connect and PySpark both count and show these results.
        df = session.sql("SELECT CAST('2020-01-01 00:00:00.123456789' AS TIMESTAMP_NTZ(9)) c")
        assert df.count() == 1
        df.show()
        # Microsecond timestamps still collect.
        micros = session.sql("SELECT CAST('2020-01-01 00:00:00.123456' AS TIMESTAMP_NTZ(6)) c")
        assert micros.collect()[0].c == datetime.datetime(2020, 1, 1, 0, 0, 0, 123456)
    finally:
        session.conf.unset("spark.sql.timestampNanosTypes.enabled")


def test_ntz_collect_is_the_wall_clock_in_the_client_time_zone_as_pyspark():
    # As PySpark's Arrow conversion: a TIMESTAMP_NTZ is the wall clock, whatever the time zone of
    # the client, from year 1 to year 9999 and inside arrays, maps and structs, while a TIMESTAMP
    # is the instant, shown in the local time zone (here seven hours behind UTC, in July).
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live-server tests")
    out = _run_in_time_zone(f"""
import datetime
from pyspark.sql import SparkSession
spark = SparkSession.builder.remote({remote!r}).getOrCreate()
print(spark.createDataFrame([(datetime.datetime(2020, 7, 1, 12, 0),)], "t timestamp_ntz").collect())
print(spark.sql("select timestamp_ntz'2020-07-01 12:00:00' t").collect())
print(spark.sql("select timestamp_ntz'0001-01-01 00:00:00' t, timestamp_ntz'9999-12-31 23:59:59.999999' u").collect())
print(spark.sql(
    "select array(timestamp_ntz'2020-07-01 12:00:00') a, map('k', timestamp_ntz'2020-07-01 12:00:00') m, "
    "named_struct('x', timestamp_ntz'2020-07-01 12:00:00') s").collect())
print(spark.sql("select timestamp'2020-07-01 12:00:00' t, timestamp_micros(9223372036854775) u").collect())
""", "America/Los_Angeles")
    lines = out.strip().splitlines()
    assert lines[0] == "[Row(t=datetime.datetime(2020, 7, 1, 12, 0))]"
    assert lines[1] == "[Row(t=datetime.datetime(2020, 7, 1, 12, 0))]"
    assert lines[2] == ("[Row(t=datetime.datetime(1, 1, 1, 0, 0), "
                        "u=datetime.datetime(9999, 12, 31, 23, 59, 59, 999999))]")
    assert lines[3] == ("[Row(a=[datetime.datetime(2020, 7, 1, 12, 0)], "
                        "m={'k': datetime.datetime(2020, 7, 1, 12, 0)}, "
                        "s=Row(x=datetime.datetime(2020, 7, 1, 12, 0)))]")
    assert lines[4] == ("[Row(t=datetime.datetime(2020, 7, 1, 5, 0), "
                        "u=datetime.datetime(2262, 4, 11, 16, 47, 16, 854775))]")


def test_year_month_intervals_are_not_collected_as_pyspark(session):
    # As PySpark's Arrow conversion of the rows: a year-month interval, also nested, is not
    # collected in Spark Connect, and raises NOT_IMPLEMENTED, even for an empty result.
    import datetime

    from pyspark.errors import PySparkNotImplementedError

    message = (r"\[NOT_IMPLEMENTED\] Collecting a year-month interval value in Spark Connect "
               r"is not implemented\.")
    df = session.sql("SELECT INTERVAL '1-2' YEAR TO MONTH ym")
    for collect in (
        lambda d: d.collect(),
        lambda d: d.first(),
        lambda d: d.head(1),
        lambda d: d.take(1),
        lambda d: list(d.toLocalIterator()),
        lambda d: d.where("1 = 0").collect(),
        lambda d: session.sql("SELECT array(INTERVAL '1-2' YEAR TO MONTH) a").collect(),
        lambda d: session.sql("SELECT map('a', INTERVAL '1-2' YEAR TO MONTH) m").collect(),
        lambda d: session.sql("SELECT named_struct('a', INTERVAL '1-2' YEAR TO MONTH) s").collect(),
    ):
        with pytest.raises(PySparkNotImplementedError, match=message):
            collect(df)
    # A day-time interval is collected as a timedelta.
    assert session.sql("SELECT INTERVAL '1 02:03:04' DAY TO SECOND dt").collect()[0][0] == (
        datetime.timedelta(days=1, seconds=7384))


def test_repartition_by_range_without_a_count_lets_the_server_pick_it(session):
    # As PySpark's: the str/Column forms send no partition count, so the server picks it
    # and returns the rows.
    df = session.range(5)
    for repartitioned in (df.repartitionByRange("id"), df.repartitionByRange(F.col("id")),
                          df.repartitionByRange("id", "id")):
        assert sorted(row.id for row in repartitioned.collect()) == [0, 1, 2, 3, 4]


def test_stream_reader_checks_its_arguments_as_pyspark(session, tmp_path):
    # As PySpark's DataStreamReader, which checks a path, a source name and the schema of
    # changes in the client, rather than sending them to the server.
    from pyspark.errors import PySparkTypeError, PySparkValueError

    source = tmp_path / "source"
    session.range(2).write.json(str(source))
    assert session.readStream.schema("id long").json(str(source)).isStreaming
    with pytest.raises(PySparkValueError) as e:
        session.readStream.format("json").schema("id long").load("")
    assert e.value.getCondition() == "VALUE_NOT_NON_EMPTY_STR"
    with pytest.raises(PySparkTypeError) as e:
        session.readStream.json([str(source)])
    assert e.value.getCondition() == "NOT_EXPECTED_TYPE"
    with pytest.raises(PySparkValueError) as e:
        session.readStream.name("a-b")
    assert e.value.getCondition() == "INVALID_STREAMING_SOURCE_NAME"
    with pytest.raises(AnalysisException) as e:
        session.readStream.schema("id long").changes("t")
    assert e.value.getCondition() == "_LEGACY_ERROR_TEMP_1189"


def test_local_relations_of_64_kib_follow_the_size_limit_as_pyspark(session):
    # Local rows and pandas frames of 64 KiB or more read the local-relation settings of a
    # session, so they fail past its size limit, as PySpark does, whichever schema form.
    import numpy as np
    import pandas as pd

    rows = [("y" * 100, i) for i in range(1000)]
    frame = pd.DataFrame({"v": ["y" * 100] * 1000, "i": np.arange(1000, dtype=np.int64)})
    schema = StructType([StructField("v", StringType()), StructField("i", LongType())])
    ddl = "v string, i long"
    spark = session.newSession()
    try:
        spark.conf.set("spark.sql.session.localRelationCacheThreshold", "1")
        spark.conf.set("spark.sql.session.localRelationSizeLimit", "100")
        for data in (rows, frame):
            for form in (schema, ddl):
                with pytest.raises(AnalysisException) as e:
                    spark.createDataFrame(data, form)
                assert e.value.getCondition() == "LOCAL_RELATION_SIZE_LIMIT_EXCEEDED"
    finally:
        spark.stop()
    # The same data works with the default settings.
    for data in (rows, frame):
        for form in (schema, ddl):
            assert session.createDataFrame(data, form).count() == 1000


def test_instr_and_regexp_replace_apply_their_optional_arguments(session):
    # Column b holds "c", so the literal "b" and the column b find different positions.
    df = session.createDataFrame([("abcabc", "c", "a", "z")], ["s", "b", "p", "r"])
    out = df.select(
        F.instr("s", "b"),
        F.instr("s", "b", 3),
        F.instr("s", "b", occurrence=2),
        F.instr(F.col("s"), F.col("b"), F.lit(4)),
        F.regexp_replace("s", "a", "z", 2),
        F.regexp_replace("s", F.col("p"), F.col("r")),
        F.call_function("instr", "s", F.lit("b")),
    )
    assert out.columns == [
        "instr(s, b)", "instr(s, b, 3, 1)", "instr(s, b, 1, 2)", "instr(s, b, 4, 1)",
        "regexp_replace(s, a, z, 2)", "regexp_replace(s, p, r, 1)", "instr(s, b)",
    ]
    assert tuple(out.first()) == (2, 5, 5, 6, "abczbc", "zbczbc", 2)


@pytest.mark.parametrize("by_name", [True, False])
def test_repartition_by_id_takes_a_column_name(session, by_name):
    df = session.range(12).withColumn("p", (F.col("id") % 3).cast("int"))
    partition_id_col = "p" if by_name else F.col("p")
    rows = (
        df.repartitionById(3, partition_id_col)
        .select("p", F.spark_partition_id().alias("pid"))
        .collect()
    )
    assert {(r.p, r.pid) for r in rows} == {(0, 0), (1, 1), (2, 2)}


@pytest.mark.parametrize(
    "schema",
    [
        StructType().add("a", LongType()).add("b", LongType()),
        StructType([StructField("a", LongType())]).add("b", LongType()),
    ],
)
def test_map_in_arrow_and_pandas_with_a_struct_type_built_with_add(session, schema):
    df = session.range(3).selectExpr("id AS a", "id * 2 AS b")
    assert df.mapInArrow(lambda batches: batches, schema).columns == ["a", "b"]
    rows = df.mapInPandas(lambda frames: frames, schema).orderBy("a").collect()
    assert [tuple(r) for r in rows] == [(0, 0), (1, 2), (2, 4)]


def test_udtf_takes_a_ddl_or_struct_return_type_as_pyspark(session):
    from pyspark.errors.exceptions.connect import ParseException

    class Echo:
        def eval(self, value):
            yield (value,)
            yield (value + 1,)

    struct = StructType([StructField("value", LongType())])
    for return_type in ["value BIGINT", "value: bigint", struct]:
        for fn in (F.udtf(Echo, returnType=return_type), F.udtf(returnType=return_type)(Echo)):
            assert fn(F.lit(1)).collect() == [Row(value=1), Row(value=2)]

    # The server parses the DDL, so an invalid one fails when the query runs.
    df = F.udtf(Echo, returnType="invalid DDL !!!")(F.lit(1))
    with pytest.raises(ParseException) as error:
        df.collect()
    assert error.value.getCondition() == "PARSE_SYNTAX_ERROR"


def test_udtf_passes_custom_analyze_result_to_handler(session):
    from dataclasses import dataclass

    from pyspark.sql.udtf import AnalyzeResult

    @dataclass
    class ResultWithBuffer(AnalyzeResult):
        buffer: str = ""

    @F.udtf(useArrow=False)
    class Echo:
        def __init__(self, analyze_result):
            self.buffer = analyze_result.buffer

        @staticmethod
        def analyze(argument):
            return ResultWithBuffer(
                StructType([StructField("buffer", StringType())]), buffer=argument.value
            )

        def eval(self, value):
            yield (self.buffer,)

    assert Echo(F.lit("kept")).collect() == [Row(buffer="kept")]


def test_udtf_handlers_that_capture_analysis_columns_run_as_pyspark(session):
    # As PySpark's test_udtf_with_scalar_analyze_returning_wrong_result: a handler that captures
    # the result of its analysis, of partitioning columns, pickles, and the Python worker of the
    # server, which unpickles them as PySpark's, raises its error of them.
    from pyspark.sql.udtf import AnalyzeResult, PartitioningColumn

    result = AnalyzeResult(StructType().add("a", StringType()), partitionBy=[PartitioningColumn("a")])

    @F.udtf
    class Partitioned:
        @staticmethod
        def analyze(**kwargs):
            return result

        def eval(self, **kwargs):
            yield (1,)

    with pytest.raises(AnalysisException, match="partitionBy"):
        Partitioned().collect()


def test_map_schemas_keep_the_metadata_edited_in_place_on_the_server_as_pyspark(session):
    # As PySpark's: the schema of mapInPandas, mapInArrow and applyInPandas is the StructType
    # with the metadata edited in place after it was built, so the result has it.
    st = StructType([StructField("a", IntegerType()), StructField("b", StringType())])
    st["a"].metadata["comment"] = "the a"
    df = session.createDataFrame([(1, "x")], "a int, b string")
    assert df.mapInPandas(lambda it: it, st).schema["a"].metadata == {"comment": "the a"}
    assert df.mapInArrow(lambda it: it, st).schema["a"].metadata == {"comment": "the a"}
    grouped = df.groupBy("a")
    assert grouped.applyInPandas(lambda pdf: pdf, st).schema["a"].metadata == {
        "comment": "the a"}


def test_arrow_udtf_takes_and_returns_arrow_data_as_pyspark(session):
    # As PySpark's arrow_udtf(), a PyArrow-native UDTF: eval() takes a pyarrow array of the
    # argument, and returns pyarrow tables.
    import pyarrow.compute as pc

    @F.arrow_udtf(returnType="x long, kind string")
    class PlusOne:
        def eval(self, x):
            yield pa.table({"x": pc.add(x, 1), "kind": pa.array([type(x).__name__] * len(x))})

    assert PlusOne(F.lit(1)).collect() == [Row(x=2, kind="Int32Array")]


def test_java_functions_are_registered_of_their_return_types_as_pyspark(session):
    # As PySpark's: spark.udf.sparkSession is the session, and registerJavaFunction sends a
    # DDL string for the server to parse, so an invalid one is its PARSE_SYNTAX_ERROR.
    from pyspark.errors import ParseException

    assert session.udf.sparkSession is session
    with pytest.raises(ParseException) as e:
        session.udf.registerJavaFunction("java_f", "com.example.NoSuchFunction", "nonsense")
    assert e.value.getCondition() == "PARSE_SYNTAX_ERROR"
    with pytest.raises(PySparkValueError) as e:
        session.udf.registerJavaFunction("java_f", "com.example.NoSuchFunction", 1)
    assert e.value.getCondition() == "UNSUPPORTED_OPERATION"


def test_udfs_are_arrow_optimized_or_of_their_type_hints_as_pyspark(session):
    # As PySpark's udf(): Arrow-optimized as the server enables by default, so Arrow
    # converts a result to the return type; and a function with pandas type hints is a
    # pandas UDF.
    import pandas as pd

    df = session.range(2)
    assert df.select(F.udf(lambda x: str(x), "int")("id").alias("r")).collect() == [
        Row(r=0), Row(r=1)
    ]
    assert df.select(F.udf(lambda x: 1.5, "int")("id").alias("r")).collect() == [
        Row(r=1), Row(r=1)
    ]

    def plus_one(s: pd.Series) -> pd.Series:
        return s + 1

    hinted = F.udf(plus_one, "long")
    assert hinted.evalType == 200
    assert df.select(hinted("id").alias("r")).collect() == [Row(r=1), Row(r=2)]


def test_map_functions_take_a_ddl_schema(session):
    df = session.createDataFrame([(1, 1.0), (1, 2.0), (2, 3.0)], ["k", "v"])
    schema = "k long, v double"
    grouped = df.groupBy("k")
    for result in (
        df.mapInPandas(lambda frames: frames, schema),
        df.mapInArrow(lambda batches: batches, schema),
        grouped.applyInPandas(lambda pdf: pdf, schema),
        grouped.applyInArrow(lambda table: table, schema),
        grouped.cogroup(grouped).applyInPandas(lambda left, right: left, schema),
        grouped.cogroup(grouped).applyInArrow(lambda left, right: left, schema),
    ):
        assert sorted(tuple(r) for r in result.collect()) == [(1, 1.0), (1, 2.0), (2, 3.0)]


def test_inheritable_threads_run_with_the_tags_of_their_parents_as_pyspark(session):
    # As PySpark's: the operation of an InheritableThread carries the tags of the thread that
    # started it, so interruptTag of a tag of the parent interrupts it.
    import time

    from pyspark import InheritableThread
    from pyspark.sql import functions as F

    slow = F.udf(lambda x: (time.sleep(30), x)[1], "long")
    results = {}

    def run():
        try:
            results["child"] = session.range(1).select(slow("id")).collect()
        except Exception as e:  # noqa: BLE001
            results["child"] = e

    session.addTag("tag_of_the_parent")
    try:
        thread = InheritableThread(target=run, session=session)
        thread.start()
        time.sleep(3)
        interrupted = session.interruptTag("tag_of_the_parent")
        thread.join(60)
    finally:
        session.clearTags()
    assert len(interrupted) == 1
    assert isinstance(results["child"], Exception)


def test_map_functions_run_in_the_stage_of_their_resource_profile(session):
    # As PySpark's, mapInPandas and mapInArrow send their resource profile, which the server
    # creates: a server that does not support it, as one in local mode, rejects it, and
    # otherwise the stage runs with the profile, of the id that the server gave it.
    from pyspark.errors import PySparkException
    from pyspark.resource import ResourceProfileBuilder, TaskResourceRequests

    for name in ("mapInPandas", "mapInArrow"):
        profile = ResourceProfileBuilder().require(TaskResourceRequests().cpus(1)).build
        try:
            mapped = getattr(session.range(3), name)(lambda it: it, "id long", profile=profile)
            rows = mapped.collect()
        except PySparkException as error:
            assert "ResourceProfiles are only supported" in str(error)
        else:
            assert sorted(r.id for r in rows) == [0, 1, 2]
            assert isinstance(profile._id, int)
    assert session.range(3).mapInPandas(lambda it: it, "id long", profile=None).count() == 3


def test_transform_with_state_in_pandas_takes_a_ddl_output_type(session, tmp_path):
    import pandas as pd
    from pyspark.sql.streaming.stateful_processor import StatefulProcessor

    class Counter(StatefulProcessor):
        def init(self, handle):
            self.count = handle.getValueState("count", "c long")

        def handleInputRows(self, key, rows, timerValues):
            total = sum(len(pdf) for pdf in rows)
            self.count.update((total,))
            yield pd.DataFrame({"k": [key[0]], "c": [total]})

        def close(self):
            pass

    confs = {
        "spark.sql.streaming.stateStore.providerClass":
            "org.apache.spark.sql.execution.streaming.state.RocksDBStateStoreProvider",
        # A state store per shuffle partition; the default 200 take half a minute.
        "spark.sql.shuffle.partitions": "2",
    }
    for key, value in confs.items():
        session.conf.set(key, value)
    try:
        source = str(tmp_path / "source")
        session.createDataFrame([(1,), (1,), (2,)], "k long").write.parquet(source)
        query = (
            session.readStream.schema("k long").parquet(source).groupBy("k")
            .transformWithStateInPandas(Counter(), "k long, c long", "Update", "None")
            .writeStream.format("memory").queryName("tws_ddl_output").outputMode("update")
            .option("checkpointLocation", str(tmp_path / "checkpoint"))
            .trigger(availableNow=True).start()
        )
        try:
            query.awaitTermination(120)
            assert query.exception() is None
            rows = session.table("tws_ddl_output").collect()
            assert sorted(tuple(r) for r in rows) == [(1, 2), (2, 1)]
        finally:
            query.stop()
    finally:
        for key in confs:
            session.conf.unset(key)


def test_writers_shared_by_threads_write_one_at_a_time_as_pyspark(session, tmp_path):
    # As PySpark's: a writer is a plain Python object, so threads may write with one
    # DataFrameWriter or DataFrameWriterV2 at once, while the main thread sets its format.
    from concurrent.futures import ThreadPoolExecutor

    df = session.range(300000)
    writer = df.write.mode("overwrite")
    table = f"r48_shared_writer_{os.getpid()}"
    writer_v2 = df.writeTo(table).using("parquet")
    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            writes = [pool.submit(writer.parquet, str(tmp_path / f"p{i}")) for i in range(3)]
            writes.append(pool.submit(writer_v2.create))
            assert writer.format("parquet") is writer
            for write in writes:
                write.result(timeout=300)
        for i in range(3):
            assert session.read.parquet(str(tmp_path / f"p{i}")).count() == 300000
        assert session.table(table).count() == 300000
    finally:
        session.sql(f"DROP TABLE IF EXISTS {table}")


def test_transform_with_state_returns_the_rows_of_its_output_schema(session, tmp_path):
    # As PySpark's, transformWithState sends its output schema, a DDL string or a
    # StructType, which the server reads as the schema of the rows of the processor.
    from pyspark.sql.streaming.stateful_processor import StatefulProcessor

    class Counter(StatefulProcessor):
        def init(self, handle):
            pass

        def handleInputRows(self, key, rows, timerValues):
            yield Row(k=key[0], c=sum(1 for _ in rows))

        def close(self):
            pass

    confs = {
        "spark.sql.streaming.stateStore.providerClass":
            "org.apache.spark.sql.execution.streaming.state.RocksDBStateStoreProvider",
        # A state store per shuffle partition; the default 200 take half a minute.
        "spark.sql.shuffle.partitions": "2",
    }
    for key, value in confs.items():
        session.conf.set(key, value)
    try:
        source = str(tmp_path / "source")
        session.createDataFrame([(1,), (1,), (2,)], "k long").write.parquet(source)
        struct = StructType([StructField("k", LongType()), StructField("c", LongType())])
        for name, schema in [("ddl", "k long, c long"), ("struct", struct)]:
            counted = (
                session.readStream.schema("k long").parquet(source).groupBy("k")
                .transformWithState(Counter(), schema, "Update", "None")
            )
            assert counted.schema == struct
            query = (
                counted.writeStream.format("memory").queryName(f"tws_rows_{name}")
                .outputMode("update")
                .option("checkpointLocation", str(tmp_path / f"checkpoint_{name}"))
                .trigger(availableNow=True).start()
            )
            try:
                query.awaitTermination(120)
                assert query.exception() is None
                rows = session.table(f"tws_rows_{name}").collect()
                assert sorted(tuple(r) for r in rows) == [(1, 2), (2, 1)]
            finally:
                query.stop()
    finally:
        for key in confs:
            session.conf.unset(key)


def _arrow_table():
    return pa.table(
        {
            "a": pa.array([1, 2, None], pa.int64()),
            "b": pa.array(["x", None, "z"]),
            "ts": pa.array(
                [datetime(2020, 1, 1, 12), None, datetime(2021, 6, 1)], pa.timestamp("us")
            ),
            "tz": pa.array(
                [datetime(2020, 1, 1, 12), None, datetime(2021, 6, 1)],
                pa.timestamp("us", tz="America/Los_Angeles"),
            ),
            "d": pa.array([date(2020, 1, 1), None, date(2021, 1, 1)]),
            "f": pa.array([1.5, None, 2.5]),
            "l": pa.array([[1, 2], None, []], pa.list_(pa.int32())),
        }
    )


def _rows(df):
    # Timestamps as strings in the session time zone, so that the client's does not matter.
    columns = [
        F.col(f.name).cast("string") if isinstance(f.dataType, TimestampType) else F.col(f.name)
        for f in df.schema.fields
    ]
    return [tuple(r) for r in df.select(*columns).collect()]


ARROW_TABLE_ROWS = [
    (1, "x", "2020-01-01 12:00:00", "2020-01-01 12:00:00", date(2020, 1, 1), 1.5, [1, 2]),
    (2, None, None, None, None, None, None),
    (None, "z", "2021-06-01 00:00:00", "2021-06-01 00:00:00", date(2021, 1, 1), 2.5, []),
]


def test_empty_sort_list_keeps_the_cached_schema_and_fails_when_it_runs(session):
    # As PySpark's: a sort of an empty list keeps the cached schema of its input, so its schema
    # is returned without asking the server, while the server rejects the sort when it runs.
    df = session.range(3)
    schema = df.schema
    for sort in (
        lambda d: d.sort([]),
        lambda d: d.orderBy([]),
        lambda d: d.sortWithinPartitions([]),
    ):
        empty = sort(df)
        assert empty.schema == schema
        with pytest.raises(Exception) as error:
            empty.collect()
        assert error.value.getCondition() == "CONNECT_INVALID_PLAN.ASSERTION_FAILURE"


def test_pickled_sort_columns_sort_on_the_server_as_pyspark(session, tmp_path):
    # As PySpark's: a pickled sort Column is a sort order again, so orderBy, a window and
    # repartitionByRange sort by it as by the Column built without the pickling, and a
    # foreachBatch function that captures it sorts by it on the Python worker of the server.
    import pickle

    from pyspark.sql.window import Window

    df = session.createDataFrame([(1, "a"), (3, "b"), (None, "c"), (2, "d")], "x int, y string")
    for direct in (F.col("x").desc(), F.col("x").asc_nulls_last()):
        copied = pickle.loads(pickle.dumps(direct))
        assert df.orderBy(copied).collect() == df.orderBy(direct).collect()
        ranked = df.select("y", F.row_number().over(Window.orderBy(copied)).alias("r"))
        assert sorted(ranked.collect()) == sorted(
            df.select("y", F.row_number().over(Window.orderBy(direct)).alias("r")).collect()
        )
        by_y = lambda row: row.y  # noqa: E731
        assert sorted(df.repartitionByRange(2, copied).collect(), key=by_y) == sorted(
            df.collect(), key=by_y)

    (tmp_path / "input.txt").write_text("a\nc\nb\n")
    captured = F.col("value").desc()
    table = "foreach_batch_captured_sort"

    def func(batch, batch_id):
        batch.orderBy(captured).limit(1).write.mode("append").saveAsTable(table)

    query = session.readStream.format("text").load(str(tmp_path)).writeStream.foreachBatch(func)
    query = query.start()
    try:
        query.processAllAvailable()
        assert [row.value for row in session.table(table).collect()] == ["c"]
    finally:
        query.stop()
        session.sql(f"DROP TABLE IF EXISTS {table}").collect()


def test_pickled_when_columns_chain_on_the_server_as_pyspark(session, tmp_path):
    # As PySpark's: a pickled when() Column keeps its CASE WHEN, so it collects as the chain
    # built without the pickling, and a foreachBatch function that captures it chains on it on
    # the Python worker of the server.
    import pickle

    (tmp_path / "input.txt").write_text("hello\nthis\nother\n")
    df = session.createDataFrame([(1,), (2,), (3,)], "a int")
    chain = F.when(F.col("a") > 2, "big").when(F.col("a") > 1, "mid").otherwise("small")
    assert df.select(pickle.loads(pickle.dumps(chain)).alias("r")).collect() == (
        df.select(chain.alias("r")).collect()
    )
    assert [row.r for row in df.select(pickle.loads(pickle.dumps(chain)).alias("r")).collect()] == [
        "small", "mid", "big"]

    captured = F.when(F.col("value") == "hello", "h")
    table = "foreach_batch_captured_when"

    def func(batch, batch_id):
        batch.select(
            captured.when(F.col("value") == "this", "t").otherwise("x").alias("result")
        ).write.mode("append").saveAsTable(table)

    query = session.readStream.format("text").load(str(tmp_path)).writeStream.foreachBatch(func)
    query = query.start()
    try:
        query.processAllAvailable()
        assert sorted(row.result for row in session.table(table).collect()) == ["h", "t", "x"]
    finally:
        query.stop()
        session.sql(f"DROP TABLE IF EXISTS {table}").collect()


def test_pickled_deep_column_on_the_server_as_pyspark(session, tmp_path):
    # As PySpark's: a pickled Column of 100 additions, deeper than protobuf's parser reads,
    # collects as the Column built without the pickling, and a foreachBatch function that captures
    # it unpickles it on the Python worker of the server, with PySpark's Column.
    import pickle

    (tmp_path / "input.txt").write_text("a\nbb\n")
    df = session.createDataFrame([(1,), (2,)], "a int")
    direct = F.length(F.col("a").cast("string"))
    for _ in range(100):
        direct = direct + 1
    copied = pickle.loads(pickle.dumps(direct))
    assert sorted(row[0] for row in df.select(copied.alias("n")).collect()) == [101, 101]
    assert df.select(copied.alias("n")).collect() == df.select(direct.alias("n")).collect()

    captured = F.length(F.col("value"))
    for _ in range(100):
        captured = captured + 1
    # The bytes of the expression that the pickle carries, as Column.__reduce__ gives them.
    expected = len(captured.__reduce__()[1][0])
    table = "foreach_batch_captured_deep_column"

    def func(batch, batch_id):
        # The worker's PySpark unpickles the Column, and gives the expression back whole. Its own
        # plans take an expression this deep only with protobuf's pure-Python backend: upb copies
        # a message added to a repeated field through its parser, which stops at 100 levels, so
        # official PySpark raises DecodeError for `batch.select(captured)` there too. The worker
        # checks the expression itself, which any backend gives back.
        size = captured._expr.to_plan(None).ByteSize()
        batch.sparkSession.createDataFrame([(batch_id, size)], "b long, size long").write.mode(
            "append"
        ).saveAsTable(table)

    query = session.readStream.format("text").load(str(tmp_path)).writeStream.foreachBatch(func)
    query = query.start()
    try:
        query.processAllAvailable()
        assert [row.size for row in session.table(table).collect()] == [expected]
    finally:
        query.stop()
        session.sql(f"DROP TABLE IF EXISTS {table}").collect()


@pytest.mark.parametrize(
    "schema, expected",
    [
        (None, "struct<a:bigint,b:string,ts:timestamp,tz:timestamp,d:date,f:double,l:array<int>>"),
        (
            ["x", "y"],
            "struct<x:bigint,y:string,_3:timestamp,_4:timestamp,_5:date,_6:double,_7:array<int>>",
        ),
        (
            "a int, b string, ts timestamp, tz timestamp, d date, f double, l array<int>",
            "struct<a:int,b:string,ts:timestamp,tz:timestamp,d:date,f:double,l:array<int>>",
        ),
        (
            StructType(
                [
                    StructField("a", IntegerType()),
                    StructField("b", StringType()),
                    StructField("ts", TimestampType()),
                    StructField("tz", TimestampType()),
                    StructField("d", DateType()),
                    StructField("f", DoubleType()),
                    StructField("l", ArrayType(IntegerType())),
                ]
            ),
            "struct<a:int,b:string,ts:timestamp,tz:timestamp,d:date,f:double,l:array<int>>",
        ),
    ],
)
def test_create_dataframe_from_arrow_table(session, schema, expected):
    session.conf.set("spark.sql.session.timeZone", "UTC")
    if schema is None:
        df = session.createDataFrame(_arrow_table())
    else:
        df = session.createDataFrame(_arrow_table(), schema)
    assert df.schema.simpleString() == expected
    assert _rows(df) == ARROW_TABLE_ROWS


def test_catalog_function_class_name_is_none_without_a_class_as_pyspark(session):
    # As PySpark's, the className of a builtin that has no class (`!=`, `||`) is None.
    functions = {f.name: f for f in session.catalog.listFunctions()}
    assert functions["!="].className is None and functions["||"].className is None
    assert functions["abs"].className.startswith("org.apache.spark.sql.catalyst.")
    assert session.catalog.getFunction("abs").className == functions["abs"].className


def test_spark_conf_and_task_context_are_those_of_pyspark(session):
    # As PySpark's: pyspark exports the SparkConf, whose settings the builder sets on the
    # session, and the TaskContext, which a UDF uses on the server.
    from pyspark import SparkConf, TaskContext
    from pyspark.sql import functions as F

    key = "spark.sql.test.conf.of.a.spark.conf"
    conf = SparkConf().set(key, "v").setIfMissing(key, "other")
    try:
        assert SparkSession.builder.config(conf=conf).getOrCreate() is session
        assert session.conf.get(key) == "v"
    finally:
        session.conf.unset(key)
    partition = F.udf(lambda: TaskContext.get().partitionId(), "int")
    rows = session.range(0, 4, 1, 2).select(partition().alias("p")).distinct().collect()
    assert sorted(row.p for row in rows) == [0, 1]


def test_add_artifacts_reach_udfs_as_pyspark(session, tmp_path):
    import uuid
    import zipfile

    tag = uuid.uuid4().hex[:8]
    module, note, bundle = f"mod_{tag}", f"note_{tag}.txt", f"bundle_{tag}.zip"
    (tmp_path / f"{module}.py").write_text("def f():\n    return 'from the module'\n")
    (tmp_path / note).write_text("file content")
    with zipfile.ZipFile(tmp_path / bundle, "w") as archive:
        archive.writestr("inner.txt", "inside")
    session.addArtifacts(str(tmp_path / f"{module}.py"), pyfile=True)
    session.addArtifacts(str(tmp_path / note), file=True)
    session.addArtifacts(str(tmp_path / bundle) + f"#alias_{tag}", archive=True)

    @F.udf("string")
    def read(_):
        import importlib
        import os

        imported = importlib.import_module(module)
        # The directory of the session's files, which holds the module. Not SparkFiles: a
        # worker started with SPARK_SKIP_CONNECT_COMPAT_TESTS, as in CI, does not set it up.
        root = os.path.dirname(imported.__file__)
        return "|".join([
            imported.f(),
            open(os.path.join(root, note)).read(),
            open(os.path.join(root, f"alias_{tag}", "inner.txt")).read(),
        ])

    assert session.range(1).select(read("id")).first()[0] == "from the module|file content|inside"


def test_foreach_foreach_partition_and_to_json_as_pyspark(session):
    # As PySpark Connect's: foreach calls f with each row, and foreachPartition with an
    # iterator of the rows of each partition, on the server; toJSON is a DataFrame of each row
    # as a JSON object string.
    import re

    from pyspark.errors import PythonException

    df = session.createDataFrame(
        [(1, "a", b"x"), (2, None, None)], "id int, name string, bin binary"
    )

    def each(row):
        raise ValueError(f"row:{row!r}")

    with pytest.raises(PythonException, match=re.escape("row:Row(id=1, name='a', bin=b'x')")):
        df.where("id = 1").foreach(each)

    def partition(rows):
        raise ValueError(f"rows:{list(rows)!r}")

    with pytest.raises(
        PythonException,
        match=re.escape("rows:[Row(id=1, name='a', bin=b'x'), Row(id=2, name=None, bin=None)]"),
    ):
        df.coalesce(1).foreachPartition(partition)
    assert df.foreach(lambda row: None) is None
    assert df.foreachPartition(lambda rows: None) is None

    json = df.toJSON()
    assert json.columns == ["value"]
    assert json.collect() == [Row(value='{"id":1,"name":"a","bin":"eA=="}'), Row(value='{"id":2}')]


def test_progress_handler_that_runs_queries_without_end_raises_recursion_error():
    # The unguarded handler of SPARK-61153 runs a query for each progress of the queries it
    # runs. Past the bound of nested callbacks, the query fails with a RecursionError, which the
    # handlers' exceptions carry out of each execution, as PySpark's result loop does, instead of
    # the native stack overflowing and the process crashing (SIGSEGV). The outer query raises it.
    # The queries nest only when the server sends an in-progress update of each one. A query that
    # is shorter than the report interval (2 s) gets one only when the server wakes for its result
    # before the execution ends, which a fast runner may not do. With an interval of 10 ms, the
    # server sends them while the job runs, and each of these shuffles runs for several intervals.
    import subprocess
    import sys

    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live-server tests")
    result = subprocess.run([sys.executable, "-c", f"""
import sys
from pyspark.sql import SparkSession
spark = SparkSession.builder.remote({remote!r}).getOrCreate()
spark.conf.set("spark.connect.progress.reportInterval", "10ms")
spark.registerProgressHandler(
    lambda **kw: spark.range(100).repartition(20).count() if not kw["done"] else None)
try:
    print(spark.range(0, 2000, 1, 4).repartition(20).selectExpr("sum(id) as t").collect())
except RecursionError as error:
    print("raised RecursionError", error)
"""], env=dict(os.environ, TZ="UTC"), capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert "raised RecursionError" in result.stdout, (result.stdout, result.stderr)


def test_plans_of_a_few_hundred_levels_are_sent_as_pyspark(session):
    # Plans of a few hundred levels are sent, which the client grows the stack for, while the
    # server runs them up to about 510: Apache's test_recursion_handling_for_plan_logging builds
    # 400 withColumns, and a subquery of 300 levels under 33 more runs too. Both were a
    # RecursionError, or a crash.
    df = session.range(10)
    for x in range(400):
        df = df.withColumn(f"col{x}", F.lit(x))
    assert len(df.schema.fields) == 401
    assert df.count() == 10
    sub = session.range(1)
    for _ in range(300):
        sub = sub.limit(100)
    outer = session.range(10)
    for _ in range(32):
        outer = outer.limit(100)
    assert outer.filter(sub.exists()).count() == 10


def test_a_deep_plan_raises_recursion_error_as_pyspark():
    # As official PySpark 4.3.0, which raises RecursionError for each of these depths while it
    # builds the plan, before any request. The server rejected 1,000 and 1,500 levels without a
    # reason, and from 2,000 the process crashed (SIGSEGV), so this runs in a process of its own.
    # The session works afterwards.
    import subprocess
    import sys

    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live-server tests")
    result = subprocess.run([sys.executable, "-c", f"""
from pyspark.sql import SparkSession, functions as F
spark = SparkSession.builder.remote({remote!r}).getOrCreate()
for levels in [1000, 1500, 2000, 3000, 5000, 10000]:
    df = spark.range(10)
    for _ in range(levels):
        df = df.filter(F.col("id") >= 0)
    try:
        df.count()
    except RecursionError:
        print(levels, "RecursionError")
print(spark.range(3).count())
"""], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "1000 RecursionError",
        "1500 RecursionError",
        "2000 RecursionError",
        "3000 RecursionError",
        "5000 RecursionError",
        "10000 RecursionError",
        "3",
    ], result.stdout


def test_the_exception_of_a_done_call_fails_the_execution_as_pyspark(session):
    # As PySpark Connect's Progress.__exit__: the exception of a progress handler's done call goes
    # out of the execution, and a registration of a function, which PySpark runs without a
    # Progress, makes no call.
    from pyspark.sql.functions import udf

    error = ValueError("the done call raises")
    calls = []

    def failing(**kwargs):
        calls.append(kwargs["done"])
        if kwargs["done"]:
            raise error

    session.registerProgressHandler(failing)
    try:
        with pytest.raises(ValueError) as raised:
            session.range(10).collect()
        assert raised.value is error and calls[-1] is True
        with pytest.raises(ValueError):
            session.range(10).toPandas()
        calls.clear()
        session.udf.register("s16_f4_done_udf", udf(lambda: 1))
        assert calls == []
    finally:
        session.removeProgressHandler(failing)
    assert session.sql("SELECT s16_f4_done_udf()").collect()[0][0] == "1"


def test_progress_handler_runs_a_query_of_its_own_as_pyspark(session):
    # As PySpark's: a handler runs while the execution runs, and a query of the handler runs
    # too, the progress of the execution having no effect on it.
    nested = []
    started = []

    def handler(*, stages, inflight_tasks, operation_id, done):
        # The first call, while the result is read, of the progress or of its end.
        if not started:
            started.append(True)
            nested.append(session.range(4).count())

    session.registerProgressHandler(handler)
    try:
        total = (
            session.range(0, 2000000, 1, 4)
            .groupBy((F.col("id") % 97).alias("k"))
            .agg(F.sum("id").alias("s"))
            .selectExpr("sum(s) as t")
            .collect()
        )
    finally:
        session.removeProgressHandler(handler)
    assert nested == [4]
    assert total == [Row(t=1999999000000)]


def test_hook_runs_a_query_of_its_own_as_pyspark(session):
    # As PySpark's: an on_execute_plan hook runs a query, which runs without the hook.
    remote = os.environ["SPARK_REMOTE"]
    nested = []
    started = []

    class QueryHook(SparkSession.Hook):
        def on_execute_plan(self, request):
            if not started:
                started.append(True)
                nested.append(session.range(2).count())
            return request

    builder = SparkSession.builder.remote(remote)
    builder._registerHook(lambda _: QueryHook())
    spark = builder.create()
    try:
        assert spark.range(3).count() == 3
    finally:
        spark.stop()
    assert nested == [2]


def test_hook_appending_a_non_reattachable_option_keeps_the_execution_reattachable(session):
    # As PySpark's: the iterator appends its own reattachable option after the hooks ran, so
    # an option of a hook that says false does not make the execution non-reattachable.
    remote = os.environ["SPARK_REMOTE"]

    class NotReattachableHook(SparkSession.Hook):
        def on_execute_plan(self, request):
            request.request_options.add().reattach_options.reattachable = False
            return request

    builder = SparkSession.builder.remote(remote)
    builder._registerHook(lambda _: NotReattachableHook())
    spark = builder.create()
    try:
        assert spark.range(2).count() == 2
        assert spark.sql("select 1 as x").collect() == [Row(x=1)]
    finally:
        spark.stop()


def test_clone_session_copies_the_state_of_the_session_as_pyspark(session):
    # As PySpark's: the clone has the configurations and the temporary views of the session,
    # of an id of its own or of the one given, and changes to either do not show in the other.
    import uuid

    session.conf.set("spark.sql.shuffle.partitions", "7")
    session.range(2).createOrReplaceTempView("cloned_view")
    try:
        clone = session.cloneSession()
        assert clone.session_id != session.session_id
        assert clone.conf.get("spark.sql.shuffle.partitions") == "7"
        assert clone.table("cloned_view").count() == 2
        clone.conf.set("spark.sql.shuffle.partitions", "9")
        assert session.conf.get("spark.sql.shuffle.partitions") == "7"
        new_id = str(uuid.uuid4())
        assert session.cloneSession(new_id).session_id == new_id
    finally:
        session.conf.unset("spark.sql.shuffle.partitions")


def test_execution_info_is_that_of_each_dataframe_as_pyspark(session):
    a = session.range(3)
    assert a.executionInfo is None
    a.collect()
    assert a.executionInfo is not None
    # Another DataFrame, and one that an action runs derived from it, have none.
    b = session.range(5)
    b.count()
    b.first()
    assert b.executionInfo is None and a.select("id").executionInfo is None
    for run in (
        lambda df: df.toArrow(),
        lambda df: df.toPandas(),
        lambda df: df.createOrReplaceTempView("execution_info_view"),
        lambda df: df.write.mode("overwrite").format("noop").save(),
    ):
        df = session.range(2)
        run(df)
        assert df.executionInfo is not None
        assert df.executionInfo.operation_id != a.executionInfo.operation_id


def test_registered_udtfs_are_called_from_sql_as_pyspark(session):
    # As PySpark's spark.udtf.register: the server knows the UDTF by its name.
    @F.udtf(returnType="x int")
    class Count:
        def eval(self, n):
            for i in range(n):
                yield (i,)

    @F.arrow_udtf(returnType=StructType([StructField("y", LongType())]))
    class Doubled:
        def eval(self, x):
            import pyarrow.compute as pc

            yield pa.table({"y": pc.multiply(x, 2)})

    assert session.udtf.register("count_udtf", Count) is Count
    assert session.sql("SELECT * FROM count_udtf(2)").collect() == [Row(x=0), Row(x=1)]
    session.udtf.register("doubled_udtf", Doubled)
    assert session.sql("SELECT * FROM doubled_udtf(21L)").collect() == [Row(y=42)]


def test_writes_the_catalog_and_streaming_queries_release_the_gil_as_pyspark(session, tmp_path):
    # As PySpark's requests to the server, which release the GIL while they wait: another
    # Python thread runs during them.
    import threading
    import time

    def ticks_beside(call):
        ticks, stop = [0], threading.Event()

        def tick():
            while not stop.is_set():
                ticks[0] += 1
                time.sleep(0.001)

        thread = threading.Thread(target=tick)
        thread.start()
        try:
            time.sleep(0.05)
            before = ticks[0]
            call()
            return ticks[0] - before
        finally:
            stop.set()
            thread.join()

    from pyspark.sql.streaming import StreamingQueryListener

    class Listener(StreamingQueryListener):
        # The times of the progress of the query, which the listener runs in a thread.
        progress = []

        def onQueryStarted(self, event):
            pass

        def onQueryProgress(self, event):
            self.progress.append(time.time())

        def onQueryIdle(self, event):
            pass

        def onQueryTerminated(self, event):
            pass

    listener = Listener()
    session.streams.addListener(listener)
    stream = session.readStream.format("rate").load().writeStream.format("noop")
    query = stream.trigger(processingTime="200 milliseconds").start()
    try:
        # The query makes progress first, which the listener receives between waits.
        for _ in range(120):
            if Listener.progress:
                break
            time.sleep(0.25)
        start = time.time()
        assert ticks_beside(lambda: query.awaitTermination(3)) > 100
        end = time.time()
    finally:
        query.stop()
        session.streams.removeListener(listener)
    # The listener runs during awaitTermination too.
    assert any(start < at < end for at in Listener.progress)
    rows = session.range(2_000_000).selectExpr("id", "id * 2 AS v")
    assert ticks_beside(lambda: rows.write.parquet(str(tmp_path / "rows"))) > 10
    assert ticks_beside(lambda: [session.catalog.currentDatabase() for _ in range(20)]) > 10


def test_catalog_and_stream_start_errors_carry_their_operation_id(session):
    for run in (
        lambda: session.catalog.setCurrentDatabase("no_such_database_60960"),
        lambda: session.readStream.format("rate").load().writeStream.format("no_such_sink_60960").start(),
    ):
        with pytest.raises(Exception) as error:
            run()
        assert isinstance(error.value.operation_id, str) and error.value.operation_id


def test_udfs_that_capture_functions_pickle_them_by_name_as_pyspark(session):
    # As PySpark's, the functions are of pyspark.sql.functions.builtin: a UDF that captures one
    # pickles it by its name, of which the Python worker of the server imports PySpark's.
    sort = F.array_sort

    @F.udf("string")
    def name_of(value):
        return f"{sort.__module__}.{sort.__name__}"

    assert session.range(1).select(name_of("id")).first()[0] == (
        "pyspark.sql.functions.builtin.array_sort"
    )


def test_create_dataframe_with_a_schema_edited_in_place(session):
    # As PySpark: the nullable of the fields of a StructType is set in place, and a field is
    # added with add(), which keeps the names in step; the DataFrame has the schema of the struct.
    from pyspark.sql.types import DoubleType, LongType, StringType, StructField, StructType

    schema = StructType([StructField("a", LongType()), StructField("b", StringType())])
    for field in schema.fields:
        field.nullable = False
    schema.add("c", DoubleType(), False)
    df = session.createDataFrame([(1, "x", 2.0)], schema)
    assert df.schema.simpleString() == "struct<a:bigint,b:string,c:double>"
    assert [f.nullable for f in df.schema.fields] == [False, False, False]
    assert df.collect() == [Row(a=1, b="x", c=2.0)]


def test_create_dataframe_of_datetime_labels_keeps_ntz_values_as_pyspark(session):
    # As PySpark's: a datetime64 column under a DatetimeIndex label (`.T` of a time-indexed
    # frame) reads back unchanged with an explicit timestamp_ntz. The naive values are not
    # shifted by a time zone, of the session or of the client's local time, which is set to
    # Asia/Seoul here as well: a client in UTC does not show the shift.
    import pandas as pd

    frame = pd.DataFrame({"x": pd.Series(pd.to_datetime(["2020-01-02 03:00", None]))})
    frame.columns = pd.DatetimeIndex(["2020-01-01"])
    previous = session.conf.get("spark.sql.session.timeZone")
    previous_local = os.environ.get("TZ")
    session.conf.set("spark.sql.session.timeZone", "Asia/Seoul")
    os.environ["TZ"] = "Asia/Seoul"
    time.tzset()
    try:
        df = session.createDataFrame(frame, "p timestamp_ntz")
        assert df.schema.simpleString() == "struct<p:timestamp_ntz>"
        assert df.collect() == [Row(p=datetime(2020, 1, 2, 3)), Row(p=None)]
    finally:
        session.conf.set("spark.sql.session.timeZone", previous)
        if previous_local is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous_local
        time.tzset()


def test_create_dataframe_of_nullable_labels_keeps_ntz_values_as_pyspark(session):
    # As PySpark's, as for the datetime labels above: a datetime64 column under an Int64 label (a
    # pivot of `convert_dtypes()` data) reads back unchanged with an explicit timestamp_ntz, not
    # shifted by the time zone of the session or of the client, set to Asia/Seoul here.
    import pandas as pd

    frame = pd.DataFrame({"x": pd.Series(pd.to_datetime(["2020-01-02 03:00", None]))})
    frame.columns = pd.Index(pd.array([1], dtype="Int64"))
    previous = session.conf.get("spark.sql.session.timeZone")
    previous_local = os.environ.get("TZ")
    session.conf.set("spark.sql.session.timeZone", "Asia/Seoul")
    os.environ["TZ"] = "Asia/Seoul"
    time.tzset()
    try:
        df = session.createDataFrame(frame, "p timestamp_ntz")
        assert df.schema.simpleString() == "struct<p:timestamp_ntz>"
        assert df.collect() == [Row(p=datetime(2020, 1, 2, 3)), Row(p=None)]
    finally:
        session.conf.set("spark.sql.session.timeZone", previous)
        if previous_local is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous_local
        time.tzset()


def test_create_dataframe_from_empty_arrow_table(session):
    empty = pa.table({"a": pa.array([], pa.int64()), "b": pa.array([], pa.string())})
    df = session.createDataFrame(empty, "a long, b string")
    assert df.schema.simpleString() == "struct<a:bigint,b:string>"
    assert df.count() == 0
    # Without types, an empty table has no schema to infer, as an empty list does not.
    for args in ((empty,), (empty, ["x", "y"])):
        with pytest.raises(PySparkValueError) as error:
            session.createDataFrame(*args)
        assert error.value.getCondition() == "CANNOT_INFER_EMPTY_SCHEMA"


_Pair = namedtuple("_Pair", ["x", "y"])


def test_grouping_sets_return_the_rows_of_pyspark(session):
    # As PySpark's: the grouping columns are those of `cols`, null in the rows of a set that
    # does not hold them, and none without `cols`.
    df = session.createDataFrame(
        [(1, "x", 10), (1, "y", 20), (2, "x", 30)], "a int, b string, c int")
    total = F.sum("c").alias("s")
    with_columns = df.groupingSets([["a"], []], "a", "b").agg(total)
    assert with_columns.columns == ["a", "b", "s"]
    assert sorted(with_columns.collect(), key=lambda r: (r.a is None, r.a)) == [
        Row(a=1, b=None, s=30), Row(a=2, b=None, s=30), Row(a=None, b=None, s=60)]
    without_columns = df.groupingSets([["a"], ["b"]]).agg(total)
    assert without_columns.columns == ["s"]
    assert sorted(r.s for r in without_columns.collect()) == [20, 30, 30, 40]
    # Of ordinals too.
    assert df.groupingSets([[2]], 2).agg(total).columns == ["b", "s"]


def test_grouped_aggregations_name_and_take_columns_as_pyspark(session):
    # As PySpark's: count is named count, the numeric aggregations of no columns aggregate
    # every numeric column, and df.agg takes a dict.
    from pyspark.errors import PySparkTypeError

    df = session.createDataFrame([(1, "b", 2.0), (2, "b", 3.0)], "x int, k string, v double")
    grouped = df.groupBy("k")
    assert grouped.count().columns == ["k", "count"]
    assert grouped.count().orderBy("count").collect() == [Row(k="b", count=2)]
    assert grouped.sum().collect() == [Row(k="b", **{"sum(x)": 3, "sum(v)": 5.0})]
    assert grouped.min().columns == ["k", "min(x)", "min(v)"]
    assert df.groupBy().count().collect() == [Row(count=2)]
    assert df.agg({"x": "max"}).collect() == [Row(**{"max(x)": 2})]
    with pytest.raises(PySparkTypeError) as error:
        grouped.avg("k")
    assert error.value.getMessageParameters() == {"invalid_columns": "['k']"}


@pytest.mark.parametrize(
    "data, schema, columns, rows",
    [
        ([(1, 2)], None, ["_1", "_2"], [(1, 2)]),
        ([[1, 2]], ["a"], ["a", "_2"], [(1, 2)]),
        ([(1, 2), (3, 4)], ["a"], ["a", "_2"], [(1, 2), (3, 4)]),
        ([(1, 2)], [], ["_1", "_2"], [(1, 2)]),
        ([(1, 2)], ("a", "b"), ["a", "b"], [(1, 2)]),
        ([Row(x=1, y=2)], ["a", "b"], ["a", "b"], [(1, 2)]),
        ([{"y": 2, "x": 1}], ["a", "b"], ["a", "b"], [(1, 2)]),
        # A namedtuple is named by its fields, as a Row is.
        ([_Pair(1, 2), _Pair(3, 4)], None, ["x", "y"], [(1, 2), (3, 4)]),
        ([_Pair(1, 2)], ["a", "b"], ["a", "b"], [(1, 2)]),
    ],
)
def test_create_dataframe_rows_column_names(session, data, schema, columns, rows):
    df = session.createDataFrame(data, schema)
    assert df.columns == columns
    assert [tuple(r) for r in df.collect()] == rows
    assert df.isLocal() is (schema is None)


@pytest.mark.parametrize(
    "frame, schema, columns",
    [
        ({"a": [1, 2], "b": ["x", "y"]}, None, ["a", "b"]),
        ({"a": [1, 2], "b": ["x", "y"]}, ["c"], ["c", "_2"]),
        ({"a": [1, 2], "b": ["x", "y"]}, "a long, b string", ["a", "b"]),
        ({0: [1, 2], 1: ["x", "y"]}, None, ["0", "1"]),
    ],
)
def test_create_dataframe_pandas_column_names(session, frame, schema, columns):
    pd = pytest.importorskip("pandas")
    df = session.createDataFrame(pd.DataFrame(frame), schema)
    assert df.columns == columns
    assert [tuple(r) for r in df.collect()] == [(1, "x"), (2, "y")]
    assert not df.isLocal()


class _Point:
    def __init__(self, b, a):
        self.b = b
        self.a = a


@pytest.mark.parametrize(
    "data, schema, columns, rows",
    [
        # As pyspark, a dict row is read by name: missing keys are null, others ignored.
        ([{"b": 1, "a": "x"}], "b long, a string", ["b", "a"], [(1, "x")]),
        ([{"a": 1}], "a long, b string", ["a", "b"], [(1, None)]),
        ([{"a": 1, "c": 3}], "a long", ["a"], [(1,)]),
        # Without a schema, the fields are the rows' sorted keys, in the order first seen.
        ([{"a": 1}, {"b": 2}], None, ["a", "b"], [(1, None), (None, 2)]),
        ([{"b": 2, "a": 1}], None, ["a", "b"], [(1, 2)]),
        ([{"b": 2, "a": 1}], ["x", "y"], ["x", "y"], [(1, 2)]),
        ([{"a": 1}, {"a": 2, "b": 3}], ["x", "y"], ["x", "y"], [(1, None), (2, 3)]),
        # Other objects are read through their __dict__.
        ([_Point(2, "x")], None, ["a", "b"], [("x", 2)]),
        ([_Point(2, "x")], "a string, b long", ["a", "b"], [("x", 2)]),
    ],
)
def test_create_dataframe_dict_rows_by_name(session, data, schema, columns, rows):
    df = session.createDataFrame(data, schema)
    assert df.columns == columns
    assert [tuple(r) for r in df.collect()] == rows


@pytest.mark.parametrize(
    "data, simple_string, rows",
    [
        ([(None, 1), (2, 3)], "struct<_1:bigint,_2:bigint>", [(None, 1), (2, 3)]),
        ([{"a": None}, {"a": 1}], "struct<a:bigint>", [(None,), (1,)]),
    ],
)
def test_create_dataframe_infers_types_past_nulls(session, data, simple_string, rows):
    # A field takes the type of its first non-null value.
    df = session.createDataFrame(data)
    assert df.schema.simpleString() == simple_string
    assert [tuple(r) for r in df.collect()] == rows


@pytest.mark.parametrize("ddl", [False, True], ids=["StructType", "DDL"])
def test_reader_schema_keeps_field_metadata(session, tmp_path, ddl):
    (tmp_path / "t.csv").write_text("1,x\n2,y\n")
    path = str(tmp_path / "t.csv")
    if ddl:
        schema = "a long COMMENT 'the key', b string"
    else:
        schema = StructType(
            [StructField("a", LongType(), True, {"comment": "the key"}), StructField("b", StringType())]
        )
    for df in (session.read.schema(schema).csv(path), session.read.csv(path, schema=schema)):
        assert df.schema.fields[0].metadata == {"comment": "the key"}
        assert [tuple(r) for r in df.collect()] == [(1, "x"), (2, "y")]
    stream = session.readStream.schema(schema).csv(str(tmp_path))
    assert stream.isStreaming
    assert stream.schema.fields[0].metadata == {"comment": "the key"}


def test_bool_stream_options_and_bytes_table_options_are_read_by_the_server(session, tmp_path):
    # As PySpark's: the stream format methods send a bool as "True", which the server reads as
    # true, and createTable sends bytes that are UTF-8 as their text, as its compression.
    (tmp_path / "s").mkdir()
    schema = StructType([StructField("a", LongType())])
    stream = session.readStream.schema(schema).csv(str(tmp_path / "s"), header=True)
    assert stream.isStreaming
    session.sql("DROP TABLE IF EXISTS e2e_bytes_table_option")
    try:
        session.catalog.createTable(
            "e2e_bytes_table_option", source="parquet", schema=schema, compression=b"snappy")
        assert session.catalog.tableExists("e2e_bytes_table_option")
    finally:
        session.sql("DROP TABLE IF EXISTS e2e_bytes_table_option")


def test_reader_none_unsets_an_option_and_external_description_is_the_comment(session, tmp_path):
    # As PySpark's: DataFrameReader.option, options and load unset an option given None, while
    # the format methods skip None and keep it. createExternalTable's description is the
    # comment of createTable, not a property.
    import warnings

    (tmp_path / "t.csv").write_text("a,b\n1,2\n")
    path = str(tmp_path / "t.csv")
    assert session.read.option("header", "true").csv(path).columns == ["a", "b"]
    unset = session.read.option("header", "true").option("header", None)
    assert unset.format("csv").load(path).columns == ["_c0", "_c1"]
    unset = session.read.option("header", "true").options(header=None)
    assert unset.format("csv").load(path).columns == ["_c0", "_c1"]
    unset = session.read.option("header", "true").format("csv")
    assert unset.load(path, header=None).columns == ["_c0", "_c1"]
    kept = session.read.option("header", "true")
    assert kept.csv(path, header=None).columns == ["a", "b"]
    assert kept.format("csv").load(path).columns == ["a", "b"]
    session.range(2).write.mode("overwrite").parquet(str(tmp_path / "ext"))
    session.sql("DROP TABLE IF EXISTS i3_described_external")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            session.catalog.createExternalTable(
                "i3_described_external", path=str(tmp_path / "ext"), source="parquet",
                description="the table")
        assert session.catalog.getTable("i3_described_external").description == "the table"
        comments = session.sql("DESCRIBE TABLE EXTENDED i3_described_external").filter(
            "col_name = 'Comment'").collect()
        assert [tuple(r)[:2] for r in comments] == [("Comment", "the table")]
        keys = [r.key for r in session.sql("SHOW TBLPROPERTIES i3_described_external").collect()]
        assert not any("description" in key for key in keys)
    finally:
        session.sql("DROP TABLE IF EXISTS i3_described_external")


def test_reader_writer_and_catalog_return_the_results_of_pyspark(session, tmp_path):
    # As PySpark's: csv, json and xml of a DataFrame of strings parse them, the readers load
    # every path, and createTable creates the table on the call, of its schema and options.
    import warnings

    strings = session.createDataFrame([("1;x",), ("2;y",)], "value string")
    assert session.read.csv(strings, sep=";", schema="a INT, b STRING").collect() == [
        Row(a=1, b="x"), Row(a=2, b="y")]
    assert session.read.csv(strings, sep=";").columns == ["_c0", "_c1"]
    documents = session.createDataFrame([('{"a": 1, "b": "x"}',)], "value string")
    assert session.read.json(documents).collect() == [Row(a=1, b="x")]
    assert session.read.json(documents, schema=StructType([StructField("a", LongType())])) \
        .collect() == [Row(a=1)]
    elements = session.createDataFrame([("<row><a>1</a></row>",)], "value string")
    assert session.read.xml(elements, rowTag="row").collect() == [Row(a=1)]
    for name in ("p1", "p2"):
        session.range(2).write.mode("overwrite").parquet(str(tmp_path / name))
    paths = [str(tmp_path / "p1"), str(tmp_path / "p2")]
    assert session.read.parquet(*paths).count() == 4
    assert session.read.format("parquet").load(paths).count() == 4
    session.range(3).write.mode(None).json(str(tmp_path / "j"))
    assert session.read.json([str(tmp_path / "j")]).count() == 3
    session.sql("DROP TABLE IF EXISTS e2e_created_eagerly")
    try:
        schema = StructType([StructField("a", LongType()), StructField("b", StringType())])
        created = session.catalog.createTable(
            "e2e_created_eagerly", source="parquet", schema=schema, description="d",
            compression="snappy")
        assert session.catalog.tableExists("e2e_created_eagerly")
        assert created.columns == ["a", "b"]
        assert session.catalog.getTable("e2e_created_eagerly").description == "d"
    finally:
        session.sql("DROP TABLE IF EXISTS e2e_created_eagerly")
    session.sql("DROP TABLE IF EXISTS e2e_external_eagerly")
    try:
        with warnings.catch_warnings(record=True) as record:
            warnings.simplefilter("always")
            external = session.catalog.createExternalTable(
                "e2e_external_eagerly", path=paths[0], source="parquet")
            assert session.catalog.tableExists("e2e_external_eagerly", "default")
        assert external.count() == 2
        assert [w.category for w in record] == [FutureWarning, FutureWarning]
    finally:
        session.sql("DROP TABLE IF EXISTS e2e_external_eagerly")


def test_window_specs_captured_by_foreach_batch_functions_run_as_pyspark(session, tmp_path):
    # As PySpark's: a WindowSpec that a foreachBatch function captures unpickles on the Python
    # worker of the server, which computes the window with it.
    from pyspark.sql import Window

    (tmp_path / "input.txt").write_text("hello\nthis\n")
    window = (
        Window.partitionBy()
        .orderBy(F.col("value").desc())
        .rowsBetween(Window.unboundedPreceding, Window.currentRow)
    )
    table = "foreach_batch_captured_window"

    def func(df, batch_id):
        df.select("value", F.row_number().over(window).alias("r")).write.mode(
            "append"
        ).saveAsTable(table)

    query = session.readStream.format("text").load(str(tmp_path)).writeStream.foreachBatch(func)
    query = query.start()
    try:
        query.processAllAvailable()
        rows = session.table(table).collect()
        assert sorted((row.value, row.r) for row in rows) == [("hello", 2), ("this", 1)]
    finally:
        query.stop()
        session.sql(f"DROP TABLE IF EXISTS {table}").collect()


def test_table_arguments_are_those_of_pyspark_sql_table_arg(session):
    # As the PySpark client: DataFrame.asTable() is a TableArg of pyspark.sql.table_arg, and a
    # TableValuedFunctionArgument of pyspark.sql.tvf_argument.
    from pyspark.sql.table_arg import TableArg
    from pyspark.sql.tvf_argument import TableValuedFunctionArgument

    table = session.range(3).asTable()
    assert isinstance(table, TableArg)
    assert isinstance(table, TableValuedFunctionArgument)
    assert isinstance(table.partitionBy("id"), TableArg)




def test_subqueries_and_table_arguments_return_the_rows_of_pyspark(session):
    # As PySpark's DataFrame.scalar(), DataFrame.exists(), Column.isin(df) and asTable():
    # scalar, correlated and IN subqueries, and the rows of a DataFrame as a UDTF argument.
    from pyspark.errors import IllegalArgumentException

    t1 = session.createDataFrame([(1, 2), (3, 4)], "a int, b int").alias("t1")
    t2 = session.createDataFrame([(1, 5), (5, 6), (3, 7)], "a int, c int").alias("t2")
    largest = t2.select(F.max("c")).scalar()
    assert sorted(t1.select("a", largest.alias("m")).collect()) == [Row(a=1, m=7), Row(a=3, m=7)]
    correlated = t2.where(F.col("t2.a") == F.col("t1.a").outer())
    assert sorted(t1.select("a", correlated.select(F.max("c")).scalar().alias("m")).collect()) == [
        Row(a=1, m=5), Row(a=3, m=7)
    ]
    assert t1.where(~correlated.exists()).collect() == []
    above = t2.where(F.col("c") > t2.select(F.min("c")).scalar()).select("a")
    assert t1.where(F.col("a").isin(above)).collect() == [Row(a=3, b=4)]
    pairs = session.createDataFrame([(1, 2)], "x int, y int")
    assert t1.where(F.struct("a", "b").isin(pairs)).collect() == [Row(a=1, b=2)]

    @F.udtf(returnType="id bigint, n int")
    class Numbered:
        def __init__(self):
            self.n = 0

        def eval(self, row):
            self.n += 1
            yield row["id"], self.n

    table = session.range(4).asTable().withSinglePartition().orderBy(F.col("id").desc())
    assert sorted(Numbered(table).collect()) == [
        Row(id=0, n=4), Row(id=1, n=3), Row(id=2, n=2), Row(id=3, n=1)
    ]
    assert sorted(Numbered(row=session.range(2).asTable()).collect()) == [
        Row(id=0, n=1), Row(id=1, n=1)
    ]
    with pytest.raises(IllegalArgumentException, match="before orderBy"):
        session.range(1).asTable().orderBy("id")


def test_columns_of_another_session_do_not_resolve_in_this_one(session):
    # As PySpark's, plan IDs come from one counter of the process, so a column of a DataFrame of
    # another session names no node of this one: CANNOT_RESOLVE_DATAFRAME_COLUMN, not the
    # column of the same name.
    other = session.newSession()
    a = session.createDataFrame([(1,)], "id int")
    b = other.createDataFrame([(2,)], "id int")
    with pytest.raises(AnalysisException) as e:
        a.select(b.id).collect()
    assert e.value.getCondition() == "CANNOT_RESOLVE_DATAFRAME_COLUMN"
    with pytest.raises(AnalysisException) as e:
        a.filter(b.id == 2).count()
    assert e.value.getCondition() == "CANNOT_RESOLVE_DATAFRAME_COLUMN"
    assert a.select(a.id).collect() == [Row(id=1)]


def test_tags_are_those_of_each_thread_as_pyspark(session):
    # As PySpark's: the tags of a thread are on its operations only, so interruptTag of the tag
    # of one thread interrupts its operation, but not that of another thread.
    import threading
    import time

    from pyspark.sql import functions as F

    slow = F.udf(lambda x: (time.sleep(8), x)[1], "long")
    results = {}

    def run(name, tag):
        if tag:
            session.addTag(tag)
        try:
            results[name] = session.range(1).select(slow("id")).collect()
        except Exception as e:  # noqa: BLE001
            results[name] = e
        finally:
            session.clearTags()

    tagged = threading.Thread(target=run, args=("tagged", "tag_of_a_thread"))
    tagged.start()
    time.sleep(1)
    untagged = threading.Thread(target=run, args=("untagged", None))
    untagged.start()
    time.sleep(3)
    assert session.getTags() == set()
    interrupted = session.interruptTag("tag_of_a_thread")
    tagged.join(60)
    untagged.join(60)
    assert len(interrupted) == 1
    assert isinstance(results["tagged"], Exception)
    assert [tuple(row) for row in results["untagged"]] == [(0,)]


def test_grouped_map_functions_take_iterators_by_their_type_hints(session):
    from collections.abc import Iterator

    import numpy as np
    import pandas as pd

    def scaled(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        for frame in batches:
            yield frame.assign(v=frame.v * 10)

    def shifted(key: tuple[np.int64], batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        for frame in batches:
            yield frame.assign(v=frame.v + key[0])

    def batches(it: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        yield from it

    def keyed_batches(
        key: tuple[pa.Scalar, ...], it: Iterator[pa.RecordBatch]
    ) -> Iterator[pa.RecordBatch]:
        yield from it

    df = session.createDataFrame([(1, 1.0), (1, 2.0), (2, 3.0)], ["k", "v"])
    schema = StructType([StructField("k", LongType()), StructField("v", DoubleType())])
    grouped = df.groupBy("k")
    for result, expected in (
        (grouped.applyInPandas(scaled, schema), [(1, 10.0), (1, 20.0), (2, 30.0)]),
        (grouped.applyInPandas(shifted, schema), [(1, 2.0), (1, 3.0), (2, 5.0)]),
        (grouped.applyInArrow(batches, schema), [(1, 1.0), (1, 2.0), (2, 3.0)]),
        (grouped.applyInArrow(keyed_batches, schema), [(1, 1.0), (1, 2.0), (2, 3.0)]),
    ):
        assert sorted(tuple(r) for r in result.collect()) == expected


def test_dataframe_columns_resolve_against_their_dataframe_as_pyspark(session):
    # As PySpark Connect's: df[name] and df.name refer to the columns of their DataFrame by its
    # plan ID, so a join selects the columns of each side, and an ambiguous self-join of a
    # derived DataFrame raises PySpark's error.
    left = session.createDataFrame([(1, "a"), (2, "b")], "id int, v string")
    right = session.createDataFrame([(1, "x"), (3, "y")], "id int, w string")
    joined = left.join(right, left["id"] == right["id"])
    assert joined.select(left["id"], right["w"]).collect() == [Row(id=1, w="x")]
    assert joined.select(left.v, right.w).collect() == [Row(v="a", w="x")]
    assert joined.select(right["*"]).collect() == [Row(id=1, w="x")]
    assert joined.select(left.colRegex("`v`")).collect() == [Row(v="a")]
    base = session.range(3)
    derived = base.withColumn("x", F.col("id") * 2)
    with pytest.raises(AnalysisException) as error:
        base.join(derived, base["id"] == derived["id"]).select(base["id"]).collect()
    assert error.value.getCondition() == "AMBIGUOUS_COLUMN_REFERENCE"


def test_reader_and_writer_check_their_arguments_as_pyspark(session, tmp_path):
    # As PySpark Connect's: a write of an unsupported save mode raises UNSUPPORTED_OPERATION
    # and writes nothing; format(None) writes and reads the default source, Parquet; the reader
    # refuses a schema in changes and asserts the bounds of a JDBC column; text of None reads
    # no rows.
    from pyspark.errors import PySparkAttributeError

    df = session.createDataFrame([(1, "a"), (2, "b")], "id int, s string")
    path = str(tmp_path / "out")
    with pytest.raises(PySparkValueError) as e:
        df.write.mode("bad").save(path)
    assert e.value.getCondition() == "UNSUPPORTED_OPERATION"
    assert e.value.getMessageParameters() == {"operation": "bad"}
    assert not os.path.exists(path)
    df.write.format("json").format(None).save(path)
    assert sorted(session.read.parquet(path).collect()) == [Row(id=1, s="a"), Row(id=2, s="b")]
    assert session.read.format("json").format(None).load(path).count() == 2
    with pytest.raises(AnalysisException) as e:
        session.read.schema("a int").changes("t")
    assert e.value.getCondition() == "_LEGACY_ERROR_TEMP_1189"
    with pytest.raises(AssertionError, match="lowerBound can not be None"):
        session.read.jdbc("jdbc:x", "t", column="c")
    with pytest.raises(PySparkAttributeError) as e:
        session.read._jreader
    assert e.value.getCondition() == "JVM_ATTRIBUTE_NOT_SUPPORTED"
    assert session.read.text(None).count() == 0


def test_readers_and_writers_set_their_options_of_set_opts_as_pyspark(session, tmp_path):
    # As PySpark's OptionUtils._set_opts, which pandas-on-Spark's to_csv and to_json call: the
    # schema of a reader, and the options that are not None.
    df = session.createDataFrame([(1, "a"), (2, None)], ["x", "y"])
    writer = df.write.mode("overwrite")
    assert writer._set_opts(sep="|", header=True, nullValue="NA", quote=None) is None
    writer.format("csv").save(str(tmp_path / "out"))
    lines = sorted(
        line for path in (tmp_path / "out").glob("*.csv") for line in path.read_text().splitlines()
    )
    assert lines == ["1|a", "2|NA", "x|y", "x|y"]
    reader = session.read
    reader._set_opts(schema="x INT, y STRING", sep="|", header=True, nullValue="NA", mode=None)
    assert sorted(reader.format("csv").load(str(tmp_path / "out")).collect()) == [
        Row(x=1, y="a"), Row(x=2, y=None),
    ]
    with pytest.raises(AttributeError):
        df.write._set_opts(schema="x INT")
    assert callable(session.readStream._set_opts) and callable(df.writeTo("t")._set_opts)


def test_writer_v2_takes_its_arguments_as_pyspark(session):
    # As PySpark's DataFrameWriterV2: tableProperty(property, None) leaves the property unset,
    # partitionedBy raises NOT_EXPECTED_TYPE of col for a list, and overwrite takes a column
    # name, which the server takes: a table of the session catalog does not overwrite by a
    # condition.
    import uuid

    from pyspark.errors import AnalysisException, PySparkTypeError

    name = f"writer_v2_arguments_{uuid.uuid4().hex[:8]}"
    df = session.createDataFrame([(1, True), (2, False)], "id int, flag boolean")
    try:
        writer = df.writeTo(name).using("parquet").tableProperty("a", "1")
        writer.tableProperty("a", None).tableProperty("b", "2").create()
        rows = session.sql(f"SHOW TBLPROPERTIES {name}").collect()
        properties = {row.key: row.value for row in rows}
        assert properties.get("b") == "2" and "a" not in properties
        with pytest.raises(PySparkTypeError) as e:
            df.writeTo(name).partitionedBy(["id"])
        assert e.value.getCondition() == "NOT_EXPECTED_TYPE"
        with pytest.raises(AnalysisException):
            df.writeTo(name).overwrite("flag")
    finally:
        session.sql(f"DROP TABLE IF EXISTS {name}")


def test_streaming_listeners_are_removed_once_per_registration_as_pyspark(session):
    # As PySpark's StreamingQueryManager: a listener added twice is in the bus twice, and each
    # removeListener removes one of its registrations; a removal through another session, whose
    # bus does not have the listener, leaves it in this bus. Once removed, it gets no events.
    from pyspark.sql.streaming import StreamingQueryListener

    class Listener(StreamingQueryListener):
        def __init__(self):
            self.events = []

        def onQueryStarted(self, event):
            self.events.append("started")

        def onQueryProgress(self, event):
            self.events.append("progress")

        def onQueryIdle(self, event):
            pass

        def onQueryTerminated(self, event):
            self.events.append("terminated")

    listener = Listener()
    other = session.newSession()
    try:
        session.streams.addListener(listener)
        session.streams.addListener(listener)
        assert session.streams._sqlb._listener_bus == [listener, listener]
        session.streams.removeListener(listener)
        assert session.streams._sqlb._listener_bus == [listener]
        session.streams.removeListener(listener)
        assert session.streams._sqlb._listener_bus == []
        session.streams.removeListener(listener)
        assert session.streams._sqlb._listener_bus == []

        session.streams.addListener(listener)
        other.streams.removeListener(listener)
        assert session.streams._sqlb._listener_bus == [listener]
        assert other.streams._sqlb._listener_bus == []
        session.streams.removeListener(listener)
        assert session.streams._sqlb._listener_bus == []

        query = session.readStream.format("rate").load().writeStream.format("noop")
        query = query.trigger(processingTime="200 milliseconds").start()
        try:
            time.sleep(2)
        finally:
            query.stop()
        time.sleep(1)
        assert listener.events == []
    finally:
        for registered in list(session.streams._sqlb._listener_bus):
            session.streams.removeListener(registered)
        other.stop()


def test_collected_rows_take_the_names_of_the_schema_of_the_result_as_pyspark(session):
    # As PySpark Connect's: the fields of nested structs have the names of the schema, which keep
    # duplicate names, and collect raises for a calendar interval.
    from pyspark.errors import PySparkTypeError

    df = session.sql(
        "SELECT named_struct('a', 1, 'a', 2) AS v, array(named_struct('b', 1, 'b', 'x')) AS w")
    row = df.collect()[0]
    assert row.v.__fields__ == ["a", "a"] and tuple(row.v) == (1, 2) and row.v.a == 1
    assert row.w[0].__fields__ == ["b", "b"]
    assert row.asDict(recursive=True) == {"v": {"a": 2}, "w": [{"b": "x"}]}
    assert next(df.toLocalIterator()).v.__fields__ == ["a", "a"]
    with pytest.raises(PySparkTypeError) as e:
        session.sql("SELECT make_interval(1) AS v").collect()
    assert e.value.getCondition() == "UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION"
    assert e.value.getMessageParameters() == {"data_type": "month_day_nano_interval"}


def test_collect_of_a_string_that_is_not_utf8_raises_as_pyspark(session):
    # As PySpark's: the bytes of the string are not UTF-8, and collect raises the UnicodeDecodeError
    # of Python's decoder for them, while a valid string of the same query collects.
    with pytest.raises(UnicodeDecodeError) as e:
        session.sql("SELECT CAST(X'C3' AS STRING) AS s").collect()
    assert type(e.value) is UnicodeDecodeError
    assert str(e.value) == (
        "'utf-8' codec can't decode byte 0xc3 in position 0: unexpected end of data")
    assert session.sql("SELECT CAST(X'61' AS STRING) AS s").collect() == [Row(s="a")]


def test_a_string_that_is_not_utf8_is_named_past_binary_values_as_pyspark(session):
    # As PySpark's: the binary values before the string are values, not strings, so the error is
    # that of the string, whether the binary value is a column or nested in an array, for each
    # action that reads rows.
    message = "'utf-8' codec can't decode byte 0xc3 in position 0: unexpected end of data"
    for query in [
        "SELECT X'FF' AS b, CAST(X'C3' AS STRING) AS s",
        "SELECT ARRAY(X'FF') AS b, CAST(X'C3' AS STRING) AS s",
    ]:
        actions = [
            lambda df: df.collect(),
            lambda df: df.first(),
            lambda df: df.head(),
            lambda df: df.take(1),
            lambda df: df.tail(1),
            lambda df: list(df.toLocalIterator()),
        ]
        for action in actions:
            with pytest.raises(UnicodeDecodeError) as e:
                action(session.sql(query))
            assert type(e.value) is UnicodeDecodeError
            assert str(e.value) == message
    # A binary value that is valid as a value still collects as bytes.
    assert session.sql("SELECT X'FF' AS b, 'a' AS s").collect() == [Row(b=b"\xff", s="a")]


def test_collect_reads_a_result_across_stream_ends(session):
    # Takes several response streams when the server's senderMaxStreamSize is small.
    n = 300_000
    assert len(session.range(n).collect()) == n
    assert session.range(n).toArrow().num_rows == n


def test_binary_values_of_collected_rows_follow_binary_as_bytes_as_pyspark(session):
    # As PySpark Connect's `DataFrame._get_binary_as_bytes`: the binary values of the rows of
    # collect and toLocalIterator, and so of first, head, take and tail, also those in arrays,
    # maps and structs, are bytes, unless spark.sql.execution.pyspark.binaryAsBytes is set to
    # another value than true, case-insensitively, which makes them bytearray.
    df = session.sql(
        "SELECT CAST('ab' AS BINARY) AS b, array(CAST('c' AS BINARY)) AS a, "
        "named_struct('x', CAST('d' AS BINARY)) AS s, map('k', CAST('e' AS BINARY)) AS m"
    )

    def types(row):
        return [type(value) for value in (row.b, row.a[0], row.s.x, row.m["k"])]

    key = "spark.sql.execution.pyspark.binaryAsBytes"
    try:
        for value, expected in [("false", bytearray), ("TRUE", bytes), ("FALSE", bytearray)]:
            session.conf.set(key, value)
            assert types(df.collect()[0]) == [expected] * 4
            assert types(next(df.toLocalIterator())) == [expected] * 4
            assert types(next(df.toLocalIterator(prefetchPartitions=True))) == [expected] * 4
            for row in [df.first(), df.head(), df.head(1)[0], df.take(1)[0], df.tail(1)[0]]:
                assert types(row) == [expected] * 4
            assert df.collect() == [Row(b=b"ab", a=[b"c"], s=Row(x=b"d"), m={"k": b"e"})]
        # A result without a binary value is the same either way.
        assert session.range(1).collect() == [Row(id=0)]
    finally:
        session.conf.unset(key)
    assert types(df.collect()[0]) == [bytes] * 4
    assert types(next(df.toLocalIterator())) == [bytes] * 4




def test_with_metadata_and_pivot_of_a_missing_column_are_dataframe_errors_as_pyspark(session):
    # As PySpark's: withMetadata and pivot refer to the column through the DataFrame, so a
    # missing column is CANNOT_RESOLVE_DATAFRAME_COLUMN, not UNRESOLVED_COLUMN.
    df = session.createDataFrame([("a", 1)], "k string, v int")
    with pytest.raises(AnalysisException) as error:
        df.withMetadata("nope", {"a": "b"}).collect()
    assert error.value.getCondition() == "CANNOT_RESOLVE_DATAFRAME_COLUMN"
    with pytest.raises(AnalysisException) as error:
        df.groupBy("k").pivot("nope").count().collect()
    assert error.value.getCondition() == "CANNOT_RESOLVE_DATAFRAME_COLUMN"
    assert df.withMetadata("v", {"a": "b"}).schema["v"].metadata == {"a": "b"}


def test_internal_values_of_collected_rows_are_those_of_spark_as_pyspark(session):
    # As PySpark's: the toInternal of a collected row is the internal value of Spark that the
    # server computes, the days and the microseconds since the epoch, and its fromInternal is
    # the row.
    df = session.sql(
        "select date'2020-01-01' as d, timestamp'2020-01-01 12:30:15.123456' as ts, "
        "timestamp_ntz'2020-01-01 12:30:15.123456' as ntz, "
        "interval '1 02:03:04.000005' day to second as i, array(date'2020-01-02') as a, "
        "map('k', date'2020-01-03') as m")
    row = df.first()
    days, micros, next_days, last_days = session.sql(
        "select unix_date(date'2020-01-01'), unix_micros(timestamp'2020-01-01 12:30:15.123456'), "
        "unix_date(date'2020-01-02'), unix_date(date'2020-01-03')").first()
    internal = df.schema.toInternal(row)
    assert internal == (days, micros, micros, 93784000005, [next_days], {"k": last_days})
    assert df.schema.fromInternal(internal) == row




def test_time_values_round_trip_as_pyspark(session):
    # As PySpark's: TIME values are sent, and collected, as datetime.time, when the session
    # enables the TIME type.
    import datetime

    t = datetime.time(1, 2, 3, 456789)
    session.conf.set("spark.sql.timeType.enabled", "true")
    try:
        assert session.createDataFrame([(t,)]).collect() == [Row(_1=t)]
        assert session.createDataFrame([(t,)], "t time").collect() == [Row(t=t)]
        assert session.range(1).select(F.lit(t).alias("t")).collect() == [Row(t=t)]
        assert session.sql("SELECT TIME'12:34:56.789' AS t").collect() == [
            Row(t=datetime.time(12, 34, 56, 789000))
        ]
        # Without it, the server does not support the type, as for PySpark. It is set to
        # false, as the server enables it by default when SPARK_TESTING is set.
        session.conf.set("spark.sql.timeType.enabled", "false")
        with pytest.raises(AnalysisException) as error:
            session.createDataFrame([(t,)], "t time").collect()
        assert error.value.getCondition() == "UNSUPPORTED_TIME_TYPE"
    finally:
        session.conf.unset("spark.sql.timeType.enabled")


def test_timedelta_is_a_day_time_interval_as_pyspark(session):
    import datetime

    import pyarrow as pa

    values = [datetime.timedelta(days=1, seconds=2, microseconds=3),
              datetime.timedelta(seconds=-1, microseconds=5)]
    df = session.range(1).select(*[F.lit(v).alias(f"t{i}") for i, v in enumerate(values)])
    assert df.dtypes == [("t0", "interval day to second"), ("t1", "interval day to second")]
    assert tuple(df.first()) == tuple(values)
    df = session.createDataFrame([(v,) for v in values] + [(None,)], ["t"])
    assert df.dtypes == [("t", "interval day to second")]
    assert [r.t for r in df.collect()] == values + [None]
    # As PyArrow's conversion to duration[us]: an int, or a float truncated, is microseconds.
    df = session.createDataFrame([(5,), (1.5,)], "d interval day to second")
    assert [r.d for r in df.collect()] == [
        datetime.timedelta(microseconds=5), datetime.timedelta(microseconds=1)]
    with pytest.raises(pa.ArrowTypeError, match="Expected integer, got bool"):
        session.createDataFrame([(True,)], "d interval day to second")
    with pytest.raises(pa.ArrowTypeError, match="object of type <class 'str'> cannot be converted to int"):
        session.createDataFrame([("x",)], "d interval day to second")


def test_errors_carry_the_dataframe_query_contexts_of_pyspark(session):
    # As PySpark's: an error in an expression that col() or a Column method built reports its
    # DataFrameQueryContext, of the function and of its call site in this file.
    import inspect

    from pyspark.errors import ArithmeticException, QueryContextType

    def contexts(df):
        with pytest.raises((AnalysisException, ArithmeticException)) as error:
            df.collect()
        return [
            (c.contextType(), c.fragment(), c.callSite().rsplit("/", 1)[-1])
            for c in error.value.getQueryContext()
        ]

    here = f"{os.path.basename(__file__)}:{{}}"
    line = inspect.currentframe().f_lineno
    assert contexts(session.range(1).select("no_such_col")) == [
        (QueryContextType.DataFrame, "col", here.format(line + 1))
    ]
    assert contexts(session.range(1).select(F.col("nope") + 1)) == [
        (QueryContextType.DataFrame, "col", here.format(line + 4))
    ]
    session.conf.set("spark.sql.ansi.enabled", "true")
    assert contexts(session.range(1).select(F.lit(1) / F.lit(0))) == [
        (QueryContextType.DataFrame, "__truediv__", here.format(line + 8))
    ]
    session.conf.unset("spark.sql.ansi.enabled")
    # repartitionById makes a column of a name with col(), while a UDF call refers to a
    # column by its name without an origin.
    assert contexts(session.range(1).repartitionById(2, "nope")) == [
        (QueryContextType.DataFrame, "col", here.format(line + 14))
    ]
    udf = F.udf(lambda x: x, "long")
    assert contexts(session.range(1).select(udf("nope"), udf(x="nope2"))) == []
    # sampleBy takes a column of a name with col(), as PySpark's F._to_col.
    assert contexts(session.range(1).sampleBy("nope", {0: 1.0})) == [
        (QueryContextType.DataFrame, "col", here.format(line + 20))
    ]
    # unpivot takes its columns of names with col(), as PySpark's F._to_col.
    assert contexts(session.range(1).unpivot("nope", ["id"], "v", "w")) == [
        (QueryContextType.DataFrame, "col", here.format(line + 24))
    ]


def test_sessions_raise_the_errors_of_tags_tables_and_jvm_attributes_as_pyspark(session):
    # As PySpark Connect's: the errors of a tag of a comma, of an empty tag, of a name of a
    # table that is not a str, and of the SparkContext of a classic session.
    from pyspark.errors import PySparkAttributeError, PySparkTypeError, PySparkValueError

    with pytest.raises(PySparkValueError) as e:
        session.addTag("a,b")
    assert e.value.getCondition() == "VALUE_ALLOWED"
    with pytest.raises(PySparkValueError) as e:
        session.removeTag("")
    assert e.value.getCondition() == "VALUE_NOT_NON_EMPTY_STR"
    with pytest.raises(PySparkTypeError) as e:
        session.table(1)
    assert e.value.getCondition() == "NOT_EXPECTED_TYPE"
    with pytest.raises(PySparkAttributeError) as e:
        session.sparkContext
    assert e.value.getCondition() == "JVM_ATTRIBUTE_NOT_SUPPORTED"




def test_server_errors_carry_their_condition_and_details(session):
    from pyspark.errors import AnalysisException, ArithmeticException, ParseException
    from pyspark.errors.exceptions.connect import SparkConnectGrpcException, SQLQueryContext

    with pytest.raises(ParseException) as error:
        session.sql("SELEC 1").collect()
    e = error.value
    assert isinstance(e, SparkConnectGrpcException)
    assert e.getCondition() == "PARSE_SYNTAX_ERROR"
    assert e.getSqlState() == "42601"
    assert e.getMessageParameters() == {"error": "'SELEC'", "hint": ""}
    # The server's message, which names the condition once.
    assert str(e).count("[PARSE_SYNTAX_ERROR]") == 1
    assert e.getStackTrace().startswith("org.apache.spark.sql.catalyst.parser.ParseException")

    with pytest.raises(AnalysisException) as error:
        session.table("no_such_table_xyz").collect()
    e = error.value
    assert e.getCondition() == "TABLE_OR_VIEW_NOT_FOUND"
    assert e.getSqlState() == "42P01"
    assert list(e.getMessageParameters()) == ["relationName", "searchPath"]
    assert e.getMessageParameters()["relationName"] == "`no_such_table_xyz`"

    with pytest.raises(ArithmeticException) as error:
        session.sql("SELECT 1/0").collect()
    e = error.value
    assert e.getCondition() == "DIVIDE_BY_ZERO"
    [context] = e.getQueryContext()
    assert isinstance(context, SQLQueryContext)
    assert context.fragment() == "1/0"
    # spark.sql.connect.serverStacktrace.enabled is true by default.
    assert "\n\nJVM stacktrace:\norg.apache.spark.SparkArithmeticException" in str(e)


def test_progress_handlers_receive_the_progress_of_executions_as_pyspark(session):
    # As PySpark Connect's: a handler is called with the keyword arguments of ProgressHandler,
    # of each execution: its stages, its running tasks, its operation, and, last, done.
    from pyspark.sql.connect.shell.progress import StageInfo

    calls = []

    def handler(*, stages, inflight_tasks, operation_id, done):
        calls.append((list(stages), inflight_tasks, operation_id, done))

    session.registerProgressHandler(handler)
    try:
        session.range(0, 8, 1, 4).selectExpr("sum(id)").collect()
    finally:
        session.removeProgressHandler(handler)
    assert calls and calls[-1][3] is True
    assert [done for *_, done in calls].count(True) == 1
    assert len({operation_id for _, _, operation_id, _ in calls}) == 1
    assert all(isinstance(stage, StageInfo) for stages, *_ in calls for stage in stages)
    assert all(isinstance(inflight, int) for _, inflight, _, _ in calls)




def test_apply_in_pandas_with_state_runs_as_pyspark(session, tmp_path):
    import pandas as pd

    def count(key, frames, state):
        total = sum(len(frame) for frame in frames)
        state.update((total,))
        yield pd.DataFrame({"k": [key[0]], "c": [total]})

    output = StructType([StructField("k", LongType()), StructField("c", LongType())])
    state = StructType([StructField("c", LongType())])
    df = session.createDataFrame([(1,), (1,), (2,)], "k long")
    # The server takes the function in a streaming query only, as with PySpark.
    with pytest.raises(Exception, match="applyInPandasWithState is unsupported in batch query"):
        df.groupBy("k").applyInPandasWithState(count, output, state, "append", "NoTimeout").collect()

    # A state store per shuffle partition; the default 200 take half a minute.
    session.conf.set("spark.sql.shuffle.partitions", "2")
    try:
        source = str(tmp_path / "source")
        df.write.parquet(source)
        query = (
            session.readStream.schema("k long").parquet(source).groupBy("k")
            .applyInPandasWithState(count, output, state, "update", "NoTimeout")
            .writeStream.format("memory").queryName("apply_in_pandas_with_state")
            .outputMode("update").option("checkpointLocation", str(tmp_path / "checkpoint"))
            .trigger(availableNow=True).start()
        )
        try:
            query.awaitTermination(120)
            assert query.exception() is None
            rows = session.table("apply_in_pandas_with_state").collect()
            assert sorted(tuple(r) for r in rows) == [(1, 2), (2, 1)]
        finally:
            query.stop()
    finally:
        session.conf.unset("spark.sql.shuffle.partitions")


def test_map_functions_reject_a_non_struct_return_type_before_the_server(session):
    # As PySpark's, a map or group function whose return type is not a struct raises on the
    # client when it is built, so the server never sees the plan.
    from pyspark.errors import PySparkTypeError

    df = session.range(2)
    with pytest.raises(PySparkTypeError) as e:
        df.mapInPandas(lambda frames: frames, "long")
    assert e.value.getCondition() == "INVALID_RETURN_TYPE_FOR_PANDAS_UDF"
    with pytest.raises(PySparkTypeError) as e:
        df.groupBy("id").applyInArrow(lambda table: table, "long")
    assert e.value.getCondition() == "INVALID_RETURN_TYPE_FOR_ARROW_UDF"
    assert e.value.getMessageParameters()["return_type"] == "LongType()"
    assert df.mapInPandas(lambda frames: frames, "id long").collect() == [Row(id=0), Row(id=1)]


def test_spatial_schema_srids_are_those_of_pyspark(session):
    # As PySpark's: the server answers the SRIDs that PySpark 4.3.0 takes, and the schema of a
    # geometry or a geography of another SRID, such as the EPSG 2154 or 4269, raises
    # ST_INVALID_SRID_VALUE when it is read.
    from pyspark.errors import IllegalArgumentException
    from pyspark.sql.types import GeographyType, GeometryType

    geometry = session.sql("SELECT CAST(NULL AS GEOMETRY(3857)) AS g").schema["g"]
    assert geometry.dataType == GeometryType(3857)
    geography = session.sql("SELECT CAST(NULL AS GEOGRAPHY(4326)) AS g").schema["g"]
    assert geography.dataType == GeographyType(4326)
    for spatial in ["GEOMETRY(2154)", "GEOGRAPHY(4269)"]:
        with pytest.raises(IllegalArgumentException) as e:
            session.sql(f"SELECT CAST(NULL AS {spatial}) AS g").schema["g"].dataType
        assert e.value.getCondition() == "ST_INVALID_SRID_VALUE"


def test_variant_spatial_and_user_defined_values_round_trip_as_pyspark(session):
    # As PySpark's: createDataFrame sends a VariantVal, a Geometry, a Geography and a value of
    # a user-defined type, and collects them as they are.
    from pyspark.sql.types import Geography, Geometry
    from pyspark.testing.objects import ExamplePoint

    wkb = bytes.fromhex("0101000000000000000000F03F0000000000000040")
    variant = session.sql("""SELECT parse_json('{"a": 1}') AS v""").first()[0]
    df = session.createDataFrame([
        (variant, Geometry.fromWKB(wkb, 4326), Geography.fromWKB(wkb, 4326), ExamplePoint(1.0, 2.0))
    ])
    assert df.schema.simpleString() == (
        "struct<_1:variant,_2:geometry(4326),_3:geography(4326),_4:udt>"
    )
    row = df.first()
    assert row[0].toJson() == '{"a":1}'
    assert row[1] == Geometry.fromWKB(wkb, 4326) and row[2] == Geography.fromWKB(wkb, 4326)
    assert row[3] == ExamplePoint(1.0, 2.0)
    # A collected GEOMETRY is a Geometry.
    geometry = session.sql(
        "SELECT ST_GeomFromWKB(X'0101000000000000000000F03F0000000000000040') AS g"
    ).first()[0]
    assert geometry == Geometry.fromWKB(wkb, 0)


def test_variant_val_subclass_round_trips_as_pyspark(session):
    # As PySpark's: a subclass of VariantVal is createDataFrame's variant, and collects as a
    # VariantVal.
    from pyspark.sql.types import VariantVal

    variant = session.sql("""SELECT parse_json('{"a": 1}') AS v""").first()[0]

    class TaggedVariant(VariantVal):
        pass

    tagged = TaggedVariant(variant.value, variant.metadata)
    collected = session.createDataFrame([(tagged,)], "v variant").first()[0]
    assert type(collected) is VariantVal
    assert str(collected) == collected.toJson() == '{"a":1}'



def test_udfs_that_refer_to_variant_and_spatial_values_run_on_the_server_as_pyspark(session):
    # As PySpark's: the VariantVal, Geometry and Geography that a UDF refers to are pickled as
    # PySpark's plain classes, which the server's Python workers, of PySpark, unpickle.
    from pyspark.sql import functions as F
    from pyspark.sql.types import Geography, Geometry

    wkb = bytes.fromhex("0101000000000000000000F03F0000000000000040")
    variant = session.sql("""SELECT parse_json('{"a": 1}') AS v""").first()[0]
    geometry, geography = Geometry.fromWKB(wkb, 4326), Geography(wkb, 3857)

    @F.udf("string")
    def described(_):
        return f"{variant.toJson()} {geometry.getSrid():d} {geography.getSrid():d}"

    assert session.range(1).select(described("id")).first()[0] == '{"a":1} 4326 3857'


def test_udfs_of_methods_of_rows_and_values_run_on_the_server_as_pyspark(session):
    # As PySpark's: a UDF that is, or captures, a method of Row, VariantVal, Geometry or Geography
    # pickles it by reference to pyspark.sql.types, which the server's Python workers import.
    from pyspark.sql import functions as F
    from pyspark.sql.types import Geography, Row, VariantVal

    geography = session.sql(
        "SELECT st_geogfromwkb(X'0101000000000000000000F03F0000000000000040') AS g")
    assert geography.select(F.udf(Geography.getSrid, "int")("g")).first()[0] == 4326
    variant = session.sql("""SELECT parse_json('{"b": 2}') AS v""")
    assert variant.select(F.udf(VariantVal.toJson, "string")("v")).first()[0] == '{"b":2}'
    row = Row(a=1, b="x")
    as_dict = Row.asDict

    @F.udf("string")
    def captured(_):
        return str(as_dict(row))

    assert session.range(1).select(captured("id")).first()[0] == "{'a': 1, 'b': 'x'}"



def test_create_dataframe_of_multi_and_categorical_labels_as_pyspark(session):
    # As PySpark's: a frame whose labels are a MultiIndex or a CategoricalIndex converts its
    # cells as a frame of default labels does. A naive datetime of a list reads back unchanged
    # under a session time zone other than UTC, the Decimals and dicts take PySpark's types, and
    # the names are the labels as strings.
    import pandas as pd

    cells = [[datetime(2020, 1, 2, 3)], [Decimal("1.5"), Decimal("10.25")],
             {"u": 1, "v": "a"}, {"u": 1, "v": 2}]
    frames = [
        (pd.DataFrame([cells], columns=pd.MultiIndex.from_tuples(
            [("a", "x"), ("b", "y"), ("c", "z"), ("d", "w")]), dtype=object),
         ["('a', 'x')", "('b', 'y')", "('c', 'z')", "('d', 'w')"]),
        (pd.DataFrame([cells], columns=pd.CategoricalIndex(["a", "b", "c", "d"]), dtype=object),
         ["a", "b", "c", "d"]),
    ]
    previous = session.conf.get("spark.sql.session.timeZone")
    session.conf.set("spark.sql.session.timeZone", "Asia/Seoul")
    try:
        for frame, names in frames:
            df = session.createDataFrame(frame)
            assert df.columns == names
            assert df.schema.fields[0].dataType.simpleString() == "array<timestamp_ntz>"
            assert df.schema.fields[1].dataType.simpleString() == "array<decimal(4,2)>"
            assert df.schema.fields[2].dataType.simpleString() == "struct<u:bigint,v:string>"
            assert df.schema.fields[3].dataType.simpleString() == "struct<u:bigint,v:bigint>"
            row = df.first()
            assert row[0] == [datetime(2020, 1, 2, 3)]
            assert row[1] == [Decimal("1.50"), Decimal("10.25")]
            assert row[2] == Row(u=1, v="a") and row[3] == Row(u=1, v=2)
    finally:
        session.conf.set("spark.sql.session.timeZone", previous)


def test_create_dataframe_of_pandas_infers_the_types_of_arrow_as_pyspark(session):
    # As PySpark Connect's: without a schema, the columns of a pandas DataFrame convert with
    # PySpark's Arrow conversion, of the types that Arrow infers, and a naive datetime is a time
    # of the session time zone.
    import numpy as np
    import pandas as pd

    frame = pd.DataFrame({"a": np.array([1, 2], dtype=np.int8), "b": [{"k": 1}, {"k": 2}],
                          "c": np.array([1.5, 2.5], dtype=np.float32)})
    df = session.createDataFrame(frame)
    assert df.schema.simpleString() == "struct<a:tinyint,b:struct<k:bigint>,c:float>"
    assert [row.b.k for row in df.orderBy("a").collect()] == [1, 2]
    naive = pd.DataFrame({"t": [pd.Timestamp("2020-07-01 12:00")]})
    rows = session.createDataFrame(naive).selectExpr("CAST(t AS STRING) AS s").collect()
    assert rows == [Row(s="2020-07-01 12:00:00")]

def test_create_dataframe_types_pandas_columns_as_pyspark(session):
    from decimal import Decimal

    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    cases = [
        # A column without values takes the type of its dtype.
        ({"id": [1, 2], "s": [np.nan, np.nan]}, "struct<id:bigint,s:double>", [(1, None), (2, None)]),
        ({"id": [1, 2], "s": [None, None]}, "struct<id:bigint,s:void>", [(1, None), (2, None)]),
        ({"id": [1, 2], "t": pd.to_datetime([None, None])}, "struct<id:bigint,t:timestamp>",
         [(1, None), (2, None)]),
        # A column of Decimals takes the precision and scale of all its values.
        ({"d": [Decimal("1.5"), Decimal("2.25")]}, "struct<d:decimal(3,2)>",
         [(Decimal("1.50"),), (Decimal("2.25"),)]),
        ({"d": [Decimal("1E+2"), Decimal("0.001")]}, "struct<d:decimal(6,3)>",
         [(Decimal("100.000"),), (Decimal("0.001"),)]),
        ({"d": [Decimal("1.5"), None]}, "struct<d:decimal(2,1)>", [(Decimal("1.5"),), (None,)]),
    ]
    for data, schema, rows in cases:
        df = session.createDataFrame(pd.DataFrame(data))
        assert df.schema.simpleString() == schema, data
        assert [tuple(r) for r in df.collect()] == rows, data


def test_session_builders_keep_the_options_set_on_them_as_pyspark(session):
    # As PySpark's: the methods of a builder set their option on it, so an option set without
    # chaining is set on the session of getOrCreate.
    import os

    from pyspark.sql import SparkSession

    builder = SparkSession.builder.remote(os.environ["SPARK_REMOTE"])
    builder.config("spark.connect.rust.test.builder", "kept")
    try:
        assert builder.getOrCreate().conf.get("spark.connect.rust.test.builder") == "kept"
    finally:
        session.conf.unset("spark.connect.rust.test.builder")


def test_take_of_a_negative_count_raises_the_server_error_as_pyspark(session):
    # As PySpark's take, which is limit(num).collect(): the server rejects a negative count with
    # INVALID_LIMIT_LIKE_EXPRESSION.IS_NEGATIVE, as for head and limit.
    with pytest.raises(AnalysisException) as error:
        session.range(3).take(-1)
    assert error.value.getCondition() == "INVALID_LIMIT_LIKE_EXPRESSION.IS_NEGATIVE"
    assert len(session.range(3).take(2)) == 2


def test_collected_rows_behave_as_pyspark_rows(session):
    import pickle

    from pyspark.sql import Row

    rows = session.range(1).collect()
    assert repr(rows) == "[Row(id=0)]"
    assert rows[0] == Row(id=0) and hash(rows[0]) == hash(Row(id=0))
    assert repr(pickle.loads(pickle.dumps(rows[0]))) == "Row(id=0)"
    row = session.range(1).selectExpr(
        "named_struct('b', 1, 'c', array(1, 2)) AS a",
        "date'2020-01-02' AS d",
        "1.50BD AS n",
        "CAST(NULL AS INT) AS z",
    ).first()
    assert repr(row) == (
        "Row(a=Row(b=1, c=[1, 2]), d=datetime.date(2020, 1, 2), n=Decimal('1.50'), z=None)"
    )
    assert row.asDict(recursive=True)["a"] == {"b": 1, "c": [1, 2]}
    # Columns of the same name, as of a join, keep both values through pickling.
    duplicate = session.range(1).selectExpr("1 AS a", "2 AS a").first()
    assert repr(duplicate) == "Row(a=1, a=2)" and duplicate.asDict() == {"a": 2}
    assert repr(pickle.loads(pickle.dumps(duplicate))) == "Row(a=1, a=2)"


def test_a_progress_handler_that_raises_fails_the_execution_as_pyspark(session):
    # As PySpark Connect's: the progress of an execution calls the handlers from its result loop,
    # so the exception of a handler goes out of collect(), toPandas() and count() as it is. The
    # handler is still called as done, and a handler that does not raise, or the session after the
    # failure, works as before.
    def slow():
        # 500 ms a task, so that the server reports its progress at each report interval.
        return session.sql(
            "SELECT id, reflect('java.lang.Thread', 'sleep', 500L) AS s FROM range(0, 4, 1, 4)")

    error = ValueError("a handler raises")
    calls = []

    def failing(*, stages, inflight_tasks, operation_id, done):
        calls.append(done)
        if not done:
            raise error

    session.conf.set("spark.connect.progress.reportInterval", "10ms")
    try:
        session.registerProgressHandler(failing)
        try:
            for run in (lambda df: df.collect(), lambda df: df.toPandas(), lambda df: df.count()):
                calls.clear()
                with pytest.raises(ValueError) as raised:
                    run(slow())
                assert raised.value is error
                assert calls[-1] is True and False in calls
        finally:
            session.removeProgressHandler(failing)

        calls.clear()

        def recording(*, stages, inflight_tasks, operation_id, done):
            calls.append(done)

        session.registerProgressHandler(recording)
        try:
            assert len(slow().collect()) == 4
            assert calls[-1] is True and False in calls
        finally:
            session.removeProgressHandler(recording)
    finally:
        session.conf.unset("spark.connect.progress.reportInterval")
    assert session.range(3).count() == 3


def test_grouped_data_are_represented_as_pyspark(session):
    # As PySpark Connect's: the repr of a GroupedData names its grouping expressions, the
    # columns of its DataFrame and their types, and the type of the grouping.
    from pyspark.sql import functions as F

    df = session.createDataFrame([(1, "a", 2.0)], "k int, s string, v double")
    assert repr(df.groupBy(F.col("k") + 1, "s")) == (
        "GroupedData[grouping expressions: [+(k, 1), s], "
        "value: [k: int, s: string, v: double], type: GroupBy]")
    assert repr(df.rollup("k")).endswith("type: RollUp]")
    assert repr(df.groupBy("k").pivot("s")).endswith("type: Pivot]")




def test_pivot_pivots_only_a_group_by_as_pyspark(session):
    # As PySpark's: the pivot of a rollup, a cube, grouping sets or a pivot raises
    # UNSUPPORTED_OPERATION, rather than pivoting as a groupBy, which pivots.
    from pyspark.errors import PySparkNotImplementedError, PySparkTypeError

    df = session.createDataFrame(
        [(1, 10, "a"), (1, 20, "b"), (2, 30, "a")], "k int, v int, s string"
    )
    for grouped in [
        df.rollup("k"),
        df.cube("k"),
        df.groupingSets([["k"]], "k"),
        df.groupBy("k").pivot("s"),
    ]:
        with pytest.raises(PySparkNotImplementedError) as e:
            grouped.pivot("s")
        assert e.value.getCondition() == "UNSUPPORTED_OPERATION"
    rows = df.groupBy("k").pivot("s", ["a", "b"]).sum("v").orderBy("k").collect()
    assert [tuple(row) for row in rows] == [(1, 10, 20), (2, 30, None)]
    with pytest.raises(PySparkTypeError) as e:
        df.groupBy("k").pivot("s", ("a",))
    assert e.value.getCondition() == "NOT_EXPECTED_TYPE"


def test_sample_sort_group_by_drop_and_select_expr_as_pyspark(session):
    # As PySpark's: sample takes a fraction and a seed positionally, sort and groupBy take the
    # 1-based ordinals of columns, drop takes Columns, and selectExpr a list of expressions.
    df = session.createDataFrame(
        [(1, "b", 2.0), (2, "a", 1.0), (3, "c", 3.0)], "x int, k string, v double"
    )
    assert [r.k for r in df.sort(2).collect()] == ["a", "b", "c"]
    assert [r.k for r in df.orderBy(-2).collect()] == ["c", "b", "a"]
    one_partition = df.coalesce(1).sortWithinPartitions(3, ascending=False)
    assert [r.v for r in one_partition.collect()] == [3.0, 2.0, 1.0]
    grouped = df.groupBy(2).agg(F.max("x").alias("m")).orderBy("k").collect()
    assert grouped == [Row(k="a", m=2), Row(k="b", m=1), Row(k="c", m=3)]
    assert df.drop(df.x, "k").columns == ["v"]
    assert df.drop(F.col("v")).columns == ["x", "k"]
    assert df.selectExpr(["x", "k AS name"]).columns == ["x", "name"]
    assert df.sample(0.5, 42).collect() == df.sample(fraction=0.5, seed=42).collect()



def test_dataframes_of_other_sessions_are_not_taken_as_pyspark(session):
    # As PySpark's, a DataFrame of another session, whose temporary views this session would
    # not find, is not taken: SESSION_NOT_SAME.
    from pyspark.errors import SessionNotSameException

    other = session.newSession()
    other.range(3).createOrReplaceTempView("view_of_another_session")
    theirs = other.table("view_of_another_session")
    for call in (lambda: session.range(3).join(theirs, "id"),
                 lambda: session.range(3).union(theirs)):
        with pytest.raises(SessionNotSameException):
            call()

def test_column_attributes_select_struct_fields_and_slices_substrings(session):
    df = session.sql("SELECT named_struct('x', 1, 'y', named_struct('z', 'q')) AS s, 'abcdef' AS t")
    out = df.select(df.s.x, df.s.y.z, F.col("s").x.alias("x2"), df.t[2:3], F.col("t")[F.lit(1):F.lit(2)])
    assert out.columns == ["s.x", "s.y.z", "x2", "substr(t, 2, 3)", "substr(t, 1, 2)"]
    assert tuple(out.first()) == (1, "q", 1, "bcd", "ab")


def test_collected_rows_keep_their_values_as_pyspark(session):
    import pickle

    # A row holds the same values for as long as it lives, as a tuple does, so it equals
    # itself with a NaN or a VARIANT value, and keeps its hash.
    row = session.sql("SELECT double('NaN') x, 1 y").first()
    assert row == row and hash(row) == hash(row)
    assert row[0] is row[0] and row.x is row[0]
    variant = session.sql("""SELECT parse_json('{"a": 1}') v""").first()
    assert variant == variant and hash(variant) == hash(variant)
    nested = session.sql("SELECT named_struct('a', double('NaN')) s").first()
    assert nested == nested and nested.s is nested.s
    # Rows of VARIANT values pickle, as in PySpark.
    rows = pickle.loads(pickle.dumps(session.sql("SELECT parse_json('1') v, 2 n").collect()))
    assert (rows[0].v.toJson(), rows[0].n) == ("1", 2)


def test_results_compressed_by_the_server_decode_as_pyspark(session):
    # As PySpark's: the server compresses the Arrow batches of the results with the codec of
    # spark.sql.execution.arrow.compression.codec, zstd or lz4, which the client decodes.
    df = session.range(0, 20000, 1, 2).selectExpr("id", "CAST(id % 7 AS STRING) AS s")
    rows, table, frame = df.collect(), df.toArrow(), df.toPandas()
    try:
        for codec in ("zstd", "lz4", "none"):
            session.conf.set("spark.sql.execution.arrow.compression.codec", codec)
            assert df.collect() == rows, codec
            assert df.toArrow().equals(table), codec
            assert df.toPandas().equals(frame), codec
    finally:
        session.conf.unset("spark.sql.execution.arrow.compression.codec")


def test_transpose_takes_a_column_name_as_pyspark(session):
    # As PySpark's transpose, of a Column or a column name as the index column.
    df = session.createDataFrame([("a", 1, 2), ("b", 3, 4)], "k string, x int, y int")
    by_name = df.transpose("k")
    assert by_name.columns == ["key", "a", "b"]
    assert by_name.collect() == df.transpose(F.col("k")).collect()


def test_getitem_indexes_filters_and_selects_as_pyspark(session):
    from pyspark.errors import PySparkTypeError

    df = session.createDataFrame([(1, "a"), (2, "b"), (3, "c")], ["id", "s"])
    assert df.select(df[0], df[-1], df[True]).columns == ["id", "s", "s"]
    assert df[df["id"] > 1].collect() == [Row(id=2, s="b"), Row(id=3, s="c")]
    assert df[["s"]].collect() == [Row(s="a"), Row(s="b"), Row(s="c")]
    assert df[("s", "id")].columns == ["s", "id"]
    assert df[[["id"]]].columns == ["id"]
    with pytest.raises(IndexError):
        df[2]
    with pytest.raises(PySparkTypeError) as error:
        df[1.5]
    assert error.value.getCondition() == "NOT_EXPECTED_TYPE"


def test_plans_name_the_udfs_of_map_functions_after_the_functions_as_pyspark(session, capfd):
    # As PySpark's: the server's plan names the UDF of a map function after the function.
    def my_arrow_func(iterator):
        yield from iterator

    def my_grouped_func(pdf):
        return pdf

    capfd.readouterr()
    session.range(1).mapInArrow(my_arrow_func, "id long").explain()
    session.range(1).groupBy("id").applyInPandas(my_grouped_func, "id long").explain()
    out = capfd.readouterr().out
    assert "MapInArrow my_arrow_func(" in out
    assert "FlatMapGroupsInPandas [id#" in out and "], my_grouped_func(" in out


def test_range_and_table_valued_functions_take_their_arguments_as_pyspark(session):
    # As PySpark's: range takes the int() of its arguments, and the table-valued functions
    # take Columns or column names, so a str is a column, else NOT_EXPECTED_TYPE of col.
    from pyspark.errors import PySparkTypeError
    from pyspark.sql import functions as F

    assert [row.id for row in session.range("3").collect()] == [0, 1, 2]
    assert [row.id for row in session.tvf.range(0, 6.5, 2.5, "2").collect()] == [0, 2, 4]
    with pytest.raises(ValueError, match="invalid literal"):
        session.range("x").collect()
    with pytest.raises(ValueError, match="Value out of range"):
        session.tvf.range(2**63).collect()
    with pytest.raises(AnalysisException) as e:
        session.tvf.json_tuple(F.lit('{"a": 1}'), "a").collect()
    assert e.value.getCondition() == "UNRESOLVED_COLUMN.WITHOUT_SUGGESTION"
    with pytest.raises(PySparkTypeError) as e:
        session.tvf.explode([1, 2])
    assert e.value.getCondition() == "NOT_EXPECTED_TYPE"
    rows = session.tvf.json_tuple(F.lit('{"a": 1}'), F.lit("a")).collect()
    assert [tuple(row) for row in rows] == [("1",)]


def test_sql_runs_statements_eagerly(session):
    # sql() sends the statement at once: a command takes effect without an action, and the
    # DataFrame of its result is the local relation the server returned.
    session.sql("CREATE OR REPLACE TEMP VIEW eager_sql_v AS SELECT 1 AS x")
    assert [tuple(r) for r in session.table("eager_sql_v").collect()] == [(1,)]
    session.sql("SET spark.connect.rust.test.eager=on")
    assert session.conf.get("spark.connect.rust.test.eager") == "on"
    databases = session.sql("SHOW DATABASES")
    assert databases.isLocal()
    assert "default" in [r[0] for r in databases.collect()]
    # A query still runs at the action.
    query = session.sql("SELECT 1 AS x")
    assert not query.isLocal()
    assert [tuple(r) for r in query.collect()] == [(1,)]


@pytest.mark.parametrize(
    "make, expected",
    [
        (lambda s: s.sql("VALUES (1)"), True),
        (lambda s: s.sql("SHOW DATABASES"), True),
        (lambda s: s.sql("SELECT 1"), False),
        (lambda s: s.createDataFrame([(1,)], "a int"), True),
        (lambda s: s.range(1), False),
    ],
)
def test_is_local_is_the_servers_answer_as_pyspark(session, make, expected):
    # As PySpark: isLocal asks the server about the plan, so a VALUES is local, though the
    # client's own plan of it is not.
    assert make(session).isLocal() is expected


def test_struct_field_from_ddl_parses_as_data_type_from_ddl_as_pyspark(session):
    # As PySpark: StructField inherits DataType.fromDDL, so it parses a DDL string to the type
    # that DataType.fromDDL gives, with the same errors, and a StructType creates a DataFrame.
    from pyspark.errors import ParseException
    from pyspark.sql import types as T

    for ddl in ["a int", "`x y` int", "a int not null", "b: string", "a int, b string", "int"]:
        assert T.StructField.fromDDL(ddl) == T.DataType.fromDDL(ddl)
    assert T.StructField.fromDDL("`x y` int") == T.StructType([
        T.StructField("x y", T.IntegerType()),
    ])
    df = session.createDataFrame([(1,)], T.StructField.fromDDL("a int"))
    assert df.collect() == [Row(a=1)]
    for ddl, condition in [("a INT,", "PARSE_SYNTAX_ERROR"), ("a BOGUS", "UNSUPPORTED_DATATYPE")]:
        with pytest.raises(ParseException) as error:
            T.StructField.fromDDL(ddl)
        assert error.value.getCondition() == condition


def test_struct_types_take_their_keys_and_fields_as_pyspark(session):
    # As PySpark's StructType: add takes the JSON value of a data type, and a missing field is
    # KEY_NOT_EXISTS.
    from pyspark.errors import PySparkKeyError
    from pyspark.sql import types as T

    schema = T.StructType().add("a", "long").add("b", "string")
    df = session.createDataFrame([(1, "x")], schema)
    assert df.schema.simpleString() == "struct<a:bigint,b:string>"
    with pytest.raises(PySparkKeyError) as e:
        df.schema["zz"]
    assert e.value.getCondition() == "KEY_NOT_EXISTS"


def test_lists_and_numpy_arrays_are_arrays_as_pyspark(session):
    np = pytest.importorskip("numpy")

    df = session.range(1).select(
        F.lit([1, 2]).alias("l"), F.lit(np.array([1, 2])).alias("n"),
        F.lit(np.array([1.5], dtype=np.float32)).alias("f"), F.lit(np.array(["x"])).alias("s"),
        F.array_contains(F.lit([[1], [2]]), [1]).alias("c"), F.array(["id", "id"]).alias("a"),
        F.struct(["id"]).alias("t"), F.when(F.lit(True), [3]).alias("w"),
    )
    assert df.dtypes == [
        ("l", "array<int>"), ("n", "array<bigint>"), ("f", "array<float>"), ("s", "array<string>"),
        ("c", "boolean"), ("a", "array<bigint>"), ("t", "struct<id:bigint>"), ("w", "array<int>"),
    ]
    assert [tuple(r) for r in df.collect()] == [
        ([1, 2], [1, 2], [1.5], ["x"], True, [0, 0], Row(id=0), [3])]


def test_stream_writer_takes_its_query_name_and_columns_as_pyspark(session, tmp_path):
    # As PySpark's DataStreamWriter: partitionBy takes a list of column names, and queryName
    # raises VALUE_NOT_NON_EMPTY_STR for an empty name.
    from pyspark.errors import PySparkValueError

    source, sink = tmp_path / "source", tmp_path / "sink"
    session.createDataFrame([(0, "a"), (1, "b")], "id long, k string").write.json(str(source))
    query = (
        session.readStream.schema("id long, k string")
        .json(str(source))
        .writeStream.format("parquet")
        .partitionBy(["k"])
        .option("checkpointLocation", str(tmp_path / "checkpoint"))
        .trigger(availableNow=True)
        .start(str(sink))
    )
    query.awaitTermination()
    assert query.exception() is None
    assert sorted(path.name for path in sink.iterdir() if path.name.startswith("k=")) == [
        "k=a",
        "k=b",
    ]
    with pytest.raises(PySparkValueError) as e:
        session.readStream.format("rate").load().writeStream.queryName("")
    assert e.value.getCondition() == "VALUE_NOT_NON_EMPTY_STR"


def test_sql_fails_at_a_statement_that_does_not_parse(session):
    with pytest.raises(AnalysisException, match="PARSE_SYNTAX_ERROR"):
        session.sql("SELEC 1")


def test_repr_html_is_the_html_table_of_eager_evaluation_as_pyspark(session):
    # As PySpark Connect's: _repr_html_ is None without eager evaluation, else the HTML table of
    # the DataFrame, after which __repr__ is the string of the schema.
    df = session.range(2)
    assert df._repr_html_() is None
    session.conf.set("spark.sql.repl.eagerEval.enabled", "true")
    try:
        html = df._repr_html_()
        assert html.startswith("<table border='1'>") and "<th>id</th>" in html
        assert repr(df) == "DataFrame[id: bigint]"
        assert "+---+" in repr(session.range(2))
    finally:
        session.conf.unset("spark.sql.repl.eagerEval.enabled")

def test_sql_formats_the_query_with_keyword_arguments_as_pyspark(session):
    from decimal import Decimal

    df = session.range(2)
    other = session.createDataFrame([(0, "a"), (1, "b")], "id long, s string")
    cases = [
        # A value becomes its string, a string its SQL literal, and a column reference its
        # name, as `str.format` with PySpark's formatter does.
        (lambda: session.sql("SELECT {x} AS v", x=1), [(1,)]),
        (lambda: session.sql("SELECT {s} AS s", s="it's a \\ test"), [("it's a \\ test",)]),
        (lambda: session.sql("SELECT {c} AS c FROM {df}", c=F.col("id"), df=df), [(0,), (1,)]),
        (lambda: session.sql("SELECT {df.id} AS c FROM {df}", df=df), [(0,), (1,)]),
        (lambda: session.sql("SELECT {x:.2f} AS v", x=1.5), [(Decimal("1.50"),)]),
        # A DataFrame is a reference sent with the query, one for each DataFrame.
        (lambda: session.sql("SELECT count(*) AS n FROM {df}", df=session.range(3)), [(3,)]),
        (lambda: session.sql("SELECT * FROM {a} UNION ALL SELECT * FROM {a}", a=df),
         [(0,), (1,), (0,), (1,)]),
        (lambda: session.sql(
            "SELECT a.id, b.s FROM {a} a JOIN {b} b ON a.id = b.id ORDER BY a.id", a=df, b=other),
         [(0, "a"), (1, "b")]),
        # `args` still binds parameters, and braces are only formatted with keyword arguments.
        (lambda: session.sql("SELECT {x} + :p AS v", args={"p": 1}, x=2), [(3,)]),
        (lambda: session.sql("SELECT '{{}}' AS v"), [("{{}}",)]),
        (lambda: session.sql("SELECT '{{}}' AS v, {x} AS x", x=1), [("{}", 1)]),
    ]
    for run, rows in cases:
        assert [tuple(r) for r in run().collect()] == rows


def test_show_prints_the_table_of_the_server_as_pyspark(session):
    import contextlib
    import io

    df = session.sql("SELECT id, repeat('x', 25) AS s FROM range(2)")

    def shown(**kwargs):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):  # Python's print, as PySpark's show.
            df.show(**kwargs)
        return out.getvalue()

    assert shown() == (
        "+---+--------------------+\n"
        "| id|                   s|\n"
        "+---+--------------------+\n"
        "|  0|xxxxxxxxxxxxxxxxx...|\n"
        "|  1|xxxxxxxxxxxxxxxxx...|\n"
        "+---+--------------------+\n\n")
    assert shown(n=1, truncate=5) == (
        "+---+-----+\n| id|    s|\n+---+-----+\n|  0|xx...|\n+---+-----+\nonly showing top 1 row\n")
    assert shown(truncate=False).splitlines()[3] == "|0  |" + "x" * 25 + "|"
    assert shown(n=1, vertical=True) == (
        "-RECORD 0-------------------\n"
        " id  | 0                    \n"
        " s   | xxxxxxxxxxxxxxxxx... \n"
        "only showing top 1 row\n")


def test_sql_keyword_arguments_are_not_parameters_as_pyspark(session):
    with pytest.raises(AnalysisException) as error:
        session.sql("SELECT :p AS v", p=5).collect()
    assert error.value.getCondition() == "UNBOUND_SQL_PARAMETER"


def test_explain_prints_with_python_print_as_pyspark(session):
    # As PySpark's: explain, and StreamingQuery.explain, print the plan with Python's print, to
    # sys.stdout, which redirect_stdout captures, and return None.
    import contextlib
    import io
    import time

    df = session.range(1)
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        assert df.explain() is None
        assert df.explain(mode="formatted") is None
    assert printed.getvalue().startswith("== Physical Plan ==\n")
    assert "\n(1) Range [codegen id : 1]\n" in printed.getvalue()
    assert df._explain_string(True).startswith("== Parsed Logical Plan ==\n")

    query = (
        session.readStream.format("rate").load()
        .writeStream.format("memory").queryName("explain_prints").start()
    )
    try:
        # The plan of a batch of the query, after its first. processAllAvailable() would wait
        # for a rate source to have no new rows, which a slow machine never reaches.
        for _ in range(240):
            if query.lastProgress is not None:
                break
            time.sleep(0.25)
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            assert query.explain(True) is None
        assert printed.getvalue().startswith("== Parsed Logical Plan ==\n")
    finally:
        query.stop()


def test_column_alias_names_every_column_and_keeps_its_metadata(session):
    df = session.sql("SELECT map('k1', 1) AS m").select(F.explode("m").alias("k", "v"))
    assert df.columns == ["k", "v"]
    assert [tuple(r) for r in df.collect()] == [("k1", 1)]
    df = session.range(1).select(F.col("id").alias("x", metadata={"k": 1, "n": [1, 2], "s": "v"}))
    assert df.schema.fields[0].metadata == {"k": 1, "n": [1, 2], "s": "v"}


def test_to_arrow_casts_to_the_arrow_schema_of_the_schema_as_pyspark(session):
    # As PySpark Connect's: the Arrow types of the Spark types, of UTC timestamps and of strings
    # even of large var types, and the errors of the types that PySpark does not convert to
    # Arrow and of duplicate names of nested fields.
    import pyarrow as pa
    from pyspark.errors import PySparkTypeError, UnsupportedOperationException

    confs = {"spark.sql.execution.arrow.useLargeVarTypes": "true",
             "spark.sql.session.timeZone": "America/Los_Angeles"}
    previous = {key: session.conf.get(key) for key in confs}
    for key, value in confs.items():
        session.conf.set(key, value)
    try:
        table = session.sql("SELECT 'x' AS s, TIMESTAMP'2020-01-01 00:00:00' AS t").toArrow()
    finally:
        for key, value in previous.items():
            session.conf.set(key, value)
    assert table.schema.field("s").type == pa.string()
    assert table.schema.field("t").type == pa.timestamp("us", tz="UTC")
    for expr, data_type in [("INTERVAL '1-2' YEAR TO MONTH", "YearMonthIntervalType(0, 1)"),
                            ("make_interval(1)", "CalendarIntervalType()")]:
        with pytest.raises(PySparkTypeError) as e:
            session.sql(f"SELECT {expr} AS v").toArrow()
        assert e.value.getCondition() == "UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION"
        assert e.value.getMessageParameters() == {"data_type": data_type}
    with pytest.raises(UnsupportedOperationException) as e:
        session.sql("SELECT named_struct('a', 1, 'a', 2) AS v").toArrow()
    assert e.value.getCondition() == "DUPLICATED_FIELD_NAME_IN_ARROW_STRUCT"

def test_a_derived_dataframe_of_a_local_checkpoint_outlives_the_checkpoint_as_pyspark(session):
    # The checkpoint is released when its last DataFrame goes: a DataFrame derived from it keeps
    # working after the checkpoint itself is deleted and collected.
    import gc

    checkpoint = session.range(5).localCheckpoint()
    derived = checkpoint.filter(F.col("id") > 1)
    del checkpoint
    gc.collect()
    assert derived.count() == 3
    assert [row.id for row in derived.orderBy("id").collect()] == [2, 3, 4]

def test_to_pandas_converts_the_columns_as_pyspark(session):
    # As official PySpark 4.3.0's SparkConnectClient.to_pandas: each column by its Spark type,
    # in the session time zone, with the metrics of the execution in attrs.
    from decimal import Decimal

    import pandas as pd
    from pyspark.ml.functions import array_to_vector
    from pyspark.ml.linalg import DenseVector

    df = session.sql(
        "SELECT TIMESTAMP'2024-01-02 03:04:05' AS ts, map('k', 1) AS m, "
        "named_struct('a', 1, 'b', 'x') AS st, 1.5BD AS d, array(1.5D, 2.0D) AS av"
    ).withColumn("vec", array_to_vector("av")).drop("av")
    pdf = df.toPandas()
    assert str(pdf.dtypes["ts"]) == "datetime64[ns]"
    assert pdf["ts"][0] == pd.Timestamp("2024-01-02 03:04:05")
    assert pdf["m"][0] == {"k": 1}
    assert pdf["st"][0] == {"a": 1, "b": "x"}
    assert pdf["d"][0] == Decimal("1.5")
    assert pdf["vec"][0] == DenseVector([1.5, 2.0])
    assert sorted(pdf.attrs) == ["metrics"]
    assert type(pdf.attrs["metrics"][0]).__name__ == "PlanMetrics"
    struct = df.select("st")._to_pandas(pandasStructHandlingMode="row")["st"][0]
    assert type(struct).__name__ == "Row" and struct == Row(a=1, b="x")
    session.conf.set("spark.sql.session.timeZone", "Asia/Seoul")
    try:
        local = session.sql("SELECT TIMESTAMP'2024-01-02 03:04:05' AS ts").toPandas()
        assert local["ts"][0] == pd.Timestamp("2024-01-02 03:04:05")
    finally:
        session.conf.unset("spark.sql.session.timeZone")


def test_na_and_stat_functions_are_of_their_dataframe_as_pyspark(session):
    # As PySpark's: the df of df.na and df.stat is df itself.
    df = session.createDataFrame([(1, None), (None, 2.0)], "a int, b double")
    assert df.na.df is df
    assert df.stat.df is df
    assert sorted(df.na.df.na.fill(0).collect()) == sorted(df.na.fill(0).collect())
    assert df.stat.df.count() == 2


def test_sample_by_samples_the_fractions_of_the_strata_as_pyspark(session):
    # As PySpark's sampleBy: each stratum by its fraction, of a column name or a Column.
    df = session.createDataFrame([(1, "a"), (2, "b"), (3, "a")], "id int, k string")
    fractions = {"a": 1.0, "b": 0.0}
    for sampled in (
        df.sampleBy("k", fractions, seed=1),
        df.sampleBy(F.col("k"), fractions, seed=1),
        df.stat.sampleBy("k", fractions, seed=1),
    ):
        assert sorted(r.id for r in sampled.collect()) == [1, 3]


def test_window_order_by_a_column_sorts_nulls_first_as_pyspark(session):
    from pyspark.sql import Window

    df = session.sql("SELECT * FROM VALUES (1, CAST(NULL AS INT)), (1, 2), (1, 1) AS t(g, o)")
    # A column that is not a sort order is ascending with nulls first, as SortOrder(expr).
    for window, expected in (
        (Window.partitionBy("g").orderBy("o"), [(None, 1), (1, 2), (2, 3)]),
        (Window.partitionBy("g").orderBy(F.col("o")), [(None, 1), (1, 2), (2, 3)]),
        (Window.orderBy("o").partitionBy("g").orderBy("o"), [(None, 1), (1, 2), (2, 3)]),
        (Window.partitionBy("g").orderBy(F.col("o").desc()), [(2, 1), (1, 2), (None, 3)]),
    ):
        rows = df.select("o", F.row_number().over(window).alias("rn")).orderBy("rn").collect()
        assert [(r.o, r.rn) for r in rows] == expected


def test_create_dataframe_of_a_numpy_array_as_pyspark(session):
    # As PySpark Connect's: a column of each column of a 1- or 2-dimensional array, named by the
    # names, else `value` for one column, else `_1`, `_2`, ..., of the types of the array, or
    # cast to the schema by the server.
    import numpy as np

    df = session.createDataFrame(np.array([1, 2]))
    assert df.schema.simpleString() == "struct<value:bigint>"
    assert sorted(df.collect()) == [Row(value=1), Row(value=2)]
    df = session.createDataFrame(np.array([[1.5, 2.5], [3.5, 4.5]]))
    assert df.schema.simpleString() == "struct<_1:double,_2:double>"
    assert session.createDataFrame(np.array([[1, 2]]), "a string, b double").collect() == [
        Row(a="1", b=2.0)]
    assert session.createDataFrame(np.array(["x", "y"]), ["s"]).columns == ["s"]
    with pytest.raises(PySparkValueError) as e:
        session.createDataFrame(np.zeros((2, 2, 2)))
    assert e.value.getCondition() == "INVALID_NDARRAY_DIMENSION"
    with pytest.raises(PySparkValueError) as e:
        session.createDataFrame(np.array([[1, 2]]), ["x"])
    assert e.value.getCondition() == "AXIS_LENGTH_MISMATCH"

def test_local_iterator_closes_as_a_generator_as_pyspark(session):
    # As PySpark's generator: contextlib.closing ends the iteration early, so the rows that are
    # left are not fetched; the iterator then ends, and closing it again does nothing.
    import contextlib

    df = session.range(1000, numPartitions=10)
    with contextlib.closing(df.toLocalIterator()) as rows:
        assert next(rows).id == 0
    assert list(rows) == []
    rows.close()


def test_local_iterator_is_a_generator_as_pyspark(session):
    # As PySpark's generator function: it yields the rows, and a close() or a throw() in the
    # middle ends it.
    import inspect
    import types

    df = session.range(1000, numPartitions=10)
    rows = df.toLocalIterator()
    assert isinstance(rows, types.GeneratorType)
    assert [row.id for row in rows] == list(range(1000))
    assert inspect.getgeneratorstate(rows) == inspect.GEN_CLOSED
    assert next(rows, None) is None

    rows = df.toLocalIterator()
    assert next(rows).id == 0
    assert inspect.getgeneratorstate(rows) == inspect.GEN_SUSPENDED
    with pytest.raises(ValueError, match="stop"):
        rows.throw(ValueError("stop"))
    assert inspect.getgeneratorstate(rows) == inspect.GEN_CLOSED
    assert list(rows) == []

    rows = df.toLocalIterator()
    next(rows)
    rows.close()
    assert inspect.getgeneratorstate(rows) == inspect.GEN_CLOSED
    assert list(rows) == []


def test_create_dataframe_takes_the_ddl_of_spark_as_pyspark(session):
    # As PySpark Connect's, which parses a DDL schema on the server: nested structs, arrays and
    # maps of them, backtick-quoted names, and NOT NULL and COMMENT clauses.
    schema = (
        "st struct<a:int,b:double>, arr array<struct<c:int>>, m map<string,struct<x:int>>, "
        "`a b` int NOT NULL COMMENT 'c'"
    )
    df = session.createDataFrame([((1, 2.0), [(3,)], {"k": (4,)}, 5)], schema)
    assert df.schema.simpleString() == (
        "struct<st:struct<a:int,b:double>,arr:array<struct<c:int>>,"
        "m:map<string,struct<x:int>>,a b:int>"
    )
    assert df.collect() == [
        Row(st=Row(a=1, b=2.0), arr=[Row(c=3)], m={"k": Row(x=4)}, **{"a b": 5})
    ]
    # Collations, which the server names, and the errors of the server's parser.
    collated = session.createDataFrame([("x",)], "s STRING COLLATE UTF8_LCASE")
    assert collated.schema.simpleString() == "struct<s:string collate UTF8_LCASE>"
    from pyspark.errors import ParseException

    for ddl, condition in [("a INT,", "PARSE_SYNTAX_ERROR"), ("a BOGUS", "UNSUPPORTED_DATATYPE")]:
        with pytest.raises(ParseException) as error:
            session.createDataFrame([], ddl)
        assert error.value.getCondition() == condition
    # As do the map functions, of their schemas.
    mapped = session.range(2).mapInPandas(lambda it: it, "id long NOT NULL COMMENT 'k'")
    assert [(f.name, f.nullable, f.metadata) for f in mapped.schema.fields] == [
        ("id", False, {"comment": "k"})]
    assert mapped.collect() == [Row(id=0), Row(id=1)]



def test_schemas_of_collated_strings_are_of_the_json_of_pyspark(session):
    # As PySpark's: the JSON of a schema has the collations of its strings in the metadata of
    # their fields, and StructType.fromJson reads them back.
    import json

    from pyspark.sql.types import StructType

    schema = session.sql(
        "select 'a' collate UTF8_LCASE as s, array('b' collate UNICODE) as a").schema
    assert schema.json() == (
        '{"fields":[{"metadata":{"__COLLATIONS":{"s":"spark.UTF8_LCASE"}},"name":"s",'
        '"nullable":false,"type":"string"},{"metadata":{"__COLLATIONS":{"a.element":'
        '"icu.UNICODE"}},"name":"a","nullable":false,"type":{"containsNull":false,'
        '"elementType":"string","type":"array"}}],"type":"struct"}')
    assert StructType.fromJson(json.loads(schema.json())) == schema




def test_ddl_types_are_parsed_by_the_server_as_pyspark(session):
    # As PySpark Connect's DataType.fromDDL and _parse_datatype_string, which ask the server
    # of the active session to parse a DDL string, as does a UDF of a DDL return type.
    from pyspark.errors import ParseException
    from pyspark.sql import types as T
    from pyspark.sql.types import _parse_datatype_string

    field = T.DataType.fromDDL("a INT NOT NULL COMMENT 'c'").fields[0]
    assert (field.nullable, field.metadata) == (False, {"comment": "c"})
    assert T.StructType.fromDDL("st struct<a:int,b:double>").simpleString() == (
        "struct<st:struct<a:int,b:double>>")
    assert T.DataType.fromDDL("s STRING COLLATE UTF8_LCASE").simpleString() == (
        "struct<s:string collate UTF8_LCASE>")
    assert _parse_datatype_string("decimal(10, 2)") == T.DecimalType(10, 2)
    for ddl, condition in [("a INT,", "PARSE_SYNTAX_ERROR"), ("a BOGUS", "UNSUPPORTED_DATATYPE")]:
        with pytest.raises(ParseException) as error:
            T.DataType.fromDDL(ddl)
        assert error.value.getCondition() == condition
    pair = F.udf(lambda x: (x, 1.5), "st struct<a:bigint,b:double>")
    assert pair.returnType.simpleString() == "struct<st:struct<a:bigint,b:double>>"
    result = session.range(1).select(F.udf(lambda x: (x, 1.5), "struct<a:bigint,b:double>")("id"))
    assert [tuple(r[0]) for r in result.collect()] == [(0, 1.5)]

def test_functions_send_their_optional_arguments_as_pyspark(session):
    # The values of official PySpark 4.3.0; each optional argument used to be dropped.
    df = session.sql(
        "SELECT '02/01/2024' AS d, 8.0D AS x, array(2, 1, 3) AS a, 'a,b,c' AS s, "
        "'a_c' AS p, 1700000000L AS t, 2.5D AS h")
    row = df.select(
        F.to_date("d", "dd/MM/yyyy").alias("date"),
        F.log(2.0, "x").alias("log2"),
        F.sort_array("a", asc=False).alias("descending"),
        F.split("s", ",", 2).alias("split"),
        F.substr("s", F.lit(1), F.lit(2)).alias("substr"),
        F.like("p", F.lit("a!_c"), F.lit("!")).alias("like"),
        F.unix_timestamp(F.timestamp_seconds("t")).alias("seconds"),
        F.months_between(F.lit("2024-03-31"), F.lit("2024-02-28"), roundOff=False).alias("months"),
        # An explicit None is no scale, as in PySpark, rather than a NULL one.
        F.bround("h", None).alias("bround"),
    ).first()
    assert row.date == date(2024, 1, 2)
    assert row.log2 == 3.0
    assert row.descending == [3, 2, 1]
    assert row.split == ["a", "b,c"]
    assert row.substr == "a,"
    assert row.like is True
    assert row.seconds == 1700000000
    assert row.months == 1.096774193548387
    assert row.bround == 2.0
    nulls = session.sql("SELECT * FROM VALUES (1, NULL), (1, 5) AS t(g, v)")
    assert nulls.groupBy("g").agg(F.first("v", ignorenulls=True)).first()[1] == 5


def test_datetime_literals_collect_the_exact_values_of_pyspark():
    # As PySpark's TimestampType: a datetime literal is the instant of its micros, exact from year
    # 1 to year 9999 (a float of seconds lost the micros after the year 2255), and a naive one is
    # in the client's time zone. A TIMESTAMP that is past the year 9999 in the zone of the client
    # (here nine hours ahead of UTC) raises OverflowError on collect, as PySpark does.
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live-server tests")
    out = _run_in_time_zone(f"""
import datetime
from pyspark.sql import SparkSession, functions as F
spark = SparkSession.builder.remote({remote!r}).getOrCreate()
utc = datetime.timezone.utc
values = [
    ("aware_1", datetime.datetime(1, 1, 1, 0, 0, 0, 1, tzinfo=utc)),
    ("aware_1969", datetime.datetime(1969, 12, 31, 23, 59, 59, 999999, tzinfo=utc)),
    ("aware_2300", datetime.datetime(2300, 1, 1, 0, 0, 0, 123457, tzinfo=utc)),
    ("aware_9999", datetime.datetime(9999, 12, 31, 23, 59, 59, 999999, tzinfo=utc)),
    ("naive_1969", datetime.datetime(1969, 12, 31, 23, 59, 59, 999999)),
    ("naive_2300", datetime.datetime(2300, 1, 1, 0, 0, 0, 123457)),
    ("naive_9999", datetime.datetime(9999, 12, 31, 23, 59, 59, 999999)),
]
for name, value in values:
    try:
        print(name, repr(spark.range(1).select(F.lit(value).alias("v")).collect()[0][0]))
    except Exception as error:
        print(name, type(error).__name__)
""", "Asia/Seoul")
    assert out.splitlines() == [
        "aware_1 datetime.datetime(1, 1, 1, 8, 27, 52, 1)",
        "aware_1969 datetime.datetime(1970, 1, 1, 8, 59, 59, 999999)",
        "aware_2300 datetime.datetime(2300, 1, 1, 9, 0, 0, 123457)",
        "aware_9999 OverflowError",
        "naive_1969 datetime.datetime(1969, 12, 31, 23, 59, 59, 999999)",
        "naive_2300 datetime.datetime(2300, 1, 1, 0, 0, 0, 123457)",
        "naive_9999 datetime.datetime(9999, 12, 31, 23, 59, 59, 999999)",
    ]


def test_datetime_literals_follow_the_preferred_timestamp_type_as_pyspark(session):
    # As PySpark under spark.sql.timestampType=TIMESTAMP_NTZ: a datetime literal is a
    # TIMESTAMP_NTZ of its wall clock, so it matches the TIMESTAMP_NTZ values of createDataFrame,
    # also when the zone of the client is not that of the session; else it is a TIMESTAMP.
    if not hasattr(time, "tzset"):
        pytest.skip("time.tzset is not available")
    value = datetime(2020, 6, 1, 12)
    zone = os.environ.get("TZ")
    os.environ["TZ"] = "Asia/Seoul"
    time.tzset()
    session.conf.set("spark.sql.timestampType", "TIMESTAMP_NTZ")
    try:
        df = session.createDataFrame([(value,)], ["t"])
        assert df.schema.simpleString() == "struct<t:timestamp_ntz>"
        assert df.filter(F.col("t") == value).count() == 1
        assert df.filter(F.col("t").isin(value)).count() == 1
        assert session.sql(
            "SELECT CAST(:t AS STRING) AS t", args={"t": value}).first().t == "2020-06-01 12:00:00"
        assert session.range(1).select(
            F.when(F.lit(True), value).alias("w"), F.lit([value]).alias("a"),
        ).schema.simpleString() == "struct<w:timestamp_ntz,a:array<timestamp_ntz>>"
    finally:
        session.conf.unset("spark.sql.timestampType")
        if zone is None:
            del os.environ["TZ"]
        else:
            os.environ["TZ"] = zone
        time.tzset()
    assert session.range(1).select(F.lit(value).alias("t")).schema.simpleString() == (
        "struct<t:timestamp>")


def test_window_frames_return_the_rows_of_pyspark(session):
    # As PySpark's: a range frame of a boundary out of the range of an int, which was
    # truncated, and the frames of the bounds of -sys.maxsize and sys.maxsize, unbounded.
    import sys

    from pyspark.errors import PySparkValueError
    from pyspark.sql import Window

    df = session.range(4)
    o = Window.orderBy("id")

    def sums(window):
        return sorted(r.s for r in df.select(F.sum("id").over(window).alias("s")).collect())

    assert sums(o.rangeBetween(-(1 << 40), 0)) == [0, 1, 3, 6]
    assert sums(o.rangeBetween(0, 1 << 40)) == [3, 5, 6, 6]
    assert sums(o.rowsBetween(-sys.maxsize, sys.maxsize)) == [6, 6, 6, 6]
    assert sums(o.rangeBetween(-sys.maxsize, sys.maxsize)) == [6, 6, 6, 6]
    with pytest.raises(PySparkValueError, match="VALUE_NOT_BETWEEN"):
        sums(o.rowsBetween(-(1 << 31) - 1, 0))


def test_struct_types_convert_to_the_ddl_of_spark_as_pyspark(session):
    # As PySpark Connect's StructType.toDDL: the server writes the DDL of the schema, which
    # createDataFrame parses back.
    from pyspark.sql import types as T

    st = T.StructType([
        T.StructField("a", T.IntegerType()),
        T.StructField("b b", T.StringType(), False),
        T.StructField("m", T.MapType(
            T.StringType(), T.StructType([T.StructField("x", T.DecimalType(10, 2))]))),
        T.StructField("c", T.StringType(), metadata={"comment": "the c"}),
    ])
    ddl = st.toDDL()
    assert ddl == (
        "a INT,`b b` STRING NOT NULL,m MAP<STRING, STRUCT<x: DECIMAL(10,2)>>,"
        "c STRING COMMENT 'the c'")
    assert session.createDataFrame([], ddl).schema == st




def test_pandas_frames_of_duplicate_nested_field_names_are_created_as_pyspark(session):
    # As PySpark's test_createDataFrame_pandas_duplicate_field_names: the structs of the schema
    # repeat field names, with Arrow and without.
    import pandas as pd
    from pyspark.sql.types import StructType

    schema = (
        StructType()
        .add("struct", StructType().add("x", StringType()).add("x", IntegerType()))
        .add(
            "struct",
            StructType().add("a", IntegerType()).add("x", IntegerType()).add("x", StringType()),
        )
    )
    data = [Row(Row("a", 1), Row(2, 3, "b")), Row(Row("x", None), Row(7, 8, None))]
    frame = pd.DataFrame.from_records(data, columns=schema.names)
    try:
        for arrow in ("true", "false"):
            session.conf.set("spark.sql.execution.arrow.pyspark.enabled", arrow)
            df = session.createDataFrame(frame, schema)
            assert df.schema == schema and df.collect() == data, arrow
    finally:
        session.conf.unset("spark.sql.execution.arrow.pyspark.enabled")




def test_maps_with_user_defined_keys_collect_their_deserialized_keys_as_pyspark(session):
    # As PySpark: the keys of a map of a user-defined type are deserialized, at the top level,
    # in a struct, in an array and in the value of a map, for collect, toLocalIterator, head,
    # first and take.
    from pyspark.sql.types import UserDefinedType

    class PairUDT(UserDefinedType):
        @classmethod
        def sqlType(cls):
            return ArrayType(DoubleType(), False)

        @classmethod
        def module(cls):
            return __name__

        def serialize(self, obj):
            return [obj[0], obj[1]]

        def deserialize(self, datum):
            return (datum[0], datum[1])

        def __eq__(self, other):
            return isinstance(other, PairUDT)

        def __hash__(self):
            return 1

    pair = PairUDT()
    schema = StructType([
        StructField("m", MapType(pair, IntegerType())),
        StructField("s", StructType([StructField("k", MapType(pair, StringType()))])),
        StructField("a", ArrayType(MapType(pair, IntegerType()))),
        StructField("v", MapType(StringType(), MapType(pair, IntegerType()))),
    ])
    expected = Row(
        m={(1.0, 2.0): 1, (3.0, 4.0): None},
        s=Row(k={(1.0, 2.0): "x"}),
        a=[{(5.0, 6.0): 7}],
        v={"a": {(1.0, 2.0): 3}},
    )
    df = session.createDataFrame(
        [({(1.0, 2.0): 1, (3.0, 4.0): None}, {"k": {(1.0, 2.0): "x"}}, [{(5.0, 6.0): 7}],
          {"a": {(1.0, 2.0): 3}})],
        schema,
    )
    assert df.collect() == [expected]
    assert list(df.toLocalIterator()) == [expected]
    assert df.head() == df.first() == expected
    assert df.take(1) == [expected]
    # A list key is unhashable, as in PySpark, also in a map with user-defined keys.
    with pytest.raises(TypeError, match="unhashable"):
        session.sql("SELECT map(array(1), 'x') AS m").collect()


def test_maps_keep_their_entries_in_order_with_keys_of_their_type(session):
    row = session.range(1).selectExpr(
        "map(2, 'b', 1, 'a') AS ints",
        "map('b', 1, 'a', 2) AS strs",
        "map(date'2020-01-02', 1) AS dates",
        "map(named_struct('a', 1), 'x') AS structs",
    ).first()
    assert list(row.ints.items()) == [(2, "b"), (1, "a")]
    assert list(row.strs.items()) == [("b", 1), ("a", 2)]
    assert row.dates == {date(2020, 1, 2): 1}
    assert row.structs == {Row(a=1): "x"}
    # A list key is unhashable, as in PySpark. (PySpark raises at collect(); this client
    # turns a row's values into Python ones when they are read.)
    with pytest.raises(TypeError, match="unhashable"):
        session.range(1).selectExpr("map(array(1), 'x') AS m").first().m
    df = session.createDataFrame([({2: "b", 1: "a"},)])
    assert df.schema.simpleString() == "struct<_1:map<bigint,string>>"
    assert list(df.first()[0].items()) == [(2, "b"), (1, "a")]


def test_types_of_schemas_have_the_attributes_of_pyspark(session):
    # As PySpark's: the types of a schema of the server have the attributes of their
    # parameters.
    schema = session.sql(
        "select 'a' collate UTF8_LCASE as s, interval '1' hour as i, time'12:30:00' as t, "
        "1.5bd as d").schema
    assert schema["s"].dataType.collation == "UTF8_LCASE"
    assert (schema["i"].dataType.startField, schema["i"].dataType.endField) == (1, 1)
    assert schema["t"].dataType.precision == 6
    assert schema["d"].dataType.hasPrecisionInfo is True
    assert schema["i"].dataType.typeName() == type(schema["i"].dataType).typeName()




def test_decimals_of_a_negative_scale_are_collected_as_pyspark(session):
    # As PySpark's, of Arrow: with spark.sql.legacy.allowNegativeScaleOfDecimal, a decimal of a
    # negative scale is collected as the Decimal of its unscaled value times ten to the negated
    # scale, in scientific notation: Decimal("1E+1") of 1 at the scale -1.
    from decimal import Decimal

    from pyspark.sql.types import DecimalType

    key = "spark.sql.legacy.allowNegativeScaleOfDecimal"
    session.conf.set(key, "true")
    try:
        df = session.createDataFrame([(1,), (11,)], ["value"])
        rows = df.select(F.col("value").cast(DecimalType(1, -1)).alias("d")).collect()
        assert [int(r.d) for r in rows] == [0, 10]
        assert rows[1].d == Decimal("10") and str(rows[1].d) == "1E+1"
        df = session.createDataFrame([(-123,), (49,), (51,)], ["value"])
        rows = df.select(F.col("value").cast(DecimalType(3, -2)).alias("d")).collect()
        assert [str(r.d) for r in rows] == ["-1E+2", "0E+2", "1E+2"]
    finally:
        session.conf.unset(key)




def test_decimal_literals_are_of_the_types_of_pyspark(session):
    # The types of official PySpark 4.3.0: at least the precision 10 of DecimalType().
    from decimal import Decimal

    df = session.range(1).select(
        F.lit(Decimal("1.5")).alias("a"),
        F.lit(Decimal(0)).alias("b"),
        F.lit(Decimal("123456789012.5")).alias("c"),
        F.lit(Decimal("-0.001")).alias("d"),
        (F.lit(Decimal("1.5")) * F.lit(Decimal("2.25"))).alias("product"),
    )
    assert df.dtypes == [
        ("a", "decimal(10,1)"), ("b", "decimal(10,0)"), ("c", "decimal(13,1)"),
        ("d", "decimal(10,3)"), ("product", "decimal(21,3)"),
    ]
    assert tuple(df.first()) == (
        Decimal("1.5"), Decimal("0"), Decimal("123456789012.5"), Decimal("-0.001"),
        Decimal("3.375"),
    )


def test_observations_receive_the_observed_metrics_as_pyspark(session):
    # As PySpark's: each action that runs a DataFrame of an observation sets in it the metrics
    # that the server observed, for each observation of a name on its own, and get raises the
    # error of collecting them.
    from pyspark.errors import PySparkException
    from pyspark.sql import Observation

    first, second = Observation("m"), Observation("m")
    df1 = session.range(3).observe(
        first, F.count(F.lit(1)).alias("c"), F.sum("id").alias("s"), F.lit(None).alias("n")
    )
    df2 = session.range(5).observe(second, F.count(F.lit(1)).alias("c"))
    assert first.get == {} and second.get == {}
    assert df1.collect() == [Row(id=0), Row(id=1), Row(id=2)]
    assert first.get == {"c": 3, "s": 3, "n": None} and second.get == {}
    df2.select("*").write.format("noop").mode("overwrite").save()
    assert second.get == {"c": 5}
    assert df1.where("id > 0").count() == 2 and first.get["c"] == 3

    failing = Observation("failing")
    df = session.range(2).observe(failing, F.raise_error(F.lit("test error")).alias("e"))
    assert df.collect() == [Row(id=0), Row(id=1)]
    with pytest.raises(PySparkException, match="test error"):
        failing.get

    # A streaming DataFrame, which the server tells, is not observed.
    from pyspark.errors import IllegalArgumentException

    stream = session.readStream.format("rate").load()
    with pytest.raises(IllegalArgumentException) as error:
        stream.observe(Observation(), F.count(F.lit(1)).alias("c"))
    assert error.value.getCondition() == "UNSUPPORTED_OPERATION"


def test_a_deep_copy_of_a_dataframe_sets_its_own_observations_as_pyspark(session):
    # As PySpark's deep copy of a DataFrame, which copies the Observations of its plan: an
    # action of the copy sets the copies and not the Observations of the DataFrame, which keep
    # their metrics as they were when copied; copy.copy shares them, and so does a DataFrame
    # over a shared one.
    import copy
    import gc

    from pyspark.sql import Observation

    observation = Observation("deep")
    df = session.range(3).observe(observation, F.count(F.lit(1)).alias("c"))
    copied = copy.deepcopy(df)
    del df
    gc.collect()
    assert copied.count() == 3
    assert observation.get == {}

    shallow = Observation("shallow")
    df = session.range(4).observe(shallow, F.count(F.lit(1)).alias("c"))
    assert copy.copy(df).count() == 4 and shallow.get == {"c": 4}

    collected = Observation("collected")
    df = session.range(2).observe(collected, F.count(F.lit(1)).alias("c"))
    assert df.collect() == [Row(id=0), Row(id=1)] and collected.get == {"c": 2}
    late = copy.deepcopy(df)
    assert late.count() == 2 and collected.get == {"c": 2}

    first, second = Observation("join_first"), Observation("join_second")
    left = session.range(3).observe(first, F.count(F.lit(1)).alias("c"))
    right = session.range(5).observe(second, F.count(F.lit(1)).alias("c"))
    joined = left.join(right, "id")
    joined_copy = copy.deepcopy(joined)
    assert joined_copy.count() == 3
    assert first.get == {} and second.get == {}
    assert joined.count() == 3
    assert first.get == {"c": 3} and second.get == {"c": 5}


def test_copies_of_a_dataframe_set_the_copies_of_its_call_or_their_own_as_pyspark(session):
    # As PySpark's DataFrame.__reduce__: a deep copy copies the plan with the memo of the call, so
    # the copy of an Observation that the call copies too, in a tuple, a list or an object that
    # holds both, is the one whose metrics the actions of the copied DataFrame set; an unpickled
    # DataFrame, or an unpickled deep copy, sets copies of its own; the Observation keeps its
    # metrics. A shallow copy has an executionInfo of its own.
    import copy
    import pickle

    from pyspark import cloudpickle
    from pyspark.sql import Observation

    class Holder:
        def __init__(self, df, observation):
            self.df, self.observation = df, observation

    observation = Observation("of_the_call")
    df = session.range(3).observe(observation, F.count(F.lit(1)).alias("c"))
    holder = copy.deepcopy(Holder(df, observation))
    pairs = [copy.deepcopy((df, observation)), copy.deepcopy([observation, df])[::-1],
             (holder.df, holder.observation)]
    for copied, copied_observation in pairs:
        assert copied.count() == 3 and copied_observation.get == {"c": 3}
    assert observation.get == {}
    for unpickled in (pickle.loads(cloudpickle.dumps(df)),
                      pickle.loads(cloudpickle.dumps(copy.deepcopy(df)))):
        assert unpickled.count() == 3
    assert observation.get == {}

    named = session.range(3).observe("named_of_a_copy", F.count(F.lit(1)).alias("c"))
    shallow = copy.copy(named)
    assert shallow.collect() == [Row(id=0), Row(id=1), Row(id=2)]
    assert named.executionInfo is None
    assert shallow.executionInfo.flows == [("named_of_a_copy", {"c": 3})]
    assert copy.copy(shallow).executionInfo is None


def test_an_observation_copies_and_pickles_as_the_plain_object_of_pyspark(session):
    # As PySpark's Observation, a plain object: a shallow copy shares the metrics that the actions
    # set later, a deep copy keeps those that it had, and one pickled into the server's Python
    # worker, whose PySpark has no Rust client, unpickles as PySpark's of its name and metrics.
    import copy

    from pyspark.sql import Observation

    observation = Observation("copied")
    df = session.range(3).observe(observation, F.count(F.lit(1)).alias("c"))
    shallow, deep = copy.copy(observation), copy.deepcopy(observation)
    assert df.count() == 3
    assert observation.get == {"c": 3} and shallow.get == {"c": 3} and deep.get == {}

    @F.udf("string")
    def unpickled(_):
        return f"{type(observation).__module__} {observation._name} {observation.get}"

    assert session.range(1).select(unpickled("id")).first()[0] == (
        "pyspark.sql.connect.observation copied {'c': 3}")


def test_try_cast_and_isin_raise_the_errors_of_pyspark(session):
    # As PySpark Connect's: try_cast of an int, and isin of a tuple, raise PySpark's errors.
    from pyspark.errors import PySparkTypeError
    from pyspark.sql import functions as F

    df = session.range(3)
    with pytest.raises(PySparkTypeError) as e:
        df.select(F.col("id").try_cast(1))
    assert e.value.getCondition() == "NOT_EXPECTED_TYPE"
    with pytest.raises(PySparkTypeError) as e:
        df.where(F.col("id").isin((1, 2)))
    assert e.value.getCondition() == "UNSUPPORTED_DATA_TYPE"
    assert df.where(F.col("id").isin([1, 2])).count() == 2


def test_column_list_values_are_typed_array_literals_on_the_server_as_pyspark(session):
    # As PySpark Connect's: a list that a Column method takes as a value is an array of its
    # first element's type, so an empty one, or one that starts with None, is an error, while
    # F.lit of a list is an array() whatever it holds.
    from pyspark.errors import PySparkTypeError
    from pyspark.sql import functions as F

    df = session.range(2)
    chosen = F.when(F.col("id") > 0, F.array(F.lit(1)))
    for value in ([], [None, 2]):
        with pytest.raises(PySparkTypeError) as e:
            df.select(chosen.otherwise(value).alias("o"))
        assert e.value.getCondition() == "CANNOT_INFER_ARRAY_ELEMENT_TYPE"
    assert df.select(chosen.otherwise([1, 2]).alias("o")).collect() == [
        Row(o=[1, 2]), Row(o=[1])]
    assert session.range(1).select(F.when(F.lit(True), []).alias("w")).collect() == [
        Row(w=[])]




def test_ml_vector_functions_as_pyspark(session):
    from pyspark.ml.functions import array_to_vector, vector_to_array

    df = session.sql("SELECT array(1.5D, 2.0D) AS a").select(array_to_vector("a").alias("vec"))
    row = df.select(vector_to_array("vec").alias("d"), vector_to_array("vec", "float32").alias("f"))
    assert tuple(row.first()) == ([1.5, 2.0], [1.5, 2.0])


def test_udfs_of_types_with_user_defined_types_run_on_the_server_as_pyspark(session):
    # As PySpark's: a UDF of a return type of an array of ExamplePointUDT, whose JSON's pyClass
    # imports no class, runs on the Python worker of the server, to which the type pickles with
    # the user-defined type it is made of.
    from pyspark.testing.objects import ExamplePoint, ExamplePointUDT

    @F.udf(returnType=ArrayType(ExamplePointUDT()))
    def points(d):
        return [ExamplePoint(d + i, 10 * d + i) for i in range(2)]

    @F.udf(returnType=DoubleType())
    def last_y(es):
        return es[-1].y

    df = session.range(1).select(F.lit(1.0).alias("d"))
    expected = [Row(p=[ExamplePoint(1.0, 10.0), ExamplePoint(2.0, 11.0)])]
    assert df.select(points("d").alias("p")).collect() == expected
    assert df.select(last_y(points("d")).alias("y")).collect() == [Row(y=11.0)]




def test_progress_handlers_run_on_the_thread_of_an_interruptible_collect_as_pyspark():
    # As PySpark's: the handlers of an execution run in the thread that collects it, while the
    # signals of the process are polled for that collect.
    import subprocess
    import sys

    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live-server tests")
    result = subprocess.run([sys.executable, "-c", f"""
import threading
from pyspark.sql import SparkSession
spark = SparkSession.builder.remote({remote!r}).getOrCreate()
spark.conf.set("spark.connect.progress.reportInterval", "10ms")
main = threading.get_ident()
threads = []
spark.registerProgressHandler(lambda **kw: threads.append(threading.get_ident()))
print(spark.range(0, 2000, 1, 4).repartition(20).selectExpr("sum(id) as t").collect())
print("handler threads", len(threads), set(t == main for t in threads))
"""], env=dict(os.environ, TZ="UTC"), capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    calls = int(lines[-1].split()[2])
    assert calls >= 1, result.stdout
    assert lines[-1] == f"handler threads {calls:d} {{True}}", result.stdout


def test_sigint_during_action_raises_keyboard_interrupt_and_the_session_works_as_pyspark():
    # As PySpark's, from the issue: a SIGINT during a long action raises KeyboardInterrupt
    # within a second or two, the operation is cancelled on the server, and the session goes on.
    import subprocess
    import sys

    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live-server tests")
    result = subprocess.run([sys.executable, "-c", f"""
import os, signal, threading, time
from pyspark.sql import SparkSession, functions as F
spark = SparkSession.builder.remote({remote!r}).getOrCreate()
slow = spark.range(0, 10**12, 1, 2).agg(F.sum(F.sqrt("id")))
threading.Timer(3.0, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
started = time.monotonic()
try:
    slow.collect()
    print("collect returned")
except KeyboardInterrupt:
    print("interrupted after %.1f" % (time.monotonic() - started))
print("interrupted ids", spark.interruptAll())
print("count", spark.range(3).count())
"""], env=dict(os.environ, TZ="UTC"), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[0].startswith("interrupted after "), result.stdout
    assert 3.0 <= float(lines[0].rsplit(" ", 1)[1]) < 5.0, result.stdout
    assert lines[1:] == ["interrupted ids []", "count 3"], result.stdout


def test_sigint_interrupts_the_iterator_that_waits_and_not_a_held_one_as_pyspark():
    # As PySpark's: a SIGINT during the next of an iterator interrupts the operation that the next
    # waits for, though another iterator that was started later is alive and half-consumed. The
    # held iterator still yields its next row, and the waiting one is cancelled at once, not when
    # its slow partitions end.
    import subprocess
    import sys

    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live-server tests")
    result = subprocess.run([sys.executable, "-c", f"""
import os, signal, threading, time
from pyspark.sql import SparkSession, functions as F
spark = SparkSession.builder.remote({remote!r}).getOrCreate()
# Partition 0 returns at once; partitions 1 and 2 sleep 20 s on the server.
a = spark.range(0, 3, 1, 3).select(
    F.expr("reflect('java.lang.Thread', 'sleep', case when id = 0 then 0L else 20000L end)")
    .alias("x"), "id")
ia = a.toLocalIterator()
print("a", next(ia).id, flush=True)
ib = spark.range(0, 5, 1, 1).toLocalIterator()
print("b", next(ib).id, flush=True)
threading.Timer(2.0, lambda: os.kill(os.getpid(), signal.SIGINT)).start()
started = time.monotonic()
try:
    next(ia)
    print("next returned", flush=True)
except KeyboardInterrupt:
    print("interrupted after %.1f" % (time.monotonic() - started), flush=True)
print("b", next(ib).id, flush=True)
"""], env=dict(os.environ, TZ="UTC"), capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines()
    assert lines[:2] == ["a 0", "b 0"], result.stdout
    assert lines[2].startswith("interrupted after "), result.stdout
    assert 2.0 <= float(lines[2].rsplit(" ", 1)[1]) < 6.0, result.stdout
    assert lines[3:] == ["b 1"], result.stdout


def test_udfs_return_user_defined_types_of_main_as_pyspark():
    # As PySpark's: a UDT that __main__ defines, of parameters, is a UDF return type, as
    # cloudpickle sends its class by value and its state to the worker. In a new
    # interpreter, whose __main__ defines the classes.
    import subprocess
    import sys

    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live-server tests")
    subprocess.run([sys.executable, "-c", """
import sys
from pyspark.sql import Row, SparkSession, functions as F
from pyspark.sql.types import ArrayType, DoubleType, UserDefinedType

class Point:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def __eq__(self, other):
        return isinstance(other, Point) and (self.x, self.y) == (other.x, other.y)

class PointUDT(UserDefinedType):
    def __init__(self, scale=1.0):
        self.scale = scale

    @classmethod
    def sqlType(cls):
        return ArrayType(DoubleType(), False)

    @classmethod
    def module(cls):
        return "__main__"

    def serialize(self, obj):
        return [obj.x * self.scale, obj.y * self.scale]

    def deserialize(self, datum):
        return Point(datum[0], datum[1])

Point.__UDT__ = PointUDT()
spark = SparkSession.builder.remote(sys.argv[1]).getOrCreate()
df = spark.range(2).select(F.udf(lambda i: Point(i, i + 1), PointUDT(10.0))("id").alias("p"))
assert df.collect() == [Row(p=Point(0.0, 10.0)), Row(p=Point(10.0, 20.0))]
assert df.select(F.to_json(F.struct("p"))).collect()[1][0] == '{"p":[10.0,20.0]}'
""", remote], check=True)


def test_create_dataframe_schema_keeps_metadata_edited_in_place(session):
    # As PySpark: the schema of a DataFrame is the StructType it is created with, so a metadata
    # edited in place of one of its fields is in df.schema.
    from pyspark.sql import types as T

    st = T.StructType([T.StructField("a", T.IntegerType()), T.StructField("b", T.StringType())])
    st["a"].metadata["comment"] = "the a"
    df = session.createDataFrame([(1, "x")], st)
    assert df.schema["a"].metadata == {"comment": "the a"}
    assert df.schema.jsonValue()["fields"][0]["metadata"] == {"comment": "the a"}
    assert df.collect() == [Row(a=1, b="x")]


def test_collected_values_of_user_defined_types_are_deserialized_as_pyspark(session):
    # As official PySpark 4.3.0: a collected value of a UDT is its deserialized object, and
    # dtypes names the UDT by its simpleString(), in every way that rows are collected.
    from pyspark.ml.functions import array_to_vector
    from pyspark.ml.linalg import DenseVector

    df = session.sql(
        "SELECT array(1.5D, 2.0D) AS a, 1 AS k UNION ALL SELECT array(3.0D), 2"
    ).select(F.when(F.col("k") == 1, array_to_vector("a")).alias("vec"), "k").orderBy("k")
    vector = DenseVector([1.5, 2.0])
    rows = [Row(vec=vector, k=1), Row(vec=None, k=2)]
    assert df.dtypes == [("vec", "vector"), ("k", "int")]
    assert df.collect() == rows and df.first() == rows[0] and df.head() == rows[0]
    assert df.take(1) == rows[:1] and df.head(1) == rows[:1] and df.tail(1) == rows[1:]
    assert list(df.toLocalIterator()) == rows and list(df.toLocalIterator(True)) == rows
    assert type(df.first().vec) is DenseVector
    nested = df.limit(1).select(
        F.array("vec", "vec").alias("arr"),
        F.struct("vec", F.lit(1).alias("n")).alias("st"),
        F.create_map(F.lit("x"), "vec").alias("m"),
    )
    assert nested.dtypes == [
        ("arr", "array<vector>"), ("st", "struct<vec:vector,n:int>"), ("m", "map<string,vector>")]
    assert nested.first() == Row(arr=[vector, vector], st=Row(vec=vector, n=1), m={"x": vector})


def test_create_dataframe_merges_the_types_of_every_row_as_pyspark(session):
    import array
    import decimal

    Point = namedtuple("Point", ["x", "y"])

    class Attributes:
        def __init__(self):
            self.b = 1
            self.a = "x"

    cases = [
        # A string absorbs another atomic type, whose values become strings, and a null
        # takes the type of another row.
        ([(1, None), ("a", 2.5), (True, None)], "struct<_1:string,_2:double>",
         [("1", None), ("a", 2.5), ("true", None)]),
        ([([1, "a"],)], "struct<_1:array<string>>", [(["1", "a"],)]),
        ([{"a": 1}, {"a": "x"}], "struct<a:string>", [("1",), ("x",)]),
        ([({"a": [1]},), ({"a": [None]},)], "struct<_1:map<string,array<bigint>>>",
         [({"a": [1]},), ({"a": [None]},)]),
        # A decimal is decimal(38,18), a tuple, a namedtuple or an object a struct, and an
        # array.array an array of its typecode.
        ([(decimal.Decimal("1.5"),)], "struct<_1:decimal(38,18)>",
         [(decimal.Decimal("1.500000000000000000"),)]),
        ([((1, 2),)], "struct<_1:struct<_1:bigint,_2:bigint>>", [(Row(_1=1, _2=2),)]),
        ([(Point(1, 2),)], "struct<_1:struct<x:bigint,y:bigint>>", [(Row(x=1, y=2),)]),
        ([(Attributes(),)], "struct<_1:struct<a:string,b:bigint>>", [(Row(a="x", b=1),)]),
        ([(array.array("i", [1, 2]),)], "struct<_1:array<int>>", [([1, 2],)]),
    ]
    for data, schema, rows in cases:
        df = session.createDataFrame(data)
        assert df.schema.simpleString() == schema, data
        assert [tuple(r) for r in df.collect()] == rows, data
    for text in ["1.25", "a\x00b", "\uffff"]:
        # A string stays text, and shares a string column with an integer.
        df = session.createDataFrame([Row(value=text), Row(value=5)])
        assert [tuple(r) for r in df.collect()] == [(text,), ("5",)]


def test_foreach_batch_functions_that_capture_dataframes_run_as_pyspark(session, tmp_path):
    # As PySpark's: a DataFrame that a foreachBatch function captures unpickles on the Python
    # worker of the server, of the session of the query there, with its plan, whose columns refer
    # to it by its plan ID.
    (tmp_path / "input.txt").write_text("a\nb\n")
    captured = session.createDataFrame([("you", "can"), ("serialize", "dataframe")])
    kept = captured.filter(captured["_1"] == "you")
    name = "foreach_batch_captured_dataframe"

    def func(df, batch_id):
        kept.createOrReplaceTempView(name)

    query = session.readStream.format("text").load(str(tmp_path)).writeStream.foreachBatch(func)
    query = query.start()
    try:
        query.processAllAvailable()
        assert session.table(name).collect() == [Row(_1="you", _2="can")]
    finally:
        query.stop()
        session.catalog.dropTempView(name)


def test_foreach_batch_functions_that_capture_columns_run_as_pyspark(session, tmp_path):
    # As PySpark's test_streaming_foreach_batch_external_column: a Column that a foreachBatch
    # function captures unpickles on the Python worker of the server as PySpark Connect's, of
    # the expression as it was sent.
    (tmp_path / "input.txt").write_text("hello\nthis\n")
    captured = F.upper(F.col("value"))
    table = "foreach_batch_captured_column"

    def func(df, batch_id):
        df.select(captured.alias("result")).write.mode("append").saveAsTable(table)

    query = session.readStream.format("text").load(str(tmp_path)).writeStream.foreachBatch(func)
    query = query.start()
    try:
        query.processAllAvailable()
        assert sorted(row.result for row in session.table(table).collect()) == ["HELLO", "THIS"]
    finally:
        query.stop()
        session.sql(f"DROP TABLE IF EXISTS {table}").collect()


def test_foreach_runs_a_function_and_a_writer_of_each_row_as_pyspark(session, tmp_path):
    # As PySpark's foreach: a function of a row, and an object of open, process and close, run
    # on each row of each batch, here writing a file of each row on the local server.
    from pyspark.errors import PySparkAttributeError

    source, out = tmp_path / "source", tmp_path / "out"
    session.range(3).write.json(str(source))
    out.mkdir()
    directory = str(out)

    def write_row(row):
        import os

        open(os.path.join(directory, f"function_{row.id}"), "w").close()

    class Writer:
        def open(self, partition_id, epoch_id):
            return True

        def process(self, row):
            import os

            open(os.path.join(directory, f"writer_{row.id}"), "w").close()

        def close(self, error):
            pass

    for name, function in [("function", write_row), ("writer", Writer())]:
        query = (
            session.readStream.schema("id long")
            .json(str(source))
            .writeStream.foreach(function)
            .option("checkpointLocation", str(tmp_path / f"checkpoint_{name}"))
            .trigger(availableNow=True)
            .start()
        )
        query.awaitTermination()
        assert query.exception() is None
    expected = [f"{name}_{i}" for name in ("function", "writer") for i in range(3)]
    assert sorted(path.name for path in out.iterdir()) == expected
    with pytest.raises(PySparkAttributeError) as e:
        session.readStream.format("rate").load().writeStream.foreach(1)
    assert e.value.getCondition() == "ATTRIBUTE_NOT_CALLABLE"


def test_progress_of_queries_sums_the_rows_of_their_sources_as_pyspark(session):
    # As PySpark's: the server leaves the total of the rows of a progress out of its JSON, of
    # which the progress sums those of its sources.
    query = (
        session.readStream.format("rate").option("rowsPerSecond", 10).load()
        .writeStream.format("noop").start()
    )
    try:
        deadline = time.time() + 60
        while not any(p.numInputRows for p in query.recentProgress) and time.time() < deadline:
            time.sleep(0.5)
        progress = next(p for p in query.recentProgress if p.numInputRows)
        assert progress.numInputRows == sum(s.numInputRows for s in progress.sources) > 0
        assert progress.inputRowsPerSecond == sum(s.inputRowsPerSecond for s in progress.sources)
    finally:
        query.stop()


def test_dataframes_of_a_stream_are_streaming_as_pyspark(session):
    # As PySpark's, which asks the server: a DataFrame with a streaming source is streaming.
    stream = session.readStream.format("rate").load()
    for df in (
        stream,
        stream.select("value"),
        stream.where("value > 1"),
        stream.union(stream),
        session.range(1).join(stream.select(F.col("value").alias("id")), "id"),
    ):
        assert df.isStreaming is True
    assert session.range(1).isStreaming is False


def test_streaming_listeners_are_those_of_the_session_as_pyspark(session):
    # As PySpark Connect's spark.streams: one manager, of the listener bus of the session, whose
    # listeners get the session, are listed by _sqlb, get the started event of a query before
    # start returns, and get no events once removed.
    from pyspark.sql.streaming import StreamingQueryListener

    class Listener(StreamingQueryListener):
        def __init__(self):
            self.events = []

        def onQueryStarted(self, event):
            self.events.append(("started", str(event.id)))

        def onQueryProgress(self, event):
            self.events.append(("progress", str(event.progress.id)))

        def onQueryIdle(self, event):
            pass

        def onQueryTerminated(self, event):
            self.events.append(("terminated", str(event.id)))

    assert session.streams is session.streams
    first, second = Listener(), Listener()
    session.streams.addListener(first)
    session.streams.addListener(second)
    query = None
    try:
        assert first.spark.range(3).count() == 3
        assert session.streams._sqlb._listener_bus == [first, second]
        query = session.readStream.format("rate").load().writeStream.format("noop").start()
        assert first.events[:1] == second.events[:1] == [("started", query.id)]
        deadline = time.time() + 60
        while ("progress", query.id) not in first.events and time.time() < deadline:
            time.sleep(0.5)
        session.streams.removeListener(first)
        assert session.streams._sqlb._listener_bus == [second]
        seen = list(first.events)
        query.stop()
        deadline = time.time() + 30
        while ("terminated", query.id) not in second.events and time.time() < deadline:
            time.sleep(0.5)
        assert ("progress", query.id) in seen and first.events == seen
        assert ("terminated", query.id) in second.events
    finally:
        for listener in session.streams._sqlb._listener_bus:
            session.streams.removeListener(listener)
        if query is not None:
            query.stop()


def test_streaming_queries_return_the_status_and_the_exception_of_pyspark(session):
    # As PySpark's: the status of a query is a dict, and the exception of a failed query is
    # the StreamingQueryException of pyspark.errors, of the message of the server and its JVM
    # stack trace, which awaitTermination raises.
    import time

    from pyspark.errors import StreamingQueryException
    from pyspark.sql import functions as F

    stream = session.readStream.format("rate").load()
    query = stream.writeStream.format("noop").start()
    try:
        assert type(query.status) is dict
        assert sorted(query.status) == ["isDataAvailable", "isTriggerActive", "message"]
        query.explain(1)
        assert query.exception() is None
    finally:
        query.stop()
    failing = stream.select(F.assert_true(F.col("value") < 0)).writeStream.format("noop")
    failed = failing.start()
    try:
        for _ in range(120):
            if not failed.isActive:
                break
            time.sleep(0.5)
        exception = failed.exception()
        assert isinstance(exception, StreamingQueryException)
        message = str(exception)
        assert message.startswith("(STREAM_FAILED) [USER_RAISED_EXCEPTION]")
        assert "JVM stacktrace:" in message
        with pytest.raises(StreamingQueryException):
            failed.awaitTermination(1)
    finally:
        failed.stop()


def test_builder_config_none_leaves_the_key_unset_as_pyspark(session):
    # As PySpark's `_set_all`: an option of None, of config or of map, leaves its key unset on
    # the session, while the other options of the builder are set.
    built = (SparkSession.builder.remote(os.environ["SPARK_REMOTE"])
             .config("spark.r12.none", None)
             .config("spark.r12.kept", "v")
             .config(map={"spark.r12.map": None})
             .getOrCreate())
    assert built.session_id == session.session_id
    assert session.conf.get("spark.r12.kept") == "v"
    assert session.conf.get("spark.r12.none", "unset") == "unset"
    assert session.conf.get("spark.r12.map", "unset") == "unset"
    assert "spark.r12.none" not in session.conf.getAll
    with pytest.raises(Exception) as e:
        session.conf.get("spark.r12.none")
    assert e.value.getCondition() == "SQL_CONF_NOT_FOUND"


def test_stopped_session_raises_no_active_session_while_the_others_work(session):
    from pyspark.errors.exceptions.connect import SparkConnectException

    other = session.newSession()
    other.stop()
    assert other.is_stopped
    with pytest.raises(SparkConnectException) as e:
        other.range(2).count()
    assert e.value.getCondition() == "NO_ACTIVE_SESSION"
    assert session.range(3).count() == 3


def test_stream_writer_starts_a_query_of_its_real_time_trigger_as_pyspark(session):
    # As PySpark's, the real-time trigger reaches the server, which runs real-time mode for
    # some sinks only, not the memory sink.
    from pyspark.errors import IllegalArgumentException

    writer = session.readStream.format("rate").load().writeStream.format("memory")
    with pytest.raises(IllegalArgumentException) as error:
        query = writer.queryName("real_time_trigger").trigger(realTime="5 seconds").start()
        try:
            query.awaitTermination(10)
        finally:
            query.stop()
    assert error.value.getCondition() == (
        "STREAMING_REAL_TIME_MODE.OPERATOR_OR_SINK_NOT_IN_ALLOWLIST")


def test_udf_profiles_are_shown_by_spark_profile_as_pyspark(session, capsys):
    # As PySpark Connect's: with the UDF profiler, spark.profile shows the profile of each UDF
    # that an execution ran, which clear removes.
    from pyspark.sql.functions import udf

    session.profile.clear()
    session.conf.set("spark.sql.pyspark.udf.profiler", "perf")
    try:
        @udf("long")
        def plus_one(x):
            return x + 1

        session.range(4).select(plus_one("id")).collect()
    finally:
        session.conf.unset("spark.sql.pyspark.udf.profiler")
    capsys.readouterr()
    session.profile.show(type="perf")
    assert "Profile of UDF<id=" in capsys.readouterr().out
    session.profile.clear()
    session.profile.show()
    assert capsys.readouterr().out == ""




def test_create_dataframe_raises_merge_errors_as_pyspark(session):
    from pyspark.errors import PySparkException

    cases = [
        ([(1,), (2.5,)], "CANNOT_MERGE_TYPE", {"data_type1": "LongType", "data_type2": "DoubleType"}),
        # A merge error within a value names its field.
        ([([1, 2.5],)], "CANNOT_INFER_TYPE_FOR_FIELD", {"field_name": "_1"}),
        ([({"a": 1, "b": 2.5},)], "CANNOT_INFER_TYPE_FOR_FIELD", {"field_name": "_1"}),
        # A type that stays null cannot be determined.
        ([({"k": None},)], "CANNOT_DETERMINE_TYPE", {}),
        ([([None],)], "CANNOT_DETERMINE_TYPE", {}),
    ]
    for data, condition, parameters in cases:
        with pytest.raises(PySparkException) as error:
            session.createDataFrame(data)
        assert error.value.getCondition() == condition, data
        assert error.value.getMessageParameters() == parameters, data


@pytest.mark.parametrize("make, schema, expected_type, expected", [
    (lambda pd, np: pd.DataFrame([[None], [3]]), "x int", "struct<x:int>", [(None,), (3,)]),
    (lambda pd, np: pd.DataFrame(np.array([[1.0, np.nan], [2.0, 3.0]])), "a int, b int",
     "struct<a:int,b:int>", [(1, None), (2, 3)]),
    (lambda pd, np: pd.DataFrame([[Row(a=1, b=2)], [Row(a=3, b=4)]]), "x array<bigint>",
     "struct<x:array<bigint>>", [([1, 2],), ([3, 4],)]),
    (lambda pd, np: pd.DataFrame([[{"u": 1, "v": "a"}], [{"u": 2, "v": "b"}]]), None,
     "struct<0:struct<u:bigint,v:string>>", [(Row(u=1, v="a"),), (Row(u=2, v="b"),)]),
    (lambda pd, np: pd.DataFrame([[[datetime(2020, 1, 1, 12)]], [[datetime(2021, 1, 1)]]]), None,
     "struct<0:array<timestamp_ntz>>", [([datetime(2020, 1, 1, 12)],), ([datetime(2021, 1, 1)],)]),
    (lambda pd, np: pd.DataFrame([[[Decimal("1.5"), Decimal("10.25")]], [[Decimal("2")]]]), None,
     "struct<0:array<decimal(4,2)>>",
     [([Decimal("1.50"), Decimal("10.25")],), ([Decimal("2.00")],)]),
])
def test_default_range_index_pandas_frames_convert_as_pyspark(
    session, make, schema, expected_type, expected
):
    # Plain frames with default (RangeIndex) labels convert with Arrow, and give the results
    # of PySpark 4.3.0 for these frames.
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    data = make(pd, np)
    df = session.createDataFrame(data) if schema is None else session.createDataFrame(data, schema)
    assert df.schema.simpleString() == expected_type
    assert [tuple(row) for row in df.collect()] == expected


def test_samples_and_splits_take_a_seed_of_the_client_as_pyspark(session):
    # As PySpark's: without a seed, the client takes one, so every action of a sample returns
    # the same rows, and the splits of randomSplit are disjoint and have every row; and the
    # metadata of withMetadata takes any JSON value.
    df = session.range(0, 2000, 1, 8)
    sample = df.sample(0.5)
    assert sorted(row.id for row in sample.collect()) == sorted(row.id for row in sample.collect())
    first, second = df.randomSplit([0.5, 0.5])
    assert first.count() + second.count() == 2000
    assert first.intersect(second).count() == 0
    metadata = {"k": 1, "l": [1, 2], "s": "v"}
    assert df.withMetadata("id", metadata).schema["id"].metadata == metadata


def test_random_split_seed_of_a_float_or_str_is_converted_with_int_as_pyspark(session):
    # As PySpark's randomSplit: the seed goes through int(), so a float seed splits as its
    # truncation and a str seed as its int, and a str that int() rejects raises its ValueError.
    df = session.range(0, 2000, 1, 8)

    def ids(part):
        return sorted(row.id for row in part.collect())

    expected = [ids(part) for part in df.randomSplit([0.5, 0.5], 3)]
    assert [ids(part) for part in df.randomSplit([0.5, 0.5], 3.7)] == expected
    assert [ids(part) for part in df.randomSplit([0.5, 0.5], "3")] == expected
    with pytest.raises(ValueError, match="invalid literal for int"):
        df.randomSplit([0.5, 0.5], "3.5")


def test_answers_that_pyspark_keeps_are_asked_once_per_dataframe_as_pyspark(session):
    # As PySpark's isStreaming (a cached_property), semanticHash, isEmpty and inputFiles
    # (functools.cache): a DataFrame keeps the server's first answer, so a redefined view or a
    # new row does not change it, and the answers stay after the session stopped.
    table = f"cached_answers_{os.getpid()}"
    session.sql(f"DROP TABLE IF EXISTS {table}")
    session.sql(f"CREATE TABLE {table} (id BIGINT) USING parquet")
    try:
        rows = session.table(table)
        assert rows.isEmpty() is True
        session.sql(f"INSERT INTO {table} VALUES (1)")
        assert rows.isEmpty() is True and session.table(table).isEmpty() is False
    finally:
        session.sql(f"DROP TABLE IF EXISTS {table}")
    session.range(1).createOrReplaceTempView("cached_answers_view")
    view = session.table("cached_answers_view")
    is_streaming, semantic_hash, input_files = (
        view.isStreaming, view.semanticHash(), view.inputFiles())
    session.readStream.format("rate").load().createOrReplaceTempView("cached_answers_view")
    assert session.table("cached_answers_view").isStreaming is True
    assert (view.isStreaming, view.semanticHash(), view.inputFiles()) == (
        False, semantic_hash, input_files)
    session.catalog.dropTempView("cached_answers_view")
    session.stop()
    assert (view.isStreaming, view.semanticHash(), view.inputFiles()) == (
        is_streaming, semantic_hash, input_files)


def test_an_observation_of_a_year_month_interval_fails_the_action_as_pyspark(session):
    # As PySpark's client, which converts the metrics of an Observation of the plan as they
    # arrive: a year-month interval, which LiteralExpression._to_value does not convert, raises
    # UNSUPPORTED_LITERAL from the action, and the Observation has no metrics.
    from pyspark.errors import PySparkTypeError
    from pyspark.sql import Observation

    observation = Observation("year_month_interval_metric")
    interval = F.make_ym_interval(F.lit(1), F.col("id").cast("int"))
    df = session.range(3).observe(observation, F.max(interval).alias("m"))
    with pytest.raises(PySparkTypeError) as error:
        df.count()
    assert error.value.getCondition() == "UNSUPPORTED_LITERAL"
    assert error.value.getMessageParameters() == {"literal": "year_month_interval: 14\n"}
    assert observation.get == {}


def test_only_the_observations_of_the_plan_of_an_action_convert_its_metrics_as_pyspark(session):
    # As PySpark's client, which converts and sets the metrics of the Observations of the plan of
    # an action, and runs localCheckpoint and a SQL statement without them: a year-month interval
    # metric fails neither those nor a subquery of the observed DataFrame, an Observation keeps
    # the metrics of the earlier actions when an action fails, and a metric of a name that holds
    # an interval as an element raises when executionInfo reads it.
    from pyspark.errors import PySparkTypeError
    from pyspark.sql import Observation

    interval = F.expr("INTERVAL '1' YEAR")
    observation = Observation("unset_by_commands_and_subqueries")
    df = session.range(3).observe(observation, F.max(interval).alias("ym"))
    assert df.localCheckpoint().count() == 3
    assert session.sql("SELECT * FROM {d}", d=df).count() == 3
    assert session.range(2).select(df.select(F.max("id")).scalar().alias("m")).collect() == [
        Row(m=2), Row(m=2)]
    assert observation.get == {}

    table = f"observation_kept_metrics_{os.getpid()}"
    session.sql(f"DROP TABLE IF EXISTS {table}")
    session.sql(f"CREATE TABLE {table} (x INT) USING parquet")
    try:
        kept = Observation("kept_after_a_failing_action")
        observed = session.table(table).observe(
            kept, F.max(F.when(F.col("x") > 0, interval)).alias("ym"),
            F.count(F.lit(1)).alias("c"))
        assert observed.count() == 0 and kept.get == {"ym": None, "c": 0}
        session.sql(f"INSERT INTO {table} VALUES (1)")
        with pytest.raises(PySparkTypeError) as error:
            observed.count()
        assert error.value.getCondition() == "UNSUPPORTED_LITERAL"
        assert error.value._operation_id is not None
        assert kept.get == {"ym": None, "c": 0}
    finally:
        session.sql(f"DROP TABLE IF EXISTS {table}")

    named = session.range(2).observe("holds_an_interval", F.collect_list(interval).alias("m"))
    assert named.collect() == [Row(id=0), Row(id=1)]
    with pytest.raises(PySparkTypeError) as error:
        named.executionInfo.flows
    assert error.value.getMessageParameters() == {"literal": "year_month_interval: 12\n"}


def test_sample_split_and_sample_by_without_a_seed_follow_python_random_as_pyspark(
        session, tmp_path):
    # As PySpark's: without a seed, sample, randomSplit and sampleBy draw one from Python's
    # random, so random.seed(42) gives the same rows in every process. The rows are those of
    # the reference client for the same seed. The rows of a sample depend on the partitions,
    # so the ranges have 4 of their own rather than the server's default parallelism.
    import json
    import subprocess
    import sys

    script = tmp_path / "seeded_rows.py"
    script.write_text(
        "import json, os, random\n"
        "from pyspark.sql import SparkSession\n"
        "spark = SparkSession.builder.remote(os.environ['SPARK_REMOTE']).getOrCreate()\n"
        "df = spark.range(0, 40, 1, 4)\n"
        "def ids(frame):\n"
        "    return sorted(row.id for row in frame.collect())\n"
        "random.seed(42)\n"
        "sample = ids(df.sample(0.5))\n"
        "random.seed(42)\n"
        "splits = [ids(part) for part in df.randomSplit([0.5, 0.5])]\n"
        "stratified = spark.range(0, 40, 1, 4).selectExpr('id', 'id % 3 AS k')\n"
        "random.seed(42)\n"
        "by = ids(stratified.sampleBy('k', {0: 0.5, 1: 0.5, 2: 0.5}))\n"
        "print(json.dumps([sample, *splits, by]))\n"
        "spark.stop()\n")

    def rows():
        result = subprocess.run(
            [sys.executable, str(script)], env=os.environ, capture_output=True, text=True,
            timeout=300)
        assert result.returncode == 0, result.stderr
        return json.loads(result.stdout.splitlines()[-1])

    expected = [
        [1, 2, 3, 8, 9, 10, 12, 13, 14, 16, 17, 19, 20, 21, 23, 25, 26, 32, 36, 37],
        [1, 2, 3, 8, 9, 10, 12, 13, 14, 16, 17, 19, 20, 21, 23, 25, 26, 32, 36, 37],
        [0, 4, 5, 6, 7, 11, 15, 18, 22, 24, 27, 28, 29, 30, 31, 33, 34, 35, 38, 39],
        [1, 2, 3, 8, 9, 10, 12, 13, 14, 16, 17, 19, 20, 21, 23, 25, 26, 32, 36, 37],
    ]
    # Verified against the reference client 4.3.0 on the same server, with the same seeding.
    assert rows() == expected
    assert rows() == expected




def test_persist_cache_and_unpersist_return_the_dataframe_itself_as_pyspark(session):
    # As PySpark's: the DataFrame itself, cached at the storage level given.
    from pyspark.storagelevel import StorageLevel

    df = session.range(3)
    try:
        assert df.persist(StorageLevel.DISK_ONLY) is df
        assert df.storageLevel == StorageLevel.DISK_ONLY
        assert df.unpersist(blocking=True) is df
        assert df.cache() is df and df.count() == 3
    finally:
        df.unpersist()


def test_persist_of_none_caches_at_the_default_storage_level_of_the_server_as_pyspark(session):
    # As PySpark's persist(None), which sends no storage level, so that the server caches at its
    # spark.sql.defaultCacheStorageLevel, while an omitted level is MEMORY_AND_DISK_DESER.
    from pyspark.storagelevel import StorageLevel

    session.conf.set("spark.sql.defaultCacheStorageLevel", "DISK_ONLY")
    given, omitted = session.range(5), session.range(6)
    try:
        assert given.persist(None) is given and given.storageLevel == StorageLevel.DISK_ONLY
        assert omitted.persist().storageLevel == StorageLevel.MEMORY_AND_DISK_DESER
    finally:
        given.unpersist()
        omitted.unpersist()
        session.conf.unset("spark.sql.defaultCacheStorageLevel")


def test_values_of_local_rows_convert_and_raise_as_pyarrow(session):
    # As PySpark Connect's: PyArrow takes the __int__ of a Decimal in an int or a date field,
    # raises its own errors, and takes the days of a date out of the dates of Python, which
    # then raise the OverflowError of PyArrow's `date(1970, 1, 1) + timedelta(days=d)`.
    import decimal

    import pyarrow as pa
    from pyspark.errors import PySparkTypeError

    D = decimal.Decimal
    df = session.createDataFrame([(D("2.7"), D("1.5"))], "i int, d date")
    assert df.collect() == [Row(i=2, d=date(1970, 1, 2))]
    with pytest.raises(pa.ArrowInvalid, match="tried to convert to int32"):
        session.createDataFrame([("x",)], "v int")
    with pytest.raises(PySparkTypeError) as e:
        session.createDataFrame([("x",)], "v char(3)")
    assert e.value.getCondition() == "UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION"
    df = session.createDataFrame([(2932897,)], "v date")
    assert df.count() == 1
    with pytest.raises(OverflowError, match="^date value out of range$"):
        # PySpark raises in collect; the client's rows convert their values when read.
        df.collect()[0].v

def test_create_dataframe_converts_numpy_scalars_as_pyspark(session):
    np = pytest.importorskip("numpy")
    df = session.createDataFrame(
        [(np.int64(1), np.float32(0.5), np.bool_(True), np.float32(1.5))],
        "a bigint, b double, c boolean, d bigint")
    assert [tuple(r) for r in df.collect()] == [(1, 0.5, True, 1)]


def test_catalogs_return_the_named_tuples_of_pyspark(session):
    # As PySpark's: the methods of the catalog return the NamedTuples of pyspark.sql.catalog,
    # and registerFunction registers a UDF with spark.udf.register, with its FutureWarning.
    from pyspark.sql.catalog import Column, Database, Function, Table

    session.sql("drop table if exists default.catalog_tuples_t")
    session.sql("create table default.catalog_tuples_t (a int, b string) using parquet")
    try:
        table = session.catalog.getTable("default.catalog_tuples_t")
        assert isinstance(table, Table)
        assert table == (
            "catalog_tuples_t", "spark_catalog", ["default"], None, "MANAGED", False)
        assert table._asdict()["tableType"] == "MANAGED" and table.database == "default"
        assert session.catalog.listColumns("default.catalog_tuples_t") == [
            Column("a", None, "int", True, False, False, False),
            Column("b", None, "string", True, False, False, False),
        ]
        assert isinstance(session.catalog.getDatabase("default"), Database)
        name, *_ = session.catalog.getFunction("abs")
        assert name == "abs" and isinstance(session.catalog.getFunction("abs"), Function)
        with pytest.warns(FutureWarning, match="Use spark.udf.register instead"):
            session.catalog.registerFunction("catalog_tuples_plus_one", lambda x: x + 1, "int")
        assert session.sql("select catalog_tuples_plus_one(1) as v").first().v == 2
    finally:
        session.sql("drop table if exists default.catalog_tuples_t")




def test_create_dataframe_infers_by_the_session_configurations_as_pyspark(session):
    naive = datetime(2020, 1, 2, 3, 4, 5)
    cases = [
        ("spark.sql.pyspark.legacy.inferMapTypeFromFirstPair.enabled", "true",
         [({"a": 1, "b": 2.5},)], "struct<_1:map<string,bigint>>", [({"a": 1, "b": 2},)]),
        ("spark.sql.pyspark.legacy.inferArrayTypeFromFirstElement.enabled", "true",
         [([1, 2.5],)], "struct<_1:array<bigint>>", [([1, 2],)]),
        ("spark.sql.pyspark.inferNestedDictAsStruct.enabled", "true",
         [({"a": {"b": 1}, "c": None},)], "struct<_1:struct<a:struct<b:bigint>>>",
         [(Row(a=Row(b=1)),)]),
        # A naive datetime is TIMESTAMP_NTZ, but not in a list, whose elements PySpark
        # infers without `prefer_timestamp_ntz`.
        ("spark.sql.timestampType", "TIMESTAMP_NTZ", [(naive,)], "struct<_1:timestamp_ntz>",
         [(naive,)]),
        ("spark.sql.timestampType", "TIMESTAMP_NTZ", [([naive],)], "struct<_1:array<timestamp>>",
         None),
        # PySpark passes the elements of a list `prefer_timestamp_ntz` for
        # `infer_map_from_first_pair`, so that a dict in a list takes the map type of its
        # first pair.
        ("spark.sql.timestampType", "TIMESTAMP_NTZ", [([{"a": 1, "b": 2.5}],)],
         "struct<_1:array<map<string,bigint>>>", [([{"a": 1, "b": 2}],)]),
    ]
    for key, value, data, schema, rows in cases:
        session.conf.set(key, value)
        try:
            df = session.createDataFrame(data)
            assert df.schema.simpleString() == schema, (key, data)
            if rows is not None:
                assert [tuple(r) for r in df.collect()] == rows, (key, data)
        finally:
            session.conf.unset(key)


def test_na_and_stat_functions_take_their_arguments_as_pyspark(session):
    # As PySpark's: the subset of dropna, a column name or a list or tuple of them, the columns
    # of describe, names or a list of them, and the quantiles of approxQuantile of a list or
    # tuple of columns, a list of those of each.
    df = session.createDataFrame(
        [(1, "a", None), (None, "b", 2.0), (3, None, 3.0)], "x int, k string, v double"
    )
    kept = [Row(x=1, k="a", v=None), Row(x=3, k=None, v=3.0)]
    assert df.dropna(subset="x").collect() == kept
    assert df.na.drop(subset=("x",)).collect() == kept
    assert df.dropna(how="all", thresh=3).collect() == []
    assert df.describe(["x", "v"]).columns == ["summary", "x", "v"]
    assert df.describe(("x", "v")).columns == ["summary", "x", "v"]
    assert df.approxQuantile(["x", "v"], [0.0, 1.0], 0.0) == [[1.0, 3.0], [2.0, 3.0]]
    assert df.stat.approxQuantile("x", (0.0, 1), 0) == [1.0, 3.0]
    linear = session.createDataFrame([(1, 2.0), (2, 4.0), (3, 6.0)], "a int, b double")
    assert linear.corr("a", "b", None) == linear.stat.corr("a", "b", "pearson") == 1.0
    assert df.freqItems(["k"]).columns == ["k_freqItems"]


def test_functions_apply_their_optional_arguments_as_pyspark(session):
    df = session.sql(
        "SELECT 'key' AS k, 'msg' AS m, parse_json('{\"a\": 1}') AS v, "
        "array(3.0F, 4.0F) AS vec, 1.0F AS one")
    row = df.select(
        F.hex(F.hmac("k", "m", F.lit("SHA-1"))).alias("sha1"),
        F.to_json(F.variant_set("v", "$.b", F.lit(2), False)).alias("kept"),
        F.to_json(F.try_variant_set("v", "$.b", F.lit(2))).alias("set"),
        F.vector_norm("vec", F.col("one")).alias("l1"),
        F.vector_normalize("vec", F.col("one")).alias("normalized"),
    ).first()
    assert row.sha1 == "102900B72B7BF1031EEC76B4804B66052376896B"
    assert row.kept == '{"a":1}'
    assert row.set == '{"a":1,"b":2}'
    assert row.l1 == 7.0
    assert row.normalized == pytest.approx([3 / 7, 4 / 7])


def test_logistic_regression_fits_the_data_and_predicts_on_new_rows(session):
    # The native estimator targets Spark's JVM LogisticRegression. Compare the same
    # behavior with that estimator in official PySpark, rather than its deprecated
    # torch-based pyspark.ml.connect.classification implementation.
    LogisticRegression = _native_ml("LogisticRegression", "pyspark.ml.classification")
    from pyspark.ml.connect.functions import array_to_vector

    training = session.sql(
        "SELECT array(CAST(x AS DOUBLE)) AS xs, "
        "CAST(CASE WHEN x > 0 THEN 1 ELSE 0 END AS DOUBLE) AS target "
        "FROM VALUES (-4), (-3), (-2), (-1), (1), (2), (3), (4) AS t(x)"
    ).withColumn("input", array_to_vector("xs"))
    held_out = session.sql(
        "SELECT id, array(CAST(x AS DOUBLE)) AS xs "
        "FROM VALUES (0, -3.5), (1, -1.5), (2, 1.5), (3, 3.5) AS t(id, x)"
    ).withColumn("input", array_to_vector("xs"))
    estimator = LogisticRegression(
        featuresCol="input", labelCol="target", predictionCol="pred", maxIter=30
    )
    first = estimator.fit(training)
    second = estimator.fit(training.withColumn("target", F.lit(1.0) - F.col("target")))
    assert [r.pred for r in first.transform(held_out).orderBy("id").collect()] == [
        0.0, 0.0, 1.0, 1.0
    ]
    assert [r.pred for r in second.transform(held_out).orderBy("id").collect()] == [
        1.0, 1.0, 0.0, 0.0
    ]
    with pytest.raises(Exception):
        estimator.fit(session.range(3))


def test_linear_svc_releases_its_cached_model_once_the_last_owner_is_gone(session):
    # As pyspark.ml does, the server deletes a fitted model's cached object once no model,
    # copy or DataFrame that the model transformed is left in Python.
    import gc

    import pyspark.sql.connect.proto as pb2

    LinearSVC = _native_ml("LinearSVC", "pyspark.ml.classification")
    from pyspark.ml.connect.functions import array_to_vector

    # A session of its own: the ML cache is per session, so the models of the other tests,
    # whose releases may still be on their way, do not change the counts below.
    spark = session.newSession()

    def cached_models():
        cmd = pb2.Command()
        cmd.ml_command.get_cache_info.SetInParent()
        _, props, _ = spark.client.execute_command(cmd)
        return len(props.get("ml_command_result").param.array.elements)

    def wait_for_cached_models(expected):
        # The release is sent without waiting for the server, so poll for it to land.
        deadline = time.time() + 30
        while cached_models() != expected and time.time() < deadline:
            time.sleep(0.05)
        return cached_models()

    try:
        training = spark.sql(
            "SELECT array(CAST(x AS DOUBLE)) AS xs, "
            "CAST(CASE WHEN x > 0 THEN 1 ELSE 0 END AS DOUBLE) AS target "
            "FROM VALUES (-4), (-3), (-2), (-1), (1), (2), (3), (4) AS t(x)"
        ).withColumn("input", array_to_vector("xs"))
        held_out = spark.sql(
            "SELECT id, array(CAST(x AS DOUBLE)) AS xs "
            "FROM VALUES (0, -3.5), (1, -1.5), (2, 1.5), (3, 3.5) AS t(id, x)"
        ).withColumn("input", array_to_vector("xs"))
        assert cached_models() == 0
        for _ in range(10):
            model = LinearSVC(featuresCol="input", labelCol="target", maxIter=2).fit(training)
            del model
        gc.collect()
        assert wait_for_cached_models(0) == 0

        model = LinearSVC(featuresCol="input", labelCol="target", predictionCol="pred",
                          maxIter=50).fit(training)
        lazy = model.transform(held_out)
        del model
        gc.collect()
        assert cached_models() == 1
        rows = lazy.orderBy("id").collect()
        assert [r.id for r in rows] == [0, 1, 2, 3]
        assert all(r.pred in (0.0, 1.0) for r in rows)
        del lazy
        gc.collect()
        assert wait_for_cached_models(0) == 0
    finally:
        spark.stop()


@pytest.mark.parametrize("use_setters", [False, True], ids=["constructor", "setters"])
def test_linear_svc_fits_the_labels_and_predicts_on_new_rows(session, use_setters):
    LinearSVC = _native_ml("LinearSVC", "pyspark.ml.classification")
    from pyspark.ml.connect.functions import array_to_vector

    training = session.sql(
        "SELECT array(CAST(x AS DOUBLE)) AS xs, "
        "CAST(CASE WHEN x > 0 THEN 1 ELSE 0 END AS DOUBLE) AS target, 1.0D AS weight "
        "FROM VALUES (-4), (-3), (-2), (-1), (1), (2), (3), (4) AS t(x)"
    ).withColumn("input", array_to_vector("xs"))
    held_out = session.sql(
        "SELECT id, array(CAST(x AS DOUBLE)) AS xs "
        "FROM VALUES (0, -3.5), (1, -1.5), (2, 1.5), (3, 3.5) AS t(id, x)"
    ).withColumn("input", array_to_vector("xs"))
    kwargs = dict(featuresCol="input", labelCol="target", predictionCol="pred",
                  rawPredictionCol="margin", maxIter=50, regParam=0.1, tol=1e-6,
                  fitIntercept=False, standardization=False, threshold=0.0,
                  weightCol="weight", aggregationDepth=2, maxBlockSizeInMB=1.0)
    if use_setters:
        estimator = LinearSVC()
        for name, value in kwargs.items():
            setter = getattr(estimator, "set" + name[0].upper() + name[1:])
            assert setter(value) is estimator
    else:
        estimator = LinearSVC(**kwargs)
    first = estimator.fit(training)
    second = estimator.fit(training.withColumn("target", F.lit(1.0) - F.col("target")))
    result = first.transform(held_out)
    assert "margin" in result.columns
    assert [r.pred for r in result.orderBy("id").collect()] == [0.0, 0.0, 1.0, 1.0]
    assert [r.pred for r in second.transform(held_out).orderBy("id").collect()] == [
        1.0, 1.0, 0.0, 0.0
    ]
    estimator.setThreshold(1e6)
    assert [r.pred for r in estimator.fit(training).transform(held_out).orderBy("id").collect()] == [0.0] * 4
    with pytest.raises(Exception):
        estimator.fit(session.range(3))
    with pytest.raises(Exception):
        estimator.fit(training.withColumn("target", F.lit(2.0)))
    with pytest.raises(Exception):
        estimator.setRegParam(-1.0).fit(training)


def _linear_regression():
    return _native_ml("LinearRegression", "pyspark.ml.regression")


@pytest.mark.parametrize("use_setters", [False, True], ids=["constructor", "setters"])
def test_linear_regression_learns_the_data_and_predicts_unseen_rows(session, use_setters):
    from pyspark.ml.connect.functions import array_to_vector
    LinearRegression = _linear_regression()
    training = session.sql(
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(2*x+3 AS DOUBLE) AS target, 1.0D AS weight "
        "FROM VALUES (-4), (-3), (-2), (-1), (1), (2), (3), (4) AS t(x)"
    ).withColumn("input", array_to_vector("xs"))
    held_out = session.sql(
        "SELECT id, array(CAST(x AS DOUBLE)) AS xs, CAST(2*x+3 AS DOUBLE) AS target "
        "FROM VALUES (0, -1.5), (1, 0.5), (2, 4.5) AS t(id,x)"
    ).withColumn("input", array_to_vector("xs"))
    kwargs = dict(featuresCol="input", labelCol="target", predictionCol="pred",
                  maxIter=100, regParam=0.0, elasticNetParam=0.0, tol=1e-8,
                  fitIntercept=True, standardization=False, solver="normal",
                  weightCol="weight", aggregationDepth=2, loss="squaredError",
                  epsilon=1.35, maxBlockSizeInMB=1.0)
    estimator = LinearRegression() if use_setters else LinearRegression(**kwargs)
    if use_setters:
        for name, value in kwargs.items():
            if name == "loss" and type(estimator).__module__ == "pyspark.ml.regression":
                # Upstream 4.3.0 setLoss uses the nonexistent lossType parameter.
                # Compare the intended loss setting through its working setParams API.
                assert estimator.setParams(loss=value) is estimator
            else:
                assert getattr(estimator, "set" + name[0].upper() + name[1:])(value) is estimator
    first = estimator.fit(training)
    second = estimator.fit(training.withColumn("target", F.lit(10.0) - F.col("target")))
    assert [r.pred for r in first.transform(held_out).orderBy("id").collect()] == pytest.approx([0, 4, 12], abs=1e-6)
    assert [r.pred for r in second.transform(held_out).orderBy("id").collect()] == pytest.approx([10, 6, -2], abs=1e-6)
    with pytest.raises(Exception):
        estimator.fit(session.range(3))
    with pytest.raises(Exception):
        estimator.setRegParam(-1.0).fit(training)


def test_linear_regression_weights_and_iterative_losses(session):
    from pyspark.ml.connect.functions import array_to_vector
    LinearRegression = _linear_regression()
    training = session.sql(
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS label, CAST(w AS DOUBLE) AS weight "
        "FROM VALUES (-3,-3,1), (-2,-1,1), (2,7,1), (3,9,1), (100,-1000,0) AS t(x,y,w)"
    ).withColumn("features", array_to_vector("xs"))
    held_out = session.sql("SELECT array(2.0D) AS xs").withColumn("features", array_to_vector("xs"))
    default = LinearRegression().fit(training.filter(F.col("weight") > 0))
    weighted = LinearRegression(weightCol="weight").fit(training)
    assert default.transform(held_out).first().prediction == pytest.approx(7, abs=1e-6)
    assert weighted.transform(held_out).first().prediction == pytest.approx(7, abs=1e-6)
    regularized = LinearRegression(solver="l-bfgs", regParam=1.0, elasticNetParam=0.5).fit(
        training.filter(F.col("weight") > 0))
    assert 3 < regularized.transform(held_out).first().prediction < 6.9
    robust_training = session.sql(
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS label "
        "FROM VALUES (-3,-3.01), (-2,-0.98), (-1,1.02), (1,4.98), (2,7.01), (3,8.99), (0,1000) AS t(x,y)"
    ).withColumn("features", array_to_vector("xs"))
    robust = LinearRegression(loss="huber", solver="l-bfgs", maxIter=200, tol=1e-8).fit(robust_training)
    assert robust.transform(held_out).first().prediction == pytest.approx(7, abs=0.2)


def _native_ml(name, reference_module):
    return native_class("pyspark.ml.connect", name, reference_module)


def _kmeans():
    return _native_ml("KMeans", "pyspark.ml.clustering")


def _assert_cluster_partition(actual, expected):
    assert len(actual) == len(expected)
    assert all((a == b) == (x == y)
               for a, x in zip(actual, expected) for b, y in zip(actual, expected))


@pytest.mark.parametrize("use_setters", [False, True], ids=["constructor", "setters"])
def test_kmeans_learns_clusters_from_the_supplied_training_data(session, use_setters):
    from pyspark.ml.connect.functions import array_to_vector
    KMeans = _kmeans()
    def vectors(sql):
        return session.sql(sql).withColumn("input", array_to_vector("xs"))
    training = vectors("SELECT array(CAST(x AS DOUBLE)) AS xs, 1.0D AS weight "
                       "FROM VALUES (-11),(-10),(-9),(9),(10),(11) AS t(x)")
    shifted = vectors("SELECT array(CAST(x AS DOUBLE)) AS xs, 1.0D AS weight "
                      "FROM VALUES (9),(10),(11),(29),(30),(31) AS t(x)")
    held_out = vectors("SELECT id, array(CAST(x AS DOUBLE)) AS xs "
                       "FROM VALUES (0,-10),(1,-1),(2,10),(3,21),(4,30) AS t(id,x)")
    kwargs = dict(featuresCol="input", predictionCol="cluster", k=2, initMode="k-means||",
                  initSteps=3, tol=1e-6, maxIter=50, seed=42, distanceMeasure="euclidean",
                  weightCol="weight", solver="block" if use_setters else "row", maxBlockSizeInMB=1.0)
    estimator = KMeans() if use_setters else KMeans(**kwargs)
    if use_setters:
        for name, value in kwargs.items():
            assert getattr(estimator, "set" + name[0].upper() + name[1:])(value) is estimator
    first = estimator.fit(training)
    second = estimator.fit(shifted)
    predictions = lambda model: [r.cluster for r in model.transform(held_out).orderBy("id").collect()]
    original = predictions(first)
    _assert_cluster_partition(original, [0,0,1,1,1])
    _assert_cluster_partition(predictions(second), [0,0,0,1,1])
    assert original == predictions(estimator.fit(training))
    with pytest.raises(Exception):
        estimator.fit(session.range(3))


def test_kmeans_weights_move_the_cluster_boundary(session):
    from pyspark.ml.connect.functions import array_to_vector
    KMeans = _kmeans()
    training = session.sql(
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(w AS DOUBLE) AS weight "
        "FROM VALUES (0,1),(2,9),(10,1),(12,9) AS t(x,w)"
    ).withColumn("features", array_to_vector("xs"))
    held_out = session.sql(
        "SELECT id, array(CAST(x AS DOUBLE)) AS xs FROM VALUES (0,0),(1,6.5),(2,12) AS t(id,x)"
    ).withColumn("features", array_to_vector("xs"))
    plain = KMeans(seed=42).fit(training)
    weighted = KMeans(seed=42, weightCol="weight").fit(training)
    predictions = lambda model: [r.prediction for r in model.transform(held_out).orderBy("id").collect()]
    _assert_cluster_partition(predictions(plain), [0,1,1])
    _assert_cluster_partition(predictions(weighted), [0,0,1])
    with pytest.raises(Exception):
        KMeans(k=1).fit(training)
    with pytest.raises(Exception):
        KMeans(solver="invalid").fit(training)
    with pytest.raises(Exception):
        KMeans(weightCol="weight").fit(training.withColumn("weight", F.lit(-1.0)))


def test_kmeans_cosine_distance_groups_vectors_by_direction(session):
    from pyspark.ml.connect.functions import array_to_vector
    KMeans = _kmeans()
    training = session.sql(
        "SELECT array(CAST(x AS DOUBLE),CAST(y AS DOUBLE)) AS xs "
        "FROM VALUES (1,0),(2,0),(0,1),(0,2) AS t(x,y)"
    ).withColumn("features", array_to_vector("xs"))
    held_out = session.sql(
        "SELECT id, array(CAST(x AS DOUBLE),CAST(y AS DOUBLE)) AS xs "
        "FROM VALUES (0,100,1),(1,1,0.01),(2,1,100),(3,0.01,1) AS t(id,x,y)"
    ).withColumn("features", array_to_vector("xs"))
    model = KMeans(seed=42, distanceMeasure="cosine", solver="row").fit(training)
    _assert_cluster_partition([r.prediction for r in model.transform(held_out).orderBy("id").collect()], [0,0,1,1])


def test_higher_order_functions_take_columns_and_reject_lists_as_pyspark(session):
    # As PySpark's, a Python list is not a column of a higher-order function, and an array
    # column is one, and the rows of the lambdas of both are those of the reference.
    from pyspark.errors import PySparkTypeError

    for build in (lambda: F.transform([1, 2, 3], lambda x: x * 10),
                  lambda: F.exists([1, 2], lambda x: x > 1)):
        with pytest.raises(PySparkTypeError) as error:
            session.range(1).select(build().alias("v")).collect()
        assert error.value.getCondition() == "NOT_EXPECTED_TYPE"
    array = F.array(F.lit(1), F.lit(2), F.lit(3))
    assert session.range(1).select(
        F.transform(array, lambda x: x * 10).alias("t")).collect() == [Row(t=[10, 20, 30])]
    assert session.range(1).select(
        F.exists(F.array(F.lit(1), F.lit(2)), lambda x: x > 1).alias("e")).collect() == [
            Row(e=True)]


# These cases run unchanged against the native binding and the official JVM-backed estimator.
def _random_forest_classifier():
    return _native_ml("RandomForestClassifier", "pyspark.ml.classification")


def _forest_vectors(session, sql, column="features"):
    from pyspark.ml.connect.functions import array_to_vector
    return session.sql(sql).withColumn(column, array_to_vector("xs"))


def _forest_predictions(model, frame, column="prediction"):
    return [r[column] for r in model.transform(frame).orderBy("id").collect()]


@pytest.mark.parametrize("use_setters", [False, True], ids=["constructor", "setters"])
def test_random_forest_learns_nonlinear_labels_and_preserves_lazy_predictions(session, use_setters):
    from pyspark.ml.connect.functions import vector_to_array
    RandomForestClassifier = _random_forest_classifier()
    training = _forest_vectors(session,
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS target, 1.0D AS weight "
        "FROM VALUES (-3,0),(-2,0),(-1,1),(0,1),(1,1),(2,0),(3,0) AS t(x,y)", "input")
    held_out = _forest_vectors(session,
        "SELECT id, array(CAST(x AS DOUBLE)) AS xs "
        "FROM VALUES (0,-2.5),(1,-0.5),(2,0.5),(3,2.5) AS t(id,x)", "input")
    kwargs = dict(featuresCol="input", labelCol="target", predictionCol="class",
                  probabilityCol="probs", rawPredictionCol="votes", maxDepth=3,
                  maxBins=16, minInstancesPerNode=1, minInfoGain=0.0, pruneTree=False,
                  maxMemoryInMB=64, cacheNodeIds=True, checkpointInterval=-1,
                  impurity="entropy", numTrees=3, featureSubsetStrategy="all", seed=42,
                  subsamplingRate=1.0, leafCol="leaves", minWeightFractionPerNode=0.0,
                  weightCol="weight", bootstrap=False)
    if use_setters:
        estimator = RandomForestClassifier()
        for name, value in kwargs.items():
            assert getattr(estimator, "set" + name[0].upper() + name[1:])(value) is estimator
    else:
        estimator = RandomForestClassifier(**kwargs)
    model = estimator.fit(training)
    other = estimator.fit(training.withColumn("target", 1.0 - F.col("target")))
    delayed = model.transform(held_out).select("id", "class")
    rows = model.transform(held_out).orderBy("id").select(
        vector_to_array("probs").alias("p"), vector_to_array("votes").alias("v"),
        vector_to_array("leaves").alias("l")).collect()
    assert [r.p for r in rows] == [[1.0,0.0],[0.0,1.0],[0.0,1.0],[1.0,0.0]]
    assert [r.v for r in rows] == [[3.0,0.0],[0.0,3.0],[0.0,3.0],[3.0,0.0]]
    assert all(len(r.l) == 3 for r in rows)
    assert _forest_predictions(model, held_out, "class") == [0.0,1.0,1.0,0.0]
    assert _forest_predictions(estimator.fit(training), held_out, "class") == [0.0,1.0,1.0,0.0]
    del model
    import gc
    gc.collect()
    assert [r["class"] for r in delayed.orderBy("id").collect()] == [0.0,1.0,1.0,0.0]
    assert _forest_predictions(other, held_out, "class") == [1.0,0.0,0.0,1.0]


def test_random_forest_multiclass_probabilities_and_leaf_indices(session):
    from pyspark.ml.connect.functions import vector_to_array
    RandomForestClassifier = _random_forest_classifier()
    training = _forest_vectors(session,
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS label "
        "FROM VALUES (-3,0),(-2,0),(-1,1),(0,1),(1,2),(2,2) AS t(x,y)")
    held_out = _forest_vectors(session,
        "SELECT id, array(CAST(x AS DOUBLE)) AS xs "
        "FROM VALUES (0,-2.5),(1,-0.5),(2,1.5) AS t(id,x)")
    model = RandomForestClassifier(seed=42, numTrees=3, maxDepth=3,
                                   featureSubsetStrategy="all", bootstrap=False,
                                   leafCol="leaves").fit(training)
    assert _forest_predictions(model, held_out) == [0.0,1.0,2.0]
    rows = model.transform(held_out).orderBy("id").select(
        vector_to_array("probability").alias("p"),
        vector_to_array("rawPrediction").alias("v"), vector_to_array("leaves").alias("l")).collect()
    assert [r.p for r in rows] == [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
    assert [r.v for r in rows] == [[3.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,3.0]]
    assert all(len(r.l)==3 for r in rows)


def test_random_forest_weights_and_class_thresholds_change_predictions(session):
    from pyspark.ml.connect.functions import vector_to_array
    RandomForestClassifier = _random_forest_classifier()
    training = _forest_vectors(session,
        "SELECT array(0.0D) AS xs, CAST(y AS DOUBLE) AS label, CAST(w AS DOUBLE) AS weight "
        "FROM VALUES (0,1),(0,1),(0,1),(1,9) AS t(y,w)")
    held_out = _forest_vectors(session, "SELECT 0 AS id, array(0.0D) AS xs")
    options = dict(seed=42, numTrees=3, maxDepth=0, featureSubsetStrategy="all", bootstrap=False)
    plain = RandomForestClassifier(**options).fit(training)
    weighted = RandomForestClassifier(**options, weightCol="weight").fit(training)
    estimator = RandomForestClassifier(**options)
    assert estimator.setThresholds([1.0,0.1]) is estimator
    thresholded = estimator.fit(training)
    assert _forest_predictions(plain, held_out) == [0.0]
    assert _forest_predictions(weighted, held_out) == [1.0]
    assert _forest_predictions(thresholded, held_out) == [1.0]
    for model, expected in [(plain,[0.75,0.25]), (weighted,[0.25,0.75]), (thresholded,[0.75,0.25])]:
        assert model.transform(held_out).select(vector_to_array("probability").alias("p")).first().p == expected


def test_random_forest_bootstrap_sampling_is_reproducible(session):
    RandomForestClassifier = _random_forest_classifier()
    training = _forest_vectors(session,
        "SELECT id, array(CAST(id AS DOUBLE),CAST(id % 3 AS DOUBLE)) AS xs, "
        "CAST(CASE WHEN id < 20 THEN 0 ELSE 1 END AS DOUBLE) AS label FROM range(40)")
    estimator = RandomForestClassifier(seed=2**33, numTrees=7, maxDepth=3,
                                       bootstrap=True, subsamplingRate=0.8,
                                       featureSubsetStrategy="sqrt")
    first = _forest_predictions(estimator.fit(training), training)
    assert first == _forest_predictions(estimator.fit(training), training)
    assert first[:15] == [0.0]*15 and first[25:] == [1.0]*15


def test_random_forest_rejects_invalid_parameters_and_training_data(session):
    RandomForestClassifier = _random_forest_classifier()
    training = _forest_vectors(session,
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS label, 1.0D AS weight "
        "FROM VALUES (-1,0),(1,1) AS t(x,y)")
    for options in [dict(numTrees=0),dict(maxDepth=31),dict(maxBins=1),
                    dict(minInstancesPerNode=0),dict(minInfoGain=-1.0),
                    dict(maxMemoryInMB=-1),dict(checkpointInterval=0),
                    dict(impurity="invalid"),dict(featureSubsetStrategy="invalid"),
                    dict(subsamplingRate=0.0),dict(minWeightFractionPerNode=0.5)]:
        with pytest.raises(Exception):
            RandomForestClassifier(**options).fit(training)
    with pytest.raises(Exception):
        RandomForestClassifier().setThresholds([1.0,1.0,1.0]).fit(training)
    with pytest.raises(Exception):
        RandomForestClassifier().fit(training.drop("label"))
    with pytest.raises(Exception):
        RandomForestClassifier().fit(training.withColumn("label",F.lit(-1.0)))
    with pytest.raises(Exception):
        RandomForestClassifier(weightCol="weight").fit(training.withColumn("weight",F.lit(-1.0)))


def _random_forest_regressor():
    try:
        from pyspark.ml.connect.regression import RandomForestRegressor
    except ImportError:
        from pyspark.ml.regression import RandomForestRegressor
    return RandomForestRegressor


@pytest.mark.parametrize("use_setters", [False, True], ids=["constructor", "setters"])
def test_random_forest_regressor_learns_nonlinear_values_and_retains_lazy_predictions(session, use_setters):
    from pyspark.ml.connect.functions import vector_to_array
    RandomForestRegressor = _random_forest_regressor()
    training = _forest_vectors(session,
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS target, 1.0D AS weight "
        "FROM VALUES (-3,-4.0),(-2,-4.0),(-1,5.5),(0,5.5),(1,5.5),(2,-4.0),(3,-4.0) AS t(x,y)", "input")
    held_out = _forest_vectors(session,
        "SELECT id, array(CAST(x AS DOUBLE)) AS xs "
        "FROM VALUES (0,-2.5),(1,-0.5),(2,0.5),(3,2.5) AS t(id,x)", "input")
    options = dict(featuresCol="input", labelCol="target", predictionCol="value",
                   maxDepth=3, maxBins=16, minInstancesPerNode=1, minInfoGain=0.0,
                   maxMemoryInMB=64, cacheNodeIds=True, checkpointInterval=-1,
                   impurity="variance", subsamplingRate=1.0, seed=42, numTrees=3,
                   featureSubsetStrategy="all", leafCol="leaves", minWeightFractionPerNode=0.0,
                   weightCol="weight", bootstrap=False)
    if use_setters:
        estimator = RandomForestRegressor()
        for name, value in options.items():
            assert getattr(estimator, "set" + name[0].upper() + name[1:])(value) is estimator
    else:
        estimator = RandomForestRegressor(**options)
    first = estimator.fit(training)
    second = estimator.fit(training.withColumn("target", F.col("target") + 10.0))
    expected = [-4.0, 5.5, 5.5, -4.0]
    assert _forest_predictions(first, held_out, "value") == expected
    assert _forest_predictions(second, held_out, "value") == [6.0, 15.5, 15.5, 6.0]
    assert _forest_predictions(estimator.fit(training), held_out, "value") == expected
    leaves = first.transform(held_out).select(vector_to_array("leaves").alias("v")).collect()
    assert all(len(r.v)==3 and all(x>=0 for x in r.v) for r in leaves)
    delayed = first.transform(held_out).select("id", "value")
    del first
    import gc
    gc.collect()
    assert [r.value for r in delayed.orderBy("id").collect()] == expected


def test_random_forest_regressor_instance_weights_change_the_mean(session):
    RandomForestRegressor = _random_forest_regressor()
    training = _forest_vectors(session,
        "SELECT 0 AS id, array(0.0D) AS xs, CAST(y AS DOUBLE) AS label, CAST(w AS DOUBLE) AS weight "
        "FROM VALUES (-2,1),(-2,1),(-2,1),(10,9) AS t(y,w)")
    options = dict(seed=42, numTrees=3, maxDepth=0, featureSubsetStrategy="all", bootstrap=False)
    plain = RandomForestRegressor(**options).fit(training)
    weighted = RandomForestRegressor(**options, weightCol="weight").fit(training)
    assert _forest_predictions(plain, training) == [1.0]*4
    assert _forest_predictions(weighted, training) == [7.0]*4


def test_random_forest_regressor_seeded_bootstrap_and_feature_sampling(session):
    RandomForestRegressor = _random_forest_regressor()
    training = _forest_vectors(session,
        "SELECT id, array(CAST(id AS DOUBLE),CAST(id % 3 AS DOUBLE)) AS xs, "
        "CAST(CASE WHEN id < 20 THEN -3 ELSE 8 END AS DOUBLE) AS label FROM range(40)")
    estimator = RandomForestRegressor(seed=2**33, numTrees=7, maxDepth=3,
                                      bootstrap=True, subsamplingRate=0.8,
                                      featureSubsetStrategy="sqrt")
    first = _forest_predictions(estimator.fit(training), training)
    assert first == _forest_predictions(estimator.fit(training), training)
    assert all(-3.0 <= x <= 8.0 for x in first)
    assert first[0] < first[-1]


def test_random_forest_regressor_rejects_invalid_parameters_and_data(session):
    RandomForestRegressor = _random_forest_regressor()
    training = _forest_vectors(session,
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS label, 1.0D AS weight "
        "FROM VALUES (-1,-2.5),(1,3.5) AS t(x,y)")
    for options in [dict(numTrees=0),dict(maxDepth=31),dict(maxBins=1),
                    dict(minInstancesPerNode=0),dict(minInfoGain=-1.0),dict(maxMemoryInMB=-1),
                    dict(checkpointInterval=0),dict(impurity="gini"),
                    dict(featureSubsetStrategy="invalid"),dict(subsamplingRate=0.0),
                    dict(minWeightFractionPerNode=0.5)]:
        with pytest.raises(Exception):
            RandomForestRegressor(**options).fit(training)
    with pytest.raises(Exception):
        RandomForestRegressor().fit(training.drop("label"))
    with pytest.raises(Exception):
        RandomForestRegressor().fit(training.withColumn("label",F.lit(float("nan"))))
    with pytest.raises(Exception):
        RandomForestRegressor(weightCol="weight").fit(training.withColumn("weight",F.lit(-1.0)))
    with pytest.raises(Exception):
        RandomForestRegressor().fit(training.withColumn("features",F.lit(1.0)))


def _decision_tree_classifier():
    try:
        from pyspark.ml.connect.classification import DecisionTreeClassifier
    except ImportError:
        from pyspark.ml.classification import DecisionTreeClassifier
    return DecisionTreeClassifier

@pytest.mark.parametrize("use_setters", [False, True], ids=["constructor", "setters"])
def test_decision_tree_classifier_learns_nonlinear_labels_and_preserves_lazy_predictions(session, use_setters):
    from pyspark.ml.connect.functions import vector_to_array
    DecisionTreeClassifier = _decision_tree_classifier()
    training = _forest_vectors(session,
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS target, 1.0D AS weight "
        "FROM VALUES (-3,0),(-2,0),(-1,1),(0,1),(1,1),(2,0),(3,0) AS t(x,y)", "input")
    held_out = _forest_vectors(session,
        "SELECT id, array(CAST(x AS DOUBLE)) AS xs "
        "FROM VALUES (0,-2.5),(1,-0.5),(2,0.5),(3,2.5) AS t(id,x)", "input")
    kwargs = dict(featuresCol="input", labelCol="target", predictionCol="class",
                  probabilityCol="probs", rawPredictionCol="votes", maxDepth=3,
                  maxBins=16, minInstancesPerNode=1, minInfoGain=0.0, pruneTree=False,
                  maxMemoryInMB=64, cacheNodeIds=True, checkpointInterval=-1,
                  impurity="entropy", seed=42,
                  leafCol="leaves", minWeightFractionPerNode=0.0,
                  weightCol="weight")
    if use_setters:
        estimator = DecisionTreeClassifier()
        for name, value in kwargs.items():
            assert getattr(estimator, "set" + name[0].upper() + name[1:])(value) is estimator
    else:
        estimator = DecisionTreeClassifier(**kwargs)
    model = estimator.fit(training)
    other = estimator.fit(training.withColumn("target", 1.0 - F.col("target")))
    delayed = model.transform(held_out).select("id", "class")
    rows = model.transform(held_out).orderBy("id").select(
        vector_to_array("probs").alias("p"), vector_to_array("votes").alias("v"),
        F.col("leaves").alias("l")).collect()
    assert [r.p for r in rows] == [[1.0,0.0],[0.0,1.0],[0.0,1.0],[1.0,0.0]]
    assert [r.v for r in rows] == [[2.0,0.0],[0.0,3.0],[0.0,3.0],[2.0,0.0]]
    assert [r.l for r in rows] == [0.0,1.0,1.0,2.0]
    assert _forest_predictions(model, held_out, "class") == [0.0,1.0,1.0,0.0]
    assert _forest_predictions(estimator.fit(training), held_out, "class") == [0.0,1.0,1.0,0.0]
    del model
    import gc
    gc.collect()
    assert [r["class"] for r in delayed.orderBy("id").collect()] == [0.0,1.0,1.0,0.0]
    assert _forest_predictions(other, held_out, "class") == [1.0,0.0,0.0,1.0]


def test_decision_tree_classifier_multiclass_probabilities_and_leaf_indices(session):
    from pyspark.ml.connect.functions import vector_to_array
    DecisionTreeClassifier = _decision_tree_classifier()
    training = _forest_vectors(session,
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS label "
        "FROM VALUES (-3,0),(-2,0),(-1,1),(0,1),(1,2),(2,2) AS t(x,y)")
    held_out = _forest_vectors(session,
        "SELECT id, array(CAST(x AS DOUBLE)) AS xs "
        "FROM VALUES (0,-2.5),(1,-0.5),(2,1.5) AS t(id,x)")
    model = DecisionTreeClassifier(seed=42, maxDepth=3,

                                   leafCol="leaves").fit(training)
    assert _forest_predictions(model, held_out) == [0.0,1.0,2.0]
    assert model.transform(held_out).schema["leaves"].dataType == DoubleType()
    rows = model.transform(held_out).orderBy("id").select(
        vector_to_array("probability").alias("p"),
        vector_to_array("rawPrediction").alias("v"), F.col("leaves").alias("l")).collect()
    assert [r.p for r in rows] == [[1.0,0.0,0.0],[0.0,1.0,0.0],[0.0,0.0,1.0]]
    assert [r.v for r in rows] == [[2.0,0.0,0.0],[0.0,2.0,0.0],[0.0,0.0,2.0]]
    assert [r.l for r in rows] == [0.0,1.0,2.0]


def test_decision_tree_classifier_weights_and_class_thresholds_change_predictions(session):
    from pyspark.ml.connect.functions import vector_to_array
    DecisionTreeClassifier = _decision_tree_classifier()
    training = _forest_vectors(session,
        "SELECT array(0.0D) AS xs, CAST(y AS DOUBLE) AS label, CAST(w AS DOUBLE) AS weight "
        "FROM VALUES (0,1),(0,1),(0,1),(1,9) AS t(y,w)")
    held_out = _forest_vectors(session, "SELECT 0 AS id, array(0.0D) AS xs")
    options = dict(seed=42, maxDepth=0)
    plain = DecisionTreeClassifier(**options).fit(training)
    weighted = DecisionTreeClassifier(**options, weightCol="weight").fit(training)
    estimator = DecisionTreeClassifier(**options)
    assert estimator.setThresholds([1.0,0.1]) is estimator
    thresholded = estimator.fit(training)
    assert _forest_predictions(plain, held_out) == [0.0]
    assert _forest_predictions(weighted, held_out) == [1.0]
    assert _forest_predictions(thresholded, held_out) == [1.0]
    for model, expected, counts in [(plain,[0.75,0.25],[3.0,1.0]),
                                     (weighted,[0.25,0.75],[3.0,9.0]),
                                     (thresholded,[0.75,0.25],[3.0,1.0])]:
        row = model.transform(held_out).select(
            vector_to_array("probability").alias("p"),
            vector_to_array("rawPrediction").alias("raw")).first()
        assert row.p == expected
        assert row.raw == counts


def test_decision_tree_classifier_rejects_invalid_parameters_and_training_data(session):
    DecisionTreeClassifier = _decision_tree_classifier()
    training = _forest_vectors(session,
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS label, 1.0D AS weight "
        "FROM VALUES (-1,0),(1,1) AS t(x,y)")
    for options in [dict(maxDepth=31),dict(maxBins=1),
                    dict(minInstancesPerNode=0),dict(minInfoGain=-1.0),
                    dict(maxMemoryInMB=-1),dict(checkpointInterval=0),
                    dict(impurity="invalid"),
                    dict(minWeightFractionPerNode=0.5)]:
        with pytest.raises(Exception):
            DecisionTreeClassifier(**options).fit(training)
    with pytest.raises(Exception):
        DecisionTreeClassifier().setThresholds([1.0,1.0,1.0]).fit(training)
    with pytest.raises(Exception):
        DecisionTreeClassifier().fit(training.drop("label"))
    with pytest.raises(Exception):
        DecisionTreeClassifier().fit(training.withColumn("label",F.lit(-1.0)))
    with pytest.raises(Exception):
        DecisionTreeClassifier(weightCol="weight").fit(training.withColumn("weight",F.lit(-1.0)))


    with pytest.raises(Exception):
        DecisionTreeClassifier().fit(training.drop("features"))
    # An array of numbers is taken as the vector of its numbers, as PySpark's estimators take
    # arrays; an array of strings is not.
    with pytest.raises(Exception):
        DecisionTreeClassifier().fit(training.withColumn("features",F.array(F.lit("a"))))


def _decision_tree_regressor():
    try:
        from pyspark.ml.connect.regression import DecisionTreeRegressor
    except ImportError:
        from pyspark.ml.regression import DecisionTreeRegressor
    return DecisionTreeRegressor


@pytest.mark.parametrize("use_setters", [False, True], ids=["constructor", "setters"])
def test_decision_tree_regressor_learns_nonlinear_values_and_retains_lazy_predictions(session, use_setters):
    DecisionTreeRegressor = _decision_tree_regressor()
    training = _forest_vectors(session,
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS target, 1.0D AS weight "
        "FROM VALUES (-3,-4.0),(-2,-4.0),(-1,5.5),(0,5.5),(1,5.5),(2,-4.0),(3,-4.0) AS t(x,y)", "input")
    held_out = _forest_vectors(session,
        "SELECT id, array(CAST(x AS DOUBLE)) AS xs "
        "FROM VALUES (0,-2.5),(1,-0.5),(2,0.5),(3,2.5) AS t(id,x)", "input")
    options = dict(featuresCol="input", labelCol="target", predictionCol="value",
                   maxDepth=3, maxBins=16, minInstancesPerNode=1, minInfoGain=0.0,
                   maxMemoryInMB=64, cacheNodeIds=True, checkpointInterval=-1,
                   impurity="variance", seed=42,
                   leafCol="leaves", varianceCol="uncertainty", minWeightFractionPerNode=0.0,
                   weightCol="weight")
    if use_setters:
        estimator = DecisionTreeRegressor()
        for name, value in options.items():
            assert getattr(estimator, "set" + name[0].upper() + name[1:])(value) is estimator
    else:
        estimator = DecisionTreeRegressor(**options)
    first = estimator.fit(training)
    second = estimator.fit(training.withColumn("target", F.col("target") + 10.0))
    expected = [-4.0, 5.5, 5.5, -4.0]
    assert _forest_predictions(first, held_out, "value") == expected
    assert _forest_predictions(second, held_out, "value") == [6.0, 15.5, 15.5, 6.0]
    assert _forest_predictions(estimator.fit(training), held_out, "value") == expected
    output = first.transform(held_out).orderBy("id")
    assert output.schema["leaves"].dataType == DoubleType()
    assert output.schema["uncertainty"].dataType == DoubleType()
    assert [r.leaves for r in output.select("leaves").collect()] == [0.0,1.0,1.0,2.0]
    assert [r.uncertainty for r in output.select("uncertainty").collect()] == [0.0]*4
    delayed = first.transform(held_out).select("id", "value")
    del first
    import gc
    gc.collect()
    assert [r.value for r in delayed.orderBy("id").collect()] == expected


def test_decision_tree_regressor_instance_weights_change_the_mean(session):
    DecisionTreeRegressor = _decision_tree_regressor()
    training = _forest_vectors(session,
        "SELECT 0 AS id, array(0.0D) AS xs, CAST(y AS DOUBLE) AS label, CAST(w AS DOUBLE) AS weight "
        "FROM VALUES (-2,1),(-2,1),(-2,1),(10,9) AS t(y,w)")
    options = dict(seed=42, maxDepth=0)
    plain = DecisionTreeRegressor(**options).fit(training)
    weighted = DecisionTreeRegressor(**options, weightCol="weight").fit(training)
    assert _forest_predictions(plain, training) == [1.0]*4
    assert _forest_predictions(weighted, training) == [7.0]*4
    assert "variance" not in plain.transform(training).columns
    assert "leaf" not in plain.transform(training).columns


def test_decision_tree_regressor_rejects_invalid_parameters_and_data(session):
    DecisionTreeRegressor = _decision_tree_regressor()
    training = _forest_vectors(session,
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS label, 1.0D AS weight "
        "FROM VALUES (-1,-2.5),(1,3.5) AS t(x,y)")
    for options in [dict(maxDepth=31),dict(maxBins=1),
                    dict(minInstancesPerNode=0),dict(minInfoGain=-1.0),dict(maxMemoryInMB=-1),
                    dict(checkpointInterval=0),dict(impurity="gini"),

                    dict(minWeightFractionPerNode=0.5)]:
        with pytest.raises(Exception):
            DecisionTreeRegressor(**options).fit(training)
    with pytest.raises(Exception):
        DecisionTreeRegressor().fit(training.drop("label"))
    with pytest.raises(Exception):
        DecisionTreeRegressor().fit(training.withColumn("label",F.lit(float("nan"))))
    with pytest.raises(Exception):
        DecisionTreeRegressor(weightCol="weight").fit(training.withColumn("weight",F.lit(-1.0)))
    with pytest.raises(Exception):
        DecisionTreeRegressor().fit(training.withColumn("features",F.lit(1.0)))




def test_decision_tree_regressor_reports_weighted_variance_in_each_leaf(session):
    DecisionTreeRegressor = _decision_tree_regressor()
    training = _forest_vectors(session,
        "SELECT array(CAST(x AS DOUBLE)) AS xs, CAST(y AS DOUBLE) AS label, "
        "CAST(w AS DOUBLE) AS weight "
        "FROM VALUES (-2,-2,1),(-1,0,3),(1,4,1),(2,8,3) AS t(x,y,w)")
    held_out = _forest_vectors(session,
        "SELECT id, array(CAST(x AS DOUBLE)) AS xs "
        "FROM VALUES (0,-1.5),(1,1.5) AS t(id,x)")
    estimator = DecisionTreeRegressor(seed=2**33, maxDepth=1, weightCol="weight",
                                      varianceCol="uncertainty", leafCol="leaf")
    first = estimator.fit(training)
    rows = first.transform(held_out).orderBy("id").select("prediction", "uncertainty", "leaf").collect()
    assert [r.prediction for r in rows] == [-0.5,7.0]
    assert [r.uncertainty for r in rows] == [0.75,3.0]
    assert [r.leaf for r in rows] == [0.0,1.0]
    assert _forest_predictions(estimator.fit(training), held_out) == [-0.5,7.0]


def test_binary_subclasses_are_sent_as_the_bytes_of_pyspark(session):
    class Bytes(bytes):
        def __bytes__(self):
            return b"other"

    class ByteArray(bytearray):
        def __bytes__(self):
            return b"other"

    rows = session.createDataFrame(
        [(Bytes(b"orig"),), (ByteArray(b"orig"),), (bytearray(b"ba"),), (None,)], "b binary"
    ).collect()
    assert [row.b for row in rows] == [b"other", b"other", b"ba", None]
    rows = session.createDataFrame([([Bytes(b"orig")],)], "a array<binary>").collect()
    assert rows[0].a == [b"other"]


def test_null_get_item_reads_its_nulls_though_the_server_declares_it_non_nullable(session):
    # As official PySpark 4.3.0: `arr.getItem(None)` is declared non-nullable by the server while
    # its values are null; the rows, the iterator, the struct field and the Arrow table keep them.
    arr = F.array(F.lit(10), F.lit(20))
    df = session.range(3).select(arr.getItem(None).alias("v"))
    assert df.schema["v"].nullable is False
    assert [row.v for row in df.collect()] == [None, None, None]
    assert [row.v for row in df.toLocalIterator()] == [None, None, None]
    rows = session.range(2).select(F.struct(arr.getItem(None).alias("x")).alias("s")).collect()
    assert [row.s.x for row in rows] == [None, None]
    # The table is cast to the non-nullable schema, which raises for a null, as PySpark's does.
    with pytest.raises(ValueError, match="Casting field 'v' with null values to non-nullable"):
        df.toArrow()
