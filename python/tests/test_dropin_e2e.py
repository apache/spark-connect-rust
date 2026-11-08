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

import pyarrow as pa
import pytest
from pyspark.errors import AnalysisException, PySparkValueError
from pyspark.sql import Row, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import (
    ArrayType,
    DateType,
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# Keep interval round trips in the live suite selected by CI.
from interval_input_cases import IntervalInputLiveTests  # noqa: F401


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


def test_ntz_collect_is_the_wall_clock_in_the_client_time_zone_as_pyspark():
    # As PySpark's Arrow conversion: a TIMESTAMP_NTZ is the wall clock, whatever the time zone of
    # the client, from year 1 to year 9999 and inside arrays, maps and structs, while a TIMESTAMP
    # is the instant, shown in the local time zone (here seven hours behind UTC, in July).
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("SPARK_REMOTE is required for live-server tests")
    out = _run_in_time_zone("""
import datetime
from pyspark.sql import SparkSession
spark = SparkSession.builder.remote(%r).getOrCreate()
print(spark.createDataFrame([(datetime.datetime(2020, 7, 1, 12, 0),)], "t timestamp_ntz").collect())
print(spark.sql("select timestamp_ntz'2020-07-01 12:00:00' t").collect())
print(spark.sql("select timestamp_ntz'0001-01-01 00:00:00' t, timestamp_ntz'9999-12-31 23:59:59.999999' u").collect())
print(spark.sql(
    "select array(timestamp_ntz'2020-07-01 12:00:00') a, map('k', timestamp_ntz'2020-07-01 12:00:00') m, "
    "named_struct('x', timestamp_ntz'2020-07-01 12:00:00') s").collect())
print(spark.sql("select timestamp'2020-07-01 12:00:00' t, timestamp_micros(9223372036854775) u").collect())
""" % remote, "America/Los_Angeles")
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
    import threading
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
    from typing import Iterator, Tuple

    import numpy as np
    import pandas as pd

    def scaled(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        for frame in batches:
            yield frame.assign(v=frame.v * 10)

    def shifted(key: Tuple[np.int64], batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        for frame in batches:
            yield frame.assign(v=frame.v + key[0])

    def batches(it: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        yield from it

    def keyed_batches(
        key: Tuple[pa.Scalar, ...], it: Iterator[pa.RecordBatch]
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
    try:
        from pyspark._pyspark import LogisticRegression
    except ImportError:
        from pyspark.ml.classification import LogisticRegression
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


@pytest.mark.parametrize("use_setters", [False, True], ids=["constructor", "setters"])
def test_linear_svc_fits_the_labels_and_predicts_on_new_rows(session, use_setters):
    try:
        from pyspark._pyspark import LinearSVC
    except ImportError:
        from pyspark.ml.classification import LinearSVC
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
