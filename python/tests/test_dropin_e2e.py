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


def test_conf_get_with_a_none_default(session):
    key = "spark.connect.rust.test.unset"
    assert session.conf.get(key, None) is None
    assert session.conf.get(key, "x") == "x"
    with pytest.raises(Exception, match="SQL_CONF_NOT_FOUND"):
        session.conf.get(key)


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


def test_collect_reads_a_result_across_stream_ends(session):
    # Takes several response streams when the server's senderMaxStreamSize is small.
    n = 300_000
    assert len(session.range(n).collect()) == n
    assert session.range(n).toArrow().num_rows == n


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


def test_sql_fails_at_a_statement_that_does_not_parse(session):
    with pytest.raises(AnalysisException, match="PARSE_SYNTAX_ERROR"):
        session.sql("SELEC 1")


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


def test_column_alias_names_every_column_and_keeps_its_metadata(session):
    df = session.sql("SELECT map('k1', 1) AS m").select(F.explode("m").alias("k", "v"))
    assert df.columns == ["k", "v"]
    assert [tuple(r) for r in df.collect()] == [("k1", 1)]
    df = session.range(1).select(F.col("id").alias("x", metadata={"k": 1, "n": [1, 2], "s": "v"}))
    assert df.schema.fields[0].metadata == {"k": 1, "n": [1, 2], "s": "v"}


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


def test_ml_vector_functions_as_pyspark(session):
    from pyspark.ml.functions import array_to_vector, vector_to_array

    df = session.sql("SELECT array(1.5D, 2.0D) AS a").select(array_to_vector("a").alias("vec"))
    row = df.select(vector_to_array("vec").alias("d"), vector_to_array("vec", "float32").alias("f"))
    assert tuple(row.first()) == ([1.5, 2.0], [1.5, 2.0])


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


def test_create_dataframe_converts_numpy_scalars_as_pyspark(session):
    np = pytest.importorskip("numpy")
    df = session.createDataFrame(
        [(np.int64(1), np.float32(0.5), np.bool_(True), np.float32(1.5))],
        "a bigint, b double, c boolean, d bigint")
    assert [tuple(r) for r in df.collect()] == [(1, 0.5, True, 1)]


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
