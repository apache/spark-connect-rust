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


def test_collect_reads_a_result_across_stream_ends(session):
    # Takes several response streams when the server's senderMaxStreamSize is small.
    n = 300_000
    assert len(session.range(n).collect()) == n
    assert session.range(n).toArrow().num_rows == n


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


def test_sql_fails_at_a_statement_that_does_not_parse(session):
    with pytest.raises(AnalysisException, match="PARSE_SYNTAX_ERROR"):
        session.sql("SELEC 1")
