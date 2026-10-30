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
from datetime import date, datetime

import pyarrow as pa
import pytest
from pyspark.errors import PySparkValueError
from pyspark.sql import SparkSession
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


def test_collect_reads_a_result_across_stream_ends(session):
    # Takes several response streams when the server's senderMaxStreamSize is small.
    n = 300_000
    assert len(session.range(n).collect()) == n
    assert session.range(n).toArrow().num_rows == n
