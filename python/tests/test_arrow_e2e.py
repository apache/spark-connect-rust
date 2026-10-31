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
"""Native Arrow input and output checks against a live Spark Connect server."""

import os
from datetime import date, datetime, timezone

import pyarrow as pa
import pytest
from pyspark.sql import SparkSession


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


def test_to_arrow_nested_null_binary_and_temporal_values(session):
    table = session.sql("""
        SELECT array(1, CAST(NULL AS INT), 3) AS items,
               named_struct('name', 'spark', 'value', CAST(NULL AS BIGINT)) AS nested,
               X'00FF' AS binary_value, DATE '2020-01-02' AS day,
               TIMESTAMP '2020-01-02 03:04:05+00:00' AS instant
    """).toArrow()
    assert isinstance(table, pa.Table)
    values = table.to_pylist()
    assert len(values) == 1
    assert values[0] == {
        "items": [1, None, 3],
        "nested": {"name": "spark", "value": None},
        "binary_value": b"\x00\xff",
        "day": date(2020, 1, 2),
        "instant": datetime(2020, 1, 2, 3, 4, 5, tzinfo=timezone.utc),
    }


def test_to_arrow_numeric_range(session):
    table = session.range(10000).toArrow()
    assert table.num_rows == 10000
    assert table.column("id").to_pylist() == list(range(10000))


def test_to_arrow_empty_result(session):
    table = session.range(0).toArrow()
    assert isinstance(table, pa.Table)
    assert table.num_rows == 0


def test_create_dataframe_arrow_cached_upload(session):
    """Materialize cached Arrow input larger than 1 MiB on a real server."""
    import numpy as np

    rows = 262144
    table = pa.table({"id": np.arange(rows, dtype=np.int64)})
    assert table.nbytes == 2 * 1024**2
    isolated = session.newSession()
    try:
        isolated.conf.set("spark.sql.session.localRelationCacheThreshold", str(1024**2))
        frame = isolated.createDataFrame(table)
        actual = frame.selectExpr("count(*) AS rows", "sum(id) AS checksum").collect()[0]
        assert actual[0] == rows
        assert actual[1] == rows * (rows - 1) // 2
        assert frame.isLocal()
    finally:
        isolated.stop()


@pytest.mark.parametrize("cached", [False, True], ids=["inline", "cached"])
@pytest.mark.parametrize("timestamp_type", ["TIMESTAMP_LTZ", "TIMESTAMP_NTZ"])
def test_create_dataframe_arrow_inferred_timestamp(session, cached, timestamp_type):
    value = datetime(2020, 1, 2, 3, 4, 5)
    table = pa.table({"t": pa.array([value], type=pa.timestamp("ns"))})
    isolated = session.newSession()
    try:
        isolated.conf.set("spark.sql.session.timeZone", "America/Los_Angeles")
        isolated.conf.set("spark.sql.timestampType", timestamp_type)
        isolated.conf.set("spark.sql.session.localRelationCacheThreshold",
                          "0" if cached else str(1024**2))
        frame = isolated.createDataFrame(table)
        spark_type = "timestamp" if timestamp_type == "TIMESTAMP_LTZ" else "timestamp_ntz"
        assert frame.schema.simpleString() == f"struct<t:{spark_type}>"
        assert frame.selectExpr("CAST(t AS STRING)").collect()[0][0] == "2020-01-02 03:04:05"

        result = frame.toArrow()
        if timestamp_type == "TIMESTAMP_LTZ":
            assert result.column("t").type.unit == "us"
            assert result.column("t").type.tz is not None
            # The native Arrow result carries the session timezone; compare instants.
            assert result.column("t").cast(pa.timestamp("us", tz="UTC")).to_pylist() == [
                datetime(2020, 1, 2, 11, 4, 5, tzinfo=timezone.utc)
            ]
        else:
            assert result.column("t").type == pa.timestamp("us")
            assert result.column("t").to_pylist() == [value]
    finally:
        isolated.stop()
