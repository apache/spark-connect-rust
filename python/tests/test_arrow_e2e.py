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
    """Exercise real-server config values and cached input above the default 1 MiB."""
    import numpy as np

    rows = 262144
    table = pa.table({"id": np.arange(rows, dtype=np.int64)})
    assert table.nbytes == 2 * 1024**2
    # Clear inherited session overrides so CI exercises registered server defaults.
    session.conf.unset("spark.sql.session.localRelationCacheThreshold")
    session.conf.unset("spark.sql.session.localRelationSizeLimit")
    threshold = int(session.conf.get("spark.sql.session.localRelationCacheThreshold"))
    assert threshold == 1024**2
    assert table.nbytes >= threshold
    # Spark 4.2 declares this setting as longConf and returns decimal bytes.
    assert session.conf.get("spark.sql.session.localRelationSizeLimit") == "3221225472"

    frame = session.createDataFrame(table)
    actual = frame.selectExpr("count(*) AS rows", "sum(id) AS checksum").collect()[0]
    assert actual[0] == rows
    assert actual[1] == rows * (rows - 1) // 2
    assert frame.isLocal()


@pytest.mark.parametrize("schema,expected_width", [
    ("x long", "1"),
    ("x long, y long, z long", "3"),
    ("long", "1"),
])
def test_create_dataframe_arrow_explicit_schema_width(session, schema, expected_width):
    from pyspark.errors import PySparkValueError

    with pytest.raises(PySparkValueError) as error:
        session.createDataFrame(pa.table({"x": [1], "y": [2]}), schema)
    assert error.value.getCondition() == "AXIS_LENGTH_MISMATCH"
    assert error.value.getMessageParameters() == {
        "expected_length": expected_width, "actual_length": "2",
    }
