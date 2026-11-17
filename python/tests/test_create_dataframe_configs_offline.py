#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements.  See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License.  You may obtain a copy of the License at
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
The settings that createDataFrame reads, as PySpark 4.3.0's: one Config get of the settings of all
of its conversions, in its order, before it converts any data that is not empty, whatever the kind
of the data, after the AnalyzePlan of a DDL schema; and no request for empty data. Each case
records the requests that the server receives for one createDataFrame, which is lazy.
"""

import datetime

import pyspark.sql.connect.proto as pb2
import pytest
from _mock_connect_server import mock_connect_session
from pyspark.sql import Row
from pyspark.sql.types import LongType, StringType, StructField, StructType

# The keys of the Config get of PySpark 4.3.0's createDataFrame, in its order.
_KEYS = [
    "spark.sql.timestampType",
    "spark.sql.session.timeZone",
    "spark.sql.session.localRelationCacheThreshold",
    "spark.sql.session.localRelationSizeLimit",
    "spark.sql.session.localRelationChunkSizeRows",
    "spark.sql.session.localRelationChunkSizeBytes",
    "spark.sql.session.localRelationBatchOfChunksSizeBytes",
    "spark.sql.execution.pandas.convertToArrowArraySafely",
    "spark.sql.execution.pandas.inferPandasDictAsMap",
    "spark.sql.pyspark.inferNestedDictAsStruct.enabled",
    "spark.sql.pyspark.legacy.inferArrayTypeFromFirstElement.enabled",
    "spark.sql.pyspark.legacy.inferMapTypeFromFirstPair.enabled",
    "spark.sql.execution.arrow.useLargeVarTypes",
]
_SCHEMA = StructType([StructField("a", LongType()), StructField("b", StringType())])


@pytest.fixture(scope="module")
def _mock():
    pytest.importorskip("grpc")
    with mock_connect_session() as session:
        yield session


@pytest.fixture
def mock(_mock):
    spark, service = _mock
    service.reset()
    return spark, service


def _requests(service):
    # The recorded requests, in order: ("get", keys) of a Config get, and the kind of any other.
    requests = []
    for kind, data in service.recorded_order:
        if kind == "config":
            operation = pb2.ConfigRequest.FromString(data).operation
            if operation.HasField("get"):
                requests.append(("get", list(operation.get.keys)))
                continue
        requests.append((kind, None))
    return requests


def _pandas_frame():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})


def _naive_timestamp_frame():
    pd = pytest.importorskip("pandas")
    return pd.DataFrame({"t": [datetime.datetime(2020, 1, 1)]})


def _numpy_array():
    np = pytest.importorskip("numpy")
    return np.array([[1, 2], [3, 4]])


def _arrow_table():
    pa = pytest.importorskip("pyarrow")
    return pa.table({"a": [1, 2]})


_CASES = {
    "tuples and names": lambda: ([(1, "x"), (2, "y")], ["a", "b"]),
    "tuples and a struct": lambda: ([(1, "x")], _SCHEMA),
    "rows inferred": lambda: ([Row(a=1, b="x")], None),
    "dicts inferred": lambda: ([{"a": 1, "b": {"k": 1}}], None),
    "datetimes inferred": lambda: ([(datetime.datetime(2020, 1, 1),)], ["t"]),
    "single values": lambda: ([1, 2], LongType()),
    "a generator": lambda: ((row for row in [(1,), (2,)]), ["a"]),
    "a pandas DataFrame": lambda: (_pandas_frame(), None),
    "a pandas DataFrame of naive timestamps": lambda: (_naive_timestamp_frame(), None),
    "a numpy array": lambda: (_numpy_array(), None),
    "a pyarrow Table": lambda: (_arrow_table(), None),
}


@pytest.mark.parametrize("name", list(_CASES))
def test_create_dataframe_reads_pysparks_settings_in_one_request(mock, name):
    # One Config get of every setting of the conversions, of PySpark's keys in its order, and no
    # other request, whatever the data: the settings that the data does not need are read too.
    spark, service = mock
    data, schema = _CASES[name]()
    spark.createDataFrame(data, schema)
    assert _requests(service) == [("get", _KEYS)]


def test_create_dataframe_parses_a_ddl_schema_before_it_reads_the_settings(mock):
    # As PySpark's: the AnalyzePlan of the DDL string first, then the settings.
    spark, service = mock
    spark.createDataFrame([(1, "x")], "a long, b string")
    assert _requests(service) == [("analyze", None), ("get", _KEYS)]


def test_create_dataframe_of_empty_data_reads_no_setting(mock):
    # Empty data of a schema is a relation of the schema, and of no schema an error, both before
    # the settings are read, as in PySpark.
    spark, service = mock
    spark.createDataFrame([], _SCHEMA)
    with pytest.raises(Exception, match="CANNOT_INFER_EMPTY_SCHEMA"):
        spark.createDataFrame([])
    assert _requests(service) == []


def test_create_dataframe_reads_the_settings_before_it_checks_the_columns_of_an_array(mock):
    # As PySpark's, whose conversion of the array, after the settings, raises AXIS_LENGTH_MISMATCH.
    np = pytest.importorskip("numpy")
    spark, service = mock
    with pytest.raises(Exception, match="AXIS_LENGTH_MISMATCH"):
        spark.createDataFrame(np.array([[1, 2]]), ["a"])
    assert _requests(service) == [("get", _KEYS)]


def test_create_dataframe_reads_the_settings_before_it_converts_the_rows(mock):
    # A row that the conversion rejects fails after the settings are read, as in PySpark.
    spark, service = mock
    with pytest.raises(Exception):
        spark.createDataFrame([("x",)], StructType([StructField("a", LongType())]))
    assert _requests(service) == [("get", _KEYS)]
