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


"""Conservative columnar pandas input; unsupported conversions use the row path."""

from pyspark.sql.types import LongType, StructType


def prepare_int64_table(frame, schema):
    """Return an Arrow table only when each column is already a signed 64-bit integer.

    In particular, never coerce a mixed pandas frame to a common NumPy dtype:
    doing so can round integers above 2**53. Other dtypes and schema conversions
    deliberately retain the existing row path until their semantics are covered.
    ``verifySchema`` is ignored for pandas Arrow input, as in Spark Connect.
    """
    import numpy as np
    import pandas as pd

    width = len(frame.columns)
    if not width or not frame.columns.is_unique or not all(
        isinstance(name, str) and "\x00" not in name for name in frame.columns
    ):
        return None
    dtypes = frame.dtypes
    if not all(dtype == np.dtype("int64") or isinstance(dtype, pd.Int64Dtype)
               for dtype in dtypes):
        return None
    # The importer's configuration RPCs outweigh row conversion for small NumPy
    # frames. Nullable extension columns still use Arrow for reliable null typing.
    if len(frame) < 1000 and not any(isinstance(dtype, pd.Int64Dtype) for dtype in dtypes):
        return None
    if isinstance(schema, StructType):
        if len(schema.fields) != width or not all(
            isinstance(field.dataType, LongType) and "\x00" not in field.name
            for field in schema.fields
        ):
            return None
    elif schema is not None:
        if not isinstance(schema, (list, tuple)) or len(schema) != width or not all(
            isinstance(name, str) and "\x00" not in name for name in schema
        ):
            return None

    import pyarrow as pa

    # Arrow consumes each Series' integer buffer and nullable mask independently.
    return pa.Table.from_arrays(
        [pa.array(frame.iloc[:, i], type=pa.int64(), from_pandas=True) for i in range(width)],
        names=list(frame.columns),
    )


def timestamp_columns(frame, schema):
    """Select only explicit, microsecond-aligned datetime64 timestamp columns.

    Inference (including TimestampNTZ preferences), object/mixed columns, other
    units and submicrosecond truncation retain the row path. Check the integer
    buffers separately so NaT's sentinel never participates in alignment tests.
    """
    import numpy as np
    import pandas as pd
    from pyspark.sql.types import TimestampType

    if not isinstance(schema, StructType) or not len(frame) or not len(frame.columns):
        return None
    if len(schema.fields) != len(frame.columns) or not frame.columns.is_unique:
        return None
    if not all(isinstance(name, str) and "\x00" not in name for name in frame.columns):
        return None
    if not all(type(field.dataType) is TimestampType and "\x00" not in field.name
               for field in schema.fields):
        return None
    columns = []
    for i, dtype in enumerate(frame.dtypes):
        if isinstance(dtype, pd.DatetimeTZDtype):
            unit = dtype.unit
        elif isinstance(dtype, np.dtype) and dtype.kind == "M":
            unit = np.datetime_data(dtype)[0]
        else:
            return None
        if unit not in ("ns", "us"):
            return None
        column = frame.iloc[:, i]
        if unit == "ns" and np.any(column.array.asi8[column.notna()] % 1000):
            return None
        columns.append(column)
    return columns


def prepare_timestamp_table(columns, schema, timezone):
    """Preserve instants without row materialization or a host-zone conversion.

    Aware columns carry their own timezone. Naive columns are deliberately
    restricted to UTC host/session pairs: the existing row path uses the host
    zone, unlike Python Connect's session-zone conversion (and DST fold policy).
    """
    import os
    import time
    import pandas as pd
    import pyarrow as pa

    if any(not isinstance(column.dtype, pd.DatetimeTZDtype) for column in columns):
        if (timezone != "UTC" or os.environ.get("TZ") != "UTC"
                or time.tzname != ("UTC", "UTC")):
            return None
    return pa.Table.from_arrays(
        [pa.array(column, type=pa.timestamp("us", tz="UTC"), from_pandas=True)
         for column in columns],
        names=schema.names,
    )
