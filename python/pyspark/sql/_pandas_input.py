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
    import sys
    from types import GetSetDescriptorType, WrapperDescriptorType

    import numpy as np
    import pandas as pd
    try:
        from pandas.core.arrays.datetimes import DatetimeArray
        from pandas.core.dtypes.dtypes import DatetimeTZDtype
        from pandas.core.frame import DataFrame
        from pandas.core.series import Series
    except ImportError:
        return None
    from pyspark._pyspark import (
        BooleanType, ByteType, ShortType, IntegerType, LongType, FloatType,
        DoubleType, DecimalType, StringType, BinaryType, DateType, TimestampType,
        StructField, StructType as NativeStructType,
    )
    from pyspark.sql import types as sql_types

    if (pd.DataFrame is not DataFrame or type(frame) is not DataFrame
            or type(schema) is not NativeStructType or not len(frame)):
        return None
    # Native row conversion reads the native schema directly. The Arrow importer
    # also consults Python descriptors and public or captured converter aliases.
    if sys.modules.get("pyspark.sql.types") is not sql_types:
        return None
    importer = sys.modules.get("pyspark.sql._arrow_input")
    if importer is not None and getattr(importer, "StructType", None) is not NativeStructType:
        return None
    converter = sys.modules.get("pyspark.sql.pandas.types")
    namespaces = [vars(sql_types)]
    if converter is not None:
        namespaces.append(vars(converter))
    if importer is not None:
        for name in ("from_arrow_schema", "to_arrow_schema",
                     "_check_arrow_table_timestamps_localize"):
            # Newer importers resolve converters locally within prepare_table.
            if name not in vars(importer):
                continue
            namespace = getattr(getattr(importer, name, None), "__globals__", None)
            if type(namespace) is not dict:
                return None
            namespaces.append(namespace)
    for name, cls in (
        ("BooleanType", BooleanType), ("ByteType", ByteType), ("ShortType", ShortType),
        ("IntegerType", IntegerType), ("LongType", LongType), ("FloatType", FloatType),
        ("DoubleType", DoubleType), ("DecimalType", DecimalType), ("StringType", StringType),
        ("BinaryType", BinaryType), ("DateType", DateType), ("TimestampType", TimestampType),
        ("StructField", StructField), ("StructType", NativeStructType),
    ):
        if any(namespace.get(name) is not cls for namespace in namespaces):
            return None
    for cls in (TimestampType, StructField, NativeStructType):
        if (cls.__getattribute__ is not object.__getattribute__
                or any("__class__" in vars(base) for base in cls.__mro__ if base is not object)):
            return None
    for cls, names, descriptor_type in (
        (NativeStructType, ("fields", "names"), GetSetDescriptorType),
        (NativeStructType, ("__iter__",), WrapperDescriptorType),
        (StructField, ("name", "dataType", "nullable", "metadata"), GetSetDescriptorType),
    ):
        for name in names:
            descriptor = vars(cls).get(name)
            if (type(descriptor) is not descriptor_type
                    or descriptor.__objclass__ is not cls or descriptor.__name__ != name):
                return None

    width = len(frame.columns)
    fields = schema.fields
    if not width or len(fields) != width or not frame.columns.is_unique:
        return None
    if not all(type(name) is str and "\x00" not in name for name in frame.columns):
        return None
    if not all(type(field) is StructField and type(field.dataType) is TimestampType
               and type(field.name) is str and "\x00" not in field.name for field in fields):
        return None
    if len(set(schema.names)) != width:
        return None
    columns = []
    datetime_dtype = type(np.dtype("datetime64[ns]"))
    for i, dtype in enumerate(frame.dtypes):
        if type(dtype) is DatetimeTZDtype:
            unit = dtype.unit
        elif type(dtype) is datetime_dtype and dtype.kind == "M":
            unit = np.datetime_data(dtype)[0]
        else:
            return None
        if unit not in ("ns", "us"):
            return None
        column = frame.iloc[:, i]
        if type(column) is not Series or type(column.array) is not DatetimeArray:
            return None
        # Preserve the row path's local null error before any timezone Config RPC.
        if not fields[i].nullable and column.hasnans:
            return None
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
    from pandas.core.dtypes.dtypes import DatetimeTZDtype
    import pyarrow as pa

    if any(type(column.dtype) is not DatetimeTZDtype for column in columns):
        if (timezone != "UTC" or os.environ.get("TZ") != "UTC"
                or time.tzname != ("UTC", "UTC")):
            return None
    return pa.Table.from_arrays(
        [pa.array(column, type=pa.timestamp("us", tz="UTC"), from_pandas=True)
         for column in columns],
        names=schema.names,
    )
