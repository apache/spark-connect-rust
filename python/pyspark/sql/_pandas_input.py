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


def prepare_date_table(frame, schema):
    """Accept only explicit date columns whose cells are exact dates or None.

    Inspect every object before Arrow conversion: date subclasses can override
    conversion hooks, and pandas treats several other objects as missing values.
    Those cases must keep the existing row conversion and its error ordering.
    """
    from datetime import date
    import sys

    import numpy as np
    import pandas as pd
    from pandas.core.frame import DataFrame
    from pandas.core.series import Series
    from types import GetSetDescriptorType, WrapperDescriptorType

    from pyspark._pyspark import (
        BooleanType, ByteType, ShortType, IntegerType, LongType, FloatType,
        DoubleType, DecimalType, StringType, BinaryType, DateType, StructField,
        StructType as NativeStructType,
    )
    from pyspark.sql import types as sql_types

    # Keep smaller frames on the row path to avoid the importer's config RPC overhead.
    if (pd.DataFrame is not DataFrame or type(frame) is not DataFrame
            or type(schema) is not NativeStructType
            or len(frame) < 10000):
        return None
    # Earlier converter branches must not recognize a date as another type.
    # Check loaded converters and globals captured by older importer functions.
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
        for name in ("from_arrow_schema", "to_arrow_schema"):
            # Newer importers resolve these functions locally in prepare_table.
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
        ("BinaryType", BinaryType), ("DateType", DateType), ("StructField", StructField),
        ("StructType", NativeStructType),
    ):
        if any(namespace.get(name) is not cls for namespace in namespaces):
            return None
    # The row path reads native fields directly. Class attributes are mutable,
    # so customized Python schema access must not select a different Arrow path.
    for name, cls in (("DateType", DateType), ("StructField", StructField),
                      ("StructType", NativeStructType)):
        if (getattr(sql_types, name, None) is not cls
                or cls.__getattribute__ is not object.__getattribute__
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
    if not width or len(schema.fields) != width or not frame.columns.is_unique:
        return None
    if not all(type(name) is str and "\x00" not in name for name in frame.columns):
        return None
    if not all(
        type(field) is StructField and type(field.dataType) is DateType
        and type(field.name) is str and "\x00" not in field.name
        for field in schema.fields
    ) or len(set(schema.names)) != width:
        return None
    if not all(type(dtype) is type(np.dtype("object")) and dtype == np.dtype("object")
               for dtype in frame.dtypes):
        return None
    # A custom ExtensionArray can report an object dtype while its Arrow hook
    # emits different cells. Trust only pandas' concrete NumPy backing array.
    try:
        from pandas.core.arrays.numpy_ import NumpyExtensionArray
    except ImportError:
        try:
            from pandas.core.arrays.numpy_ import PandasArray as NumpyExtensionArray
        except ImportError:
            return None
    columns = [frame.iloc[:, i] for i in range(width)]
    for column, field in zip(columns, schema.fields):
        if type(column) is not Series or type(column.array) is not NumpyExtensionArray:
            return None
        # Arrow rejects nulls in non-nullable fields; let the row path retain
        # its existing behavior for that input instead.
        if not all(type(value) is date or (value is None and field.nullable)
                   for value in column.to_numpy(copy=False)):
            return None

    import pyarrow as pa

    return pa.Table.from_arrays(
        [pa.array(column, type=pa.date32(), from_pandas=False) for column in columns],
        names=schema.names,
    )
