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


def prepare_int64_table(frame, schema):
    """Return an Arrow table only when each column is already a signed 64-bit integer.

    In particular, never coerce a mixed pandas frame to a common NumPy dtype:
    doing so can round integers above 2**53. Other dtypes and schema conversions
    deliberately retain the existing row path until their semantics are covered.
    ``verifySchema`` is ignored for pandas Arrow input, as in Spark Connect.
    """
    import sys
    from types import GetSetDescriptorType, WrapperDescriptorType

    import numpy as np
    import pandas as pd
    try:
        from pandas.core.arrays.integer import Int64Dtype, IntegerArray
        from pandas.core.arrays.numpy_ import NumpyExtensionArray
        from pandas.core.frame import DataFrame
        from pandas.core.series import Series
    except ImportError:
        # Older supported pandas versions call the NumPy wrapper PandasArray.
        # Keep their existing row conversion when these concrete types are absent.
        return None

    from pyspark._pyspark import (
        BooleanType, ByteType, ShortType, IntegerType, LongType, StructField, StructType,
    )
    from pyspark.sql import types as sql_types

    # A subclass or custom extension array can change the values exposed to Arrow.
    # Check concrete identities before consulting dtype equality or conversion hooks.
    if pd.DataFrame is not DataFrame or type(frame) is not DataFrame:
        return None

    width = len(frame.columns)
    if not width or not frame.columns.is_unique or not all(
        type(name) is str and "\x00" not in name for name in frame.columns
    ):
        return None
    dtypes = frame.dtypes
    numpy_dtype = np.dtype("int64")
    if not all((type(dtype) is type(numpy_dtype) and dtype == numpy_dtype)
               or type(dtype) is Int64Dtype
               for dtype in dtypes):
        return None
    # The importer's configuration RPCs outweigh row conversion for small NumPy
    # frames. Nullable extension columns still use Arrow for reliable null typing.
    if len(frame) < 1000 and not any(type(dtype) is Int64Dtype for dtype in dtypes):
        return None

    # The native row path reads native schema fields directly. The Arrow converter
    # instead uses Python descriptors and imported class aliases, including aliases
    # captured before this helper was loaded. Customized schemas retain the row path.
    if sys.modules.get("pyspark.sql.types") is not sql_types:
        return None
    importer = sys.modules.get("pyspark.sql._arrow_input")
    if importer is not None and getattr(importer, "StructType", None) is not StructType:
        return None
    converter = sys.modules.get("pyspark.sql.pandas.types")
    namespaces = [vars(sql_types)]
    if converter is not None:
        namespaces.append(vars(converter))
    if importer is not None:
        # An importer may still hold functions from a removed converter module.
        for name in ("from_arrow_schema", "to_arrow_schema"):
            # Later importers resolve these converters locally inside prepare_table.
            if name not in vars(importer):
                continue
            namespace = getattr(getattr(importer, name, None), "__globals__", None)
            if type(namespace) is not dict:
                return None
            namespaces.append(namespace)
    for name, cls in (
        ("BooleanType", BooleanType), ("ByteType", ByteType), ("ShortType", ShortType),
        ("IntegerType", IntegerType), ("LongType", LongType),
        ("StructField", StructField), ("StructType", StructType),
    ):
        if any(namespace.get(name) is not cls for namespace in namespaces):
            return None
    for cls in (LongType, StructField, StructType):
        if (cls.__getattribute__ is not object.__getattribute__
                or any("__class__" in vars(base) for base in cls.__mro__ if base is not object)):
            return None
    for cls, names, descriptor_type in (
        (StructType, ("fields", "names"), GetSetDescriptorType),
        (StructType, ("__iter__",), WrapperDescriptorType),
        (StructField, ("name", "dataType", "nullable", "metadata"), GetSetDescriptorType),
    ):
        for name in names:
            descriptor = vars(cls).get(name)
            if (type(descriptor) is not descriptor_type
                    or descriptor.__objclass__ is not cls or descriptor.__name__ != name):
                return None

    names = list(frame.columns)
    fields = None
    if type(schema) is StructType:
        fields = schema.fields
        if len(fields) != width or not all(
            type(field) is StructField and type(field.dataType) is LongType
            and type(field.name) is str and "\x00" not in field.name
            for field in fields
        ):
            return None
        names = schema.names
        if len(set(names)) != width:
            return None
    elif schema is not None:
        if type(schema) not in (list, tuple) or len(schema) != width or not all(
            type(name) is str and "\x00" not in name for name in schema
        ):
            return None

    columns = [frame.iloc[:, i] for i in range(width)]
    for i, (column, dtype) in enumerate(zip(columns, dtypes)):
        if type(column) is not Series:
            return None
        expected = IntegerArray if type(dtype) is Int64Dtype else NumpyExtensionArray
        if type(column.array) is not expected:
            return None
        # The native row path rejects invalid nullability before any Config RPC.
        if (fields is not None and not fields[i].nullable
                and type(dtype) is Int64Dtype and column.hasnans):
            return None

    import pyarrow as pa

    # Arrow consumes each Series' integer buffer and nullable mask independently.
    return pa.Table.from_arrays(
        [pa.array(column, type=pa.int64(), from_pandas=True) for column in columns],
        names=names,
    )
