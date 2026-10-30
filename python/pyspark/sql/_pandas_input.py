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


def _has_canonical_columns(columns):
    """Decline label containers whose row-path ``tolist`` can run custom hooks."""
    from types import FunctionType

    import numpy as np
    import pandas as pd
    from pandas.core import base as base_module
    from pandas.core.indexes import base as index_module

    Index = index_module.Index
    if (pd.Index is not Index or type(columns) is not Index
            or Index.__getattribute__ is not object.__getattribute__
            or "tolist" in vars(columns)):
        return False

    def ordinary(method, name, module):
        return (type(method) is FunctionType and method.__name__ == name
                and method.__module__ == module.__name__
                and method.__globals__ is vars(module))

    # The native row path calls columns.tolist() before normalizing any cells.
    # Decline overrides without invoking them, so fallback preserves that order.
    if (Index.tolist is not base_module.IndexOpsMixin.tolist
            or not ordinary(Index.tolist, "tolist", base_module)):
        return False
    values = vars(Index).get("_values")
    if (type(values) is not property
            or not ordinary(values.fget, "_values", index_module)):
        return False
    if any("_data" in vars(base) for base in Index.__mro__):
        return False
    backing = vars(columns).get("_data")
    if type(backing) is np.ndarray:
        return True

    try:
        from pandas.core.arrays import numpy_ as numpy_module, string_ as string_module
        from pandas.core.arrays import string_arrow as string_arrow_module
        from pandas.core.arrays.arrow import array as arrow_module
    except ImportError:
        # Older pandas releases may lack these concrete string storage types.
        return False

    python_strings = tuple(vars(string_module).get(name) for name in
                           ("StringArray", "StringArrayNumpySemantics"))
    arrow_strings = tuple(vars(string_arrow_module).get(name) for name in
                          ("ArrowStringArray", "ArrowStringArrayNumpySemantics"))
    cls = type(backing)
    if not any(cls is candidate for candidate in python_strings + arrow_strings):
        return False
    if (cls.__getattribute__ is not object.__getattribute__
            or any(name in vars(backing) for name in
                   ("tolist", "to_numpy", "isna", "fillna", "_hasna"))
            or getattr(cls, "tolist", None) is not getattr(
                vars(string_module).get("BaseStringArray"), "tolist", None)
            or not ordinary(getattr(cls, "tolist", None), "tolist", string_module)):
        return False
    if any(cls is candidate for candidate in python_strings):
        owner, module = vars(numpy_module).get("NumpyExtensionArray"), numpy_module
        string_array = vars(string_module).get("StringArray")
        if owner is None or string_array is None:
            return False
        try:
            storage = object.__getattribute__(backing, "_ndarray")
        except AttributeError:
            return False
        if type(storage) is not np.ndarray:
            return False
        isna = vars(string_array).get("isna", getattr(owner, "isna", None))
        isna_module = string_module if "isna" in vars(string_array) else module
    else:
        import pyarrow as pa

        owner, module = vars(arrow_module).get("ArrowExtensionArray"), arrow_module
        if owner is None:
            return False
        try:
            storage = object.__getattribute__(backing, "_pa_array")
        except AttributeError:
            return False
        if type(storage) is not pa.ChunkedArray:
            return False
        isna, isna_module = getattr(owner, "isna", None), module
        hasna = getattr(cls, "_hasna", None)
        if (type(hasna) is not property
                or not ordinary(hasna.fget, "_hasna", module)
                or getattr(cls, "fillna", None) is not getattr(owner, "fillna", None)
                or not ordinary(getattr(cls, "fillna", None), "fillna", module)):
            return False
    # BaseStringArray.tolist delegates to to_numpy, which consults isna (or
    # Arrow's _hasna). Authenticate those direct hooks as well as the storage.
    return (getattr(cls, "to_numpy", None) is getattr(owner, "to_numpy", None)
            and ordinary(getattr(cls, "to_numpy", None), "to_numpy", module)
            and getattr(cls, "isna", None) is isna
            and ordinary(getattr(cls, "isna", None), "isna", isna_module))


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

    if not _has_canonical_columns(frame.columns):
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
