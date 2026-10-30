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


def prepare_string_input(frame, schema):
    """Prepare inline IPC for explicit columns of exact strings or None.

    Inspect every object before Arrow conversion: str subclasses can override
    conversion hooks, and pandas treats several other objects as missing values.
    Those cases must keep the existing row conversion and its error ordering.
    """
    import sys

    import numpy as np
    import pandas as pd
    from pandas.core import frame as frame_module, generic as generic_module
    from pandas.core.frame import DataFrame
    from pandas.core.series import Series
    from types import FunctionType, GetSetDescriptorType, MethodDescriptorType, WrapperDescriptorType

    from pyspark._pyspark import (
        BooleanType, ByteType, ShortType, IntegerType, LongType, FloatType,
        DoubleType, DecimalType, StringType, StructField, StructType as NativeStructType,
    )
    from pyspark.sql import types as sql_types

    # Keep small frames on the established row path.
    if (pd.DataFrame is not DataFrame or type(frame) is not DataFrame
            or type(schema) is not NativeStructType
            or len(frame) < 10000
            or any(name in vars(frame) for name in ("astype", "notna", "where"))):
        return None
    # Preserve the direct normalization hooks used by the existing row path.
    for name, module in (("astype", generic_module), ("notna", frame_module),
                         ("where", generic_module)):
        method = getattr(DataFrame, name)
        if (type(method) is not FunctionType or method.__name__ != name
                or method.__module__ != module.__name__ or method.__globals__ is not vars(module)):
            return None
    if (DataFrame.astype is not generic_module.NDFrame.astype
            or DataFrame.where is not generic_module.NDFrame.where):
        return None
    # Earlier converter branches must not recognize a string as another type.
    # Check a loaded converter too: it keeps its own imported type aliases.
    if sys.modules.get("pyspark.sql.types") is not sql_types:
        return None
    importer = sys.modules.get("pyspark.sql._arrow_input")
    if importer is not None and getattr(importer, "StructType", None) is not NativeStructType:
        return None
    converter = sys.modules.get("pyspark.sql.pandas.types")
    for name, cls in (
        ("BooleanType", BooleanType), ("ByteType", ByteType), ("ShortType", ShortType),
        ("IntegerType", IntegerType), ("LongType", LongType), ("FloatType", FloatType),
        ("DoubleType", DoubleType), ("DecimalType", DecimalType), ("StringType", StringType),
        ("StructField", StructField), ("StructType", NativeStructType),
    ):
        if (getattr(sql_types, name, None) is not cls
                or converter is not None and getattr(converter, name, None) is not cls):
            return None
    # The row path reads native fields directly. Class attributes are mutable,
    # so customized Python schema access must not select a different Arrow path.
    for name, cls in (("StringType", StringType), ("StructField", StructField),
                      ("StructType", NativeStructType)):
        if (getattr(sql_types, name, None) is not cls
                or cls.__getattribute__ is not object.__getattribute__
                or any("__class__" in vars(base) for base in cls.__mro__ if base is not object)):
            return None
    for cls, names, descriptor_type in (
        (NativeStructType, ("fields", "names"), GetSetDescriptorType),
        (NativeStructType, ("__iter__",), WrapperDescriptorType),
        (StructField, ("name", "dataType", "nullable", "metadata"), GetSetDescriptorType),
        (StringType, ("isUTF8BinaryCollation",), MethodDescriptorType),
    ):
        for name in names:
            descriptor = vars(cls).get(name)
            if (type(descriptor) is not descriptor_type
                    or descriptor.__objclass__ is not cls or descriptor.__name__ != name):
                return None
    if not _has_canonical_columns(frame.columns):
        return None
    width = len(frame.columns)
    if not width or len(schema.fields) != width or not frame.columns.is_unique:
        return None
    if not all(type(name) is str and "\x00" not in name for name in frame.columns):
        return None
    if not all(
        type(field) is StructField and type(field.dataType) is StringType
        and field.dataType.isUTF8BinaryCollation()
        and type(field.name) is str and "\x00" not in field.name
        for field in schema.fields
    ) or len(set(schema.names)) != width:
        return None
    if not all(type(dtype) is type(np.dtype("object")) and dtype == np.dtype("object")
               for dtype in frame.dtypes):
        return None
    # A custom ExtensionArray may report a genuine NumPy dtype while its
    # Arrow hook emits different values from to_numpy(). Trust only pandas'
    # concrete NumPy backing array, including its name in older pandas releases.
    try:
        from pandas.core.arrays.numpy_ import NumpyExtensionArray
    except ImportError:
        try:
            from pandas.core.arrays.numpy_ import PandasArray as NumpyExtensionArray
        except ImportError:
            return None
    columns = [frame.iloc[:, i] for i in range(width)]
    if any(type(column) is not Series or type(column.array) is not NumpyExtensionArray
           for column in columns):
        return None

    from pyspark._pyspark import _pandas_string_ipc

    # Preflight exact object types before the native Arrow buffer traversal.
    # Unsupported values, non-nullable None, and surrogates return to the row
    # path before serialization. Keep its inline transport and configuration
    # behavior rather than selecting the public Arrow import/upload policy.
    arrays = [column.array._ndarray for column in columns]
    if any(type(array) is not np.ndarray or array.ndim != 1
           or type(array.dtype) is not type(np.dtype("object"))
           or array.dtype != np.dtype("object") or len(array) != len(frame)
           for array in arrays):
        return None
    return _pandas_string_ipc(arrays, schema)
