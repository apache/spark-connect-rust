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


_ARROW_STRING_HOOKS = ("__arrow_array__", "__array__", "__getitem__", "isna",
                       "to_numpy", "__len__")
_ARROW_STRING_CONSTRUCTION_HOOKS = ("__init__", "_from_pyarrow_array", "__new__")


def _defined_in(method, name, *modules):
    """Whether `method` is a plain function named `name` whose module name and
    globals belong to one of `modules`.
    """
    from types import FunctionType

    return (type(method) is FunctionType and method.__name__ == name
            and any(method.__module__ == module.__name__
                    and method.__globals__ is vars(module) for module in modules))


def _raw_class_attribute(cls, name, start=0):
    """Return the defining class and raw attribute, or (None, None) if absent."""
    if cls is not None:
        for base in cls.__mro__[start:]:
            namespace = vars(base)
            if name in namespace:
                return base, namespace[name]
    return None, None


def _has_canonical_arrow_string_super_method(cls, start, name, owner, module):
    """Authenticate the immediate method reached through ArrowStringArray's super()."""
    actual_owner, method = _raw_class_attribute(cls, name, start)
    return actual_owner is owner and _defined_in(method, name, module)


def _arrow_string_class_is_canonical(cls):
    """Authenticate the shared row/Arrow conversion hooks of a concrete pandas class."""
    import pandas as pd
    from pandas.core.arrays import base as array_base_module, string_ as string_module, string_arrow
    from pandas.core.arrays.arrow import array as arrow_module

    numpy_semantics = getattr(string_arrow, "ArrowStringArrayNumpySemantics", None)
    # Reject subclasses by identity before inspecting them. In particular, do not
    # hash an untrusted class whose metaclass could customize __hash__.
    if (not any(cls is candidate for candidate in
                (string_arrow.ArrowStringArray, numpy_semantics))
            or pd.arrays.ArrowStringArray is not string_arrow.ArrowStringArray
            or pd.StringDtype is not string_module.StringDtype):
        return False
    try:
        super_start = cls.__mro__.index(string_arrow.ArrowStringArray) + 1
    except ValueError:
        return False

    # Authenticate row/Arrow conversion hooks and __len__, which sizes the table.
    # to_numpy indexes the non-null portion through __getitem__ before reading storage.
    # pandas 2.2's NumPy-semantics variant has its own string-method accessor.
    access = _raw_class_attribute(cls, "__getattribute__")[1]
    if access is not object.__getattribute__:
        if cls is not numpy_semantics or not _defined_in(
                access, "__getattribute__", string_arrow):
            return False
        from functools import partial
        from pandas.core.arrays import _arrow_string_mixins as string_mixins

        mixin = string_mixins.ArrowStringArrayMixin
        # This accessor dispatches through the mixin before the ordinary MRO.
        # Authenticate its globals and reject intercepted conversion names.
        # Construction itself uses class slots, checked separately below;
        # the mixin's canonical __init__ is not an instance conversion hook.
        if (not any(base is mixin for base in cls.__mro__)
                or vars(string_arrow).get("ArrowStringArrayMixin") is not mixin
                or vars(string_arrow).get("partial") is not partial
                or any(name in vars(mixin) for name in
                       _ARROW_STRING_HOOKS + ("astype", "dtype", "_hasna", "fillna",
                                              "_from_pyarrow_array", "__new__"))):
            return False
    if (any(not _defined_in(_raw_class_attribute(cls, name)[1], name,
                           string_arrow, arrow_module) for name in _ARROW_STRING_HOOKS)
            or not _defined_in(_raw_class_attribute(cls, "astype")[1], "astype", string_arrow)
            or not _has_canonical_arrow_string_super_method(
                cls, super_start, "astype", array_base_module.ExtensionArray, array_base_module)):
        return False
    # Null-subarray conversion may construct another array. Older pandas versions
    # have no _from_pyarrow_array helper; a present custom descriptor must decline.
    if (not _defined_in(_raw_class_attribute(cls, "__init__")[1], "__init__", string_arrow)
            or not _has_canonical_arrow_string_super_method(
                cls, super_start, "__init__", arrow_module.ArrowExtensionArray, arrow_module)):
        return False
    factory_owner, factory = _raw_class_attribute(cls, "_from_pyarrow_array")
    if factory_owner is not None and not _defined_in(factory, "_from_pyarrow_array", string_arrow):
        return False
    if _raw_class_attribute(cls, "__new__")[1] is not object.__new__:
        return False
    dtype = _raw_class_attribute(cls, "dtype")[1]
    hasna = _raw_class_attribute(cls, "_hasna")[1]
    missing_value = _raw_class_attribute(string_module.StringDtype, "na_value")[1]
    return (type(dtype) is property
            and _defined_in(dtype.fget, "dtype", string_arrow, arrow_module)
            and type(hasna) is property
            and _defined_in(hasna.fget, "_hasna", arrow_module)
            and type(missing_value) is property
            and _defined_in(missing_value.fget, "na_value", string_module))


def _has_canonical_arrow_string_array(array):
    """Authenticate a column at this point without caching mutable hooks.

    Callers may already have read frame.dtypes, as the native row path does.
    """
    # Column lookup can run customized pandas accessors between these checks.
    # An earlier column authenticates neither the next instance nor its class.
    if not _arrow_string_class_is_canonical(type(array)):
        return False
    # Deliberately include descriptors and special names for future explicit lookups.
    return not any(name in vars(array) for name in
                   _ARROW_STRING_HOOKS + _ARROW_STRING_CONSTRUCTION_HOOKS
                   + ("astype", "dtype", "_hasna"))


def _has_canonical_columns(columns):
    """Decline label containers whose row-path iteration can run custom hooks."""
    import sys

    import numpy as np
    import pandas as pd
    from pandas.core import base as base_module
    from pandas.core.indexes import base as index_module

    Index = index_module.Index
    if (pd.Index is not Index or type(columns) is not Index
            or Index.__getattribute__ is not object.__getattribute__
            or "__iter__" in vars(columns)):
        return False

    # The native row path iterates columns before normalizing any cells.
    # Decline overrides without invoking them, so fallback preserves that order.
    if (Index.__iter__ is not base_module.IndexOpsMixin.__iter__
            or not _defined_in(Index.__iter__, "__iter__", base_module)):
        return False
    values = vars(Index).get("_values")
    if (type(values) is not property
            or not _defined_in(values.fget, "_values", index_module)):
        return False
    if any("_data" in vars(base) for base in Index.__mro__):
        return False
    backing = vars(columns).get("_data")
    if type(backing) is np.ndarray:
        return True

    # Arrow-backed labels already hold PyArrow storage. Do not import optional
    # Arrow modules just to decide whether a fallback input is eligible.
    pa = sys.modules.get("pyarrow")
    if pa is None:
        return False

    try:
        from pandas.core.arrays import numpy_ as numpy_module, string_ as string_module
        from pandas.core.arrays import base as array_base_module, _mixins as mixins_module
        from pandas._libs.arrays import NDArrayBacked
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
    arrow_backed = any(cls is candidate for candidate in arrow_strings)
    if arrow_backed and not _arrow_string_class_is_canonical(cls):
        return False

    def class_attribute(owner, name):
        # Arrow labels share the raw lookup; Python-string class lookup stays unchanged.
        if arrow_backed:
            return _raw_class_attribute(owner, name)[1]
        return getattr(owner, name, None)

    # The shared instance-shadow list conservatively includes descriptors and special
    # names for both Arrow-backed and Python-backed string labels.
    if (class_attribute(cls, "__getattribute__") is not object.__getattribute__
            or any(name in vars(backing) for name in
                   ("__iter__", "__getitem__", "__len__", "__array__", "_box_func",
                    "tolist", "to_numpy", "isna", "fillna", "_hasna")
                   + _ARROW_STRING_CONSTRUCTION_HOOKS)
            or class_attribute(cls, "tolist") is not class_attribute(
                vars(string_module).get("BaseStringArray"), "tolist")
            or not _defined_in(class_attribute(cls, "tolist"), "tolist", string_module)):
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
        # ExtensionArray iteration dispatches through len and indexed boxing.
        for name, hook_module in (("__iter__", array_base_module),
                                  ("__getitem__", mixins_module), ("_box_func", mixins_module)):
            if not _defined_in(getattr(cls, name, None), name, hook_module):
                return False
        if (getattr(cls, "__len__", None) is not NDArrayBacked.__len__
                or getattr(cls, "ndim", None) is not NDArrayBacked.ndim
                or any("_ndarray" in vars(base) for base in cls.__mro__
                       if base is not NDArrayBacked)):
            return False
        isna = vars(string_array).get("isna", getattr(owner, "isna", None))
        isna_module = string_module if "isna" in vars(string_array) else module
    else:
        owner, module = vars(arrow_module).get("ArrowExtensionArray"), arrow_module
        if owner is None:
            return False
        # pandas builds an Arrow string Index's uniqueness engine with astype(object).
        # Authenticate the recognized object-cast methods and dtype/missing-value
        # properties before is_unique. This is a bounded check of pandas internals.
        if "astype" in vars(backing):
            return False
        try:
            storage = object.__getattribute__(backing, "_pa_array")
        except AttributeError:
            return False
        if (type(storage) is not pa.ChunkedArray
                or not _defined_in(class_attribute(cls, "__iter__"), "__iter__", module)):
            return False
        isna, isna_module = class_attribute(owner, "isna"), module
        hasna = class_attribute(cls, "_hasna")
        if (type(hasna) is not property
                or not _defined_in(hasna.fget, "_hasna", module)
                or class_attribute(cls, "fillna") is not class_attribute(owner, "fillna")
                or not _defined_in(class_attribute(cls, "fillna"), "fillna", module)):
            return False
    # Keep the existing storage/conversion authentication as well as the
    # iterator hooks: these objects also supply labels to pandas normalization.
    return (class_attribute(cls, "to_numpy") is class_attribute(owner, "to_numpy")
            and _defined_in(class_attribute(cls, "to_numpy"), "to_numpy", module)
            and class_attribute(cls, "isna") is isna
            and _defined_in(class_attribute(cls, "isna"), "isna", isna_module))


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
        AtomicType, BooleanType, ByteType, ShortType, IntegerType, LongType, StructField, StructType,
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
    # The importer's configuration RPC outweighs row conversion for small NumPy frames.
    # Nullable Int64 frames use Arrow at any size, which keeps their null masks.
    if len(frame) < 1000 and not any(type(dtype) is Int64Dtype for dtype in dtypes):
        return None

    # The native row path reads native schema fields directly. The Arrow converter
    # instead uses Python descriptors and imported class aliases, including aliases
    # captured before this helper was loaded. Customized schemas retain the row path.
    if sys.modules.get("pyspark.sql.types") is not sql_types:
        return None
    importer = sys.modules.get("pyspark.sql._arrow_input")
    if vars(sql_types).get("AtomicType") is not AtomicType:
        return None
    if importer is not None:
        if getattr(importer, "StructType", None) is not StructType:
            return None
        # The importer uses this alias while validating schema widths.
        if vars(importer).get("AtomicType") is not AtomicType:
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
        # As PySpark Connect, the Arrow table keeps the pandas names, and the list is
        # applied with toDF.

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
    from types import FunctionType, GetSetDescriptorType, WrapperDescriptorType

    import numpy as np
    import pandas as pd
    from pandas.core import frame as frame_module, generic as generic_module
    try:
        from pandas.core.arrays.datetimes import DatetimeArray
        from pandas.core.dtypes.dtypes import DatetimeTZDtype
        from pandas.core.frame import DataFrame
        from pandas.core.series import Series
    except ImportError:
        return None
    from pyspark._pyspark import (
        AtomicType, BooleanType, ByteType, ShortType, IntegerType, LongType, FloatType,
        DoubleType, DecimalType, StringType, BinaryType, DateType, TimestampType,
        StructField, StructType as NativeStructType,
    )
    from pyspark.sql import types as sql_types

    if (pd.DataFrame is not DataFrame or type(frame) is not DataFrame
            or type(schema) is not NativeStructType or not len(frame)
            or any(name in vars(frame) for name in ("astype", "notna", "where"))):
        return None
    # The row path calls these normalization methods before reading cells.
    # Preserve customized transformations and errors before any Arrow Config RPC.
    for name, module, qualname in (
        ("astype", generic_module, "NDFrame.astype"),
        ("notna", frame_module, "DataFrame.notna"),
        ("where", generic_module, "NDFrame.where"),
    ):
        method = getattr(DataFrame, name)
        if (type(method) is not FunctionType or method.__name__ != name
                or method.__qualname__ != qualname or method.__module__ != module.__name__
                or method.__globals__ is not vars(module)):
            return None
    if (DataFrame.astype is not generic_module.NDFrame.astype
            or DataFrame.where is not generic_module.NDFrame.where):
        return None
    if not _has_canonical_columns(frame.columns):
        return None
    # Native row conversion reads the native schema directly. The Arrow importer
    # also consults Python descriptors and public or captured converter aliases.
    if sys.modules.get("pyspark.sql.types") is not sql_types:
        return None
    importer = sys.modules.get("pyspark.sql._arrow_input")
    if vars(sql_types).get("AtomicType") is not AtomicType:
        return None
    if importer is not None:
        if getattr(importer, "StructType", None) is not NativeStructType:
            return None
        if vars(importer).get("AtomicType") is not AtomicType:
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
    # Reject unsupported host zones before Rust queries the session time zone.
    # Their row fallback builds a local relation without any configuration RPC.
    if not timestamp_columns_are_aware(columns) and not _timestamp_host_is_utc():
        return None
    return columns


def timestamp_columns_are_aware(columns):
    """Whether every column carries its time zone, so the session's is not needed."""
    from pandas.core.dtypes.dtypes import DatetimeTZDtype

    return all(type(column.dtype) is DatetimeTZDtype for column in columns)


def _timestamp_host_is_utc():
    """Require an explicit UTC host setting and an initialized UTC runtime zone."""
    import os
    import time

    return os.environ.get("TZ") == "UTC" and time.tzname == ("UTC", "UTC")


def prepare_timestamp_table(columns, schema, timezone):
    """Preserve instants without row materialization or a host-zone conversion.

    Aware columns carry their own timezone. Naive columns are deliberately
    restricted to literal ``TZ=UTC`` and session ``UTC``, with a UTC runtime
    zone: the existing row path uses the host zone, unlike Python Connect's
    session-zone conversion (and DST fold policy).
    """
    import pyarrow as pa

    if not timestamp_columns_are_aware(columns):
        if timezone != "UTC" or not _timestamp_host_is_utc():
            return None
    return pa.Table.from_arrays(
        [pa.array(column, type=pa.timestamp("us", tz="UTC"), from_pandas=True)
         for column in columns],
        names=schema.names,
    )


def prepare_float64_table(frame, schema):
    """Keep explicit native double columns columnar without changing pandas nulls."""
    import numpy as np
    import pandas as pd
    from pandas.core import frame as frame_module, generic as generic_module
    from pandas.core.frame import DataFrame
    from pandas.core.series import Series
    from types import FunctionType, GetSetDescriptorType, WrapperDescriptorType

    from pyspark import _pyspark as native
    from pyspark.sql import types as sql_types

    # First-use Arrow imports outweigh the conversion saving for smaller frames.
    # Keep those on the existing row path.
    if (pd.DataFrame is not DataFrame or type(frame) is not DataFrame
            or type(schema) is not native.StructType or len(frame) < 100000
            or any(name in vars(frame) for name in ("astype", "notna"))):
        return None
    if not _has_canonical_columns(frame.columns):
        return None
    # Class or instance conversion hooks must retain the row path's calls.
    for name, module in (("astype", generic_module), ("notna", frame_module),
                         ("where", generic_module)):
        method = getattr(DataFrame, name)
        if (type(method) is not FunctionType or method.__name__ != name
                or method.__module__ != module.__name__ or method.__globals__ is not vars(module)):
            return None
    if (DataFrame.astype is not generic_module.NDFrame.astype
            or DataFrame.where is not generic_module.NDFrame.where):
        return None
    width = len(frame.columns)
    if not width or not frame.columns.is_unique or not all(
        type(name) is str and "\x00" not in name for name in frame.columns
    ):
        return None
    double_dtype = np.dtype("float64")
    if not all(type(dtype) is type(double_dtype) and dtype == double_dtype
               for dtype in frame.dtypes):
        return None
    # A custom ExtensionArray can advertise the literal NumPy dtype while its
    # Arrow hook emits other values. Accept only pandas' concrete NumPy storage.
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
    # NumPy subclasses can customize the row path's astype conversion even
    # inside pandas' concrete NumPy wrapper. Keep their hooks, accepting the
    # row path's cost for subclasses such as np.memmap.
    if any(type(column.array._ndarray) is not np.ndarray for column in columns):
        return None
    # Native row conversion reads Rust fields, bypassing mutable Python accessors.
    for name in ("DoubleType", "StructField", "StructType"):
        cls = getattr(native, name)
        if (vars(sql_types).get(name) is not cls
                or cls.__getattribute__ is not object.__getattribute__
                or any("__class__" in vars(base) for base in cls.__mro__ if base is not object)):
            return None
    for cls, names, descriptor_type in (
        (native.StructType, ("fields", "names"), GetSetDescriptorType),
        (native.StructType, ("__iter__",), WrapperDescriptorType),
        (native.StructField, ("name", "dataType", "nullable", "metadata"), GetSetDescriptorType),
    ):
        for name in names:
            descriptor = vars(cls).get(name)
            if (type(descriptor) is not descriptor_type
                    or descriptor.__objclass__ is not cls or descriptor.__name__ != name):
                return None
    fields = schema.fields
    if len(fields) != width or not all(
        type(field) is native.StructField and type(field.dataType) is native.DoubleType
        and type(field.name) is str and "\x00" not in field.name for field in fields
    ) or len(set(schema.names)) != width:
        return None

    # Use the same pandas missing-value semantics as the row path, including the
    # legacy use_inf_as_na option. Infinities and signed zero otherwise stay intact.
    valid = frame.notna()
    if any(not field.nullable and not valid.iloc[:, i].all()
           for i, field in enumerate(fields)):
        return None

    import pyarrow as pa

    return pa.Table.from_arrays(
        [pa.array(column, type=pa.float64(),
                  mask=~valid.iloc[:, i], from_pandas=False)
         for i, column in enumerate(columns)],
        # Match the row serializer's physical fields. The complete Spark
        # metadata remains in the explicit logical schema, not Arrow metadata.
        schema=pa.schema([pa.field(field.name, pa.float64(), nullable=field.nullable)
                          for field in fields]),
    )


def prepare_float64_input(frame, schema):
    """Serialize supported double columns with the existing row transport policy."""
    table = prepare_float64_table(frame, schema)
    if table is None:
        return None

    import pyarrow as pa

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return sink.getvalue().to_pybytes()


def prepare_string_input(frame, schema):
    """Prepare inline IPC for explicit columns of exact strings or None, or an Arrow table for
    columns that pandas keeps in Arrow string storage (see _arrow_string_table).

    Inspect every object before Arrow conversion: str subclasses can override
    conversion hooks, and pandas treats several other objects as missing values.
    Those cases must keep the existing row conversion and its error ordering.
    """
    import numpy as np
    import pandas as pd
    from pandas.core import frame as frame_module, generic as generic_module
    from pandas.core.frame import DataFrame
    from pandas.core.series import Series
    from types import GetSetDescriptorType, MethodDescriptorType, WrapperDescriptorType

    from pyspark._pyspark import StringType, StructField, StructType as NativeStructType
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
        if not _defined_in(method, name, module):
            return None
    if (DataFrame.astype is not generic_module.NDFrame.astype
            or DataFrame.where is not generic_module.NDFrame.where):
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
    dtypes = frame.dtypes
    if not all(type(dtype) is type(np.dtype("object")) and dtype == np.dtype("object")
               for dtype in dtypes):
        return _arrow_string_table(frame, schema, dtypes)
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


def _arrow_string_table(frame, schema, dtypes):
    """The columns of a frame kept in pandas' Arrow string storage (string[pyarrow], and the
    str dtype of pandas 3), as an Arrow table of the schema, or None.

    For the row path, pandas converts such a column from that storage, with its missing
    values as None, so the storage holds the same values. Only pandas' own array classes are
    trusted with their original conversion hooks: customization could convert differently.
    The caller's guards of prepare_string_input (frame, schema, names, size) have passed.
    """
    from pandas.core.arrays import string_ as string_module

    if not all(type(dtype) is string_module.StringDtype
               and dtype.storage in ("pyarrow", "pyarrow_numpy")
               for dtype in dtypes):
        return None
    # The importer uses mutable public/captured schema names. Decline a rebound
    # namespace before conversion or configuration/upload RPCs, as the generic route does.
    if not _converter_namespaces_intact():
        return None

    import pyarrow as pa
    from pandas.core.series import Series

    arrays = []
    for index, field in enumerate(schema.fields):
        column = frame.iloc[:, index]
        if type(column) is not Series:
            return None
        array = column.array
        if not _has_canonical_arrow_string_array(array):
            return None
        values = array.__arrow_array__()
        # pandas returns a ChunkedArray; normalize a plain Array defensively if
        # an instance's _pa_array storage was replaced.
        if issubclass(type(values), pa.Array):
            values = pa.chunked_array([values])
        elif type(values) is not pa.ChunkedArray:
            return None
        if values.type not in (pa.string(), pa.large_string()):
            return None
        try:
            # Malformed offsets or UTF-8 keep the row path and its error.
            values.validate(full=True)
            # Check normal-string capacity before configuration/upload RPCs. The importer
            # may widen back to large_string when useLargeVarTypes is enabled.
            # pandas 2.2+ stores large_string, so this narrowing copies each column's offsets.
            values = values.cast(pa.string())
        except (pa.ArrowException, OverflowError):
            return None
        if not field.nullable and values.null_count:
            return None
        arrays.append(values)
    # The importer applies the nullability and metadata of the schema, and uploads the table
    # when it is at least spark.sql.session.localRelationCacheThreshold.
    return pa.Table.from_arrays(arrays, names=schema.names)


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
    from pandas.core import frame as frame_module, generic as generic_module
    from pandas.core.frame import DataFrame
    from pandas.core.series import Series
    from types import FunctionType, GetSetDescriptorType, WrapperDescriptorType

    from pyspark._pyspark import (
        AtomicType, BooleanType, ByteType, ShortType, IntegerType, LongType, FloatType,
        DoubleType, DecimalType, StringType, BinaryType, DateType, StructField,
        StructType as NativeStructType,
    )
    from pyspark.sql import types as sql_types

    # Keep smaller frames on the row path to avoid the importer's config RPC overhead.
    if (pd.DataFrame is not DataFrame or type(frame) is not DataFrame
            or type(schema) is not NativeStructType
            or len(frame) < 10000
            or any(name in vars(frame) for name in ("astype", "notna", "where"))):
        return None
    if not _has_canonical_columns(frame.columns):
        return None
    # The row path calls these normalization methods. Custom class or instance
    # hooks must retain those calls and the values they produce before Arrow RPCs.
    for name, module, qualname in (
        ("astype", generic_module, "NDFrame.astype"),
        ("notna", frame_module, "DataFrame.notna"),
        ("where", generic_module, "NDFrame.where"),
    ):
        method = getattr(DataFrame, name)
        if (type(method) is not FunctionType or method.__name__ != name
                or method.__qualname__ != qualname or method.__module__ != module.__name__
                or method.__globals__ is not vars(module)):
            return None
    if (DataFrame.astype is not generic_module.NDFrame.astype
            or DataFrame.where is not generic_module.NDFrame.where):
        return None
    # Earlier converter branches must not recognize a date as another type.
    # Check loaded converters and globals captured by older importer functions.
    if sys.modules.get("pyspark.sql.types") is not sql_types:
        return None
    importer = sys.modules.get("pyspark.sql._arrow_input")
    if vars(sql_types).get("AtomicType") is not AtomicType:
        return None
    if importer is not None:
        if getattr(importer, "StructType", None) is not NativeStructType:
            return None
        if vars(importer).get("AtomicType") is not AtomicType:
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
    arrays = []
    for column, field in zip(columns, schema.fields):
        if type(column) is not Series:
            return None
        backing = column.array
        if type(backing) is not NumpyExtensionArray:
            return None
        array = backing._ndarray
        if (type(array) is not np.ndarray or array.ndim != 1
                or type(array.dtype) is not type(np.dtype("object"))
                or array.dtype != np.dtype("object") or len(array) != len(frame)):
            return None
        # Arrow rejects nulls in non-nullable fields; let the row path retain
        # its existing behavior for that input instead.
        if not all(type(value) is date or (value is None and field.nullable)
                   for value in array):
            return None
        arrays.append(array)

    import pyarrow as pa

    return pa.Table.from_arrays(
        # Series accessors can be customized independently of the DataFrame row
        # path. Emit the same concrete storage whose cells were checked above.
        [pa.array(array, type=pa.date32(), from_pandas=False) for array in arrays],
        names=schema.names,
    )


def _native_names_intact(namespace):
    """Whether each public native class name in `namespace` is still the native class."""
    from pyspark import _pyspark as native

    return all(
        namespace[name] is value
        for name, value in vars(native).items()
        if not name.startswith("_") and isinstance(value, type) and name in namespace
    )


def _converter_namespaces_intact():
    """Whether loaded schema converters/importers still resolve native type names."""
    import sys

    from pyspark.sql import types as sql_types

    if sys.modules.get("pyspark.sql.types") is not sql_types or not _native_names_intact(
        vars(sql_types)
    ):
        return False
    for name in ("pyspark.sql.pandas.types", "pyspark.sql._arrow_input"):
        module = sys.modules.get(name)
        if module is not None and not _native_names_intact(vars(module)):
            return False
    # The importer resolves its schema classes by name, and its conversion functions may
    # keep the globals of an older type module.
    importer = sys.modules.get("pyspark.sql._arrow_input")
    if importer is not None:
        from pyspark._pyspark import AtomicType, StructType

        if any(vars(importer).get(name) is not cls
               for name, cls in (("StructType", StructType), ("AtomicType", AtomicType))):
            return False
        for function in list(vars(importer).values()):
            namespace = getattr(function, "__globals__", None)
            if type(namespace) is dict and not _native_names_intact(namespace):
                return False
        # The importer calls its converters by these names, which must still be functions.
        converter = sys.modules.get("pyspark.sql.pandas.types")
        for name in ("to_arrow_schema", "from_arrow_schema"):
            if name in vars(importer) and (
                getattr(vars(importer)[name], "__globals__", None) is None
                or (converter is not None
                    and vars(importer)[name] is not getattr(converter, name, None))
            ):
                return False
    return True


def _canonical_struct(schema):
    """Whether a native struct schema and its fields keep their own classes' behavior.

    The conversion reads the fields, names and data types through the classes, so a
    customized class keeps the row path.
    """
    from types import GetSetDescriptorType, WrapperDescriptorType

    from pyspark._pyspark import StructField, StructType

    if type(schema) is not StructType:
        return False
    for cls in (StructType, StructField):
        if (cls.__getattribute__ is not object.__getattribute__
                or any("__class__" in vars(base) for base in cls.__mro__ if base is not object)):
            return False
    for cls, names, descriptor_type in (
        (StructType, ("fields", "names"), GetSetDescriptorType),
        (StructType, ("__iter__",), WrapperDescriptorType),
        (StructField, ("name", "dataType", "nullable", "metadata"), GetSetDescriptorType),
    ):
        for name in names:
            descriptor = vars(cls).get(name)
            if (type(descriptor) is not descriptor_type
                    or descriptor.__objclass__ is not cls or descriptor.__name__ != name):
                return False
    return all(
        type(field) is StructField and type(field.name) is str and "\x00" not in field.name
        and type(field.dataType).__getattribute__ is object.__getattribute__
        for field in schema.fields
    )


def accepts_pandas_table(frame, schema):
    """Whether `prepare_pandas_table` may convert this frame under this schema.

    Customized frames, column labels, schema classes and rebound type names keep the row
    path, which reads them in its own order. Conversion-hook authentication covers Arrow
    string storage, including inherited hooks. Other pandas arrays retain the existing
    module-origin checks; rebinding their canonical classes' hooks is not covered here.
    The conversion and importer resolve type names by lookup, so each namespace must hold
    the native classes. The schema is the caller's object or a DDL string, which is parsed.
    """
    return accepts_pandas_frame(frame) and (type(schema) is str or _canonical_struct(schema))


def accepts_pandas_frame(frame):
    """Check frame eligibility for explicit or inferred Arrow schema conversion."""
    import sys

    import pandas as pd
    from pandas.core.frame import DataFrame

    if pd.DataFrame is not DataFrame or type(frame) is not DataFrame:
        return False
    if not _has_canonical_columns(frame.columns):
        return False
    # Custom extension-array/dtype classes outside pandas and NumPy keep the row path.
    # Canonical non-string array classes retain the inherited module-origin policy.
    # Arrow string storage already loads its defining module; do not import optional
    # Arrow dependencies merely to inspect unrelated pandas columns.
    # The generic routes assume this sys.modules entry still defines live arrays;
    # removing or replacing that module while keeping old instances is not covered.
    string_arrow = sys.modules.get("pandas.core.arrays.string_arrow")
    arrow_strings = () if string_arrow is None else tuple(
        vars(string_arrow).get(name)
        for name in ("ArrowStringArray", "ArrowStringArrayNumpySemantics")
    )
    # Direct type checks bypass __subclasscheck__ and __mro__ metaclass callbacks.
    # Use the real type: isinstance can also consult a custom __class__ property.
    # Non-class aliases cannot own array storage.
    arrow_strings = tuple(cls for cls in arrow_strings
                          if type.__subclasscheck__(type, type(cls)))
    for i in range(len(frame.columns)):
        column = frame.iloc[:, i]
        array = column.array
        # A canonical class can still have customized hooks. Generic explicit and inferred
        # schema routes must preserve the specialized routes' row fallback.
        if (any(type.__subclasscheck__(cls, type(array)) for cls in arrow_strings)
                and not _has_canonical_arrow_string_array(array)):
            return False
        if not all(type(value).__module__.split(".")[0] in ("pandas", "numpy")
                   for value in (array, column.dtype)):
            return False
    # The native row path normalizes cells through these methods, which a customized class
    # or instance keeps on the row path.
    from pandas.core import frame as frame_module, generic as generic_module

    if any(name in vars(frame) for name in ("astype", "notna", "where")):
        return False
    for name, module, qualname in (
        ("astype", generic_module, "NDFrame.astype"),
        ("notna", frame_module, "DataFrame.notna"),
        ("where", generic_module, "NDFrame.where"),
    ):
        method = getattr(DataFrame, name)
        if not _defined_in(method, name, module) or method.__qualname__ != qualname:
            return False
    if not _converter_namespaces_intact():
        return False
    return True


def prepare_pandas_table(frame, schema, timezone, safecheck, prefers_large_types):
    """Convert a pandas frame under an explicit struct schema as PySpark Connect does.

    Each column goes through PySpark's own Arrow conversion, with the session time zone
    and the safe-cast setting, and the table is then renamed and cast to the schema's
    Arrow types. The values and errors therefore match PySpark, including the nullability
    and overflow messages. The caller checks the frame with `accepts_pandas_table` and
    the column count against the schema first.
    """
    from pyspark.sql.pandas.conversion import create_arrow_table_from_pandas
    from pyspark.sql.pandas.types import _deduplicate_field_names, to_arrow_schema

    import pyarrow as pa

    deduplicated = _deduplicate_field_names(schema)
    arrow_schema = to_arrow_schema(
        deduplicated, timezone="UTC", prefers_large_types=prefers_large_types
    )
    # Top-level names are the caller's, which may repeat: the importer casts to them.
    arrow_schema = pa.schema([
        pa.field(name, field.type, field.nullable, field.metadata)
        for name, field in zip(schema.names, arrow_schema)
    ])
    series_with_types = [
        (column, field.dataType)
        for (_, column), field in zip(frame.items(), deduplicated.fields)
    ]
    table = create_arrow_table_from_pandas(
        series_with_types,
        timezone=timezone,
        safecheck=safecheck,
        prefers_large_types=prefers_large_types,
    )
    return table.rename_columns(schema.names).cast(arrow_schema)


def arrow_decimal_column(values, dtype, name):
    """The precision, scale and values of a pandas object column of Decimals and other values,
    as Arrow types and converts it: PySpark's conversion of pandas leaves such a column to
    Arrow, which takes ints as Decimals and raises for the others."""
    import pyarrow as pa
    from pyspark.errors import PySparkTypeError, PySparkValueError

    message = ("Exception thrown when converting pandas.Series (%s) with name '%s' to Arrow "
               "Array (None)." % (dtype, name))
    try:
        array = pa.array(values, from_pandas=True)
    except pa.lib.ArrowException as e:
        if isinstance(e, TypeError):
            raise PySparkTypeError(message) from e
        raise PySparkValueError(
            message + " It can be caused by overflows or other unsafe conversions warned by "
            "Arrow. Arrow safe type check can be disabled by using SQL config "
            "`spark.sql.execution.pandas.convertToArrowArraySafely`.") from e
    if not pa.types.is_decimal(array.type):
        raise PySparkTypeError(message)
    return array.type.precision, array.type.scale, array.to_pylist()


def pyarrow_available():
    """Whether pyarrow can be imported, checked without importing it. Without it, a frame
    without a schema keeps the row path, which does not need it."""
    import importlib.util

    try:
        return importlib.util.find_spec("pyarrow") is not None
    except (ImportError, ValueError):
        return False


def prepare_inferred_pandas_table(frame, timezone, safecheck, prefers_large_types,
                                  infer_dict_as_map, prefer_timestamp_ntz):
    """Convert a pandas frame without a schema, or of names, as PySpark Connect does.

    Each column goes through PySpark's own Arrow conversion, of the type that Arrow infers,
    except that datetimes are timestamps, of the session time zone, and timedeltas day-time
    intervals, for the server to infer the types of the table. With `inferPandasDictAsMap`
    (`infer_dict_as_map`), the frame converts under the struct of the types that Arrow infers,
    where a column of dicts is a map, and the table takes the names and the Arrow types of the
    struct. Returns the table and the names of that struct, or None. The caller checks the
    frame with `accepts_pandas_frame`.
    """
    import pandas as pd
    import pyarrow as pa
    from pandas.api.types import is_datetime64_dtype, is_timedelta64_dtype

    from pyspark.errors import PySparkValueError
    from pyspark.sql.pandas.conversion import create_arrow_table_from_pandas
    from pyspark.sql.pandas.types import (
        _deduplicate_field_names,
        from_arrow_type,
        to_arrow_schema,
    )
    from pyspark.sql.types import (
        DayTimeIntervalType,
        MapType,
        StringType,
        StructType,
        TimestampType,
    )

    struct = None
    if infer_dict_as_map:
        struct = StructType()
        for field in pa.Schema.from_pandas(frame):
            field_type = field.type
            if isinstance(field_type, pa.StructType):
                if len(field_type) == 0:
                    raise PySparkValueError(
                        errorClass="CANNOT_INFER_EMPTY_SCHEMA",
                        messageParameters={},
                    )
                spark_type = MapType(
                    StringType(), from_arrow_type(field_type.field(0).type, prefer_timestamp_ntz)
                )
            else:
                spark_type = from_arrow_type(field_type, prefer_timestamp_ntz)
            struct = struct.add(field.name, spark_type, nullable=field.nullable)
        deduplicated = _deduplicate_field_names(struct)
        spark_types = [field.dataType for field in deduplicated.fields]
    else:
        spark_types = [
            (
                TimestampType()
                if is_datetime64_dtype(t) or isinstance(t, pd.DatetimeTZDtype)
                else DayTimeIntervalType()
                if is_timedelta64_dtype(t)
                else None
            )
            for t in frame.dtypes
        ]
    if len(frame.columns) == 0:
        # pa.RecordBatch.from_pandas keeps the number of rows of a frame of no columns.
        table = pa.Table.from_batches([pa.RecordBatch.from_pandas(frame)])
    else:
        table = create_arrow_table_from_pandas(
            [(c, st) for (_, c), st in zip(frame.items(), spark_types)],
            timezone=timezone,
            safecheck=safecheck,
            prefers_large_types=prefers_large_types,
        )
    if struct is None:
        return table, None
    if len(struct.fields) > 0:
        arrow_schema = to_arrow_schema(
            deduplicated, timezone="UTC", prefers_large_types=prefers_large_types
        )
        table = table.rename_columns(deduplicated.names).cast(arrow_schema)
    return table, [str(name) for name in struct.fieldNames()]
