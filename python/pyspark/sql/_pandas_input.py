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


"""The conversions of the pandas DataFrames of createDataFrame.

As PySpark's createDataFrame, every pandas DataFrame, of any class, labels and arrays, converts
with PySpark's conversion of each column to Arrow (`prepare_pandas_table` and
`prepare_inferred_pandas_table`), of the conversions and the types that the module of the session
binds as PySpark Connect's module binds them (`_session_module`), so of a patched class or a
rebound type name too. The fast paths of this module take only plain frames of common columns, of
intact names and classes, of which they make the tables that the conversion makes, faster; they
decline the others, which take the conversion. The native row path of the session converts the
rows of a frame only where the conversion does not run: without PyArrow.
"""


_ARROW_STRING_HOOKS = ("__arrow_array__", "__array__", "__getitem__", "isna",
                       "to_numpy", "__len__")
_ARROW_STRING_CONSTRUCTION_HOOKS = ("__init__", "_from_pyarrow_array", "__new__")
_ARROW_STRING_MIXIN_ALLOWED_NAMES = (
    "__module__", "__qualname__", "__doc__", "__dict__", "__weakref__",
    "__firstlineno__", "__static_attributes__", "_pa_array", "__init__",
)


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
    """Check the identity and supported hooks of a concrete pandas Arrow-string class.

    Authenticate conversion, construction and allocation, dtype/missing-value
    properties, and NumPy-semantics accessor dispatch using raw MRO lookups.
    """
    import pandas as pd
    from pandas.core.arrays import base as array_base_module
    from pandas.core.arrays import string_ as string_module
    from pandas.core.arrays import string_arrow
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
        # A rebound optional NumPy-semantics class alias may lack that ancestry.
        return False

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
        # Check the mixin/partial aliases and allow only string-method names, class
        # metadata, and initializer/storage declarations in the mixin namespace.
        # Construction itself uses class slots, checked separately below;
        # the mixin's canonical __init__ is not an instance conversion hook.
        if (not any(base is mixin for base in cls.__mro__)
                or vars(string_arrow).get("ArrowStringArrayMixin") is not mixin
                or vars(string_arrow).get("partial") is not partial
                or any(name not in _ARROW_STRING_MIXIN_ALLOWED_NAMES
                       and not name.startswith("_str_") for name in vars(mixin))):
            return False
    # Authenticate row/Arrow conversion hooks and __len__, which sizes the table.
    # to_numpy indexes the non-null portion through __getitem__ before reading storage.
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

    Callers may already have read frame.dtypes.
    """
    # Column lookup can run customized pandas accessors between these checks.
    # An earlier column authenticates neither the next instance nor its class.
    if not _arrow_string_class_is_canonical(type(array)):
        return False
    # Deliberately include descriptors and special names for future explicit lookups.
    return not any(name in vars(array) for name in
                   _ARROW_STRING_HOOKS + _ARROW_STRING_CONSTRUCTION_HOOKS
                   + ("astype", "dtype", "_hasna"))


def _has_canonical_range_columns(columns, range_module):
    """Accept the default RangeIndex of column labels, which iterates its own int range.

    As for an Index, subclasses and instance overrides decline the fast paths.
    """
    RangeIndex = range_module.RangeIndex
    if (RangeIndex.__getattribute__ is not object.__getattribute__
            or "__iter__" in vars(columns) or "__len__" in vars(columns)
            or not _defined_in(RangeIndex.__iter__, "__iter__", range_module)
            or not _defined_in(RangeIndex.__len__, "__len__", range_module)):
        return False
    return type(vars(columns).get("_range")) is range


def _has_canonical_structured_columns(columns, module, multi):
    """Accept a plain MultiIndex or CategoricalIndex of column labels, as an Index is accepted.

    The labels only name the columns, and the Arrow routes name them as PySpark does, by `str`.
    The override rules of an Index apply: the exact class, no override of `__getattribute__`,
    `__iter__` or `__len__` on the class or the instance, and pandas' own iteration and values.
    The storage must be pandas' own too: the levels and codes of a MultiIndex, and the values
    and categories of a CategoricalIndex. Each label container inside is checked as a label
    index (see `_has_canonical_columns`), so a DatetimeIndex level is checked as one too.
    """
    import numpy as np
    from pandas.core import base as base_module
    from pandas.core.arrays import categorical as categorical_module
    from pandas.core.indexes import base as index_module
    from pandas.core.indexes import frozen as frozen_module

    cls = type(columns)
    if (cls.__getattribute__ is not object.__getattribute__
            or "__iter__" in vars(columns) or "__len__" in vars(columns)
            or cls.__iter__ is not base_module.IndexOpsMixin.__iter__
            or not _defined_in(base_module.IndexOpsMixin.__iter__, "__iter__", base_module)
            or not _defined_in(cls.__len__, "__len__", module, index_module)):
        return False
    _, values = _raw_class_attribute(cls, "_values")
    if not _defined_in(getattr(values, "fget", None), "_values", module, index_module,
                       base_module):
        return False

    if multi:
        levels = vars(columns).get("_levels")
        codes = vars(columns).get("_codes")
        if (type(levels) is not frozen_module.FrozenList
                or type(codes) is not frozen_module.FrozenList):
            return False
        return (all(_has_canonical_columns(level) for level in list.__iter__(levels))
                and all(type(code) is np.ndarray for code in list.__iter__(codes)))

    data = vars(columns).get("_data")
    categorical = categorical_module.Categorical
    if (type(data) is not categorical or "__iter__" in vars(data)
            or not _defined_in(categorical.__iter__, "__iter__", categorical_module)):
        return False
    # Read the categories through the pandas property, not an instance attribute.
    return _has_canonical_columns(categorical.categories.__get__(data))


def _has_canonical_label_index(columns, array_check, array_module, array_name):
    """Accept a plain DatetimeIndex, TimedeltaIndex, PeriodIndex or IntervalIndex of labels.

    As for an Index, the labels only name the columns. The override rules of an Index apply:
    the exact class, no override of `__getattribute__`, `__iter__` or `__len__` on the class
    or the instance, pandas' own iteration and values, and no class in the MRO that defines
    the `_data` storage. The storage must pass `array_check`, which takes the exact array
    class `array_name` of `array_module`.
    """
    from pandas.core import base as base_module
    from pandas.core.indexes import base as index_module

    cls = type(columns)
    if (cls.__getattribute__ is not object.__getattribute__
            or "__iter__" in vars(columns) or "__len__" in vars(columns)
            or cls.__iter__ is not base_module.IndexOpsMixin.__iter__
            or not _defined_in(base_module.IndexOpsMixin.__iter__, "__iter__", base_module)
            or not _defined_in(cls.__len__, "__len__", index_module)
            or any("_data" in vars(base) for base in cls.__mro__)):
        return False
    _, values = _raw_class_attribute(cls, "_values")
    if not _defined_in(getattr(values, "fget", None), "_values", index_module):
        return False
    return array_check(vars(columns).get("_data"), array_module, array_name)


def _has_canonical_datetimelike_array(backing, module, name):
    """Accept the exact DatetimeArray, TimedeltaArray or PeriodArray storage of a label index.

    Iteration boxes the int64 `_ndarray` through pandas' own `__iter__` and `_box_func`, and
    indexing goes through pandas' own `__getitem__`. No instance may shadow those hooks.
    """
    import numpy as np
    from pandas._libs.arrays import NDArrayBacked
    from pandas.core.arrays import _mixins as mixins_module
    from pandas.core.arrays import datetimelike as datetimelike_module

    cls = vars(module).get(name)
    if type(backing) is not cls:
        return False
    if cls.__getattribute__ is not object.__getattribute__ or any(
        hook in vars(backing) for hook in
        ("__iter__", "__getitem__", "__len__", "__array__", "_box_func", "asi8", "_ndarray")
    ):
        return False
    if (not _defined_in(_raw_class_attribute(cls, "__iter__")[1], "__iter__", module,
                        datetimelike_module)
            or not _defined_in(_raw_class_attribute(cls, "__getitem__")[1], "__getitem__",
                               datetimelike_module, mixins_module)
            or _raw_class_attribute(cls, "__len__")[1] is not NDArrayBacked.__len__
            or not _defined_in(_raw_class_attribute(cls, "_box_func")[1], "_box_func", module)):
        return False
    try:
        storage = object.__getattribute__(backing, "_ndarray")
    except AttributeError:
        return False
    return type(storage) is np.ndarray


def _has_canonical_interval_array(backing, module, name):
    """Accept the exact IntervalArray storage of an IntervalIndex label.

    Iteration boxes `_left` and `_right` through pandas' own `__array__`, `isna`, `closed` and
    `__getitem__`. Each endpoint is a NumPy array or a canonical datetime-like array.
    """
    import numpy as np
    from pandas.core.arrays import datetimes as datetimes_module
    from pandas.core.arrays import timedeltas as timedeltas_module
    from pandas.core.dtypes.dtypes import IntervalDtype

    cls = vars(module).get(name)
    if type(backing) is not cls:
        return False
    if cls.__getattribute__ is not object.__getattribute__ or any(
        hook in vars(backing) for hook in
        ("__iter__", "__len__", "__getitem__", "__array__", "isna", "closed", "dtype")
    ):
        return False
    if (not all(_defined_in(_raw_class_attribute(cls, hook)[1], hook, module)
                for hook in ("__iter__", "__len__", "__getitem__", "__array__", "isna"))
            or not all(_defined_in(getattr(_raw_class_attribute(cls, hook)[1], "fget", None),
                                   hook, module) for hook in ("closed", "dtype"))
            or type(vars(backing).get("_dtype")) is not IntervalDtype):
        return False
    for side in ("_left", "_right"):
        endpoint = vars(backing).get(side)
        if type(endpoint) is np.ndarray:
            continue
        if not (_has_canonical_datetimelike_array(endpoint, datetimes_module, "DatetimeArray")
                or _has_canonical_datetimelike_array(endpoint, timedeltas_module,
                                                     "TimedeltaArray")):
            return False
    return True


# The hooks of a label array that iterating, measuring and converting the labels of an Index
# reach, in pandas 2 and 3.
_LABEL_ARRAY_HOOKS = ("__iter__", "__getitem__", "__len__", "__array__", "__contains__",
                      "to_numpy", "astype", "isna", "tolist", "fillna", "_hasna", "dtype")


def _has_canonical_label_array(backing, pa):
    """Accept the exact masked or Arrow storage of an Index of labels.

    pandas' nullable labels (Int64, Float64 or boolean, as a pivot or `.T` of `convert_dtypes()`
    data has them) are an IntegerArray, FloatingArray or BooleanArray over the ndarrays `_data`
    and `_mask`, and Arrow-backed labels of any type (`dtype_backend="pyarrow"`) are an
    ArrowExtensionArray over the ChunkedArray `_pa_array`. As for the string labels, the class
    must be pandas' own, and every hook that reading the labels reaches must be a function (or
    the getter of a property or a cached property) of pandas' own array modules, shadowed by no
    instance attribute, so that reading the labels runs no user code.
    """
    import numpy as np

    try:
        from pandas._libs.properties import CachedProperty
        from pandas.core.arrays import base as base_module
        from pandas.core.arrays import boolean as boolean_module
        from pandas.core.arrays import floating as floating_module
        from pandas.core.arrays import integer as integer_module
        from pandas.core.arrays import masked as masked_module
        from pandas.core.arrays import numeric as numeric_module
        from pandas.core.arrays.arrow import array as arrow_module
    except ImportError:
        return False
    cls = type(backing)
    masked = (vars(integer_module).get("IntegerArray"), vars(floating_module).get("FloatingArray"),
              vars(boolean_module).get("BooleanArray"))
    if any(cls is candidate for candidate in masked if candidate is not None):
        data, mask = vars(backing).get("_data"), vars(backing).get("_mask")
        if (type(data) is not np.ndarray or type(mask) is not np.ndarray
                or mask.dtype != np.bool_):
            return False
        modules = (masked_module, numeric_module, boolean_module, integer_module,
                   floating_module, base_module)
    elif pa is not None and cls is vars(arrow_module).get("ArrowExtensionArray"):
        if type(vars(backing).get("_pa_array")) is not pa.ChunkedArray:
            return False
        modules = (arrow_module, base_module)
    else:
        return False
    if _raw_class_attribute(cls, "__getattribute__")[1] is not object.__getattribute__:
        return False
    for name in _LABEL_ARRAY_HOOKS:
        if name in vars(backing):
            return False
        hook = _raw_class_attribute(cls, name)[1]
        if type(hook) is property or type(hook) is CachedProperty:
            hook = hook.fget
        if not _defined_in(hook, name, *modules):
            return False
    return True


def _has_canonical_columns(columns):
    """Decline label containers whose iteration can run custom hooks: the fast paths read the
    labels, and run no user code, and a frame that they decline takes PySpark's conversion."""
    import sys

    import numpy as np
    import pandas as pd
    from pandas.core import base as base_module
    from pandas.core.arrays import datetimes as datetimes_array_module
    from pandas.core.arrays import interval as interval_array_module
    from pandas.core.arrays import period as period_array_module
    from pandas.core.arrays import timedeltas as timedeltas_array_module
    from pandas.core.indexes import base as index_module
    from pandas.core.indexes import category as category_module
    from pandas.core.indexes import datetimes as datetimes_module
    from pandas.core.indexes import interval as interval_module
    from pandas.core.indexes import multi as multi_module
    from pandas.core.indexes import period as period_module
    from pandas.core.indexes import range as range_module
    from pandas.core.indexes import timedeltas as timedeltas_module

    # A default RangeIndex (pd.DataFrame of a list or ndarray) takes the Arrow routes too.
    if pd.RangeIndex is range_module.RangeIndex and type(columns) is pd.RangeIndex:
        return _has_canonical_range_columns(columns, range_module)

    # Plain MultiIndex and CategoricalIndex labels (`pd.DataFrame` with tuples or categories)
    # take the Arrow routes too: their labels are only names, as with a default RangeIndex.
    if pd.MultiIndex is multi_module.MultiIndex and type(columns) is pd.MultiIndex:
        return _has_canonical_structured_columns(columns, multi_module, True)
    if (pd.CategoricalIndex is category_module.CategoricalIndex
            and type(columns) is pd.CategoricalIndex):
        return _has_canonical_structured_columns(columns, category_module, False)

    # Plain DatetimeIndex, TimedeltaIndex, PeriodIndex and IntervalIndex labels (`.T` of a
    # time-indexed frame, a pivot of datetimes) take the Arrow routes too: their values are
    # pandas' own and only name the columns.
    if pd.DatetimeIndex is datetimes_module.DatetimeIndex and type(columns) is pd.DatetimeIndex:
        return _has_canonical_label_index(
            columns, _has_canonical_datetimelike_array, datetimes_array_module, "DatetimeArray")
    if (pd.TimedeltaIndex is timedeltas_module.TimedeltaIndex
            and type(columns) is pd.TimedeltaIndex):
        return _has_canonical_label_index(
            columns, _has_canonical_datetimelike_array, timedeltas_array_module, "TimedeltaArray")
    if pd.PeriodIndex is period_module.PeriodIndex and type(columns) is pd.PeriodIndex:
        return _has_canonical_label_index(
            columns, _has_canonical_datetimelike_array, period_array_module, "PeriodArray")
    if pd.IntervalIndex is interval_module.IntervalIndex and type(columns) is pd.IntervalIndex:
        return _has_canonical_label_index(
            columns, _has_canonical_interval_array, interval_array_module, "IntervalArray")

    Index = index_module.Index
    if (pd.Index is not Index or type(columns) is not Index
            or Index.__getattribute__ is not object.__getattribute__
            or "__iter__" in vars(columns)):
        return False

    # Decline overrides without invoking them: PySpark's conversion, which the frame takes
    # then, runs them in its own order.
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
    # Nullable and Arrow-backed labels of other types than strings take the Arrow routes too.
    if _has_canonical_label_array(backing, sys.modules.get("pyarrow")):
        return True

    # Arrow-backed labels already hold PyArrow storage. Do not import optional
    # Arrow modules just to decide whether a fallback input is eligible.
    pa = sys.modules.get("pyarrow")
    if pa is None:
        return False

    try:
        from pandas._libs.arrays import NDArrayBacked
        from pandas.core.arrays import _mixins as mixins_module
        from pandas.core.arrays import base as array_base_module
        from pandas.core.arrays import numpy_ as numpy_module
        from pandas.core.arrays import string_ as string_module
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


def _encodes_as_utf8(text):
    """Whether UTF-8 encodes the str `text`, as a name of an Arrow table does."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    return True


def prepare_int64_table(frame, schema):
    """Return an Arrow table only when each column is already a signed 64-bit integer.

    In particular, never coerce a mixed pandas frame to a common NumPy dtype:
    doing so can round integers above 2**53. Other dtypes and schema conversions
    are left to PySpark's conversion.
    ``verifySchema`` is ignored for pandas Arrow input, as in Spark Connect.
    """
    from types import WrapperDescriptorType

    import numpy as np
    import pandas as pd
    try:
        from pandas.core.arrays.integer import Int64Dtype, IntegerArray
        from pandas.core.arrays.numpy_ import NumpyExtensionArray
        from pandas.core.frame import DataFrame
        from pandas.core.series import Series
    except ImportError:
        # Older supported pandas versions call the NumPy wrapper PandasArray.
        # Leave them to PySpark's conversion when these concrete types are absent.
        return None

    from pyspark._pyspark import (
        LongType,
        StructField,
        StructType,
    )

    # A subclass or custom extension array can change the values exposed to Arrow.
    # Check concrete identities before consulting dtype equality or conversion hooks.
    if pd.DataFrame is not DataFrame or type(frame) is not DataFrame:
        return None

    if not _has_canonical_columns(frame.columns):
        return None
    width = len(frame.columns)
    # The table is named by the labels, which Arrow encodes as UTF-8 when it is made. PySpark's
    # conversion names its table _0, _1, ..., and applies the labels with toDF, whose plan sets
    # them when a request builds it: a label that UTF-8 cannot encode takes that conversion.
    if not width or not frame.columns.is_unique or not all(
        type(name) is str and _encodes_as_utf8(name) for name in frame.columns
    ):
        return None
    dtypes = frame.dtypes
    numpy_dtype = np.dtype("int64")
    if not all((type(dtype) is type(numpy_dtype) and dtype == numpy_dtype)
               or type(dtype) is Int64Dtype
               for dtype in dtypes):
        return None
    # Frames of fewer than 1,000 rows with only NumPy columns are left to the general
    # conversions of pandas, which give the same values; nullable Int64 frames take this path
    # at any size, which keeps their null masks.
    if len(frame) < 1000 and not any(type(dtype) is Int64Dtype for dtype in dtypes):
        return None

    # The table holds the values of the native fields, which PySpark's conversion reads through
    # the classes, of the type names and the converters that it resolves, which must be intact.
    if not _converter_namespaces_intact():
        return None
    for cls in (LongType, StructField, StructType):
        if (cls.__getattribute__ is not object.__getattribute__
                or any("__class__" in vars(base) for base in cls.__mro__ if base is not object)):
            return None
    for cls, names, descriptor_type in (
        (StructType, ("fields", "names"), None),
        (StructType, ("__iter__",), WrapperDescriptorType),
        (StructField, ("name", "dataType", "nullable", "metadata"), None),
    ):
        for name in names:
            # A parameter of a struct or a field is an entry of its `__dict__`, as PySpark's
            # (SPARK-62251), of no attribute of the class, which a patch of it would set.
            if descriptor_type is None:
                if name in vars(cls):
                    return None
                continue
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
            and type(field.name) is str
            for field in fields
        ):
            return None
        names = schema.names
        if len(set(names)) != width:
            return None
    elif schema is not None:
        if type(schema) not in (list, tuple) or len(schema) != width or not all(
            type(name) is str for name in schema
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
        # A null in a non-nullable field is left to the general conversion, which raises
        # PySpark's error for it.
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
    units and submicrosecond truncation are left to PySpark's conversion. Check the
    integer buffers separately so NaT's sentinel never participates in alignment tests.
    """
    from types import FunctionType, WrapperDescriptorType

    import numpy as np
    import pandas as pd
    from pandas.core import frame as frame_module
    from pandas.core import generic as generic_module
    try:
        from pandas.core.arrays.datetimes import DatetimeArray
        from pandas.core.dtypes.dtypes import DatetimeTZDtype
        from pandas.core.frame import DataFrame
        from pandas.core.series import Series
    except ImportError:
        return None
    from pyspark._pyspark import (
        StructField,
        TimestampType,
    )
    from pyspark._pyspark import (
        StructType as NativeStructType,
    )

    if (pd.DataFrame is not DataFrame or type(frame) is not DataFrame
            or type(schema) is not NativeStructType or not len(frame)
            or any(name in vars(frame) for name in ("astype", "notna", "where"))):
        return None
    # A frame that customizes these methods of pandas is left to PySpark's conversion, which
    # reads its columns as PySpark does.
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
    # PySpark's conversion and the Arrow importer resolve the type names and the converters,
    # which must be intact.
    if not _converter_namespaces_intact():
        return None
    for cls in (TimestampType, StructField, NativeStructType):
        if (cls.__getattribute__ is not object.__getattribute__
                or any("__class__" in vars(base) for base in cls.__mro__ if base is not object)):
            return None
    for cls, names, descriptor_type in (
        (NativeStructType, ("fields", "names"), None),
        (NativeStructType, ("__iter__",), WrapperDescriptorType),
        (StructField, ("name", "dataType", "nullable", "metadata"), None),
    ):
        for name in names:
            # A parameter of a struct or a field is an entry of its `__dict__`, as PySpark's
            # (SPARK-62251), of no attribute of the class, which a patch of it would set.
            if descriptor_type is None:
                if name in vars(cls):
                    return None
                continue
            descriptor = vars(cls).get(name)
            if (type(descriptor) is not descriptor_type
                    or descriptor.__objclass__ is not cls or descriptor.__name__ != name):
                return None

    width = len(frame.columns)
    fields = schema.fields
    if not width or len(fields) != width or not frame.columns.is_unique:
        return None
    if not all(type(name) is str for name in frame.columns):
        return None
    if not all(type(field) is StructField and type(field.dataType) is TimestampType
               and type(field.name) is str for field in fields):
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
        # A null in a non-nullable field is left to the general conversion, which raises
        # PySpark's error for it.
        if not fields[i].nullable and column.hasnans:
            return None
        if unit == "ns" and np.any(column.array.asi8[column.notna()] % 1000):
            return None
        columns.append(column)
    # A naive column takes this path on a UTC host only, as `prepare_timestamp_table`
    # requires; on another host, the general conversion takes it in the session time zone.
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
    zone, of which their values are the instants as they are; in another zone,
    PySpark's conversion localizes them in the session zone, with its DST fold policy.
    As that conversion of the column of a timestamp field, each column checks the version
    of pandas (`require_minimum_pandas_version`), which warns of pandas 3.
    """
    import pyarrow as pa

    from pyspark.sql.pandas.utils import require_minimum_pandas_version

    if not timestamp_columns_are_aware(columns):
        if timezone != "UTC" or not _timestamp_host_is_utc():
            return None
    for _ in columns:
        require_minimum_pandas_version()
    return pa.Table.from_arrays(
        [pa.array(column, type=pa.timestamp("us", tz="UTC"), from_pandas=True)
         for column in columns],
        names=schema.names,
    )


def prepare_float64_table(frame, schema):
    """Keep explicit native double columns columnar without changing pandas nulls."""
    from types import FunctionType, WrapperDescriptorType

    import numpy as np
    import pandas as pd
    from pandas.core import frame as frame_module
    from pandas.core import generic as generic_module
    from pandas.core.frame import DataFrame
    from pandas.core.series import Series

    from pyspark import _pyspark as native
    from pyspark.sql import types as sql_types

    # First-use Arrow imports outweigh the conversion saving for smaller frames,
    # which are left to PySpark's conversion.
    if (pd.DataFrame is not DataFrame or type(frame) is not DataFrame
            or type(schema) is not native.StructType or len(frame) < 100000
            or any(name in vars(frame) for name in ("astype", "notna"))):
        return None
    if not _has_canonical_columns(frame.columns):
        return None
    # A frame whose conversion hooks are customized, of its class or instance, is left to
    # PySpark's conversion.
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
        type(name) is str for name in frame.columns
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
    # NumPy subclasses can customize their conversion even inside pandas' concrete
    # NumPy wrapper: they are left to PySpark's conversion, which converts them as
    # PySpark does, at its cost for subclasses such as np.memmap.
    if any(type(column.array._ndarray) is not np.ndarray for column in columns):
        return None
    # The serialization reads Rust fields, bypassing mutable Python accessors, which must
    # keep their behavior.
    for name in ("DoubleType", "StructField", "StructType"):
        cls = getattr(native, name)
        if (vars(sql_types).get(name) is not cls
                or cls.__getattribute__ is not object.__getattribute__
                or any("__class__" in vars(base) for base in cls.__mro__ if base is not object)):
            return None
    for cls, names, descriptor_type in (
        (native.StructType, ("fields", "names"), None),
        (native.StructType, ("__iter__",), WrapperDescriptorType),
        (native.StructField, ("name", "dataType", "nullable", "metadata"), None),
    ):
        for name in names:
            # A parameter of a struct or a field is an entry of its `__dict__`, as PySpark's
            # (SPARK-62251), of no attribute of the class, which a patch of it would set.
            if descriptor_type is None:
                if name in vars(cls):
                    return None
                continue
            descriptor = vars(cls).get(name)
            if (type(descriptor) is not descriptor_type
                    or descriptor.__objclass__ is not cls or descriptor.__name__ != name):
                return None
    # The type names and the converters that PySpark's conversion resolves must be intact.
    if not _converter_namespaces_intact():
        return None
    fields = schema.fields
    if len(fields) != width or not all(
        type(field) is native.StructField and type(field.dataType) is native.DoubleType
        and type(field.name) is str for field in fields
    ) or len(set(schema.names)) != width:
        return None

    # Use pandas' missing-value semantics, as PySpark's conversion does (`isnull`), including
    # the legacy use_inf_as_na option. Infinities and signed zero otherwise stay intact.
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
        # metadata remains in the explicit logical schema, not Arrow metadata. The names are
        # PySpark's of its table, `schema.names`, which of a `fields` list changed in place are
        # the names before the change.
        schema=pa.schema([pa.field(name, pa.float64(), nullable=field.nullable)
                          for name, field in zip(schema.names, fields)]),
    )


def prepare_float64_input(frame, schema):
    """Serialize supported double columns as an inline Arrow IPC stream, which the session sends
    inline below the cache threshold."""
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
    Those cases are left to PySpark's conversion, with its values and errors.
    """
    from types import MethodDescriptorType, WrapperDescriptorType

    import numpy as np
    import pandas as pd
    from pandas.core import frame as frame_module
    from pandas.core import generic as generic_module
    from pandas.core.frame import DataFrame
    from pandas.core.series import Series

    from pyspark._pyspark import StringType, StructField
    from pyspark._pyspark import StructType as NativeStructType
    from pyspark.sql import types as sql_types

    # Small frames are left to PySpark's conversion, and so are frames whose normalization
    # methods are customized: the cells must be the frame's own.
    if (pd.DataFrame is not DataFrame or type(frame) is not DataFrame
            or type(schema) is not NativeStructType
            or len(frame) < 10000
            or any(name in vars(frame) for name in ("astype", "notna", "where"))):
        return None
    for name, module in (("astype", generic_module), ("notna", frame_module),
                         ("where", generic_module)):
        method = getattr(DataFrame, name)
        if not _defined_in(method, name, module):
            return None
    if (DataFrame.astype is not generic_module.NDFrame.astype
            or DataFrame.where is not generic_module.NDFrame.where):
        return None
    # The encoding reads native fields directly. Class attributes are mutable, so
    # customized Python schema access is left to the general conversion.
    for name, cls in (("StringType", StringType), ("StructField", StructField),
                      ("StructType", NativeStructType)):
        if (getattr(sql_types, name, None) is not cls
                or cls.__getattribute__ is not object.__getattribute__
                or any("__class__" in vars(base) for base in cls.__mro__ if base is not object)):
            return None
    for cls, names, descriptor_type in (
        (NativeStructType, ("fields", "names"), None),
        (NativeStructType, ("__iter__",), WrapperDescriptorType),
        (StructField, ("name", "dataType", "nullable", "metadata"), None),
        (StringType, ("isUTF8BinaryCollation",), MethodDescriptorType),
    ):
        for name in names:
            # A parameter of a struct or a field is an entry of its `__dict__`, as PySpark's
            # (SPARK-62251), of no attribute of the class, which a patch of it would set.
            if descriptor_type is None:
                if name in vars(cls):
                    return None
                continue
            descriptor = vars(cls).get(name)
            if (type(descriptor) is not descriptor_type
                    or descriptor.__objclass__ is not cls or descriptor.__name__ != name):
                return None
    if not _has_canonical_columns(frame.columns):
        return None
    # The type names and the converters that PySpark's conversion resolves must be intact.
    if not _converter_namespaces_intact():
        return None
    width = len(frame.columns)
    if not width or len(schema.fields) != width or not frame.columns.is_unique:
        return None
    if not all(type(name) is str for name in frame.columns):
        return None
    if not all(
        type(field) is StructField and type(field.dataType) is StringType
        and field.dataType.isUTF8BinaryCollation()
        and type(field.name) is str
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
    # The IPC is of the names of the fields, where PySpark's table is of `schema.names`, which of
    # a `fields` list changed in place are the names before the change: that takes PySpark's
    # conversion.
    if list(schema.names) != [field.name for field in schema.fields]:
        return None
    return _pandas_string_ipc(arrays, schema)


def _arrow_string_table(frame, schema, dtypes):
    """The columns of a frame kept in pandas' Arrow string storage (string[pyarrow], and the
    str dtype of pandas 3), as an Arrow table of the schema, or None.

    PySpark's conversion takes such a column from that storage (`__arrow_array__`), with its
    missing values as nulls, so the storage holds the same values. Only pandas' own array
    classes are trusted with their original conversion hooks: customization could convert
    differently, which the general conversion does as PySpark's.
    The caller's guards of prepare_string_input (frame, schema, names, size) have passed.
    """
    from pandas.core.arrays import string_ as string_module

    if not all(type(dtype) is string_module.StringDtype
               and dtype.storage in ("pyarrow", "pyarrow_numpy")
               for dtype in dtypes):
        return None
    # The importer uses mutable public/captured schema names. Decline a rebound
    # namespace before conversion or an upload, as the generic route does.
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
        if len(values) != len(frame):
            return None
        if values.type not in (pa.string(), pa.large_string()):
            return None
        try:
            # Malformed offsets or UTF-8 are left to the general conversion and its error.
            values.validate(full=True)
            # Check normal-string capacity before an upload. The importer
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
    Those cases are left to PySpark's conversion, with its values and errors.
    """
    from datetime import date
    from types import FunctionType, WrapperDescriptorType

    import numpy as np
    import pandas as pd
    from pandas.core import frame as frame_module
    from pandas.core import generic as generic_module
    from pandas.core.frame import DataFrame
    from pandas.core.series import Series

    from pyspark._pyspark import (
        DateType,
        StructField,
    )
    from pyspark._pyspark import (
        StructType as NativeStructType,
    )
    from pyspark.sql import types as sql_types

    # Frames of fewer than 10,000 rows are left to the general conversion of the schema,
    # which gives the same values.
    if (pd.DataFrame is not DataFrame or type(frame) is not DataFrame
            or type(schema) is not NativeStructType
            or len(frame) < 10000
            or any(name in vars(frame) for name in ("astype", "notna", "where"))):
        return None
    if not _has_canonical_columns(frame.columns):
        return None
    # A frame whose normalization methods are customized, of its class or instance, is left
    # to PySpark's conversion.
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
    # Earlier converter branches must not recognize a date as another type: the type names and
    # the converters that PySpark's conversion and the importer resolve must be intact.
    if not _converter_namespaces_intact():
        return None
    # The table reads native fields directly. Class attributes are mutable, so
    # customized Python schema access is left to the general conversion.
    for name, cls in (("DateType", DateType), ("StructField", StructField),
                      ("StructType", NativeStructType)):
        if (getattr(sql_types, name, None) is not cls
                or cls.__getattribute__ is not object.__getattribute__
                or any("__class__" in vars(base) for base in cls.__mro__ if base is not object)):
            return None
    for cls, names, descriptor_type in (
        (NativeStructType, ("fields", "names"), None),
        (NativeStructType, ("__iter__",), WrapperDescriptorType),
        (StructField, ("name", "dataType", "nullable", "metadata"), None),
    ):
        for name in names:
            # A parameter of a struct or a field is an entry of its `__dict__`, as PySpark's
            # (SPARK-62251), of no attribute of the class, which a patch of it would set.
            if descriptor_type is None:
                if name in vars(cls):
                    return None
                continue
            descriptor = vars(cls).get(name)
            if (type(descriptor) is not descriptor_type
                    or descriptor.__objclass__ is not cls or descriptor.__name__ != name):
                return None
    width = len(frame.columns)
    if not width or len(schema.fields) != width or not frame.columns.is_unique:
        return None
    if not all(type(name) is str for name in frame.columns):
        return None
    if not all(
        type(field) is StructField and type(field.dataType) is DateType
        and type(field.name) is str
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
        # A null in a non-nullable field is left to the general conversion, which raises
        # PySpark's error for it.
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
    """Whether each public name of a native type class in `namespace` (of `DataType` and its
    subclasses, `StructField` too), which the conversions resolve, is still the native class. A
    name of another native class, such as the classic `SparkSession` of a module, is not checked."""
    from pyspark import _pyspark as native

    return all(
        namespace[name] is value
        for name, value in vars(native).items()
        if not name.startswith("_") and isinstance(value, type)
        and issubclass(value, native.DataType) and name in namespace
    )


# The functions of `pyspark.sql.pandas.types` that PySpark's conversion of a pandas DataFrame calls
# by their names, and those of `pyspark.sql.pandas.conversion` that it calls so.
_CONVERTER_FUNCTIONS = ("_deduplicate_field_names", "_dedup_names", "to_arrow_schema",
                        "to_arrow_type", "to_arrow_metadata", "from_arrow_type",
                        "_create_converter_from_pandas",
                        "_check_series_convert_timestamps_internal")
_CONVERSION_FUNCTIONS = ("create_arrow_table_from_pandas", "create_arrow_array_from_pandas")


def _own_functions_intact(namespace, names):
    """Whether the module namespace `namespace` holds the native classes, and, as each of `names`,
    a function of its own of that name, which no rebinding replaced, or the cache of one
    (`functools.lru_cache`, as of `_create_converter_from_pandas`)."""
    from functools import _lru_cache_wrapper
    from types import FunctionType

    def own(name, function):
        if type(function) is _lru_cache_wrapper:
            function = function.__wrapped__
        return (type(function) is FunctionType and function.__name__ == name
                and function.__globals__ is namespace)

    return _native_names_intact(namespace) and all(
        own(name, namespace.get(name)) for name in names
    )


def _session_module():
    """The module of the session, `pyspark.sql.connect.session`, which binds the conversions of
    createDataFrame and the types that they make as PySpark Connect's module binds them, when it
    is imported (see it): PySpark's createDataFrame calls them by their names there."""
    import sys

    module = sys.modules.get("pyspark.sql.connect.session")
    if module is None:
        import pyspark.sql.connect.session as module
    return module


def _session_conversions_intact():
    """Whether the conversions and the types that the module of the session binds, which PySpark's
    conversion of a pandas DataFrame calls and makes, are the native classes and the converters'
    own functions, of modules that hold the native classes too."""
    from types import FunctionType

    namespace = vars(_session_module())
    if not _native_names_intact(namespace):
        return False
    for name in ("to_arrow_schema", "_deduplicate_field_names", "from_arrow_type",
                 "create_arrow_table_from_pandas"):
        function = namespace.get(name)
        if type(function) is not FunctionType or function.__name__ != name:
            return False
        names = (_CONVERSION_FUNCTIONS if name == "create_arrow_table_from_pandas"
                 else _CONVERTER_FUNCTIONS)
        if not _own_functions_intact(function.__globals__, names):
            return False
    return True


def _converter_namespaces_intact():
    """Whether the type names and the converters that PySpark's conversion of a pandas DataFrame
    resolves, and the importer of the fast paths' tables, are the native classes and the
    converters' own functions: in `pyspark.sql.types`, of which a converter module imported anew
    binds them; in the converter module of `sys.modules`, whose converters the conversion of a
    column imports when it is called; in the module of the session and the modules of the
    conversions that it bound when it was imported (`_session_conversions_intact`); and in the
    importer. A fast path declines a frame otherwise, which then takes PySpark's conversion, of
    the names as they are."""
    import sys

    from pyspark.sql import types as sql_types

    if sys.modules.get("pyspark.sql.types") is not sql_types or not _native_names_intact(
        vars(sql_types)
    ):
        return False
    converter = sys.modules.get("pyspark.sql.pandas.types")
    if converter is not None and not _own_functions_intact(vars(converter), _CONVERTER_FUNCTIONS):
        return False
    if not _session_conversions_intact():
        return False
    # The importer resolves its schema classes by name, and its conversion functions may
    # keep the globals of an older type module.
    importer = sys.modules.get("pyspark.sql._arrow_input")
    if importer is not None:
        from pyspark._pyspark import AtomicType, StructType

        if not _native_names_intact(vars(importer)) or any(
            vars(importer).get(name) is not cls
            for name, cls in (("StructType", StructType), ("AtomicType", AtomicType))
        ):
            return False
        for function in list(vars(importer).values()):
            namespace = getattr(function, "__globals__", None)
            if type(namespace) is dict and not _native_names_intact(namespace):
                return False
        # The importer calls its converters by these names, which must still be functions.
        for name in ("to_arrow_schema", "from_arrow_schema", "_check_arrow_table_timestamps_localize"):
            if name in vars(importer) and (
                getattr(vars(importer)[name], "__globals__", None) is None
                or (converter is not None
                    and vars(importer)[name] is not getattr(converter, name, None))
            ):
                return False
    return True


_STRUCT_READS = ("fields", "names", "__iter__")
_FIELD_READS = ("name", "dataType", "nullable", "metadata")


def _reads_as(cls, base, names):
    """Whether `cls`, `base` or a Python subclass of it, reads `names` as `base` does: whether no
    class before `base` in its MRO defines one of them, `__getattribute__` or `__class__`."""
    for klass in cls.__mro__:
        if klass is base:
            return True
        namespace = vars(klass)
        if ("__getattribute__" in namespace or "__class__" in namespace
                or any(name in namespace for name in names)):
            return False
    return False


def _native_struct_classes_intact():
    """Whether the native `StructType` and `StructField` keep their own reads: no class of them, or
    a base, defines `__getattribute__` or `__class__`, the `__iter__` of `StructType` is its own,
    and neither class has an attribute of the class over its parameters (`fields` and `names`;
    `name`, `dataType`, `nullable` and `metadata`), which are entries of the `__dict__` of its
    objects."""
    from types import WrapperDescriptorType

    from pyspark._pyspark import StructField, StructType

    for cls in (StructType, StructField):
        if (cls.__getattribute__ is not object.__getattribute__
                or any("__class__" in vars(base) for base in cls.__mro__ if base is not object)):
            return False
    for cls, names, descriptor_type in (
        (StructType, ("fields", "names"), None),
        (StructType, ("__iter__",), WrapperDescriptorType),
        (StructField, ("name", "dataType", "nullable", "metadata"), None),
    ):
        for name in names:
            # A parameter of a struct or a field is an entry of its `__dict__`, as PySpark's
            # (SPARK-62251), of no attribute of the class, which a patch of it would set.
            if descriptor_type is None:
                if name in vars(cls):
                    return False
                continue
            descriptor = vars(cls).get(name)
            if (type(descriptor) is not descriptor_type
                    or descriptor.__objclass__ is not cls or descriptor.__name__ != name):
                return False
    return True


def _canonical_struct(schema):
    """Whether a native struct schema and its fields keep their own classes' behavior.

    The conversion reads the fields, names and data types through the classes, and the relation
    of the core the native types that they hold, so a frame under a class that overrides one of
    these reads takes the row path, which converts it with PySpark's conversion too, of the
    relation of PySpark's `json()` of the struct, which reads them as the conversion does
    (`create_from_pandas_table` of the session). As PySpark's createDataFrame, which converts a
    frame under any StructType of StructFields with its conversion, a Python subclass that
    overrides none takes this route too.
    """
    from pyspark._pyspark import StructField, StructType

    if not isinstance(schema, StructType) or not _reads_as(type(schema), StructType, _STRUCT_READS):
        return False
    if not _native_struct_classes_intact():
        return False
    return all(
        isinstance(field, StructField) and _reads_as(type(field), StructField, _FIELD_READS)
        and type(field.name) is str
        and type(field.dataType).__getattribute__ is object.__getattribute__
        for field in schema.fields
    )


def accepts_pandas_schema(schema=None):
    """Whether the session converts a pandas frame of rows under this schema, a DDL string or a
    struct type, with PySpark's conversion (`prepare_pandas_table`), of the relation of the struct
    that the core lowers: of a struct whose classes, and its fields', keep their own reads
    (`_canonical_struct`). The row path converts a frame under another struct with PySpark's
    conversion too, where PyArrow is, of the relation of PySpark's `json()` of the struct.
    """
    return schema is None or type(schema) is str or _canonical_struct(schema)


def prepare_pandas_table(frame, schema, timezone, safecheck, prefers_large_types):
    """Convert a pandas frame under an explicit struct schema as PySpark Connect does: the table,
    and the names that its `toDF` applies, `schema.fieldNames()`, of strs.

    As PySpark's createDataFrame, in its order, of the conversions and the types that its module
    of the session binds (`_session_module`), so a rebound name or a patched class changes what it
    changes in PySpark, of its errors too. Each column goes through PySpark's own Arrow
    conversion, with the session time zone and the safe-cast setting, and the table is then
    renamed and cast to the schema's Arrow types. The values and errors therefore match PySpark,
    including the nullability and overflow messages. As PySpark's, the columns are zipped with the
    fields: the columns past the fields are not converted and are dropped, of no fields the table
    has no column and no row, and of fewer columns than the fields the rename raises PyArrow's
    `ArrowInvalid`. A frame of no columns is the table of `pa.RecordBatch.from_pandas`, which
    keeps its number of rows, with a column of each level of its index unless it is a
    RangeIndex; the caller checks the number of columns of the table against the number of
    fields, as PySpark does.
    """
    import pyarrow as pa

    session = _session_module()
    # As PySpark's, the table is of the names of `_deduplicate_field_names`, of the top level too
    # (`a_0`, `a_1`), which the relation's schema has as given, and `toDF` applies.
    deduped_schema = session._deduplicate_field_names(schema)
    spark_types = [field.dataType for field in deduped_schema.fields]
    arrow_schema = session.to_arrow_schema(
        deduped_schema, timezone="UTC", prefers_large_types=prefers_large_types
    )
    names = [str(x) if not isinstance(x, str) else x for x in schema.fieldNames()]
    if len(frame.columns) == 0:
        table = pa.Table.from_batches([pa.RecordBatch.from_pandas(frame)])
    else:
        table = session.create_arrow_table_from_pandas(
            [(c, st) for (_, c), st in zip(frame.items(), spark_types)],
            timezone=timezone,
            safecheck=safecheck,
            prefers_large_types=prefers_large_types,
        )
    # The names of `_deduplicate_field_names`, made again, of a `fields` list changed in place of
    # its cached `names`, as many as the fields, of which fewer raise its `IndexError`.
    if len(schema.fields) > 0:
        table = table.rename_columns(session._deduplicate_field_names(schema).names).cast(
            arrow_schema
        )
    return table, names


def arrow_decimal_column(values, dtype, name):
    """The precision, scale and values of a pandas object column of Decimals and other values,
    as Arrow types and converts it: PySpark's conversion of pandas leaves such a column to
    Arrow, which takes ints as Decimals and raises for the others."""
    import pyarrow as pa

    from pyspark.errors import PySparkTypeError, PySparkValueError

    message = (f"Exception thrown when converting pandas.Series ({dtype}) with name '{name}' to Arrow "
               "Array (None).")
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
    """Whether pyarrow can be imported, checked without importing it. Without it, the row path
    converts every pandas frame: it does not need PyArrow, which PySpark's conversion and the
    fast paths of plain frames do."""
    import importlib.util

    try:
        return importlib.util.find_spec("pyarrow") is not None
    except (ImportError, ValueError):
        return False


def localized_timestamp_cells(frame, index, timezone):
    """The cells of the naive datetime64 column `index` of a frame that the row path converts,
    as the UTC Timestamps of their instants, and None of NaT.

    PySpark's conversion takes the values of such a column as times of the session time zone,
    `timezone`, of a timestamp field or of no schema (`_check_series_convert_timestamps_internal`,
    of the standard time of an ambiguous time, and of the errors of pandas for a time that the
    zone skips and for a zone that it does not know), whatever the zone of the host, and as wall
    clocks of a timestamp_ntz field, of no `timezone`.
    """
    from pyspark.sql.pandas.types import _check_series_convert_timestamps_internal

    column = frame.iloc[:, index]
    if timezone is None:
        # The wall clocks, as the instants of the same clocks of UTC.
        localized = column.dt.tz_localize("UTC")
    else:
        localized = _check_series_convert_timestamps_internal(column, timezone)
    return localized.astype(object).where(localized.notna(), None).tolist()


def prepare_inferred_pandas_table(frame, timezone, safecheck, prefers_large_types,
                                  infer_dict_as_map, prefer_timestamp_ntz):
    """Convert a pandas frame without a schema, or of names, as PySpark Connect does.

    As PySpark's createDataFrame, in its order, of the conversions and the types that its module
    of the session binds (`_session_module`). Each column goes through PySpark's own Arrow
    conversion, of the type that Arrow infers, except that datetimes are timestamps, of the
    session time zone, and timedeltas day-time intervals, for the server to infer the types of the
    table. With `inferPandasDictAsMap` (`infer_dict_as_map`), the frame converts under the struct
    of the types that Arrow infers, where a column of dicts is a map, and the table takes the
    names and the Arrow types of the struct. Returns the table and the names of that struct, of
    strs, or None.
    """
    import pandas as pd
    import pyarrow as pa
    from pandas.api.types import is_datetime64_dtype, is_timedelta64_dtype

    from pyspark.errors import PySparkValueError

    session = _session_module()
    struct = None
    if infer_dict_as_map:
        struct = session.StructType()
        for field in pa.Schema.from_pandas(frame):
            field_type = field.type
            if isinstance(field_type, pa.StructType):
                if len(field_type) == 0:
                    raise PySparkValueError(
                        errorClass="CANNOT_INFER_EMPTY_SCHEMA",
                        messageParameters={},
                    )
                arrow_type = field_type.field(0).type
                spark_type = session.MapType(
                    session.StringType(),
                    session.from_arrow_type(arrow_type, prefer_timestamp_ntz),
                )
            else:
                spark_type = session.from_arrow_type(field_type, prefer_timestamp_ntz)
            struct.add(field.name, spark_type, nullable=field.nullable)
    names = None
    if struct is not None:
        deduped_schema = session._deduplicate_field_names(struct)
        spark_types = [field.dataType for field in deduped_schema.fields]
        arrow_schema = session.to_arrow_schema(
            deduped_schema, timezone="UTC", prefers_large_types=prefers_large_types
        )
        names = [str(x) if not isinstance(x, str) else x for x in struct.fieldNames()]
    else:
        spark_types = [
            (
                session.TimestampType()
                if is_datetime64_dtype(t) or isinstance(t, pd.DatetimeTZDtype)
                else session.DayTimeIntervalType()
                if is_timedelta64_dtype(t)
                else None
            )
            for t in frame.dtypes
        ]
    if len(frame.columns) == 0:
        # pa.RecordBatch.from_pandas keeps the number of rows of a frame of no columns.
        table = pa.Table.from_batches([pa.RecordBatch.from_pandas(frame)])
    else:
        table = session.create_arrow_table_from_pandas(
            [(c, st) for (_, c), st in zip(frame.items(), spark_types)],
            timezone=timezone,
            safecheck=safecheck,
            prefers_large_types=prefers_large_types,
        )
    if struct is not None and len(struct.fields) > 0:
        table = table.rename_columns(session._deduplicate_field_names(struct).names).cast(
            arrow_schema
        )
    return table, names
