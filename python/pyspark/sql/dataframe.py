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
# The public DataFrame base and its concrete Rust-backed implementation.
#
# As in PySpark, ``pyspark.sql.DataFrame`` is a Python base/factory, whereas
# ``pyspark.sql.connect.dataframe.DataFrame`` is the concrete remote DataFrame.  Keeping
# the public base in Python lets user-defined subclasses allocate their own Python state
# with ``object.__new__``.  A concrete frame must still be allocated by the native class:
# its Rust payload cannot be initialized by Python's object allocator.

from abc import ABCMeta
from functools import cached_property, wraps
from inspect import signature
from types import (
    ClassMethodDescriptorType,
    FunctionType,
    GetSetDescriptorType,
    MethodDescriptorType,
    WrapperDescriptorType,
)
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Tuple, Union, overload

from pyspark import _NoValue
from pyspark._globals import _NoValueType
from pyspark._pyspark import (
    DataFrame as _NativeDataFrame,
)
from pyspark.sql._classic_bases import _ClassicMeta

# PySpark's docstrings of the members of its classic DataFrame that differ from those of the native
# class, which are those that `inspect.getdoc` and `help()` find of PySpark's Connect DataFrame
# (scripts/gen_connect_docstrings.py): the public base takes them in their place.
from pyspark.sql._classic_docs_generated import DATAFRAME as _classic_docstrings

# PySpark's, of the annotations of the classes of this module, as its module imports them.
from pyspark.sql.column import Column
from pyspark.sql.table_arg import TableArg

# Of PySpark's pyspark.sql.dataframe, whose namespace the official doctests of the DataFrame take
# as their globals (pyspark.sql.connect.dataframe._test), and whose examples use these names.
from pyspark.sql.types import Row, StructType  # noqa: F401
from pyspark.storagelevel import StorageLevel  # noqa: F401

if TYPE_CHECKING:
    # The names of PySpark's annotations below, as its module imports them.
    from py4j.java_gateway import JavaObject

    from pyspark.core.context import SparkContext
    from pyspark.core.rdd import RDD
    from pyspark.sql._typing import LiteralType, OptionalPrimitiveType
    from pyspark.sql.context import SQLContext
    from pyspark.sql.pandas._typing import DataFrameLike as PandasDataFrameLike
    from pyspark.sql.session import SparkSession

__all__ = ["DataFrame", "DataFrameNaFunctions", "DataFrameStatFunctions"]

_public_dataframe = None
_initial_public_attributes = {}
_native_overrides = {}
_missing = object()


# As PySpark Connect's `DataFrame.isStreaming`, a `functools.cached_property` of the class body of
# its pyspark.sql.connect.dataframe: the first access keeps the answer in the `__dict__` of the
# DataFrame, which the next ones read, of the request of the native class. It is installed on
# the native class here, before the public base reads the attributes of that class.
def isStreaming(self: Any) -> bool:
    return self._is_streaming()


isStreaming.__qualname__ = "DataFrame.isStreaming"
isStreaming.__module__ = "pyspark.sql.connect.dataframe"
if not isinstance(vars(_NativeDataFrame).get("isStreaming"), cached_property):
    _cached_is_streaming = cached_property(isStreaming)
    _cached_is_streaming.__set_name__(_NativeDataFrame, "isStreaming")
    _NativeDataFrame.isStreaming = _cached_is_streaming
    del _cached_is_streaming
del isStreaming

# The native class's own attributes, which a patch of the public base may replace.
_native_originals = dict(vars(_NativeDataFrame))

# The undecorated, empty-bodied public methods in official PySpark 4.3.0
# pyspark/sql/dataframe.py, which the class below defines as PySpark does. Re-derive this list on
# every PySpark version bump.
_empty_public_methods = frozenset({
    "checkpoint", "localCheckpoint", "lateralJoin", "nearestByJoin", "toPandas",
    "asTable", "scalar", "exists",
})

# The names that PySpark 4.3.0's pyspark.sql.connect.dataframe.DataFrame defines itself (its
# vars()): every name of the public base, but __arrow_c_stream__, _preapare_args_for_sample and
# _preapare_cols_for_sort, which it inherits, and helpers of its own. Its frames find these
# first, so a patch of one on the public base does not reach them; an explicit
# DataFrame.<name>(df) reads it. Re-derive this list on every PySpark version bump.
_connect_names = frozenset({
    "__dir__", "__doc__", "__getattr__", "__getitem__", "__init__", "__module__", "__new__",
    "__reduce__", "__repr__", "_check_same_session", "_col", "_explain_string",
    "_get_binary_as_bytes", "_ipython_key_completions_", "_joinAsOf", "_map_partitions",
    "_merge_cached_schema", "_repr_html_", "_schema", "_show_string", "_to_cols", "_to_pandas",
    "_to_table", "agg", "alias", "approxQuantile", "asTable", "cache", "checkpoint", "coalesce",
    "colRegex", "collect", "columns", "corr", "count", "cov", "createGlobalTempView",
    "createOrReplaceGlobalTempView", "createOrReplaceTempView", "createTempView", "crossJoin",
    "crosstab", "cube", "describe", "distinct", "drop", "dropDuplicates",
    "dropDuplicatesWithinWatermark", "drop_duplicates", "dropna", "dtypes", "exceptAll",
    "executionInfo", "exists", "explain", "fillna", "filter", "first", "foreach",
    "foreachPartition", "freqItems", "groupBy", "groupby", "groupingSets", "head", "hint",
    "inputFiles", "intersect", "intersectAll", "isEmpty", "isLocal", "isStreaming", "is_cached",
    "join", "lateralJoin", "limit", "localCheckpoint", "mapInArrow", "mapInPandas", "melt",
    "mergeInto", "metadataColumn", "na", "nearestByJoin", "observe", "offset", "orderBy",
    "pandas_api", "persist", "plot", "printSchema", "randomSplit", "registerTempTable",
    "repartition", "repartitionById", "repartitionByRange", "replace", "rollup",
    "sameSemantics", "sample", "sampleBy", "scalar", "schema", "select", "selectExpr",
    "semanticHash", "show", "sort", "sortWithinPartitions", "sparkSession", "stat",
    "storageLevel", "subtract", "summary", "tail", "take", "to", "toArrow", "toDF", "toJSON",
    "toLocalIterator", "toPandas", "transform", "transpose", "union", "unionAll", "unionByName",
    "unpersist", "unpivot", "where", "withColumn", "withColumnRenamed", "withColumns",
    "withColumnsRenamed", "withMetadata", "withWatermark", "write", "writeStream", "writeTo",
    "zip", "zipWithIndex",
})

# The attributes of PySpark's public base that its Connect DataFrame inherits, and that the
# public base here has: the native class's own stands for each.
_inherited_base_attributes = frozenset({"__arrow_c_stream__"})

# The attributes that the __init__ of PySpark's Connect DataFrame sets on each frame. They hide
# a patch of the public base from its frames, unless the patch is a data descriptor, such as a
# property, which Python's attribute lookup takes before the attributes of an instance.
_connect_frame_attributes = frozenset({
    "_plan", "_session", "_support_repr_html", "_cached_schema", "_cached_schema_serialized",
    "_execution_info",
})


class _NativeOverride(NamedTuple):
    previous_native: object
    forwarded: object


# These describe the Python class's layout, instance attribute protocols or
# import/ABC bookkeeping. They belong to the public base and must not overwrite
# the concrete type's own behavior or metadata.
_local_class_attributes = frozenset({
    "__new__", "__init__", "__name__", "__module__", "__qualname__", "__doc__", "__dict__",
    "__weakref__", "__bases__", "__base__", "__mro__", "__slots__", "__class__",
    "__annotations__", "__abstractmethods__", "__classcell__", "__parameters__",
    "__type_params__", "__init_subclass__", "__subclasshook__", "__class_getitem__",
    "__orig_bases__", "__getattribute__", "__setattr__", "__delattr__", "_abc_impl",
})


def _set_native(name, value):
    if value is _missing:
        if name in vars(_NativeDataFrame):
            delattr(_NativeDataFrame, name)
    else:
        setattr(_NativeDataFrame, name, value)


def _native_entries(name):
    """The native class's entry of `name`, and its own: the one that a patch of the public base,
    forwarded to it and still in place, replaced."""
    current = vars(_NativeDataFrame).get(name, _missing)
    override = _native_overrides.get(name)
    if override is not None and current is override.forwarded:
        return current, override.previous_native
    return current, current


def _is_data_descriptor(value):
    kind = type(value)
    return hasattr(kind, "__set__") or hasattr(kind, "__delete__")


class _DataFrameMeta(ABCMeta):
    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        if _public_dataframe is not None and type(cls) is _DataFrameMeta:
            # A by-value cloudpickled user subclass must name a metaclass that
            # official workers can import. Only the public base needs forwarding;
            # ordinary subclasses retain standard ABC behavior. Preserve explicit
            # user metaclass subclasses, which own their construction semantics.
            type.__setattr__(cls, "__class__", ABCMeta)

    def __setattr__(cls, name, value):
        if cls is not _public_dataframe or name in _local_class_attributes:
            return super().__setattr__(name, value)

        # The native class stands for PySpark's Connect class, which inherits from the public
        # base: a patch of the base reaches the frames where it reaches PySpark's, as a new
        # helper does. It does not where PySpark's Connect class defines the name, as count or
        # __getattr__, or where the concrete class has a patch of its own: either hides it.
        # Restoring an initial public dispatcher restores the native attribute, and never
        # installs the dispatcher itself on the native type, which would make dynamic dispatch
        # recursive.
        current, own = _native_entries(name)
        original = _native_originals.get(name, _missing)
        if name in _connect_names or (own is not original and own is not _missing):
            seen, target = False, own
        else:
            # Of an attribute that PySpark's frames set on themselves, they see a data
            # descriptor only.
            seen = value is not _initial_public_attributes.get(name, _missing) and (
                name not in _connect_frame_attributes or _is_data_descriptor(value)
            )
            target = value if seen else original
        try:
            if target is not current:
                _set_native(name, target)
            super().__setattr__(name, value)
        except BaseException:
            if target is not current:
                _set_native(name, current)
            raise
        if seen:
            _native_overrides[name] = _NativeOverride(previous_native=own, forwarded=value)
        else:
            _native_overrides.pop(name, None)

    def __delattr__(cls, name):
        if (
            cls is not _public_dataframe
            or name in _local_class_attributes
            or name not in vars(cls)
        ):
            return super().__delattr__(name)

        # Deleting a forwarded patch restores the native attribute that it replaced, and keeps
        # a patch that the concrete class made since. PySpark's Connect frames have none of an
        # attribute that they inherit from the base without it.
        current, own = _native_entries(name)
        target = own
        if name in _inherited_base_attributes and own is _native_originals.get(name, _missing):
            target = _missing
        try:
            if target is not current:
                _set_native(name, target)
            super().__delattr__(name)
        except BaseException:
            if target is not current:
                _set_native(name, current)
            raise
        if target is own:
            _native_overrides.pop(name, None)
        else:
            _native_overrides[name] = _NativeOverride(previous_native=own, forwarded=target)


def _concrete_attribute(name, original):
    # As PySpark's dispatch_df_method, getattr(ConnectDataFrame, name) at the call: the concrete
    # class's attribute, or its patch, of a name that PySpark's Connect class defines. Its frames
    # inherit the others from the public base, whose own implementation the native class's
    # original stands for. So a chain of wrappers ends at the implementation, as in PySpark:
    # with f, a wrapper of the original DataFrame.__copy__, set as DataFrame.__copy__, and then
    # g, a wrapper of f, set as ConnectDataFrame.__copy__, a copy calls g, f and the original.
    if name not in _connect_names:
        return original
    current = vars(_NativeDataFrame).get(name, _missing)
    if current is _missing:
        return original
    get = getattr(type(current), "__get__", None)
    return current if get is None else get(current, None, _NativeDataFrame)


def _not_implemented(name):
    from pyspark.errors import PySparkNotImplementedError

    # PySpark dispatches public-base methods to the concrete implementation. An
    # independent Python subclass supplies the operations for its own data model.
    return PySparkNotImplementedError(
        errorClass="NOT_IMPLEMENTED",
        messageParameters={"feature": f"DataFrame.{name}"},
    )


def _docstring(name, attribute):
    """The docstring of the member `name` of the public base, PySpark's of its classic DataFrame:
    that of the native class's `attribute`, or of `_classic_docstrings` where they differ."""
    return _classic_docstrings.get(name, attribute.__doc__)


def _dispatch_method(name, method):
    @wraps(method)
    def dispatched(self, *args, **kwargs):
        if not isinstance(self, _NativeDataFrame):
            if name == "__arrow_c_stream__":
                from pyspark.sql.interchange import SparkArrowCStreamer

                arguments = signature(method).bind(self, *args, **kwargs)
                return SparkArrowCStreamer(self).__arrow_c_stream__(
                    arguments.arguments.get("requested_schema")
                )
            raise _not_implemented(name)
        implementation = _concrete_attribute(name, method)
        if implementation is dispatched:
            implementation = method
        return implementation(self, *args, **kwargs)

    dispatched.__module__ = __name__
    dispatched.__qualname__ = f"DataFrame.{name}"
    dispatched.__doc__ = _docstring(name, method)
    return dispatched


def _empty_property(name, descriptor):
    # PySpark's public-base properties have empty bodies and no dispatch decorator: read through
    # the base, of any frame, they are None. A frame reads its own class's property, and an
    # independent Python subclass reads None unless it overrides them.
    def get(self):
        return None

    get.__name__ = name
    get.__module__ = __name__
    get.__qualname__ = f"DataFrame.{name}"
    return property(get, doc=_docstring(name, descriptor))


class DataFrame(metaclass=_DataFrameMeta):
    __doc__ = _classic_docstrings.get("", _NativeDataFrame.__doc__)
    # PySpark's annotations of the attributes of its frames, which make no attribute.
    _sql_ctx: Optional["SQLContext"]
    _session: "SparkSession"
    _sc: "SparkContext"
    _jdf: "JavaObject"
    is_cached: bool
    _schema: Optional[StructType]
    _lazy_rdd: Optional["RDD[Row]"]
    _support_repr_html: bool

    def __new__(cls, plan, session):
        # Retain this client's Connect constructor, including its keyword arguments.
        # A Python subclass can override __new__ to use object.__new__(cls), as the
        # reference client's MockDataset does, without allocating any Rust storage.
        # As in the reference public factory, inheriting this __new__ still returns
        # a concrete frame, rather than allocating an instance of the subclass.
        return _NativeDataFrame(plan, session)

    def __getattr__(self, name: str) -> Column:
        if not isinstance(self, _NativeDataFrame):
            raise _not_implemented("__getattr__")
        # An explicit __getattr__ call resolves a column even when a regular
        # attribute has that name. PyO3's resolver has no class-dict entry until a
        # user patches the concrete class, so use its ordinary native bridge as the default.
        implementation = _concrete_attribute(
            "__getattr__", _NativeDataFrame._get_column_by_attribute
        )
        if implementation is _initial_public_attributes["__getattr__"]:
            implementation = _NativeDataFrame._get_column_by_attribute
        return implementation(self, name)

    # PySpark's own: empty bodies of PySpark's parameters, without a dispatch decorator. Called
    # through the public base, of any frame, each returns None once Python binds its arguments,
    # and raises CPython's TypeError of them; a frame calls its own class's method.
    def checkpoint(self, eager: bool = True) -> "DataFrame": ...

    def localCheckpoint(
        self, eager: bool = True, storageLevel: Optional[StorageLevel] = None
    ) -> "DataFrame": ...

    def lateralJoin(
        self,
        other: "DataFrame",
        on: Optional[Column] = None,
        how: Optional[str] = None,
    ) -> "DataFrame": ...

    def nearestByJoin(
        self,
        other: "DataFrame",
        rankingExpression: Column,
        numResults: int,
        mode: str,
        direction: str,
        *,
        joinType: str = "inner",
    ) -> "DataFrame": ...

    def toPandas(self) -> "PandasDataFrameLike": ...

    def asTable(self) -> TableArg: ...

    def scalar(self) -> Column: ...

    def exists(self) -> Column: ...


_initial_public_attributes["__getattr__"] = vars(DataFrame)["__getattr__"]


# Expose the concrete implementation's real API on the public base. Native instances
# continue to call their native methods directly. Explicit public-base calls
# (DataFrame.select(frame, ...)) and inherited calls on independent Python subclasses
# go through these dispatchers, as PySpark's dispatch_df_method; the properties and the
# empty methods above are PySpark's, which do not dispatch. There is no per-result wrapper
# and no change to the native object's layout or allocator.
for _name, _attribute in vars(_NativeDataFrame).items():
    if _name in _local_class_attributes or _name in {
        "__reduce__", "__reduce_ex__", "__getstate__", "__hash__", "__sizeof__",
    }:
        continue
    if _name in {"is_cached", "_schema", "_is_streaming"}:
        # These access native storage, of no member of PySpark's public base. Independent Python
        # subclasses can assign ordinary instance attributes instead of inheriting them.
        continue
    if _name in {"__getattr__", "_get_column_by_attribute"}:
        # The explicit public hook above owns the special-method dispatch. Its
        # private native bridge is an implementation detail of the concrete type.
        continue
    if _name in _empty_public_methods:
        _public_attribute = vars(DataFrame)[_name]
        _public_attribute.__doc__ = _docstring(_name, _attribute)
    elif isinstance(_attribute, staticmethod):
        _public_attribute = _attribute
    elif isinstance(_attribute, (classmethod, ClassMethodDescriptorType)):
        # There is no native DataFrame classmethod today. Require an explicit
        # binding decision and real API tests when one is introduced.
        raise TypeError(
            f"Native DataFrame classmethod requires an explicit public binding: {_name}"
        )
    elif isinstance(_attribute, (GetSetDescriptorType, cached_property)):
        _public_attribute = _empty_property(_name, _attribute)
    elif isinstance(_attribute, (FunctionType, MethodDescriptorType, WrapperDescriptorType)):
        _public_attribute = _dispatch_method(_name, _attribute)
    elif hasattr(type(_attribute), "__get__"):
        raise TypeError(f"Unsupported native DataFrame descriptor: {_name}")
    else:
        _public_attribute = _attribute
    # Initial exposure must not forward a Python dispatcher onto the native type.
    type.__setattr__(DataFrame, _name, _public_attribute)
    _initial_public_attributes[_name] = _public_attribute

# Virtual inheritance preserves isinstance/issubclass without changing the native
# object's layout. Its actual MRO stays native, while Python subclasses have the
# ordinary Python layout required by object.__new__.
DataFrame.register(_NativeDataFrame)
_public_dataframe = DataFrame

del _name, _attribute, _public_attribute


# PySpark's classic classes of `df.na` and `df.stat`, of its module: the native Connect classes of
# pyspark.sql.connect.dataframe are their virtual subclasses, as PySpark Connect's are their
# subclasses (pyspark.sql._classic_bases), and their explicit calls dispatch as PySpark's
# dispatch_df_method does. PySpark's classes are of `type`.
from pyspark.sql.utils import dispatch_df_method  # noqa: E402


class DataFrameNaFunctions(metaclass=_ClassicMeta):
    """Functionality for working with missing data in :class:`DataFrame`.

    .. versionadded:: 1.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.
    """

    def __init__(self, df: DataFrame):
        self.df = df

    @dispatch_df_method
    def drop(
        self,
        how: str = "any",
        thresh: Optional[int] = None,
        subset: Optional[Union[str, Tuple[str, ...], List[str]]] = None,
    ) -> DataFrame: ...

    drop.__doc__ = DataFrame.dropna.__doc__

    @overload
    def fill(self, value: "LiteralType", subset: Optional[List[str]] = ...) -> DataFrame: ...

    @overload
    def fill(self, value: Dict[str, "LiteralType"]) -> DataFrame: ...

    @dispatch_df_method
    def fill(
        self,
        value: Union["LiteralType", Dict[str, "LiteralType"]],
        subset: Optional[List[str]] = None,
    ) -> DataFrame: ...

    fill.__doc__ = DataFrame.fillna.__doc__

    @overload
    def replace(
        self,
        to_replace: List["LiteralType"],
        value: List["OptionalPrimitiveType"],
        subset: Optional[List[str]] = ...,
    ) -> DataFrame: ...

    @overload
    def replace(
        self,
        to_replace: Dict["LiteralType", "OptionalPrimitiveType"],
        subset: Optional[List[str]] = ...,
    ) -> DataFrame: ...

    @overload
    def replace(
        self,
        to_replace: List["LiteralType"],
        value: "OptionalPrimitiveType",
        subset: Optional[List[str]] = ...,
    ) -> DataFrame: ...

    @dispatch_df_method  # type: ignore[misc]
    def replace(
        self,
        to_replace: Union[List["LiteralType"], Dict["LiteralType", "OptionalPrimitiveType"]],
        value: Optional[
            Union["OptionalPrimitiveType", List["OptionalPrimitiveType"], _NoValueType]
        ] = _NoValue,
        subset: Optional[List[str]] = None,
    ) -> DataFrame: ...

    replace.__doc__ = DataFrame.replace.__doc__


class DataFrameStatFunctions(metaclass=_ClassicMeta):
    """Functionality for statistic functions with :class:`DataFrame`.

    .. versionadded:: 1.4.0

    .. versionchanged:: 3.4.0
        Supports Spark Connect.
    """

    def __init__(self, df: DataFrame):
        self.df = df

    @overload
    def approxQuantile(
        self,
        col: str,
        probabilities: Union[List[float], Tuple[float]],
        relativeError: float,
    ) -> List[float]: ...

    @overload
    def approxQuantile(
        self,
        col: Union[List[str], Tuple[str]],
        probabilities: Union[List[float], Tuple[float]],
        relativeError: float,
    ) -> List[List[float]]: ...

    @dispatch_df_method
    def approxQuantile(
        self,
        col: Union[str, List[str], Tuple[str]],
        probabilities: Union[List[float], Tuple[float]],
        relativeError: float,
    ) -> Union[List[float], List[List[float]]]: ...

    approxQuantile.__doc__ = DataFrame.approxQuantile.__doc__

    @dispatch_df_method
    def corr(self, col1: str, col2: str, method: Optional[str] = None) -> float: ...

    corr.__doc__ = DataFrame.corr.__doc__

    @dispatch_df_method
    def cov(self, col1: str, col2: str) -> float: ...

    cov.__doc__ = DataFrame.cov.__doc__

    @dispatch_df_method
    def crosstab(self, col1: str, col2: str) -> DataFrame: ...

    crosstab.__doc__ = DataFrame.crosstab.__doc__

    @dispatch_df_method
    def freqItems(self, cols: List[str], support: Optional[float] = None) -> DataFrame: ...

    freqItems.__doc__ = DataFrame.freqItems.__doc__

    @dispatch_df_method
    def sampleBy(
        self, col: str, fractions: Dict[Any, float], seed: Optional[int] = None
    ) -> DataFrame: ...

    sampleBy.__doc__ = DataFrame.sampleBy.__doc__
