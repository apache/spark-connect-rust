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
"""The metaclass of PySpark's classic classes whose Connect classes are the drop-in's native ones.

PySpark's ``pyspark.sql.connect.column.Column`` is a subclass of the classic
``pyspark.sql.column.Column``, and so are the Connect ``DataFrameNaFunctions``,
``DataFrameStatFunctions``, ``WindowSpec`` and ``TableArg`` of theirs. The drop-in's Connect
classes are native, and a native class cannot have a Python class as its base. So the classic
classes, which are PySpark's, are of this metaclass, which makes each native class a virtual
subclass of its classic class: ``isinstance`` and ``issubclass`` hold of it, as in PySpark, while
its MRO stays its own. The ``pyspark.sql.DataFrame`` of ``pyspark.sql.dataframe`` does the same of
a metaclass of its own.

A patch of a classic class or of its public base (``TableValuedFunctionArgument``), a name that a
library sets on it, reaches the objects of PySpark's Connect class through its MRO, unless a class
before it in the MRO has the name, or the objects set it on themselves and it is not a data
descriptor. The metaclass forwards such a patch to the native class, in the same cases, and a
deletion of the patch restores the native class's own attribute.
"""

import sys
from abc import ABCMeta
from typing import Any, NamedTuple

__all__: list[str] = []


class _Binding(NamedTuple):
    # The name of the native class in pyspark._pyspark.
    native_name: str
    # The names that PySpark 4.3.0's Connect class defines itself (its vars()), which its objects
    # find before those of the classic class. Re-derive these on every PySpark version bump.
    connect_names: frozenset[str]
    # The attributes that the __init__ of PySpark's Connect class sets on each object, which hide
    # a patch of the class unless the patch is a data descriptor.
    instance_names: frozenset[str]


# The classic classes of PySpark whose Connect classes are native, by their module and name.
_BINDINGS: dict[str, _Binding] = {
    "pyspark.sql.column.Column": _Binding(
        "Column",
        frozenset({
            "__add__", "__and__", "__bool__", "__contains__", "__div__", "__doc__", "__eq__",
            "__ge__", "__getattr__", "__getitem__", "__getnewargs__", "__gt__", "__hash__",
            "__init__", "__invert__", "__iter__", "__le__", "__lt__", "__mod__", "__module__",
            "__mul__", "__ne__", "__neg__", "__new__", "__nonzero__", "__or__", "__pow__",
            "__radd__", "__rand__", "__rdiv__", "__repr__", "__rmod__", "__rmul__", "__ror__",
            "__rpow__", "__rsub__", "__rtruediv__", "__sub__", "__truediv__", "alias", "asc",
            "asc_nulls_first", "asc_nulls_last", "astype", "between", "bitwiseAND",
            "bitwiseOR", "bitwiseXOR", "cast", "contains", "desc", "desc_nulls_first",
            "desc_nulls_last", "dropFields", "endswith", "eqNullSafe", "getField", "getItem",
            "ilike", "isNaN", "isNotNull", "isNull", "isin", "like", "name", "otherwise",
            "outer", "over", "rlike", "startswith", "substr", "to_plan", "transform",
            "try_cast", "when", "withField",
        }),
        frozenset({"_expr"}),
    ),
    "pyspark.sql.dataframe.DataFrameNaFunctions": _Binding(
        "DataFrameNaFunctions",
        frozenset({"__doc__", "__init__", "__module__", "drop", "fill", "replace"}),
        frozenset({"df"}),
    ),
    "pyspark.sql.dataframe.DataFrameStatFunctions": _Binding(
        "DataFrameStatFunctions",
        frozenset({
            "__doc__", "__init__", "__module__", "approxQuantile", "corr", "cov", "crosstab",
            "freqItems", "sampleBy",
        }),
        frozenset({"df"}),
    ),
    "pyspark.sql.window.WindowSpec": _Binding(
        "WindowSpec",
        frozenset({
            "__doc__", "__getnewargs__", "__init__", "__module__", "__new__", "__repr__",
            "orderBy", "partitionBy", "rangeBetween", "rowsBetween",
        }),
        frozenset({"_frame", "_orderSpec", "_partitionSpec"}),
    ),
    "pyspark.sql.table_arg.TableArg": _Binding(
        "TableArg",
        frozenset({
            "__doc__", "__init__", "__module__", "_is_partitioned", "orderBy", "partitionBy",
            "to_plan", "withSinglePartition",
        }),
        frozenset({"_subquery_expr"}),
    ),
}

# The public bases of the classic classes, which are of this metaclass too, without a native
# class of their own.
_PUBLIC_BASES = frozenset({"pyspark.sql.tvf_argument.TableValuedFunctionArgument"})

# These describe the Python class's layout, its attribute protocols or the bookkeeping of its
# imports and of the ABC: they belong to the classic class, and are not forwarded.
_LOCAL_CLASS_ATTRIBUTES = frozenset({
    "__new__", "__init__", "__name__", "__module__", "__qualname__", "__doc__", "__dict__",
    "__weakref__", "__bases__", "__base__", "__mro__", "__slots__", "__class__",
    "__annotations__", "__abstractmethods__", "__classcell__", "__parameters__",
    "__type_params__", "__init_subclass__", "__subclasshook__", "__class_getitem__",
    "__orig_bases__", "__getattribute__", "__setattr__", "__delattr__", "_abc_impl",
})

_missing = object()


class _Native:
    """A native Connect class, of the classic class whose virtual subclass it is."""

    def __init__(self, binding: _Binding, native: type, classic: type) -> None:
        self.binding = binding
        self.native = native
        self.classic = classic
        # The native class's own attributes, which a forwarded patch may replace.
        self.originals: dict[str, Any] = dict(vars(native))
        # Of each name of a forwarded patch: the patch, and the native class's own attribute
        # that it replaced.
        self.forwarded: dict[str, Any] = {}

    def chain(self) -> list[type]:
        """The classes after PySpark's Connect class in its MRO: the classic class and its bases,
        of which its objects inherit a patch."""
        return [c for c in self.classic.__mro__ if c is not object]

    def own(self, name: str) -> Any:
        """The native class's own attribute of ``name``: the one that a forwarded patch, still in
        place, replaced."""
        current = vars(self.native).get(name, _missing)
        forwarded = self.forwarded.get(name)
        if forwarded is not None and current is forwarded[0]:
            return forwarded[1]
        return current

    def update(self, name: str) -> None:
        """Makes the native class's attribute of ``name`` the one that PySpark's Connect objects
        find: their class's own, else a patch of a class of the chain, else the native one."""
        current = vars(self.native).get(name, _missing)
        own = self.own(name)
        patch = _missing
        if name not in self.binding.connect_names and (
            own is self.originals.get(name, _missing) or own is _missing
        ):
            # Not a name of PySpark's Connect class, nor of a patch of the native class itself,
            # which would hide the patch of a classic class: the first class of the chain that
            # has the name gives it, unless that is its own attribute, which PySpark's objects
            # inherit, and whose implementation the native class's own stands for.
            for cls in self.chain():
                if name in vars(cls):
                    value = vars(cls)[name]
                    if value is not _initial[cls].get(name, _missing) and (
                        name not in self.binding.instance_names or _is_data_descriptor(value)
                    ):
                        patch = value
                    break
        target = own if patch is _missing else patch
        if target is not current:
            if target is _missing:
                delattr(self.native, name)
            else:
                setattr(self.native, name, target)
        if patch is _missing:
            self.forwarded.pop(name, None)
        else:
            self.forwarded[name] = (patch, own)


# Each native Connect class, of the classic class whose virtual subclass it is.
_natives: dict[type, _Native] = {}
# The own attributes of each classic class and public base when it was made, which are PySpark's.
_initial: dict[type, dict[str, Any]] = {}


def _is_data_descriptor(value: Any) -> bool:
    kind = type(value)
    return hasattr(kind, "__set__") or hasattr(kind, "__delete__")


def _update_natives(cls: type, name: str) -> None:
    for native in _natives.values():
        if cls in native.chain():
            native.update(name)


class _ClassicMeta(ABCMeta):
    def __new__(mcls, name: str, bases: Any, namespace: Any, **kwargs: Any) -> Any:
        if "__module__" not in namespace:
            # As type() of PySpark's classes, of `type`: the module of a class of no __module__,
            # such as type("C", (Column,), {}), is the caller's, not that of ABCMeta.__new__.
            namespace = dict(namespace)
            namespace["__module__"] = sys._getframe(1).f_globals.get("__name__")
        return super().__new__(mcls, name, bases, namespace, **kwargs)

    def __init__(cls, name: str, bases: Any, namespace: Any, **kwargs: Any) -> None:
        super().__init__(name, bases, namespace, **kwargs)
        key = f"{cls.__module__}.{cls.__qualname__}"
        binding = _BINDINGS.get(key)
        if binding is not None:
            # The classic class of PySpark itself, of its module (also of a reload of it).
            from pyspark import _pyspark

            native = getattr(_pyspark, binding.native_name)
            cls.register(native)
            _initial[cls] = dict(vars(cls))
            _natives[native] = _Native(binding, native, cls)
        elif key in _PUBLIC_BASES:
            _initial[cls] = dict(vars(cls))
        else:
            # A subclass that a user makes, which PySpark makes of `type`: of the standard
            # ABCMeta, which a by-value cloudpickle of it names, and which the Python workers of
            # PySpark import. It keeps the ABC behavior of its bases.
            type.__setattr__(cls, "__class__", ABCMeta)

    def __setattr__(cls, name: str, value: Any) -> None:
        super().__setattr__(name, value)
        if cls in _initial and name not in _LOCAL_CLASS_ATTRIBUTES:
            _update_natives(cls, name)

    def __delattr__(cls, name: str) -> None:
        super().__delattr__(name)
        if cls in _initial and name not in _LOCAL_CLASS_ATTRIBUTES:
            _update_natives(cls, name)
