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
"""The public DataFrame base and its concrete Rust-backed implementation.

As in PySpark, ``pyspark.sql.DataFrame`` is a Python base/factory, whereas
``pyspark.sql.connect.dataframe.DataFrame`` is the concrete remote DataFrame.  Keeping
the public base in Python lets user-defined subclasses allocate their own Python state
with ``object.__new__``.  A concrete frame must still be allocated by the native class:
its Rust payload cannot be initialized by Python's object allocator.
"""

from abc import ABCMeta
from functools import wraps
from inspect import signature
from types import (
    ClassMethodDescriptorType,
    FunctionType,
    GetSetDescriptorType,
    MethodDescriptorType,
    WrapperDescriptorType,
)
from typing import NamedTuple

from pyspark._pyspark import (
    Column,
    DataFrame as _NativeDataFrame,
    DataFrameNaFunctions,
    DataFrameStatFunctions,
)

__all__ = ["DataFrame", "DataFrameNaFunctions", "DataFrameStatFunctions"]

_public_dataframe = None
_initial_public_attributes = {}
_native_overrides = {}
_missing = object()

# The undecorated, empty-bodied public methods in official PySpark 4.3.0
# pyspark/sql/dataframe.py. Re-derive this list on every PySpark version bump.
# They return None on independent Python subclasses instead of native dispatch.
_empty_public_methods = frozenset({
    "checkpoint", "localCheckpoint", "lateralJoin", "nearestByJoin", "toPandas",
    "asTable", "scalar", "exists",
})


class _NativeOverride(NamedTuple):
    previous_native: object
    forwarded: object
    previous_lookup: object


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


def _apply_native_state(name, value, attribute_lookup):
    _set_native(name, value)
    if name == "__getattr__":
        # Apply generic lookup for a forwarded hook, or reinstall the saved PyO3
        # wrapper to restore its original combined native lookup slot.
        _set_native("__getattribute__", attribute_lookup)


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

        # User extensions of the public base must be visible on native frames too.
        # Restoring an initial public dispatcher must preserve any concrete-class
        # patch that preceded the public override. Never install that dispatcher
        # itself on the native type, which would make dynamic dispatch recursive.
        initial = _initial_public_attributes.get(name, _missing)
        native_before = vars(_NativeDataFrame).get(name, _missing)
        lookup_before = vars(_NativeDataFrame).get("__getattribute__", _missing)
        restoring = initial is not _missing and value is initial
        override = _native_overrides.get(name)
        still_forwarded = override is not None and native_before is override.forwarded
        previous_native = override.previous_native if still_forwarded else native_before
        previous_lookup = (
            override.previous_lookup
            if still_forwarded and lookup_before is object.__getattribute__
            else lookup_before
        )
        if restoring:
            native_value = previous_native
            lookup_value = previous_lookup
        else:
            native_value = value
            # PyO3's __getattribute__ wrapper already resolves missing names.
            # A Python hook needs generic lookup first so it actually receives
            # misses, rather than being bypassed by that native resolver.
            lookup_value = object.__getattribute__
        try:
            _apply_native_state(name, native_value, lookup_value)
            super().__setattr__(name, value)
        except BaseException:
            _apply_native_state(name, native_before, lookup_before)
            raise
        if restoring:
            _native_overrides.pop(name, None)
        else:
            # A concrete change since the last forwarding becomes the new saved
            # state, for initial members and newly added helpers alike.
            _native_overrides[name] = _NativeOverride(
                previous_native=previous_native,
                forwarded=native_value,
                previous_lookup=previous_lookup,
            )

    def __delattr__(cls, name):
        if (
            cls is not _public_dataframe
            or name in _local_class_attributes
            or name not in vars(cls)
        ):
            return super().__delattr__(name)

        native_before = vars(_NativeDataFrame).get(name, _missing)
        lookup_before = vars(_NativeDataFrame).get("__getattribute__", _missing)
        # Deleting an initial public method removes only the base's exposure. Keep
        # an independent concrete patch, or restore the pre-public-override method.
        override = _native_overrides.get(name)
        still_forwarded = override is not None and native_before is override.forwarded
        native_value = override.previous_native if still_forwarded else native_before
        lookup_value = (
            override.previous_lookup
            if still_forwarded and lookup_before is object.__getattribute__
            else lookup_before
        )
        try:
            _apply_native_state(name, native_value, lookup_value)
            super().__delattr__(name)
        except BaseException:
            _apply_native_state(name, native_before, lookup_before)
            raise
        _native_overrides.pop(name, None)


def _not_implemented(name):
    from pyspark.errors import PySparkNotImplementedError

    # PySpark dispatches public-base methods to the concrete implementation. An
    # independent Python subclass supplies the operations for its own data model.
    return PySparkNotImplementedError(
        errorClass="NOT_IMPLEMENTED",
        messageParameters={"feature": f"DataFrame.{name}"},
    )


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
            if name in _empty_public_methods:
                # Empty official bodies still enforce their Python signatures.
                # Native calls keep their current implementation and argument handling.
                signature(method).bind(self, *args, **kwargs)
                return None
            raise _not_implemented(name)
        implementation = getattr(_NativeDataFrame, name)
        if implementation is dispatched:
            implementation = method
        return implementation(self, *args, **kwargs)

    dispatched.__module__ = __name__
    dispatched.__qualname__ = f"DataFrame.{name}"
    return dispatched


def _dispatch_property(name, descriptor):
    def get(self):
        if not isinstance(self, _NativeDataFrame):
            # Official public-base properties have empty bodies. Independent
            # Python subclasses therefore read None unless they override them.
            return None
        implementation = getattr(_NativeDataFrame, name)
        if implementation is public_property:
            implementation = descriptor
        getter = getattr(implementation, "__get__", None)
        return getter(self, type(self)) if getter is not None else implementation

    get.__name__ = name
    get.__module__ = __name__
    get.__qualname__ = f"DataFrame.{name}"
    public_property = property(get, doc=descriptor.__doc__)
    return public_property


class DataFrame(metaclass=_DataFrameMeta):
    __doc__ = _NativeDataFrame.__doc__
    is_cached: bool

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
        # user overrides it, so use its ordinary native bridge as the default.
        implementation = getattr(
            _NativeDataFrame, "__getattr__", _NativeDataFrame._get_column_by_attribute
        )
        if implementation is _initial_public_attributes["__getattr__"]:
            implementation = _NativeDataFrame._get_column_by_attribute
        return implementation(self, name)


_initial_public_attributes["__getattr__"] = vars(DataFrame)["__getattr__"]


# Expose the concrete implementation's real API on the public base. Native instances
# continue to call their native methods directly. Explicit public-base calls
# (DataFrame.select(frame, ...)) and inherited calls on independent Python subclasses
# go through these dispatchers. There is no per-result wrapper and no change to the
# native object's layout or allocator.
for _name, _attribute in vars(_NativeDataFrame).items():
    if _name in _local_class_attributes or _name in {
        "__reduce__", "__reduce_ex__", "__getstate__", "__hash__", "__sizeof__",
    }:
        continue
    if _name in {"_session", "_plan", "is_cached"}:
        # These getters access native storage. Independent Python subclasses can
        # assign ordinary instance attributes instead of inheriting those descriptors.
        continue
    if _name in {"__getattr__", "_get_column_by_attribute"}:
        # The explicit public hook above owns the special-method dispatch. Its
        # private native bridge is an implementation detail of the concrete type.
        continue
    if isinstance(_attribute, staticmethod):
        _public_attribute = _attribute
    elif isinstance(_attribute, (classmethod, ClassMethodDescriptorType)):
        # There is no native DataFrame classmethod today. Require an explicit
        # binding decision and real API tests when one is introduced.
        raise TypeError(
            f"Native DataFrame classmethod requires an explicit public binding: {_name}"
        )
    elif isinstance(_attribute, GetSetDescriptorType):
        _public_attribute = _dispatch_property(_name, _attribute)
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
