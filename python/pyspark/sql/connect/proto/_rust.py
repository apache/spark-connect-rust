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
"""The messages of Spark Connect of the drop-in's Rust protobuf runtime.

``pyspark.sql.connect.proto`` exports these: a class for each message of the protos of Spark
Connect, with the API of protobuf's Python messages, whose fields and wire format the Rust runtime
holds and encodes. Each class has the ``DESCRIPTOR`` of its message and, as an attribute, the
classes of its nested messages, its enums, their values and the numbers of its fields, as the
modules that protoc generates give them; the classes of the well-known types
``google.protobuf.Any`` and ``google.protobuf.Timestamp`` have the methods of protobuf's. Each
module of a proto (``pyspark.sql.connect.proto.base_pb2``, ...) has the names of the generated
module of the proto.

Where the ``protobuf`` package imports, the classes are of the shape of upb's (the implementation
of a ``protobuf`` install): of the metaclass ``MessageMeta``, subclasses of the runtime's
``Message`` and of protobuf's ``google.protobuf.message.Message`` (and of protobuf's methods of a
well-known type, ``well_known_types.WKTBASES``), whose ``DESCRIPTOR`` is protobuf's descriptor of
the message, of its default descriptor pool, to which the files of the protos are added, as the
generated modules add them (see ``_add_files_to_protobuf``); a parse that fails raises protobuf's
``DecodeError``. The modules that protoc generates of other protos that import them, and of the
files of their package that they import, get the classes of this runtime, and protobuf's lookups of
their types give them (see ``_hook_message_types`` and ``_hold_files``), its
``unknown_fields.UnknownFieldSet`` takes the messages of the runtime (see
``_hook_unknown_field_set``), and its parses follow the limit of the levels of messages of
protobuf's parsers (see ``_hook_depth_limit``). Without protobuf, the classes are of the runtime's
own descriptors, base and methods of protobuf's, and of the runtime's ``DecodeError``.
"""
import calendar
import collections.abc
import copyreg
import datetime
import functools
import importlib
import importlib.abc
import importlib.machinery
import os
import sys
import threading
import types
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any as _Any
from typing import Optional, Union

from pyspark import _pyspark

Message = _pyspark.ProtoMessage
RepeatedScalarContainer = _pyspark.ProtoRepeatedScalarContainer
RepeatedCompositeContainer = _pyspark.ProtoRepeatedCompositeContainer
MapContainer = _pyspark.ProtoMapContainer
ScalarMapContainer = _pyspark.ProtoScalarMapContainer
MessageMapContainer = _pyspark.ProtoMessageMapContainer

# As upb's containers: a repeated field is a MutableSequence, and a map a MutableMapping (of either
# class of the base MapContainer, ScalarMapContainer or MessageMapContainer, neither a subclass of
# the other).
collections.abc.MutableSequence.register(RepeatedScalarContainer)
collections.abc.MutableSequence.register(RepeatedCompositeContainer)
collections.abc.MutableMapping.register(MapContainer)

__all__ = [
    "Message",
    "MessageMeta",
    "DecodeError",
    "EnumTypeWrapper",
    "install",
    "add_pipeline_analysis_context",
]


def _protobuf_modules() -> Optional[types.SimpleNamespace]:
    """The modules of protobuf that the runtime follows where protobuf imports, else None: then
    the runtime needs none of it."""
    try:
        from google.protobuf import (
            descriptor_pool,
            message,
            message_factory,
            reflection,
            symbol_database,
        )
        from google.protobuf.internal import (
            api_implementation,
            enum_type_wrapper,
            well_known_types,
        )

        return types.SimpleNamespace(
            pool=descriptor_pool.Default(),
            message=message,
            message_factory=message_factory,
            reflection=reflection,
            database=symbol_database.Default(),
            enum_type_wrapper=enum_type_wrapper.EnumTypeWrapper,
            well_known_bases=dict(well_known_types.WKTBASES),
            api_implementation=api_implementation,
            # Of protobuf's pure-Python implementation, whose descriptors hold the class of their
            # message type (``_concrete_class``), which its containers and lookups read.
            python=api_implementation.Type() == "python",
        )
    except Exception:  # pragma: no cover  # noqa: BLE001 - no protobuf, or one that fails
        return None


_PROTOBUF = _protobuf_modules()

# As protobuf's ``Message.__reduce__``, of which upb's messages pickle: a message of a nested type
# pickles of protobuf's ``_InternalConstructMessage`` of the full name of its type, where protobuf
# imports, else of its class's qualified name (``Expression.Literal``), which loads in PySpark too.
_pyspark._proto_set_internal_construct(
    None if _PROTOBUF is None else getattr(_PROTOBUF.message, "_InternalConstructMessage", None)
)


if _PROTOBUF is None:

    class DecodeError(_pyspark.ProtoDecodeError):
        """The error of a parse of a message that fails (``ParseFromString``, ``MergeFromString``,
        ``FromString`` and the load of its pickle) where protobuf does not import, of upb's
        message: ``Error parsing message with type 'spark.connect.Plan': Wire format was
        corrupt``. It pickles by its name here. Where protobuf imports, the runtime raises
        protobuf's ``google.protobuf.message.DecodeError`` itself, as upb does."""

else:
    DecodeError = _PROTOBUF.message.DecodeError  # type: ignore[misc]

_pyspark._proto_set_decode_error(DecodeError)

if _PROTOBUF is None:

    class EncodeError(_pyspark.ProtoEncodeError):
        """The error of a serialization of a message that lacks a required field (of proto2,
        ``SerializeToString`` and ``ByteSize``) where protobuf does not import, of upb's message:
        ``Message thirdparty.T is missing required fields: id``. Where protobuf imports, the
        runtime raises protobuf's ``google.protobuf.message.EncodeError`` itself, as upb does."""

else:
    EncodeError = _PROTOBUF.message.EncodeError  # type: ignore[misc]

_pyspark._proto_set_encode_error(EncodeError)


class EnumTypeWrapper:
    """An enum of a proto, as protobuf's ``EnumTypeWrapper``: ``Name`` and ``Value`` of its values,
    their ``keys``, ``values`` and ``items``, and each value as an attribute, with protobuf's
    errors."""

    DESCRIPTOR = None
    ValueType = int

    def __init__(self, enum_type: _Any) -> None:
        self._enum_type = enum_type
        self.DESCRIPTOR = enum_type
        self._values: list[tuple[str, int]] = [(v.name, v.number) for v in enum_type.values]
        self._by_name = dict(self._values)
        self._by_number: dict[int, str] = {}
        for name, number in self._values:
            self._by_number.setdefault(number, name)

    def Name(self, number: int) -> str:
        try:
            return self._by_number[number]
        except KeyError:
            pass
        if not isinstance(number, int):
            raise TypeError(
                f"Enum value for {self._enum_type.name} must be an int, but got "
                f"{type(number)} {number!r}."
            )
        raise ValueError(f"Enum {self._enum_type.name} has no name defined for value {number!r}")

    def Value(self, name: str) -> int:
        try:
            return self._by_name[name]
        except KeyError:
            pass
        raise ValueError(f"Enum {self._enum_type.name} has no value defined for name {name!r}")

    def keys(self) -> list[str]:
        return [name for name, _ in self._values]

    def values(self) -> list[int]:
        return [number for _, number in self._values]

    def items(self) -> list[tuple[str, int]]:
        return list(self._values)

    def __getattr__(self, name: str) -> int:
        try:
            return self.__dict__["_by_name"][name]
        except KeyError:
            pass
        raise AttributeError(
            f"Enum {self.__dict__['_enum_type'].name} has no value defined for name {name!r}"
        )

    def __or__(self, other: _Any) -> _Any:
        return type(self) | other


def _full_name_of(value: _Any) -> Optional[str]:
    """The full name of the message type of ``value``, a message of this runtime or of protobuf's,
    or None of another object."""
    try:
        full_name = value.DESCRIPTOR.full_name
    except AttributeError:
        return None
    return full_name if isinstance(full_name, str) else None


class _AnyMethods:
    """The methods of ``google.protobuf.Any``, which protobuf's class of it has (its
    ``well_known_types.Any``): ``Pack`` and ``Unpack`` of another message, of its type URL."""

    __slots__ = ()

    def Pack(
        self, msg: _Any, type_url_prefix: str = "type.googleapis.com/", deterministic: _Any = None
    ) -> None:
        """Hold ``msg``, of the type URL of its full name under ``type_url_prefix``."""
        if len(type_url_prefix) < 1 or type_url_prefix[-1] != "/":
            self.type_url = f"{type_url_prefix}/{msg.DESCRIPTOR.full_name}"
        else:
            self.type_url = f"{type_url_prefix}{msg.DESCRIPTOR.full_name}"
        self.value = msg.SerializeToString(deterministic=deterministic)

    def Unpack(self, msg: _Any) -> bool:
        """Parse the held message into ``msg`` if it is of its type: whether it is."""
        if not self.Is(msg.DESCRIPTOR):
            return False
        msg.ParseFromString(self.value)
        return True

    def TypeName(self) -> str:
        """The full name of the type of the held message: the last part of its type URL."""
        return self.type_url.rpartition("/")[2]

    def Is(self, descriptor: _Any) -> bool:
        """Whether the held message is of the type of ``descriptor``."""
        return "/" in self.type_url and self.TypeName() == descriptor.full_name


_NANOS_PER_SECOND = 1_000_000_000
_NANOS_PER_MILLISECOND = 1_000_000
_NANOS_PER_MICROSECOND = 1_000
_SECONDS_PER_DAY = 24 * 3600
_TIMESTAMP_SECONDS_MIN = -62135596800
_TIMESTAMP_SECONDS_MAX = 253402300799
_EPOCH_NAIVE = datetime.datetime(1970, 1, 1)
_EPOCH_AWARE = _EPOCH_NAIVE.replace(tzinfo=datetime.timezone.utc)


def _check_timestamp(seconds: int, nanos: int) -> None:
    """protobuf's errors of a Timestamp out of its range (of the years 1 to 9999)."""
    if seconds < _TIMESTAMP_SECONDS_MIN or seconds > _TIMESTAMP_SECONDS_MAX:
        raise ValueError(
            f"Timestamp is not valid: Seconds {seconds} must be in range "
            "[-62135596800, 253402300799]."
        )
    if nanos < 0 or nanos >= _NANOS_PER_SECOND:
        raise ValueError(f"Timestamp is not valid: Nanos {nanos} must be in a range [0, 999999].")


def _is_message_of(value: _Any, full_name: str) -> bool:
    return _full_name_of(value) == full_name


class _TimestampMethods:
    """The methods of ``google.protobuf.Timestamp``, which protobuf's class of it has (its
    ``well_known_types.Timestamp``), of protobuf 7's results and errors: its RFC 3339 text, its
    conversions from and to the units since the epoch and ``datetime``, and its arithmetic with
    ``timedelta``, ``datetime`` and Durations. A Timestamp field of a message takes a ``datetime``
    as protobuf's does, of ``_internal_assign``."""

    __slots__ = ()

    def ToJsonString(self) -> str:
        """The RFC 3339 text of the Timestamp in UTC, of 0, 3, 6 or 9 fractional digits, as few
        as its nanos need: ``1972-01-01T10:00:20.021Z``."""
        _check_timestamp(self.seconds, self.nanos)
        seconds_of_day = self.seconds % _SECONDS_PER_DAY
        days = (self.seconds - seconds_of_day) // _SECONDS_PER_DAY
        text = (_EPOCH_NAIVE + datetime.timedelta(days, seconds_of_day)).isoformat()
        nanos = self.nanos
        # As protobuf's, of float arithmetic, whose quotients its %d truncates.
        if (nanos % 1e9) == 0:
            return text + "Z"
        if (nanos % 1e6) == 0:
            return f"{text}.{int(nanos / 1e6):03d}Z"
        if (nanos % 1e3) == 0:
            return f"{text}.{int(nanos / 1e3):06d}Z"
        return f"{text}.{nanos:09d}Z"

    def FromJsonString(self, value: str) -> None:
        """Set the Timestamp of the RFC 3339 text ``value``, of any offset and of up to 9
        fractional digits, such as ``1972-01-01T10:00:20.021-05:00``."""
        if not isinstance(value, str):
            raise ValueError(f"Timestamp JSON value not a string: {value!r}")
        offset = value.find("Z")
        if offset == -1:
            offset = value.find("+")
        if offset == -1:
            offset = value.rfind("-")
        if offset == -1:
            raise ValueError("Failed to parse timestamp: missing valid timezone offset.")
        second_value, _, nano_value = value[:offset].partition(".")
        if "t" in second_value:
            raise ValueError(
                f"time data '{second_value}' does not match format '%Y-%m-%dT%H:%M:%S', "
                "lowercase 't' is not accepted"
            )
        since_epoch = datetime.datetime.strptime(second_value, "%Y-%m-%dT%H:%M:%S") - _EPOCH_NAIVE
        seconds = since_epoch.seconds + since_epoch.days * _SECONDS_PER_DAY
        if len(nano_value) > 9:
            raise ValueError(
                f"Failed to parse Timestamp: nanos {nano_value} more than 9 fractional digits."
            )
        nanos = round(float("0." + nano_value) * 1e9) if nano_value else 0
        if value[offset] == "Z":
            if len(value) != offset + 1:
                raise ValueError(f"Failed to parse timestamp: invalid trailing data {value}.")
        else:
            timezone = value[offset:]
            colon = timezone.find(":")
            if colon == -1:
                raise ValueError(f"Invalid timezone offset value: {timezone}.")
            shift = (int(timezone[1:colon]) * 60 + int(timezone[colon + 1 :])) * 60
            seconds = seconds - shift if timezone[0] == "+" else seconds + shift
        _check_timestamp(seconds, nanos)
        self.seconds = int(seconds)
        self.nanos = int(nanos)

    def GetCurrentTime(self) -> None:
        """Set the Timestamp of the current time."""
        self.FromDatetime(datetime.datetime.now(tz=datetime.timezone.utc))

    def ToNanoseconds(self) -> int:
        _check_timestamp(self.seconds, self.nanos)
        return self.seconds * _NANOS_PER_SECOND + self.nanos

    def ToMicroseconds(self) -> int:
        _check_timestamp(self.seconds, self.nanos)
        return self.seconds * 1_000_000 + self.nanos // _NANOS_PER_MICROSECOND

    def ToMilliseconds(self) -> int:
        _check_timestamp(self.seconds, self.nanos)
        return self.seconds * 1_000 + self.nanos // _NANOS_PER_MILLISECOND

    def ToSeconds(self) -> int:
        _check_timestamp(self.seconds, self.nanos)
        return self.seconds

    def _set_valid(self, seconds: int, nanos: int) -> None:
        _check_timestamp(seconds, nanos)
        self.seconds = seconds
        self.nanos = nanos

    def FromNanoseconds(self, nanos: int) -> None:
        self._set_valid(nanos // _NANOS_PER_SECOND, nanos % _NANOS_PER_SECOND)

    def FromMicroseconds(self, micros: int) -> None:
        self._set_valid(micros // 1_000_000, (micros % 1_000_000) * _NANOS_PER_MICROSECOND)

    def FromMilliseconds(self, millis: int) -> None:
        self._set_valid(millis // 1_000, (millis % 1_000) * _NANOS_PER_MILLISECOND)

    def FromSeconds(self, seconds: int) -> None:
        self._set_valid(seconds, 0)

    def ToDatetime(self, tzinfo: Optional[datetime.tzinfo] = None) -> datetime.datetime:
        """The Timestamp as a naive ``datetime`` in UTC, or as an aware one in ``tzinfo``, of its
        microseconds."""
        _check_timestamp(self.seconds, self.nanos)
        delta = datetime.timedelta(
            seconds=self.seconds, microseconds=self.nanos // _NANOS_PER_MICROSECOND
        )
        if tzinfo is None:
            return _EPOCH_NAIVE + delta
        return (_EPOCH_AWARE + delta).astimezone(tzinfo)

    def FromDatetime(self, dt: datetime.datetime) -> None:
        """Set the Timestamp of ``dt``, a ``datetime``, which is in UTC if it is naive."""
        if type(dt).__name__ != "datetime" and not isinstance(dt, datetime.datetime):
            raise TypeError(
                "Fail to convert to Timestamp. Expected a datetime object got "
                f"{type(dt).__name__}"
            )
        try:
            seconds = calendar.timegm(dt.utctimetuple())
            nanos = dt.microsecond * _NANOS_PER_MICROSECOND
        except AttributeError as e:
            raise AttributeError(
                "Fail to convert to Timestamp. Expected a datetime like object got "
                f"{type(dt).__name__} : {e}"
            ) from e
        self._set_valid(seconds, nanos)

    def _internal_assign(self, dt: datetime.datetime) -> None:
        self.FromDatetime(dt)

    def __add__(self, value: _Any) -> datetime.datetime:
        if _is_message_of(value, "google.protobuf.Duration"):
            return self.ToDatetime() + value.ToTimedelta()
        return self.ToDatetime() + value

    __radd__ = __add__

    def __sub__(self, value: _Any) -> Union[datetime.datetime, datetime.timedelta]:
        if _is_message_of(value, "google.protobuf.Timestamp"):
            return self.ToDatetime() - value.ToDatetime()
        if _is_message_of(value, "google.protobuf.Duration"):
            return self.ToDatetime() - value.ToTimedelta()
        return self.ToDatetime() - value

    def __rsub__(self, dt: datetime.datetime) -> datetime.timedelta:
        return dt - self.ToDatetime()


# The wrapper of the enums of the protos: protobuf's, where it imports, as upb's classes give them.
_ENUM_TYPE_WRAPPER: type = EnumTypeWrapper if _PROTOBUF is None else _PROTOBUF.enum_type_wrapper


class _MessageBase:
    """The base of the classes of the runtime where protobuf does not import, in place of
    protobuf's ``google.protobuf.message.Message``, of which upb's classes take these methods:
    the ``dir()`` of a message, of its fields too, and the abstract methods that protobuf's has of
    no implementation. The runtime's ``Message`` pickles and copies a message as protobuf's
    methods do (``__reduce__``, ``__getstate__``, ``__setstate__``, ``__copy__``)."""

    __slots__ = ()
    DESCRIPTOR = None

    def _SetListener(self, message_listener: _Any) -> None:
        raise NotImplementedError

    def __unicode__(self) -> str:
        raise NotImplementedError

    def __dir__(self) -> list[str]:
        names = set(super().__dir__())
        if self.DESCRIPTOR is not None:
            names.update(field.name for field in self.DESCRIPTOR.fields)
        # As protobuf's: not ``Extensions``, of a message of no extension ranges.
        try:
            self.Extensions  # noqa: B018 - whether the attribute raises
        except AttributeError:
            names.discard("Extensions")
        return sorted(names)


# The base of the classes of the messages, as protobuf's builder gives it to upb's metaclass:
# protobuf's ``google.protobuf.message.Message``, where it imports.
_MESSAGE_BASE: type = _MessageBase if _PROTOBUF is None else _PROTOBUF.message.Message


def _well_known_base(full_name: str) -> Optional[type]:
    """The class of the methods of the well-known type ``full_name``, which its class has as a
    base: protobuf's, where it imports (``well_known_types.WKTBASES``, of which upb's classes of
    the types take them), else the runtime's of ``Any`` and ``Timestamp``."""
    if _PROTOBUF is not None:
        return _PROTOBUF.well_known_bases.get(full_name)
    return {"google.protobuf.Any": _AnyMethods, "google.protobuf.Timestamp": _TimestampMethods}.get(
        full_name
    )


# The class of each message, by the full name of the message.
_CLASSES: dict[str, type] = {}

# The classes of the runtime that protobuf's lookups of their types give (its metaclass and
# ``message_factory.GetMessageClass``, see ``_class_for``), by protobuf's descriptor of their type:
# those of the protos of Spark Connect, of a plugin's file that imports them and of the files of its
# package that it imports (see ``_hold_files``), as upb gives its one class of a type. The classes
# of the well-known types, and of the files of other packages that a plugin's file imports, are
# not among them: their lookups give protobuf's classes.
_OWNED: dict[_Any, type] = {}


# The classes that the metaclass made that the runtime registers at once, of the thread that makes
# them (see ``_registered_together``), else none.
_REGISTERING = threading.local()


@contextmanager
def _registered_together() -> Generator[None, None, None]:
    """Register the classes that the metaclass makes in the block (of this thread) with the Rust
    runtime at once, at its end, rather than one each: no message of them is made in the block (as
    of the classes of a file, which the runtime makes before their messages)."""
    if getattr(_REGISTERING, "classes", None) is not None:
        yield
        return
    _REGISTERING.classes = {}
    try:
        yield
    finally:
        classes, _REGISTERING.classes = _REGISTERING.classes, None
        if classes:
            _pyspark._proto_register_classes(classes)


@functools.cache
def _names_of_bases(bases: tuple) -> frozenset:
    """The names of the attributes of the classes ``bases``, of their own bases too."""
    return frozenset(name for base in bases for name in dir(base))


def _add_attributes(namespace: dict, bases: tuple, descriptor: _Any) -> None:
    """Add to ``namespace``, of the class of the message type ``descriptor`` of ``bases``, the
    attributes that upb's metaclass gives of a type: the number of each field
    (``Relation.READ_FIELD_NUMBER``), the wrapper of each nested enum and the number of each value
    of a nested enum (``Join.JOIN_TYPE_INNER``), as upb binds them once they are read, but for a
    name that the class has (of the namespace, of its nested messages, or of its bases), as upb
    reads that first. A name of the suffix of the numbers of fields is only of the number of a
    field, not of a nested enum or a value of one, as upb gives the numbers of the fields of a read
    of any name of their suffix (and raises ``AttributeError`` of one of no field)."""
    taken = _names_of_bases(bases)
    for field in descriptor.fields:
        name = field.name.upper() + "_FIELD_NUMBER"
        if name not in namespace and name not in taken:
            namespace[name] = field.number
    for enum in descriptor.enum_types:
        name = enum.name
        if name not in namespace and name not in taken and not _is_field_number_name(name):
            namespace[name] = _ENUM_TYPE_WRAPPER(enum)
        for value in enum.values:
            name = value.name
            if name not in namespace and name not in taken and not _is_field_number_name(name):
                namespace[name] = value.number


def _is_field_number_name(name: str) -> bool:
    """Whether ``name`` is of the suffix of the names of the numbers of fields
    (``READ_FIELD_NUMBER``), of which upb's class gives only the number of a field, and no nested
    enum or value of one."""
    return len(name) > len("_FIELD_NUMBER") and name.endswith("_FIELD_NUMBER")


class MessageMeta(type):
    """The metaclass of the classes of the messages, as upb's ``MessageMeta``: a class of a message
    type is of the base ``Message`` of protobuf (or ``()``) and of a ``DESCRIPTOR`` of the type, and
    there is one class of each type, so a message class cannot be subclassed, with upb's error. The
    class of a type is the runtime's ``Message``, ``Message`` of protobuf (and protobuf's methods of
    a well-known type), of ``__slots__ = ()``, and has the attributes of upb's class of the type:
    the class of a nested message, the wrapper of a nested enum, the number of a value of a nested
    enum (``Join.JOIN_TYPE_INNER``) and the number of a field (``Relation.READ_FIELD_NUMBER``),
    bound when the class is made (see ``_add_attributes``)."""

    def __new__(mcs, name: str, bases: tuple, namespace: dict, **kwargs: _Any) -> "MessageMeta":
        if not (bases == () or (len(bases) == 1 and bases[0] is _MESSAGE_BASE)):
            raise TypeError(f"A Message class can only inherit from Message, not {bases}")
        descriptor = namespace.get("DESCRIPTOR")
        if descriptor is None:
            raise TypeError("Message class has no DESCRIPTOR")
        made = _class_of_descriptor(descriptor)
        if made is not None:
            return made  # type: ignore[return-value]
        full_name = descriptor.full_name
        well_known = _well_known_base(full_name)
        namespace = dict(namespace, __slots__=())
        if _PROTOBUF is not None and _PROTOBUF.python:
            # As protobuf's pure-Python messages hold their unknown fields, which its ``==`` reads
            # of the other message, of a message of the runtime too.
            namespace["_unknown_fields"] = property(_pyspark._proto_unknown_field_pairs)
        real_bases = (Message, _MESSAGE_BASE) + ((well_known,) if well_known else ())
        _add_attributes(namespace, real_bases, descriptor)
        cls = super().__new__(mcs, name, real_bases, namespace, **kwargs)
        _CLASSES[full_name] = cls
        pending = getattr(_REGISTERING, "classes", None)
        if pending is not None:
            pending[full_name] = cls
        else:
            _pyspark._proto_register_classes({full_name: cls})
        return cls


def _class_of_descriptor(descriptor: _Any) -> Optional[type]:
    """The class of the runtime of the message type ``descriptor``, if it has one: of the
    runtime's descriptor, or, where protobuf imports, of the descriptor of protobuf's default pool
    that the class has."""
    try:
        full_name = descriptor.full_name
    except AttributeError:
        return None
    cls = _CLASSES.get(full_name)
    if cls is None:
        return None
    if cls.__dict__.get("DESCRIPTOR") is descriptor or isinstance(
        descriptor, _pyspark.ProtoDescriptor
    ):
        return cls
    return None


def _descriptor_of(full_name: str) -> _Any:
    """The descriptor of the message type ``full_name`` that its class has: protobuf's, of its
    default pool, where protobuf imports and the pool holds the file of the type as the runtime does
    (see ``_add_files_to_protobuf``), as upb's classes have, else the runtime's."""
    if _PROTOBUF is not None and _USE_PROTOBUF_DESCRIPTORS:
        return _PROTOBUF.pool.FindMessageTypeByName(full_name)
    return _pyspark._proto_message_descriptor(full_name)


def _enum_descriptor_of(full_name: str) -> _Any:
    """The descriptor of the enum type ``full_name``, as ``_descriptor_of`` gives that of a
    message."""
    if _PROTOBUF is not None and _USE_PROTOBUF_DESCRIPTORS:
        return _PROTOBUF.pool.FindEnumTypeByName(full_name)
    return _pyspark._proto_enum_descriptor(full_name)


def _file_descriptor_of(name: str) -> _Any:
    """The descriptor of the file ``name``, as ``_descriptor_of`` gives one of a message."""
    if _PROTOBUF is not None and _USE_PROTOBUF_DESCRIPTORS:
        return _PROTOBUF.pool.FindFileByName(name)
    return _pyspark._proto_file_descriptor(name)


def _make_class(
    full_name: str, qualname: str, module_name: Optional[str], owned: bool = True
) -> type:
    """The class of the message type ``full_name``, of its nested messages too, made of the
    metaclass as protobuf's builder makes a class of a generated module, of the module
    ``module_name`` and the qualified name ``qualname``. An ``owned`` class is one that protobuf's
    lookups of its type give (see ``_own``); any other pickles as protobuf's class of its type
    (see ``_reduce_as_protobufs``)."""
    made = _CLASSES.get(full_name)
    if made is not None:
        return made
    descriptor = _descriptor_of(full_name)
    namespace: dict[str, _Any] = {"DESCRIPTOR": descriptor, "__module__": module_name}
    for nested in descriptor.nested_types:
        namespace[nested.name] = _make_class(
            nested.full_name, qualname + "." + nested.name, module_name, owned
        )
    namespace["__qualname__"] = qualname
    cls = MessageMeta(descriptor.name, (_MESSAGE_BASE,), namespace)
    if owned:
        _own(cls)
    elif _PROTOBUF is not None:
        cls.__reduce__ = _reduce_as_protobufs  # type: ignore[method-assign]
        _PROTOBUF_CLASSES.pop(cls, None)
    return cls


def _own(cls: type) -> None:
    """Make ``cls`` the class that protobuf's lookups of its message type give (``_OWNED``), where
    protobuf imports: of protobuf's pure-Python implementation, the class of protobuf's descriptor
    of the type too (``_concrete_class``), as its metaclass sets it of a class it makes, but not in
    place of a class that the descriptor has (protobuf's, of a module that its builder built). A
    class that pickled as protobuf's class (see ``_make_class``) no longer does."""
    if _PROTOBUF is None or not _USE_PROTOBUF_DESCRIPTORS:
        return
    descriptor = cls.__dict__["DESCRIPTOR"]
    _OWNED[descriptor] = cls
    if "__reduce__" in cls.__dict__:
        type.__delattr__(cls, "__reduce__")
    if _PROTOBUF.python and getattr(descriptor, "_concrete_class", None) is None:
        descriptor._concrete_class = cls


def _reduce_as_protobufs(self: _Any) -> tuple[_Any, ...]:
    """The pickle of a message of a class of the runtime whose type protobuf's lookups give
    protobuf's class of (see ``_make_class``): ``Any``, ``Timestamp`` and the other well-known
    types, and the types of the files of other packages that a plugin's file imports, as upb's
    pickle of the message, of protobuf's ``Message.__reduce__`` of protobuf's class of the type
    (``google.protobuf.any_pb2.Any``, ``google.rpc.status_pb2.Status``, and of a nested type
    protobuf's ``_InternalConstructMessage``): it loads where protobuf and the module of the type
    import, as a message of protobuf's class, which the message fields of the runtime take."""
    cls = type(self)
    state = self.__getstate__()
    if cls in _NESTED:
        return _PROTOBUF.message._InternalConstructMessage, (_NESTED[cls],), state
    protobuf_class = _PROTOBUF_CLASSES.get(cls)
    if protobuf_class is None:
        descriptor = cls.__dict__["DESCRIPTOR"]
        if descriptor.containing_type is not None:
            _NESTED[cls] = descriptor.full_name
            return _PROTOBUF.message._InternalConstructMessage, (descriptor.full_name,), state
        protobuf_class = _PROTOBUF_CLASSES[cls] = _protobuf_class_of(descriptor)
    return protobuf_class, (), state


# protobuf's class of the type of each class of the runtime whose messages pickle as protobuf's (see
# ``_reduce_as_protobufs``), and the full name of each such class of a nested type, known once.
_PROTOBUF_CLASSES: dict[type, type] = {}
_NESTED: dict[type, str] = {}


def _protobuf_class_of(descriptor: _Any) -> type:
    """protobuf's class of the message type ``descriptor`` of its default pool, of protobuf's own
    ``message_factory.GetMessageClass``: upb's one class of the type, of the module of its file."""
    get_message_class = _PROTOBUF.message_factory.GetMessageClass  # type: ignore[union-attr]
    return getattr(get_message_class, "_protobuf_function", get_message_class)(descriptor)


def _reduce_message_class(cls: type) -> _Any:
    """The pickle of ``cls``, a class of the metaclass (of ``copyreg``, which ``pickle`` reads of a
    class): of its name, as a class pickles, but of a class whose messages pickle as protobuf's
    (see ``_reduce_as_protobufs``), protobuf's class of its type, which its name names, as upb's
    class of the type pickles: the type of a message of protobuf's class of the type."""
    if "__reduce__" not in cls.__dict__ or _PROTOBUF is None:
        return cls.__qualname__
    reduced = _reduce_as_protobufs(cls())
    return type, (reduced[0](*reduced[1]),)


copyreg.pickle(MessageMeta, _reduce_message_class)


def _module_name(file_name: str, package: str) -> str:
    """The name of the module of a proto file: that of its generated module for a proto of Spark
    Connect, such as ``pyspark.sql.connect.proto.base_pb2``, else one of this module."""
    stem = file_name.rsplit("/", 1)[-1][: -len(".proto")]
    if package == "spark.connect":
        return f"pyspark.sql.connect.proto.{stem}_pb2"
    return f"{__name__}.{stem}_pb2"


def _protoc_module_name(file_name: str) -> str:
    """The name of the module that protoc generates of the file ``file_name``, of its path:
    ``spark.connect.relations_pb2`` of ``spark/connect/relations.proto``."""
    return file_name[: -len(".proto")].replace("/", ".") + "_pb2"


def _module_alias(file_name: str) -> str:
    """The name that the generated module of a proto that imports the file ``file_name`` binds
    the module of the file to, as protoc names it: ``spark_dot_connect_dot_relations__pb2``."""
    return _protoc_module_name(file_name).replace("_", "__").replace(".", "_dot_")


def _add_descriptors(module: types.ModuleType, file_descriptor: _Any) -> None:
    """Bind the descriptors of the messages, enums and services of the file to the module, as
    protobuf's builder binds them in a generated module: ``_RELATION`` of ``Relation``,
    ``_JOIN_JOINTYPE`` of ``Join.JoinType`` and ``_SPARKCONNECTSERVICE``."""

    def add_nested(descriptor: _Any, prefix: str) -> None:
        for name, nested in descriptor.nested_types_by_name.items():
            setattr(module, prefix + name.upper(), nested)
            add_nested(nested, prefix + name.upper() + "_")
        for enum in descriptor.enum_types:
            setattr(module, prefix + enum.name.upper(), enum)

    for name, descriptor in file_descriptor.message_types_by_name.items():
        setattr(module, "_" + name.upper(), descriptor)
        add_nested(descriptor, "_" + name.upper() + "_")
    for name, enum in file_descriptor.enum_types_by_name.items():
        setattr(module, "_" + name.upper(), enum)
    for name, service in file_descriptor.services_by_name.items():
        setattr(module, "_" + name.upper(), service)


class _Loader(importlib.abc.Loader):
    """The loader of the modules of this runtime, which this module makes: importing one again
    (after it is removed from ``sys.modules``) or reloading it gives it as it is."""

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> types.ModuleType:
        return _MODULES[spec.name]

    def exec_module(self, module: types.ModuleType) -> None:
        pass


class _Finder(importlib.abc.MetaPathFinder):
    """The finder of the modules of the protos of Spark Connect of this runtime, which the runtime
    makes at run time: ``pyspark.sql.connect.proto.base_pb2`` and the others, whose files in the
    package are only the source that type checkers find next to their stubs."""

    def find_spec(
        self, fullname: str, path: Optional[Sequence[str]], target: _Any = None
    ) -> Optional[importlib.machinery.ModuleSpec]:
        module = _MODULES.get(fullname)
        if module is None or not _is_exported(fullname):
            return None
        return module.__spec__


_LOADER = _Loader()


def _is_exported(module_name: str) -> bool:
    """Whether the module is one of ``pyspark.sql.connect.proto``, of a proto of Spark Connect."""
    return module_name.startswith("pyspark.sql.connect.proto.") and not module_name.startswith(
        __name__ + "."
    )


# The files of the descriptor set of the runtime, and those of them of the protos of Spark Connect.
# The files of the descriptor set of the runtime, of their names, packages, messages and enums
# (read once), and the names of those of them of the protos of Spark Connect.
_FILES = _pyspark._proto_files()
_EMBEDDED_FILES = frozenset(name for name, _, _, _ in _FILES)
_SPARK_CONNECT_FILES = frozenset(
    name for name, package, _, _ in _FILES if package == "spark.connect"
)


def _same_files(protobuf_file: _Any, name: str) -> bool:
    """Whether ``protobuf_file``, the file ``name`` of protobuf's default pool, declares the
    messages of the runtime's file ``name``, of the same fields: as it does when it is the file
    that the runtime added, or that a copy of Apache Spark's generated modules of the same protos
    added before."""
    if protobuf_file.serialized_pb == _pyspark._proto_file_descriptor(name).serialized_pb:
        return True

    def shape(descriptor: _Any) -> list[_Any]:
        fields = [
            (f.name, f.number, f.type, getattr(f.message_type, "full_name", None))
            for f in descriptor.fields
        ]
        return [descriptor.full_name, fields] + [shape(n) for n in descriptor.nested_types]

    ours = _pyspark._proto_file_descriptor(name)
    try:
        return [shape(m) for m in protobuf_file.message_types_by_name.values()] == [
            shape(m) for m in ours.message_types_by_name.values()
        ]
    except Exception:  # pragma: no cover  # noqa: BLE001 - descriptors of another shape
        return False


def _add_files_to_protobuf() -> bool:
    """Add the files of the protos to the default descriptor pool of protobuf, when protobuf
    imports, as its generated modules of the protos add them: the modules that protoc generates of
    other protos that import them (those of a plugin, such as Delta Lake's) then build, and
    protobuf's ``json_format``, ``text_format`` and ``symbol_database`` find their types, of an
    ``Any`` too. Whether the classes of the runtime take protobuf's descriptors of the files: the
    pool has them as the runtime does.

    protobuf's own modules of the well-known types add their files, of protobuf's copies of them.
    A file that the pool has is left as it is. The files are those that protoc writes, the bytes
    that Apache Spark's generated modules embed, so that one of those (of another package) adds
    its file again, as the pool takes a file twice. A pool that refuses a file, or that has
    another file of its name (as only a copy of Apache Spark's generated modules of other protos,
    imported first, adds), keeps the runtime's descriptors for the classes."""
    if _PROTOBUF is None:  # pragma: no cover - the runtime needs none of this
        return False
    pool = _PROTOBUF.pool
    same = True
    for file_name, package, _, _ in _FILES:
        if package == "google.protobuf":
            stem = file_name.rsplit("/", 1)[-1][: -len(".proto")]
            try:
                importlib.import_module(f"google.protobuf.{stem}_pb2")
            except Exception:  # pragma: no cover  # noqa: BLE001
                pass
        try:
            file = pool.FindFileByName(file_name)
        except KeyError:
            try:
                serialized = _pyspark._proto_file_descriptor(file_name).serialized_pb
                file = pool.AddSerializedFile(serialized)
            except Exception:  # pragma: no cover  # noqa: BLE001 - other descriptors of its names
                same = False
                continue
        same = same and _same_files(file, file_name)
    return same


_USE_PROTOBUF_DESCRIPTORS = _add_files_to_protobuf()


class _OwnModule(types.ModuleType):
    """A module of the runtime of its own, of a file that no proto of Spark Connect is
    (``pyspark.sql.connect.proto._rust.any_pb2``): it pickles by its name, so that a class of a
    dotted name of a global of a module that binds it (``base_pb2``'s
    ``google_dot_protobuf_dot_any__pb2.Any``, without protobuf) pickles of the protocols before 4
    too, of which ``pickle`` pickles such a class as the attribute of that module."""

    def __reduce__(self) -> tuple[_Any, ...]:
        return importlib.import_module, (self.__name__,)


def _build() -> dict[str, types.ModuleType]:
    modules = {}
    files = []
    for file_name, package, messages, enums in _FILES:
        module_name = _module_name(file_name, package)
        module = (types.ModuleType if _is_exported(module_name) else _OwnModule)(module_name)
        module.__doc__ = f"The messages of {file_name}, of the drop-in's Rust protobuf runtime."
        # As the generated module: of the file of the module, of the package (the source that
        # type checkers find next to its stub), which the finder of the runtime shadows.
        source = os.path.join(os.path.dirname(__file__), module_name.rpartition(".")[2] + ".py")
        origin = source if _is_exported(module_name) and os.path.exists(source) else None
        module.__spec__ = importlib.machinery.ModuleSpec(module_name, _LOADER, origin=origin)
        if origin is not None:
            module.__spec__.has_location = True
            module.__file__ = origin
        module.__loader__ = _LOADER
        module.__package__ = module_name.rpartition(".")[0]
        file_descriptor = _file_descriptor_of(file_name)
        module.DESCRIPTOR = file_descriptor
        _add_descriptors(module, file_descriptor)
        # The classes of the well-known types are not those of protobuf's lookups of their types,
        # which give protobuf's classes, as the modules of the types stay protobuf's.
        owned = package != "google.protobuf"
        for full_name in messages:
            name = full_name.rsplit(".", 1)[-1]
            setattr(module, name, _make_class(full_name, name, module_name, owned))
        for full_name in enums:
            descriptor = _enum_descriptor_of(full_name)
            setattr(module, descriptor.name, _ENUM_TYPE_WRAPPER(descriptor))
            for value in descriptor.values:
                setattr(module, value.name, value.number)
        modules[file_name] = module
        files.append(_pyspark._proto_file_descriptor(file_name))
    # The module of each file that a file imports, under protoc's name of it.
    for file_descriptor in files:
        for dependency in file_descriptor.dependencies:
            setattr(
                modules[file_descriptor.name],
                _module_alias(dependency.name),
                modules[dependency.name],
            )
    return {module.__name__: module for module in modules.values()}


with _registered_together():
    _MODULES = _build()

# The well-known types that the protos of Spark Connect import.
Any = _CLASSES["google.protobuf.Any"]
Timestamp = _CLASSES["google.protobuf.Timestamp"]


# The generated modules that ``pyspark.sql.connect.proto`` star-imports, in its order.
_EXPORTED = ("base", "types", "commands", "expressions", "relations", "catalog", "common", "ml",
             "ml_common", "pipelines")  # fmt: skip


def _path_in_package(file_name: str) -> tuple[str, str]:
    """The module of the first proto of Spark Connect that ``pyspark.sql.connect.proto`` exports
    that imports the file ``file_name`` (of a well-known type), and the name that the module binds
    the module of the file to: ``pyspark.sql.connect.proto.base_pb2`` and
    ``google_dot_protobuf_dot_any__pb2`` of ``google/protobuf/any.proto``."""
    for stem in _EXPORTED:
        module_name = f"pyspark.sql.connect.proto.{stem}_pb2"
        if any(d.name == file_name for d in _MODULES[module_name].DESCRIPTOR.dependencies):
            return module_name, _module_alias(file_name)
    raise KeyError(file_name)  # pragma: no cover - the protos import Any and Timestamp


def _new_message(full_name: str) -> _Any:
    """A new message of the class of the runtime of the message type ``full_name``, of which a
    pickle of ``Any`` or ``Timestamp`` of a protocol before 4 loads without protobuf (see
    ``_reduce_ex_by_path``)."""
    return _CLASSES[full_name]()


def _reduce_ex_by_path(self: _Any, protocol: int) -> tuple[_Any, ...]:
    """The pickle of a message of ``Any`` or ``Timestamp`` without protobuf: of protocol 4 and
    later, of the class by the name that the generated module of the first proto of Spark Connect
    that imports its file binds it to, which is the class's own (``Any`` is
    ``pyspark.sql.connect.proto.base_pb2``'s ``google_dot_protobuf_dot_any__pb2.Any``), so that the
    pickle loads in PySpark as a message of protobuf's class (upb's), as upb's own pickle of it
    does, and here as one of the runtime's class, with or without protobuf; of an earlier protocol,
    which names no such dotted name, of ``_new_message``, which only the drop-in has."""
    if protocol >= 4:
        return self.__reduce__()
    return _new_message, (self.DESCRIPTOR.full_name,), self.__getstate__()


if _PROTOBUF is None:  # pragma: no cover - of the runs without protobuf (a subprocess's of the tests)
    for _cls in (Any, Timestamp):
        _module_name_of, _alias = _path_in_package(_cls.DESCRIPTOR.file.name)
        _cls.__module__ = _module_name_of
        _cls.__qualname__ = f"{_alias}.{_cls.__name__}"
        _cls.__reduce_ex__ = _reduce_ex_by_path  # type: ignore[method-assign]
    del _cls, _module_name_of, _alias


def _register_classes_in_protobuf() -> None:
    """Register the classes of the messages of the protos of Spark Connect in protobuf's default
    symbol database, by the descriptors of the default pool, as protobuf's builder registers the
    classes of its generated modules (``RegisterMessage``): ``symbol_database.Default().GetSymbol``
    then gives them, which is how PySpark's pickle of a message of a nested type
    (``Expression.Literal``) loads, through protobuf's ``_InternalConstructMessage`` of the full name
    of its type. A class that the database has stays (see ``_hold_built_modules``)."""
    if _PROTOBUF is None:  # pragma: no cover - the runtime needs none of this
        return
    registered = _PROTOBUF.database._classes
    for descriptor, cls in list(_OWNED.items()):
        if descriptor.file.name in _SPARK_CONNECT_FILES:
            registered.setdefault(descriptor, cls)


# The lock of the files that the hooks add to the runtime and of their classes: the generated
# modules of two threads may build at once.
_HOOK_LOCK = threading.RLock()

# The files of protobuf's default pool that the runtime holds with its classes, which their modules
# have (see ``_hold_files``), by name.
_HELD: set[str] = set()

# The names of the files of protobuf's default pool of a plugin that the runtime refused (see
# ``_hold_files``), which keep protobuf's classes: a file's imports do not change.
_UNHELD: set[str] = set()

# Whether each file of protobuf's default pool imports a proto of Spark Connect or a file that the
# runtime holds with its classes, directly or through the files that it imports, or is one, by name
# (see ``_imports_held``): of a file that does not, known until the runtime holds more files.
_IMPORTS_HELD: dict[str, bool] = {}

# The message types of protobuf's default pool of which protobuf's lookups give protobuf's
# classes, as ``_class_for`` found, which it does not look at again: dropped when the runtime holds
# more files (see ``_held_more``).
_NOT_OWNED: set[_Any] = set()

# protobuf's class of each message type (of a nested type too) of protobuf's symbol database that
# the runtime's class replaced there (see ``_register_in_protobuf``), by its descriptor: the class
# that the type gets back if the runtime gives its file protobuf's classes again (see ``_release``).
_PROTOBUFS: dict[_Any, type] = {}

# Whether each file of protobuf's default pool is a proto of Spark Connect or imports one, directly
# or through the files that it imports, by name (see ``_imports_spark_connect``), and whether it
# declares what the runtime does not hold (see ``_declares_unheld``): a file does not change.
_IMPORTS_SPARK_CONNECT: dict[str, bool] = {}
_DECLARES_UNHELD: dict[str, bool] = {}


def _imports_held(file: _Any) -> bool:
    """Whether ``file``, a file of protobuf's default pool, imports a proto of Spark Connect or a
    file that the runtime holds with its classes (``_HELD``), directly or through the files that it
    imports, or is one: known of each file once, of those that it imports first, so that a closure
    is walked once for the files of many modules (until the runtime holds more files)."""
    known = _IMPORTS_HELD.get(file.name)
    if known is not None:
        return known
    stack: list[tuple[_Any, bool]] = [(file, False)]
    while stack:
        current, imports_known = stack.pop()
        name = current.name
        if name in _IMPORTS_HELD:
            continue
        if name in _EMBEDDED_FILES:
            _IMPORTS_HELD[name] = name in _SPARK_CONNECT_FILES
        elif name in _HELD:
            _IMPORTS_HELD[name] = True
        elif imports_known:
            _IMPORTS_HELD[name] = any(_IMPORTS_HELD[d.name] for d in current.dependencies)
        else:
            stack.append((current, True))
            stack.extend((d, False) for d in current.dependencies)
    return _IMPORTS_HELD[file.name]


def _held_more() -> None:
    """Forget what is known of the files that imported no file that the runtime holds, and of the
    types of protobuf's classes, as the runtime holds more files."""
    for name in [name for name, imports in _IMPORTS_HELD.items() if not imports]:
        del _IMPORTS_HELD[name]
    _NOT_OWNED.clear()


def _imports_spark_connect(file: _Any) -> bool:
    """Whether ``file``, a file of protobuf's default pool, is a proto of Spark Connect or imports
    one, directly or through the files that it imports: a file that the runtime holds so keeps its
    classes (see ``_release_imports_of``). Known of each file once, as ``_imports_held``."""
    known = _IMPORTS_SPARK_CONNECT.get(file.name)
    if known is not None:
        return known
    stack: list[tuple[_Any, bool]] = [(file, False)]
    while stack:
        current, imports_known = stack.pop()
        name = current.name
        if name in _IMPORTS_SPARK_CONNECT:
            continue
        if name in _EMBEDDED_FILES:
            _IMPORTS_SPARK_CONNECT[name] = name in _SPARK_CONNECT_FILES
        elif imports_known:
            _IMPORTS_SPARK_CONNECT[name] = any(
                _IMPORTS_SPARK_CONNECT[d.name] for d in current.dependencies
            )
        else:
            stack.append((current, True))
            stack.extend((d, False) for d in current.dependencies)
    return _IMPORTS_SPARK_CONNECT[file.name]


def _declares_unheld(file: _Any) -> bool:
    """Whether ``file`` declares what the runtime does not hold, as its pool refuses it (see
    ``pbmsg::pool::register``): extensions, of the file or of a message, extension ranges, or the
    message-set wire format of a message. Known of each file once (``_DECLARES_UNHELD``)."""
    known = _DECLARES_UNHELD.get(file.name)
    if known is not None:
        return known
    declares = bool(file.extensions_by_name)
    stack = list(file.message_types_by_name.values())
    while stack and not declares:
        message = stack.pop()
        declares = bool(message.extension_ranges or message.extensions) or bool(
            message.GetOptions().message_set_wire_format
        )
        stack.extend(message.nested_types)
    _DECLARES_UNHELD[file.name] = declares
    return declares


def _closure_of(file: _Any) -> list[_Any]:
    """The files of protobuf's pool that the runtime needs to hold the messages of the file
    ``file``: the file and the files that it imports, directly or through others, but those of the
    descriptor set of the runtime, each after those that it imports."""
    found: list[_Any] = []
    seen: set[str] = set()
    stack: list[tuple[_Any, bool]] = [(file, False)]
    while stack:
        current, imports_found = stack.pop()
        if imports_found:
            found.append(current)
            continue
        if current.name in seen or current.name in _EMBEDDED_FILES:
            continue
        seen.add(current.name)
        stack.append((current, True))
        stack.extend((dependency, False) for dependency in reversed(current.dependencies))
    return found


def _is_protobufs_own(file: _Any) -> bool:
    """Whether ``file`` is one of protobuf's own files (its well-known types), whose module and
    classes, protobuf's, the runtime leaves as they are: the runtime has classes of their messages
    of its own, of protobuf's methods of them."""
    return file.package == "google.protobuf"


def _is_googles(package: str) -> bool:
    """Whether ``package`` is one of Google's (``google.protobuf``, ``google.rpc``, ...), whose
    files the runtime never takes over: protobuf's and grpcio's modules of them stay as they are."""
    return package == "google" or package.startswith("google.")


def _globals_of(module: _Any) -> Optional[dict[str, _Any]]:
    """The globals of ``module``, an object of ``sys.modules``, as they are: read without its
    ``__getattribute__``, so that a lazy module (of ``importlib.util.LazyLoader``) is not loaded
    by the read, nor a module's ``__getattr__`` run. None of an object of no such dict."""
    try:
        namespace = object.__getattribute__(module, "__dict__")
    except Exception:  # noqa: BLE001 - an object of no globals, or of a failing one
        return None
    return namespace if isinstance(namespace, dict) else None


def _is_generated_module_of(module: _Any, file: _Any) -> bool:
    """Whether ``module`` is a generated module of ``file``: whose ``DESCRIPTOR`` is the file and
    which has the descriptors that protobuf's builder binds in a generated module (``_TABLE`` of
    ``Table``), which a package that star-imports the module does not have."""
    namespace = _globals_of(module)
    if namespace is None or namespace.get("DESCRIPTOR") is not file:
        return False
    for name, descriptor in file.message_types_by_name.items():
        return namespace.get("_" + name.upper()) is descriptor
    return True


def _imported_module_of_file(
    file: _Any, near: Optional[str] = None
) -> Optional[types.ModuleType]:
    """The imported generated module of the file ``file`` of protobuf's default pool, if any: the
    module of the class of a message of the file that protobuf's symbol database has (which
    protobuf's builder registers as it builds the module), else one of the top-level package of the
    name of that class's module or of ``near`` (the module of a plugin's file that imports it), as a
    module of the name of protoc's path of the file (``dl.connect.base_pb2``) is imported of another
    (``dl.connect.proto.base_pb2``): a generated module of the file (see
    ``_is_generated_module_of``), so not a package that star-imports it, as Delta Lake's
    ``delta/connect/proto/__init__.py`` imports ``base_pb2``, nor any module that it does not
    name."""
    if _PROTOBUF is None:  # pragma: no cover - the runtime needs none of this
        return None
    registered = _PROTOBUF.database._classes
    tops = set()
    for descriptor in file.message_types_by_name.values():
        module_name = getattr(registered.get(descriptor), "__module__", None)
        if not isinstance(module_name, str):
            continue
        module = sys.modules.get(module_name)
        if module is not None and _is_generated_module_of(module, file):
            return module  # type: ignore[no-any-return]
        tops.add(module_name.partition(".")[0])
    if near:
        tops.add(near.partition(".")[0])
    for module in _modules_of_packages(tops):
        if _is_generated_module_of(module, file):
            return module  # type: ignore[no-any-return]
    return None


# Of a hold of files of this thread (see ``_held_together``): the imported modules of each
# top-level package that it found (see ``_modules_of_packages``), and the classes that its
# take-overs replaced in the generated modules of each (see ``_rebind``), by the name of the
# package; else None.
_HOLDING = threading.local()


@contextmanager
def _held_together() -> Generator[None, None, None]:
    """Find the imported modules of a top-level package once in the block (of this thread, which
    imports none), and rebind the names of the modules of each package once, at its end, to the
    classes that the block's take-overs replaced (see ``_rebind``), rather than of each file."""
    if getattr(_HOLDING, "modules", None) is not None:
        yield
        return
    _HOLDING.modules, _HOLDING.replaced = {}, {}
    try:
        yield
        for top, replaced in _HOLDING.replaced.items():
            _rebind_in_package(replaced, top)
    finally:
        _HOLDING.modules = _HOLDING.replaced = None


def _modules_of_packages(tops: set[str]) -> list[_Any]:
    """The imported modules of the top-level packages ``tops`` (and the packages), of
    ``sys.modules``: of a hold of files (see ``_held_together``), those that it found first."""
    if not tops:
        return []
    known = getattr(_HOLDING, "modules", None)
    if known is None:
        known = {}
    missing = {top for top in tops if top not in known}
    if missing:
        prefixes = tuple(top + "." for top in missing)
        modules = sys.modules
        names = [
            name
            for name in list(modules)
            if isinstance(name, str) and (name in missing or name.startswith(prefixes))
        ]
        for top in missing:
            known[top] = []
        for name in names:
            module = modules.get(name)
            if module is not None:
                known[name.partition(".")[0]].append(module)
    return [module for top in tops for module in known[top]]


def _rebind(replaced: dict[int, tuple[_Any, _Any]], module_name: str) -> None:
    """Rebind each name of the globals of the imported modules of the package of the module
    ``module_name`` (of its top-level package, ``delta`` of ``delta.connect.proto.base_pb2``) that
    is bound to an object of ``replaced`` (protobuf's class of a message, by its ``id``, with the
    runtime's class of it), as a module of the package that star-imports a generated module binds
    its classes (Delta Lake's ``delta/connect/proto/__init__.py`` binds those of ``base_pb2``): no
    other name, nor any module of another package, changes. Of a hold of files (see
    ``_held_together``), at its end."""
    if not replaced:
        return
    top = module_name.partition(".")[0]
    pending = getattr(_HOLDING, "replaced", None)
    if pending is not None:
        pending.setdefault(top, {}).update(replaced)
        return
    _rebind_in_package(replaced, top)


def _rebind_in_package(replaced: dict[int, tuple[_Any, _Any]], top: str) -> None:
    """Rebind the names of the imported modules of the top-level package ``top`` that are bound
    to an object of ``replaced`` (see ``_rebind``)."""
    for module in _modules_of_packages({top}):
        namespace = _globals_of(module)
        if namespace is None:
            continue
        for key, value in list(namespace.items()):
            found = replaced.get(id(value))
            if found is not None and found[0] is value:
                namespace[key] = found[1]


def _take_over(
    file: _Any, module: types.ModuleType, replaced: dict[int, tuple[_Any, _Any]]
) -> None:
    """Give ``module``, the generated module of ``file``, of protobuf's classes of the messages of
    the file, the runtime's classes of them, which protobuf's symbol database gives too, recording
    in ``replaced`` each class that they replace. A name of the module that is bound to another
    object (a class of the application of the name of a message) stays."""
    namespace = _globals_of(module) or {}
    for name, descriptor in file.message_types_by_name.items():
        cls = _OWNED.get(descriptor)
        old = namespace.get(name)
        if cls is None or old is None or old is cls:
            continue
        if not (isinstance(old, type) and vars(old).get("DESCRIPTOR") is descriptor):
            continue
        replaced[id(old)] = (old, cls)
        namespace[name] = cls
    _register_in_protobuf(file)


def _register_in_protobuf(file: _Any) -> None:
    """Register the runtime's classes of the messages of ``file`` in protobuf's symbol database,
    in place of protobuf's, as its builder registers those of a generated module: its
    ``GetSymbol``, ``GetMessages`` and the load of a pickle of a nested type give them."""
    database = _PROTOBUF.database._classes  # type: ignore[union-attr]

    def register(descriptor: _Any) -> None:
        cls = _OWNED.get(descriptor)
        if cls is not None:
            previous = database.get(descriptor)
            if previous is not None and not isinstance(previous, MessageMeta):
                _PROTOBUFS.setdefault(descriptor, previous)
            database[descriptor] = cls
        for nested in descriptor.nested_types:
            register(nested)

    for descriptor in file.message_types_by_name.values():
        register(descriptor)


def _module_name_of_file(file: _Any) -> Optional[str]:
    """The name of the generated module of the file ``file`` of protobuf's default pool, of
    protobuf's class of one of its messages that protobuf's symbol database has (which protobuf's
    builder registers), else None, as upb's class of a type of no module (of
    ``message_factory.GetMessageClass``) has."""
    registered = _PROTOBUF.database._classes  # type: ignore[union-attr]
    for descriptor in file.message_types_by_name.values():
        module_name = getattr(registered.get(descriptor), "__module__", None)
        if isinstance(module_name, str):
            return module_name
    return None


def _own_file(file: _Any) -> None:
    """Make the runtime's classes of the messages of ``file``, of their nested messages too, the
    classes that protobuf's lookups give (see ``_own``): also those that the runtime made of the
    file as a file of another package that a plugin's file imports (see ``_hold_files``), once a
    plugin's file of the file's own package imports it."""

    def own(descriptor: _Any) -> None:
        cls = _CLASSES.get(descriptor.full_name)
        if cls is not None and cls.__dict__.get("DESCRIPTOR") is descriptor:
            _own(cls)
        for nested in descriptor.nested_types:
            own(nested)

    for descriptor in file.message_types_by_name.values():
        own(descriptor)


def _hold_files(file: _Any, module_name: Optional[str]) -> bool:
    """Give the messages of ``file``, a file of protobuf's default pool, the runtime's classes, of
    the module ``module_name`` (the generated module of the file, which its builder builds), when
    it imports a proto of Spark Connect, directly or through the files that it imports, as a
    plugin's does (Delta Lake's ``delta/connect/relations.proto``), or a file that the runtime
    holds so: their message fields then take the messages of this package as they are, as
    protobuf's take PySpark's. Whether it does.

    The files are added to the pool of the runtime first, with those that it imports that the
    runtime does not hold. Those of the package of the file, of a package (Delta Lake's
    ``delta/connect/base.proto``, which imports no proto of Spark Connect), and those that import a
    proto of Spark Connect or a file that the runtime holds, get the runtime's classes too, of their
    generated modules, which the runtime takes over (see ``_take_over``) if protobuf's builder built
    them before, with the names of the modules of their package bound to protobuf's classes that
    they replace; so do the files that protobuf's builder built before that import those, as Delta
    Lake's ``commands.proto`` imports ``base.proto`` (see ``_hold_importers``). A file whose
    module the runtime does not find keeps protobuf's classes. The files of other packages, of no
    package, and Google's (``google/protobuf/duration.proto``, ``google/rpc/status.proto``), keep
    protobuf's classes in their modules and protobuf's lookups, as the modules of other libraries
    that import them do: the runtime has classes of their messages of its own, which its message
    fields of their types give, which pickle as protobuf's classes (see ``_reduce_as_protobufs``),
    and its message fields take protobuf's messages of their types, of their bytes. A file that the
    runtime does not hold as protobuf does (one that declares extensions, or that imports one)
    keeps protobuf's classes, with the files that it imports: those that the runtime holds get
    protobuf's classes back (see ``_release_imports_of``)."""
    name = file.name
    if name in _HELD:
        return True
    if name in _UNHELD or _is_protobufs_own(file) or not _imports_held(file):
        return False
    with _HOOK_LOCK:
        if name in _HELD:
            return True
        if name in _UNHELD:  # pragma: no cover - of another thread
            return False
        files = _closure_of(file)
        try:
            _pyspark._proto_register_files([f.serialized_pb for f in files])
        except ValueError:
            _UNHELD.add(name)
            # upb's classes of the file take only upb's messages of the files that it imports.
            _release_imports_of(file)
            return False
        taken: list[tuple[_Any, Optional[types.ModuleType]]] = []
        with _held_together():
            with _registered_together():
                for held in files:
                    if held.name in _HELD:
                        continue
                    # As upb's class of the type: of the module that protobuf's builder named.
                    of_module = _module_name_of_file(held)
                    module = None
                    if held.name == name:
                        owned = True
                        if module_name is not None:
                            of_module = module_name
                    else:
                        owned = (
                            not _is_googles(held.package)
                            and held.name not in _UNHELD
                            and (
                                (held.package != "" and held.package == file.package)
                                or _imports_held(held)
                            )
                        )
                        if owned:
                            module = _imported_module_of_file(held, module_name)
                            # protobuf's classes of a module that the runtime does not find stay.
                            owned = module is not None or of_module is None
                    for descriptor in held.message_types_by_name.values():
                        _make_class(descriptor.full_name, descriptor.name, of_module, owned)
                    if owned:
                        _own_file(held)
                        _HELD.add(held.name)
                        if held.name != name:
                            taken.append((held, module))
            _held_more()
            for held, module in taken:
                if module is None:
                    _register_in_protobuf(held)
                    continue
                replaced: dict[int, tuple[_Any, _Any]] = {}
                _take_over(held, module, replaced)
                _rebind(replaced, module.__name__)
            if taken:
                _hold_importers()
    return True


def _hold_importers() -> None:
    """Give the runtime's classes to the generated modules that protobuf's builder built of files
    that import a file that the runtime holds with its classes (see ``_hold_files``), as Delta
    Lake's ``commands.proto`` imports ``base.proto``, which a plugin's file of its package took
    over: their message fields take the runtime's messages of the file then, and the runtime's
    those of theirs. The modules are those of the classes of protobuf's symbol database (which
    protobuf's builder registers), so no other module is read. A file that the runtime cannot hold
    (of extensions), or whose module it does not find, keeps protobuf's classes, and the files that
    the runtime holds that it imports get protobuf's classes back (see ``_release_imports_of``),
    before the runtime holds any other: as upb's classes of the file take only upb's messages of
    the types of the files that it imports, so that the others then may import none."""
    if _PROTOBUF is None:  # pragma: no cover - the runtime needs none of this
        return
    while True:
        found = _built_importers()
        if not found:
            return
        refused = [file for file in found if any(map(_declares_unheld, _closure_of(file)))]
        if refused:
            for file in refused:
                _UNHELD.add(file.name)
                _release_imports_of(file)
            continue
        # The files found at once, of one read of the symbol database; those that the runtime holds
        # then may make others importers of a file that it holds, which the next read finds.
        for file in found:
            name = file.name
            if name in _HELD or name in _UNHELD or not _imports_held(file):
                continue
            module = _imported_module_of_file(file)
            held = module is not None and _hold_files(
                file, _module_name_of_file(file) or module.__name__  # type: ignore[union-attr]
            )
            if not held:
                _UNHELD.add(name)
                _release_imports_of(file)
                continue
            replaced: dict[int, tuple[_Any, _Any]] = {}
            _take_over(file, module, replaced)  # type: ignore[arg-type]
            _rebind(replaced, module.__name__)  # type: ignore[union-attr]


def _built_importers() -> list[_Any]:
    """The files of the classes of protobuf's symbol database (whose modules protobuf's builder
    built) that import a file that the runtime holds with its classes, directly or through the
    files that they import, and that the runtime neither holds nor refused, in their order."""
    found: dict[str, _Any] = {}
    pool = _PROTOBUF.pool  # type: ignore[union-attr]
    for descriptor, cls in list(_PROTOBUF.database._classes.items()):  # type: ignore[union-attr]
        try:
            if cls is _OWNED.get(descriptor):
                continue
            file = descriptor.file
            name = file.name
            if name in found or name in _HELD or name in _UNHELD:
                continue
            if file.pool is not pool or _is_protobufs_own(file):
                continue
            if _imports_held(file):
                found[name] = file
        except Exception:  # noqa: BLE001 - an entry of another kind
            continue
    return list(found.values())


def _release_imports_of(file: _Any) -> None:
    """Give protobuf's classes back to the files that the runtime holds with its classes that
    ``file``, a file that keeps protobuf's classes, imports (or that a file that it imports imports
    publicly), and to those that they import (see ``_release``): upb's classes of ``file`` take only
    upb's messages of the types of the files that it imports, which the modules of these files then
    give again, whether ``file`` was built before the runtime took them over or after. A proto of
    Spark Connect, and a file that imports one (a plugin's), keep the runtime's classes."""
    seen: set[str] = set()
    released = False
    stack = list(file.dependencies)
    while stack:
        imported = stack.pop()
        if imported.name in seen:
            continue
        seen.add(imported.name)
        if imported.name in _HELD and not _imports_spark_connect(imported):
            _release(imported)
            released = True
            stack.extend(imported.dependencies)
        else:
            stack.extend(imported.public_dependencies)
    if released:
        _IMPORTS_HELD.clear()
        _NOT_OWNED.clear()


def _release(file: _Any) -> None:
    """Give the messages of ``file``, a file that the runtime holds with its classes and that
    imports no proto of Spark Connect, protobuf's classes again (see ``_release_imports_of``), as if
    the runtime had not held it: in protobuf's lookups and symbol database, in its module and in the
    names of the modules of its package that are bound to the runtime's classes (see ``_rebind``),
    and never again. The runtime's classes of its types are then of types that keep protobuf's
    classes, as those of the files of other packages that a plugin's file imports (see
    ``_make_class``): their messages pickle as protobuf's, and the message fields of the runtime's
    classes take protobuf's messages of their types."""
    _HELD.discard(file.name)
    _UNHELD.add(file.name)
    module = _imported_module_of_file(file)
    database = _PROTOBUF.database._classes  # type: ignore[union-attr]
    restored: dict[int, tuple[_Any, _Any]] = {}

    def release(descriptor: _Any) -> None:
        cls = _OWNED.pop(descriptor, None)
        if cls is not None:
            # Of protobuf's pure-Python implementation, a class of the runtime that the descriptor
            # holds (of a type of no protobuf's class) makes way for one that protobuf makes.
            python = _PROTOBUF.python  # type: ignore[union-attr]
            if python and getattr(descriptor, "_concrete_class", None) is cls:
                descriptor._concrete_class = None
            protobuf_class = _PROTOBUFS.pop(descriptor, None) or _protobuf_class_of(descriptor)
            database[descriptor] = protobuf_class
            cls.__reduce__ = _reduce_as_protobufs  # type: ignore[method-assign]
            _PROTOBUF_CLASSES.pop(cls, None)
            restored[id(cls)] = (cls, protobuf_class)
        for nested in descriptor.nested_types:
            release(nested)

    for descriptor in file.message_types_by_name.values():
        release(descriptor)
    if module is not None:
        namespace = _globals_of(module) or {}
        for name in file.message_types_by_name:
            found = restored.get(id(namespace.get(name)))
            if found is not None and found[0] is namespace.get(name):
                namespace[name] = found[1]
        _restore_bindings(restored, module.__name__)


def _restore_bindings(restored: dict[int, tuple[_Any, _Any]], module_name: str) -> None:
    """Rebind the names of the modules of the package of the module ``module_name`` that are bound
    to a class of ``restored`` (the runtime's class of a type that a release gives protobuf's class
    again, by its ``id``, with protobuf's class) to protobuf's class, as ``_rebind`` does, and keep
    the rebinds of a hold of files (see ``_held_together``) from binding protobuf's classes of them
    to the runtime's."""
    if not restored:
        return
    pending = getattr(_HOLDING, "replaced", None)
    if pending is not None:
        protobufs = {id(protobuf) for _, protobuf in restored.values()}
        replaced = pending.get(module_name.partition(".")[0], {})
        for key in [key for key in replaced if key in protobufs]:
            del replaced[key]
    _rebind(restored, module_name)


def _class_for(descriptor: _Any, module_name: Optional[str]) -> Optional[type]:
    """The class of the runtime that protobuf's metaclass, called of ``descriptor``, gives in
    place of its own, and protobuf's ``GetMessageClass`` too: of a message type of protobuf's
    default pool of a file of Spark Connect or that ``_hold_files`` gives the runtime's classes
    (of the module ``module_name``). None of any other descriptor, which protobuf's metaclass
    takes, and which is not looked at again."""
    if _PROTOBUF is None or not _USE_PROTOBUF_DESCRIPTORS:
        return None
    try:
        cls = _OWNED.get(descriptor)
        if cls is not None or descriptor in _NOT_OWNED:
            return cls
    except TypeError:  # an object that is no descriptor, which protobuf's function takes
        return None
    try:
        file = descriptor.file
        if file.pool is not _PROTOBUF.pool:
            return None
    except AttributeError:
        return None
    if _hold_files(file, module_name):
        cls = _OWNED.get(descriptor)
        if cls is not None:
            return cls
    _NOT_OWNED.add(descriptor)
    return None


class _WrapperType(type):
    """The metaclass of the wrappers of protobuf's classes and functions of this module: what
    ``isinstance`` and ``issubclass`` tell of the wrapper is what they tell of protobuf's class."""

    def __instancecheck__(cls, instance: _Any) -> bool:
        return isinstance(instance, cls._protobuf_class)

    def __subclasscheck__(cls, subclass: type) -> bool:
        return issubclass(subclass, cls._protobuf_class)


def _wrap_metaclass(protobuf_class: type) -> type:
    """A wrapper of ``protobuf_class``, protobuf's metaclass of its message classes
    (``reflection.GeneratedProtocolMessageType``, upb's ``MessageMeta``), that gives the runtime's
    class of a message type that the runtime holds (see ``_class_for``), as upb's gives the one
    class of a type, and protobuf's class of any other."""

    class GeneratedProtocolMessageType(metaclass=_WrapperType):
        __doc__ = protobuf_class.__doc__
        _protobuf_class = protobuf_class
        _DESCRIPTOR_KEY = "DESCRIPTOR"

        def __new__(cls, name: str, bases: tuple, namespace: dict, **kwargs: _Any) -> _Any:
            descriptor = namespace.get("DESCRIPTOR") if isinstance(namespace, dict) else None
            module_name = None if descriptor is None else namespace.get("__module__")
            made = None if descriptor is None else _class_for(descriptor, module_name)
            if made is not None:
                # A class that the runtime made of no module (of a lookup of its type before its
                # module built, such as ``message_factory.GetMessageClass``) is of the module that
                # the builder builds, as upb's, so that the module is found of it later (see
                # ``_imported_module_of_file``).
                if made.__dict__.get("__module__") is None and isinstance(module_name, str):
                    made.__module__ = module_name
                return made
            return protobuf_class(name, bases, namespace, **kwargs)

    GeneratedProtocolMessageType.__module__ = protobuf_class.__module__
    GeneratedProtocolMessageType.__qualname__ = protobuf_class.__qualname__
    return GeneratedProtocolMessageType


def _hook_message_types() -> None:
    """Make protobuf's metaclass of its message classes (``reflection.GeneratedProtocolMessageType``
    and the one of ``message_factory``, one wrapper of the two, as upb has one metaclass) and its
    ``message_factory.GetMessageClass`` give the runtime's classes of the message types that the
    runtime holds, when protobuf imports.

    A module that protoc generates makes the classes of the messages of its file with that
    metaclass, of protobuf's default pool, after it imports the modules of the files that its file
    imports, which import this package and so install the hooks first, for a file that imports the
    protos of Spark Connect: through protobuf's builder (``_BuildMessage``), as the modules of
    protoc 3.20 and later do, or by itself, as the modules of an older protoc do. Such a module so
    gets the runtime's classes of the messages of its file (see ``_hold_files``), which protobuf's
    builder registers in protobuf's symbol database, and ``GetMessageClass`` gives them, as upb's
    of its one class of a type. A file of another pool, such as one that a test builds, keeps
    protobuf's classes. One hook wraps protobuf's own function and metaclass, of the runtime of the
    last install. A protobuf that lacks them is passed over, as the drop-in itself needs none of
    this."""
    if _PROTOBUF is None:  # pragma: no cover - the runtime needs none of this
        return
    reflection, message_factory = _PROTOBUF.reflection, _PROTOBUF.message_factory
    wrappers: dict[int, type] = {}

    def wrapper_of(metaclass: type) -> type:
        metaclass = getattr(metaclass, "_protobuf_class", metaclass)
        wrapper = wrappers.get(id(metaclass))
        if wrapper is None:
            wrapper = wrappers[id(metaclass)] = _wrap_metaclass(metaclass)
        return wrapper

    metaclass = getattr(reflection, "GeneratedProtocolMessageType", None)
    if metaclass is not None:
        reflection.GeneratedProtocolMessageType = wrapper_of(metaclass)
    factory_metaclass = getattr(message_factory, "_GENERATED_PROTOCOL_MESSAGE_TYPE", None)
    if factory_metaclass is not None:
        message_factory._GENERATED_PROTOCOL_MESSAGE_TYPE = wrapper_of(factory_metaclass)
    get_message_class = getattr(message_factory, "GetMessageClass", None)
    if get_message_class is None:  # pragma: no cover - a protobuf before 4.21
        return
    get_message_class = getattr(get_message_class, "_protobuf_function", get_message_class)

    default_pool = _PROTOBUF.pool

    @functools.wraps(get_message_class)
    def GetMessageClass(descriptor: _Any) -> _Any:
        # Of a type known to keep protobuf's class, or of another pool, protobuf's function at once.
        try:
            if descriptor in _NOT_OWNED:
                return get_message_class(descriptor)
            cls = _OWNED.get(descriptor)
        except TypeError:  # an object that is no descriptor, which protobuf's function takes
            return get_message_class(descriptor)
        if cls is not None:
            return cls
        try:
            if descriptor.file.pool is not default_pool:
                return get_message_class(descriptor)
        except AttributeError:
            return get_message_class(descriptor)
        cls = _class_for(descriptor, None)
        if cls is not None:
            return cls
        return get_message_class(descriptor)

    GetMessageClass._protobuf_function = get_message_class  # type: ignore[attr-defined]
    message_factory.GetMessageClass = GetMessageClass


def _hold_built_modules() -> None:
    """Give the runtime's classes to the generated modules that protobuf's builder built before
    the hooks were installed, as only a module that imports another copy of Apache Spark's
    generated modules of the protos (of the same files) can: a module of a proto of Spark Connect
    gets the classes of the package, as upb gives the module of a file built again the class that
    it made of it, and one of a file that imports them (or a file that the runtime holds) gets the
    runtime's (see ``_hold_files``), with the names of the modules of its package bound to the
    classes that they replace. The modules are those of the classes of protobuf's symbol database
    (which protobuf's builder registers), so no other module is read."""
    if _PROTOBUF is None or not _USE_PROTOBUF_DESCRIPTORS:  # pragma: no cover
        return
    with _HOOK_LOCK:
        files: dict[str, _Any] = {}
        for descriptor, cls in list(_PROTOBUF.database._classes.items()):
            try:
                if cls is _OWNED.get(descriptor):
                    continue
                file = descriptor.file
                if file.pool is not _PROTOBUF.pool or _is_protobufs_own(file):
                    continue
                files.setdefault(file.name, file)
            except Exception:  # noqa: BLE001 - an entry of another kind
                continue
        with _held_together():
            for name, file in files.items():
                module = _imported_module_of_file(file)
                if name in _SPARK_CONNECT_FILES:
                    held = module is not None and not _is_exported(module.__name__)
                else:
                    # A module that the runtime does not find keeps protobuf's classes.
                    held = (
                        name not in _EMBEDDED_FILES
                        and module is not None
                        and _hold_files(file, _module_name_of_file(file) or module.__name__)
                    )
                if held:
                    replaced: dict[int, tuple[_Any, _Any]] = {}
                    _take_over(file, module, replaced)  # type: ignore[arg-type]
                    _rebind(replaced, module.__name__)  # type: ignore[union-attr]


class _UnknownFieldSetType(type):
    """The metaclass of ``UnknownFieldSet``, the wrapper of protobuf's class: what ``isinstance``
    and ``issubclass`` tell of the wrapper is what they tell of protobuf's class, of which the
    wrapper makes its sets."""

    def __instancecheck__(cls, instance: _Any) -> bool:
        return cls._protobuf_class is not None and isinstance(instance, cls._protobuf_class)

    def __subclasscheck__(cls, subclass: type) -> bool:
        return cls._protobuf_class is not None and issubclass(subclass, cls._protobuf_class)


class UnknownFieldSet(metaclass=_UnknownFieldSetType):
    """protobuf's ``unknown_fields.UnknownFieldSet``, of the messages of this runtime too (see
    ``_hook_unknown_field_set``): the set of the unknown fields of a message, as protobuf's class
    of the implementation of the install (upb's) makes it."""

    __slots__ = ()
    # protobuf's class, and its class of google.protobuf.Empty, which the hook sets.
    _protobuf_class: _Any = None
    _empty_class: _Any = None

    def __new__(cls, *args: _Any, **kwargs: _Any) -> _Any:
        # protobuf's class takes a message of the runtime as a message of protobuf's
        # google.protobuf.Empty, which has no fields, whose unknown fields are those of the
        # message; it takes the arguments as they are otherwise, with its errors of them.
        if len(args) == 1 and not kwargs and not isinstance(args[0], Message):
            return cls._protobuf_class(args[0])

        def taken(value: _Any) -> _Any:
            if isinstance(value, Message):
                return cls._empty_class.FromString(_pyspark._proto_unknown_fields(value))
            return value

        return cls._protobuf_class(
            *map(taken, args), **{name: taken(value) for name, value in kwargs.items()}
        )


def _hook_unknown_field_set() -> None:
    """Make ``UnknownFieldSet`` of protobuf's ``google.protobuf.unknown_fields`` the wrapper of this
    module, when protobuf is installed, so that it takes the messages of this runtime, which
    protobuf's class (upb's, or the pure-Python one) refuses ("Expected a message object"): the set
    of a message of the runtime is protobuf's set of a message of protobuf's ``Empty`` that holds
    the encoded unknown fields of the message, the fields, values and order of upb's set of a
    message of upb that read the same bytes. Of any other argument, it is protobuf's set, or
    protobuf's error. ``text_format`` reads the set of each message of the module's attribute, so
    that ``MessageToString(message, print_unknown_fields=True)`` prints them. One wrapper wraps
    protobuf's own class, of the runtime of the last install. A protobuf that has no such module,
    or no ``Empty``, is passed over, as the drop-in itself needs none of this."""
    try:
        from google.protobuf import empty_pb2, unknown_fields

        protobuf_class = unknown_fields.UnknownFieldSet
        empty_class = empty_pb2.Empty
    except Exception:  # pragma: no cover  # noqa: BLE001 - no protobuf, or one that fails or lacks it
        return
    UnknownFieldSet._protobuf_class = getattr(protobuf_class, "_protobuf_class", protobuf_class)
    UnknownFieldSet._empty_class = empty_class
    unknown_fields.UnknownFieldSet = UnknownFieldSet


# The limits of the levels of messages of upb's parsers: its default
# (``kUpb_WireFormat_DefaultDepthLimit``), and of once its ``SetAllowOversizeProtos(True)`` is
# called (``UINT16_MAX``).
_UPB_DEPTH_LIMIT = 100
_UPB_OVERSIZE_DEPTH_LIMIT = 65535


def _upb_depth_limit() -> int:
    """The limit of the levels of messages of upb's parsers: 65535 once upb's
    ``SetAllowOversizeProtos(True)`` is called, else 100, which upb tells of no other way than of a
    parse: whether upb parses a ``google.protobuf.ListValue`` of 102 levels of messages below it."""
    from google.protobuf import struct_pb2

    size, prefixes = 0, []
    for _ in range(51):
        # ListValue.values (1) of a Value, whose list_value (6) is the ListValue below.
        value = b"\x32" + _varint(size)
        size += len(value)
        values = b"\x0a" + _varint(size)
        size += len(values)
        prefixes.append(values + value)
    try:
        struct_pb2.ListValue.FromString(b"".join(reversed(prefixes)))
    except Exception:  # noqa: BLE001 - upb's DecodeError past its limit
        return _UPB_DEPTH_LIMIT
    return _UPB_OVERSIZE_DEPTH_LIMIT


def _python_depth_limit() -> int:
    """The limit of the levels of messages of the parsers of protobuf's pure-Python implementation,
    which its ``decoder.SetRecursionLimit`` sets (``decoder._recursion_limit``, 100 of none)."""
    from google.protobuf.internal import decoder

    return int(getattr(decoder, "_recursion_limit", _UPB_DEPTH_LIMIT))


def _varint(value: int) -> bytes:
    """The bytes of ``value`` as a varint of the wire format."""
    out = bytearray()
    while value > 0x7F:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _hook_depth_limit() -> None:
    """Make the parses of the runtime follow the limit of the levels of messages of the parsers of
    the ``protobuf`` of the process: of upb, its ``SetAllowOversizeProtos``
    (``api_implementation._c_module``), once called with True of 65535 levels, of 100 before; of
    protobuf's pure-Python implementation, its ``decoder.SetRecursionLimit`` (see ``codec.rs``).
    The runtime takes the limit when it installs, which tells a call before; wraps the function,
    which tells a call after; and asks for the limit again when a parse reaches its limit, which
    tells a call of a reference to the function taken before the install."""
    if _PROTOBUF is None:  # pragma: no cover - the runtime needs none of this
        return
    if _PROTOBUF.python:
        try:
            from google.protobuf.internal import decoder

            owner, name = decoder, "SetRecursionLimit"
            protobuf_function = decoder.SetRecursionLimit
        except Exception:  # pragma: no cover  # noqa: BLE001 - a decoder of no such function
            return
        probe = _python_depth_limit

        def limit_of(argument: _Any) -> int:
            return _python_depth_limit()

    else:
        try:
            owner, name = _PROTOBUF.api_implementation._c_module, "SetAllowOversizeProtos"
            protobuf_function = owner.SetAllowOversizeProtos
        except Exception:  # noqa: BLE001 - a protobuf of no such function
            return
        probe = _upb_depth_limit

        def limit_of(argument: _Any) -> int:
            return _UPB_OVERSIZE_DEPTH_LIMIT if argument else _UPB_DEPTH_LIMIT

    protobuf_function = getattr(protobuf_function, "_protobuf_function", protobuf_function)

    @functools.wraps(protobuf_function)
    def hooked(argument: _Any) -> _Any:
        result = protobuf_function(argument)
        _pyspark._proto_set_depth_limit(limit_of(argument))
        return result

    hooked._protobuf_function = protobuf_function  # type: ignore[attr-defined]
    setattr(owner, name, hooked)
    try:
        _pyspark._proto_set_depth_limit(probe())
    except Exception:  # pragma: no cover  # noqa: BLE001 - a protobuf without struct_pb2
        return
    _pyspark._proto_set_depth_probe(probe)


def install(package: dict[str, _Any]) -> None:
    """Export the messages of Spark Connect into ``package``, the globals of
    ``pyspark.sql.connect.proto``, as its ``from ..._pb2 import *`` of the generated modules does
    (so its ``DESCRIPTOR`` is that of the last, ``pipelines_pb2``), make each generated module
    (``pyspark.sql.connect.proto.base_pb2``, ...) the one of this runtime, put the modules of the
    well-known types of the runtime (``pyspark.sql.connect.proto._rust.any_pb2`` and
    ``timestamp_pb2``) in ``sys.modules``, so that their earlier pickles load, and, when protobuf
    imports, register the classes of the runtime in its symbol database, give the modules that
    protoc generates of other protos that import them the classes of the runtime, make its
    ``UnknownFieldSet`` take the messages of the runtime, and make the parses of the runtime follow
    the limit of the levels of messages of protobuf's parsers (see ``_hook_depth_limit``)."""
    prefix = "pyspark.sql.connect.proto."
    for module_name, module in _MODULES.items():
        sys.modules[module_name] = module
        if _is_exported(module_name):
            package[module_name[len(prefix) :]] = module
    for stem in _EXPORTED:
        for name, value in vars(_MODULES[f"{prefix}{stem}_pb2"]).items():
            if not name.startswith("_"):
                package[name] = value
    # One finder of the modules, of the runtime of the last install.
    sys.meta_path[:] = [
        f
        for f in sys.meta_path
        if not (type(f).__name__ == "_Finder" and type(f).__module__ == __name__)
    ]
    sys.meta_path.insert(0, _Finder())
    _register_classes_in_protobuf()
    _hook_message_types()
    _hold_built_modules()
    _hook_unknown_field_set()
    _hook_depth_limit()


@contextmanager
def add_pipeline_analysis_context(
    spark: _Any, dataflow_graph_id: str, flow_name: Optional[str]
) -> Generator[None, None, None]:
    """``pyspark.pipelines.add_pipeline_analysis_context`` of this runtime's ``Any``: the context
    manager that adds the ``PipelineAnalysisContext`` extension to the user context of the
    requests, for the analysis of a pipeline, as the vendored one does with protobuf's ``Any``
    (see ``pyspark._proto_hooks``)."""
    extension_id = None
    client = spark.client
    try:
        import pyspark.sql.connect.proto as pb2

        analysis_context = pb2.PipelineAnalysisContext(
            dataflow_graph_id=dataflow_graph_id, flow_name=flow_name
        )
        extension = Any()
        extension.Pack(analysis_context)
        extension_id = client.add_threadlocal_user_context_extension(extension)
        yield
    finally:
        # extension_id stays None if registering the extension above failed; skip cleanup in that
        # case so we don't call remove_user_context_extension(None) and mask the original error.
        if extension_id is not None:
            client.remove_user_context_extension(extension_id)
