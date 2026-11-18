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
Connect, with the API of protobuf's Python messages, whose fields, descriptors and wire format the
Rust runtime holds and encodes. Each class has the ``DESCRIPTOR`` of its message and, as an
attribute, the classes of its nested messages and its enums, with their values, as the modules
that protoc generates give them; the classes of the well-known types ``google.protobuf.Any`` and
``google.protobuf.Timestamp`` have the methods of protobuf's. Each module of a proto
(``pyspark.sql.connect.proto.base_pb2``, ...) has the names of the generated module of the proto.

Where the ``protobuf`` package is installed, the files of the protos are also added to its default
descriptor pool, as the generated modules add them (see ``_add_files_to_protobuf``), the modules
that protoc generates of other protos that import them get the classes of this runtime (see
``_hook_protobuf_builder``), and its ``unknown_fields.UnknownFieldSet`` takes the messages of the
runtime (see ``_hook_unknown_field_set``).
"""
import calendar
import collections.abc
import datetime
import functools
import importlib
import importlib.abc
import importlib.machinery
import sys
import threading
import types
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from typing import Any as _Any
from typing import Optional, Union

from pyspark import _pyspark

Message = _pyspark.ProtoMessage
DecodeError = _pyspark.ProtoDecodeError
RepeatedScalarContainer = _pyspark.ProtoRepeatedScalarContainer
RepeatedCompositeContainer = _pyspark.ProtoRepeatedCompositeContainer
MapContainer = _pyspark.ProtoMapContainer
MessageMapContainer = _pyspark.ProtoMessageMapContainer

# As upb's containers: a repeated field is a MutableSequence, and a map a MutableMapping.
collections.abc.MutableSequence.register(RepeatedScalarContainer)
collections.abc.MutableSequence.register(RepeatedCompositeContainer)
collections.abc.MutableMapping.register(MapContainer)

__all__ = ["Message", "DecodeError", "EnumTypeWrapper", "install", "add_pipeline_analysis_context"]


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


# The methods of the classes of the well-known types, as protobuf's classes of them have.
_WELL_KNOWN_BASES: dict[str, tuple[type, ...]] = {
    "google.protobuf.Any": (_AnyMethods,),
    "google.protobuf.Timestamp": (_TimestampMethods,),
}

# The class of each message, by the full name of the message.
_CLASSES: dict[str, type] = {}


def _make_class(full_name: str, qualname: str, module_name: str) -> type:
    descriptor = _pyspark._proto_message_descriptor(full_name)
    cls = type(
        descriptor.name,
        _WELL_KNOWN_BASES.get(full_name, ()) + (Message,),
        {
            "__slots__": (),
            "DESCRIPTOR": descriptor,
            "__module__": module_name,
            "__qualname__": qualname,
        },
    )
    _CLASSES[full_name] = cls
    # As upb's classes: the number of each field, as ``Literal.STRING_FIELD_NUMBER``.
    for field in descriptor.fields:
        setattr(cls, field.name.upper() + "_FIELD_NUMBER", field.number)
    for nested in descriptor.nested_types:
        setattr(
            cls,
            nested.name,
            _make_class(nested.full_name, qualname + "." + nested.name, module_name),
        )
    for enum in descriptor.enum_types:
        setattr(cls, enum.name, EnumTypeWrapper(enum))
        for value in enum.values:
            setattr(cls, value.name, value.number)
    return cls


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
    """The finder of the modules of the protos of Spark Connect of this runtime, which have no
    files: ``pyspark.sql.connect.proto.base_pb2`` and the others."""

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


def _build() -> dict[str, types.ModuleType]:
    modules = {}
    files = []
    for file_name, package, messages, enums in _pyspark._proto_files():
        module_name = _module_name(file_name, package)
        module = types.ModuleType(module_name)
        module.__doc__ = f"The messages of {file_name}, of the drop-in's Rust protobuf runtime."
        module.__spec__ = importlib.machinery.ModuleSpec(module_name, _LOADER)
        module.__loader__ = _LOADER
        module.__package__ = module_name.rpartition(".")[0]
        file_descriptor = _pyspark._proto_file_descriptor(file_name)
        module.DESCRIPTOR = file_descriptor
        _add_descriptors(module, file_descriptor)
        for full_name in messages:
            name = full_name.rsplit(".", 1)[-1]
            setattr(module, name, _make_class(full_name, name, module_name))
        for full_name in enums:
            descriptor = _pyspark._proto_enum_descriptor(full_name)
            setattr(module, descriptor.name, EnumTypeWrapper(descriptor))
            for value in descriptor.values:
                setattr(module, value.name, value.number)
        modules[file_name] = module
        files.append(file_descriptor)
    # The module of each file that a file imports, under protoc's name of it.
    for file_descriptor in files:
        for dependency in file_descriptor.dependencies:
            setattr(
                modules[file_descriptor.name],
                _module_alias(dependency.name),
                modules[dependency.name],
            )
    _pyspark._proto_register_classes(_CLASSES)
    return {module.__name__: module for module in modules.values()}


_MODULES = _build()

# The well-known types that the protos of Spark Connect import.
Any = _CLASSES["google.protobuf.Any"]
Timestamp = _CLASSES["google.protobuf.Timestamp"]


# The generated modules that ``pyspark.sql.connect.proto`` star-imports, in its order.
_EXPORTED = ("base", "types", "commands", "expressions", "relations", "catalog", "common", "ml",
             "ml_common", "pipelines")  # fmt: skip


def _add_files_to_protobuf() -> None:
    """Add the files of the protos to the default descriptor pool of protobuf, when protobuf is
    installed, as its generated modules of the protos add them: the modules that protoc generates
    of other protos that import them (those of a plugin, such as Delta Lake's) then build, and
    protobuf's ``json_format``, ``text_format`` and ``symbol_database`` find their types, of an
    ``Any`` too. Those modules get the classes of this runtime (see ``_hook_protobuf_builder``).

    protobuf's own modules of the well-known types add their files, of protobuf's copies of them.
    A file that the pool has is left as it is. The files are those that protoc writes, the bytes
    that Apache Spark's generated modules embed, so that one of those (of another package) adds
    its file again, as the pool takes a file twice. The drop-in itself needs none of this: a
    protobuf that does not import, or a pool that refuses a file, is passed over."""
    try:
        from google.protobuf import descriptor_pool

        pool = descriptor_pool.Default()
    except Exception:  # pragma: no cover  # noqa: BLE001 - no protobuf, or one that fails
        return
    for file_name, package, _, _ in _pyspark._proto_files():
        if package == "google.protobuf":
            stem = file_name.rsplit("/", 1)[-1][: -len(".proto")]
            try:
                importlib.import_module(f"google.protobuf.{stem}_pb2")
            except Exception:  # pragma: no cover  # noqa: BLE001
                pass
        try:
            pool.FindFileByName(file_name)
            continue
        except KeyError:
            pass
        try:
            pool.AddSerializedFile(_pyspark._proto_file_descriptor(file_name).serialized_pb)
        except Exception:  # pragma: no cover  # noqa: BLE001 - other descriptors of its names
            pass
    _register_classes_in_protobuf(pool)


def _register_classes_in_protobuf(pool: _Any) -> None:
    """Register the classes of the messages of the files of the descriptor set of the runtime
    (``_EMBEDDED_FILES``) in protobuf's default symbol database, by the descriptors of the default
    pool, as protobuf's builder registers the classes of its generated modules
    (``RegisterMessage``): ``symbol_database.Default().GetSymbol`` then gives them, which is how
    PySpark's pickle of a message of a nested type (``Expression.Literal``) loads, through
    protobuf's ``_InternalConstructMessage`` of the full name of its type. The database's own
    classes (of the well-known types) stay. So do those of the files that the hook added to the
    runtime for a plugin's module, when the package installs again: protobuf's lookups give
    protobuf's classes of them (see ``_hook_protobuf_builder``)."""
    try:
        from google.protobuf import symbol_database

        registered = symbol_database.Default()._classes
    except Exception:  # pragma: no cover  # noqa: BLE001 - a protobuf without the database
        return
    for full_name, cls in _CLASSES.items():
        try:
            descriptor = pool.FindMessageTypeByName(full_name)
        except KeyError:  # pragma: no cover - a file that the pool refused
            continue
        if descriptor.file.name in _EMBEDDED_FILES:
            registered.setdefault(descriptor, cls)


# The files of the descriptor set of the runtime, and those of them of the protos of Spark Connect.
_EMBEDDED_FILES = frozenset(name for name, _, _, _ in _pyspark._proto_files())
_SPARK_CONNECT_FILES = frozenset(
    name for name, package, _, _ in _pyspark._proto_files() if package == "spark.connect"
)

# The name of the generated module of each file of protobuf's default pool that protobuf's
# builder built since the hook was installed, or that ``_module_of_file`` found of one built
# before: the module of the classes of the runtime of the messages of a file that the runtime
# holds for a plugin's file that imports it.
_GENERATED_MODULES: dict[str, str] = {}

# The lock of the files that the hook adds to the runtime and of their classes: the generated
# modules of two threads may build at once.
_HOOK_LOCK = threading.RLock()


def _files_for_runtime(file_des: _Any) -> Optional[list[_Any]]:
    """The files of protobuf's pool that the runtime needs to hold the messages of the file
    ``file_des``: the file and the files that it imports, directly or through others, but those of
    the descriptor set of the runtime, each after those that it imports. None when it imports no
    proto of Spark Connect, so that it keeps protobuf's classes, as the modules of protobuf and of
    other libraries do, or when one of them is a file of protobuf's own package other than those
    of the set, the well-known types ``Any`` and ``Timestamp``, whose classes of the runtime have
    the methods of protobuf's: the runtime would have classes of the others (``Duration``,
    ``Struct``, ...) without them."""
    found: list[_Any] = []
    seen: set[str] = set()
    imports_spark_connect = False
    stack: list[tuple[_Any, bool]] = [(file_des, False)]
    while stack:
        file, imports_found = stack.pop()
        if imports_found:
            found.append(file)
            continue
        if file.name in seen:
            continue
        seen.add(file.name)
        if file.name in _EMBEDDED_FILES:
            imports_spark_connect = imports_spark_connect or file.name in _SPARK_CONNECT_FILES
            continue
        if file.package == "google.protobuf":
            return None
        stack.append((file, True))
        stack.extend((dependency, False) for dependency in reversed(file.dependencies))
    return found if imports_spark_connect else None


def _module_of_file(name: str) -> str:
    """The name of the generated module of the file ``name`` of protobuf's default pool, which the
    classes of the runtime of its messages take, as protobuf's classes of them do: that of the
    module that protobuf's builder built of it since the hook was installed, else that of a
    module of it that the builder built before (as Delta Lake's ``delta/connect/proto/__init__.py``
    imports ``base_pb2``, of ``delta/connect/base.proto``, before the modules that import the
    protos of Spark Connect, which install the hook), else protoc's name of its path."""
    module_name = _GENERATED_MODULES.get(name)
    if module_name is None:
        module_name = _imported_module_of_file(name) or _protoc_module_name(name)
        _GENERATED_MODULES[name] = module_name
    return module_name


def _imported_module_of_file(name: str) -> Optional[str]:
    """The module name of protobuf's classes of the messages of the file ``name`` of protobuf's
    default pool, of an imported module of the file (whose ``DESCRIPTOR`` is the file), which
    protobuf's builder gave them: None if no imported module has them. Only the globals of the
    modules are read, so no module's ``__getattr__`` runs."""
    from google.protobuf import descriptor_pool

    pool = descriptor_pool.Default()
    for module in list(sys.modules.values()):
        try:
            namespace = vars(module)
            file = namespace.get("DESCRIPTOR")
            if file is None or file.name != name or file.pool is not pool:
                continue
            for message in file.message_types_by_name:
                cls = namespace.get(message)
                if isinstance(cls, type):
                    return cls.__module__
        except Exception:  # noqa: BLE001 - an object of other globals, or of none
            continue
    return None


def _use_runtime_classes(file_des: _Any, module_name: str, module: dict[str, _Any]) -> None:
    """Give ``module``, the globals of the generated module ``module_name`` of the file
    ``file_des`` of protobuf's default pool, the classes of this runtime of the messages of the
    file, and its wrappers of the enums, when the file imports a proto of Spark Connect, directly
    or through the files that it imports, as a plugin's does (Delta Lake's
    ``delta/connect/relations.proto``): their message fields then take the messages of this
    package as they are, as protobuf's take PySpark's. The module of the same file built again
    gets the same classes, as protobuf's builder gives a file the classes that it made of it.

    The files are added to the pool of the runtime first, with those that the file imports that
    it does not have, whose messages the runtime makes classes of too: a file that imports no
    proto of Spark Connect (Delta Lake's ``delta/connect/base.proto``) keeps protobuf's classes in
    its own module, and a message field of the runtime of one of its types takes protobuf's
    messages of the type, of their full name and bytes, as it takes any. A file that the runtime
    does not hold as protobuf does (of proto2 or of editions, or that imports one) keeps
    protobuf's classes.

    The other names of the module stay protobuf's: its ``DESCRIPTOR``, the descriptors that the
    builder's ``BuildMessageAndEnumDescriptors`` binds (``_DELETEFROMTABLE``), whose serialized
    ranges and options the generated code sets with protobuf's pure-Python implementation, and
    its extensions and services."""
    _GENERATED_MODULES.setdefault(file_des.name, module_name)
    files = _files_for_runtime(file_des)
    if files is None:
        return
    with _HOOK_LOCK:
        if files:
            try:
                held = _pyspark._proto_register_files([file.serialized_pb for file in files])
            except ValueError:
                return
            made = set(_CLASSES)
            for name, _, messages, _ in held:
                of_module = module_name if name == file_des.name else _module_of_file(name)
                for full_name in messages:
                    if full_name not in _CLASSES:
                        _make_class(full_name, full_name.rpartition(".")[2], of_module)
            _pyspark._proto_register_classes(
                {name: cls for name, cls in _CLASSES.items() if name not in made}
            )
        for name, descriptor in file_des.message_types_by_name.items():
            module[name] = _CLASSES[descriptor.full_name]
        for name, enum in file_des.enum_types_by_name.items():
            module[name] = EnumTypeWrapper(_pyspark._proto_enum_descriptor(enum.full_name))


def _hook_protobuf_builder() -> None:
    """Wrap ``BuildTopDescriptorsAndMessages`` of protobuf's builder (``google.protobuf.internal
    .builder``), when protobuf is installed, with ``_use_runtime_classes``. A module that protoc
    generates calls it to make the classes of the messages of its file, of protobuf's default pool,
    after it imports the modules of the files that its file imports, which import this package and
    so install the hook first, for a file that imports the protos of Spark Connect. The builder
    makes protobuf's classes of the file first, as it does without the hook, so protobuf's own
    lookups of the file (``message_factory.GetMessageClass``, ``symbol_database``) still give them.
    A file of another pool, such as one that a test builds, keeps protobuf's classes. One hook
    wraps the builder's own function, of the runtime of the last install. A protobuf of no builder
    (before 3.20), or whose builder has no such function, is passed over, as the drop-in itself
    needs none of this. A module of protoc before 3.20 makes its classes with protobuf's
    ``reflection.GeneratedProtocolMessageType``, not with the builder, so it keeps protobuf's
    classes."""
    try:
        from google.protobuf import descriptor_pool
        from google.protobuf.internal import builder

        default_pool = descriptor_pool.Default()
        build = builder.BuildTopDescriptorsAndMessages
    except Exception:  # pragma: no cover  # noqa: BLE001 - no protobuf, or one that fails or lacks it
        return
    build = getattr(build, "_protobuf_build", build)

    @functools.wraps(build)
    def BuildTopDescriptorsAndMessages(
        file_des: _Any, module_name: str, module: dict[str, _Any]
    ) -> None:
        build(file_des, module_name, module)
        if file_des.pool is default_pool:
            _use_runtime_classes(file_des, module_name, module)

    BuildTopDescriptorsAndMessages._protobuf_build = build  # type: ignore[attr-defined]
    builder.BuildTopDescriptorsAndMessages = BuildTopDescriptorsAndMessages


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


def install(package: dict[str, _Any]) -> None:
    """Export the messages of Spark Connect into ``package``, the globals of
    ``pyspark.sql.connect.proto``, as its ``from ..._pb2 import *`` of the generated modules does
    (so its ``DESCRIPTOR`` is that of the last, ``pipelines_pb2``), make each generated module
    (``pyspark.sql.connect.proto.base_pb2``, ...) the one of this runtime, and, when protobuf is
    installed, add the files of the protos to its pool, give the modules that protoc generates of
    other protos that import them the classes of the runtime, and make its ``UnknownFieldSet``
    take the messages of the runtime."""
    prefix = "pyspark.sql.connect.proto."
    for module_name, module in _MODULES.items():
        if _is_exported(module_name):
            sys.modules[module_name] = module
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
    _add_files_to_protobuf()
    _hook_protobuf_builder()
    _hook_unknown_field_set()


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
