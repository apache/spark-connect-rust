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
descriptor pool, as the generated modules add them (see ``_add_files_to_protobuf``).
"""
import calendar
import collections.abc
import datetime
import importlib
import importlib.abc
import importlib.machinery
import sys
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


def _module_alias(file_name: str) -> str:
    """The name that the generated module of a proto that imports the file ``file_name`` binds
    the module of the file to, as protoc names it: ``spark_dot_connect_dot_relations__pb2``."""
    module = file_name[: -len(".proto")].replace("/", ".") + "_pb2"
    return module.replace("_", "__").replace(".", "_dot_")


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
    ``Any`` too. The messages of those modules are protobuf's, whose message fields take a message
    of this runtime through its bytes or a dict.

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


def install(package: dict[str, _Any]) -> None:
    """Export the messages of Spark Connect into ``package``, the globals of
    ``pyspark.sql.connect.proto``, as its ``from ..._pb2 import *`` of the generated modules does
    (so its ``DESCRIPTOR`` is that of the last, ``pipelines_pb2``), make each generated module
    (``pyspark.sql.connect.proto.base_pb2``, ...) the one of this runtime, and add the files of
    the protos to protobuf's pool when protobuf is installed."""
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
