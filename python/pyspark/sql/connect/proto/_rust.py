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

``pyspark.sql.connect.proto`` exports these when the ``protobuf`` package is not installed (or
is too old for the generated ``*_pb2`` modules, or ``PYSPARK_CONNECT_PROTO_IMPLEMENTATION=rust``
asks for them): a class for each message of the protos of Spark Connect, with the API of
protobuf's Python messages, whose fields, descriptors and wire format the Rust runtime holds and
encodes. Each class has the ``DESCRIPTOR`` of its message and, as an attribute, the classes of
its nested messages and its enums, with their values, as the generated modules give them.
"""
import collections.abc
import sys
import types
from collections.abc import Generator, Iterator
from contextlib import contextmanager
from typing import Any as _Any
from typing import Optional

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
    and each value as an attribute."""

    def __init__(self, descriptor: _Any) -> None:
        self.DESCRIPTOR = descriptor
        self._values: list[tuple[str, int]] = [(v.name, v.number) for v in descriptor.values]
        self._by_name = dict(self._values)
        self._by_number: dict[int, str] = {}
        for name, number in self._values:
            self._by_number.setdefault(number, name)

    def Name(self, number: int) -> str:
        try:
            return self._by_number[number]
        except KeyError:
            raise ValueError(
                f"Enum {self.DESCRIPTOR.name} has no name defined for value {number!r}"
            ) from None

    def Value(self, name: str) -> int:
        try:
            return self._by_name[name]
        except KeyError:
            raise ValueError(
                f"Enum {self.DESCRIPTOR.name} has no value defined for name {name!r}"
            ) from None

    def keys(self) -> list[str]:
        return [name for name, _ in self._values]

    def values(self) -> list[int]:
        return [number for _, number in self._values]

    def items(self) -> list[tuple[str, int]]:
        return list(self._values)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def __getattr__(self, name: str) -> int:
        try:
            return self.__dict__["_by_name"][name]
        except KeyError:
            raise AttributeError(
                f"Enum {self.DESCRIPTOR.name} has no value named {name!r}"
            ) from None

    def __repr__(self) -> str:
        return f"<EnumTypeWrapper {self.DESCRIPTOR.full_name}>"


# The class of each message, by the full name of the message.
_CLASSES: dict[str, type] = {}


def _make_class(full_name: str, qualname: str, module_name: str) -> type:
    descriptor = _pyspark._proto_message_descriptor(full_name)
    cls = type(
        descriptor.name,
        (Message,),
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


def _build() -> dict[str, types.ModuleType]:
    modules = {}
    for file_name, package, messages, enums in _pyspark._proto_files():
        module_name = _module_name(file_name, package)
        module = types.ModuleType(module_name)
        module.__doc__ = f"The messages of {file_name}, of the drop-in's Rust protobuf runtime."
        for full_name in messages:
            name = full_name.rsplit(".", 1)[-1]
            setattr(module, name, _make_class(full_name, name, module_name))
        for full_name in enums:
            descriptor = _pyspark._proto_enum_descriptor(full_name)
            setattr(module, descriptor.name, EnumTypeWrapper(descriptor))
            for value in descriptor.values:
                setattr(module, value.name, value.number)
        modules[module_name] = module
    _pyspark._proto_register_classes(_CLASSES)
    return modules


_MODULES = _build()

# The well-known types that the protos of Spark Connect import.
Any = _CLASSES["google.protobuf.Any"]
Timestamp = _CLASSES["google.protobuf.Timestamp"]


def install(package: dict[str, _Any]) -> None:
    """Export the messages of Spark Connect into ``package``, the globals of
    ``pyspark.sql.connect.proto``, as its ``from ..._pb2 import *`` does, and make each generated
    module (``pyspark.sql.connect.proto.base_pb2``, ...) the one of this runtime."""
    for module_name, module in _MODULES.items():
        if not module_name.startswith("pyspark.sql.connect.proto.") or module_name.startswith(
            __name__ + "."
        ):
            continue
        sys.modules[module_name] = module
        package[module_name.rsplit(".", 1)[-1]] = module
        for name, value in vars(module).items():
            if not name.startswith("_"):
                package[name] = value


@contextmanager
def add_pipeline_analysis_context(
    spark: _Any, dataflow_graph_id: str, flow_name: Optional[str]
) -> Generator[None, None, None]:
    """``pyspark.pipelines.add_pipeline_analysis_context`` of this runtime's ``Any``: the context
    manager that adds the ``PipelineAnalysisContext`` extension to the user context of the
    requests, for the analysis of a pipeline, as the vendored one does with protobuf's ``Any``
    (see ``pyspark._proto_fallback``)."""
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
