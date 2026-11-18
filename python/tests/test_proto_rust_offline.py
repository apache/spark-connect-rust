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
"""The Rust protobuf runtime of the drop-in (``pyspark.sql.connect.proto._rust``), the messages of
``pyspark.sql.connect.proto``, against upb, the implementation of a ``protobuf`` install: the
same operations on a message of each agree in their bytes, text, fields, containers, presence,
errors, descriptors, options, well-known types and the modules of the protos, and protobuf's own
modules (``json_format``, ``text_format``, the default descriptor pool and the modules that protoc
generates of protos that import Spark Connect's) take the messages. Two tests run the runtime
where ``protobuf`` cannot be imported, as the drop-in runs without it.

The comparisons need protobuf of upb installed, and are skipped otherwise. upb's messages are of
the files of the protos that ``pyspark.sql.connect.proto`` adds to protobuf's default pool, which
are the bytes of the files that Apache Spark's generated modules add. The runtime follows the
errors of protobuf 7, which a few of protobuf 6 differ from.
"""

import collections.abc as abc
import copy
import datetime
import importlib
import json
import os
import pathlib
import pickle
import random
import subprocess
import sys
import textwrap
import threading
import time
import warnings

import pytest
from pyspark import _pyspark
from pyspark.sql.connect import proto
from pyspark.sql.connect.proto import _rust

# The classes of upb, by full name, that _upb_classes makes.
_UPB_CLASSES = {}


def _upb_classes_of(files):
    """The classes of upb of the messages of ``files``, each its serialized file and the name of
    its module, by full name, as the modules that protoc generates of the files make them with
    protobuf's builder (of their module and qualified names, as Apache Spark's do). The files are
    those of the protos and protobuf's own of the well-known types that they import, then
    ``files``, in a pool of their own, so that no class that protobuf made before of the default
    pool, of another name, stands in for one: the builder gives a file of another pool than the
    default one protobuf's classes, which the drop-in leaves as they are."""
    from google.protobuf import any_pb2, descriptor_pool, timestamp_pb2
    from google.protobuf.internal import builder

    pool = descriptor_pool.DescriptorPool()
    spark_files = [(m.DESCRIPTOR.serialized_pb, m.__name__) for m in (any_pb2, timestamp_pb2)]
    spark_files += [
        (_pyspark._proto_file_descriptor(name).serialized_pb, _rust._module_name(name, package))
        for name, package, _, _ in _pyspark._proto_files()
        if package == "spark.connect"
    ]
    classes = {}

    def add(cls):
        classes[cls.DESCRIPTOR.full_name] = cls
        for nested in cls.DESCRIPTOR.nested_types:
            add(getattr(cls, nested.name))

    for serialized, module_name in spark_files + list(files):
        file = pool.AddSerializedFile(serialized)
        namespace = {}
        builder.BuildMessageAndEnumDescriptors(file, namespace)
        builder.BuildTopDescriptorsAndMessages(file, module_name, namespace)
        for name in file.message_types_by_name:
            add(namespace[name])
    return classes


def _upb_classes():
    """The class of a message of upb by full name, of the protos (see ``_upb_classes_of``), or
    None without protobuf of upb."""
    try:
        from google.protobuf.internal import api_implementation
    except ImportError:
        return None
    if api_implementation.Type() != "upb":
        return None
    if not _UPB_CLASSES:
        _UPB_CLASSES.update(_upb_classes_of([]))
    return _UPB_CLASSES.__getitem__


@pytest.fixture(scope="module")
def upb():
    """The class of a message of upb, by full name."""
    classes = _upb_classes()
    if classes is None:
        pytest.skip("the runtime follows upb, the implementation of a protobuf install")
    return classes


def _protobuf_7():
    """Whether protobuf is of 7 or later, whose errors the runtime gives where those of protobuf 6
    differ."""
    import google.protobuf

    return int(google.protobuf.__version__.split(".")[0]) >= 7


def ours(full_name):
    return _rust._CLASSES[full_name]


def _outcome(fn):
    """The result of `fn`, or the type and message of what it raises."""
    try:
        return ("ok", _comparable(fn()))
    except Exception as e:  # noqa: BLE001
        return ("raise", type(e).__name__, str(e))


def _comparable(value):
    if hasattr(value, "SerializeToString") and hasattr(value, "DESCRIPTOR"):
        # A message, by its deterministic bytes: those of either runtime parse alike.
        return ("message", value.DESCRIPTOR.full_name, value.SerializeToString(deterministic=True))
    if isinstance(value, (list, tuple)):
        return [_comparable(v) for v in value]
    if isinstance(value, dict):
        return sorted((repr(k), repr(_comparable(v))) for k, v in value.items())
    if isinstance(value, float) and value != value:
        return "nan"
    if isinstance(value, (bool, int, float, str, bytes, type(None))):
        return value
    name = type(value).__name__
    if name in ("RepeatedScalarContainer", "RepeatedCompositeContainer"):
        return ("repeated", [_comparable(v) for v in value])
    if name in ("ScalarMapContainer", "MessageMapContainer", "MapContainer"):
        return ("map", sorted((repr(k), repr(_comparable(value[k]))) for k in list(value)))
    if isinstance(value, (abc.KeysView, abc.ValuesView, abc.ItemsView)):
        return (type(value).__name__, sorted(repr(_comparable(v)) for v in value))
    return ("object", name)


def _all_types(upb):
    """The full names of every message type of the protos, but the map entries."""
    return [n for n in sorted(_rust._CLASSES) if not upb(n).DESCRIPTOR.GetOptions().map_entry]


class _Generator:
    """Random messages of upb, of every kind of field, as Spark Connect's are."""

    def __init__(self, upb, seed):
        self.upb = upb
        self.rng = random.Random(seed)

    def scalar(self, f):
        rng = self.rng
        t = f.type
        if t in (f.TYPE_INT32, f.TYPE_SINT32, f.TYPE_SFIXED32):
            return rng.choice([0, 1, -1, 2**31 - 1, -(2**31), rng.randint(-1000, 1000)])
        if t in (f.TYPE_INT64, f.TYPE_SINT64, f.TYPE_SFIXED64):
            return rng.choice([0, 1, -1, 2**63 - 1, -(2**63), rng.randint(-(10**12), 10**12)])
        if t in (f.TYPE_UINT32, f.TYPE_FIXED32):
            return rng.choice([0, 1, 2**32 - 1, rng.randint(0, 10**6)])
        if t in (f.TYPE_UINT64, f.TYPE_FIXED64):
            return rng.choice([0, 1, 2**64 - 1, rng.randint(0, 10**15)])
        if t == f.TYPE_DOUBLE:
            return rng.choice([0.0, -0.0, 1.5, -1.0, 1e300, float("inf"), 0.1, 1e-310, 1 / 3])
        if t == f.TYPE_FLOAT:
            return rng.choice([0.0, -0.0, 1.5, 0.1, 3.4e38, float("inf"), 1e-40, 1 / 3])
        if t == f.TYPE_BOOL:
            return rng.choice([True, False])
        if t == f.TYPE_STRING:
            return rng.choice(["", "a", "bytes", "héllo", 'q"\\\n', "\x00\x7f", "日本語", "x" * 300])
        if t == f.TYPE_BYTES:
            return rng.choice([b"", b"\x00\xff\x80", b"abc", bytes(range(256))])
        if t == f.TYPE_ENUM:
            values = [v.number for v in f.enum_type.values]
            return rng.choice(values + [values[-1] + 7])  # an unknown value of an open enum
        raise AssertionError(t)

    def fill(self, message, depth):
        rng = self.rng
        d = message.DESCRIPTOR
        if d.full_name == "google.protobuf.Any":
            choice = rng.randint(0, 2)
            if choice == 0:
                inner = self.upb("spark.connect.Relation")()
                if depth > 0:
                    self.fill(inner, depth - 1)
                message.Pack(inner)
            elif choice == 1:
                message.type_url = "type.googleapis.com/x.y.Unknown"
                message.value = b"\x08\x01"
            return
        fields = list(d.fields)
        rng.shuffle(fields)
        for f in fields[: rng.randint(0, min(len(fields), 6))]:
            if f.type == f.TYPE_MESSAGE and f.message_type.GetOptions().map_entry:
                key_f = f.message_type.fields_by_name["key"]
                value_f = f.message_type.fields_by_name["value"]
                container = getattr(message, f.name)
                for _ in range(rng.randint(0, 3)):
                    key = self.scalar(key_f)
                    if value_f.type == value_f.TYPE_MESSAGE:
                        child = container[key]
                        if depth > 0:
                            self.fill(child, depth - 1)
                    else:
                        container[key] = self.scalar(value_f)
            elif f.is_repeated:
                container = getattr(message, f.name)
                for _ in range(rng.randint(0, 3)):
                    if f.type == f.TYPE_MESSAGE:
                        child = container.add()
                        if depth > 0:
                            self.fill(child, depth - 1)
                    else:
                        container.append(self.scalar(f))
            elif f.type == f.TYPE_MESSAGE:
                child = getattr(message, f.name)
                if depth > 0:
                    self.fill(child, depth - 1)
                child.SetInParent()
            else:
                setattr(message, f.name, self.scalar(f))


@pytest.mark.parametrize("seed", range(8))
def test_random_messages_of_every_type_encode_parse_and_print_as_upb(upb, seed):
    from google.protobuf import text_format

    generator = _Generator(upb, seed)
    problems = []
    for full_name in _all_types(upb):
        real = upb(full_name)()
        generator.fill(real, depth=generator.rng.randint(0, 3))
        data = real.SerializeToString(deterministic=True)
        message = ours(full_name).FromString(data)
        found = []
        if message.SerializeToString() != data:
            found.append("bytes")
        if upb(full_name).FromString(message.SerializeToString()).SerializeToString(
            deterministic=True
        ) != data:
            found.append("bytes as upb parses them")
        try:
            real_text = str(real)
        except KeyError:  # upb's text of an Any of an unknown type
            real_text = None
        if real_text is not None and str(message) != real_text:
            found.append("str()")
        try:
            real_one_line = text_format.MessageToString(real, as_one_line=True)
        except KeyError:
            real_one_line = None
        if real_one_line is not None and (
            _pyspark._proto_message_to_string(message, as_one_line=True) != real_one_line
        ):
            found.append("text_format")
        if [f.name for f, _ in message.ListFields()] != [f.name for f, _ in real.ListFields()]:
            found.append("ListFields")
        if message.ByteSize() != real.ByteSize():
            found.append("ByteSize")
        if found:
            problems.append(f"{full_name}: {', '.join(found)}")
    assert not problems, problems[:10]


def _varint(n, length=None):
    """The varint of `n`, in `length` bytes (an overlong one) if given."""
    out = bytearray()
    while True:
        out.append(n & 0x7F)
        n >>= 7
        if not n and (length is None or len(out) >= length):
            return bytes(out)
        out[-1] |= 0x80


def _tag(number, wire_type, length=None):
    return _varint((number << 3) | wire_type, length)


def _field(number, payload):
    """The field `number` of the length-delimited `payload`."""
    return _tag(number, 2) + _varint(len(payload)) + payload


def _parsed(cls, data):
    """What a message of `cls` that parses `data` is: its bytes, text and size, or the error."""

    def parse():
        message = cls.FromString(data)
        return message.SerializeToString(deterministic=True), str(message), message.ByteSize()

    return _outcome(parse)


def test_tags_of_more_than_32_bits_or_five_bytes_are_corrupt_as_upbs(upb):
    # upb reads a tag as a varint of at most five bytes that holds 32 bits, so a field number of
    # 2^29 or more is corrupt, rather than cut to its bits (2^32 + 1 as field 1) or kept as an
    # unknown field: of a field, of a group and its fields, and of the fields of a map entry. A
    # field number of 0 is corrupt, but among the fields of a group. A failure in a message that
    # the message holds is the error of the message parsed, which names its type.
    group, end = _tag(5, 3), _tag(5, 4)
    fields = [
        _tag(2**32 + 1, 0) + _varint(5),
        _tag(2**32, 0) + _varint(5),
        _tag(2**31, 0) + _varint(5),
        _tag(2**29, 0) + _varint(5),
        _tag(2**29 - 1, 0) + _varint(5),
        _tag(2**61 + 1, 0) + _varint(5),
        _tag(1, 0, length=5) + _varint(5),
        _tag(1, 0, length=6) + _varint(5),
        _tag(2**29 - 1, 0, length=6) + _varint(5),
        _tag(2**29 - 1, 7),
        _tag(0, 0) + _varint(5),
        end,
        _tag(2**29 + 3, 3) + _tag(2**29 + 3, 4),
        _tag(2**29 - 1, 3) + _tag(2**29 - 1, 4),
        group + _tag(2**29, 0) + _varint(1) + end,
        group + _tag(2**29 - 1, 0) + _varint(1) + end,
        group + _tag(1, 0, length=6) + _varint(1) + end,
        group + _tag(0, 0) + _varint(1) + end,
        group + _tag(1, 7) + end,
        group + _varint(((2**32 + 5) << 3) | 4),
        group + _tag(5, 4, length=5),
        group + _tag(5, 4, length=6),
        group + _tag(6, 4),
    ]
    for data in fields:
        expected = _parsed(upb("spark.connect.Ints"), data)
        assert _parsed(ours("spark.connect.Ints"), data) == expected, data.hex()
    key, value = _field(1, b"k"), _field(2, b"v")
    entries = [
        _tag(2**32 + 1, 2) + _varint(1) + b"k" + value,
        _tag(2**29, 2) + _varint(1) + b"k" + value,
        key + _tag(2**32 + 2, 2) + _varint(1) + b"v",
        _tag(1, 2, length=5) + _varint(1) + b"k" + value,
        _tag(1, 2, length=6) + _varint(1) + b"k" + value,
        key + value + _tag(0, 0) + _varint(1),
        key + value + _tag(3, 3) + _tag(0, 0) + _varint(1) + _tag(3, 4),
    ]
    for entry in entries:
        data = _field(3, entry)  # an entry of Read.DataSource.options
        expected = _parsed(upb("spark.connect.Read.DataSource"), data)
        assert _parsed(ours("spark.connect.Read.DataSource"), data) == expected, entry.hex()
    corrupt = _tag(2**29, 0) + _varint(1)
    for full_name, data in [
        ("spark.connect.Plan", _field(1, corrupt)),  # of its root
        ("spark.connect.Plan", _field(1, _field(2, _field(2, _field(2, b"\xff"))))),  # bad UTF-8
        ("spark.connect.Plan", _field(1, b"\x0a\x05")),  # truncated
        ("spark.connect.SQL", _field(4, _field(1, b"a") + _field(2, corrupt))),  # of a map value
        ("spark.connect.Expression.UnresolvedFunction", _field(2, corrupt)),  # of a repeated one
    ]:
        assert _parsed(ours(full_name), data) == _parsed(upb(full_name), data), data.hex()


def test_map_entries_of_fields_of_no_field_of_their_type_are_unknown_fields_as_upbs(upb):
    # upb decodes a map entry as a message of the entry type and sets its key and value in the
    # map, but not when the entry has a field that the type has no field for, or a key or a value
    # of another wire type (which upb reads as such a field): the message keeps the entry in its
    # unknown fields, among the others, encoded again as upb encodes the entry message (its key
    # and value, a scalar of the default not, then those fields), and DiscardUnknownFields drops
    # it. A key, a value or an entry that is not there is the default, and the last of a key or a
    # value that is there twice wins, as of the fields of a message.
    def parsed(cls, full_name, field, data, before=None):
        def parse():
            message = cls(full_name)()
            if before:
                before(getattr(message, field))
            message.MergeFromString(data)
            found = [_comparable(getattr(message, field)), message.ByteSize(), str(message)]
            found.append(message.SerializeToString(deterministic=True))
            message.DiscardUnknownFields()
            return found + [message.SerializeToString(deterministic=True)]

        return _outcome(parse)

    key, value, unknown = _field(1, b"k"), _field(2, b"v"), _tag(3, 0) + _varint(7)
    string_entries = [
        key + value,
        key + value + unknown,
        unknown + key + value,
        key + unknown + value,
        unknown,
        b"",
        key,
        value,
        value + key,
        key + value + _tag(4, 1) + bytes(8),
        key + value + _tag(4, 5) + bytes(4),
        key + value + _field(5, b"\xff"),  # not read as UTF-8
        key + value + _tag(6, 3) + _tag(1, 0) + _varint(1) + _tag(6, 4),
        key + value + _tag(2**29 - 1, 0) + _varint(1),
        _tag(1, 0) + _varint(5) + value,  # a key of another wire type
        _tag(1, 5) + b"abcd" + value,
        _tag(1, 3) + _tag(1, 4) + value,
        key + _tag(1, 0) + _varint(5) + value,
        key + _tag(2, 0) + _varint(5),  # a value of another wire type
        key + _tag(2, 3) + _tag(2, 4),
        _field(1, b"") + value + unknown,
        key + _field(2, b"") + unknown,
        _field(1, b"") + _field(2, b"") + unknown,
        _field(1, b"a") + _field(1, b"b") + value + unknown,
        key + _field(2, b"x") + _field(2, b"y") + unknown,
        _tag(1, 2, length=2) + _varint(1) + b"k" + value + unknown,
        key + value + _tag(3, 4),
        key + value + _tag(5, 2) + _varint(5) + b"ab",
        _field(1, b"\xff") + value + unknown,
    ]
    data_source = "spark.connect.Read.DataSource"
    for entry in string_entries:
        data = _field(3, entry)
        expected = parsed(upb, data_source, "options", data)
        assert parsed(ours, data_source, "options", data) == expected, entry.hex()
    # Among the other unknown fields of the message, and into a map that has the key.
    data = _tag(9, 0) + _varint(1) + _field(3, key + value + unknown) + _tag(10, 0) + _varint(2)
    data += _field(3, _field(1, b"a") + value) + _field(3, _field(1, b"b") + value + unknown)
    assert parsed(ours, data_source, "options", data) == parsed(upb, data_source, "options", data)

    def set_k(options):
        options["k"] = "old"

    data = _field(3, key + value + unknown)
    expected = parsed(upb, data_source, "options", data, before=set_k)
    assert parsed(ours, data_source, "options", data, before=set_k) == expected

    name, unknown = _field(1, b"a"), _tag(5, 0) + _varint(1)
    message_entries = [
        name + _field(2, b""),
        name + _field(2, b"") + unknown,
        name + unknown,  # of no value, which is then not written
        unknown,
        name,
        name + _tag(2, 0) + _varint(1),
        name + _tag(2, 0) + _varint(1) + _field(2, b""),
        name + _field(2, _tag(999, 0) + _varint(1)),  # of an unknown field of the value
        name + _field(2, _tag(999, 0) + _varint(1)) + unknown,
        name + _field(2, _field(1, b"\x10\x03")) + _field(2, _field(1, b"\x18\x01")) + unknown,
        name + _field(2, _field(1, _tag(3, 0) + _varint(7))) + unknown,
        _tag(1, 0) + _varint(5) + _field(2, b""),
        name + _field(2, b"\xff") + unknown,
    ]
    for entry in message_entries:
        data = _field(4, entry)  # an entry of SQL.named_arguments, of Expression values
        expected = parsed(upb, "spark.connect.SQL", "named_arguments", data)
        assert parsed(ours, "spark.connect.SQL", "named_arguments", data) == expected, entry.hex()


def test_messages_that_fail_to_parse_keep_what_they_read_as_upbs(upb):
    # As upb's: a parse that fails keeps what it read before the failure. A message of a message
    # field is set, or appended, before its bytes are read, so a failure in them, or a length past
    # the end of the data, leaves it set of the fields read before; a map entry is set once it is
    # read. A packed field of a fixed width whose bytes are not a whole number of values keeps
    # none of them. MergeFromString keeps the fields that the message had, and ParseFromString
    # clears them first.
    def parsed(cls, full_name, before, data, method):
        def parse():
            message = cls(full_name)()
            message.MergeFromString(before)
            try:
                getattr(message, method)(data)
            except Exception as e:  # noqa: BLE001
                failure = (type(e).__name__, str(e))
            else:
                failure = None
            fields = [field.name for field, _ in message.ListFields()]
            return failure, fields, message.SerializeToString(deterministic=True), str(message)

        return _outcome(parse)

    expression = upb("spark.connect.Expression")(literal={"integer": 1}).SerializeToString()
    common = upb("spark.connect.Relation")(common={"plan_id": 7}).SerializeToString()
    sql = upb("spark.connect.Relation")(sql={"query": "q"}).SerializeToString()
    cut = _tag(6, 0) + _varint(5) + b"\xff"  # Literal.integer, then a tag cut short
    function = "spark.connect.Expression.UnresolvedFunction"
    cases = [
        ("spark.connect.Relation", common, sql + b"\xff"),
        ("spark.connect.Relation", common, sql[:-1]),
        ("spark.connect.Expression", b"", _field(1, cut)),  # of a set message
        (function, b"", _field(2, expression) + _field(2, _field(1, cut))),  # of an element
        ("spark.connect.SQL", b"", _field(4, _field(1, b"a") + _field(2, _field(1, cut)))),
        ("spark.connect.Expression", b"", _tag(1, 2) + _varint(9) + _tag(6, 0)),
        (function, b"", _field(2, expression) + _tag(2, 2) + _varint(9) + b"\x0a"),
        ("spark.connect.SQL", b"", _tag(4, 2) + _varint(9) + b"\x0a"),
        ("spark.connect.Strings", b"", _field(1, b"a") + _tag(1, 2) + _varint(5) + b"ab"),
        ("spark.connect.Ints", _field(1, b"\x07"), _tag(1, 2) + _varint(9) + b"\x01"),
        ("spark.connect.Doubles", _field(1, bytes(8)), _field(1, bytes(12))),
    ]
    for full_name, before, data in cases:
        for method in ("MergeFromString", "ParseFromString"):
            expected = parsed(upb, full_name, before, data, method)
            assert expected[1][0] is not None, (full_name, data.hex())
            assert parsed(ours, full_name, before, data, method) == expected, (
                full_name,
                data.hex(),
                method,
            )


_UNKNOWN = _tag(999, 0) + _varint(1)


def _messages_of_levels(cls, levels, shape):
    """A message of `levels` levels and its messages of every level, by depth: Relations of
    limits ("message"), whose last has a SQL of a map value too ("map"), or Expressions of a
    function of one argument, of a repeated field ("repeated")."""
    if shape == "repeated":
        root = cls("spark.connect.Expression")()
        messages, current = [root], root
        for _ in range(levels):
            function = current.unresolved_function
            current = function.arguments.add()
            messages += [function, current]
        return root, messages
    root = cls("spark.connect.Relation")()
    messages, current = [root], root
    for _ in range(levels):
        limit = current.limit
        limit.limit = 1
        current = limit.input
        messages += [limit, current]
    if shape == "map":
        sql = current.sql
        messages += [sql, sql.named_arguments["a"]]
    return root, messages


def test_discard_unknown_fields_discards_those_of_63_levels_as_upbs(upb):
    # upb's DiscardUnknownFields discards the unknown fields of the message and of its messages
    # down to 62 levels below it (upb_Message_DiscardUnknown of a maximum depth of 64), of message
    # fields, repeated fields and map values, and leaves those of the deeper messages.
    def discarded(cls, levels, shape, start):
        root, messages = _messages_of_levels(cls, levels, shape)
        for message in messages:
            message.MergeFromString(_UNKNOWN)
        messages[start].DiscardUnknownFields()
        return root.SerializeToString(deterministic=True)

    for case in [
        (10, "message", 0),
        (40, "message", 0),
        (100, "message", 0),
        (60, "message", 41),
        (40, "repeated", 0),
        (30, "map", 0),
        (31, "map", 0),
        (32, "map", 0),
    ]:
        assert discarded(ours, *case) == discarded(upb, *case), case


_LOCKSTEP_TYPES = [
    "spark.connect.Relation",
    "spark.connect.Expression",
    "spark.connect.Command",
    "spark.connect.MlCommand",
    "spark.connect.MlParams",
    "spark.connect.Expression.Literal",
    "spark.connect.Plan",
    "spark.connect.ExecutePlanRequest",
    "spark.connect.WriteOperation",
    "spark.connect.Read",
    "spark.connect.Aggregate",
    "spark.connect.MlRelation",
    "spark.connect.PipelineCommand",
    "spark.connect.ConfigRequest",
    "spark.connect.Expression.UnresolvedFunction",
    "spark.connect.Expression.Literal.Array",
    "spark.connect.Join",
]


class _Lockstep:
    """The same random operations on a message of upb and on one of the runtime: setting fields
    (with values of the wrong types too), unset children, containers, maps, copies, merges and
    clears."""

    def __init__(self, upb, seed):
        self.upb = upb
        self.rng = random.Random(seed)

    def value(self, f):
        rng = self.rng
        t = f.type
        if t in (f.TYPE_INT32, f.TYPE_SINT32, f.TYPE_SFIXED32, f.TYPE_INT64, f.TYPE_SINT64):
            return rng.choice([0, 1, -5, 2**31 - 1, 2**40, True, "1"])
        if t in (f.TYPE_UINT32, f.TYPE_FIXED32, f.TYPE_UINT64, f.TYPE_FIXED64, f.TYPE_SFIXED64):
            return rng.choice([0, 3, 2**32 - 1, -1])
        if t in (f.TYPE_DOUBLE, f.TYPE_FLOAT):
            return rng.choice([0.0, 2.5, -1.0, 7, 1.0, "x"])
        if t == f.TYPE_BOOL:
            return rng.choice([True, False, 0, 1, 2.0])
        if t == f.TYPE_STRING:
            return rng.choice(["", "s", "日本", b"bytes", 5])
        if t == f.TYPE_BYTES:
            return rng.choice([b"", b"\x01\x02", "str"])
        values = [v.number for v in f.enum_type.values]
        return rng.choice(values + [99, f.enum_type.values[-1].name, "NOPE", True])

    def both(self, label, a, b, fn):
        r1, r2 = _outcome(lambda: fn(a)), _outcome(lambda: fn(b))
        assert r1[:2] == r2[:2], f"{label}: upb={r1} ours={r2}"
        return r1[0] == "ok"

    def step(self, a, b, log, depth=0):
        rng = self.rng
        fields = list(a.DESCRIPTOR.fields)
        if not fields:
            return
        f = rng.choice(fields)
        kind = rng.random()
        name = f.name
        if f.type == f.TYPE_MESSAGE and f.message_type.GetOptions().map_entry:
            key = self.value(f.message_type.fields_by_name["key"])
            value_f = f.message_type.fields_by_name["value"]
            if value_f.type == value_f.TYPE_MESSAGE:
                if kind < 0.6:
                    log.append(f"{name}[{key!r}] then a step in it")
                    if self.both(log[-1], a, b, lambda m: getattr(m, name)[key]) and depth < 2:
                        self.step(getattr(a, name)[key], getattr(b, name)[key], log, depth + 1)
                else:
                    log.append(f"del {name}[{key!r}]")
                    self.both(log[-1], a, b, lambda m: getattr(m, name).__delitem__(key))
            else:
                value = self.value(value_f)
                log.append(f"{name}[{key!r}] = {value!r}")
                self.both(log[-1], a, b, lambda m: getattr(m, name).__setitem__(key, value))
            return
        if f.is_repeated:
            if f.type == f.TYPE_MESSAGE:
                if kind < 0.5 or len(getattr(a, name)) == 0:
                    log.append(f"{name}.add() then a step in it")
                    child_a, child_b = getattr(a, name).add(), getattr(b, name).add()
                    if depth < 2:
                        self.step(child_a, child_b, log, depth + 1)
                elif kind < 0.7:
                    i = rng.randrange(len(getattr(a, name)))
                    log.append(f"del {name}[{i}]")
                    self.both(log[-1], a, b, lambda m: getattr(m, name).__delitem__(i))
                else:
                    log.append(f"{name}.extend(copies)")
                    self.both(log[-1], a, b, lambda m: getattr(m, name).extend(list(getattr(m, name))))
            else:
                value = self.value(f)
                ops = [
                    (f"{name}.append({value!r})", lambda m: getattr(m, name).append(value)),
                    (f"{name}.extend([{value!r}])", lambda m: getattr(m, name).extend([value])),
                    (f"del {name}[:1]", lambda m: getattr(m, name).__delitem__(slice(0, 1))),
                    (f"{name}.sort()", lambda m: getattr(m, name).sort()),
                ]
                label, op = rng.choice(ops)
                log.append(label)
                self.both(label, a, b, op)
            return
        if f.type == f.TYPE_MESSAGE:
            if kind < 0.45:
                log.append(f"step in {name}")
                if depth < 3:
                    self.step(getattr(a, name), getattr(b, name), log, depth + 1)
            elif kind < 0.55:
                log.append(f"{name}.SetInParent()")
                self.both(log[-1], a, b, lambda m: getattr(m, name).SetInParent())
            elif kind < 0.75:
                log.append(f"{name}.CopyFrom(new)")
                new_a = self.upb(f.message_type.full_name)()
                new_b = ours(f.message_type.full_name)()
                if depth < 2:
                    self.step(new_a, new_b, log, depth + 1)
                getattr(a, name).CopyFrom(new_a)
                getattr(b, name).CopyFrom(new_b)
            elif kind < 0.9:
                log.append(f"{name}.MergeFrom(a copy of it)")
                for m in (a, b):
                    child = getattr(m, name)
                    copied = type(child)()
                    copied.CopyFrom(child)
                    child.MergeFrom(copied)
            else:
                log.append(f"ClearField({name})")
                self.both(log[-1], a, b, lambda m: m.ClearField(name))
            return
        if kind < 0.85:
            value = self.value(f)
            log.append(f"{name} = {value!r}")
            self.both(log[-1], a, b, lambda m: setattr(m, name, value))
        else:
            log.append(f"ClearField({name})")
            self.both(log[-1], a, b, lambda m: m.ClearField(name))

    def compare(self, full_name, a, b, log):
        where = " | ".join(log[-6:])
        data = a.SerializeToString(deterministic=True)
        again = self.upb(full_name).FromString(b.SerializeToString())
        assert again.SerializeToString(deterministic=True) == data, f"content after {where}"
        for f in a.DESCRIPTOR.fields:
            if f.has_presence:
                assert a.HasField(f.name) == b.HasField(f.name), f"HasField({f.name}) after {where}"
        for o in a.DESCRIPTOR.oneofs:
            assert a.WhichOneof(o.name) == b.WhichOneof(o.name), f"{o.name} after {where}"
        try:
            text = str(a)
        except KeyError:
            text = None
        assert text is None or str(b) == text, f"str() after {where}"


@pytest.mark.parametrize("seed", range(8))
def test_random_operations_on_messages_agree_with_upb(upb, seed):
    lockstep = _Lockstep(upb, seed)
    for full_name in _LOCKSTEP_TYPES:
        for _ in range(3):
            a, b = upb(full_name)(), ours(full_name)()
            log = []
            for _ in range(40):
                lockstep.step(a, b, log)
                lockstep.compare(full_name, a, b, log)
            copied_a, copied_b = upb(full_name)(), ours(full_name)()
            copied_a.CopyFrom(a)
            copied_b.CopyFrom(b)
            copied_a.MergeFrom(a)
            copied_b.MergeFrom(b)
            lockstep.compare(full_name, copied_a, copied_b, log + ["CopyFrom, MergeFrom"])


def _literal(cls, field):
    message = cls("spark.connect.Expression.Literal")()
    values = {"longs": [3, 1, 2], "doubles": [3.5, 1.0, 2.0], "strings": ["b", "a", "c"]}[field]
    getattr(message.specialized_array, field).values.extend(values)
    return message, getattr(message.specialized_array, field).values


def _arguments(cls):
    message = cls("spark.connect.Expression.UnresolvedFunction")()
    for name in ("b", "a", "c"):
        message.arguments.add().unresolved_attribute.unparsed_identifier = name
    return message, message.arguments


def _options(cls):
    message = cls("spark.connect.Read.DataSource")()
    message.options["b"] = "2"
    message.options["a"] = "1"
    return message, message.options


def _params(cls):
    message = cls("spark.connect.MlParams")()
    message.params["b"].integer = 2
    message.params["a"].string = "x"
    return message, message.params


def _expression(cls):
    message = cls("spark.connect.Expression")()
    message.unresolved_function.function_name = "f"
    message.unresolved_function.arguments.add().literal.integer = 1
    return message, message


# Values of each type of a repeated scalar field, by the type of the field.
_VALUES_OF_TYPES = {
    1: [1.5, float("nan")],  # double
    2: [1.5, -0.0],  # float
    3: [1, -(2**62)],  # int64
    4: [1, 2**64 - 1],  # uint64
    5: [1, -2],  # int32
    6: [1, 2**64 - 1],  # fixed64
    7: [1, 2**32 - 1],  # fixed32
    8: [True, False],  # bool
    9: ["a", "b\u00e9"],  # string
    12: [b"a", b"\xff"],  # bytes
    13: [1, 2**32 - 1],  # uint32
    14: [0, 1],  # enum
    15: [1, -2],  # sfixed32
    16: [1, -(2**62)],  # sfixed64
    17: [1, -2],  # sint32
    18: [1, -(2**62)],  # sint64
}


@pytest.mark.filterwarnings("ignore:invalid value encountered in cast:RuntimeWarning")
def test_arrays_of_repeated_scalar_fields_are_upbs(upb):
    # As upb's `__array__`: a new numpy array of the values, of the type of the field (int32 of an
    # enum, S of bytes and str of strings), or of `numpy.dtype(dtype)` when a dtype is given, None
    # too (float64), so that `numpy.asarray` of a field of floats is of float32, and its errors.
    np = pytest.importorskip("numpy")
    calls = [
        ("asarray", lambda c: np.asarray(c)),
        ("array int8", lambda c: np.array(c, dtype=np.int8)),
        ("__array__()", lambda c: c.__array__()),
        ("__array__(None)", lambda c: c.__array__(None)),
        ("__array__(f4)", lambda c: c.__array__("f4")),
        ("__array__(dtype, copy)", lambda c: c.__array__(dtype=np.int64, copy=False)),
        ("__array__(None, True)", lambda c: c.__array__(None, True)),
        ("__array__(str)", lambda c: c.__array__(str)),
        ("__array__(bool)", lambda c: c.__array__(bool)),
        ("__array__(1, 2, 3)", lambda c: c.__array__(1, 2, 3)),
        ("__array__(nope)", lambda c: c.__array__(nope=1)),
        ("__array__(True)", lambda c: c.__array__(True)),
    ]

    def array(fn, container):
        values = fn(container)
        is_new = not np.shares_memory(values, fn(container))
        return type(values).__name__, values.dtype.str, values.shape, values.tolist(), is_new

    found = {}
    for full_name, cls in sorted(_rust._CLASSES.items()):
        message = cls()
        for field in cls.DESCRIPTOR.fields:
            container = type(getattr(message, field.name)).__name__
            if field.type in _VALUES_OF_TYPES and container == "RepeatedScalarContainer":
                found.setdefault(field.type, (full_name, field.name))
    assert {1, 2, 3, 5, 8, 9} <= set(found), found  # of the arrays of Literal.SpecializedArray
    for field_type, (full_name, name) in sorted(found.items()):
        for values in (_VALUES_OF_TYPES[field_type], []):
            real, mine = getattr(upb(full_name)(), name), getattr(ours(full_name)(), name)
            real.extend(values)
            mine.extend(values)
            for label, fn in calls:
                expected = _outcome(lambda: array(fn, real))
                assert _outcome(lambda: array(fn, mine)) == expected, (full_name, name, label)


_SEQUENCE_CASES = [
    ("abc", lambda c: [isinstance(c, t) for t in (abc.Sequence, abc.MutableSequence, abc.Mapping)]),
    ("len", len),
    ("list", list),
    ("reversed", lambda c: list(reversed(c))),
    ("[-1]", lambda c: c[-1]),
    ("[3]", lambda c: c[3]),
    ("[::-1]", lambda c: c[::-1]),
    ('["a"]', lambda c: c["a"]),
    ("[2**100]", lambda c: c[2**100]),
    ("del [2**100]", lambda c: c.__delitem__(2**100)),
    ("== list", lambda c: c == list(c)),
    ("in", lambda c: c[0] in c),
    ("in, of no __contains__", lambda c: hasattr(c, "__contains__")),
    ("hash", hash),
    ("+ list", lambda c: c + []),
    ("deepcopy", copy.deepcopy),
    ("pickle", lambda c: __import__("pickle").dumps(c)),
    ("pop", lambda c: (c.pop(), list(c))),
    ("pop 9", lambda c: (c.pop(9), list(c))),
    ("pop empty", lambda c: (c.clear(), c.pop())),
    ("remove missing", lambda c: c.remove(object())),
    ("reverse", lambda c: (c.reverse(), list(c))),
    ("sort key", lambda c: (c.sort(key=repr, reverse=True), list(c))),
    ("sort sort_function", lambda c: c.sort(sort_function=None)),
    ("clear", lambda c: (c.clear(), list(c))),
    ("insert", lambda c: (c.insert(-1, c[0]), list(c))),
    ("delitem slice", lambda c: (c.__delitem__(slice(None, None, 2)), list(c))),
    ("MergeFrom", lambda c: (c.MergeFrom(list(c)), len(c))),
    ("extend 5", lambda c: c.extend(5)),
]

_SCALAR_SEQUENCE_CASES = [
    ("np.array", lambda c: __import__("numpy").array(c).tolist()),
    ("np.array dtype", lambda c: str(__import__("numpy").array(c).dtype)),
    ("append object", lambda c: c.append(object())),
    ("setitem", lambda c: (c.__setitem__(0, c[1]), list(c))),
    ("setitem extended slice", lambda c: c.__setitem__(slice(None, None, 2), [c[0]])),
    # As upb's: a slice of another length first moves the items after it, and then the values are
    # set in turn, so one that fails leaves those before it set.
    ("setitem slice, failing", lambda c: c.__setitem__(slice(0, 2), [c[2], object()])),
    ("setitem longer slice, failing", lambda c: c.__setitem__(slice(0, 1), [c[2], c[1], object()])),
    ("setitem shorter slice, failing", lambda c: c.__setitem__(slice(0, 2), [object()])),
    ("setitem extended slice of a generator", lambda c: c.__setitem__(slice(None, None, 2), (v for v in c[:2]))),
    ("setitem slice of 5", lambda c: c.__setitem__(slice(0, 1), 5)),
    ("extend, failing", lambda c: c.extend([c[0], object()])),
    ("count", lambda c: c.count(c[0])),
    ("index", lambda c: c.index(c[0])),
]

_MESSAGE_SEQUENCE_CASES = [
    ("add", lambda c: (c.add(unresolved_attribute={"unparsed_identifier": "z"}), len(c))),
    ("add bad", lambda c: c.add(nope=1)),
    ("append dict", lambda c: c.append({})),
    ("append other type", lambda c: (_outcome(lambda: c.append(c[0].unresolved_attribute)), len(c))),
    ("insert 5", lambda c: c.insert(0, 5)),
    ("setitem", lambda c: c.__setitem__(0, c[1])),
    ('setitem ["a"]', lambda c: c.__setitem__("a", c[1])),
    ("setitem 9", lambda c: c.__setitem__(9, c[1])),
    ("remove", lambda c: (c.remove(c[1]), len(c))),
    # As upb's: append gives the message that it appends, a copy of its argument.
    ("append returns", lambda c: (c.append(c[0]), len(c))),
    ("append returns the element", lambda c: (c.append(c[0]) is c[-1], c.append(c[0]) is c[0])),
    ("add positional", lambda c: c.add(1)),
]

_MAP_CASES = [
    ("abc", lambda c: [isinstance(c, t) for t in (abc.Mapping, abc.MutableMapping, abc.Sequence)]),
    ("keys", lambda c: (type(c.keys()).__name__, sorted(c.keys()))),
    ("items", lambda c: type(c.items()).__name__),
    ("keys == set", lambda c: c.keys() == {"a", "b"}),
    ("missing inserts", lambda c: (c["zz"], len(c))),
    ("[1]", lambda c: c[1]),
    ("get 1", lambda c: c.get(1)),
    ("in 1", lambda c: 1 in c),
    ("== dict", lambda c: c == dict(c)),
    ("pop missing", lambda c: (c.pop("zz"), len(c))),
    ("popitem empty", lambda c: (c.clear(), c.popitem())),
    ("setdefault", lambda c: c.setdefault("q")),
    ("setdefault value", lambda c: c.setdefault("q", "v")),
    ("update pairs", lambda c: c.update([("p", "q")])),
    ("delitem missing", lambda c: c.__delitem__("zz")),
    ("setitem", lambda c: c.__setitem__("n", "v")),
    ("setitem 1", lambda c: c.__setitem__(1, "v")),
    ("MergeFrom dict", lambda c: c.MergeFrom({})),
    ("MergeFrom pairs", lambda c: c.MergeFrom([("m", "n")])),
    ("GetEntryClass", lambda c: c.GetEntryClass().DESCRIPTOR.full_name),
    # As upb's: a map of messages (a MessageMapContainer) has `get_or_create`, the message of the
    # key, which it inserts if it is missing, and a map of scalars has none.
    ("get_or_create", lambda c: (c.get_or_create("zz"), len(c))),
    ("get_or_create present", lambda c: (c.get_or_create("a") is c["a"], len(c))),
    ("get_or_create 1", lambda c: c.get_or_create(1)),
    ("get_or_create no key", lambda c: c.get_or_create()),
    ("get_or_create keyword", lambda c: c.get_or_create(key="a")),
]

_MESSAGE_CASES = [
    ("in", lambda m: ("unresolved_function" in m, "literal" in m)),
    ("in no presence", lambda m: "function_name" in m.unresolved_function),
    ("in 1", lambda m: 1 in m),
    ("HasField unknown", lambda m: m.HasField("nope")),
    ("HasField repeated", lambda m: m.unresolved_function.HasField("arguments")),
    ("WhichOneof field", lambda m: m.WhichOneof("literal")),
    ("ClearField 1", lambda m: m.ClearField(1)),
    ("set unknown", lambda m: setattr(m, "nope", 1)),
    ("set message", lambda m: setattr(m, "literal", 1)),
    ("set repeated", lambda m: setattr(m.unresolved_function, "arguments", [])),
    ("get unknown", lambda m: m.nope),
    ("del", lambda m: delattr(m, "literal")),
    ("init positional", lambda m: type(m)(1)),
    ("init unknown", lambda m: type(m)(nope=1)),
    ("init message 5", lambda m: type(m)(literal=5)),
    ("init repeated 5", lambda m: type(m.unresolved_function)(arguments=5)),
    ("MergeFrom 5", lambda m: m.MergeFrom(5)),
    ("CopyFrom other type", lambda m: m.CopyFrom(m.unresolved_function)),
    ("SerializeToString kwargs", lambda m: m.SerializeToString(nope=True)),
    ("SerializeToString positional", lambda m: m.SerializeToString(True)),
    ("ParseFromString str", lambda m: m.ParseFromString("x")),
    ("FromString corrupt", lambda m: type(m).FromString(b"\xff\xff")),
    ("FromString bad UTF-8", lambda m: type(m.unresolved_attribute).FromString(b"\n\x02\xff\xfe")),
    ("__getstate__", lambda m: m.__getstate__()),
    ("__setstate__", lambda m: (m.__setstate__({"serialized": b""}), m)[1]),
    ("dir", lambda m: "unresolved_function" in dir(m) and "HasField" in dir(m)),
    # As upb's: a message that declares no extension ranges has no `Extensions` (an
    # AttributeError of no message), which `dir` does not list.
    ("Extensions", lambda m: m.Extensions),
    ("dir Extensions", lambda m: ("Extensions" in dir(m), "Extensions" in dir(type(m)))),
    # As upb's: the names of fields of HasField, ClearField and WhichOneof (and `in`) are str or
    # bytes, of UTF-8.
    ("HasField bytes", lambda m: (m.HasField(b"unresolved_function"), m.HasField(b"literal"))),
    ("HasField bytes of a oneof", lambda m: m.HasField(b"expr_type")),
    ("HasField bytes of no field", lambda m: m.HasField(b"nope")),
    ("HasField bytes not UTF-8", lambda m: m.HasField(b"\xff")),
    ("HasField bytearray", lambda m: m.HasField(bytearray(b"literal"))),
    ("WhichOneof bytes", lambda m: m.WhichOneof(b"expr_type")),
    ("WhichOneof bytes of a field", lambda m: m.WhichOneof(b"literal")),
    ("ClearField bytes", lambda m: (m.ClearField(b"unresolved_function"), m.WhichOneof("expr_type"))),
    ("ClearField bytes of a oneof", lambda m: (m.ClearField(b"expr_type"), m.ListFields())),
    ("in bytes", lambda m: (b"unresolved_function" in m, b"literal" in m)),
    # As upb's: the classes have the number of each field, and a message has the attributes of its
    # class and of the metaclass of its class.
    ("FIELD_NUMBER", lambda m: (type(m).LITERAL_FIELD_NUMBER, m.UNRESOLVED_FUNCTION_FIELD_NUMBER)),
    ("attributes of the metaclass", lambda m: (m.__name__, m.__qualname__, m.__module__)),
    ("hash", hash),
    ("== None", lambda m: m == None),  # noqa: E711
    ("deepcopy", lambda m: copy.deepcopy(m) == m),
]


@pytest.mark.filterwarnings("ignore:pop index out of range:FutureWarning")
@pytest.mark.parametrize(
    "make, cases",
    [
        (lambda cls: _literal(cls, "longs"), _SEQUENCE_CASES + _SCALAR_SEQUENCE_CASES),
        (lambda cls: _literal(cls, "doubles"), _SEQUENCE_CASES + _SCALAR_SEQUENCE_CASES),
        (lambda cls: _literal(cls, "strings"), _SEQUENCE_CASES + _SCALAR_SEQUENCE_CASES),
        (_arguments, _SEQUENCE_CASES + _MESSAGE_SEQUENCE_CASES),
        (_options, _MAP_CASES),
        (_params, [c for c in _MAP_CASES if c[0] not in ("setdefault value", "update pairs")]),
        (_expression, _MESSAGE_CASES),
    ],
    ids=["int64s", "doubles", "strings", "messages", "string map", "message map", "message"],
)
def test_containers_and_messages_behave_as_upbs(upb, make, cases):
    # As upb's: the result of each operation (or the type and message of its error), and the
    # message after it. The C names of upb's types are of its module (google._upb._message), and
    # it has a map type of scalars and one of messages.
    def normalized(outcome):
        if outcome[0] == "raise":
            text = outcome[2].replace("google._upb._message.", "")
            for name in ("ScalarMapContainer", "MessageMapContainer"):
                text = text.replace(name, "MapContainer")
            return outcome[:2] + (text,)
        return outcome

    for label, fn in cases:
        real_message, real_value = make(upb)
        message, value = make(ours)
        expected = normalized(_outcome(lambda: fn(real_value)))
        assert normalized(_outcome(lambda: fn(value))) == expected, label
        assert message.SerializeToString() == real_message.SerializeToString(deterministic=True), (
            label
        )


class _Index:
    """An object whose `__index__` is what `index` gives, which may use a container first."""

    def __init__(self, index):
        self.index = index

    def __index__(self):
        return self.index()


class _Equal:
    """An object whose `__eq__` is `result` of anything, after it runs `effect`."""

    def __init__(self, result, effect):
        self.result, self.effect = result, effect

    def __eq__(self, other):
        self.effect()
        return self.result


def _then(values, effect):
    """The values of `values`, each followed by `effect`."""
    for value in values:
        yield value
        effect()


def _then_raising(values, effect):
    """The values of `_then`, then a KeyError."""
    yield from _then(values, effect)
    raise KeyError("values")


def _extend_of_a_list_its_value_appends_to(container):
    values = [2]
    values.insert(0, _Index(lambda: values.append(5) or 1))
    container.extend(values)


def _pop_out_of_range(container, effect):
    """`container.pop(9)` of a container of fewer items, whose warning runs `effect`."""
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = lambda *args, **kwargs: effect()
        return container.pop(9)


def _name(expression):
    return expression.unresolved_attribute.unparsed_identifier


# Python code that a method of a container of the ints 3, 4, ... runs, which uses the container.
_PYTHON_CODE_OF_INTS = [
    # The __index__ of a value: upb converts it, then appends, inserts or sets it in the container
    # as it is by then, where a write past its end is lost.
    ("append, reading the length", [], lambda c: c.append(_Index(lambda: len(c)))),
    ("append, appending", [3], lambda c: c.append(_Index(lambda: c.append(7) or 1))),
    ("append, clearing", [3], lambda c: c.append(_Index(lambda: c.clear() or 1))),
    ("insert, appending", [3, 4], lambda c: c.insert(0, _Index(lambda: c.append(7) or 1))),
    ("set, appending", [3], lambda c: c.__setitem__(0, _Index(lambda: c.append(7) or 1))),
    ("set, clearing", [3, 4], lambda c: c.__setitem__(1, _Index(lambda: c.clear() or 9))),
    ("set, popping", [3, 4], lambda c: c.__setitem__(-1, _Index(lambda: c.pop() and 9))),
    ("set a slice, appending", [3, 4], lambda c: c.__setitem__(slice(0, 1), [_Index(lambda: c.append(7) or 1)])),
    ("extend, appending", [3], lambda c: c.extend([_Index(lambda i=i: c.append(7) or i) for i in (1, 2)])),
    ("extend, clearing and failing", [3, 4], lambda c: c.extend([_Index(lambda: c.clear() or 1), "x"])),
    ("extend, of a list that a value appends to", [3], _extend_of_a_list_its_value_appends_to),
    # The __index__ of an index or of a bound of a slice: upb reads the length of the container
    # before it, and checks the index against that.
    ("insert at, appending", [3, 4], lambda c: c.insert(_Index(lambda: c.append(7) or 5), 1)),
    ("insert at, clearing", [3, 4], lambda c: c.insert(_Index(lambda: c.clear() or 1), 1)),
    ("get at, appending", [3], lambda c: c[_Index(lambda: c.append(7) or 1)]),
    ("get from, appending", [3], lambda c: c[_Index(lambda: c.append(7) or 0) :]),
    ("set at, appending", [3], lambda c: c.__setitem__(_Index(lambda: c.append(7) or 1), 5)),
    ("set at, clearing", [3, 4], lambda c: c.__setitem__(_Index(lambda: c.clear() or 1), 5)),
    ("set from, appending", [3, 4], lambda c: c.__setitem__(slice(_Index(lambda: c.append(7) or 0), 1), [5])),
    ("delete at, appending", [3], lambda c: c.__delitem__(_Index(lambda: c.append(7) or 1))),
    ("delete from, appending", [3, 4], lambda c: c.__delitem__(slice(_Index(lambda: c.append(7) or 0), 1))),
    ("pop at, appending", [3, 4], lambda c: c.pop(_Index(lambda: c.append(7) or 0))),
    ("pop at, clearing", [3, 4], lambda c: c.pop(_Index(lambda: c.clear() or 0))),
    ("pop past the end, warning and appending", [3, 4], lambda c: _pop_out_of_range(c, lambda: c.append(7))),
    ("pop past the end, warning and clearing", [3, 4], lambda c: _pop_out_of_range(c, c.clear)),
    # An iterable: upb takes its values (of PySequence_Fast) before it appends or sets them, and
    # an extend that fails leaves the container of the length that it had.
    ("extend, of a generator that appends", [], lambda c: c.extend(c.append(1) or v for v in [2])),
    ("extend, of a generator that appends after", [], lambda c: c.extend(_then([1, 2], lambda: c.append(9)))),
    ("extend, of a generator that clears", [3], lambda c: c.extend(_then([1, 2], c.clear))),
    ("extend, of a generator that appends and fails", [3], lambda c: c.extend(_then_raising([1], lambda: c.append(9)))),
    ("set a slice, of a generator that appends", [3, 4], lambda c: c.__setitem__(slice(0, 1), (c.append(8) or v for v in [5, 6]))),
    ("set an extended slice, of a generator that appends", [3, 4, 5], lambda c: c.__setitem__(slice(None, None, 2), (c.append(1) or v for v in [7, 8]))),
    # An __eq__: upb's remove compares the items of the container as it is, among as many as it
    # had, `in` reads them in turn, and == compares a list of them.
    ("remove, appending", [3, 4], lambda c: c.remove(_Equal(True, lambda: c.append(7)))),
    ("remove, popping", [3, 4, 5], lambda c: c.remove(_Equal(False, c.pop))),
    ("in, clearing", [3, 4], lambda c: _Equal(False, c.clear) in c),
    ("==, appending", [3, 4], lambda c: c == [_Equal(True, lambda: c.append(7))] * 2),
    ("==, clearing", [3, 4], lambda c: c == [_Equal(True, c.clear)] * 2),
    # The key of a sort: upb sets the sorted items at their positions of the container as it is.
    ("sort, appending", [3, 1, 2], lambda c: c.sort(key=lambda v: c.append(9) or v)),
    ("sort, popping", [3, 1, 2], lambda c: c.sort(key=lambda v: c.pop() and v)),
]  # fmt: skip

# Where upb then reads or writes past the end of the array of a container that the Python code
# cleared (it reads a stale item, or the process ends): a read raises upb's IndexError, and an
# assignment, an insertion or a deletion takes the container as it is.
_PYTHON_CODE_OF_INTS_PAST_THE_END = [
    ("get at, clearing", [3, 4], lambda c: c[_Index(lambda: c.clear() or 1)], ("raise", "IndexError", "list index out of range"), []),
    ("get from, clearing", [3, 4], lambda c: c[_Index(lambda: c.clear() or 0) :], ("ok", []), []),
    ("insert, clearing", [3, 4], lambda c: c.insert(5, _Index(lambda: c.clear() or 1)), ("ok", None), [1]),
    ("set a slice, clearing", [3, 4, 5], lambda c: c.__setitem__(slice(0, 1), _then([5, 6], c.clear)), ("ok", None), [5, 6]),
    ("delete at, clearing", [3, 4], lambda c: c.__delitem__(_Index(lambda: c.clear() or 1)), ("ok", None), []),
    ("delete from, clearing", [3, 4], lambda c: c.__delitem__(slice(_Index(lambda: c.clear() or 0), 2)), ("ok", None), []),
    ("remove, clearing", [3, 4], lambda c: c.remove(_Equal(True, c.clear)), ("ok", None), []),
]  # fmt: skip

# The same of a container of the messages a, b and c.
_PYTHON_CODE_OF_MESSAGES = [
    ("sort, adding", lambda c: c.sort(key=lambda e: (c.add(), _name(e))[1])),
    ("sort, popping", lambda c: c.sort(key=lambda e: (len(c) > 1 and c.pop(), _name(e))[1])),
    ("sort, clearing", lambda c: c.sort(key=lambda e: (c.clear(), _name(e))[1])),
    ("remove, adding", lambda c: c.remove(_Equal(True, c.add))),
    ("remove, popping", lambda c: c.remove(_Equal(False, c.pop))),
    ("==, adding", lambda c: c == [_Equal(True, c.add)] * 3),
    ("get at, adding", lambda c: c[_Index(lambda: (c.add(), 3)[1])]),
    ("insert at, adding", lambda c: c.insert(_Index(lambda: (c.add(), 5)[1]), c[0])),
    ("delete at, adding", lambda c: c.__delitem__(_Index(lambda: (c.add(), 3)[1]))),
    ("delete from, adding", lambda c: c.__delitem__(slice(_Index(lambda: (c.add(), 0)[1]), 1))),
    ("pop past the end, warning and adding", lambda c: _pop_out_of_range(c, c.add)),
    ("pop past the end, warning and clearing", lambda c: _pop_out_of_range(c, c.clear)),
]  # fmt: skip

_PYTHON_CODE_OF_MESSAGES_PAST_THE_END = [
    ("remove, clearing", lambda c: c.remove(_Equal(True, c.clear))),
    ("delete at, clearing", lambda c: c.__delitem__(_Index(lambda: (c.clear(), 1)[1]))),
]

# The same of a map of the strings a: 1 and b: 2.
_PYTHON_CODE_OF_MAPS = [
    ("==, setting", lambda c: c == {"a": _Equal(True, lambda: c.__setitem__("z", "9")), "b": "2"}),
    ("==, clearing", lambda c: c == {"a": _Equal(True, c.clear), "b": "2"}),
    ("update, of a generator that sets", lambda c: c.update((c.__setitem__("g", "0") or p) for p in [("x", "1"), ("y", "2")])),
    ("MergeFrom, of a generator that sets", lambda c: c.MergeFrom((c.__setitem__("g", "0") or p) for p in [("x", "1")])),
    ("update, of keys that clear", lambda c: c.update(type("M", (), {"keys": lambda self: c.clear() or ["x"], "__getitem__": lambda self, k: "1"})())),
]  # fmt: skip


@pytest.mark.filterwarnings("ignore:pop index out of range:FutureWarning")
def test_containers_that_their_python_code_changes_behave_as_upbs(upb):
    # The Python code that a method of a container runs (the __index__ of a value, of an index or
    # of a bound of a slice, an iterable, an __eq__, the key of a sort, a warning) may use the
    # container, as upb's allow, and the method then changes the container as it is, in upb's
    # order: none raises PyO3's PanicException ("Already borrowed", a BaseException), of a borrow
    # of the container that it held while the code ran. Nor does one of another thread.
    def run(cls, full_name, field, items, fn):
        message = cls(full_name)(**{field: items})
        container = getattr(message, field)
        outcome = _outcome(lambda: fn(container))
        return outcome, _comparable(container), message.SerializeToString(deterministic=True)

    for label, items, fn in _PYTHON_CODE_OF_INTS:
        expected = run(upb, "spark.connect.Ints", "values", items, fn)
        assert run(ours, "spark.connect.Ints", "values", items, fn) == expected, label
    for label, items, fn, outcome, after in _PYTHON_CODE_OF_INTS_PAST_THE_END:
        found = run(ours, "spark.connect.Ints", "values", items, fn)
        assert found[:2] == (outcome, ("repeated", after)), label
    arguments = [{"unresolved_attribute": {"unparsed_identifier": n}} for n in "abc"]
    function = "spark.connect.Expression.UnresolvedFunction"
    for label, fn in _PYTHON_CODE_OF_MESSAGES:
        expected = run(upb, function, "arguments", arguments, fn)
        assert run(ours, function, "arguments", arguments, fn) == expected, label
    for label, fn in _PYTHON_CODE_OF_MESSAGES_PAST_THE_END:
        found = run(ours, function, "arguments", arguments, fn)
        assert found[:2] == (("ok", None), ("repeated", [])), label
    options = {"a": "1", "b": "2"}
    for label, fn in _PYTHON_CODE_OF_MAPS:
        expected = run(upb, "spark.connect.Read.DataSource", "options", options, fn)
        assert run(ours, "spark.connect.Read.DataSource", "options", options, fn) == expected, label

    # One thread extends a container of the values of a slow generator while another appends.
    def extend_while_appending(cls):
        message = cls("spark.connect.Ints")()
        errors = []

        def slowly():
            for i in range(200):
                time.sleep(0.0002)
                yield i

        def append():
            for i in range(50):
                try:
                    message.values.append(i)
                except BaseException as e:  # a PanicException is one
                    errors.append(type(e).__name__)
                    return
                time.sleep(0.0005)

        thread = threading.Thread(target=append)
        thread.start()
        try:
            message.values.extend(slowly())
        except BaseException as e:
            errors.append(type(e).__name__)
        thread.join()
        return errors, len(message.values)

    assert extend_while_appending(ours) == extend_while_appending(upb) == ([], 250)


def test_maps_keep_the_order_of_insertion_and_upbs_entries_through_removals(upb):
    # A map finds a key in a hash table of its keys, as upb's does, and keeps the order in which
    # its keys were inserted (which upb's does not, see docs/compatibility.md), through removals,
    # which leave holes among its entries until they are dropped at once: the order of a dict,
    # but of popitem, which takes the first key, as MutableMapping's does. After each of random
    # operations on a few keys, which come and go, its entries and bytes are upb's.
    rng = random.Random(11)
    keys = [f"k{i}" for i in range(24)]
    real, mine = upb("spark.connect.Read.DataSource")(), ours("spark.connect.Read.DataSource")()
    order = {}
    for step in range(4000):
        key, value, kind = rng.choice(keys), f"v{step}", rng.random()
        if kind < 0.35:
            for m in (real, mine):
                m.options[key] = value
            order[key] = value
        elif kind < 0.65:
            outcomes = [_outcome(lambda m=m: m.options.__delitem__(key)) for m in (real, mine)]
            assert outcomes[0] == outcomes[1], (step, key)
            order.pop(key, None)
        elif kind < 0.75:
            # pop of a missing key gives the default, and keeps no entry of it.
            assert real.options.pop(key) == mine.options.pop(key) == order.pop(key, ""), step
        elif kind < 0.8:
            if order:
                first = next(iter(order))
                assert mine.options.popitem() == (first, order.pop(first)), step
                del real.options[first]
        elif kind < 0.9:
            pairs = [(rng.choice(keys), f"u{step}") for _ in range(3)]
            for m in (real, mine):
                m.options.update(pairs)
            order.update(pairs)
        elif kind < 0.98:
            # Reading a missing key inserts it, of the default.
            assert real.options[key] == mine.options[key] == order.setdefault(key, ""), step
        else:
            for m in (real, mine):
                m.options.clear()
            order.clear()
        assert list(mine.options.items()) == list(order.items()), step
        assert len(mine.options) == len(order) and all(k in mine.options for k in order), step
        assert dict(real.options) == order, step
        assert mine.SerializeToString() == real.SerializeToString(deterministic=True), step
    # The same of a map of messages, of which reading a missing key inserts a new message.
    real, mine = upb("spark.connect.SQL")(), ours("spark.connect.SQL")()
    order = {}
    for step in range(2000):
        key = rng.choice(keys)
        if rng.random() < 0.55:
            for m in (real, mine):
                m.named_arguments[key].literal.integer = step
            order.setdefault(key, None)
        else:
            outcomes = [_outcome(lambda m=m: m.named_arguments.__delitem__(key)) for m in (real, mine)]
            assert outcomes[0] == outcomes[1], (step, key)
            order.pop(key, None)
        assert list(mine.named_arguments) == list(order), step
        assert mine.SerializeToString() == real.SerializeToString(deterministic=True), step
    copied = copy.deepcopy(mine)
    assert list(copied.named_arguments) == list(order) and copied == mine


def test_operations_on_maps_of_many_keys_take_no_scan_of_their_keys(upb):
    # As upb's, whose hash table finds a key at once: the parse, the constructor, update,
    # MergeFrom, a copy, ==, the lookups and the removals of a map of 100,000 keys each take a
    # fraction of a second, also of a map of messages, and give upb's entries. The bounds are
    # generous: the maps that scanned their keys for each key took about 20 s for the constructor
    # of a map of 100,000 keys, and as long for the others.
    count = 100_000

    def quick(operation):
        started = time.perf_counter()
        result = operation()
        elapsed = time.perf_counter() - started
        assert elapsed < 3.0, elapsed
        return result

    source = ours("spark.connect.Read.DataSource")
    options = {f"k{i}": f"v{i}" for i in range(count)}
    built = quick(lambda: source(options=options))
    data = built.SerializeToString()
    assert data == upb("spark.connect.Read.DataSource")(options=options).SerializeToString(
        deterministic=True
    )
    assert list(built.options) == list(options)
    parsed = quick(lambda: source.FromString(data))
    assert quick(lambda: parsed == built)
    updated = source()
    quick(lambda: updated.options.update(options))
    merged = source()
    quick(lambda: merged.MergeFrom(built))
    copied = quick(lambda: copy.deepcopy(built))
    assert updated == merged == copied == built
    assert list(copied.options) == list(options)
    assert quick(lambda: sum(f"k{i}" in parsed.options for i in range(count))) == count
    assert quick(lambda: all(parsed.options[f"k{i}"] == f"v{i}" for i in range(count)))
    quick(lambda: [updated.options.__delitem__(f"k{i}") for i in range(0, count, 2)])
    assert list(updated.options) == [f"k{i}" for i in range(1, count, 2)]
    quick(lambda: [updated.options.popitem() for _ in range(count // 4)])
    assert list(updated.options) == [f"k{i}" for i in range(count // 2 + 1, count, 2)]

    sql = ours("spark.connect.SQL")
    arguments = {f"k{i}": {"literal": {"integer": i}} for i in range(count)}
    built = quick(lambda: sql(named_arguments=arguments))
    data = built.SerializeToString()
    assert data == upb("spark.connect.SQL")(named_arguments=arguments).SerializeToString(
        deterministic=True
    )
    parsed = quick(lambda: sql.FromString(data))
    assert quick(lambda: parsed == built)
    merged = sql()
    quick(lambda: merged.MergeFrom(parsed))
    assert merged == built and list(merged.named_arguments) == list(parsed.named_arguments)


_UNSET_CHILD_OPERATIONS = {
    "SerializeToString": lambda c: c.SerializeToString(),
    "ListFields": lambda c: c.ListFields(),
    "HasField": lambda c: c.HasField("named_table"),
    "DiscardUnknownFields": lambda c: c.DiscardUnknownFields(),
    "MergeFromString": lambda c: c.MergeFromString(b""),
    "MergeFromString str": lambda c: c.MergeFromString("x"),
    "ParseFromString": lambda c: c.ParseFromString(b""),
    "CopyFrom": lambda c: c.CopyFrom(type(c)()),
    "CopyFrom wrong": lambda c: c.CopyFrom(1),
    "MergeFrom": lambda c: c.MergeFrom(type(c)()),
    "Clear": lambda c: c.Clear(),
    "ClearField": lambda c: c.ClearField("is_streaming"),
    "ClearField oneof": lambda c: c.ClearField("read_type"),
    "ClearField unknown": lambda c: c.ClearField("nope"),
    "SetInParent": lambda c: c.SetInParent(),
    "set default": lambda c: setattr(c, "is_streaming", False),
    "set bad value": lambda c: setattr(c, "is_streaming", "x"),
    "set message": lambda c: setattr(c, "named_table", 1),
    "set unknown": lambda c: setattr(c, "nope", 1),
    "child set bad": lambda c: setattr(c.named_table, "unparsed_identifier", 5),
    "repeated read": lambda c: (len(c.data_source.paths), c.data_source.paths == []),
    "repeated append bad": lambda c: c.data_source.paths.append(1),
    "repeated extend": lambda c: c.data_source.paths.extend([]),
    "repeated pop": lambda c: c.data_source.paths.pop(),
    "repeated sort": lambda c: c.data_source.paths.sort(),
    "repeated clear": lambda c: c.data_source.paths.clear(),
    "repeated reverse": lambda c: c.data_source.paths.reverse(),
    "map read": lambda c: (c.data_source.options.get("k"), "k" in c.data_source.options),
    "map missing": lambda c: c.data_source.options["k"],
    "map bad key": lambda c: c.data_source.options[1],
    "map delitem": lambda c: c.data_source.options.__delitem__("k"),
    "map setdefault": lambda c: c.data_source.options.setdefault("k"),
    "map MergeFrom dict": lambda c: c.data_source.options.MergeFrom({}),
    "map clear": lambda c: c.data_source.options.clear(),
}


def test_unset_children_set_in_their_parents_as_upbs(upb):
    # An unset child (the `read` of a new Relation) is set in its parent by the operations that
    # upb sets it by, even those that then fail, and only by them.
    def run(cls, operation):
        parent = cls("spark.connect.Relation")()
        child = parent.read
        result = _outcome(lambda: operation(child))
        return result[:2], parent.HasField("read"), child.HasField("data_source")

    for label, operation in _UNSET_CHILD_OPERATIONS.items():
        assert run(ours, operation) == run(upb, operation), label

    # An assignment to an item of a repeated message field sets its unset message too, and then
    # fails, of its index or not.
    def assign(cls, index):
        relation = cls("spark.connect.Relation")()
        expressions = relation.project.expressions
        outcome = _outcome(lambda: expressions.__setitem__(index, cls("spark.connect.Expression")()))
        return outcome, relation.HasField("project")

    for index in (0, "a"):
        assert assign(ours, index) == assign(upb, index), index

    # An unset child of a member of a oneof stays that member's after another member is set, a
    # merge of its parent sets it, and a copy of its parent that has no such field leaves it so.
    for cls in (upb, ours):
        command = cls("spark.connect.MlCommand")()
        write = command.write
        command.delete.SetInParent()
        write.path = "p"
        assert command.WhichOneof("command") == "write"
        relation = cls("spark.connect.Relation")()
        sql = relation.sql
        source = cls("spark.connect.Relation")()
        source.sql.query = "q"
        relation.MergeFrom(source)
        sql.query += "!"
        assert relation.sql.query == "q!"


def _nested_limits(levels):
    """The mapping of the fields of a Relation of `levels` limits."""
    fields = {}
    for _ in range(levels):
        fields = {"limit": {"limit": 1, "input": fields}}
    return fields


def _nested_arguments(levels):
    """The mapping of the fields of an Expression of `levels` functions of one argument."""
    fields = {}
    for _ in range(levels):
        fields = {"unresolved_function": {"function_name": "f", "arguments": [fields]}}
    return fields


def _relation_bytes(levels, unknown_at):
    """The bytes of a Relation of `levels` limits (`Relation{limit: Limit{input: Relation{...},
    limit: 1}}`), the Relation of each level that `unknown_at` takes of the unknown field 999."""
    data = _UNKNOWN if unknown_at(levels) else b""
    for level in reversed(range(levels)):
        limit = _field(1, data) + _tag(2, 0) + _varint(1)
        data = _field(8, limit) + (_UNKNOWN if unknown_at(level) else b"")
    return data


def test_messages_of_nested_mappings_build_as_upbs(upb):
    # A constructor, add(**kwargs) and a map of messages take a mapping of the fields of a
    # message, whose message fields take mappings too, a mapping a level: as upb's, here of 300
    # levels (see test_deep_messages_discard_and_build_on_a_small_stack for thousands).
    def built(cls):
        relation = cls("spark.connect.Relation")(**_nested_limits(300))
        function = cls("spark.connect.Expression.UnresolvedFunction")()
        function.arguments.add(**_nested_arguments(300))
        sql = cls("spark.connect.SQL")(named_arguments={"a": _nested_arguments(300)})
        return [m.SerializeToString(deterministic=True) for m in (relation, function, sql)]

    assert built(ours) == built(upb)


def test_deep_messages_discard_and_build_on_a_small_stack():
    # DiscardUnknownFields of a message of thousands of levels (here parsed, which upb refuses
    # past 100 levels) and the messages of mappings nested as deep finish on a thread of a small
    # stack, as upb's (which builds a message of 1,000 levels of mappings on it, of the same
    # bytes), where they recursed over the levels on the stack and overflowed it from 1,500 and
    # 200 levels. A stack overflow ends the process, so they run in one of their own.
    script = textwrap.dedent(
        """
        import threading

        import pyspark.sql.connect.proto as proto
        from tests.test_proto_rust_offline import (
            _nested_arguments,
            _nested_limits,
            _relation_bytes,
        )

        LEVELS = 5000
        results = []


        def run():
            relation = proto.Relation.FromString(_relation_bytes(LEVELS, lambda level: True))
            relation.DiscardUnknownFields()
            # As upb's: the Relations of 64 levels below it or deeper keep their unknown fields.
            kept = _relation_bytes(LEVELS, lambda level: 2 * level >= 64)
            results.append(relation.SerializeToString() == kept)
            built = proto.Relation(**_nested_limits(LEVELS))
            results.append(built.SerializeToString() == _relation_bytes(LEVELS, lambda level: False))
            function = proto.Expression.UnresolvedFunction()
            function.arguments.add(**_nested_arguments(LEVELS))
            sql = proto.SQL(named_arguments={"a": _nested_arguments(LEVELS)})
            results.append(sql.named_arguments["a"] == function.arguments[0])


        threading.stack_size(512 * 1024)
        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        print(results)
        """
    )
    root = str(pathlib.Path(__file__).resolve().parent.parent)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, cwd=root
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.strip() == "[True, True, True]", result.stderr[-3000:]


# The messages of a large message of the tests of the checkpoints of its operations: steps enough
# for another thread to wait for the GIL at the checkpoints, every 1,024 messages, for a few tens
# of milliseconds at least.
_LARGE = 262144


class _References:
    """Whether an operation is in progress, by `test` of the references to `target` beyond those
    that it has when armed, right before the operation."""

    def __init__(self, target, test):
        self.target = target
        self.test = test
        self.base = None

    def arm(self):
        self.base = sys.getrefcount(self.target)

    def __call__(self):
        return self.test(sys.getrefcount(self.target) - self.base)


class _Length:
    """Whether an operation that sets the entries of `container` is in progress, of the number of
    entries that it has set: more than none, fewer than `count`. Armed, the container is empty."""

    def __init__(self, container, count):
        self.container = container
        self.count = count

    def arm(self):
        self.container.clear()

    def __call__(self):
        return 0 < len(self.container) < self.count


def _large_operation(operation):
    """An operation of `operation` on a message of `_LARGE` messages, what it raises, and whether
    it is in progress, of the references that it holds while it runs."""
    literal_class = ours("spark.connect.Expression.Literal")
    element = _field(literal_class.Array.ELEMENTS_FIELD_NUMBER, literal_class(string="w").SerializeToString())
    data = _field(literal_class.ARRAY_FIELD_NUMBER, element * _LARGE)
    # Each message holds a reference to its class while it lives: those that an operation makes.
    made = _References(literal_class, lambda extra: extra >= 1024)
    if operation in ("parse", "map parse"):
        # The data ends with a corrupt tag, so that the parse fails once it has made its messages,
        # which it releases before it returns, running no Python code: each message holds a
        # reference to its class until then. A map of SQL.args has a Literal of each entry.
        if operation == "map parse":
            value = _field(2, literal_class(string="w").SerializeToString())
            data = b"".join(_field(2, _field(1, b"k%d" % i) + value) for i in range(_LARGE))
            parsed = ours("spark.connect.SQL")
        else:
            parsed = literal_class
        corrupt = data + b"\xff"
        return lambda: parsed.FromString(corrupt), _rust.DecodeError, made
    literal = literal_class.FromString(data)
    # The constructor, the extend and the update of many values (here of a value that fails last,
    # so that the messages that they made are released before they return, running no Python
    # code) take one budget, not one of each message.
    elements = list(literal.array.elements) + [5]
    if operation == "constructor of a list":
        return lambda: literal_class.Array(elements=elements), TypeError, made
    if operation == "extend":
        return lambda: literal_class.Array().elements.extend(elements), TypeError, made
    if operation == "map constructor":
        args = {f"k{i}": value for i, value in enumerate(elements)}
        return lambda: ours("spark.connect.SQL")(args=args), TypeError, made
    if operation == "map update":
        options = {f"k{i}": "v" for i in range(_LARGE)}
        options["bad"] = 5
        source = ours("spark.connect.Read.DataSource")()
        # The map is read by its length: the other thread reads it at the checkpoints, where the
        # update holds no borrow of it.
        in_progress = _Length(source.options, _LARGE)
        return lambda: source.options.update(options), TypeError, in_progress
    operations = {
        "constructor": lambda: ours("spark.connect.Fetch.Method.Args")(param=literal),
        "CopyFrom": lambda: literal_class().CopyFrom(literal),
        "MergeFrom": lambda: literal_class().MergeFrom(literal),
        "deepcopy": lambda: copy.deepcopy(literal),
        "deepcopy of a container": lambda: copy.deepcopy(literal.array.elements),
    }
    # The copy holds a reference to each message that it copies until it has copied it: of the
    # last one, until it ends.
    in_progress = _References(literal.array.elements[-1], lambda extra: extra == 1)
    return operations[operation], None, in_progress


@pytest.mark.parametrize(
    "operation",
    [
        "parse",
        "constructor",
        "CopyFrom",
        "MergeFrom",
        "deepcopy",
        "deepcopy of a container",
        "constructor of a list",
        "extend",
        "map parse",
        "map constructor",
        "map update",
    ],
)
@pytest.mark.parametrize("cancel", [False, True])
def test_python_thread_runs_before_operations_on_large_messages_finish(operation, cancel):
    # The parse, the constructor, the copies, the merges, the extend of a repeated field of
    # messages and the update of a map of many messages, values or entries, which take longer than
    # upb's, let another thread take the GIL every 1,024 of them, and a signal interrupt them, so
    # that the GIL is held as long as upb's at most, for a thousand messages; upb holds it until
    # the operation ends (but in the update of a map, of MutableMapping's Python code). The other
    # thread reads the operation in progress from the references that it holds, or from the
    # entries of the map; a signal raises KeyboardInterrupt in it.
    from _local_data_conversion_probe import LocalDataConversionProbe

    action, failure, in_progress = _large_operation(operation)
    # The other thread takes the GIL at the checkpoints that the operating system schedules it
    # at: an operation in which it read none is run again, a few times, and so is one whose
    # signal came only after it ended, as a read near its end leaves no checkpoint after it.
    # With no checkpoint, it reads none.
    for _ in range(5):
        in_progress.arm()
        probe = LocalDataConversionProbe(in_progress, in_progress, cancel=cancel)
        interrupted = False
        with probe:
            try:
                if failure is None:
                    probe.convert(action)
                else:
                    with pytest.raises(failure):
                        probe.convert(action)
            except KeyboardInterrupt:
                interrupted = True
        assert not probe.failures, probe.failures
        if "progress" in probe.observed and not (
            cancel and "late_or_unproven_signal" in probe.observed
        ):
            break
    progress = probe.observed.get("progress")
    assert progress is not None, probe.observed
    assert progress["captured_at"] < probe.observed["conversion_ended_at"], probe.observed
    assert interrupted is cancel, probe.observed
    if cancel:
        assert probe.observed.get("signal_in_conversion"), probe.observed
    # The messages take the next operation.
    assert ours("spark.connect.Expression.Literal").FromString(b"\x6a\x01w").string == "w"


_AWKWARD_VALUES = [
    "a", b"a", 1.5, True, 2**64, -(2**63) - 1, 2**31, -1, None, b"\xff", "x\udc80",
    bytearray(b"a"), 1e300, float("nan"), "OPERATOR_TYPE_MODEL", "NOPE", 2**1024,
]  # fmt: skip


def test_values_of_fields_convert_and_fail_as_upbs(upb):
    import numpy as np

    values = _AWKWARD_VALUES + [np.float64(1.0), np.int64(2), np.bool_(True), np.uint64(2**63)]
    fields = [
        ("spark.connect.Expression.Literal", f)
        for f in ("integer", "long", "double", "float", "boolean", "string", "binary", "short")
    ] + [("spark.connect.MlOperator", "type")]

    def run(cls, message, field, value):
        m = cls(message)()
        return _outcome(lambda: (setattr(m, field, value), getattr(m, field))[1])

    for message, field in fields:
        for value in values:
            expected = run(upb, message, field, value)
            assert run(ours, message, field, value) == expected, (field, value)


def test_descriptors_are_upbs(upb):
    def fields_of(descriptor):
        return [
            (
                f.name,
                f.full_name,
                f.number,
                f.index,
                f.type,
                f.cpp_type,
                f.is_repeated,
                f.has_presence,
                f.json_name,
                f.is_packed,
                f.has_options,
                f.containing_oneof.name if f.containing_oneof else None,
                f.message_type.full_name if f.message_type else None,
                f.enum_type.full_name if f.enum_type else None,
                _comparable(f.default_value),
            )
            for f in descriptor.fields
        ]

    for full_name, cls in _rust._CLASSES.items():
        real, mine = upb(full_name).DESCRIPTOR, cls.DESCRIPTOR
        assert mine._concrete_class is cls
        assert (mine.name, mine.full_name, mine.has_options, mine.file.name) == (
            real.name,
            real.full_name,
            real.has_options,
            real.file.name,
        )
        assert fields_of(mine) == fields_of(real), full_name
        assert [(o.name, [f.name for f in o.fields]) for o in mine.oneofs] == [
            (o.name, [f.name for f in o.fields]) for o in real.oneofs
        ]
        assert [n.full_name for n in mine.nested_types] == [n.full_name for n in real.nested_types]
        assert [(e.full_name, [(v.name, v.number) for v in e.values]) for e in mine.enum_types] == [
            (e.full_name, [(v.name, v.number) for v in e.values]) for e in real.enum_types
        ]
        file, real_file = mine.file, real.file
        assert (file.package, file.serialized_pb, [d.name for d in file.dependencies]) == (
            real_file.package,
            real_file.serialized_pb,
            [d.name for d in real_file.dependencies],
        )


def _text_cases(cls):
    """A relation of the escapes, floats and maps of the text format, and a user context of an
    Any of a known type."""
    relation = cls("spark.connect.Relation")()
    relation.sql.query = "日本\x7f\n\"'\\"
    for key in ("", "b", "bytes", "日本", "a"):
        relation.sql.named_arguments[key].literal.double = -1.0
    relation.sql.pos_arguments.add().literal.float = 1 / 3
    relation.sql.pos_arguments.add().literal.binary = bytes(range(250, 256))
    relation.sql.pos_arguments.add().literal.double = 0.1
    context = cls("spark.connect.UserContext")()
    context.user_id = "u"
    context.extensions.add().Pack(cls("spark.connect.Relation")(range={"end": 3}))
    return [relation, context]


def test_text_of_messages_as_upbs_and_text_formats(upb):
    # str() of a message is upb's text (doubles as C's %g, every key and value of a map in upb's
    # order of the keys, an Any as its bytes), and the client's one-line text of plan logging is
    # text_format's (doubles as repr, maps in Python's order of the keys without defaults, an Any
    # of a known type expanded).
    from google.protobuf import text_format

    for real, mine in zip(_text_cases(upb), _text_cases(ours)):
        assert mine.SerializeToString() == real.SerializeToString(deterministic=True)
        assert str(mine) == str(real) and repr(mine) == repr(real)
        one_line = text_format.MessageToString(real, as_one_line=True)
        assert _pyspark._proto_message_to_string(mine, as_one_line=True) == one_line
        assert _pyspark._proto_message_to_string(mine) == text_format.MessageToString(real)


def test_messages_work_without_protobuf():
    # Where protobuf cannot be imported, pyspark.sql.connect.proto is of the runtime: messages
    # build, encode, parse, print, copy and pickle, and their containers are sequences and
    # mappings, as protobuf's are; the example plugin's take the package's, the pool of the files
    # finds their descriptors and a Timestamp converts. With upb installed, the bytes and the text
    # are upb's.
    script = textwrap.dedent(
        """
        import collections.abc, copy, datetime, importlib.abc, json, pickle, sys

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "google.protobuf" or name.startswith("google.protobuf."):
                    raise ImportError("no protobuf")

        sys.meta_path.insert(0, Block())
        import numpy as np
        import pyspark.sql.connect.proto as proto
        from pyspark import _pyspark
        from pyspark.sql.connect.proto import example_plugins_pb2, expressions_pb2
        from tests.test_proto_rust_offline import _text_cases

        assert issubclass(proto.Relation, _pyspark.ProtoMessage)
        assert expressions_pb2.Expression is proto.Expression
        assert "google.protobuf" not in sys.modules
        plugin = example_plugins_pb2.ExamplePluginRelation(input=proto.Relation(range={"end": 3}))
        assert plugin.input.range.end == 3
        pool = proto.Relation.DESCRIPTOR.file.pool
        assert pool.FindMessageTypeByName("spark.connect.Relation") is proto.Relation.DESCRIPTOR
        event = proto.PipelineEvent(timestamp=datetime.datetime(2020, 1, 2, 3, 4, 5, 678000))
        assert event.timestamp.ToJsonString() == "2020-01-02T03:04:05.678Z"
        # The vendored context manager, which packs into protobuf's Any, is the runtime's.
        from pyspark.pipelines import add_pipeline_analysis_context as analysis
        from pyspark.sql.connect.proto import _rust

        assert analysis.add_pipeline_analysis_context is _rust.add_pipeline_analysis_context

        relation, context = _text_cases(lambda name: _rust._CLASSES[name])
        values = relation.sql.pos_arguments.add().literal.specialized_array.doubles.values
        values.extend([1.5, 2.5])
        unpacked = proto.Relation()
        assert context.extensions[0].Unpack(unpacked) and unpacked.range.end == 3
        nested = proto.Expression.Literal(string="nested")
        try:
            proto.Plan.FromString(b"\\xff")
        except _rust.DecodeError as error:
            decode_error = str(error)
        out = {
            "bytes": [relation.SerializeToString().hex(), context.SerializeToString().hex()],
            "str": [str(relation), str(context)],
            "pickle": pickle.loads(pickle.dumps(relation)) == relation,
            "pickle nested": pickle.loads(pickle.dumps(nested)) == nested,
            "deepcopy": copy.deepcopy(context) == context,
            "numpy": np.array(values).tolist(),
            "sequence": isinstance(values, collections.abc.MutableSequence),
            "mapping": isinstance(relation.sql.named_arguments, collections.abc.MutableMapping),
            "enum": proto.MlOperator(type="OPERATOR_TYPE_MODEL").type,
            "decode error": decode_error,
        }
        print(json.dumps(out))
        """
    )
    root = str(pathlib.Path(__file__).resolve().parent.parent)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, cwd=root
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["pickle"] and out["pickle nested"] and out["deepcopy"]
    assert out["numpy"] == [1.5, 2.5] and out["sequence"] and out["mapping"]
    assert out["enum"] == 4
    assert out["decode error"] == (
        "Error parsing message with type 'spark.connect.Plan': Wire format was corrupt"
    )
    upb = _upb_classes()
    if upb is not None:
        relation, context = _text_cases(upb)
        relation.sql.pos_arguments.add().literal.specialized_array.doubles.values.extend([1.5, 2.5])
        assert out["bytes"] == [
            relation.SerializeToString(deterministic=True).hex(),
            context.SerializeToString(deterministic=True).hex(),
        ]
        assert out["str"] == [str(relation), str(context)]


def test_enums_of_the_classes_and_modules_are_upbs(upb):
    # The enum wrappers of the classes, as protobuf's EnumTypeWrapper, of their errors too.
    real_operator, mine_operator = upb("spark.connect.MlOperator"), ours("spark.connect.MlOperator")
    for real, mine in [
        (real_operator.OperatorType, mine_operator.OperatorType),
        (upb("spark.connect.Join").JoinType, ours("spark.connect.Join").JoinType),
    ]:
        assert mine.keys() == real.keys() and mine.values() == real.values()
        assert mine.items() == real.items() and mine.ValueType is real.ValueType
        name = real.keys()[1]
        assert mine.Name(real.Value(name)) == name and mine.Value(name) == real.Value(name)
        assert getattr(mine, name) == getattr(real, name)
        assert mine.DESCRIPTOR.full_name == real.DESCRIPTOR.full_name
        for fn in (
            lambda e: e.Name(12345),
            lambda e: e.Name("x"),
            lambda e: e.Name([]),
            lambda e: e.Value("NOPE"),
            lambda e: e.NOPE,
            list,
            lambda e: e | None,
        ):
            outcome, expected = _outcome(lambda: fn(mine)), _outcome(lambda: fn(real))
            assert outcome[:2] == expected[:2]
            if outcome[0] == "raise" and expected[1] != "TypeError":
                assert outcome[2] == expected[2]
    assert mine_operator.OPERATOR_TYPE_MODEL == real_operator.OPERATOR_TYPE_MODEL


def test_install_exports_the_messages_and_modules_of_the_runtime(upb):
    # install(): the package's names and its generated modules become the runtime's, as the
    # package's star imports of the generated modules give protobuf's: the public names of the
    # generated modules of the ten protos that it imports, and the modules.
    from google.protobuf import descriptor_pool

    names = {}
    saved = {k: v for k, v in sys.modules.items() if k.startswith("pyspark.sql.connect.proto.")}
    try:
        _rust.install(names)
        assert sys.modules["pyspark.sql.connect.proto.base_pb2"].Plan is ours("spark.connect.Plan")
    finally:
        sys.modules.update(saved)
    assert names["Relation"] is ours("spark.connect.Relation")
    assert names["relations_pb2"].Relation is names["Relation"]
    pool = descriptor_pool.Default()
    exported = set()
    for stem in _rust._EXPORTED:
        generated = _generated_names(pool.FindFileByName(f"spark/connect/{stem}.proto"))
        exported |= {n for n in generated if not n.startswith("_")}
    modules = {name.rpartition(".")[2] for name in _rust._MODULES if _rust._is_exported(name)}
    assert set(names) == exported | modules, sorted(set(names) ^ (exported | modules))[:5]
    assert "example_plugins_pb2" in modules and "ExamplePluginRelation" not in names


def test_pipeline_analysis_context_of_the_runtime_is_the_vendored_ones(upb, monkeypatch):
    # The context manager of the runtime adds the user context extension that the vendored one
    # adds with protobuf's Any, and removes it, also when its body raises. With protobuf, the
    # vendored module keeps its own, which packs a message of the runtime into protobuf's Any (as
    # Apache Spark's tests of it patch that Any); without, it has the runtime's (see
    # pyspark._proto_hooks and test_messages_work_without_protobuf).
    import types

    import google.protobuf
    import pyspark.pipelines.add_pipeline_analysis_context as module
    from pyspark import _proto_hooks

    vendored = module.add_pipeline_analysis_context
    assert vendored is not _rust.add_pipeline_analysis_context
    assert _proto_hooks._protobuf_any_imports()
    monkeypatch.setitem(sys.modules, "google.protobuf.any_pb2", None)
    monkeypatch.delattr(google.protobuf, "any_pb2", raising=False)
    assert not _proto_hooks._protobuf_any_imports()
    monkeypatch.undo()
    name = _rust.add_pipeline_analysis_context.__module__
    try:
        fake = types.ModuleType("_vendored_analysis_context")
        _proto_hooks._give_runtime_context(fake)
        assert fake.add_pipeline_analysis_context is _rust.add_pipeline_analysis_context
        assert _rust.add_pipeline_analysis_context.__module__ == fake.__name__
    finally:
        _rust.add_pipeline_analysis_context.__module__ = name

    class Client:
        def __init__(self):
            self.added, self.removed = [], []

        def add_threadlocal_user_context_extension(self, extension):
            self.added.append(extension.SerializeToString())
            return f"id{len(self.added)}"

        def remove_user_context_extension(self, extension_id):
            self.removed.append(extension_id)

    class Spark:
        def __init__(self):
            self.client = Client()

    for flow_name in ("flow", None):
        mine, real = Spark(), Spark()
        with _rust.add_pipeline_analysis_context(mine, "graph", flow_name):
            pass
        with vendored(real, "graph", flow_name):
            pass
        assert (mine.client.added, mine.client.removed) == (real.client.added, real.client.removed)
    failing = Spark()
    with pytest.raises(KeyError):
        with _rust.add_pipeline_analysis_context(failing, "graph", "flow"):
            raise KeyError("body")
    assert failing.client.removed == ["id1"]


def _module_alias(file_name):
    """protoc's name of the module of the file ``file_name`` in a generated module that imports
    it: ``spark_dot_connect_dot_relations__pb2``."""
    module = file_name[: -len(".proto")].replace("/", ".") + "_pb2"
    return module.replace("_", "__").replace(".", "_dot_")


def _generated_names(file_descriptor):
    """The names of the module that protoc generates of the file of ``file_descriptor`` (one of
    upb's), as protobuf's builder binds them, but those of protobuf itself (``_globals``,
    ``_sym_db``, ...): its messages, enums, top-level enum values and services and their
    descriptors, its ``DESCRIPTOR`` and the modules of the files that it imports."""
    names = {"DESCRIPTOR"} | {_module_alias(d.name) for d in file_descriptor.dependencies}

    def nested(descriptor, prefix):
        for name, message in descriptor.nested_types_by_name.items():
            names.add(prefix + name.upper())
            nested(message, prefix + name.upper() + "_")
        for enum in descriptor.enum_types:
            names.add(prefix + enum.name.upper())

    for name, message in file_descriptor.message_types_by_name.items():
        names |= {name, "_" + name.upper()}
        nested(message, "_" + name.upper() + "_")
    for name, enum in file_descriptor.enum_types_by_name.items():
        names |= {name, "_" + name.upper()} | {v.name for v in enum.values}
    names |= {"_" + name.upper() for name in file_descriptor.services_by_name}
    return names


def test_modules_of_the_protos_are_the_generated_modules(upb):
    # Each module of a proto of Spark Connect has the names of the module that protoc generates of
    # it, and imports, reloads and is found as a module.
    from google.protobuf import descriptor_pool

    pool = descriptor_pool.Default()
    for module_name, module in _rust._MODULES.items():
        if not _rust._is_exported(module_name):
            continue
        names = {n for n in vars(module) if not (n.startswith("__") and n.endswith("__"))}
        assert names == _generated_names(pool.FindFileByName(module.DESCRIPTOR.name)), module_name
        assert sys.modules[module_name] is module
        assert importlib.util.find_spec(module_name) is module.__spec__
        assert getattr(proto, module_name.rpartition(".")[2]) is module
    relations = proto.relations_pb2
    assert relations._RELATION is proto.Relation.DESCRIPTOR
    assert relations._JOIN_JOINTYPE is proto.Join.DESCRIPTOR.enum_types_by_name["JoinType"]
    assert relations.spark_dot_connect_dot_expressions__pb2 is proto.expressions_pb2
    assert proto.base_pb2._SPARKCONNECTSERVICE.full_name == "spark.connect.SparkConnectService"
    # A module that is no longer in sys.modules imports again as it is, and reloads as it is.
    name = "pyspark.sql.connect.proto.relations_pb2"
    del sys.modules[name]
    try:
        assert importlib.import_module(name) is relations
    finally:
        sys.modules[name] = relations
    assert importlib.reload(relations) is relations and relations.Relation is proto.Relation


def test_options_and_protos_of_the_descriptors_are_upbs(upb):
    # GetOptions() gives the options of protobuf's descriptor_pb2, the same object each time, of
    # which has_options tells whether there are any, and CopyToProto merges the proto of the
    # descriptor into a message of its type, as upb's of the messages, fields, oneofs, enums,
    # enum values, files, services and methods of the protos of Spark Connect.
    from google.protobuf import descriptor_pb2, descriptor_pool

    def options(descriptor):
        found = descriptor.GetOptions()
        assert descriptor.GetOptions() is found
        return (type(found).__name__, found.SerializeToString(), descriptor.has_options)

    def copied(descriptor, proto_class):
        copy = proto_class()
        copy.MergeFromString(b"")
        return (_outcome(lambda: descriptor.CopyToProto(copy)), copy)

    def check_enum(real, mine):
        assert options(mine) == options(real), mine.full_name
        assert mine.is_closed is real.is_closed is False
        assert copied(mine, descriptor_pb2.EnumDescriptorProto) == copied(
            real, descriptor_pb2.EnumDescriptorProto
        )
        for real_value, value in zip(real.values, mine.values):
            assert options(value) == options(real_value), value.name

    pool = descriptor_pool.Default()
    files = set()
    for full_name, cls in _rust._CLASSES.items():
        if not full_name.startswith("spark.connect."):
            continue
        real, mine = upb(full_name).DESCRIPTOR, cls.DESCRIPTOR
        assert options(mine) == options(real), full_name
        assert copied(mine, descriptor_pb2.DescriptorProto) == copied(
            real, descriptor_pb2.DescriptorProto
        )
        for real_field, field in zip(real.fields, mine.fields):
            assert options(field) == options(real_field), field.full_name
        for real_oneof, oneof in zip(real.oneofs, mine.oneofs):
            assert options(oneof) == options(real_oneof), oneof.full_name
        for real_enum, enum in zip(real.enum_types, mine.enum_types):
            check_enum(real_enum, enum)
        files.add(mine.file.name)
    # A map entry's options are its map_entry, of which json_format and text_format tell a map.
    entry = proto.SQL.DESCRIPTOR.nested_types_by_name["NamedArgumentsEntry"]
    assert entry.GetOptions().map_entry and entry.has_options
    deprecated = proto.SQL.DESCRIPTOR.fields_by_name["args"]
    assert deprecated.GetOptions().deprecated and deprecated.has_options

    for name in sorted(files):
        real, mine = pool.FindFileByName(name), proto.Relation.DESCRIPTOR.file.pool.FindFileByName(name)
        assert options(mine) == options(real), name
        assert copied(mine, descriptor_pb2.FileDescriptorProto) == copied(
            real, descriptor_pb2.FileDescriptorProto
        )
        for enum_name in real.enum_types_by_name:
            check_enum(real.enum_types_by_name[enum_name], mine.enum_types_by_name[enum_name])
        assert list(mine.services_by_name) == list(real.services_by_name)
        for service_name, real_service in real.services_by_name.items():
            service = mine.services_by_name[service_name]
            attributes = ("name", "full_name", "index", "has_options")
            assert [getattr(service, a) for a in attributes] == [
                getattr(real_service, a) for a in attributes
            ]
            assert service.file is mine and options(service) == options(real_service)
            assert copied(service, descriptor_pb2.ServiceDescriptorProto) == copied(
                real_service, descriptor_pb2.ServiceDescriptorProto
            )
            assert list(service.methods_by_name) == list(real_service.methods_by_name)
            for real_method, method in zip(real_service.methods, service.methods):
                assert service.FindMethodByName(method.name) is method
                assert method.containing_service is service
                attributes = (
                    "name",
                    "full_name",
                    "index",
                    "client_streaming",
                    "server_streaming",
                    "has_options",
                )
                assert [getattr(method, a) for a in attributes] == [
                    getattr(real_method, a) for a in attributes
                ]
                assert (method.input_type.full_name, method.output_type.full_name) == (
                    real_method.input_type.full_name,
                    real_method.output_type.full_name,
                )
                assert method.input_type is ours(method.input_type.full_name).DESCRIPTOR
                assert options(method) == options(real_method)
                assert copied(method, descriptor_pb2.MethodDescriptorProto) == copied(
                    real_method, descriptor_pb2.MethodDescriptorProto
                )
            assert _outcome(lambda: service.FindMethodByName("nope")) == _outcome(
                lambda: real_service.FindMethodByName("nope")
            )

    # The errors of CopyToProto of a message of another type and of an object that is no message.
    real, mine = upb("spark.connect.Relation").DESCRIPTOR, proto.Relation.DESCRIPTOR
    wrong = descriptor_pb2.FileDescriptorProto
    assert _outcome(lambda: mine.CopyToProto(wrong())) == _outcome(lambda: real.CopyToProto(wrong()))
    outcome, expected = _outcome(lambda: mine.CopyToProto(1)), _outcome(lambda: real.CopyToProto(1))
    assert outcome == expected


def test_pool_of_the_files_finds_their_descriptors_as_upbs(upb):
    # The pool of a file of the runtime finds the descriptors of the runtime by name, as upb's
    # default pool finds its own, and fails on the names that it does not have as upb's does.
    from google.protobuf import descriptor_pool

    real_pool, pool = descriptor_pool.Default(), proto.Relation.DESCRIPTOR.file.pool
    assert proto.Plan.DESCRIPTOR.file.pool is pool

    def file_of(pool, symbol):
        return pool.FindFileContainingSymbol(symbol).name

    def check_enum(enum, scope):
        assert pool.FindEnumTypeByName(enum.full_name) is enum
        for value in enum.values:
            symbol = f"{scope}.{value.name}"
            assert file_of(pool, symbol) == file_of(real_pool, symbol), symbol

    files = {}
    for full_name, cls in _rust._CLASSES.items():
        descriptor = cls.DESCRIPTOR
        assert pool.FindMessageTypeByName(full_name) is descriptor
        assert pool.FindFileContainingSymbol(full_name) is descriptor.file
        files[descriptor.file.name] = descriptor.file
        for field in descriptor.fields:
            assert pool.FindFieldByName(field.full_name) is field
            assert file_of(pool, field.full_name) == file_of(real_pool, field.full_name)
        for oneof in descriptor.oneofs:
            assert pool.FindOneofByName(oneof.full_name) is oneof
            assert file_of(pool, oneof.full_name) == file_of(real_pool, oneof.full_name)
        for enum in descriptor.enum_types:
            check_enum(enum, full_name)
    for name, file in files.items():
        assert pool.FindFileByName(name) is file
        for enum in file.enum_types_by_name.values():
            check_enum(enum, file.package)
        for service in file.services_by_name.values():
            assert pool.FindServiceByName(service.full_name) is service
            assert pool.FindFileContainingSymbol(service.full_name) is file
            for method in service.methods:
                assert pool.FindMethodByName(method.full_name) is method
                # Neither finds the file of a method.
                assert _outcome(lambda: file_of(pool, method.full_name)) == _outcome(
                    lambda: file_of(real_pool, method.full_name)
                )

    real_relation = upb("spark.connect.Relation").DESCRIPTOR
    for method, args, real_args in [
        ("FindMessageTypeByName", ("nope.X",), None),
        ("FindMessageTypeByName", ("spark.connect.Relation.sql",), None),
        ("FindFileByName", ("nope.proto",), None),
        ("FindEnumTypeByName", ("nope.E",), None),
        ("FindEnumTypeByName", ("spark.connect.Relation",), None),
        ("FindFieldByName", ("spark.connect.Relation.nope",), None),
        ("FindFieldByName", ("nope",), None),
        ("FindOneofByName", ("spark.connect.Relation.nope",), None),
        ("FindExtensionByName", ("spark.connect.nope",), None),
        ("FindServiceByName", ("spark.connect.Nope",), None),
        ("FindMethodByName", ("spark.connect.SparkConnectService.Nope",), None),
        ("FindMethodByName", ("Nope",), None),
        ("FindFileContainingSymbol", ("spark.connect.Nope",), None),
        ("FindFileContainingSymbol", ("spark.connect.Relation.nope",), None),
        ("FindExtensionByNumber", (proto.Relation.DESCRIPTOR, 5), (real_relation, 5)),
        ("FindAllExtensions", (proto.Relation.DESCRIPTOR,), (real_relation,)),
    ]:
        outcome = _outcome(lambda: getattr(pool, method)(*args))
        expected = _outcome(lambda: getattr(real_pool, method)(*(real_args or args)))
        if method == "FindFieldByName" and not _protobuf_7():
            # protobuf 6 names the field a message in its error.
            outcome, expected = outcome[:2], expected[:2]
        assert outcome == expected, (method, args)


def _result(fn):
    """The value of `fn`, or the type and message of what it raises, as it is (a datetime too)."""
    try:
        return ("ok", fn())
    except Exception as e:  # noqa: BLE001
        return ("raise", type(e).__name__, str(e))


def _same_results(mine, expected, exact=True):
    """Whether two results agree: in full, or (``exact`` false, of an error message that
    protobuf 6 words otherwise) in whether they raise."""
    return mine == expected if exact else mine[0] == expected[0]


def test_well_known_types_have_the_methods_of_protobufs(upb):
    # The classes of google.protobuf.Any and google.protobuf.Timestamp have the methods of
    # protobuf's, of their results and errors, and the other classes none of those.
    from google.protobuf import duration_pb2

    methods = {"Pack", "Unpack", "Is", "TypeName", "ToJsonString", "FromDatetime", "ToDatetime"}
    for full_name, cls in _rust._CLASSES.items():
        real = upb(full_name)
        assert {m for m in methods if hasattr(cls, m)} == {m for m in methods if hasattr(real, m)}

    # Any: a message of either runtime packs into the Any of either, of each type URL prefix.
    def pack(any_of, message_of, prefix):
        packed = any_of("google.protobuf.Any")()
        packed.Pack(message_of("spark.connect.Range")(start=1, end=3), prefix)
        return (packed.type_url, packed.value)

    for prefix in ("type.googleapis.com/", "", "x", "x/", "a/b/"):
        expected = pack(upb, upb, prefix)
        assert pack(ours, ours, prefix) == pack(ours, upb, prefix) == expected
    for type_url in (
        "",
        "spark.connect.Range",
        "x/spark.connect.Range",
        "a/b/spark.connect.Range",
        "x/spark.connect.Relation",
        "x/",
    ):

        def unpack(of):
            packed = of("google.protobuf.Any")(type_url=type_url, value=b"\x08\x01\x10\x03")
            target = of("spark.connect.Range")()
            return (
                _outcome(lambda: packed.Unpack(target)),
                target.SerializeToString(),
                packed.TypeName(),
                packed.Is(target.DESCRIPTOR),
            )

        assert unpack(ours) == unpack(upb), type_url

    # Timestamp: its text, conversions and arithmetic, in its range and out of it.
    stamps = [(0, 0), (1, 5000), (-1, 999_999_999), (1_600_000_000, 123_000_000)]
    stamps += [(1_600_000_000, 123_456_000), (1_600_000_000, 123_456_789), (-62135596800, 0)]
    stamps += [(253402300799, 999_999_999), (253402300800, 0), (-62135596801, 0)]
    stamps += [(0, -1), (0, 1_000_000_000)]
    eastern = datetime.timezone(datetime.timedelta(hours=-5))
    readers = [
        lambda t: t.ToJsonString(),
        lambda t: t.ToNanoseconds(),
        lambda t: t.ToMicroseconds(),
        lambda t: t.ToMilliseconds(),
        lambda t: t.ToSeconds(),
        lambda t: t.ToDatetime(),
        lambda t: t.ToDatetime(datetime.timezone.utc),
        lambda t: t.ToDatetime(eastern),
        lambda t: t + datetime.timedelta(days=1, microseconds=7),
        lambda t: datetime.timedelta(seconds=-3) + t,
        lambda t: t - datetime.timedelta(hours=2),
        lambda t: t - datetime.datetime(2000, 1, 1),
        lambda t: datetime.datetime(2000, 1, 1) - t,
        lambda t: t + duration_pb2.Duration(seconds=5, nanos=1000),
        lambda t: t - duration_pb2.Duration(seconds=5),
        lambda t: t - type(t)(seconds=10, nanos=500_000),
        lambda t: t - upb("google.protobuf.Timestamp")(seconds=10),
    ]
    real_class, mine_class = upb("google.protobuf.Timestamp"), ours("google.protobuf.Timestamp")
    for seconds, nanos in stamps:
        for read in readers:
            expected = _result(lambda: read(real_class(seconds=seconds, nanos=nanos)))
            assert _result(lambda: read(mine_class(seconds=seconds, nanos=nanos))) == expected, (
                seconds,
                nanos,
            )

    def set_by(cls, setter):
        stamp = cls()
        return _result(lambda: (setter(stamp), (stamp.seconds, stamp.nanos))[1])

    texts = [
        "1970-01-01T00:00:00Z",
        "2020-01-02T03:04:05.678Z",
        "2020-01-02T03:04:05.123456789Z",
        "2020-01-02T03:04:05.1234567891Z",
        "2020-01-02T03:04:05.5+09:30",
        "2020-01-02T03:04:05-05:00",
        "0001-01-01T00:00:00Z",
        "9999-12-31T23:59:59.999999999Z",
        "2020-01-02T03:04:05",
        "2020-01-02t03:04:05Z",
        "2020-01-02T03:04:05Zx",
        "2020-01-02T03:04:05+0900",
        "2020-01-02T03:04:05.x1Z",
        "2020-01-02",
        "",
        "Z",
        5,
    ]
    for text in texts:
        expected = set_by(real_class, lambda t: t.FromJsonString(text))
        assert set_by(mine_class, lambda t: t.FromJsonString(text)) == expected, text
    for value in (0, 1, -1, 1_500, -1_500, 1_234_567_891, -(10**20), 10**20):
        for name in ("FromNanoseconds", "FromMicroseconds", "FromMilliseconds", "FromSeconds"):
            expected = set_by(real_class, lambda t: getattr(t, name)(value))
            assert set_by(mine_class, lambda t: getattr(t, name)(value)) == expected, (name, value)
    for value, exact in [
        (datetime.datetime(2020, 1, 2, 3, 4, 5, 678901), True),
        (datetime.datetime(2020, 1, 2, 3, 4, 5, tzinfo=eastern), True),
        (datetime.datetime(1, 1, 1), True),
        (datetime.datetime(9999, 12, 31, 23, 59, 59, 999999), True),
        (datetime.date(2020, 1, 2), _protobuf_7()),
        ("2020-01-02", _protobuf_7()),
        (None, _protobuf_7()),
        (0, _protobuf_7()),
        # An object of a class named datetime that is not one, as protobuf takes it.
        (type("datetime", (), {})(), True),
    ]:
        expected = set_by(real_class, lambda t: t.FromDatetime(value))
        assert _same_results(set_by(mine_class, lambda t: t.FromDatetime(value)), expected, exact)
    now = mine_class()
    now.GetCurrentTime()
    assert abs(now.ToDatetime() - datetime.datetime.utcnow()) < datetime.timedelta(minutes=1)


def test_timestamp_fields_take_datetimes_as_upbs(upb):
    # A Timestamp field takes a datetime, through its message's _internal_assign, in the
    # constructor and as an assignment, as upb's does, and fails on other values as upb's.
    dt = datetime.datetime(2020, 1, 2, 3, 4, 5, 678901)
    aware = dt.replace(tzinfo=datetime.timezone(datetime.timedelta(hours=9)))

    def construct(cls, value):
        event = cls("spark.connect.PipelineEvent")(timestamp=value, message="m")
        return (event.HasField("timestamp"), event.SerializeToString())

    def assign(cls, value):
        event = cls("spark.connect.PipelineEvent")()
        outcome = _result(lambda: setattr(event, "timestamp", value))
        return (outcome, event.HasField("timestamp"), event.SerializeToString())

    values = [(dt, True), (aware, True), ({"seconds": 5}, True), (None, True)]
    values += [(1, _protobuf_7()), ("x", _protobuf_7()), (datetime.date(2020, 1, 2), _protobuf_7())]
    for value, exact in values:
        expected = _result(lambda: construct(upb, value))
        assert _same_results(_result(lambda: construct(ours, value)), expected, exact), value
        expected = assign(upb, value)
        mine = assign(ours, value)
        assert mine[1:] == expected[1:] and _same_results(mine[0], expected[0], exact), value
    stamp = upb("google.protobuf.Timestamp")(seconds=7)
    assert construct(ours, ours("google.protobuf.Timestamp")(seconds=7)) == construct(upb, stamp)
    # A field of a message type of Spark Connect is not assigned, as upb's.
    relation = proto.Relation()
    real = upb("spark.connect.Relation")()
    assert _result(lambda: setattr(relation, "sql", proto.SQL())) == _result(
        lambda: setattr(real, "sql", real.sql)
    )


def test_extensions_of_the_messages_are_upbs(upb):
    # The messages declare no extension ranges: HasExtension and ClearExtension fail on any
    # field, and Extensions is an AttributeError, as upb's.
    real, mine = upb("spark.connect.Relation")(), proto.Relation()
    real_field = real.DESCRIPTOR.fields_by_name["sql"]
    field = mine.DESCRIPTOR.fields_by_name["sql"]
    for method in ("HasExtension", "ClearExtension"):
        assert _outcome(lambda: getattr(mine, method)(field)) == _outcome(
            lambda: getattr(real, method)(real_field)
        )
        for other in ("x", 1, None):
            assert _outcome(lambda: getattr(mine, method)(other)) == _outcome(
                lambda: getattr(real, method)(other)
            )
    assert _outcome(lambda: mine.Extensions) == _outcome(lambda: real.Extensions)
    assert hasattr(mine, "Extensions") is hasattr(real, "Extensions") is False
    assert ("Extensions" in dir(type(mine))) is ("Extensions" in dir(type(real))) is True


def test_pickles_of_nested_messages_load_as_upbs(upb):
    # As upb's: a pickle of a message of a nested type made by upb, which protobuf reduces to its
    # `_InternalConstructMessage` of the full name of the type, a class of protobuf's default
    # symbol database, loads where protobuf is installed: the runtime registers its classes in
    # the database of the files that it adds to protobuf's default pool, so it loads as a message
    # of the runtime. A pickle of the runtime's, of its class (by its module and qualified name,
    # those of the class of upb), loads of its class.
    import pickle

    from google.protobuf import symbol_database

    for full_name, fields in [
        ("spark.connect.Expression.Literal", {"string": "x"}),
        ("spark.connect.Expression.UnresolvedFunction", {"function_name": "f"}),
    ]:
        loaded = pickle.loads(pickle.dumps(upb(full_name)(**fields)))
        assert type(loaded) is ours(full_name), full_name
        assert loaded.SerializeToString() == ours(full_name)(**fields).SerializeToString()
        assert symbol_database.Default().GetSymbol(full_name) is ours(full_name)
    for full_name, fields in [
        ("spark.connect.Expression.Literal", {"string": "x"}),
        ("spark.connect.Relation", {"sql": {"query": "q"}}),
    ]:
        mine = ours(full_name)(**fields)
        loaded = pickle.loads(pickle.dumps(mine))
        assert type(loaded) is ours(full_name) and loaded == mine, full_name


# Unknown fields of each wire type: a varint, a fixed64, bytes, a group of a varint and bytes, a
# fixed32 and bytes of a message.
_UNKNOWN_FIELDS = (
    _tag(501, 0) + _varint(5)
    + _tag(502, 1) + (7).to_bytes(8, "little")
    + _field(503, b"abc")
    + _tag(504, 3) + _tag(1, 0) + _varint(9) + _field(2, b"x") + _tag(504, 4)
    + _tag(505, 5) + (9).to_bytes(4, "little")
    + _field(506, _tag(1, 0) + _varint(1))
)  # fmt: skip


def _unknown_field_cases(cls):
    """Messages of unknown fields, of their own and in the messages that they hold (of a map value
    too), a map entry that upb keeps among them, and messages of none, an unset child too."""
    plan = cls("spark.connect.Plan")()
    plan.MergeFromString(_UNKNOWN_FIELDS)
    relation = cls("spark.connect.Relation")(sql={"query": "q"})
    relation.sql.MergeFromString(_UNKNOWN_FIELDS)
    relation.sql.named_arguments["a"].MergeFromString(_tag(900, 0) + _varint(1))
    relation.MergeFromString(_tag(999, 0) + _varint(2))
    source = cls("spark.connect.Read.DataSource")()
    source.MergeFromString(_field(3, _field(1, b"k") + _tag(3, 0) + _varint(7)))
    empty = cls("spark.connect.Relation")(range={"end": 1})
    return [plan, relation, relation.sql, source, empty, cls("spark.connect.Relation")().read]


def _unknown_field_sets(cls):
    """The unknown fields of `_unknown_field_cases`, of protobuf's UnknownFieldSet (each its
    number, wire type and data, of a group its fields), and their text of text_format."""
    from google.protobuf import text_format, unknown_fields

    def fields(field_set):
        return [
            (f.field_number, f.wire_type, fields(f.data) if f.wire_type == 3 else repr(f.data))
            for f in field_set
        ]

    return [
        (
            fields(unknown_fields.UnknownFieldSet(message)),
            text_format.MessageToString(message, print_unknown_fields=True),
            text_format.MessageToString(message, print_unknown_fields=True, as_one_line=True),
        )
        for message in _unknown_field_cases(cls)
    ]


def test_unknown_field_sets_of_the_messages_are_upbs(upb):
    # protobuf's unknown_fields.UnknownFieldSet takes the messages of the runtime, which protobuf's
    # class refuses (the drop-in wraps it, as it does protobuf's builder): their sets have the
    # fields, wire types, data and order of upb's, a group as a set of its own, and text_format
    # prints them with print_unknown_fields=True as upb's, in the messages that a message holds
    # too. Of other arguments, the sets and errors are protobuf's, and the sets are of protobuf's
    # class. With protobuf's pure-Python implementation too, whose class is another.
    from google.protobuf import unknown_fields

    assert _unknown_field_sets(ours) == _unknown_field_sets(upb)
    field_set = unknown_fields.UnknownFieldSet
    real, mine = upb("spark.connect.Plan")(), ours("spark.connect.Plan")()
    real.MergeFromString(_UNKNOWN_FIELDS)
    mine.MergeFromString(_UNKNOWN_FIELDS)
    for label, call in [
        ("message=", lambda m: len(field_set(message=m))),
        ("two", lambda m: field_set(m, 1)),
        ("msg=", lambda m: field_set(msg=m)),
        ("none", lambda m: field_set()),
        ("5", lambda m: field_set(5)),
        ("None", lambda m: field_set(None)),
        ("dict", lambda m: field_set({})),
        ("index", lambda m: field_set(m)[-1]),
        ("past the end", lambda m: field_set(m)[6]),
        ("hash", lambda m: hash(field_set(m))),
        ("of the class", lambda m: isinstance(field_set(m), field_set)),
        ("of a group", lambda m: isinstance(field_set(m)[3].data, field_set)),
        ("after a discard", lambda m: (m.DiscardUnknownFields(), len(field_set(m)))[1]),
    ]:
        assert _outcome(lambda: call(mine)) == _outcome(lambda: call(real)), label
    assert type(field_set(ours("spark.connect.Plan")())) is type(field_set(upb("spark.connect.Plan")()))
    # One wrapper wraps protobuf's own class, also when the package installs again.
    from google.protobuf.internal import api_implementation

    _rust._hook_unknown_field_set()
    protobuf_class = api_implementation._c_module.UnknownFieldSet
    assert unknown_fields.UnknownFieldSet._protobuf_class is protobuf_class

    script = textwrap.dedent(
        """
        import json

        from google.protobuf.internal import api_implementation
        from pyspark.sql.connect.proto import _rust
        from tests.test_proto_rust_offline import _unknown_field_sets

        print(json.dumps([api_implementation.Type(), _unknown_field_sets(_rust._CLASSES.__getitem__)]))
        """
    )
    root = pathlib.Path(__file__).resolve().parent.parent
    env = dict(os.environ, PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python")
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, cwd=str(root), env=env
    )
    assert result.returncode == 0, result.stderr[-3000:]
    implementation, found = json.loads(result.stdout)
    assert implementation == "python"
    assert found == json.loads(json.dumps(_unknown_field_sets(upb)))


def _protobuf_cases(cls):
    """Messages of the maps, Anys of a type of the protos and Timestamps that protobuf's
    json_format and text_format print and parse."""
    relation = cls("spark.connect.Relation")()
    relation.sql.query = "select :a, :b"
    relation.sql.named_arguments["a"].literal.integer = 1
    relation.sql.named_arguments["b"].unresolved_attribute.unparsed_identifier = "x"
    relation.sql.args["c"].string = "y"
    extension = cls("spark.connect.Relation")()
    extension.extension.Pack(cls("spark.connect.Range")(start=1, end=3))
    context = cls("spark.connect.UserContext")(user_id="u")
    context.extensions.add().Pack(cls("spark.connect.SQL")(query="q"))
    event = cls("spark.connect.PipelineEvent")(message="m")
    event.timestamp.FromJsonString("2020-01-02T03:04:05.678Z")
    return [relation, extension, context, event]


def test_json_and_text_formats_of_protobuf_take_the_messages(upb):
    # protobuf's json_format and text_format print and parse the messages as upb's: of maps,
    # whose map entries' options they read, of an Any of a type of the protos, which they find in
    # protobuf's default pool, and of a Timestamp, as its RFC 3339 text.
    from google.protobuf import json_format, text_format

    for real, mine in zip(_protobuf_cases(upb), _protobuf_cases(ours)):
        # json_format writes the entries of a map in the order that the map iterates them, the
        # order of insertion of the runtime's and that of upb's hash table (see
        # docs/compatibility.md), so the objects compare, not their texts.
        assert json.loads(json_format.MessageToJson(mine)) == json.loads(
            json_format.MessageToJson(real)
        )
        assert json_format.MessageToDict(mine) == json_format.MessageToDict(real)
        assert text_format.MessageToString(mine) == text_format.MessageToString(real)
        one_line = text_format.MessageToString(real, as_one_line=True)
        assert text_format.MessageToString(mine, as_one_line=True) == one_line
        bytes_of = real.SerializeToString(deterministic=True)
        parsed = json_format.Parse(json_format.MessageToJson(real), type(mine)())
        assert parsed.SerializeToString() == bytes_of
        parsed = json_format.ParseDict(json_format.MessageToDict(real), type(mine)())
        assert parsed.SerializeToString() == bytes_of
        parsed = text_format.Parse(text_format.MessageToString(real), type(mine)())
        assert parsed.SerializeToString() == bytes_of


@pytest.fixture(scope="module")
def plugin(upb):
    """The generated modules of the plugin of the tests (``plugin_protos/plugin.proto``, which
    imports the protos of Spark Connect, and ``plugin_protos/base.proto``, which does not, as
    Delta Lake's ``delta/connect/relations.proto`` and ``base.proto``), and the class of upb of a
    message of their files and of the protos by full name, of a pool of their own, as the classes
    of the official client are upb's of the modules."""
    try:
        from google.protobuf import runtime_version
    except ImportError:
        runtime_version = None
    if runtime_version is None or (runtime_version.MAJOR, runtime_version.MINOR) < (6, 33):
        pytest.skip("the modules of plugin_protos, of protoc 33.0, need protobuf 6.33 or later")
    from plugin_protos import base_pb2, plugin_pb2

    classes = _upb_classes_of(
        [(m.DESCRIPTOR.serialized_pb, m.__name__) for m in (base_pb2, plugin_pb2)]
    )
    return plugin_pb2, base_pb2, classes.__getitem__


def _plugin_cases(cls, table):
    """Messages of the plugin of the tests, of the classes of ``cls`` by full name, as Delta
    Lake's plan code builds its relations: a relation and an expression of the protos copied and
    merged into their fields, given to their constructors and to their containers and maps, a
    message of ``table``, the class of ``thirdparty.Table`` (of base.proto), copied and merged into
    a field of its type, and an ``Any`` of the protos that packs one."""
    relation = cls("spark.connect.Relation")(sql={"query": "select 1"})
    relation.common.plan_id = 7
    condition = cls("spark.connect.Expression")(expression_string={"expression": "id > 1"})
    delete = cls("thirdparty.DeleteFromTable")()
    delete.target.CopyFrom(relation)
    delete.condition.CopyFrom(condition)
    built = cls("thirdparty.DeleteFromTable")(target=relation, condition=condition)
    merged = cls("thirdparty.DeleteFromTable")()
    merged.target.MergeFrom(relation)
    merged.MergeFrom(built)
    merge = cls("thirdparty.MergeIntoTable")(
        source=relation, mode="MODE_UPSERT", defaults={"d": condition}
    )
    merge.target.CopyFrom(relation)
    merge.condition.CopyFrom(condition)
    assignment = cls("thirdparty.Assignment")(field=condition, value=condition)
    action = merge.matched_actions.add(condition=condition)
    action.update_action.assignments.extend([assignment])
    action.update_action.assignments.append(assignment)
    merge.matched_actions.add().delete_action.SetInParent()
    merge.matched_actions.append(
        cls("thirdparty.MergeIntoTable.Action")(update_action={"assignments": [assignment]})
    )
    merge.defaults["e"].CopyFrom(condition)
    merge.with_schema_evolution = False
    scan = cls("thirdparty.Scan")(table=table(path="/p", options={"k": "v"}))
    scanned = cls("thirdparty.Scan")()
    scanned.table.CopyFrom(table(name="t"))
    scanned.table.MergeFrom(table(options={"k": "v"}))
    wrapper = cls("thirdparty.PluginRelation")(merge_into_table=merge)
    wrapper.scan.CopyFrom(scan)
    extension = cls("spark.connect.Relation")()
    extension.extension.Pack(delete)
    return [delete, built, merged, merge, scan, scanned, wrapper, extension]


def test_generated_modules_of_plugins_have_the_classes_of_the_runtime(plugin):
    # The module that protoc generates of a plugin's file that imports the protos of Spark Connect
    # has the classes of the runtime, of its module and qualified names, as protobuf's builder
    # names them, and the runtime's wrappers of its enums; the module of its file that does not
    # import them, as protobuf's own modules, keeps protobuf's classes.
    from google.protobuf import message as protobuf_message
    from google.protobuf import message_factory

    plugin_pb2, base_pb2, _ = plugin
    delete = plugin_pb2.DeleteFromTable
    assert issubclass(delete, _pyspark.ProtoMessage)
    assert delete is ours("thirdparty.DeleteFromTable")
    assert delete.__module__ == "plugin_protos.plugin_pb2"
    assert delete.__qualname__ == "DeleteFromTable"
    update = plugin_pb2.MergeIntoTable.Action.UpdateAction
    assert update.__qualname__ == "MergeIntoTable.Action.UpdateAction"
    assert update.DESCRIPTOR.full_name == "thirdparty.MergeIntoTable.Action.UpdateAction"
    assert isinstance(plugin_pb2.Mode, _rust.EnumTypeWrapper)
    assert (plugin_pb2.Mode.Value("MODE_UPSERT"), plugin_pb2.MODE_INSERT) == (2, 1)
    assert plugin_pb2.Mode.DESCRIPTOR is plugin_pb2.MergeIntoTable.DESCRIPTOR.fields_by_name[
        "mode"
    ].enum_type
    assert not issubclass(base_pb2.Table, _pyspark.ProtoMessage)
    assert issubclass(base_pb2.Table, protobuf_message.Message)

    # The descriptors of the module stay protobuf's, which the generated code of protobuf's
    # pure-Python implementation changes; those of the classes are the runtime's, of its pool.
    assert plugin_pb2._DELETEFROMTABLE.full_name == "thirdparty.DeleteFromTable"
    assert not isinstance(plugin_pb2._DELETEFROMTABLE, _pyspark.ProtoDescriptor)
    assert not isinstance(plugin_pb2.DESCRIPTOR, _pyspark.ProtoFileDescriptor)
    file = delete.DESCRIPTOR.file
    assert isinstance(file, _pyspark.ProtoFileDescriptor)
    assert (file.name, file.package) == ("plugin_protos/plugin.proto", "thirdparty")
    assert file.serialized_pb == plugin_pb2.DESCRIPTOR.serialized_pb
    assert [d.name for d in file.dependencies] == [
        "spark/connect/relations.proto",
        "spark/connect/expressions.proto",
        "plugin_protos/base.proto",
    ]
    pool = proto.Relation.DESCRIPTOR.file.pool
    assert pool.FindMessageTypeByName("thirdparty.DeleteFromTable") is delete.DESCRIPTOR
    assert pool.FindFileContainingSymbol("thirdparty.MODE_UPSERT") is file
    # As protobuf's builder made them first, protobuf's own lookups give protobuf's classes.
    made = message_factory.GetMessageClass(plugin_pb2._DELETEFROMTABLE)
    assert issubclass(made, protobuf_message.Message) and made is not delete
    # One hook wraps the builder's own function, however many times the runtime installs it.
    from google.protobuf.internal import builder

    own = builder.BuildTopDescriptorsAndMessages._protobuf_build
    assert own.__module__ == "google.protobuf.internal.builder"
    _rust._hook_protobuf_builder()
    assert builder.BuildTopDescriptorsAndMessages._protobuf_build is own

    # A module of the file built again has the same classes, as upb's has.
    assert importlib.reload(plugin_pb2).DeleteFromTable is delete
    assert plugin_pb2.MergeIntoTable.Action.UpdateAction is update
    # So has a module of a file of the protos, as a copy of Apache Spark's generated modules
    # builds it of the file of the pool: the classes of the package.
    module = _build_generated_module(_protobuf_file("spark/connect/example_plugins.proto"))
    assert module["ExamplePluginRelation"] is ours("spark.connect.ExamplePluginRelation")


def test_installs_register_the_classes_of_the_protos_only_in_protobufs_database(plugin):
    # The runtime registers the classes of its own files in protobuf's symbol database, where
    # PySpark's pickles of nested messages load (Expression.Literal), but not those that it made
    # for a plugin's module: protobuf's lookups of a plugin's file give protobuf's classes, also
    # when the package installs again, of a file whose class the database lacks too.
    from google.protobuf import symbol_database

    plugin_pb2, _, _ = plugin
    database = symbol_database.Default()
    descriptor = plugin_pb2._DELETEFROMTABLE
    kept = database._classes.pop(descriptor)
    try:
        _rust._add_files_to_protobuf()
        assert descriptor not in database._classes
    finally:
        database._classes[descriptor] = kept
    assert database.GetSymbol("thirdparty.DeleteFromTable") is kept
    assert not issubclass(kept, _pyspark.ProtoMessage)
    literal = "spark.connect.Expression.Literal"
    assert database.GetSymbol(literal) is ours(literal)
    assert all(
        descriptor.file.name in _rust._EMBEDDED_FILES
        for descriptor, cls in database._classes.items()
        if issubclass(cls, _pyspark.ProtoMessage)
    )


def test_messages_of_plugins_take_the_messages_of_the_package_as_upbs(plugin):
    # The messages of the module of a plugin take those of pyspark.sql.connect.proto as they are,
    # as upb's take PySpark's of the official client, in CopyFrom, MergeFrom, their constructors,
    # containers and maps (relation.delete_from_table.target.CopyFrom(plan) of Delta Lake's plan
    # code), and a field of a type of a file that keeps protobuf's classes takes protobuf's
    # messages of the type. The messages are upb's, of their bytes, text, JSON, pickles and Any.
    from google.protobuf import json_format, text_format

    plugin_pb2, base_pb2, upb = plugin
    mine = _plugin_cases(ours, base_pb2.Table)
    real = _plugin_cases(upb, upb("thirdparty.Table"))
    for m, r in zip(mine, real):
        assert m.SerializeToString() == r.SerializeToString(deterministic=True)
        assert str(m) == str(r)
        assert text_format.MessageToString(m) == text_format.MessageToString(r)
        assert json_format.MessageToDict(m) == json_format.MessageToDict(r)
        assert type(m).FromString(r.SerializeToString(deterministic=True)) == m
        assert json_format.Parse(json_format.MessageToJson(r), type(m)()) == m
        assert text_format.Parse(text_format.MessageToString(r), type(m)()) == m
        assert pickle.loads(pickle.dumps(m)) == m and copy.deepcopy(m) == m

    delete, built, _, merge, scan, _, _, extension = mine
    assert type(delete.target) is proto.Relation and type(built.condition) is proto.Expression
    assert isinstance(merge.defaults["d"], proto.Expression)
    assert isinstance(merge.matched_actions[0], plugin_pb2.MergeIntoTable.Action)
    unpacked = plugin_pb2.DeleteFromTable()
    assert extension.extension.Unpack(unpacked) and unpacked == delete
    assert extension.extension.TypeName() == "thirdparty.DeleteFromTable"
    # The message of a field of a type of base.proto is of the runtime's class of the type, which
    # takes protobuf's messages of it, not of the class of its module, which is protobuf's.
    assert type(scan.table) is not base_pb2.Table
    assert scan.table.DESCRIPTOR.full_name == "thirdparty.Table"
    table = base_pb2.Table(path="/p", options={"k": "v"})
    assert scan.table.SerializeToString() == table.SerializeToString(deterministic=True)

    # The errors of a message of another type are upb's.
    for make in [
        lambda cls: cls("thirdparty.DeleteFromTable")().target.CopyFrom(
            cls("spark.connect.Expression")()
        ),
        lambda cls: cls("thirdparty.DeleteFromTable")(target=cls("spark.connect.Expression")()),
        lambda cls: cls("thirdparty.DeleteFromTable")(nope=1),
        lambda cls: cls("thirdparty.MergeIntoTable")(mode="MODE_NONE"),
        lambda cls: cls("thirdparty.MergeIntoTable")().matched_actions.append(
            cls("thirdparty.Assignment")()
        ),
    ]:
        assert _outcome(lambda: make(ours)) == _outcome(lambda: make(upb))


def test_generated_modules_of_plugins_build_with_protobufs_pure_python_implementation(plugin):
    # With protobuf's pure-Python implementation, whose generated code sets the serialized ranges
    # and options of the module's descriptors (_DELETEFROMTABLE._serialized_start), which stay
    # protobuf's, the module of the plugin has the classes of the runtime too, of upb's messages,
    # and a message of the runtime equals protobuf's message of the type of the same contents (a
    # message of protobuf's class of base.proto's Table), on either side, and no other one.
    _, base_pb2, upb = plugin
    script = textwrap.dedent(
        """
        import json, sys

        sys.path.insert(0, "tests")
        from google.protobuf.internal import api_implementation
        from pyspark import _pyspark
        from pyspark.sql.connect.proto import _rust
        from plugin_protos import base_pb2, plugin_pb2
        from tests.test_proto_rust_offline import _plugin_cases

        cases = _plugin_cases(_rust._CLASSES.__getitem__, base_pb2.Table)
        table = cases[4].table
        print(json.dumps({
            "implementation": api_implementation.Type(),
            "runtime": issubclass(plugin_pb2.DeleteFromTable, _pyspark.ProtoMessage),
            "protobuf": not issubclass(base_pb2.Table, _pyspark.ProtoMessage),
            "start": plugin_pb2._DELETEFROMTABLE._serialized_start,
            "bytes": [case.SerializeToString().hex() for case in cases],
            "equal": [
                table == base_pb2.Table(path="/p", options={"k": "v"}),
                base_pb2.Table(path="/p", options={"k": "v"}) == table,
                table != base_pb2.Table(path="/p"),
                table == base_pb2.Table,
            ],
        }))
        """
    )
    root = pathlib.Path(__file__).resolve().parent.parent
    env = dict(os.environ, PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python")
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(root),
        env=env,
    )
    assert result.returncode == 0, result.stderr
    out = json.loads(result.stdout)
    assert out["implementation"] == "python" and out["runtime"] and out["protobuf"]
    assert out["start"] > 0
    real = _plugin_cases(upb, upb("thirdparty.Table"))
    assert out["bytes"] == [r.SerializeToString(deterministic=True).hex() for r in real]
    assert out["equal"] == [True, True, True, False]


def _add_plugin_file(name, dependencies, fields, syntax="proto3"):
    """The file ``name`` of protobuf's default pool, as a generated module adds it, of one message
    ``M`` of ``fields``, each its name and its message type, of the package of the file's stem. A
    file of the syntax ``editions`` is of the edition 2023."""
    from google.protobuf import descriptor_pb2, descriptor_pool

    pool = descriptor_pool.Default()
    try:
        return pool.FindFileByName(name)
    except KeyError:
        pass
    package = "tests." + name.rsplit("/", 1)[-1][: -len(".proto")]
    file = descriptor_pb2.FileDescriptorProto(
        name=name, package=package, dependency=dependencies, syntax=syntax
    )
    if syntax == "editions":
        file.edition = descriptor_pb2.EDITION_2023
    message = file.message_type.add(name="M")
    for number, (field, type_name) in enumerate(fields, 1):
        message.field.add(
            name=field,
            number=number,
            type=descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
            type_name=type_name,
        )
    return pool.AddSerializedFile(file.SerializeToString())


def _protobuf_file(name):
    """The file ``name`` of protobuf's default pool."""
    from google.protobuf import descriptor_pool

    return descriptor_pool.Default().FindFileByName(name)


def _build_generated_module(file):
    """The globals of a module of ``file`` that protoc generates, as protobuf's builder builds
    them."""
    from google.protobuf.internal import builder

    module = {}
    builder.BuildMessageAndEnumDescriptors(file, module)
    module_name = file.name[: -len(".proto")].replace("/", ".") + "_pb2"
    builder.BuildTopDescriptorsAndMessages(file, module_name, module)
    return module


def test_generated_modules_of_plugins_build_in_threads_at_once(upb):
    # The modules of plugins that threads build at once, as their imports do, have the classes of
    # the runtime, and the message of a field of a type of a file that they all import, which
    # imports no proto of Spark Connect, is of one class of the runtime of the type.
    shared = _add_plugin_file("tests/threads/shared.proto", [], [])
    files = [
        _add_plugin_file(
            f"tests/threads/plugin{i}.proto",
            ["spark/connect/relations.proto", shared.name],
            [("input", ".spark.connect.Relation"), ("shared", ".tests.shared.M")],
        )
        for i in range(8)
    ]
    barrier = threading.Barrier(len(files))
    modules = [None] * len(files)

    def build(i):
        barrier.wait(timeout=60)
        modules[i] = _build_generated_module(files[i])

    threads = [threading.Thread(target=build, args=(i,)) for i in range(len(files))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    shared_classes = set()
    for module in modules:
        message = module["M"](input=proto.Relation(sql={"query": "q"}))
        assert isinstance(message, _pyspark.ProtoMessage) and type(message.input) is proto.Relation
        shared_classes.add(type(message.shared))
    assert shared_classes == {ours("tests.shared.M")}


def test_files_that_the_runtime_does_not_hold_keep_protobufs_classes(upb):
    # A module of a file keeps protobuf's classes, as without the drop-in, when the file imports
    # no proto of Spark Connect, when it is not proto3 (the runtime reads the fields of proto3,
    # whose presence, defaults and packing are not those of proto2), when it imports a well-known
    # type other than Any and Timestamp, whose protobuf classes have methods that the runtime's
    # classes would lack (Duration, Struct, ...), and when it is of another pool than the default
    # one (as _upb_classes_of builds the classes of upb of the protos).
    from google.protobuf import duration_pb2
    from google.protobuf import message as protobuf_message

    assert duration_pb2.Duration.DESCRIPTOR.full_name == "google.protobuf.Duration"
    relations = "spark/connect/relations.proto"
    cases = [
        _add_plugin_file("tests/keeps/no_spark.proto", [], []),
        _add_plugin_file(
            "tests/keeps/proto2.proto",
            [relations],
            [("input", ".spark.connect.Relation")],
            syntax="proto2",
        ),
        _add_plugin_file(
            "tests/keeps/duration.proto",
            [relations, "google/protobuf/duration.proto"],
            [("input", ".spark.connect.Relation"), ("timeout", ".google.protobuf.Duration")],
        ),
    ]
    for file in cases:
        cls = _build_generated_module(file)["M"]
        assert issubclass(cls, protobuf_message.Message), file.name
        assert not issubclass(cls, _pyspark.ProtoMessage), file.name
        with pytest.raises(KeyError):
            _pyspark._proto_message_descriptor(cls.DESCRIPTOR.full_name)
    assert not issubclass(upb("spark.connect.Relation"), _pyspark.ProtoMessage)


def test_classes_of_a_file_built_before_the_hook_are_of_its_generated_module(upb):
    # A file that a plugin's file imports and whose generated module protobuf's builder built
    # before the hook was installed, as Delta Lake's delta/connect/proto/__init__.py imports
    # base_pb2 (of delta/connect/base.proto) before the modules that import the protos of Spark
    # Connect: the runtime's classes of its messages are of that module, as protobuf's are
    # (delta.connect.proto.base_pb2), not of protoc's name of the file's path.
    import types

    from google.protobuf.internal import builder

    early = _add_plugin_file("tests/before/early.proto", [], [])
    late = _add_plugin_file(
        "tests/before/late.proto",
        ["spark/connect/relations.proto", early.name],
        [("input", ".spark.connect.Relation"), ("early", ".tests.early.M")],
    )
    module = types.ModuleType("tests_before.generated.early_pb2")
    namespace = vars(module)
    namespace["DESCRIPTOR"] = early
    builder.BuildMessageAndEnumDescriptors(early, namespace)
    # As the module of the file, which protobuf's own builder built before the hook.
    builder.BuildTopDescriptorsAndMessages._protobuf_build(early, module.__name__, namespace)
    assert namespace["M"].__module__ == module.__name__
    sys.modules[module.__name__] = module
    try:
        built = _build_generated_module(late)
    finally:
        del sys.modules[module.__name__]
    assert issubclass(built["M"], _pyspark.ProtoMessage)
    assert not issubclass(namespace["M"], _pyspark.ProtoMessage)
    held = type(built["M"]().early)
    assert held is ours("tests.early.M")
    assert (held.__module__, held.__qualname__) == (module.__name__, "M")


def test_a_protobuf_whose_builder_lacks_the_hooked_function_imports_the_package(upb):
    # The hook wraps a function of protobuf's internal builder. Where a protobuf's builder lacks
    # it, as where there is no builder (protobuf before 3.20), the package imports without the
    # hook, of the classes of the runtime.
    script = textwrap.dedent(
        """
        from google.protobuf.internal import builder

        del builder.BuildTopDescriptorsAndMessages
        from pyspark import _pyspark
        import pyspark.sql.connect.proto as proto

        assert issubclass(proto.Relation, _pyspark.ProtoMessage)
        assert proto.Relation(sql={"query": "q"}).sql.query == "q"
        assert not hasattr(builder, "BuildTopDescriptorsAndMessages")
        print("imported")
        """
    )
    root = pathlib.Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(root),
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.split() == ["imported"]


def test_protobufs_own_classes_of_a_plugins_file_take_the_messages_through_bytes(upb):
    # The files of the protos are in protobuf's default pool, as the generated modules add them,
    # so that a file of a plugin that imports them builds, as the module that protoc generates of
    # it adds it (Delta Lake's do). A class that protobuf's message_factory makes of a message of
    # such a file, rather than its generated module, is protobuf's, which takes a message of
    # pyspark.sql.connect.proto through its bytes or a dict (not as it is, which protobuf's
    # messages take of their own classes only), and which an Any of the package packs and unpacks.
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    pool = descriptor_pool.Default()
    for name, _, _, _ in _pyspark._proto_files():
        found = pool.FindFileByName(name)
        if name.startswith("spark/connect/"):
            assert found.serialized_pb == _pyspark._proto_file_descriptor(name).serialized_pb
            # As a generated module of Apache Spark's would add it again: the pool takes it.
            pool.AddSerializedFile(found.serialized_pb)

    file = descriptor_pb2.FileDescriptorProto(
        name="tests/plugin_of_test_proto_rust_offline.proto",
        package="tests.plugin",
        syntax="proto3",
        dependency=["spark/connect/relations.proto"],
    )
    message = file.message_type.add(name="PluginRelation")
    message.field.add(
        name="input",
        number=1,
        type=descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
        type_name=".spark.connect.Relation",
        json_name="input",
    )
    message.field.add(
        name="custom",
        number=2,
        type=descriptor_pb2.FieldDescriptorProto.TYPE_STRING,
        label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
        json_name="custom",
    )
    try:
        pool.FindFileByName(file.name)
    except KeyError:
        pool.AddSerializedFile(file.SerializeToString())
    plugin_class = message_factory.GetMessageClass(
        pool.FindMessageTypeByName("tests.plugin.PluginRelation")
    )

    relation = proto.Relation(sql=proto.SQL(query="select 1"))
    plugin = plugin_class(custom="c")
    plugin.input.ParseFromString(relation.SerializeToString())
    assert plugin == plugin_class(input={"sql": {"query": "select 1"}}, custom="c")
    with pytest.raises(TypeError):
        plugin_class(input=relation)
    with pytest.raises(TypeError):
        plugin.input.CopyFrom(relation)
    wrapped = proto.Relation()
    wrapped.extension.Pack(plugin)
    assert wrapped.extension.TypeName() == "tests.plugin.PluginRelation"
    unpacked = plugin_class()
    assert wrapped.extension.Unpack(unpacked) and unpacked == plugin
    back = proto.Relation()
    back.CopyFrom(unpacked.input)
    assert back == relation and back.SerializeToString() == plugin.input.SerializeToString()


def test_messages_of_a_type_of_protobufs_classes_and_of_the_runtimes_compare_as_upbs(plugin):
    # upb has one class of a type of protobuf's default pool, so its messages of the type compare
    # by their contents. The runtime has a class of the type besides protobuf's, which protobuf's
    # own lookups give (message_factory, symbol_database, the pickles of upb's nested messages) and
    # which the module of a plugin's file that imports no proto of Spark Connect keeps (base.proto,
    # as Delta Lake's): a message of the runtime equals protobuf's message of the type of the same
    # contents, on either side of == and !=, and so in a container and a map, as upb's messages of
    # one class do, and equals no message of another pool (of another class for upb too), class or
    # other object.
    from google.protobuf import descriptor_pool, message_factory

    plugin_pb2, base_pb2, upb = plugin
    pool = descriptor_pool.Default()

    def protobufs(full_name):
        return message_factory.GetMessageClass(pool.FindMessageTypeByName(full_name))

    scan = plugin_pb2.Scan(table=base_pb2.Table(path="/p", options={"k": "v"}))
    table = base_pb2.Table(path="/p", options={"k": "v"})
    other = base_pb2.Table(path="/q")
    assert type(scan.table) is not base_pb2.Table
    assert scan.table == table and table == scan.table
    assert not (scan.table != table) and not (table != scan.table)
    assert scan.table != other and other != scan.table and not (scan.table == other)
    real_scan = upb("thirdparty.Scan")(table={"path": "/p", "options": {"k": "v"}})
    real_table = upb("thirdparty.Table")(path="/p", options={"k": "v"})
    assert real_scan.table == real_table and real_scan.table != upb("thirdparty.Table")(path="/q")

    relation = proto.Relation(sql=proto.SQL(query="select 1"))
    of_protobuf = protobufs("spark.connect.Relation")(sql={"query": "select 1"})
    assert relation == of_protobuf and of_protobuf == relation and not (relation != of_protobuf)
    delete = plugin_pb2.DeleteFromTable(target=relation)
    made = protobufs("thirdparty.DeleteFromTable").FromString(delete.SerializeToString())
    assert delete == made and made == delete

    merge = plugin_pb2.MergeIntoTable(defaults={"d": {"expression_string": {"expression": "1"}}})
    merge.matched_actions.add().update_action.SetInParent()
    merge.matched_actions.add().delete_action.SetInParent()
    action = protobufs("thirdparty.MergeIntoTable.Action")()
    action.delete_action.SetInParent()
    expression = protobufs("spark.connect.Expression")(expression_string={"expression": "1"})
    assert action in merge.matched_actions and merge.matched_actions[1] == action
    assert merge.matched_actions == [merge.matched_actions[0], action]
    assert merge.defaults["d"] == expression and merge.defaults == {"d": expression}
    merge.matched_actions.remove(action)
    assert len(merge.matched_actions) == 1 and action not in merge.matched_actions
    real_merge = upb("thirdparty.MergeIntoTable")(
        defaults={"d": {"expression_string": {"expression": "1"}}},
        matched_actions=[{"update_action": {}}],
    )
    assert merge.SerializeToString() == real_merge.SerializeToString(deterministic=True)

    # A message of the type of another pool, which upb's == of its other class gives
    # NotImplemented of, is not equal, nor is a class, a dict or another object.
    assert table != real_table and not (table == real_table)
    assert scan.table != real_table and real_table != scan.table and not (scan.table == real_table)
    for value in (base_pb2.Table, type(scan.table), {"path": "/p"}, None, "/p"):
        assert not (scan.table == value) and scan.table != value

    # A message of protobuf's of the type of a field named keys (Expression.Literal.Map), which is
    # not a mapping, initializes a field as it does upb's.
    literal_map = protobufs("spark.connect.Expression.Literal.Map")(
        keys=[{"integer": 1}], values=[{"integer": 2}]
    )
    literal = proto.Expression.Literal(map=literal_map)
    real_literal = upb("spark.connect.Expression.Literal")(
        map={"keys": [{"integer": 1}], "values": [{"integer": 2}]}
    )
    assert literal.SerializeToString() == real_literal.SerializeToString()
    assert literal.map == literal_map and literal_map == literal.map


def test_modules_that_keep_protobufs_classes_take_the_runtimes_messages_through_bytes(upb):
    # A module of a plugin's file that keeps protobuf's classes (a proto3 file that imports a
    # well-known type other than Any and Timestamp, a proto2 file or one of editions) and imports
    # a file whose module has the runtime's classes (a proto3 file that imports the protos of
    # Spark Connect) takes the messages of that module only through their bytes or dicts, as it
    # takes those of pyspark.sql.connect.proto: its fields of their type are protobuf's, where
    # upb's take the messages of its one class of the type as they are. Its messages and those of
    # the runtime of the same contents compare equal, as upb's of one class do.
    from google.protobuf import descriptor_pb2, duration_pb2
    from google.protobuf import message as protobuf_message

    held = _add_plugin_file(
        "tests/importers/runtime_held.proto",
        ["spark/connect/relations.proto"],
        [("input", ".spark.connect.Relation")],
    )
    dependency = [("held", ".tests.runtime_held.M")]
    kept = [
        _add_plugin_file(
            "tests/importers/with_duration.proto",
            [held.name, duration_pb2.DESCRIPTOR.name],
            dependency + [("timeout", ".google.protobuf.Duration")],
        ),
        _add_plugin_file("tests/importers/of_proto2.proto", [held.name], dependency, "proto2"),
    ]
    if hasattr(descriptor_pb2, "EDITION_2023"):
        kept.append(
            _add_plugin_file(
                "tests/importers/of_editions.proto", [held.name], dependency, "editions"
            )
        )
    runtime_class = _build_generated_module(held)["M"]
    assert issubclass(runtime_class, _pyspark.ProtoMessage)
    real = _upb_classes_of(
        [(duration_pb2.DESCRIPTOR.serialized_pb, duration_pb2.__name__)]
        + [(file.serialized_pb, file.name[: -len(".proto")].replace("/", ".") + "_pb2")
           for file in [held] + kept]
    )
    message = runtime_class(input={"sql": {"query": "select 1"}})
    for file in kept:
        kept_class = _build_generated_module(file)["M"]
        assert issubclass(kept_class, protobuf_message.Message), file.name
        assert not issubclass(kept_class, _pyspark.ProtoMessage), file.name
        with pytest.raises(TypeError):
            kept_class(held=message)
        with pytest.raises(TypeError):
            kept_class().held.CopyFrom(message)
        with pytest.raises(TypeError):
            kept_class().held.MergeFrom(message)
        through_bytes = kept_class()
        through_bytes.held.ParseFromString(message.SerializeToString())
        through_dict = kept_class(held={"input": {"sql": {"query": "select 1"}}})
        assert through_bytes == through_dict
        assert through_bytes.held == message and message == through_dict.held
        # upb's take the messages of the class of the type of the file that they import as they are.
        real_kept = real[kept_class.DESCRIPTOR.full_name]
        real_message = real["tests.runtime_held.M"](input={"sql": {"query": "select 1"}})
        assert real_kept(held=real_message).SerializeToString() == through_dict.SerializeToString()


def test_modules_of_protoc_before_3_20_keep_protobufs_classes(upb):
    # A module of protoc before 3.20 (plugin_protos/old_style_pb2.py, of protoc 3.19's form) makes
    # its classes with protobuf's reflection.GeneratedProtocolMessageType, not with protobuf's
    # builder, which the drop-in wraps, so its classes stay protobuf's, of a file that imports the
    # protos of Spark Connect too: they take the messages of pyspark.sql.connect.proto only through
    # their bytes or dicts, where upb's take PySpark's as they are, and their messages and those of
    # the runtime of the same contents compare equal.
    from google.protobuf import message as protobuf_message
    from plugin_protos import old_style_pb2

    old_style = old_style_pb2.OldStyleRelation
    assert issubclass(old_style, protobuf_message.Message)
    assert not issubclass(old_style, _pyspark.ProtoMessage)
    with pytest.raises(KeyError):
        _pyspark._proto_message_descriptor("thirdparty.old.OldStyleRelation")
    relation = proto.Relation(sql=proto.SQL(query="select 1"))
    with pytest.raises(TypeError):
        old_style(input=relation)
    with pytest.raises(TypeError):
        old_style().input.CopyFrom(relation)
    through_bytes = old_style(name="n")
    through_bytes.input.ParseFromString(relation.SerializeToString())
    through_dict = old_style(input={"sql": {"query": "select 1"}}, name="n")
    assert through_bytes == through_dict and through_dict.input == relation
    # As upb's of a pool of their own, as the official client's of its upb classes of the protos.
    real = _upb_classes_of([(old_style_pb2.DESCRIPTOR.serialized_pb, old_style_pb2.__name__)])
    real_relation = real["spark.connect.Relation"](sql={"query": "select 1"})
    real_old_style = real["thirdparty.old.OldStyleRelation"](input=real_relation, name="n")
    assert real_old_style.SerializeToString() == through_dict.SerializeToString()


def test_example_plugins_are_messages_of_the_runtime(upb):
    # pyspark.sql.connect.proto.example_plugins_pb2, of the messages of Spark's example plugin, is
    # of the runtime: its messages take those of the package as they are, and are upb's.
    from pyspark.sql.connect.proto import example_plugins_pb2 as plugins

    assert sys.modules["pyspark.sql.connect.proto.example_plugins_pb2"] is plugins
    relation = proto.Relation(sql=proto.SQL(query="select 1"))
    plugin = plugins.ExamplePluginRelation(input=relation, custom_field="c")
    assert plugin.input == relation and isinstance(plugin, _pyspark.ProtoMessage)
    child = proto.Expression(literal=proto.Expression.Literal(integer=1))
    expression = plugins.ExamplePluginExpression(child=child, custom_field="e")
    wrapped = proto.Relation()
    wrapped.extension.Pack(plugin)
    assert wrapped.extension.TypeName() == "spark.connect.ExamplePluginRelation"
    # As PySpark's package, which does not import the module: its messages are not exported.
    assert not hasattr(proto, "ExamplePluginRelation")
    real_plugin = upb("spark.connect.ExamplePluginRelation")(
        input={"sql": {"query": "select 1"}}, custom_field="c"
    )
    real_expression = upb("spark.connect.ExamplePluginExpression")(
        child={"literal": {"integer": 1}}, custom_field="e"
    )
    assert plugin.SerializeToString() == real_plugin.SerializeToString()
    assert expression.SerializeToString() == real_expression.SerializeToString()
    assert str(plugin) == str(real_plugin)
