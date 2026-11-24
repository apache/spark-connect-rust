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

import collections
import collections.abc as abc
import copy
import datetime
import gc
import importlib
import io
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
import types
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


def _own(full_name):
    """The runtime's own descriptor of the message type ``full_name``, which its class has where
    protobuf does not import: where it imports, the class has protobuf's (see
    ``_protobuf_descriptor``), as upb's has."""
    return _pyspark._proto_message_descriptor(full_name)


def _protobuf_descriptor(full_name):
    """protobuf's descriptor of the message type ``full_name`` of its default pool."""
    from google.protobuf import descriptor_pool

    return descriptor_pool.Default().FindMessageTypeByName(full_name)


_OWN_CLASSES = {}


def _own_class(full_name):
    """The class of the message type ``full_name`` as the runtime makes it where protobuf does not
    import: of the runtime's own descriptor, of its base ``_rust._MessageBase`` and, of ``Any``
    and ``Timestamp``, of its own methods of them, rather than protobuf's (``WKTBASES``)."""
    cls = _OWN_CLASSES.get(full_name)
    if cls is None:
        methods = {"google.protobuf.Any": _rust._AnyMethods, "google.protobuf.Timestamp": _rust._TimestampMethods}
        bases = (_pyspark.ProtoMessage, _rust._MessageBase) + (
            (methods[full_name],) if full_name in methods else ()
        )
        descriptor = _own(full_name)
        cls = type(descriptor.name, bases, {"__slots__": (), "DESCRIPTOR": descriptor})
        _OWN_CLASSES[full_name] = cls
    return cls


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


def test_parse_errors_are_protobufs_decode_errors_as_upbs(upb):
    # As upb's: a parse that fails raises protobuf's google.protobuf.message.DecodeError itself,
    # of upb's message and arguments, which pickles and copies as itself. With protobuf's
    # pure-Python implementation too, whose DecodeError is the same class. (Without protobuf, the
    # runtime's own DecodeError: see test_messages_work_without_protobuf.)
    from google.protobuf import message

    literal, relation = "spark.connect.Expression.Literal", "spark.connect.Relation"
    cases = [
        (literal, b"\xff"),
        (literal, _tag(1, 7)),
        (literal, _tag(1, 4)),
        (literal, _tag(0, 0) + _varint(1)),
        (literal, _tag(2, 2) + _varint(5) + b"ab"),
        (literal, _field(ours(literal).STRING_FIELD_NUMBER, b"\xff")),
        (relation, _field(ours(relation).SQL_FIELD_NUMBER, b"\xff")),
    ]
    parses = {
        "FromString": lambda cls, data: cls.FromString(data),
        "ParseFromString": lambda cls, data: cls().ParseFromString(data),
        "MergeFromString": lambda cls, data: cls().MergeFromString(data),
        "__setstate__": lambda cls, data: cls().__setstate__({"serialized": data}),
    }

    def raised(cls, parse, data):
        try:
            parse(cls, data)
        except message.DecodeError as error:
            return error
        raise AssertionError(data)

    for full_name, data in cases:
        for name, parse in parses.items():
            mine, real = raised(ours(full_name), parse, data), raised(upb(full_name), parse, data)
            assert type(real) is message.DecodeError and type(mine) is message.DecodeError
            assert (str(mine), mine.args, repr(mine)) == (str(real), real.args, repr(real)), (
                full_name,
                data.hex(),
                name,
            )
    assert _rust.DecodeError is message.DecodeError
    error = raised(ours("spark.connect.Plan"), parses["FromString"], b"\xff")
    copies = [pickle.loads(pickle.dumps(error, p)) for p in range(pickle.HIGHEST_PROTOCOL + 1)]
    for copied in copies + [copy.copy(error), copy.deepcopy(error)]:
        assert type(copied) is message.DecodeError and copied.args == error.args

    script = textwrap.dedent(
        """
        import json
        import pickle

        from google.protobuf import message
        from google.protobuf.internal import api_implementation
        from pyspark.sql.connect.proto import _rust

        try:
            _rust._CLASSES["spark.connect.Plan"].FromString(b"\\xff")
        except message.DecodeError as error:
            loaded = pickle.loads(pickle.dumps(error))
            out = [api_implementation.Type(), str(error), type(error) is message.DecodeError]
            out.append(type(loaded) is message.DecodeError and _rust.DecodeError is message.DecodeError)
        print(json.dumps(out))
        """
    )
    root = pathlib.Path(__file__).resolve().parent.parent
    env = dict(os.environ, PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python")
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, cwd=str(root), env=env
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert json.loads(result.stdout) == ["python", str(error), True, True]


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
        # The objects of the fields read at a comparison (a message, a container, its first
        # message), of either runtime, held so that each later comparison tells whether the field
        # gives the same object still (see `compare_identities`).
        self.kept = []

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
        self.compare_identities(a, b, where)

    def compare_identities(self, a, b, where):
        """Whether each field of a message (a message field, a repeated message field and its first
        message) gives the object that it gave at an earlier comparison, of either runtime alike;
        the objects that it gives now are held for the later ones."""

        def objects(m, f):
            value = getattr(m, f.name)
            if not f.is_repeated:
                return [value]
            return [value] + ([value[0]] if len(value) else [None])

        for f in a.DESCRIPTOR.fields:
            if f.type != f.TYPE_MESSAGE or f.message_type.GetOptions().map_entry:
                continue
            now_a, now_b = objects(a, f), objects(b, f)
            for ka, kb, name, then_a, then_b in self.kept:
                if ka is a and kb is b and name == f.name:
                    same = [x is y for x, y in zip(then_a, now_a)], [x is y for x, y in zip(then_b, now_b)]
                    assert same[0] == same[1], f"the objects of {name} (upb's, ours: {same}) after {where}"
            if not any(ka is a and name == f.name for ka, _, name, _, _ in self.kept):
                self.kept.append((a, b, f.name, now_a, now_b))


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


def _protobuf_class(full_name):
    """protobuf's own class of the type ``full_name`` of its default pool, of protobuf's metaclass
    (upb's ``MessageMeta``) that the runtime's hook wraps, as a module of the type built before
    the drop-in installs has: a class of the type besides the runtime's, whose maps are
    protobuf's. (protobuf's lookups give the runtime's class.)"""
    from google.protobuf import descriptor_pool, reflection
    from google.protobuf import message as protobuf_message

    metaclass = reflection.GeneratedProtocolMessageType._protobuf_class
    descriptor = descriptor_pool.Default().FindMessageTypeByName(full_name)
    return metaclass(
        descriptor.name, (protobuf_message.Message,), {"DESCRIPTOR": descriptor, "__module__": None}
    )


class _KeysOnly:
    """An object of ``keys`` and ``__getitem__`` that is not a mapping, which ``update`` takes by
    its keys."""

    def __init__(self, items):
        self.items = dict(items)

    def keys(self):
        return list(self.items)

    def __getitem__(self, key):
        return self.items[key]


class _MissingValues(abc.Mapping):
    """A mapping of ``keys`` whose values cannot be read."""

    def __init__(self, keys):
        self.keys_ = list(keys)

    def __iter__(self):
        return iter(self.keys_)

    def __len__(self):
        return len(self.keys_)

    def __getitem__(self, key):
        raise KeyError(key)


def test_merge_from_of_a_map_takes_its_argument_as_upbs(upb):
    # upb's MergeFrom of a map refuses a dict. Of a map of scalars, it takes any other argument as
    # `update` (MutableMapping's) does: the items of a mapping (a map of either runtime, protobuf's
    # map of a message of protobuf's class of the type, MappingProxyType, UserDict), of an object
    # with `keys`, or the pairs of an iterable, each key and value converted, with its errors. Of a
    # map of messages, it reads the keys of the iteration of the argument, converts each, reads
    # its value, then copies it (CopyFrom) into the message of the key, made if the map has none,
    # or sets the message's fields of a dict. The runtime took the iteration of any argument but
    # its own maps as pairs, so a map of 2-character keys merged {"ab": "zz"} as {"a": "b"}, and
    # took the entries of its own maps of other types as they were, so a map held keys and values
    # of other types (a message in a map of strings, an Expression in a map of Literals).
    import types

    def scalar_cases(new, protobufs):
        source = new("spark.connect.Read.DataSource")
        return {
            "the same map": source(options={"ab": "zz"}).options,
            "a map of strings of another field": new("spark.connect.WithColumnsRenamed")(
                rename_columns_map={"ab": "zz", "cd": "yy"}
            ).rename_columns_map,
            "protobuf's map of the type": protobufs("spark.connect.Read.DataSource")(
                options={"ab": "zz", "cd": "yy"}
            ).options,
            "a map of messages": new("spark.connect.SQL")(args={"ab": {"integer": 1}}).args,
            "MappingProxyType": types.MappingProxyType({"ab": "zz", "cd": "yy"}),
            "MappingProxyType of a bad value": types.MappingProxyType({"ab": "zz", "cd": 5}),
            "UserDict": collections.UserDict({"ab": "zz"}),
            "an object with keys": _KeysOnly({"ab": "zz"}),
            "pairs": [("ab", "zz"), ("cd", "yy")],
            "pairs of a bad value": [("ab", "zz"), ("cd", 5)],
            "triples": [("ab", "zz", "q")],
            "a generator of pairs": (pair for pair in [("ab", "zz")]),
            "a str": "ab",
            "a keys view": {"ab": "zz"}.keys(),
            "an int": 5,
            "None": None,
            "an empty list": [],
            "a dict": {"ab": "zz"},
            "an OrderedDict": collections.OrderedDict(ab="zz"),
            # upb calls `update` of a tuple as of its items as arguments.
            "a tuple of a pair": (("ab", "zz"),),
            "a tuple of a mapping": (types.MappingProxyType({"ab": "zz"}),),
            "an empty tuple": (),
            "a tuple of two mappings": ({"ab": "zz"}, {"cd": "yy"}),
        }

    def message_cases(new, protobufs):
        sql = new("spark.connect.SQL")
        literal = new("spark.connect.Expression.Literal")
        return {
            "the same map": sql(args={"k": {"integer": 1}, "x": {"string": "s"}}).args,
            "protobuf's map of the type": protobufs("spark.connect.SQL")(
                args={"k": {"integer": 1}}
            ).args,
            "a map of messages of another type": sql(
                named_arguments={"k": {"literal": {"integer": 1}}}
            ).named_arguments,
            "a map of strings": new("spark.connect.Read.DataSource")(options={"k": "v"}).options,
            "MappingProxyType of messages": types.MappingProxyType({"k": literal(integer=1)}),
            "MappingProxyType of dicts": types.MappingProxyType({"k": {"integer": 1}}),
            "a dict onto an entry": types.MappingProxyType({"x": {"integer": 1}}),
            "UserDict of messages": collections.UserDict({"k": literal(integer=1)}),
            "a value of None": types.MappingProxyType({"k": None}),
            "a dict of no such field": types.MappingProxyType({"k": {"nope": 1}}),
            "values that cannot be read": _MissingValues(["k"]),
            "a list of keys": ["k"],
            "pairs": [("k", literal(integer=1))],
            "a str": "ab",
            "an int": 5,
            "None": None,
            "an empty list": [],
            "a dict": {"k": literal(integer=1)},
            "a tuple of a mapping": (types.MappingProxyType({"k": literal(integer=1)}),),
        }

    def run(new, protobufs, kind, label):
        if kind == "scalar":
            target = new("spark.connect.Read.DataSource")(options={"x": "1"})
            container = target.options
            argument = scalar_cases(new, protobufs)[label]
        else:
            target = new("spark.connect.SQL")(args={"x": {"long": 9}})
            target.named_arguments["e"].common.origin.python_origin.fragment = "f"
            container = target.args
            argument = message_cases(new, protobufs)[label]
        outcome = _outcome(lambda: container.MergeFrom(argument))
        return outcome, target

    for kind, labels in (
        ("scalar", scalar_cases(ours, _protobuf_class)),
        ("message", message_cases(ours, _protobuf_class)),
    ):
        for label in labels:
            # upb's class of the type is protobuf's class of the type for upb.
            real_outcome, real = run(upb, upb, kind, label)
            mine_outcome, mine = run(ours, _protobuf_class, kind, label)
            assert mine_outcome == real_outcome, (kind, label)
            assert mine.SerializeToString() == real.SerializeToString(deterministic=True), (
                kind,
                label,
            )
            assert str(mine) == str(real), (kind, label)

    # A dict sets the fields of the message of its key as a constructor's mapping does, keeping
    # the fields that it does not set, where a message replaces it.
    mine = ours("spark.connect.SQL")()
    mine.named_arguments["e"].common.origin.python_origin.fragment = "f"
    mine.named_arguments.MergeFrom(types.MappingProxyType({"e": {"literal": {"integer": 1}}}))
    assert mine.named_arguments["e"].common.origin.python_origin.fragment == "f"
    assert mine.named_arguments["e"].literal.integer == 1

    # The message of a key that the map has stays the message of the key, of its new fields, as
    # upb's CopyFrom of the value into it keeps it, where a MergeFrom of the message replaces it.
    def kept(new, argument_of, merge_message=False):
        sql = new("spark.connect.SQL")(args={"k": {"integer": 1}})
        held = sql.args["k"]
        if merge_message:
            sql.MergeFrom(new("spark.connect.SQL")(args={"k": {"string": "s"}}))
        else:
            sql.args.MergeFrom(argument_of(new, sql))
        return held is sql.args["k"], str(held), str(sql.args["k"])

    def of_the_type(new, sql):
        return new("spark.connect.SQL")(args={"k": {"string": "s"}}).args

    def of_a_message(new, sql):
        return types.MappingProxyType({"k": new("spark.connect.Expression.Literal")(string="s")})

    def of_a_dict(new, sql):
        return types.MappingProxyType({"k": {"string": "s"}})

    def itself(new, sql):
        return sql.args

    for label, argument_of, merge_message in [
        ("a map of the type", of_the_type, False),
        ("a mapping of a message", of_a_message, False),
        ("a mapping of a dict", of_a_dict, False),
        ("the map itself", itself, False),
        ("MergeFrom of the message", None, True),
    ]:
        assert kept(ours, argument_of, merge_message) == kept(upb, argument_of, merge_message), label


def test_update_of_a_map_takes_its_arguments_as_upbs(upb):
    # upb's update of a map is MutableMapping's, of at most one positional argument, which it
    # iterates when it is neither a mapping nor of `keys`: None is the error of its iteration.
    # The runtime took None for no argument, and refused a second one in PyO3's words.
    cases = {
        "None": lambda m: m.update(None),
        "nothing": lambda m: m.update(),
        "two arguments": lambda m: m.update({}, {}),
        "a keyword other": lambda m: m.update(other="v"),
        "a mapping and keywords": lambda m: m.update({"a": "b"}, c="d"),
        "pairs": lambda m: m.update([("a", "b")]),
        "an int": lambda m: m.update(5),
    }
    for label, update in cases.items():
        outcomes = []
        for new in (upb, ours):
            message = new("spark.connect.Read.DataSource")(options={"x": "1"})
            outcomes.append((_outcome(lambda: update(message.options)), sorted(message.options.items())))
        assert outcomes[0] == outcomes[1], label


_MAP_METHOD_CALLS = {
    "MergeFrom()": lambda m: m.MergeFrom(),
    "MergeFrom(a, b)": lambda m: m.MergeFrom([], []),
    "MergeFrom(other=a)": lambda m: m.MergeFrom(other=[]),
    "MergeFrom(a, x=1)": lambda m: m.MergeFrom([], x=1),
    "clear(1)": lambda m: m.clear(1),
    "clear(x=1)": lambda m: m.clear(x=1),
    "clear(1, x=2)": lambda m: m.clear(1, x=2),
    "GetEntryClass(1)": lambda m: m.GetEntryClass(1),
    "GetEntryClass(x=1)": lambda m: m.GetEntryClass(x=1),
    "get()": lambda m: m.get(),
    "get(k, d, x)": lambda m: m.get("k", 1, 2),
    "get(key=k)": lambda m: m.get(key="k"),
    "get(zz, default=1)": lambda m: m.get("zz", default=1),
    "get(x=1)": lambda m: m.get(x=1),
    "get(default=1)": lambda m: m.get(default=1),
    "get(k, x=1)": lambda m: m.get("k", x=1),
    "get(k, key=z)": lambda m: m.get("k", key="z"),
    "get(k, 1, default=2)": lambda m: m.get("k", 1, default=2),
    "get(key=k, default=1, x=2)": lambda m: m.get(key="k", default=1, x=2),
    "get(k, 2, 3, x=4)": lambda m: m.get("k", 2, 3, x=4),
    "pop()": lambda m: m.pop(),
    "pop(k, d, x)": lambda m: m.pop("k", 1, 2),
    "pop(key=k)": lambda m: m.pop(key="k"),
    "pop(k, key=z)": lambda m: m.pop("k", key="z"),
    "pop(x=1)": lambda m: m.pop(x=1),
    "popitem(1)": lambda m: m.popitem(1),
    "popitem(x=1)": lambda m: m.popitem(x=1),
    "popitem() of an empty map": lambda m: (m.clear(), m.popitem()),
    "setdefault()": lambda m: m.setdefault(),
    "setdefault(k, d, x)": lambda m: m.setdefault("k", "d", 2),
    "setdefault(key=k)": lambda m: m.setdefault(key="k"),
    "setdefault(q, default=d)": lambda m: m.setdefault("q", default="d"),
    "setdefault(q, d)": lambda m: m.setdefault("q", "d"),
    "keys(1)": lambda m: m.keys(1),
    "keys(x=1)": lambda m: m.keys(x=1),
    "values(1)": lambda m: m.values(1),
    "items(x=1)": lambda m: m.items(x=1),
    "sorted keys(), values(), items()": lambda m: (
        sorted(m.keys()),
        len(m.values()),
        [k for k, _ in m.items()],
    ),
    "update(a, b)": lambda m: m.update({}, {}),
    "get_or_create()": lambda m: m.get_or_create(),
    "get_or_create(a, b)": lambda m: m.get_or_create("a", "b"),
    "get_or_create(key=k)": lambda m: m.get_or_create(key="k"),
}


_REPEATED_METHOD_CALLS = {
    "append()": lambda c, v: c.append(),
    "append(a, b)": lambda c, v: c.append(v, v),
    "append(value=v)": lambda c, v: c.append(value=v),
    "append(v, x=1)": lambda c, v: c.append(v, x=1),
    "extend()": lambda c, v: c.extend(),
    "extend(a, b)": lambda c, v: c.extend([], []),
    "extend(elem_seq=[])": lambda c, v: c.extend(elem_seq=[]),
    "insert()": lambda c, v: c.insert(),
    "insert(0)": lambda c, v: c.insert(0),
    "insert(0, v, 1)": lambda c, v: c.insert(0, v, 1),
    "insert(key=0, value=v)": lambda c, v: c.insert(key=0, value=v),
    "insert(a, v)": lambda c, v: c.insert("a", v),
    "insert(2**70, v)": lambda c, v: c.insert(2**70, v),
    "insert(-1, v)": lambda c, v: c.insert(-1, v),
    "pop(0, 1)": lambda c, v: c.pop(0, 1),
    "pop(key=0)": lambda c, v: c.pop(key=0),
    "pop(a)": lambda c, v: c.pop("a"),
    "pop(2**70)": lambda c, v: c.pop(2**70),
    "pop()": lambda c, v: c.pop(),
    "pop(0)": lambda c, v: c.pop(0),
    "remove()": lambda c, v: c.remove(),
    "remove(a, b)": lambda c, v: c.remove(v, v),
    "remove(elem=v)": lambda c, v: c.remove(elem=v),
    "MergeFrom()": lambda c, v: c.MergeFrom(),
    "MergeFrom(a, b)": lambda c, v: c.MergeFrom([], []),
    "MergeFrom(other=[])": lambda c, v: c.MergeFrom(other=[]),
    "MergeFrom([v])": lambda c, v: c.MergeFrom([v]),
    "reverse(1)": lambda c, v: c.reverse(1),
    "clear(1)": lambda c, v: c.clear(1),
    "sort(1)": lambda c, v: c.sort(1),
    "__deepcopy__(1, 2)": lambda c, v: c.__deepcopy__(1, 2),
    "__deepcopy__(memo={})": lambda c, v: c.__deepcopy__(memo={}),
    "copy.deepcopy": lambda c, v: copy.deepcopy(c),
}


def test_methods_of_containers_raise_upbs_errors_of_their_arguments(upb):
    # upb's maps are of its C types ScalarMapContainer and MessageMapContainer, whose own methods
    # (MergeFrom, clear, GetEntryClass, setdefault, get_or_create) take positional arguments only,
    # with CPython's errors of their C functions naming the class, and whose get parses its
    # arguments with PyArg_ParseTupleAndKeywords; its pop, popitem, update, keys, values and items
    # are those of collections.abc's MutableMapping and Mapping, in Python. upb's repeated
    # containers take positional arguments only too, append, extend, remove and MergeFrom one,
    # insert and pop of PyArg_ParseTuple's "nO" and "|n", with their errors. The runtime's methods
    # had PyO3's arguments and errors: they took keyword arguments (MergeFrom(other=...),
    # append(value=...), setdefault(key, default=...)), and the errors of the maps named
    # MapContainer.
    for kind, of in (("scalar", "Read.DataSource"), ("message", "SQL")):
        for label, call in _MAP_METHOD_CALLS.items():
            if kind == "scalar" and label.startswith("get_or_create"):
                continue
            outcomes = []
            for new in (upb, ours):
                if kind == "scalar":
                    message = new(f"spark.connect.{of}")(options={"k": "v"})
                    container = message.options
                else:
                    message = new(f"spark.connect.{of}")(args={"k": {"integer": 1}})
                    container = message.args
                outcome = _outcome(lambda: call(container))
                outcomes.append((outcome, message.SerializeToString(deterministic=True)))
            assert outcomes[0] == outcomes[1], (kind, label)
    for kind in ("scalar", "composite"):
        for label, call in _REPEATED_METHOD_CALLS.items():
            outcomes = []
            for new in (upb, ours):
                if kind == "scalar":
                    message = new("spark.connect.StatApproxQuantile")(probabilities=[1.0, 2.0])
                    container, value = message.probabilities, 3.0
                else:
                    message = new("spark.connect.Expression.Literal.Array")(
                        elements=[{"integer": 1}, {"integer": 2}]
                    )
                    container = message.elements
                    value = new("spark.connect.Expression.Literal")(integer=3)
                outcome = _outcome(lambda: call(container, value))
                outcomes.append((outcome, message.SerializeToString()))
            assert outcomes[0] == outcomes[1], (kind, label)
    # A keyword argument of upb's methods of positional arguments (METH_VARARGS) is CPython's
    # error of the method of its class when it is called of its object, and of the method only
    # when CPython calls it as a bound method, as it calls each before Python 3.12 while a trace
    # or a profile function is set (as coverage sets one): the runtime's errors follow.
    calls = {
        "setdefault": lambda new: new("spark.connect.Read.DataSource")().options.setdefault(
            key="k"
        ),
        "insert": lambda new: new("spark.connect.StatApproxQuantile")().probabilities.insert(
            key=0, value=1.0
        ),
        "pop": lambda new: new("spark.connect.Expression.Literal.Array")().elements.pop(key=0),
        "__deepcopy__": lambda new: new(
            "spark.connect.StatApproxQuantile"
        )().probabilities.__deepcopy__(memo={}),
    }
    previous = sys.getprofile()
    for profile in (None, lambda *args: None):
        sys.setprofile(profile)
        try:
            for label, call in calls.items():
                outcomes = [_outcome(lambda new=new: call(new)) for new in (upb, ours)]
                assert outcomes[0] == outcomes[1], (label, profile)
        finally:
            sys.setprofile(previous)


def _iterate_while(container, operation):
    """What a step of an iteration of `container` gives after `operation` ran between its first
    two steps: the outcome of the operation, then the keys of the rest of the iteration, or the
    error of its step."""
    iterator = iter(container)
    next(iterator)
    done = _outcome(operation)
    return done, _outcome(lambda: list(iterator))


def test_maps_changed_during_their_iteration_fail_it_as_upbs(upb):
    # upb's map takes a new version at the start of each method that may change its keys, even
    # one that then fails or changes nothing, but a set of a key that the map has: the next step of
    # an iteration of the map in progress, of its keys, values or items, then raises
    # `RuntimeError: Map modified during iteration.`. The runtime iterated a copy of the keys, so
    # `items()` and `values()` read the value of a key that the loop had removed, which inserted
    # it again, of the default value, in the message that it then serialized.
    def scalar(message):
        return message.options

    def messages(message):
        return message.args

    operations = {
        "set of a key it has": (scalar, lambda m: m.options.__setitem__("k0", "x")),
        "set of a new key": (scalar, lambda m: m.options.__setitem__("new", "x")),
        "set of a bad value": (scalar, lambda m: m.options.__setitem__("k0", 5)),
        "set of a bad key": (scalar, lambda m: m.options.__setitem__(5, "x")),
        "del": (scalar, lambda m: m.options.__delitem__("k1")),
        "del of a missing key": (scalar, lambda m: m.options.__delitem__("zz")),
        "del of a bad key": (scalar, lambda m: m.options.__delitem__(5)),
        "read of a key it has": (scalar, lambda m: m.options["k1"]),
        "read of a missing key": (scalar, lambda m: m.options["zz"]),
        "read of a bad key": (scalar, lambda m: m.options[5]),
        "get": (scalar, lambda m: (m.options.get("zz"), m.options.get("k1"))),
        "in": (scalar, lambda m: ("zz" in m.options, "k1" in m.options)),
        "len, ==, str": (scalar, lambda m: (len(m.options), m.options == {}, str(m))),
        "clear": (scalar, lambda m: m.options.clear()),
        "pop": (scalar, lambda m: m.options.pop("k1")),
        "pop of a missing key": (scalar, lambda m: m.options.pop("zz", None)),
        "pop of a bad key": (scalar, lambda m: m.options.pop(5, None)),
        "popitem": (scalar, lambda m: m.options.popitem()),
        "setdefault of a key it has": (scalar, lambda m: m.options.setdefault("k1", "q")),
        "setdefault of a new key": (scalar, lambda m: m.options.setdefault("zz", "q")),
        "setdefault of no value": (scalar, lambda m: m.options.setdefault("k1")),
        "update of keys it has": (scalar, lambda m: m.options.update({"k1": "q"})),
        "update of a new key": (scalar, lambda m: m.options.update({"zz": "q"})),
        "update of a bad value": (scalar, lambda m: m.options.update({"k1": 5})),
        "update of nothing": (scalar, lambda m: m.options.update({})),
        "MergeFrom of keys it has": (scalar, lambda m: m.options.MergeFrom([("k1", "q")])),
        "MergeFrom of nothing": (scalar, lambda m: m.options.MergeFrom([])),
        "MergeFrom of a dict": (scalar, lambda m: m.options.MergeFrom({})),
        "ClearField of the map": (scalar, lambda m: m.ClearField("options")),
        "ClearField of another field": (scalar, lambda m: m.ClearField("format")),
        "Clear of the message": (scalar, lambda m: m.Clear()),
        "MergeFrom of the message": (
            scalar,
            lambda m: m.MergeFrom(type(m)(options={"n": "v"})),
        ),
        "a parse into the message": (
            scalar,
            lambda m: m.MergeFromString(type(m)(options={"n": "v"}).SerializeToString()),
        ),
        "read of a message it has": (messages, lambda m: m.args["k1"]),
        "read of a missing message": (messages, lambda m: m.args["zz"]),
        "get_or_create of a key it has": (messages, lambda m: m.args.get_or_create("k1")),
        "get_or_create of a new key": (messages, lambda m: m.args.get_or_create("zz")),
        "get_or_create of a bad key": (messages, lambda m: m.args.get_or_create(5)),
        "a change of a message": (messages, lambda m: setattr(m.args["k1"], "integer", 5)),
        "set of a message": (
            messages,
            lambda m: m.args.__setitem__("k1", type(m.args["k1"])()),
        ),
        "del of a message": (messages, lambda m: m.args.__delitem__("k1")),
        "MergeFrom of a map": (messages, lambda m: m.args.MergeFrom(type(m)(args={"k1": {}}).args)),
        "ClearField of a map of messages": (messages, lambda m: m.ClearField("args")),
    }

    def build(new, container):
        if container is scalar:
            return new("spark.connect.Read.DataSource")(
                options={"k0": "v0", "k1": "v1", "k2": "v2"}
            )
        return new("spark.connect.SQL")(
            args={"k0": {"integer": 0}, "k1": {"integer": 1}, "k2": {"integer": 2}}
        )

    # An iteration of a map whose message adds a key goes on: upb's reads the new key or not, of
    # the order of its hash table, where the runtime's reads it, as the last.
    adding = {"MergeFrom of the message", "a parse into the message"}
    for label, (container, operation) in operations.items():
        results = []
        for new in (upb, ours):
            message = build(new, container)
            done, rest = _iterate_while(container(message), lambda: operation(message))
            # The keys of upb's iteration are of the order of its hash table: their number.
            if rest[0] == "ok":
                rest = ("ok",) if label in adding else ("ok", len(rest[1]))
            results.append((done[0], rest))
        assert results[0] == results[1], label

    # items() and values() of a loop that removes a key that it has not read yet stop at the next
    # step, and the message keeps no entry of the key.
    for view in ("items", "values"):
        for new in (upb, ours):
            message = build(new, scalar)
            iterator = iter(getattr(message.options, view)())
            next(iterator)
            del message.options["k1"]
            del message.options["k2"]
            with pytest.raises(RuntimeError, match="^Map modified during iteration[.]$"):
                next(iterator)
            assert sorted(message.options) == ["k0"], (view, new)
    # A change before the first step fails it, and so does one after the last.
    for new in (upb, ours):
        options = build(new, scalar).options
        iterator = iter(options)
        options["n"] = "x"
        with pytest.raises(RuntimeError, match="^Map modified during iteration[.]$"):
            next(iterator)
        iterator = iter(options)
        assert len(list(iterator)) == 4 and iter(iterator) is iterator
        options["m"] = "x"
        with pytest.raises(RuntimeError, match="^Map modified during iteration[.]$"):
            next(iterator)
    # An iteration with no change reads the keys of the map in their order, of a map of messages
    # too, and values() and items() give their values.
    options = build(ours, scalar).options
    assert list(options) == ["k0", "k1", "k2"] and list(options.values()) == ["v0", "v1", "v2"]
    arguments = build(ours, messages).args
    assert [(k, v.integer) for k, v in arguments.items()] == [("k0", 0), ("k1", 1), ("k2", 2)]


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
    "map setdefault bad key": lambda c: c.data_source.options.setdefault(1, "v"),
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


class _KeysAndItems:
    """An object of `keys()` and of its items, which is no mapping and has no `__iter__`:
    `MutableMapping.update` takes its items of its keys, and its iteration, of `__getitem__`,
    reads the item of 0 first."""

    def __init__(self, items):
        self.items = dict(items)

    def keys(self):
        return list(self.items)

    def __getitem__(self, key):
        return self.items[key]


class _IterableOfKeys:
    """An object that iterates its keys and gives its items, which is no mapping and has no
    `keys()`: `MutableMapping.update` takes it for pairs."""

    def __init__(self, items):
        self.items = dict(items)

    def __iter__(self):
        return iter(self.items)

    def __getitem__(self, key):
        return self.items[key]


class _KeyTwice:
    """An object that iterates the key "a" once for each of `values`, whose items are `values`, in
    turn."""

    def __init__(self, values):
        self.values = list(values)
        self.read = 0

    def __iter__(self):
        return iter(["a"] * len(self.values))

    def __getitem__(self, key):
        self.read += 1
        return self.values[self.read - 1]


class _ItemsOf(dict):
    """A dict whose item of each key, as `[key]` reads it, is `item`, not the one it holds."""

    def __init__(self, data, item):
        super().__init__(data)
        self.item = item

    def __getitem__(self, key):
        return self.item


class _Counted:
    """A pair that iterates `n` keys, counting those taken of it."""

    def __init__(self, n):
        self.n = n
        self.taken = 0

    def __iter__(self):
        while self.taken < self.n:
            self.taken += 1
            yield f"k{self.taken}"


def _map_field_cases():
    """The values of map fields of a constructor, each made of the classes of `cls`: of a map of
    scalars (Read.DataSource.options) and of maps of messages (MlParams.params, of Literals, and
    SQL.named_arguments, of Expressions)."""

    def literal(cls, **fields):
        return cls("spark.connect.Expression.Literal")(**fields)

    def expression(cls, **fields):
        return cls("spark.connect.Expression")(**fields)

    options = ("spark.connect.Read.DataSource", "options")
    params = ("spark.connect.MlParams", "params")
    named = ("spark.connect.SQL", "named_arguments")
    origin = {"common": {"origin": {"python_origin": {"fragment": "f"}}}}
    return [
        # A map of scalars takes its value of the map's update, MutableMapping.update: the items
        # of a mapping (read with [key]) or of an object of keys(), else pairs, as their
        # iteration and unpacking give them, with Python's errors of another shape.
        (options, lambda cls: {"a": "b", "c": "d"}),
        (options, lambda cls: [("a", "b"), ("c", "d")]),
        (options, lambda cls: iter([("a", "b")])),
        (options, lambda cls: [["a", "b"]]),
        (options, lambda cls: ["ab"]),
        (options, lambda cls: {("a", "b")}),
        (options, lambda cls: zip("ac", "bd")),
        (options, lambda cls: [{"a": 1, "b": 2}]),
        (options, lambda cls: [(b"a", b"b")]),
        (options, lambda cls: [("a", "b"), ("a", "c")]),
        (options, lambda cls: []),
        (options, lambda cls: ""),
        (options, lambda cls: types.MappingProxyType({"a": "b"})),
        (options, lambda cls: _KeysAndItems({"a": "b"})),
        (options, lambda cls: _ItemsOf({"a": "b"}, "c")),
        (options, lambda cls: cls("spark.connect.Read.DataSource")(options={"a": "b"}).options),
        (options, lambda cls: "ab"),
        (options, lambda cls: ["abc"]),
        (options, lambda cls: [("a",)]),
        (options, lambda cls: [("a", "b", "c")]),
        (options, lambda cls: [1]),
        (options, lambda cls: [None]),
        (options, lambda cls: [("a", "b"), 1]),
        (options, lambda cls: 1),
        (options, lambda cls: _IterableOfKeys({"a": "b"})),
        (options, lambda cls: [("a", 1)]),
        (options, lambda cls: [(1, "b")]),
        (options, lambda cls: {"a": None}),
        (options, lambda cls: [("a", b"\xff")]),
        (options, lambda cls: [("\ud800", "b")]),
        (options, lambda cls: cls("spark.connect.MlParams")(params={"a": {}}).params),
        # upb gives a tuple to the update as its arguments (PyObject_CallMethod of "O"): the
        # update of a mapping, of the pairs of one pair (which are not pairs), or of too many.
        (options, lambda cls: ()),
        (options, lambda cls: ({"a": "b"},)),
        (options, lambda cls: ([("a", "b")],)),
        (options, lambda cls: (("a", "b"),)),
        (options, lambda cls: (("a", "b"), ("c", "d"))),
        (options, lambda cls: (None,)),
        # A map of messages takes the keys that its value iterates, of their items ([key]): of a
        # mapping, but not of pairs, which the map refuses for keys after their items fail.
        (params, lambda cls: {"a": literal(cls, integer=1), "b": literal(cls, string="x")}),
        (params, lambda cls: {"a": {"integer": 1}, "b": {}}),
        (params, lambda cls: types.MappingProxyType({"a": literal(cls, integer=1)})),
        (params, lambda cls: _IterableOfKeys({"a": {"integer": 2}})),
        (params, lambda cls: _ItemsOf({"a": {"integer": 1}}, literal(cls, string="item"))),
        (params, lambda cls: cls("spark.connect.MlParams")(params={"a": {"integer": 3}}).params),
        (params, lambda cls: {b"a": {"integer": 1}}),
        (params, lambda cls: []),
        (params, lambda cls: ""),
        (params, lambda cls: [("a", literal(cls, integer=1))]),
        (params, lambda cls: [("a", {"integer": 1})]),
        (params, lambda cls: iter([("a", literal(cls, integer=1))])),
        (params, lambda cls: {"a": literal(cls)}.items()),
        (params, lambda cls: (("a", literal(cls)),)),
        (params, lambda cls: "ab"),
        (params, lambda cls: ["a"]),
        (params, lambda cls: 1),
        (params, lambda cls: _KeysAndItems({"a": literal(cls)})),
        (params, lambda cls: {1: literal(cls)}),
        (params, lambda cls: {"\ud800": literal(cls)}),
        # A dict is the fields of the message of its key, and anything else is copied into it
        # with CopyFrom, of its errors.
        (params, lambda cls: {"a": 1}),
        (params, lambda cls: {"a": None}),
        (params, lambda cls: {"a": "x"}),
        (params, lambda cls: {"a": literal(cls).SerializeToString()}),
        (params, lambda cls: {"a": cls("spark.connect.Relation")()}),
        (params, lambda cls: {"a": types.MappingProxyType({"integer": 1})}),
        (params, lambda cls: {"a": _ItemsOf({"integer": 1}, 2)}),
        (params, lambda cls: {"a": {"nope": 1}}),
        (params, lambda cls: {"a": {"integer": "x"}}),
        (params, lambda cls: {"a": {"integer": None}}),
        (params, lambda cls: {"a": literal(cls, integer=1), "b": 5}),
        (params, lambda cls: cls("spark.connect.Read.DataSource")(options={"a": "b"}).options),
        # A key given twice: a dict sets its fields in the message of the key, and a message is
        # copied over it.
        (named, lambda cls: _KeyTwice([expression(cls, **origin), {"literal": {"integer": 1}}])),
        (named, lambda cls: _KeyTwice([origin, expression(cls, literal={"integer": 1})])),
        (named, lambda cls: _KeyTwice([origin, {"literal": {"integer": 1}}])),
        (named, lambda cls: _KeyTwice([expression(cls, **origin), expression(cls, literal={})])),
    ]


def test_map_fields_of_constructors_take_mappings_and_pairs_as_upbs(upb):
    # As upb's InitMapAttributes: a map of scalars takes its value as the map's update does, of
    # pairs too, and a map of messages takes the items of the keys that its value iterates, a
    # dict as the fields of the message of its key and anything else through CopyFrom. A nested
    # message's map takes its value so too.
    for (full_name, field), make in _map_field_cases():
        real = make(upb)
        label = (full_name, field, repr(real))
        expected = _outcome(lambda: upb(full_name)(**{field: real}))
        assert _outcome(lambda: ours(full_name)(**{field: make(ours)})) == expected, label
        outer = {"spark.connect.Read.DataSource": "data_source", "spark.connect.SQL": "sql"}
        if full_name in outer:
            container = "spark.connect.Read" if field == "options" else "spark.connect.Relation"
            nested = {outer[full_name]: {field: make(upb)}}
            expected = _outcome(lambda: upb(container)(**nested))
            nested = {outer[full_name]: {field: make(ours)}}
            assert _outcome(lambda: ours(container)(**nested)) == expected, label
    # The unpacking of a pair takes three of its items at most, as Python's does.
    for cls in (ours, upb):
        pair = _Counted(1000)
        outcome = _outcome(lambda: cls("spark.connect.Read.DataSource")(options=[pair]))
        assert (outcome, pair.taken) == (
            ("raise", "ValueError", "too many values to unpack (expected 2)"),
            3,
        )
    # A message value is copied, of its contents then.
    for cls in (ours, upb):
        literal = cls("spark.connect.Expression.Literal")(integer=1)
        params = cls("spark.connect.MlParams")(params={"a": literal})
        literal.integer = 2
        assert params.params["a"] is not literal and params.params["a"].integer == 1


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
    # DiscardUnknownFields of a message of thousands of levels (here of the drop-in's own parse of
    # any depth, as FromString refuses past 100 levels, as upb's does) and the messages of mappings
    # nested as deep finish on a thread of a small stack, as upb's (which builds a message of
    # 1,000 levels of mappings on it, of the same bytes), where they recursed over the levels on
    # the stack and overflowed it from 1,500 and 200 levels. A stack overflow ends the process, so
    # they run in one of their own.
    script = textwrap.dedent(
        """
        import threading

        import pyspark.sql.connect.proto as proto
        from pyspark import _pyspark
        from pyspark.sql.connect.proto._rust import DecodeError
        from tests.test_proto_rust_offline import (
            _nested_arguments,
            _nested_limits,
            _relation_bytes,
        )

        LEVELS = 5000
        results = []


        def run():
            data = _relation_bytes(LEVELS, lambda level: True)
            try:
                proto.Relation.FromString(data)
            except DecodeError:
                pass
            relation = _pyspark._proto_parse_deep(proto.Relation, data)
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
        # The data ends with a corrupt tag, so that the parse fails once it has read its messages.
        # The parse reads them natively, of no object, so it is in progress while its call holds
        # the data, its argument. A map of SQL.args has a Literal of each entry.
        if operation == "map parse":
            value = _field(2, literal_class(string="w").SerializeToString())
            data = b"".join(_field(2, _field(1, b"k%d" % i) + value) for i in range(_LARGE))
            parsed = ours("spark.connect.SQL")
        else:
            parsed = literal_class
        corrupt = data + b"\xff"
        in_call = _References(corrupt, lambda extra: extra >= 1)
        return lambda: parsed.FromString(corrupt), _rust.DecodeError, in_call
    if operation == "map of scalars constructor":
        # The constructor takes the pairs of a map of scalars through MutableMapping.update, in
        # Python, as upb's does, which holds them while it runs.
        pairs = [(f"k{i}", "v") for i in range(_LARGE)] + [("bad", 5)]
        in_progress = _References(pairs, lambda extra: extra >= 1)
        return lambda: ours("spark.connect.Read.DataSource")(options=pairs), TypeError, in_progress
    literal = literal_class.FromString(data)
    # The constructor, the extend and the update of many values (here of a value that fails last,
    # so that the messages that they made are released before they return, running no Python
    # code) take one budget, not one of each message.
    elements = list(literal.array.elements) + [5]
    if operation == "constructor of a list":
        # The constructor copies the messages natively, making no object of them: it is in
        # progress while it holds the list, of its iteration, beyond the keyword argument.
        in_list = _References(elements, lambda extra: extra >= 2)
        return lambda: literal_class.Array(elements=elements), TypeError, in_list
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
        "map of scalars constructor",
        "map update",
    ],
)
@pytest.mark.parametrize("cancel", [False, True])
def test_python_thread_runs_before_operations_on_large_messages_finish(operation, cancel):
    # The parse, the constructor, the copies, the merges, the extend of a repeated field of
    # messages and the update of a map of many messages, values or entries, which take longer than
    # upb's, let another thread take the GIL every 1,024 of them, and a signal interrupt them, so
    # that the GIL is held as long as upb's at most, for a thousand messages; upb holds it until
    # the operation ends (but in the update of a map and in the constructor of a map of scalars of
    # pairs, of MutableMapping's Python code, which the runtime's constructor runs too). The other
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
        real, mine = upb(full_name).DESCRIPTOR, _own(full_name)
        assert cls.DESCRIPTOR is _protobuf_descriptor(full_name)
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
    # finds their descriptors and a Timestamp converts. A parse that fails raises the runtime's
    # DecodeError, which pickles. With upb installed, the bytes and the text are upb's.
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
        from tests.test_proto_rust_offline import _relation_bytes, _text_cases

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
        except _pyspark.ProtoDecodeError as error:
            decode_error = str(error)
            decode_error_class = type(error) is _rust.DecodeError
            decode_error_pickle = type(pickle.loads(pickle.dumps(error))) is _rust.DecodeError
        deep = _pyspark._proto_parse_deep(proto.Relation, _relation_bytes(60, lambda level: False))
        try:
            proto.Relation.FromString(deep.SerializeToString())
        except _rust.DecodeError as error:
            too_deep = str(error)
        try:
            type("Sub", (proto.Relation,), {})
        except TypeError as error:
            subclassing = str(error)
        out = {
            "metaclass": type(proto.Relation).__name__,
            "bases": [c.__qualname__ for c in proto.Relation.__mro__[1:]],
            "subclassing": subclassing,
            "too deep": too_deep,
            "maps": [type(proto.Read.DataSource().options).__name__,
                     type(proto.SQL().named_arguments).__name__],
            "copy": copy.copy(nested) == nested and type(copy.copy(nested)) is type(nested),
            "dir": "string" in dir(nested) and "Extensions" not in dir(nested),
            "any pickle": b"google_dot_protobuf_dot_any__pb2" in pickle.dumps(context.extensions[0]),
            "any pickle 2": [
                b"_new_message" in pickle.dumps(context.extensions[0], 2),
                type(pickle.loads(pickle.dumps(context.extensions[0], 2))) is _rust.Any,
                repr(_rust.Any),
            ],
            "class pickles": [
                pickle.loads(pickle.dumps(cls, protocol)) is cls
                for cls in (_rust.Any, _rust.Timestamp, proto.Expression.Literal)
                for protocol in range(pickle.HIGHEST_PROTOCOL + 1)
            ],
            "bytes": [relation.SerializeToString().hex(), context.SerializeToString().hex()],
            "str": [str(relation), str(context)],
            "pickle": pickle.loads(pickle.dumps(relation)) == relation,
            "pickle nested": pickle.loads(pickle.dumps(nested)) == nested,
            "pickle well-known": [
                type(pickle.loads(pickle.dumps(m))) is type(m) and pickle.loads(pickle.dumps(m)) == m
                for m in (context.extensions[0], event.timestamp)
            ],
            "deepcopy": copy.deepcopy(context) == context,
            "numpy": np.array(values).tolist(),
            "sequence": isinstance(values, collections.abc.MutableSequence),
            "mapping": isinstance(relation.sql.named_arguments, collections.abc.MutableMapping),
            "enum": proto.MlOperator(type="OPERATOR_TYPE_MODEL").type,
            "decode error": decode_error,
            "decode error class": decode_error_class and decode_error_pickle,
            "decode error bases": [f"{c.__module__}.{c.__qualname__}" for c in _rust.DecodeError.__mro__],
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
    assert out["pickle well-known"] == [True, True] and out["any pickle"]
    # Of a protocol before 4, of no dotted name, of the runtime's function of the type; of 4 and
    # later, of the name that base_pb2 binds the class to, which is the class's.
    assert out["any pickle 2"] == [
        True,
        True,
        "<class 'pyspark.sql.connect.proto.base_pb2.google_dot_protobuf_dot_any__pb2.Any'>",
    ]
    # The classes pickle at every protocol: of one before 4, the class of the dotted name pickles
    # as the attribute of the runtime's module of its file, which pickles by its name.
    assert out["class pickles"] == [True] * 3 * (pickle.HIGHEST_PROTOCOL + 1)
    for name in ("any_pb2", "timestamp_pb2"):
        module = sys.modules[f"pyspark.sql.connect.proto._rust.{name}"]
        assert pickle.loads(pickle.dumps(module, 2)) is module
    # The classes are of the shape of upb's, of the runtime's base of protobuf's methods of
    # Message, and of upb's errors of a subclass and of a message too deep to parse.
    assert out["metaclass"] == "MessageMeta"
    assert out["bases"] == ["Message", "_MessageBase", "object"]
    assert out["subclassing"] == (
        "A Message class can only inherit from Message, not "
        "(<class 'pyspark.sql.connect.proto.relations_pb2.Relation'>,)"
    )
    assert out["too deep"] == (
        "Error parsing message with type 'spark.connect.Relation': Exceeded upb_DecodeOptions_MaxDepth"
    )
    assert out["maps"] == ["ScalarMapContainer", "MessageMapContainer"]
    assert out["copy"] and out["dir"]
    assert out["numpy"] == [1.5, 2.5] and out["sequence"] and out["mapping"]
    assert out["enum"] == 4
    assert out["decode error"] == (
        "Error parsing message with type 'spark.connect.Plan': Wire format was corrupt"
    )
    # Without protobuf, the runtime's DecodeError is of the native ProtoDecodeError only.
    assert out["decode error class"]
    assert out["decode error bases"] == [
        "pyspark.sql.connect.proto._rust.DecodeError",
        "pyspark.sql.connect.proto.DecodeError",
        "builtins.Exception",
        "builtins.BaseException",
        "builtins.object",
    ]
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
    # Of protobuf's EnumTypeWrapper where protobuf imports, as upb's, and of the runtime's own of
    # its descriptors where it does not.
    own = [
        _rust.EnumTypeWrapper(_pyspark._proto_enum_descriptor(name))
        for name in ("spark.connect.MlOperator.OperatorType", "spark.connect.Join.JoinType")
    ]
    for real, mine in [
        (real_operator.OperatorType, mine_operator.OperatorType),
        (upb("spark.connect.Join").JoinType, ours("spark.connect.Join").JoinType),
        (real_operator.OperatorType, own[0]),
        (upb("spark.connect.Join").JoinType, own[1]),
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
        real, mine = upb(full_name).DESCRIPTOR, _own(full_name)
        assert cls.DESCRIPTOR is _protobuf_descriptor(full_name)
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
    entry = _own("spark.connect.SQL").nested_types_by_name["NamedArgumentsEntry"]
    assert entry.GetOptions().map_entry and entry.has_options
    deprecated = _own("spark.connect.SQL").fields_by_name["args"]
    assert deprecated.GetOptions().deprecated and deprecated.has_options

    for name in sorted(files):
        own_pool = _own("spark.connect.Relation").file.pool
        real, mine = pool.FindFileByName(name), own_pool.FindFileByName(name)
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
                assert method.input_type is _own(method.input_type.full_name)
                assert options(method) == options(real_method)
                assert copied(method, descriptor_pb2.MethodDescriptorProto) == copied(
                    real_method, descriptor_pb2.MethodDescriptorProto
                )
            assert _outcome(lambda: service.FindMethodByName("nope")) == _outcome(
                lambda: real_service.FindMethodByName("nope")
            )

    # The errors of CopyToProto of a message of another type and of an object that is no message.
    real, mine = upb("spark.connect.Relation").DESCRIPTOR, _own("spark.connect.Relation")
    wrong = descriptor_pb2.FileDescriptorProto
    assert _outcome(lambda: mine.CopyToProto(wrong())) == _outcome(lambda: real.CopyToProto(wrong()))
    outcome, expected = _outcome(lambda: mine.CopyToProto(1)), _outcome(lambda: real.CopyToProto(1))
    assert outcome == expected


def test_pool_of_the_files_finds_their_descriptors_as_upbs(upb):
    # The pool of a file of the runtime finds the descriptors of the runtime by name, as upb's
    # default pool finds its own, and fails on the names that it does not have as upb's does.
    from google.protobuf import descriptor_pool

    real_pool, pool = descriptor_pool.Default(), _own("spark.connect.Relation").file.pool
    assert _own("spark.connect.Plan").file.pool is pool

    def file_of(pool, symbol):
        return pool.FindFileContainingSymbol(symbol).name

    def check_enum(enum, scope):
        assert pool.FindEnumTypeByName(enum.full_name) is enum
        for value in enum.values:
            symbol = f"{scope}.{value.name}"
            assert file_of(pool, symbol) == file_of(real_pool, symbol), symbol

    files = {}
    for full_name in _rust._CLASSES:
        descriptor = _own(full_name)
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
        ("FindExtensionByNumber", (_own("spark.connect.Relation"), 5), (real_relation, 5)),
        ("FindAllExtensions", (_own("spark.connect.Relation"),), (real_relation,)),
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


@pytest.mark.parametrize("bases", ["protobuf", "own"])
def test_well_known_types_have_the_methods_of_protobufs(upb, bases):
    # The classes of google.protobuf.Any and google.protobuf.Timestamp have the methods of
    # protobuf's, of their results and errors, and the other classes none of those: protobuf's own
    # (WKTBASES) where protobuf imports, and the runtime's of the same results and errors (of
    # protobuf 7) where it does not, as _own_class makes them.
    from google.protobuf import duration_pb2

    def ours(full_name):  # noqa: F811 - the classes of the runtime of `bases`
        if bases == "own" and full_name.startswith("google.protobuf."):
            return _own_class(full_name)
        return _rust._CLASSES[full_name]

    methods = {"Pack", "Unpack", "Is", "TypeName", "ToJsonString", "FromDatetime", "ToDatetime"}
    for full_name in _rust._CLASSES:
        if _own(full_name).file.name not in _rust._EMBEDDED_FILES:
            continue
        cls, real = ours(full_name), upb(full_name)
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


def test_pickles_of_the_well_known_types_are_upbs_where_protobuf_imports():
    # Where protobuf imports, a message of the runtime's class of a well-known type, Any or
    # Timestamp, pickles as upb's does, of protobuf's class of the type
    # (google.protobuf.any_pb2.Any), so that it loads wherever protobuf imports, as a message of
    # protobuf's class, which the fields of the runtime take as they are. Without protobuf, it
    # pickles of the runtime's class (see test_messages_work_without_protobuf), which loads of the
    # runtime's class in another process of the drop-in, with protobuf too. copy.copy keeps the
    # runtime's class. The runtime's modules of the types (pyspark.sql.connect.proto._rust.any_pb2
    # and timestamp_pb2) are in sys.modules, so that the pickles of earlier drop-ins load.
    from pyspark import cloudpickle

    packed = _rust.Any()
    packed.Pack(proto.Relation(range=proto.Range(end=3)))
    stamp = _rust.Timestamp(seconds=7, nanos=8)
    assert sys.modules["pyspark.sql.connect.proto._rust.any_pb2"].Any is _rust.Any
    assert sys.modules["pyspark.sql.connect.proto._rust.timestamp_pb2"].Timestamp is _rust.Timestamp
    try:
        from google.protobuf import any_pb2, timestamp_pb2
    except ImportError:  # the drop-in runs without protobuf
        any_pb2 = timestamp_pb2 = None
    expected = {}
    if any_pb2 is not None:
        expected = {"google.protobuf.Any": any_pb2.Any, "google.protobuf.Timestamp": timestamp_pb2.Timestamp}
    for message in (packed, stamp, proto.Relation(extension=packed).extension):
        assert type(copy.copy(message)) is type(message) and copy.copy(message) == message
        full_name = message.DESCRIPTOR.full_name
        for dumps in (pickle.dumps, cloudpickle.dumps):
            loaded = pickle.loads(dumps(message))
            assert type(loaded) is expected.get(full_name, type(message))
            assert loaded == message and message == loaded
        if any_pb2 is not None:
            theirs = expected[full_name].FromString(message.SerializeToString())
            for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
                assert pickle.dumps(message, protocol) == pickle.dumps(theirs, protocol)
    assert pickle.loads(cloudpickle.dumps(lambda: packed))() == packed
    if any_pb2 is not None:
        assert proto.Relation(extension=pickle.loads(pickle.dumps(packed))).extension == packed
    script = textwrap.dedent(
        """
        import json, pickle, sys

        packed, stamp = pickle.loads(sys.stdin.buffer.read())
        from pyspark.sql.connect import proto
        from pyspark.sql.connect.proto import _rust

        relation = proto.Relation()
        runtime = [type(packed) is _rust.Any, type(stamp) is _rust.Timestamp]
        out = [type(packed).__module__, packed.Unpack(relation), relation.range.end]
        out += [type(stamp).__module__, stamp.ToJsonString(), runtime]
        print(json.dumps(out))
        """
    )
    root = str(pathlib.Path(__file__).resolve().parent.parent)
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=pickle.dumps([packed, stamp]),
        capture_output=True,
        timeout=120,
        cwd=root,
    )
    assert result.returncode == 0, result.stderr.decode()[-3000:]
    # (Of the runtime's classes in another process of the drop-in, of protobuf, of their modules
    # there. Without protobuf, a class is named by the module of the first proto that imports its
    # file, as that module's name of the file's module, so that the pickle loads in PySpark as
    # protobuf's class: Any is base_pb2's google_dot_protobuf_dot_any__pb2.Any.)
    modules = ["google.protobuf.any_pb2", "google.protobuf.timestamp_pb2", [False, False]]
    if any_pb2 is None:
        modules = [
            "pyspark.sql.connect.proto.base_pb2",
            "pyspark.sql.connect.proto.pipelines_pb2",
            [True, True],
        ]
    assert json.loads(result.stdout) == [
        modules[0], True, 3, modules[1], "1970-01-01T00:00:07.000000008Z", modules[2]
    ]


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
    # names them, which protobuf's lookups of the types give, as upb gives its one class of a
    # type. So does the module of a file that it imports that does not import them (base.proto,
    # as Delta Lake's delta/connect/base.proto), which protobuf's builder built first.
    from google.protobuf import descriptor_pool, message_factory, reflection, symbol_database
    from google.protobuf import message as protobuf_message
    from google.protobuf.internal import enum_type_wrapper

    plugin_pb2, base_pb2, _ = plugin
    delete = plugin_pb2.DeleteFromTable
    assert issubclass(delete, _pyspark.ProtoMessage) and issubclass(delete, protobuf_message.Message)
    assert delete is ours("thirdparty.DeleteFromTable")
    assert type(delete) is type(proto.Relation) and type(delete).__name__ == "MessageMeta"
    assert delete.__module__ == "plugin_protos.plugin_pb2"
    assert delete.__qualname__ == "DeleteFromTable"
    update = plugin_pb2.MergeIntoTable.Action.UpdateAction
    assert update.__qualname__ == "MergeIntoTable.Action.UpdateAction"
    assert update.DESCRIPTOR.full_name == "thirdparty.MergeIntoTable.Action.UpdateAction"
    assert isinstance(plugin_pb2.Mode, enum_type_wrapper.EnumTypeWrapper)
    assert (plugin_pb2.Mode.Value("MODE_UPSERT"), plugin_pb2.MODE_INSERT) == (2, 1)
    assert plugin_pb2.Mode.DESCRIPTOR is plugin_pb2.MergeIntoTable.DESCRIPTOR.fields_by_name[
        "mode"
    ].enum_type
    table = base_pb2.Table
    assert table is ours("thirdparty.Table") and table.__module__ == "plugin_protos.base_pb2"

    # The descriptors of the classes are protobuf's, those of the module, as upb's.
    pool = descriptor_pool.Default()
    assert plugin_pb2._DELETEFROMTABLE is delete.DESCRIPTOR
    assert delete.DESCRIPTOR.file is plugin_pb2.DESCRIPTOR
    assert table.DESCRIPTOR.file is base_pb2.DESCRIPTOR
    assert proto.Relation.DESCRIPTOR.file.pool is pool
    assert pool.FindMessageTypeByName("thirdparty.DeleteFromTable") is delete.DESCRIPTOR
    # The runtime's pool holds the files, of the same bytes.
    file = _pyspark._proto_file_descriptor("plugin_protos/plugin.proto")
    assert file.serialized_pb == plugin_pb2.DESCRIPTOR.serialized_pb
    assert [d.name for d in file.dependencies] == [
        "spark/connect/relations.proto",
        "spark/connect/expressions.proto",
        "plugin_protos/base.proto",
    ]
    # protobuf's lookups give the runtime's classes, of nested types and of base.proto too.
    database = symbol_database.Default()
    for full_name in (
        "thirdparty.DeleteFromTable",
        "thirdparty.MergeIntoTable.Action",
        "thirdparty.Table",
    ):
        cls = ours(full_name)
        assert message_factory.GetMessageClass(cls.DESCRIPTOR) is cls
        assert database.GetSymbol(full_name) is cls
        assert reflection.GeneratedProtocolMessageType(
            cls.__name__, (protobuf_message.Message,), {"DESCRIPTOR": cls.DESCRIPTOR}
        ) is cls
    assert database.GetMessages(["plugin_protos/base.proto"]) == {
        "thirdparty.Table": table,
        "thirdparty.Table.OptionsEntry": table.OptionsEntry,
    }
    # One hook wraps protobuf's own metaclass and function, however many times the runtime
    # installs them, which protobuf's other classes keep.
    hooked = reflection.GeneratedProtocolMessageType
    own = hooked._protobuf_class
    assert own.__module__ != _rust.__name__ and isinstance(type(proto.Relation.DESCRIPTOR), type)
    _rust._hook_message_types()
    assert reflection.GeneratedProtocolMessageType._protobuf_class is own
    assert message_factory.GetMessageClass._protobuf_function.__module__ == message_factory.__name__
    # As of protobuf's: a class of upb's metaclass is not of the class of the hook, nor of
    # protobuf's subclass of upb's metaclass that it wraps.
    for cls in (proto.Relation, _upb_classes()("spark.connect.Relation")):
        assert not isinstance(cls, reflection.GeneratedProtocolMessageType)
    assert repr(reflection.GeneratedProtocolMessageType) == repr(own)

    # A module of the file built again has the same classes, as upb's has.
    assert importlib.reload(plugin_pb2).DeleteFromTable is delete
    assert plugin_pb2.MergeIntoTable.Action.UpdateAction is update
    # So has a module of a file of the protos, as a copy of Apache Spark's generated modules
    # builds it of the file of the pool: the classes of the package.
    module = _build_generated_module(_protobuf_file("spark/connect/example_plugins.proto"))
    assert module["ExamplePluginRelation"] is ours("spark.connect.ExamplePluginRelation")


def test_installs_register_the_classes_of_the_runtime_in_protobufs_database(plugin):
    # The runtime registers the classes of its own files in protobuf's symbol database, where
    # PySpark's pickles of nested messages load (Expression.Literal), as protobuf's builder
    # registers those of a plugin's module, and keeps the database's own classes of the
    # well-known types. A message of a nested type pickles as upb's, of protobuf's
    # _InternalConstructMessage of the database.
    from google.protobuf import any_pb2, symbol_database
    from google.protobuf import message as protobuf_message

    plugin_pb2, _, _ = plugin
    database = symbol_database.Default()
    literal = "spark.connect.Expression.Literal"
    assert database.GetSymbol(literal) is ours(literal)
    assert database.GetSymbol("google.protobuf.Any") is any_pb2.Any
    _rust._register_classes_in_protobuf()
    assert database.GetSymbol("google.protobuf.Any") is any_pb2.Any
    action = plugin_pb2.MergeIntoTable.Action(condition={"literal": {"integer": 1}})
    assert action.__reduce__()[:2] == (
        protobuf_message._InternalConstructMessage,
        ("thirdparty.MergeIntoTable.Action",),
    )
    assert pickle.loads(pickle.dumps(action)) == action
    assert type(pickle.loads(pickle.dumps(action))) is plugin_pb2.MergeIntoTable.Action


def test_messages_of_plugins_take_the_messages_of_the_package_as_upbs(plugin):
    # The messages of the module of a plugin take those of pyspark.sql.connect.proto as they are,
    # as upb's take PySpark's of the official client, in CopyFrom, MergeFrom, their constructors,
    # containers and maps (relation.delete_from_table.target.CopyFrom(plan) of Delta Lake's plan
    # code), and those of the module of base.proto. The messages are upb's, of their bytes, text,
    # JSON, pickles and Any.
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
    # The message of a field of a type of base.proto is of the class of its module, the runtime's,
    # as upb's is of its one class of the type: the module's class takes it as it is, and it
    # pickles alone, of its class.
    assert type(scan.table) is base_pb2.Table and isinstance(scan.table, base_pb2.Table)
    copied = base_pb2.Table()
    copied.CopyFrom(scan.table)
    assert copied == scan.table and type(pickle.loads(pickle.dumps(scan.table))) is base_pb2.Table
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
    # and options of the module's descriptors (_DELETEFROMTABLE._serialized_start), the modules of
    # the plugin have the classes of the runtime too, of upb's messages, whose descriptors are
    # those of the module, which give the runtime's classes as their _concrete_class.
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
            "base": base_pb2.Table is _rust._CLASSES["thirdparty.Table"],
            "descriptor": plugin_pb2.DeleteFromTable.DESCRIPTOR is plugin_pb2._DELETEFROMTABLE
            and plugin_pb2._DELETEFROMTABLE._concrete_class is plugin_pb2.DeleteFromTable,
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
    assert out["implementation"] == "python" and out["runtime"] and out["base"]
    assert out["descriptor"]
    assert out["start"] > 0
    real = _plugin_cases(upb, upb("thirdparty.Table"))
    assert out["bytes"] == [r.SerializeToString(deterministic=True).hex() for r in real]
    assert out["equal"] == [True, True, True, False]


def _add_plugin_file(
    name,
    dependencies,
    fields,
    syntax="proto3",
    package=None,
    message="M",
    extendable=False,
    nested=None,
):
    """The file ``name`` of protobuf's default pool, as a generated module adds it, of one message
    ``message`` of ``fields``, each its name and its message type, of ``package``, else of the
    package of the file's stem, of the extension range 100 to 200 if ``extendable`` (of proto2),
    and of a nested message ``nested`` of no fields if given. A file of the syntax ``editions`` is
    of the edition 2023."""
    from google.protobuf import descriptor_pb2, descriptor_pool

    pool = descriptor_pool.Default()
    try:
        return pool.FindFileByName(name)
    except KeyError:
        pass
    if package is None:
        package = "tests." + name.rsplit("/", 1)[-1][: -len(".proto")]
    file = descriptor_pb2.FileDescriptorProto(
        name=name, package=package, dependency=dependencies, syntax=syntax
    )
    if syntax == "editions":
        file.edition = descriptor_pb2.EDITION_2023
    message = file.message_type.add(name=message)
    if extendable:
        message.extension_range.add(start=100, end=201)
    if nested is not None:
        message.nested_type.add(name=nested)
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


def _add_extendable_file(name, dependencies, fields):
    """The file ``name`` of protobuf's default pool, as ``_add_plugin_file`` adds one, of proto2,
    whose message ``M`` declares an extension range."""
    from google.protobuf import descriptor_pb2, descriptor_pool

    pool = descriptor_pool.Default()
    try:
        return pool.FindFileByName(name)
    except KeyError:
        pass
    package = "tests." + name.rsplit("/", 1)[-1][: -len(".proto")]
    file = descriptor_pb2.FileDescriptorProto(
        name=name, package=package, dependency=dependencies, syntax="proto2"
    )
    message = file.message_type.add(name="M")
    for number, (field, type_name) in enumerate(fields, 1):
        message.field.add(
            name=field,
            number=number,
            type=descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
            type_name=type_name,
        )
    message.extension_range.add(start=100, end=200)
    return pool.AddSerializedFile(file.SerializeToString())


def test_files_that_the_runtime_does_not_hold_keep_protobufs_classes(upb):
    # A module of a file keeps protobuf's classes, as without the drop-in, when the file imports
    # no proto of Spark Connect, when it declares extensions or extension ranges, which the
    # runtime does not hold, or imports a file that does (as one of custom options imports
    # google/protobuf/descriptor.proto), and when it is of another pool than the default one (as
    # _upb_classes_of builds the classes of upb of the protos). A file of proto2, of editions, or
    # that imports well-known types other than Any and Timestamp has the runtime's classes.
    from google.protobuf import descriptor_pb2, duration_pb2
    from google.protobuf import message as protobuf_message

    relations = "spark/connect/relations.proto"
    assert duration_pb2.Duration.DESCRIPTOR.full_name == "google.protobuf.Duration"
    kept = [
        _add_plugin_file("tests/keeps/no_spark.proto", [], []),
        _add_extendable_file(
            "tests/keeps/extendable.proto", [relations], [("input", ".spark.connect.Relation")]
        ),
        _add_plugin_file(
            "tests/keeps/options.proto",
            [relations, descriptor_pb2.DESCRIPTOR.name],
            [("input", ".spark.connect.Relation"), ("options", ".google.protobuf.FileOptions")],
        ),
    ]
    held = [
        _add_plugin_file(
            "tests/keeps/proto2.proto",
            [relations],
            [("input", ".spark.connect.Relation")],
            syntax="proto2",
        ),
        _add_plugin_file(
            "tests/keeps/editions.proto",
            [relations],
            [("input", ".spark.connect.Relation")],
            syntax="editions",
        ),
        _add_plugin_file(
            "tests/keeps/duration.proto",
            [relations, "google/protobuf/duration.proto"],
            [("input", ".spark.connect.Relation"), ("timeout", ".google.protobuf.Duration")],
        ),
    ]
    for file in kept:
        cls = _build_generated_module(file)["M"]
        assert issubclass(cls, protobuf_message.Message), file.name
        assert not issubclass(cls, _pyspark.ProtoMessage), file.name
        with pytest.raises(KeyError):
            _pyspark._proto_message_descriptor(cls.DESCRIPTOR.full_name)
    for file in held:
        cls = _build_generated_module(file)["M"]
        assert cls is ours(cls.DESCRIPTOR.full_name), file.name
        assert type(cls(input=proto.Relation()).input) is proto.Relation
    assert not issubclass(upb("spark.connect.Relation"), _pyspark.ProtoMessage)


def _generated_module(file, name=None):
    """A generated module of ``file`` of the name of its path, as protoc names it
    (``tests.same.early_pb2`` of ``tests/same/early.proto``), or of ``name``, of the globals that
    protobuf's builder builds, which protobuf's symbol database registers the classes of, put in
    ``sys.modules``."""
    module_name = file.name[: -len(".proto")].replace("/", ".") + "_pb2"
    module = types.ModuleType(name or module_name)
    vars(module).update(_build_generated_module(file), DESCRIPTOR=file)
    sys.modules[module.__name__] = module
    return module


def test_a_file_of_the_package_of_a_plugins_file_that_it_imports_gets_the_runtimes_classes(upb):
    # A file that imports no proto of Spark Connect keeps protobuf's classes, as protobuf's other
    # modules do, until a plugin's file of its package that imports the protos imports it, as
    # Delta Lake's delta/connect/relations.proto imports delta/connect/base.proto, whose module
    # delta/connect/proto/__init__.py star-imports first: then its module gets the runtime's
    # classes, of its name, as does a name of a module of its package bound to one of protobuf's
    # classes that they replace (the package's), and protobuf's lookups give them, as upb's give
    # its one class. The module is the one of protobuf's class, not the package that star-imports
    # it (which binds the DESCRIPTOR of the file too); a name of the module bound to another object
    # (a class of the application of the name of a message), and any name of a module of another
    # package, stay.
    from google.protobuf import message as protobuf_message
    from google.protobuf import message_factory, symbol_database

    early = _add_plugin_file("tests/same/early.proto", [], [], package="tests.same", message="Early")
    other = _add_plugin_file("tests/same/app.proto", [], [], package="tests.same", message="App")
    late = _add_plugin_file(
        "tests/same/late.proto",
        ["spark/connect/relations.proto", early.name, other.name],
        [("input", ".spark.connect.Relation"), ("early", ".tests.same.Early")],
        package="tests.same",
    )
    # The package, which star-imports the module first, is in sys.modules before it.
    package = types.ModuleType("tests.same")
    sys.modules[package.__name__] = package
    names = [package.__name__]
    try:
        module = _generated_module(early)
        apps = _generated_module(other)
        names += [module.__name__, apps.__name__]
        kept = vars(module)["Early"]
        assert kept.__module__ == "tests.same.early_pb2"
        assert not issubclass(kept, _pyspark.ProtoMessage)
        assert issubclass(kept, protobuf_message.Message)
        vars(package).update(Early=kept, DESCRIPTOR=early, kept=[kept])
        unrelated = types.ModuleType("tests_other_package.unrelated")
        vars(unrelated).update(Early=kept)
        sys.modules[unrelated.__name__] = unrelated
        names.append(unrelated.__name__)

        class App:
            """A class of the application of the name of a message of the module."""

        vars(apps)["App"] = App
        built = _build_generated_module(late)
    finally:
        for name in names:
            del sys.modules[name]
    assert issubclass(built["M"], _pyspark.ProtoMessage)
    held = ours("tests.same.Early")
    assert vars(module)["Early"] is held and type(built["M"]().early) is held
    assert vars(package)["Early"] is held and vars(package)["kept"] == [kept]
    assert vars(package)["DESCRIPTOR"] is early and kept.DESCRIPTOR is held.DESCRIPTOR
    assert vars(unrelated)["Early"] is kept and vars(apps)["App"] is App
    assert (held.__module__, held.__qualname__) == (module.__name__, "Early")
    assert message_factory.GetMessageClass(held.DESCRIPTOR) is held
    assert symbol_database.Default().GetSymbol("tests.same.Early") is held
    # A message of protobuf's class that the module had before compares equal to the runtime's,
    # and the runtime's fields take it.
    assert kept() == held() and built["M"](early=kept()).HasField("early")


def test_a_file_of_another_package_that_a_plugins_file_imports_keeps_protobufs_classes(upb):
    # A file of another package that a plugin's file imports (a company's shared proto, or Google's
    # google/rpc/status.proto), which imports no proto of Spark Connect, keeps protobuf's classes,
    # in its module and of protobuf's lookups, so that the modules of upb that import it take its
    # messages as they do without the drop-in: in constructors, CopyFrom, containers, maps,
    # isinstance and pickles, of messages made before the plugin's module built too. The plugin's
    # message fields of its types hold the runtime's class of the type, of the module and
    # qualified names of protobuf's, which pickles as upb's of protobuf's class, and take
    # protobuf's messages of the type, of their bytes.
    from google.protobuf import message_factory, symbol_database

    common = _add_plugin_file(
        "tests/shared/common.proto", [], [], package="tests.shared", message="Common"
    )
    user = _add_plugin_file(
        "tests/shared/user.proto",
        [common.name],
        [("c", ".tests.shared.Common")],
        package="tests.user",
        message="User",
    )
    plugin_file = _add_plugin_file(
        "tests/shared/plugin.proto",
        ["spark/connect/relations.proto", common.name],
        [("input", ".spark.connect.Relation"), ("c", ".tests.shared.Common")],
        package="tests.shared_plugin",
    )
    module = _generated_module(common)
    try:
        common_class = vars(module)["Common"]
        early = common_class()
        plugin = _build_generated_module(plugin_file)
        user_class = _build_generated_module(user)["User"]
        assert vars(module)["Common"] is common_class
        assert not issubclass(common_class, _pyspark.ProtoMessage)
        assert message_factory.GetMessageClass(common_class.DESCRIPTOR) is common_class
        assert symbol_database.Default().GetSymbol("tests.shared.Common") is common_class
        # upb's messages of a file that imports it take protobuf's messages of it.
        assert user_class(c=common_class()).HasField("c")
        copied = user_class()
        copied.c.CopyFrom(common_class())
        assert copied.HasField("c") and isinstance(user_class().c, common_class)
        assert type(pickle.loads(pickle.dumps(early))) is common_class
        assert type(pickle.loads(pickle.dumps(copied.c))) is common_class
        # The plugin's messages of its type: of the runtime's class of the type.
        message = plugin["M"](c=common_class(), input=proto.Relation())
        held = type(message.c)
        assert message.HasField("c") and issubclass(held, _pyspark.ProtoMessage)
        assert held is not common_class and held.DESCRIPTOR is common_class.DESCRIPTOR
        assert (held.__module__, held.__qualname__) == (module.__name__, "Common")
        assert message.c == common_class() and common_class() == message.c
        assert pickle.dumps(message.c) == pickle.dumps(common_class())
        assert type(pickle.loads(pickle.dumps(message.c))) is common_class
        assert type(copy.copy(message.c)) is held
    finally:
        del sys.modules[module.__name__]


def test_googles_files_that_a_plugins_file_imports_keep_protobufs_classes(upb):
    # Google's files (google/rpc/status.proto of googleapis-common-protos) that a plugin's file
    # imports keep protobuf's classes, of whatever package the plugin's file is, so that the
    # modules of other libraries that import them (grpcio-status, google/longrunning) take their
    # messages, as they do without the drop-in.
    status_pb2 = pytest.importorskip("google.rpc.status_pb2")
    from google.protobuf import message_factory

    status = status_pb2.Status
    plugin_file = _add_plugin_file(
        "google/rpc/tests_plugin.proto",
        ["spark/connect/relations.proto", "google/rpc/status.proto"],
        [("input", ".spark.connect.Relation"), ("status", ".google.rpc.Status")],
        package="google.rpc",
    )
    plugin = _build_generated_module(plugin_file)
    assert status_pb2.Status is status and not issubclass(status, _pyspark.ProtoMessage)
    assert message_factory.GetMessageClass(status.DESCRIPTOR) is status
    message = plugin["M"](status=status(code=5, message="m"))
    assert message.status.code == 5 and message.status == status(code=5, message="m")
    assert type(pickle.loads(pickle.dumps(message.status))) is status
    try:
        from google.longrunning import operations_proto_pb2
    except ImportError:
        return
    operation = operations_proto_pb2.Operation(name="op", error=status(code=5))
    operation.error.CopyFrom(status(code=6))
    assert operation.error.code == 6 and isinstance(operation.error, status)


def test_a_protobuf_that_lacks_the_hooked_metaclass_and_function_imports_the_package(upb):
    # The hooks wrap protobuf's metaclass of its message classes and message_factory's
    # GetMessageClass. Where a protobuf lacks them, the package imports without them, of the
    # classes of the runtime.
    script = textwrap.dedent(
        """
        from google.protobuf import message_factory, reflection

        del reflection.GeneratedProtocolMessageType
        del message_factory._GENERATED_PROTOCOL_MESSAGE_TYPE
        del message_factory.GetMessageClass
        from pyspark import _pyspark
        import pyspark.sql.connect.proto as proto

        assert issubclass(proto.Relation, _pyspark.ProtoMessage)
        assert proto.Relation(sql={"query": "q"}).sql.query == "q"
        assert not hasattr(reflection, "GeneratedProtocolMessageType")
        assert not hasattr(message_factory, "GetMessageClass")
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


def test_classes_of_a_plugins_file_that_message_factory_makes_are_the_runtimes(upb):
    # The files of the protos are in protobuf's default pool, as the generated modules add them,
    # so that a file of a plugin that imports them builds, as the module that protoc generates of
    # it adds it (Delta Lake's do). The class that protobuf's message_factory gives of a message
    # of such a file, of no generated module, is the runtime's, as upb's gives its one class of
    # the type (of no module): it takes a message of pyspark.sql.connect.proto as it is, and an
    # Any of the package packs and unpacks it. protobuf's own class of the type (as a module built
    # before the drop-in installs has) takes the package's messages through their bytes or a dict.
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
    descriptor = pool.FindMessageTypeByName("tests.plugin.PluginRelation")
    plugin_class = message_factory.GetMessageClass(descriptor)
    assert plugin_class is ours("tests.plugin.PluginRelation") and plugin_class.__module__ is None
    assert plugin_class.DESCRIPTOR is descriptor

    relation = proto.Relation(sql=proto.SQL(query="select 1"))
    plugin = plugin_class(input=relation, custom="c")
    assert type(plugin.input) is proto.Relation and plugin.input == relation
    assert plugin == plugin_class(input={"sql": {"query": "select 1"}}, custom="c")
    wrapped = proto.Relation()
    wrapped.extension.Pack(plugin)
    assert wrapped.extension.TypeName() == "tests.plugin.PluginRelation"
    unpacked = plugin_class()
    assert wrapped.extension.Unpack(unpacked) and unpacked == plugin
    back = proto.Relation()
    back.CopyFrom(unpacked.input)
    assert back == relation and back.SerializeToString() == plugin.input.SerializeToString()

    protobufs = _protobuf_class("tests.plugin.PluginRelation")
    assert protobufs is not plugin_class and not issubclass(protobufs, _pyspark.ProtoMessage)
    kept = protobufs(custom="c")
    kept.input.ParseFromString(relation.SerializeToString())
    assert kept == protobufs(input={"sql": {"query": "select 1"}}, custom="c") == plugin
    with pytest.raises(TypeError):
        protobufs(input=relation)
    with pytest.raises(TypeError):
        kept.input.CopyFrom(relation)


def test_messages_of_a_type_of_protobufs_classes_and_of_the_runtimes_compare_as_upbs(plugin):
    # upb has one class of a type of protobuf's default pool, so its messages of the type compare
    # by their contents. The runtime has a class of the type besides protobuf's own, which a
    # module of the type that protobuf's builder built before the drop-in installed has, as the
    # well-known types other than Any and Timestamp do: a message of the runtime equals protobuf's
    # message of the type of the same contents, on either side of == and !=, and so in a container
    # and a map, as upb's messages of one class do, and equals no message of another pool (of
    # another class for upb too), class or other object.
    plugin_pb2, base_pb2, upb = plugin
    protobufs = _protobuf_class
    protobuf_table = protobufs("thirdparty.Table")

    scan = plugin_pb2.Scan(table=protobuf_table(path="/p", options={"k": "v"}))
    table = protobuf_table(path="/p", options={"k": "v"})
    other = protobuf_table(path="/q")
    assert type(scan.table) is base_pb2.Table and type(table) is not base_pb2.Table
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
    for value in (base_pb2.Table, protobuf_table, {"path": "/p"}, None, "/p"):
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


def test_equality_of_messages_compares_floats_of_their_bits_as_upbs(upb):
    # upb's == of two messages (upb_Message_IsEqual) compares a float or a double field of its
    # bits, as it compares an integer: a NaN equals a NaN of the same bits, a message of a NaN
    # equals its copy, and -0.0 is not 0.0, nor the unset default of a field without presence. So
    # do `in` and `remove` of the containers of messages, and == of two messages' maps. The
    # runtime compared their values, as Python's ==, so a message of a NaN never equaled itself
    # but by its identity. == of the containers and maps themselves compares their values as
    # Python's == does, on both runtimes.
    import struct

    nan = float("nan")
    other_nan = struct.unpack("<d", struct.pack("<Q", 0x7FF8000000000001))[0]

    def cases(new):
        literal = new("spark.connect.Expression.Literal")
        sample = new("spark.connect.Sample")
        quantile = new("spark.connect.StatApproxQuantile")
        sql = new("spark.connect.SQL")
        relation = new("spark.connect.Relation")
        return {
            "double NaN": (literal(double=nan), literal(double=nan)),
            "double NaNs of other bits": (literal(double=nan), literal(double=other_nan)),
            "double -0.0 and 0.0": (literal(double=-0.0), literal(double=0.0)),
            "float NaN": (literal(float=nan), literal(float=nan)),
            "float -0.0 and 0.0": (literal(float=-0.0), literal(float=0.0)),
            "a NaN of no presence": (sample(upper_bound=nan), sample(upper_bound=nan)),
            "-0.0 of no presence and the default": (sample(lower_bound=-0.0), sample()),
            "-0.0 and 0.0 of no presence": (sample(lower_bound=-0.0), sample(lower_bound=0.0)),
            "repeated NaN": (
                quantile(probabilities=[nan, 1.0]),
                quantile(probabilities=[nan, 1.0]),
            ),
            "repeated -0.0 and 0.0": (
                quantile(probabilities=[-0.0]),
                quantile(probabilities=[0.0]),
            ),
            "a map of a NaN": (
                sql(args={"k": {"double": nan}}),
                sql(args={"k": {"double": nan}}),
            ),
            "a nested NaN": (
                relation(sample={"upper_bound": nan}),
                relation(sample={"upper_bound": nan}),
            ),
        }

    real_cases, mine_cases = cases(upb), cases(ours)
    for label in real_cases:
        (r1, r2), (m1, m2) = real_cases[label], mine_cases[label]
        real = (r1 == r2, r1 != r2, r2 == r1, r1 == copy.deepcopy(r1))
        real += (r1 == type(r1).FromString(r1.SerializeToString()),)
        mine = (m1 == m2, m1 != m2, m2 == m1, m1 == copy.deepcopy(m1))
        mine += (m1 == type(m1).FromString(m1.SerializeToString()),)
        assert mine == real, label
    for new in (upb, ours):
        literal = new("spark.connect.Expression.Literal")
        array = new("spark.connect.Expression.Literal.Array")(elements=[{"double": nan}])
        assert literal(double=nan) in array.elements
        array.elements.remove(literal(double=nan))
        assert len(array.elements) == 0
        # The containers and maps compare their values as Python's == does.
        quantile = new("spark.connect.StatApproxQuantile")
        with_nan = quantile(probabilities=[nan]).probabilities
        assert with_nan != quantile(probabilities=[nan]).probabilities and with_nan != with_nan
        assert quantile(probabilities=[-0.0]).probabilities == [0.0]


def test_unknown_fields_of_messages_compare_as_upbs(upb):
    # upb's == compares the unknown fields of two messages by their fields, of their tags and
    # values, in the order of their tags (upb_Message_UnknownFieldsAreEqual): the order of the
    # fields of other numbers or wire types and the encoding of a varint do not count, in a group
    # either, and fields of the same tag keep their order. The runtime compared their bytes. So do
    # a message of the runtime and protobuf's message of its type.
    from google.protobuf import descriptor_pool, message_factory

    def group(number, body):
        return _tag(number, 3) + body + _tag(number, 4)

    first, second = _tag(1, 0) + _varint(7), _tag(2, 0) + _varint(8)
    varint, fixed32 = _tag(501, 0) + _varint(5), _tag(501, 5) + b"\x05\x00\x00\x00"
    cases = {
        "the same fields": (varint + _field(502, b"abc"), varint + _field(502, b"abc")),
        "fields of other numbers swapped": (
            varint + _field(502, b"abc"),
            _field(502, b"abc") + varint,
        ),
        "three numbers rotated": (
            _tag(503, 0) + b"\x01" + _tag(501, 0) + b"\x02" + _tag(502, 0) + b"\x03",
            _tag(501, 0) + b"\x02" + _tag(502, 0) + b"\x03" + _tag(503, 0) + b"\x01",
        ),
        "fields of the same number and wire type swapped": (
            varint + _tag(501, 0) + _varint(6),
            _tag(501, 0) + _varint(6) + varint,
        ),
        "fields of the same number of other wire types swapped": (
            varint + fixed32,
            fixed32 + varint,
        ),
        "a varint of more bytes": (_tag(501, 0) + b"\x85\x00", varint),
        "a varint of ten bytes": (_tag(501, 0) + b"\x85" + b"\x80" * 8 + b"\x00", varint),
        "a varint and a fixed32 of the same value": (varint, fixed32),
        "length-delimited fields of their fields swapped": (
            _field(502, first + second),
            _field(502, second + first),
        ),
        "a group of its fields swapped": (group(503, first + second), group(503, second + first)),
        "groups swapped": (
            group(503, first) + group(504, second),
            group(504, second) + group(503, first),
        ),
        "nested groups of their fields swapped": (
            group(503, group(5, first + second)),
            group(503, group(5, second + first)),
        ),
        "a fixed64 of other bytes": (_tag(501, 1) + b"\x01" * 8, _tag(501, 1) + b"\x02" * 8),
        "one field and none": (varint, b""),
    }
    pool = descriptor_pool.Default()
    protobufs = message_factory.GetMessageClass(
        pool.FindMessageTypeByName("spark.connect.Expression.Literal")
    )
    real, mine = upb("spark.connect.Expression.Literal"), ours("spark.connect.Expression.Literal")
    for label, (a, b) in cases.items():
        expected = real.FromString(a) == real.FromString(b)
        assert (mine.FromString(a) == mine.FromString(b)) is expected, label
        assert (mine.FromString(a) != mine.FromString(b)) is not expected, label
        assert (mine.FromString(a) == protobufs.FromString(b)) is expected, label
        assert (protobufs.FromString(b) == mine.FromString(a)) is expected, label
        # In a message that the message holds too.
        held_real = upb("spark.connect.Relation")()
        held_real.limit.MergeFromString(a)
        held_mine = ours("spark.connect.Relation")()
        held_mine.limit.MergeFromString(a)
        other_real, other_mine = upb("spark.connect.Relation")(), ours("spark.connect.Relation")()
        other_real.limit.MergeFromString(b)
        other_mine.limit.MergeFromString(b)
        assert (held_mine == other_mine) is (held_real == other_real) is expected, label


class _Lookalike:
    """An object of the `DESCRIPTOR` and `SerializeToString` of a message, which is not one."""

    def __init__(self, cls, data):
        self.DESCRIPTOR = cls.DESCRIPTOR
        self.data = data

    def SerializeToString(self, **kwargs):
        return self.data

    def __repr__(self):
        return "<a lookalike>"


def test_messages_take_only_protobufs_messages_of_their_type_of_its_default_pool_as_upbs(upb):
    # upb's messages take a message of their class only. upb has one class of a type of a pool, so
    # the runtime takes protobuf's message of its type of protobuf's default pool, whose files it
    # holds, as upb takes its own. It also took a message of a type of the same name of another
    # pool, and any object of a DESCRIPTOR of the name and a SerializeToString, of their bytes, in
    # CopyFrom, MergeFrom, the constructors and the containers, which upb refuses, with the errors
    # of another class. == already compared them unequal.
    import types

    others = _upb_classes_of([])

    def run(new, other_of):
        literal = new("spark.connect.Expression.Literal")
        array = new("spark.connect.Expression.Literal.Array")
        expression = new("spark.connect.Expression")
        sql = new("spark.connect.SQL")
        other = other_of()
        return {
            "CopyFrom": lambda: literal().CopyFrom(other),
            "MergeFrom": lambda: literal().MergeFrom(other),
            "CopyFrom of a field": lambda: expression().literal.CopyFrom(other),
            "the constructor of a field": lambda: expression(literal=other),
            "the constructor of a repeated field": lambda: array(elements=[other]),
            "append": lambda: array().elements.append(other),
            "extend": lambda: array().elements.extend([other]),
            "insert": lambda: array().elements.insert(0, other),
            "==": lambda: literal(integer=5) == other,
            "!=": lambda: literal(integer=5) != other,
            "in": lambda: other in array(elements=[{"integer": 5}]).elements,
            "a value of a map of the constructor": lambda: sql(args={"k": other}),
            "MergeFrom of a map": lambda: sql().args.MergeFrom(types.MappingProxyType({"k": other})),
        }

    name = "spark.connect.Expression.Literal"
    data = ours(name)(integer=5).SerializeToString()
    # A message of another pool: of the pool of _upb_classes_of for the runtime, and of a pool of
    # its own for upb's classes. A lookalike of the descriptor of the class.
    for kind, real_other, mine_other in [
        ("another pool", lambda: others[name](integer=5), lambda: upb(name)(integer=5)),
        ("a lookalike", lambda: _Lookalike(upb(name), data), lambda: _Lookalike(ours(name), data)),
    ]:
        real_cases, mine_cases = run(upb, real_other), run(ours, mine_other)
        for label in real_cases:
            assert _outcome(mine_cases[label]) == _outcome(real_cases[label]), (kind, label)


def test_modules_of_files_of_proto2_editions_and_well_known_types_take_the_runtimes_messages(upb):
    # A module of a plugin's file of proto2, of editions or that imports a well-known type other
    # than Any and Timestamp, which imports a file whose module has the runtime's classes (a
    # proto3 file that imports the protos of Spark Connect), has the runtime's classes too, so it
    # takes the messages of that module as they are, as upb's take those of its one class of the
    # type. One that keeps protobuf's classes (of an extension range) takes them only through
    # their bytes or dicts, and its messages and those of the runtime of the same contents
    # compare equal, as upb's of one class do.
    from google.protobuf import duration_pb2

    held = _add_plugin_file(
        "tests/importers/runtime_held.proto",
        ["spark/connect/relations.proto"],
        [("input", ".spark.connect.Relation")],
    )
    dependency = [("held", ".tests.runtime_held.M")]
    importers = [
        _add_plugin_file(
            "tests/importers/with_duration.proto",
            [held.name, duration_pb2.DESCRIPTOR.name],
            dependency + [("timeout", ".google.protobuf.Duration")],
        ),
        _add_plugin_file("tests/importers/of_proto2.proto", [held.name], dependency, "proto2"),
        _add_plugin_file("tests/importers/of_editions.proto", [held.name], dependency, "editions"),
    ]
    runtime_class = _build_generated_module(held)["M"]
    assert issubclass(runtime_class, _pyspark.ProtoMessage)
    real = _upb_classes_of(
        [(duration_pb2.DESCRIPTOR.serialized_pb, duration_pb2.__name__)]
        + [(file.serialized_pb, file.name[: -len(".proto")].replace("/", ".") + "_pb2")
           for file in [held] + importers]
    )
    message = runtime_class(input={"sql": {"query": "select 1"}})
    for file in importers:
        cls = _build_generated_module(file)["M"]
        assert issubclass(cls, _pyspark.ProtoMessage), file.name
        built = cls(held=message)
        copied = cls()
        copied.held.CopyFrom(message)
        merged = cls()
        merged.held.MergeFrom(message)
        assert type(built.held) is runtime_class and built == copied == merged
        # upb's take the messages of the class of the type of the file that they import as they are.
        real_kept = real[cls.DESCRIPTOR.full_name]
        real_message = real["tests.runtime_held.M"](input={"sql": {"query": "select 1"}})
        assert real_kept(held=real_message).SerializeToString() == built.SerializeToString()

    kept = _build_generated_module(
        _add_extendable_file("tests/importers/extendable_importer.proto", [held.name], dependency)
    )["M"]
    assert not issubclass(kept, _pyspark.ProtoMessage)
    with pytest.raises(TypeError):
        kept(held=message)
    with pytest.raises(TypeError):
        kept().held.CopyFrom(message)
    through_bytes = kept()
    through_bytes.held.ParseFromString(message.SerializeToString())
    through_dict = kept(held={"input": {"sql": {"query": "select 1"}}})
    assert through_bytes == through_dict
    assert through_bytes.held == message and message == through_dict.held


def test_modules_of_protoc_before_3_20_have_the_classes_of_the_runtime(upb):
    # A module of protoc before 3.20 (plugin_protos/old_style_pb2.py, of protoc 3.19's form) makes
    # its classes with protobuf's reflection.GeneratedProtocolMessageType, not with protobuf's
    # builder: the drop-in wraps it, so the module of a file that imports the protos of Spark
    # Connect has the classes of the runtime, as a module of a later protoc, which take the
    # messages of pyspark.sql.connect.proto as they are, as upb's take PySpark's, and which
    # protobuf's symbol database registers.
    from google.protobuf import message as protobuf_message
    from google.protobuf import symbol_database
    from plugin_protos import old_style_pb2

    old_style = old_style_pb2.OldStyleRelation
    assert old_style is ours("thirdparty.old.OldStyleRelation")
    assert issubclass(old_style, protobuf_message.Message)
    assert old_style.__module__ == "plugin_protos.old_style_pb2"
    assert old_style.DESCRIPTOR is old_style_pb2._OLDSTYLERELATION
    assert symbol_database.Default().GetSymbol("thirdparty.old.OldStyleRelation") is old_style
    relation = proto.Relation(sql=proto.SQL(query="select 1"))
    built = old_style(input=relation, name="n")
    assert type(built.input) is proto.Relation
    copied = old_style()
    copied.input.CopyFrom(relation)
    copied.name = "n"
    through_dict = old_style(input={"sql": {"query": "select 1"}}, name="n")
    assert built == copied == through_dict and through_dict.input == relation
    # As upb's of a pool of their own, as the official client's of its upb classes of the protos.
    real = _upb_classes_of([(old_style_pb2.DESCRIPTOR.serialized_pb, old_style_pb2.__name__)])
    real_relation = real["spark.connect.Relation"](sql={"query": "select 1"})
    real_old_style = real["thirdparty.old.OldStyleRelation"](input=real_relation, name="n")
    assert real_old_style.SerializeToString() == built.SerializeToString()


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


_INTEGER_MAPS = "tests/intmaps/maps.proto"


def _integer_maps(upb):
    """The class of the runtime and upb's class of ``tests.intmaps.M``, a message of a plugin's file
    (which imports Spark Connect's relations.proto) of maps of each integer type of key, of bool
    keys and of string keys, to strings: the protos of Spark Connect have maps of string keys only.
    The runtime's class is that of the module that protoc generates of the file, which the hook
    gives the runtime's classes; upb's is of a pool of its own, as ``_upb_classes_of`` makes
    them."""
    from google.protobuf import descriptor_pb2, descriptor_pool

    field = descriptor_pb2.FieldDescriptorProto
    file = descriptor_pb2.FileDescriptorProto(
        name=_INTEGER_MAPS,
        package="tests.intmaps",
        dependency=["spark/connect/relations.proto"],
        syntax="proto3",
    )
    message = file.message_type.add(name="M")
    maps = [
        ("i32", field.TYPE_INT32, field.TYPE_STRING),
        ("i64", field.TYPE_INT64, field.TYPE_STRING),
        ("u32", field.TYPE_UINT32, field.TYPE_STRING),
        ("u64", field.TYPE_UINT64, field.TYPE_STRING),
        ("s32", field.TYPE_SINT32, field.TYPE_STRING),
        ("sf64", field.TYPE_SFIXED64, field.TYPE_STRING),
        ("b", field.TYPE_BOOL, field.TYPE_STRING),
        ("s", field.TYPE_STRING, field.TYPE_STRING),
    ]
    for number, (name, key_type, value_type) in enumerate(maps, 1):
        entry = message.nested_type.add(name=name.capitalize() + "Entry")
        entry.options.map_entry = True
        entry.field.add(name="key", number=1, type=key_type, label=field.LABEL_OPTIONAL)
        entry.field.add(name="value", number=2, type=value_type, label=field.LABEL_OPTIONAL)
        message.field.add(
            name=name,
            number=number,
            type=field.TYPE_MESSAGE,
            label=field.LABEL_REPEATED,
            type_name=f".tests.intmaps.M.{entry.name}",
        )
    message.field.add(
        name="input",
        number=len(maps) + 1,
        type=field.TYPE_MESSAGE,
        label=field.LABEL_OPTIONAL,
        type_name=".spark.connect.Relation",
    )
    serialized = file.SerializeToString()
    pool = descriptor_pool.Default()
    try:
        added = pool.FindFileByName(_INTEGER_MAPS)
    except KeyError:
        added = pool.AddSerializedFile(serialized)
    mine = _build_generated_module(added)["M"]
    assert issubclass(mine, _pyspark.ProtoMessage)
    real = _upb_classes_of([(serialized, "tests.intmaps.maps_pb2")])["tests.intmaps.M"]
    return mine, real


def test_maps_of_integer_keys_merge_serialize_and_print_as_upbs(upb):
    # The protos of Spark Connect have maps of string keys only, so a plugin's file has the maps of
    # the other types of keys. (The test comes after those that compare each class of the runtime
    # with upb's, as the file adds its classes to the runtime's.) Their MergeFrom converts the keys
    # of a map of another type, of either runtime, as upb's does, with its errors. upb orders the
    # keys of a map of a signed integer type of their bits as unsigned numbers, so a negative key
    # comes after the others: of its deterministic serialization (the order of the runtime's
    # serialization) and of its text (str()). The runtime ordered them as signed numbers. protobuf's
    # text_format sorts them itself, as signed numbers, of either runtime.
    from google.protobuf import text_format

    mine_class, real_class = _integer_maps(upb)
    for source_field, key in (("u32", 4), ("i64", 4), ("i64", 2**40), ("b", True), ("s", "k")):
        outcomes = []
        for cls in (real_class, mine_class):
            target, source = cls(i32={1: "x"}), cls()
            getattr(source, source_field)[key] = "q"
            outcome = _outcome(lambda: target.i32.MergeFrom(getattr(source, source_field)))
            outcomes.append(
                (outcome, sorted(target.i32.items()), target.SerializeToString(deterministic=True))
            )
        assert outcomes[0] == outcomes[1], (source_field, key)
    keys = {
        "i32": [3, -5, 1, 100, 0, -1, -(2**31), 2**31 - 1],
        "i64": [3, 2**40, 1, -7, 0, -(2**63), 2**63 - 1],
        "s32": [3, -5, 1, 0, -1],
        "sf64": [3, -5, 1, 0, -(2**63)],
        "u32": [3, 2**31, 1, 0, 2**32 - 1],
        "u64": [3, 2**63, 1, 0, 2**64 - 1],
        "b": [True, False],
    }
    real, mine = real_class(), mine_class()
    for field, values in keys.items():
        for value in values:
            for message in (real, mine):
                getattr(message, field)[value] = f"v{value}"
    assert mine.SerializeToString(deterministic=True) == real.SerializeToString(deterministic=True)
    assert mine.SerializeToString() == real.SerializeToString(deterministic=True)
    assert mine.ByteSize() == real.ByteSize()
    assert str(mine) == str(real) and repr(mine) == repr(real)
    assert text_format.MessageToString(mine) == text_format.MessageToString(real)
    assert mine_class.FromString(real.SerializeToString()) == mine


_EQUALITY_MAPS = "tests/eqmaps/maps.proto"


def _maps_of_kinds(upb):
    """The class of the runtime and upb's class of ``tests.eqmaps.M``, a message of a plugin's file
    (which imports Spark Connect's relations.proto) of maps of int32, int64, uint32 and bool keys to
    strings, of a map of int32 to int32 and one of int64 to int64, and of a map of strings to
    doubles: the protos of Spark Connect have maps of string keys only, of no floats. The runtime's
    class is that of the module that protoc generates of the file, which the hook gives the
    runtime's classes; upb's is of a pool of its own, as ``_upb_classes_of`` makes them."""
    from google.protobuf import descriptor_pb2, descriptor_pool

    field = descriptor_pb2.FieldDescriptorProto
    file = descriptor_pb2.FileDescriptorProto(
        name=_EQUALITY_MAPS,
        package="tests.eqmaps",
        dependency=["spark/connect/relations.proto"],
        syntax="proto3",
    )
    message = file.message_type.add(name="M")
    maps = [
        ("i32", field.TYPE_INT32, field.TYPE_STRING),
        ("i64", field.TYPE_INT64, field.TYPE_STRING),
        ("u32", field.TYPE_UINT32, field.TYPE_STRING),
        ("b", field.TYPE_BOOL, field.TYPE_STRING),
        ("ii", field.TYPE_INT32, field.TYPE_INT32),
        ("ll", field.TYPE_INT64, field.TYPE_INT64),
        ("d", field.TYPE_STRING, field.TYPE_DOUBLE),
    ]
    for number, (name, key_type, value_type) in enumerate(maps, 1):
        entry = message.nested_type.add(name=name.capitalize() + "Entry")
        entry.options.map_entry = True
        entry.field.add(name="key", number=1, type=key_type, label=field.LABEL_OPTIONAL)
        entry.field.add(name="value", number=2, type=value_type, label=field.LABEL_OPTIONAL)
        message.field.add(
            name=name,
            number=number,
            type=field.TYPE_MESSAGE,
            label=field.LABEL_REPEATED,
            type_name=f".tests.eqmaps.M.{entry.name}",
        )
    message.field.add(
        name="input",
        number=len(maps) + 1,
        type=field.TYPE_MESSAGE,
        label=field.LABEL_OPTIONAL,
        type_name=".spark.connect.Relation",
    )
    serialized = file.SerializeToString()
    pool = descriptor_pool.Default()
    try:
        added = pool.FindFileByName(_EQUALITY_MAPS)
    except KeyError:
        added = pool.AddSerializedFile(serialized)
    mine = _build_generated_module(added)["M"]
    assert issubclass(mine, _pyspark.ProtoMessage)
    real = _upb_classes_of([(serialized, "tests.eqmaps.maps_pb2")])["tests.eqmaps.M"]
    return mine, real


def test_maps_of_integer_keys_and_of_doubles_compare_as_upbs(upb):
    # The protos of Spark Connect have maps of string keys only, of no floats, so a plugin's file
    # has the others. (The test comes after those that compare each class of the runtime with
    # upb's, as the file adds its classes to the runtime's.) upb's == of two maps is
    # Mapping.__eq__, of the dicts of their items: a map of int32 keys equals a map of int64 or
    # bool keys of equal keys and values, as dicts of them are equal, and a map of a NaN value
    # equals no map, itself neither, where == of two messages compares the values of their maps of
    # their bits, as their fields. The runtime compared the maps of other types unequal and a map
    # of a NaN equal to itself, and the messages of a NaN unequal and of -0.0 equal to 0.0.
    nan = float("nan")
    mine_class, real_class = _maps_of_kinds(upb)
    for cls in (real_class, mine_class):
        message = cls(i32={1: "t"}, i64={1: "t"}, u32={1: "t"}, b={True: "t"})
        message.ii[1] = 2
        message.ll[1] = 2
        assert message.i32 == message.i64 and message.i64 == message.i32, cls
        assert message.i32 == message.u32 and message.i32 == message.b, cls
        assert not (message.i32 != message.b) and message.ii == message.ll, cls
        assert message.i32 != cls(i64={1: "u"}).i64 and message.i32 == {1: "t"}, cls
        with_nan = cls(d={"x": nan})
        assert with_nan == cls(d={"x": nan}) and with_nan == copy.deepcopy(with_nan), cls
        assert with_nan.d != cls(d={"x": nan}).d and with_nan.d != with_nan.d, cls
        assert not (with_nan.d == with_nan.d) and with_nan.d != {"x": nan}, cls
        assert cls(d={"x": -0.0}) != cls(d={"x": 0.0}), cls
        assert cls(d={"x": -0.0}).d == cls(d={"x": 0.0}).d, cls


def test_classes_are_of_upbs_metaclass_bases_and_attributes(upb):
    # As upb's: the classes are of a metaclass MessageMeta, of the bases of a C message (the
    # runtime's Message), protobuf's Message and protobuf's methods of a well-known type, of the
    # attributes of the builder (DESCRIPTOR, the nested classes, __slots__), of the dir() of their
    # class and of their messages (and a __copy__ of the runtime's Message, which copies a message
    # of its fields, where upb's copy parses its bytes), and of the attributes that upb's
    # metaclass binds when they are read, which the runtime's binds when it makes the class: the
    # wrappers of the nested enums, the numbers of their values, and the numbers of all the
    # fields. A message class cannot be subclassed, and the metaclass gives the class of a
    # descriptor, with upb's errors.
    from google.protobuf import message as protobuf_message
    from google.protobuf.internal import enum_type_wrapper, well_known_types

    for name in ("spark.connect.Join", "spark.connect.Expression.Literal", "google.protobuf.Any"):
        mine, real = ours(name), upb(name)
        assert type(mine).__name__ == type(real).__name__ == "MessageMeta"
        assert mine.__mro__[1] is _pyspark.ProtoMessage and mine.__mro__[2:] == real.__mro__[2:]
        # (The runtime's class of a well-known type is of a module of its own.)
        assert mine.__qualname__ == real.__qualname__
        assert mine.__module__ == real.__module__ or name.startswith("google.protobuf.")
        # (Of the attributes of the class but those that the metaclass binds once they are read.)
        bound = {e.name for e in mine.DESCRIPTOR.enum_types} | set(mine.DESCRIPTOR.enum_values_by_name)

        def names(obj):
            return {n for n in dir(obj) if not n.endswith("_FIELD_NUMBER") and n not in bound}

        assert names(mine) - {"__copy__"} == names(real) and "__copy__" not in names(real)
        assert names(mine()) - {"__copy__"} == names(real())
        assert isinstance(mine(), protobuf_message.Message)
        for cls in (mine, real):
            subclassing = _outcome(lambda: type("Sub", (cls,), {}))
            assert subclassing[2].endswith(f"not (<class '{cls.__module__}.{cls.__qualname__}'>,)")
        meta = type(mine)
        descriptor = {"DESCRIPTOR": mine.DESCRIPTOR}
        assert meta(mine.__name__, (protobuf_message.Message,), descriptor) is mine
        assert meta(mine.__name__, (), {"DESCRIPTOR": mine.DESCRIPTOR}) is mine
        assert _outcome(lambda: meta("X", (protobuf_message.Message,), {})) == _outcome(
            lambda: type(real)("X", (protobuf_message.Message,), {})
        )
        mine_error = _outcome(lambda: meta("X", (mine,), {"DESCRIPTOR": 1}))
        real_error = _outcome(lambda: type(real)("X", (real,), {"DESCRIPTOR": 1}))
        assert mine_error[:2] == real_error[:2] == ("raise", "TypeError")
        assert mine_error[2].replace(mine.__module__, real.__module__) == real_error[2]
    assert ours("google.protobuf.Any").__mro__[3] is well_known_types.WKTBASES["google.protobuf.Any"]

    join, real_join = ours("spark.connect.Join"), upb("spark.connect.Join")
    fresh = [n for n in vars(join) if n.isupper() or n.endswith("Type")]
    for cls in (join, real_join):
        assert isinstance(cls.JoinType, enum_type_wrapper.EnumTypeWrapper) and "JoinType" in vars(cls)
        assert cls.JOIN_TYPE_INNER == 1 and "JOIN_TYPE_INNER" in vars(cls)
        assert cls.LEFT_FIELD_NUMBER == 1 and "JOIN_CONDITION_FIELD_NUMBER" in vars(cls)
        assert _outcome(lambda: cls.nope)[:2] == ("raise", "AttributeError")
    assert set(vars(join)) == set(vars(real_join)) | set(fresh)
    # (upb's error of a name that a class lacks is of the name; the runtime's, of no __getattr__
    # of its metaclass, Python's.)
    assert _outcome(lambda: join.nope)[2] == "type object 'Join' has no attribute 'nope'"
    assert _outcome(lambda: real_join.nope)[2] == "nope"

    # __new__ makes an empty message of any arguments, and __init__ sets the fields of its
    # keyword arguments over those that the message has.
    for cls in (ours("spark.connect.Relation"), upb("spark.connect.Relation")):
        made = cls.__new__(cls, 1, range={"end": 3})
        assert made.ListFields() == []
        made.__init__(common={"plan_id": 1})
        made.__init__(range={"end": 3})
        assert [f.name for f, _ in made.ListFields()] == ["common", "range"]
        assert _outcome(lambda: made.__init__(1)) == (
            "raise", "TypeError", "No positional arguments allowed"
        )


class _AllowList(pickle.Unpickler):
    """An unpickler of an allow-list of globals, as Python's documentation restricts them
    ("Restricting Globals"): those of pyspark and protobuf only, and the function of the bytes of
    the protocols before 3."""

    def find_class(self, module, name):
        if module.startswith(("pyspark.", "google.protobuf.")) or (module, name) == (
            "_codecs",
            "encode",
        ):
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"global '{module}.{name}' is forbidden")


def test_pickles_are_upbs_and_those_of_any_and_timestamp_load_in_pyspark(upb, tmp_path):
    # A message pickles as upb's, of protobuf's Message.__reduce__: of its class, or of protobuf's
    # _InternalConstructMessage of the full name of a nested type, which protobuf's symbol
    # database gives the class of. An Any and a Timestamp pickle as upb's of protobuf's classes
    # (google.protobuf.any_pb2.Any), which load where protobuf does. Without protobuf, they pickle
    # of the name that the module of the first proto of Spark Connect that imports their file
    # binds their class to, of protocols 4 and later
    # (pyspark.sql.connect.proto.base_pb2's google_dot_protobuf_dot_any__pb2.Any), which is
    # protobuf's class in PySpark, whose generated module binds protobuf's module there, and the
    # runtime's in the drop-in, with or without protobuf: of no other global, so that an
    # unpickler of an allow-list of modules loads them.
    from google.protobuf import message as protobuf_message

    for name in ("spark.connect.Relation", "spark.connect.Expression.Literal"):
        mine, real = ours(name)(), upb(name)()
        assert mine.__reduce__()[1:] == real.__reduce__()[1:]
        assert (mine.__reduce__()[0] is protobuf_message._InternalConstructMessage) == (
            real.__reduce__()[0] is protobuf_message._InternalConstructMessage
        )
    literal = proto.Expression.Literal(string="x")
    assert type(copy.copy(literal)) is type(literal) and copy.copy(literal) == literal
    relation = proto.Relation(range={"end": 3})
    extension = _rust.Any()
    extension.Pack(relation)
    stamp = _rust.Timestamp(seconds=5, nanos=6)
    upbs = [b"google.protobuf.any_pb2", b"google.protobuf.timestamp_pb2"]
    for message, module in zip((extension, stamp), upbs):
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            data = pickle.dumps(message, protocol)
            assert module in data and b"pyspark" not in data
            assert _AllowList(io.BytesIO(data)).load() == message

    # Without protobuf, of the names of the generated modules of the protos: PySpark's modules of
    # the protos, which bind protobuf's modules there, load them of protobuf's classes.
    script = textwrap.dedent(
        """
        import importlib.abc, json, pickle, sys

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "google.protobuf" or name.startswith("google.protobuf."):
                    raise ImportError("no protobuf")

        sys.meta_path.insert(0, Block())
        import pyspark.sql.connect.proto as proto
        from pyspark.sql.connect.proto import _rust

        extension = _rust.Any()
        extension.Pack(proto.Relation(range={"end": 3}))
        stamp = _rust.Timestamp(seconds=5, nanos=6)
        data = [pickle.dumps(m, protocol) for m in (extension, stamp) for protocol in (4, 5)]
        print(json.dumps([d.hex() for d in data]))
        """
    )
    root = str(pathlib.Path(__file__).resolve().parent.parent)
    dumped = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, cwd=root
    )
    assert dumped.returncode == 0, dumped.stderr[-3000:]
    data = [bytes.fromhex(d) for d in json.loads(dumped.stdout)]
    for d in data:
        assert b"pyspark.sql.connect.proto" in d and b"google_dot_protobuf_dot_" in d
        assert b"importlib" not in d and b"getattr" not in d
    assert [type(_AllowList(io.BytesIO(d)).load()) for d in data] == [_rust.Any] * 2 + [
        _rust.Timestamp
    ] * 2
    package = tmp_path / "pyspark" / "sql" / "connect" / "proto"
    package.mkdir(parents=True)
    for parent in [tmp_path / "pyspark", *package.parents][:3]:
        (parent / "__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "base_pb2.py").write_text(
        "from google.protobuf import any_pb2 as google_dot_protobuf_dot_any__pb2\n"
    )
    (package / "pipelines_pb2.py").write_text(
        "from google.protobuf import any_pb2 as google_dot_protobuf_dot_any__pb2\n"
        "from google.protobuf import timestamp_pb2 as google_dot_protobuf_dot_timestamp__pb2\n"
    )
    script = textwrap.dedent(
        """
        import json, pickle, sys
        from google.protobuf import any_pb2, timestamp_pb2

        loaded = [pickle.loads(bytes.fromhex(data)) for data in json.load(sys.stdin)]
        print(json.dumps([[type(m).__module__ + "." + type(m).__name__, m.SerializeToString().hex()]
                          for m in loaded]))
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps([d.hex() for d in data]),
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(tmp_path),
        env=dict(os.environ, PYTHONPATH=str(tmp_path)),
    )
    assert result.returncode == 0, result.stderr[-3000:]
    expected = [extension.SerializeToString().hex()] * 2 + [stamp.SerializeToString().hex()] * 2
    assert json.loads(result.stdout) == [
        [name, value]
        for name, value in zip(
            ["google.protobuf.any_pb2.Any"] * 2 + ["google.protobuf.timestamp_pb2.Timestamp"] * 2,
            expected,
        )
    ]


def _nested_limits_of(levels):
    """A Relation of `levels` limits, of the mappings of _nested_limits."""
    return _nested_limits(levels)


def _depth_cases(cls):
    """The operations whose message of 51 limits (102 levels below it) upb refuses, as it copies
    their message by a parse of its bytes, and those that it does not, of what each keeps."""
    relation = cls("spark.connect.Relation")
    deep = relation(**_nested_limits(51))
    fine = relation(**_nested_limits(50))
    literal = cls("spark.connect.Expression.Literal")
    any_of = cls("spark.connect.Relation")()
    any_of.extension.Pack(deep)

    def kept(container):
        return [m.SerializeToString(deterministic=True) for m in container]

    def into(make):
        def run():
            target = make()
            try:
                operation = target[1](target[0])
            except Exception as e:  # noqa: BLE001
                operation = (type(e).__name__, str(e))
            return operation, target[2](target[0])

        return run

    with_relations = cls("spark.connect.WithRelations")
    plan = cls("spark.connect.Plan")

    def message_bytes(message):
        return message.SerializeToString(deterministic=True)

    def references(message):
        return kept(message.references)

    return {
        "FromString": lambda: relation.FromString(deep.SerializeToString()),
        "pickle": lambda: pickle.loads(pickle.dumps(deep)),
        "copy.copy": lambda: copy.copy(deep),
        "Unpack": into(lambda: (relation(), any_of.extension.Unpack, message_bytes)),
        "MergeFrom": into(lambda: (relation(), lambda m: m.MergeFrom(deep), message_bytes)),
        "MergeFrom of a field": into(
            lambda: (plan(), lambda m: m.root.MergeFrom(deep), message_bytes)
        ),
        "constructor": lambda: plan(root=deep),
        "constructor of a repeated field": lambda: with_relations(references=[deep]),
        "append": into(
            lambda: (with_relations(), lambda m: m.references.append(deep), references)
        ),
        "extend": into(
            lambda: (with_relations(), lambda m: m.references.extend([fine, deep, fine]), references)
        ),
        "insert": into(
            lambda: (with_relations(), lambda m: m.references.insert(0, deep), references)
        ),
        "fine": lambda: relation.FromString(fine.SerializeToString()),
        "CopyFrom": lambda: plan().root.CopyFrom(deep),
        "deepcopy": lambda: copy.deepcopy(deep),
        "map value": lambda: cls("spark.connect.SQL")(
            named_arguments={"a": {"literal": {"integer": 1}}}
        ),
        "add": lambda: with_relations().references.add(**_nested_limits(51)),
        "SerializeToString": lambda: deep.SerializeToString(),
        "literal": lambda: literal(string="x"),
    }


def test_messages_of_more_than_100_levels_fail_where_upb_parses_their_bytes(upb):
    # As upb's: a parse of a message of more than 100 levels of messages below it fails, of upb's
    # error, keeping what it read: of FromString, of a pickle, of Unpack, and of the operations
    # that upb makes of a parse of the bytes of the message, MergeFrom, copy.copy, a constructor
    # given a message for a field or a repeated field, and append, extend and insert (whose
    # messages upb appends before the parse, so that a failed one stays, but of insert). CopyFrom,
    # deepcopy, the value of a map and the mappings of add copy any depth. The drop-in's own parse
    # of a message that PySpark builds in Python takes any depth.
    real_cases = _depth_cases(upb)
    for label, mine in _depth_cases(ours).items():
        if label == "pickle":
            # (upb's classes of the tests are of the names of the package's, so do not pickle.)
            real = lambda: upb("spark.connect.Relation").FromString(  # noqa: E731
                ours("spark.connect.Relation")(**_nested_limits(51)).SerializeToString()
            )
        else:
            real = real_cases[label]
        assert _outcome(mine) == _outcome(real), label
    deep = proto.Relation(**_nested_limits(51))
    assert _pyspark._proto_parse_deep(proto.Relation, deep.SerializeToString()) == deep
    message = _outcome(lambda: proto.Relation.FromString(deep.SerializeToString()))
    assert message[1:] == (
        "DecodeError",
        "Error parsing message with type 'spark.connect.Relation': Exceeded upb_DecodeOptions_MaxDepth",
    )
    from pyspark.sql.connect.types import pyspark_types_to_proto_types
    from pyspark.sql.types import IntegerType, StructField, StructType

    nested = IntegerType()
    for _ in range(60):
        nested = StructType([StructField("a", nested)])
    assert pyspark_types_to_proto_types(nested).struct.fields[0].name == "a"


def _corrupt(rng, data):
    """`data` corrupted: cut short, of a changed byte, of bytes inserted or of bytes removed."""
    data = bytearray(data)
    kind = rng.randint(0, 3)
    if kind == 0 and data:
        del data[rng.randint(0, len(data) - 1) :]
    elif kind == 1 and data:
        data[rng.randint(0, len(data) - 1)] = rng.randint(0, 255)
    elif kind == 2:
        at = rng.randint(0, len(data))
        data[at:at] = bytes(rng.randint(0, 255) for _ in range(rng.randint(1, 3)))
    elif data:
        at = rng.randint(0, len(data) - 1)
        del data[at : at + rng.randint(1, 4)]
    return bytes(data)


@pytest.mark.parametrize("seed", range(3))
def test_corrupt_bytes_fail_and_keep_what_upbs_parse_keeps(upb, seed):
    # As upb's decoder of its input stream: a value, a tag or a length that runs past the end of
    # its message reads the bytes that follow it in the data, and zeros past the end of the data,
    # and is set before the parse fails; a string reads past the end of its message but not of
    # the data, and a packed field of a fixed width up to 16 bytes past the data once its stream
    # is in its patch buffer; a field of presence is set before its bytes are read; the run of
    # fields of numbers of no field is kept at the next field of a number of the type, or at the
    # end of the message, so a failure before either drops it. Random messages of every type,
    # cut short, of a byte changed and of bytes inserted or removed, keep what upb's keep.
    def state(cls, data):
        message = cls()
        try:
            message.MergeFromString(data)
            failure = None
        except Exception as e:  # noqa: BLE001
            failure = (type(e).__name__, str(e))
        try:
            text = str(message)
        except Exception as e:  # noqa: BLE001
            text = type(e).__name__
        fields = [f.name for f, _ in message.ListFields()]
        return failure, fields, message.SerializeToString(deterministic=True), text

    generator = _Generator(upb, seed)
    rng = random.Random(seed + 1000)
    mismatches = []
    embedded = [
        name
        for name in sorted(_rust._CLASSES)
        if ours(name).DESCRIPTOR.file.name in _rust._EMBEDDED_FILES
        and not ours(name).DESCRIPTOR.GetOptions().map_entry
    ]
    for full_name in embedded:
        real = upb(full_name)()
        generator.fill(real, depth=generator.rng.randint(0, 3))
        data = real.SerializeToString(deterministic=True)
        for _ in range(2):
            bad = _corrupt(rng, data)
            if state(ours(full_name), bad) != state(upb(full_name), bad):
                mismatches.append((full_name, bad.hex()))
    assert not mismatches, mismatches[:5]


@pytest.fixture(scope="module")
def kinds(plugin):
    """The modules of the plugin's files of proto2, of the edition 2023 and of well-known types,
    and the class of upb of a message of their files by full name, of a pool of their own."""
    from google.protobuf import duration_pb2, empty_pb2, field_mask_pb2, struct_pb2, wrappers_pb2
    from plugin_protos import editions_pb2, legacy_pb2, wkt_pb2

    known = [duration_pb2, empty_pb2, field_mask_pb2, struct_pb2, wrappers_pb2]
    classes = _upb_classes_of(
        [
            (m.DESCRIPTOR.serialized_pb, m.__name__)
            for m in known + [legacy_pb2, editions_pb2, wkt_pb2]
        ]
    )
    return legacy_pb2, editions_pb2, wkt_pb2, classes.__getitem__


def _kind_cases(cls):
    """Operations on the messages of the plugin's files of proto2, editions and well-known types,
    of the classes of ``cls`` by full name: defaults and presence, required fields, closed enums,
    packing, groups, strings of no UTF-8, the features of editions, and the methods, assignments,
    comparisons and membership of the well-known types."""
    from google.protobuf import json_format, text_format

    defaults, legacy = cls("thirdparty.legacy.Defaults"), cls("thirdparty.legacy.Legacy")
    required = cls("thirdparty.legacy.Required")
    featured, wkt = cls("thirdparty.editions.Featured"), cls("thirdparty.wkt.Wkt")
    names = ["count", "name", "data", "ratio", "scale", "flag", "color", "first_color", "big",
             "negative", "plain", "plain_name"]  # fmt: skip

    def nested():
        message = legacy()
        message.required.next.next.SetInParent()
        message.many_required.add(a=1)
        message.many_required.add()
        message.item.x = 1
        message.by_name["k"].SetInParent()
        return message

    def groups():
        message = legacy(id="x")
        message.item.x, message.item.label = 7, "l"
        message.entry.add(key=1).value.sql.query = "q"
        return message

    def featured_set():
        message = featured(required="r", packed=[1, 2], expanded=[3, 4], implicit=0, explicit=0)
        message.delimited.value = 5
        message.delimited_values.add(value=6)
        return message

    def struct():
        return wkt(options={"a": 1, "b": "x"}, list=[1, 2])

    errors = []
    unknown_enums = bytes.fromhex("0a0178280930093001" "3a03010902")
    bad_utf8 = bytes.fromhex("0a01ff" "a20103ff6162")
    return [
        lambda: [repr(getattr(defaults(), n)) for n in names],
        lambda: [defaults().HasField(n) for n in names],
        lambda: defaults(count=5, plain=0, name="", color=1),
        lambda: defaults(count=5) == defaults(),
        lambda: legacy().IsInitialized(),
        lambda: nested().FindInitializationErrors(),
        lambda: nested().SerializeToString(),
        lambda: nested().ByteSize(),
        lambda: nested().SerializePartialToString(),
        lambda: (nested().IsInitialized(errors), errors),
        lambda: nested().IsInitialized(5),
        lambda: legacy.FromString(b""),
        lambda: cls("spark.connect.Relation")().extension.Pack(required()),
        lambda: legacy(color=9),
        lambda: legacy(colors=[1, 9]),
        lambda: legacy(id="x").colors.extend([1, 9]),
        lambda: legacy(color="PURPLE"),
        lambda: legacy.FromString(unknown_enums),
        lambda: list(legacy.FromString(unknown_enums).packed_colors),
        lambda: legacy.FromString(bytes.fromhex("0a0178" "9801" "09")).WhichOneof("choice"),
        lambda: legacy(id="x", unpacked=[1, 2], packed=[3, 4]),
        lambda: list(legacy.FromString(bytes.fromhex("0a01781a020102")).unpacked),
        lambda: groups(),
        lambda: str(groups()),
        lambda: text_format.MessageToString(groups()),
        lambda: json_format.MessageToJson(groups()),
        lambda: legacy.FromString(groups().SerializeToString()) == groups(),
        lambda: str(legacy.FromString(bytes.fromhex("0a0178" "4202" "4801"))),
        lambda: legacy.FromString(bytes.fromhex("0a0178" "43" "4801")),
        lambda: legacy.FromString(bytes.fromhex("0a0178" "43" "4801" "5c")),
        lambda: text_format.Parse('id: "x" Item { x: 3 label: "q" }', legacy()),
        lambda: legacy(id="x", input=cls("spark.connect.Relation")(sql={"query": "q"})),
        lambda: [(f.name, v) for f, v in legacy.FromString(bad_utf8).ListFields()],
        lambda: (str(legacy.FromString(bad_utf8)), legacy.FromString(bad_utf8).SerializeToString()),
        lambda: json_format.MessageToJson(legacy.FromString(bad_utf8)),
        lambda: setattr(legacy(), "raw", b"\xff"),
        lambda: featured().HasField("implicit"),
        lambda: (featured().explicit, featured().HasField("explicit"), featured().closed),
        lambda: featured().FindInitializationErrors(),
        lambda: featured_set(),
        lambda: str(featured_set()),
        lambda: json_format.MessageToJson(featured_set()),
        lambda: setattr(featured(), "closed", 7),
        lambda: str(featured.FromString(bytes.fromhex("1a017230073807"))),
        lambda: featured.FromString(bytes.fromhex("1a017262" "01ff")).unchecked,
        lambda: featured.FromString(bytes.fromhex("1a01ff")),
        lambda: wkt(timeout=datetime.timedelta(seconds=3, microseconds=5)).timeout.ToJsonString(),
        lambda: str(wkt(options={"a": 1, "b": [1, "x"], "c": {"d": None}})),
        lambda: wkt(options={"fields": {"a": {"number_value": 1}}}),
        lambda: wkt(list=[1, "a", None, True]),
        lambda: wkt(at=datetime.datetime(2020, 1, 2)),
        lambda: wkt(mask={"paths": ["a.b"]}).mask.ToJsonString(),
        lambda: [value in struct().options for value in ("a", "z")]
        + [value in struct().list for value in (1, 3)],
        lambda: (struct().options == {"a": 1, "b": "x"}, struct().options != {"a": 1, "b": "x"}),
        lambda: (struct().list == [1, 2], struct().list != [1, 2], struct().list == [2]),
        lambda: sorted(struct().options.items()),
        lambda: setattr(struct(), "value", 3),
        lambda: struct().timeouts.append(datetime.timedelta(seconds=1)),
        lambda: json.loads(json_format.MessageToJson(struct())),
        lambda: wkt(timeout={"seconds": 1}).timeout + datetime.timedelta(seconds=1),
        lambda: wkt(input=cls("spark.connect.Relation")(sql={"query": "q"})),
        lambda: (wkt(nothing={}).HasField("nothing"), str(wkt(nothing={}))),
    ]


def test_messages_of_files_of_proto2_editions_and_well_known_types_are_upbs(kinds):
    # The runtime's classes of the plugin's files of proto2, of editions and of well-known types
    # other than Any and Timestamp behave as upb's: the defaults and presence of proto2 and of the
    # features of editions, required fields (IsInitialized, FindInitializationErrors and the
    # EncodeError of SerializeToString), closed enums (whose unknown values a parse keeps in the
    # unknown fields), packing, groups and delimited messages, strings that a parse does not check
    # (of bytes that are not UTF-8), and protobuf's methods of the well-known types, of upb's
    # assignments, comparisons and membership of a Struct and a ListValue. A class of a well-known
    # type is the runtime's own, which errors name. A map pickles and serializes of the order of
    # its keys, as upb's deterministic serialization.
    legacy_pb2, editions_pb2, wkt_pb2, upb = kinds
    for module in (legacy_pb2, editions_pb2, wkt_pb2):
        for name, cls in vars(module).items():
            if isinstance(cls, type) and hasattr(cls, "DESCRIPTOR"):
                assert cls is ours(cls.DESCRIPTOR.full_name), name

    def normalized(outcome):
        text = repr(outcome)
        for stem in ("struct", "duration", "wrappers", "field_mask", "empty"):
            ours_module = f"pyspark.sql.connect.proto._rust.{stem}_pb2"
            text = text.replace(ours_module, f"google.protobuf.{stem}_pb2")
        return text

    mine, real = _kind_cases(ours), _kind_cases(upb)
    for at, (m, r) in enumerate(zip(mine, real)):
        assert normalized(_outcome(m)) == normalized(_outcome(r)), at
    struct = wkt_pb2.Wkt(options={"a": 1, "b": "x"}, list=[1, 2])
    assert pickle.loads(pickle.dumps(struct)) == struct
    timeout = wkt_pb2.Wkt(timeout={"seconds": 1}).timeout
    assert pickle.dumps(timeout).count(b"google.protobuf.duration_pb2") == 1
    assert type(pickle.loads(pickle.dumps(timeout))).__module__ == "google.protobuf.duration_pb2"
    assert type(copy.copy(timeout)) is type(timeout)


def test_modules_built_before_the_package_get_its_classes():
    # A module of a proto of Spark Connect or of a plugin's file that imports them, which
    # protobuf's builder built before pyspark.sql.connect.proto was imported (as only a copy of
    # Apache Spark's generated modules of the same protos can), gets the classes of the package,
    # as upb gives the module of a file built again the class that it made of it, or the
    # runtime's, as do the names that other modules bound to its classes. (Of protobuf's builder.)
    pytest.importorskip("google.protobuf")
    script = textwrap.dedent(
        """
        import sys, types
        from google.protobuf import descriptor_pb2, descriptor_pool
        from google.protobuf.internal import builder
        from pyspark import _pyspark

        pool = descriptor_pool.Default()
        for name, _, _, _ in _pyspark._proto_files():
            if name.startswith("google/"):
                __import__("google.protobuf." + name.rsplit("/", 1)[1][:-6] + "_pb2")
            else:
                pool.AddSerializedFile(_pyspark._proto_file_descriptor(name).serialized_pb)

        def build(file, name):
            module = types.ModuleType(name)
            module.DESCRIPTOR = file
            builder.BuildMessageAndEnumDescriptors(file, vars(module))
            builder.BuildTopDescriptorsAndMessages(file, name, vars(module))
            sys.modules[name] = module
            return module

        copy = build(pool.FindFileByName("spark/connect/relations.proto"), "copy.relations_pb2")
        plugin = descriptor_pb2.FileDescriptorProto(
            name="early/plugin.proto", package="early", syntax="proto3",
            dependency=["spark/connect/relations.proto"])
        field = plugin.message_type.add(name="M").field.add(
            name="input", number=1, type=11, label=1, type_name=".spark.connect.Relation")
        early = build(pool.AddSerializedFile(plugin.SerializeToString()), "early.plugin_pb2")
        bound = types.ModuleType("early.bound")
        bound.M = early.M
        sys.modules["early.bound"] = bound
        before = (issubclass(copy.Relation, _pyspark.ProtoMessage), issubclass(early.M, _pyspark.ProtoMessage))

        import pyspark.sql.connect.proto as proto
        after = [copy.Relation is proto.Relation, issubclass(early.M, _pyspark.ProtoMessage),
                 bound.M is early.M, type(early.M(input=proto.Relation()).input) is proto.Relation]
        print(before, after)
        """
    )
    root = str(pathlib.Path(__file__).resolve().parent.parent)
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120, cwd=root
    )
    assert result.returncode == 0, result.stderr[-3000:]
    assert result.stdout.strip() == "(False, False) [True, True, True, True]"


def test_modules_of_the_messages_have_files_for_type_checkers():
    # Each module of messages, of a stub of PySpark (base_pb2.pyi), has a file next to it, which
    # pyright takes as its source (with no reportMissingModuleSource), as it takes PySpark's
    # generated module: the runtime's finder gives the module that the runtime makes, of the file
    # as its origin, rather than the file, which holds only its docstring.
    import ast

    package = pathlib.Path(proto.__file__).parent
    for stub in sorted(package.glob("*_pb2.pyi")):
        source = stub.with_suffix(".py")
        assert source.exists(), source
        module = importlib.import_module(f"pyspark.sql.connect.proto.{stub.stem}")
        assert module.__file__ == str(source) and module.__spec__.loader is _rust._LOADER
        assert module.DESCRIPTOR.name == f"spark/connect/{stub.stem[: -len('_pb2')]}.proto"
        body = ast.parse(source.read_text()).body
        assert len(body) == 1 and isinstance(body[0].value, ast.Constant), source


def test_the_runtimes_own_base_and_descriptors_where_protobuf_does_not_import_are_upbs(upb):
    # Where protobuf does not import, the classes are of the runtime's base _MessageBase, in place
    # of protobuf's Message, of its methods: the pickle of a message, of its class and its bytes,
    # its copy and its dir(); and of the runtime's descriptors, which _descriptor_of and the others
    # give where the classes do not take protobuf's (as where protobuf's pool has other files of the
    # names of the protos, which _same_files tells, of their shapes).
    from google.protobuf import descriptor_pb2, descriptor_pool, reflection

    relation = _own_class("spark.connect.Relation")
    message = relation(sql={"query": "q"})
    # ListFields gives the descriptors of the fields of the DESCRIPTOR of the class of the type,
    # protobuf's, read of the class once.
    listed = [proto.Relation(sql={"query": "q"}).ListFields()[0][0] for _ in range(2)]
    assert listed[0] is listed[1] is proto.Relation.DESCRIPTOR.fields_by_name["sql"]
    reduced = message.__reduce__()
    assert reduced[0] is relation and reduced[1] == () and reduced[2] == {
        "serialized": message.SerializeToString()
    }
    restored = relation()
    restored.__setstate__({"serialized": message.SerializeToString().decode("latin1")})
    assert restored == message and copy.copy(message) == message
    assert type(copy.copy(message)) is relation
    assert set(dir(message)) - {"__copy__"} == set(
        dir(upb("spark.connect.Relation")(sql={"query": "q"}))
    )

    import pyspark.sql.connect.proto._rust as runtime

    saved = runtime._USE_PROTOBUF_DESCRIPTORS
    runtime._USE_PROTOBUF_DESCRIPTORS = False
    try:
        assert runtime._descriptor_of("spark.connect.Relation") is _own("spark.connect.Relation")
        enum = runtime._enum_descriptor_of("spark.connect.Join.JoinType")
        assert enum is _pyspark._proto_enum_descriptor("spark.connect.Join.JoinType")
        file = runtime._file_descriptor_of("spark/connect/base.proto")
        assert file is _pyspark._proto_file_descriptor("spark/connect/base.proto")
        assert runtime._class_for(proto.Relation.DESCRIPTOR, None) is None
    finally:
        runtime._USE_PROTOBUF_DESCRIPTORS = saved

    # The files of protobuf's pool of the names of the protos are the runtime's of the same bytes,
    # or of the same shapes of other bytes (of JSON names), and no other.
    name = "spark/connect/common.proto"
    same = descriptor_pb2.FileDescriptorProto.FromString(
        _pyspark._proto_file_descriptor(name).serialized_pb
    )
    for message_proto in same.message_type:
        for field in message_proto.field:
            field.json_name = field.json_name or field.name
    other = descriptor_pb2.FileDescriptorProto()
    other.CopyFrom(same)
    other.message_type[0].field[0].number += 1000
    pool = descriptor_pool.DescriptorPool()
    for dependency in same.dependency:
        pool.AddSerializedFile(_pyspark._proto_file_descriptor(dependency).serialized_pb)
    assert runtime._same_files(pool.AddSerializedFile(same.SerializeToString()), name)
    other_pool = descriptor_pool.DescriptorPool()
    for dependency in other.dependency:
        other_pool.AddSerializedFile(_pyspark._proto_file_descriptor(dependency).serialized_pb)
    assert not runtime._same_files(other_pool.AddSerializedFile(other.SerializeToString()), name)

    # The hooks pass over what is not a descriptor of protobuf's default pool, and an object of
    # sys.modules of no globals.
    assert runtime._class_for(object(), None) is None
    assert issubclass(type(upb("spark.connect.Relation")), reflection.GeneratedProtocolMessageType) is (
        issubclass(type(upb("spark.connect.Relation")), reflection.GeneratedProtocolMessageType._protobuf_class)
    )
    sys.modules["tests_of_no_globals"] = 1  # type: ignore[assignment]
    try:
        assert runtime._imported_module_of_file(proto.Relation.DESCRIPTOR.file) is not None
        runtime._rebind({id(object()): (object(), object())}, "tests_of_no_globals")
    finally:
        del sys.modules["tests_of_no_globals"]


def test_holds_of_files_find_the_modules_of_a_package_once_and_rebind_them_at_their_end(upb):
    # A hold of files (of the take-overs of many generated modules) finds the imported modules of a
    # top-level package once, and rebinds the names of its modules once, at its end, to the
    # runtime's classes that replaced protobuf's: of the package only, and of no other name.
    import pyspark.sql.connect.proto._rust as runtime

    old, new, other = type("Old", (), {}), type("New", (), {}), type("Other", (), {})
    names = ["tests_hold_pkg", "tests_hold_pkg.sub", "tests_hold_pkgx", "tests_hold_pkg.late"]
    modules = {name: types.ModuleType(name) for name in names}
    for module in modules.values():
        module.Old, module.Other = old, other
    for name in names[:3]:
        sys.modules[name] = modules[name]
    try:
        with runtime._held_together():
            found = runtime._modules_of_packages({"tests_hold_pkg"})
            assert {m.__name__ for m in found} == {"tests_hold_pkg", "tests_hold_pkg.sub"}
            with runtime._held_together():  # a hold in a hold is the one hold
                runtime._rebind({id(old): (old, new)}, "tests_hold_pkg.sub")
            sys.modules["tests_hold_pkg.late"] = modules["tests_hold_pkg.late"]
            assert runtime._modules_of_packages({"tests_hold_pkg"}) == found
            assert modules["tests_hold_pkg"].Old is old
        assert [modules[name].Old for name in names] == [new, new, old, old]
        assert {m.Other for m in modules.values()} == {other}
        assert len(runtime._modules_of_packages({"tests_hold_pkg"})) == 3
        runtime._rebind({id(old): (old, new)}, "tests_hold_pkg.late")
        assert modules["tests_hold_pkg.late"].Old is new and modules["tests_hold_pkgx"].Old is old
        assert runtime._modules_of_packages(set()) == []
    finally:
        for name in names:
            sys.modules.pop(name, None)


def _run_script(script, **env):
    """The JSON that ``script`` prints of a Python of the drop-in, of the environment of this one
    and of ``env`` (without ``PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION`` unless it is given)."""
    root = str(pathlib.Path(__file__).resolve().parent.parent)
    environment = {k: v for k, v in os.environ.items() if k != "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"}
    environment.update(env)
    result = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=180,
        cwd=root,
        env=environment,
    )
    assert result.returncode == 0, result.stderr[-3000:]
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_protobufs_classes_stay_those_of_its_descriptors_of_its_pure_python_implementation(upb):
    # Of protobuf's pure-Python implementation, whose descriptors hold the class of their type
    # (_concrete_class), which its containers and message_factory read: the runtime sets it of a
    # descriptor of a type that it holds of no class, as protobuf's metaclass sets it, and never in
    # place of protobuf's class of a descriptor, of the well-known types and of the files that a
    # plugin's file imports, so protobuf's messages of other modules make and take theirs.
    out = _run_script(
        """
        import json
        from google.protobuf import any_pb2, duration_pb2, message_factory, struct_pb2
        from google.protobuf import timestamp_pb2
        from google.protobuf.internal import api_implementation

        import pyspark.sql.connect.proto as proto
        from plugin_protos import wkt_pb2

        out = [api_implementation.Type()]
        out += [m.DESCRIPTOR._concrete_class is m for m in (any_pb2.Any, timestamp_pb2.Timestamp,
                                                             duration_pb2.Duration, struct_pb2.Struct)]
        out.append(message_factory.GetMessageClass(any_pb2.Any.DESCRIPTOR) is any_pb2.Any)
        out.append(proto.Relation.DESCRIPTOR._concrete_class is proto.Relation)
        out.append(wkt_pb2.Wkt.DESCRIPTOR._concrete_class is wkt_pb2.Wkt)
        value = struct_pb2.Value()
        value.struct_value.fields["a"].number_value = 1
        listed = struct_pb2.ListValue()
        listed.values.add().number_value = 2
        out.append([type(value.struct_value) is struct_pb2.Struct, listed.values[0].number_value])
        try:
            from google.rpc import status_pb2
        except ImportError:
            out.append(True)
        else:
            status = status_pb2.Status()
            out.append(type(status.details.add()) is any_pb2.Any)
        print(json.dumps(out))
        """,
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python",
        PYTHONPATH=str(pathlib.Path(__file__).resolve().parent),
    )
    assert out == ["python", True, True, True, True, True, True, True, [True, 2.0], True]


def test_maps_of_strings_of_no_utf8_validation_take_bytes_that_are_not_utf8_as_upbs(kinds):
    # A map of proto2 (whose entries take any bytes of a string) and one of the NONE validation of
    # editions parse a key or a value that is not UTF-8, as upb's do, which then gives it as
    # bytes; one of the VERIFY validation refuses it.
    legacy_pb2, editions_pb2, _, upb = kinds

    def outcome(cls, data):
        message = cls()
        failure = _outcome(lambda: message.MergeFromString(data))
        keys = {name: list(getattr(message, name)) for name in ("by_name", "names", "checked_names")
                if name in cls.DESCRIPTOR.fields_by_name}
        return failure[:2], message.SerializePartialToString(), keys, str(message)

    cases = [
        (legacy_pb2.Legacy, "thirdparty.legacy.Legacy", bytes.fromhex("7a030a01ff")),
        (editions_pb2.Strict, "thirdparty.editions.Strict", bytes.fromhex("12060a01ff120161")),
        (editions_pb2.Strict, "thirdparty.editions.Strict", bytes.fromhex("12060a016112016c") + b"\xff"),
        (editions_pb2.Strict, "thirdparty.editions.Strict", bytes.fromhex("12060a016112") + b"\x01\xff"),
        (editions_pb2.Strict, "thirdparty.editions.Strict", bytes.fromhex("1a060a01ff120161")),
    ]
    for cls, full_name, data in cases:
        assert outcome(cls, data) == outcome(upb(full_name), data), data.hex()
    parsed = legacy_pb2.Legacy.FromString(bytes.fromhex("7a030a01ff"))
    assert list(parsed.by_name) == [b"\xff"]
    with pytest.raises(Exception, match="String field had bad UTF-8"):
        editions_pb2.Strict.FromString(bytes.fromhex("1a060a01ff120161"))


def test_required_message_fields_of_editions_are_required_as_upbs(kinds):
    # A message field of the LEGACY_REQUIRED presence of editions is required, as upb's: a message
    # that lacks it is not initialized and does not serialize.
    _, editions_pb2, _, upb = kinds
    for cls in (editions_pb2.Strict, upb("thirdparty.editions.Strict")):
        message = cls()
        assert message.FindInitializationErrors() == ["inner"] and not message.IsInitialized()
        assert _outcome(message.SerializeToString)[2] == (
            "Message thirdparty.editions.Strict is missing required fields: inner"
        )
        message.inner.value = 1
        assert message.IsInitialized() and message.SerializeToString() == b"\x0a\x02\x08\x01"


def _parse_outcome(cls, data):
    """What a merge of ``data`` into a message of ``cls`` gives: its failure, the bytes of the
    message (of its required fields too) and its text."""
    message = cls()
    failure = _outcome(lambda: message.MergeFromString(data))
    return failure[:3] if failure[0] == "raise" else None, message.SerializePartialToString(), str(
        message
    )


def test_the_decoder_keeps_and_refuses_what_upbs_does(kinds):
    # As upb's decoder: a value of a closed enum that it does not declare is kept in the unknown
    # fields of the bytes of its field as they are (of a packed field, of a tag of its own and the
    # 64 bits of the value), and not kept when it is read past the end of its message; a length of
    # more than five bytes is corrupt, even of a small value; a field of the number 0 is an unknown
    # field of a message type of no fields; a string field that is not UTF-8 prints its UTF-8
    # characters and the escapes of the other bytes; a group past the limit of the levels is
    # corrupt ("Wire format was corrupt").
    legacy_pb2, editions_pb2, _, upb = kinds
    negative = b"\xff" * 9 + b"\x01"
    legacy = "thirdparty.legacy.Legacy"
    defaults = "thirdparty.legacy.Defaults"
    cases = [
        (legacy, b"\x28" + negative),  # color = -1, 10 bytes
        (legacy, b"\x30" + negative),  # colors (repeated) = -1
        (legacy, b"\xa8\x80\x00\x33"),  # color = 51, of a tag of 3 bytes
        (legacy, b"\x28\xb3\x80\x00"),  # color = 51, of 3 bytes
        (legacy, b"\x3a\x0b" + negative + b"\x01"),  # packed_colors = [-1, 1]
        (legacy, b"\x3a\x05\xb3\x80\x80\x80\x10"),  # packed_colors = [2^32 + 51]
        (legacy, b"\x7a\x0e\x0a\x01k\x10" + negative),  # by_name's value, a message, of a varint
        (defaults, b"\x40"),  # first_color, read past the end
        (defaults, b"\x08\x01\x40"),
        (defaults, b"\x40\x81"),  # first_color = 1, cut short
        (legacy, b"\x72\x01\x40\x05"),  # defaults.first_color = 5, read past its message
        (legacy, b"\x72\x01\x40\x02"),  # defaults.first_color = 2, read past its message
        (legacy, b"\x28\x80"),
        ("spark.connect.Relation", b"\x1a\x80\x80\x80\x80\x80\x00"),
        ("spark.connect.RelationCommon", b"\x0a\x80\x80\x80\x80\x80\x00"),
        ("spark.connect.AnalyzePlanRequest.Explain", b"\x0a\x80\x80\x80\x80\x80"),
        ("spark.connect.Relation", b"\x1a\x80\x80\x80\x80\x00"),
        ("spark.connect.Relation", b"\xfa\x3e\x80\x80\x80\x80\x80\x00"),
        ("spark.connect.ExecutePlanResponse.ResultComplete", b"\x00\x00"),
        ("spark.connect.ExecutePlanResponse.ResultComplete", b"\x05\x01\x02\x03\x04\x02\x01\x00"),
        ("spark.connect.ExecutePlanResponse.ResultComplete", b"\x03\x00"),
        # Groups of the number 0 in a message type of no fields, which are unknown fields.
        ("spark.connect.ExecutePlanResponse.ResultComplete", b"\x03\x04"),
        ("spark.connect.ExecutePlanResponse.ResultComplete", bytes.fromhex("03080104")),
        ("spark.connect.ExecutePlanResponse.ResultComplete", bytes.fromhex("03030404")),
        ("spark.connect.ExecutePlanResponse.ResultComplete", bytes.fromhex("030b0c04")),
        ("spark.connect.ExecutePlanResponse.ResultComplete", b"\x04"),
        ("spark.connect.CurrentDatabase", b"\x03\x04"),
        ("spark.connect.ExecutePlanResponse", bytes.fromhex("72020304")),
        ("spark.connect.ExecutePlanResponse", b"\x72\x02\x00\x00"),
        ("spark.connect.Relation", b"\x00\x00"),
        ("thirdparty.legacy.Legacy.Item", bytes.fromhex("520668c3a96cc66f")),
        ("spark.connect.Relation", b"\xc3\x3e" * 100 + b"\xc4\x3e" * 100),
        ("spark.connect.Relation", b"\xc3\x3e" * 101 + b"\xc4\x3e" * 101),
        # Undeclared values of a packed closed enum read past the end of the packed values, which
        # are kept.
        (legacy, b"\x3a\x02\x29\x94\x02"),
        ("thirdparty.editions.Featured", bytes.fromhex("3a01e697a5e69cace8aa9e01")),
        # A map entry of an unknown field, of a key of the default that it read (of proto2 and of
        # editions, of entry fields of presence, written again), and of proto3 (not written).
        (legacy, bytes.fromhex("7a0d0a001200212be6708b68120000")),
        ("thirdparty.editions.Strict", bytes.fromhex("12060a0012001805")),
        ("spark.connect.Read.DataSource", bytes.fromhex("1a060a0012001805")),
    ]
    mine = {
        defaults: legacy_pb2.Defaults,
        legacy: legacy_pb2.Legacy,
        "thirdparty.legacy.Legacy.Item": legacy_pb2.Legacy.Item,
        "thirdparty.editions.Featured": editions_pb2.Featured,
        "thirdparty.editions.Strict": editions_pb2.Strict,
    }
    for full_name, data in cases:
        cls = mine.get(full_name) or ours(full_name)
        assert _parse_outcome(cls, data) == _parse_outcome(upb(full_name), data), (full_name, data.hex())
    item = legacy_pb2.Legacy.Item.FromString(bytes.fromhex("520668c3a96cc66f"))
    assert str(item) == 'label: "h\u00e9l\\306o"\n'
    assert _parse_outcome(ours("spark.connect.Relation"), b"\xc3\x3e" * 101 + b"\xc4\x3e" * 101)[0][2] == (
        "Error parsing message with type 'spark.connect.Relation': Wire format was corrupt"
    )
    assert legacy_pb2.Legacy.FromString(b"\x28" + negative).SerializePartialToString() == b"\x28" + negative
    # A key of a map that is not UTF-8, which the runtime holds as bytes, sorts among the others of
    # its bytes, of the text and of a deterministic serialization, as upb's.
    data = bytes.fromhex("7a050a016112007a060a02fc7f1200")
    mine_parsed, real_parsed = legacy_pb2.Legacy.FromString(data), upb(legacy).FromString(data)
    assert set(mine_parsed.by_name) == set(real_parsed.by_name) == {"a", b"\xfc\x7f"}
    assert str(mine_parsed) == str(real_parsed)
    assert mine_parsed.SerializePartialToString(deterministic=True) == (
        real_parsed.SerializePartialToString(deterministic=True)
    )


def test_copy_from_and_eq_of_protobufs_messages_take_any_depth_as_upbs(upb):
    # CopyFrom of protobuf's message of the type of a message, and ==, copy and compare messages of
    # any depth, as upb's deep copy and comparison do, where a parse of their bytes stops at 100
    # levels.
    from google.protobuf import struct_pb2

    deep = struct_pb2.Struct()
    inner = deep
    for _ in range(60):
        inner = inner.fields["a"].struct_value
    inner.fields["b"].number_value = 1
    shadow = _rust._CLASSES.get("google.protobuf.Struct")
    if shadow is None:  # the runtime holds Struct once a plugin's file imports it
        pytest.importorskip("plugin_protos.wkt_pb2")
        shadow = _rust._CLASSES["google.protobuf.Struct"]
    held = shadow()
    held.CopyFrom(deep)
    assert held.SerializeToString(deterministic=True) == deep.SerializeToString(deterministic=True)
    assert held == deep and deep == held
    relation_class = _protobuf_class("spark.connect.Relation")
    relation = relation_class()
    top = relation
    for _ in range(120):
        relation = relation.limit.input
    relation.range.end = 3
    mine = proto.Relation()
    mine.CopyFrom(top)
    assert mine.SerializeToString() == top.SerializeToString() and mine == top and top == mine
    with pytest.raises(Exception, match="Exceeded upb_DecodeOptions_MaxDepth"):
        proto.Relation(limit={"input": top})


def test_parses_follow_upbs_set_allow_oversize_protos(upb):
    # Once upb's SetAllowOversizeProtos(True) is called, before the package imports or after, upb's
    # parsers read messages of up to 65535 levels, and so do the runtime's, of 100 levels before.
    script = """
        import json, sys
        from google.protobuf.internal import api_implementation

        before = len(sys.argv) > 1
        if "{when}" == "before":
            api_implementation._c_module.SetAllowOversizeProtos(True)
        import pyspark.sql.connect.proto as proto
        from pyspark import _pyspark

        data = b""
        for _ in range(150):
            data = bytes([0x42, len(data) + 2 if len(data) < 126 else 0]) if False else data
        deep = proto.Relation(range={{"end": 1}})
        for _ in range(150):
            outer = proto.Relation()
            outer.limit.input.CopyFrom(deep)
            deep = outer
        data = deep.SerializeToString()
        out = []
        def parse():
            try:
                return len(proto.Relation.FromString(data).SerializeToString())
            except Exception as e:
                return type(e).__name__
        if "{when}" == "after":
            out.append(parse())
            api_implementation._c_module.SetAllowOversizeProtos(True)
        out.append(parse())
        api_implementation._c_module.SetAllowOversizeProtos(False)
        out.append(parse())
        print(json.dumps(out))
        """
    assert _run_script(script.format(when="before")) == [_relation_size(150), "DecodeError"]
    assert _run_script(script.format(when="after")) == [
        "DecodeError",
        _relation_size(150),
        "DecodeError",
    ]
    # Of that limit, a copy of a message of tens of thousands of levels (copy.copy walks its levels
    # to tell whether upb would parse it) runs on a stack that grows, in a thread of a small stack
    # too, where upb's parser can end the process.
    deep = _run_script(
        """
        import copy, json, threading
        from google.protobuf.internal import api_implementation
        import pyspark.sql.connect.proto as proto
        from pyspark import _pyspark

        def varint(value):
            out = bytearray()
            while value > 0x7F:
                out.append((value & 0x7F) | 0x80)
                value >>= 7
            out.append(value)
            return bytes(out)

        api_implementation._c_module.SetAllowOversizeProtos(True)
        size, prefixes = 0, []
        for _ in range(30000):
            # Relation.limit (8) of a Limit, whose input (1) is the Relation below.
            limit = b"\\x0a" + varint(size)
            size += len(limit)
            relation = b"\\x42" + varint(size)
            size += len(relation)
            prefixes.append(relation + limit)
        data = b"".join(reversed(prefixes))
        message = _pyspark._proto_parse_deep(proto.Relation, data)
        out = []

        def run():
            out.append(len(copy.copy(message).SerializeToString()) == len(data))
            out.append(len(proto.Plan(root=message).SerializeToString()) > len(data))

        threading.stack_size(256 * 1024)
        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        print(json.dumps(out))
        """
    )
    assert deep == [True, True]


def _relation_size(levels):
    """The size of a Relation of ``levels`` limits over a range of end 1."""
    deep = proto.Relation(range={"end": 1})
    for _ in range(levels):
        outer = proto.Relation()
        outer.limit.input.CopyFrom(deep)
        deep = outer
    return len(deep.SerializeToString())


def test_lazy_modules_of_sys_modules_stay_lazy(upb):
    # The package and the hooks read the globals of the modules of sys.modules without their
    # __getattribute__, so that a module of importlib.util.LazyLoader is not loaded (its error, of
    # an import that fails, is raised at its first use, as without the drop-in).
    tests_dir = pathlib.Path(__file__).resolve().parent
    out = _run_script(
        """
        import importlib.abc, importlib.machinery, importlib.util, json, sys

        class Source(importlib.abc.SourceLoader):
            def get_filename(self, fullname):
                return "<broken_lazy>"

            def get_data(self, path):
                return b"import this_module_does_not_exist_of_the_drop_in\\nX = 1\\n"

        spec = importlib.machinery.ModuleSpec("broken_lazy", importlib.util.LazyLoader(Source()))
        module = importlib.util.module_from_spec(spec)
        sys.modules["broken_lazy"] = module
        spec.loader.exec_module(module)
        import pyspark.sql.connect.proto
        from plugin_protos import plugin_pb2

        lazy = type(sys.modules["broken_lazy"]).__name__
        try:
            module.X
            first_use = None
        except Exception as error:
            first_use = type(error).__name__
        print(json.dumps([lazy, first_use]))
        """,
        PYTHONPATH=str(tests_dir),
    )
    assert out == ["_LazyModule", "ModuleNotFoundError"]


def test_classes_bind_the_attributes_of_upbs_metaclass_when_they_are_made():
    # The attributes that upb's metaclass binds when they are read (the wrappers of nested enums,
    # the numbers of their values and of the fields) are bound when the class is made, so that the
    # metaclass has no __getattr__, which would slow every read of an attribute of a class.
    out = _run_script(
        """
        import copy, json
        import pyspark.sql.connect.proto as proto

        join = vars(proto.Join)
        print(json.dumps([
            "__getattr__" in vars(type(proto.Relation)),
            "READ_FIELD_NUMBER" in vars(proto.Relation),
            "JOIN_TYPE_INNER" in join and "JoinType" in join and "LEFT_FIELD_NUMBER" in join,
            "__copy__" in vars(proto.Relation.__mro__[1]),
        ]))
        """
    )
    assert out == [False, True, True, True]


def test_lookups_of_types_that_keep_protobufs_classes_are_known_once(upb):
    # message_factory.GetMessageClass of a type that keeps protobuf's class looks it up once: the
    # type is known after (of _NOT_OWNED), until the runtime takes a file over. Whether a file
    # imports the protos of Spark Connect is known of each file once.
    from google.protobuf import duration_pb2, message_factory

    descriptor = duration_pb2.Duration.DESCRIPTOR
    assert message_factory.GetMessageClass(descriptor) is duration_pb2.Duration
    assert descriptor in _rust._NOT_OWNED
    assert message_factory.GetMessageClass(descriptor) is duration_pb2.Duration
    assert message_factory.GetMessageClass(proto.Relation.DESCRIPTOR) is proto.Relation
    assert _rust._IMPORTS_HELD.get(descriptor.file.name) is None or not (
        _rust._IMPORTS_HELD[descriptor.file.name]
    )
    from google.protobuf import reflection

    # One wrapper of protobuf's metaclass, which reflection and message_factory both name.
    assert reflection.GeneratedProtocolMessageType is message_factory._GENERATED_PROTOCOL_MESSAGE_TYPE


def test_the_hooks_take_over_a_file_of_the_runtimes_classes_of_another_package(upb):
    # A file of another package that a plugin's file imports gets the runtime's own classes of its
    # types, which pickle as protobuf's classes (of a nested type, of protobuf's
    # _InternalConstructMessage); once a plugin's file of the file's own package imports it, those
    # classes become the ones that protobuf's lookups give, and pickle of their own, and the file,
    # of no generated module, is registered in protobuf's symbol database.
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory, symbol_database
    from google.protobuf import message as protobuf_message

    pool = descriptor_pool.Default()
    name = "tests/convert/shared.proto"
    try:
        shared = pool.FindFileByName(name)
    except KeyError:
        proto_file = descriptor_pb2.FileDescriptorProto(
            name=name, package="tests.convert", syntax="proto3"
        )
        outer = proto_file.message_type.add(name="Shared")
        outer.nested_type.add(name="Inner").field.add(
            name="x", number=1, type=descriptor_pb2.FieldDescriptorProto.TYPE_INT32,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
        )
        outer.field.add(
            name="inner", number=1, type=descriptor_pb2.FieldDescriptorProto.TYPE_MESSAGE,
            label=descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL,
            type_name=".tests.convert.Shared.Inner",
        )
        shared = pool.AddSerializedFile(proto_file.SerializeToString())
    other = _add_plugin_file(
        "tests/convert/other.proto",
        ["spark/connect/relations.proto", name],
        [("input", ".spark.connect.Relation"), ("shared", ".tests.convert.Shared")],
        package="tests.convert_other",
    )
    own = _add_plugin_file(
        "tests/convert/own.proto",
        ["spark/connect/relations.proto", name],
        [("input", ".spark.connect.Relation"), ("shared", ".tests.convert.Shared")],
        package="tests.convert",
        message="Own",
    )
    held = _build_generated_module(other)["M"](shared={"inner": {"x": 3}}).shared
    shadow, inner = type(held), type(held.inner)
    descriptor = shared.message_types_by_name["Shared"]
    assert shadow.DESCRIPTOR is descriptor and shadow.__module__ is None
    assert "__reduce__" in vars(shadow)
    reduced = held.inner.__reduce__()
    assert reduced[:2] == (protobuf_message._InternalConstructMessage, ("tests.convert.Shared.Inner",))
    assert message_factory.GetMessageClass(descriptor) is not shadow
    built = _build_generated_module(own)
    assert type(built["Own"]().shared) is shadow and "__reduce__" not in vars(shadow)
    assert message_factory.GetMessageClass(descriptor) is shadow
    assert symbol_database.Default().GetSymbol("tests.convert.Shared.Inner") is inner
    # (Of its own class, of no module, as upb's class of a type of no module pickles.)
    assert held.__reduce__()[0] is shadow


def test_the_hooks_pass_over_what_is_no_descriptor_and_hold_modules_built_before(upb):
    # message_factory.GetMessageClass of an object that is no descriptor (of no hash) is
    # protobuf's, with its error, and a module that protobuf's own metaclass built of a plugin's
    # file, whose class protobuf's symbol database has, gets the runtime's classes once the hooks
    # hold the modules built before them (as at the install of the package).
    from google.protobuf import message as protobuf_message
    from google.protobuf import message_factory, reflection, symbol_database

    unhashable = []
    assert _rust._class_for(unhashable, None) is None
    get = message_factory.GetMessageClass._protobuf_function
    assert _outcome(lambda: message_factory.GetMessageClass(unhashable)) == _outcome(
        lambda: get(unhashable)
    )
    file = _add_plugin_file(
        "tests/built/plugin.proto",
        ["spark/connect/relations.proto"],
        [("input", ".spark.connect.Relation")],
        package="tests.built",
    )
    descriptor = file.message_types_by_name["M"]
    metaclass = reflection.GeneratedProtocolMessageType._protobuf_class
    module = types.ModuleType("tests.built.plugin_pb2")
    old = metaclass(
        "M", (protobuf_message.Message,), {"DESCRIPTOR": descriptor, "__module__": module.__name__}
    )
    vars(module).update(M=old, _M=descriptor, DESCRIPTOR=file)
    database = symbol_database.Default()
    database._classes[descriptor] = old
    sys.modules[module.__name__] = module
    try:
        _rust._hold_built_modules()
    finally:
        del sys.modules[module.__name__]
    cls = vars(module)["M"]
    assert cls is not old and issubclass(cls, _pyspark.ProtoMessage)
    assert database.GetSymbol("tests.built.M") is cls and cls.__module__ == module.__name__
    assert type(cls(input=proto.Relation()).input) is proto.Relation


def test_the_runtimes_helpers_of_pickles_without_protobuf_and_of_python_protobuf(upb):
    # The helpers of the pickles of Any and Timestamp without protobuf (the name that base_pb2
    # binds the class to, and the function of the protocols before 4), the unknown fields of a
    # message as protobuf's pure-Python messages hold them, and the hook of upb's
    # SetAllowOversizeProtos (a protobuf of no such function is passed over).
    from google.protobuf import any_pb2
    from google.protobuf.internal import api_implementation

    assert _rust._path_in_package("google/protobuf/any.proto") == (
        "pyspark.sql.connect.proto.base_pb2",
        "google_dot_protobuf_dot_any__pb2",
    )
    assert _rust._path_in_package("google/protobuf/timestamp.proto")[0] == (
        "pyspark.sql.connect.proto.pipelines_pb2"
    )
    assert type(_rust._new_message("google.protobuf.Any")) is _rust.Any
    packed = _rust.Any(type_url="x/y", value=b"z")
    assert _rust._reduce_ex_by_path(packed, 2) == (
        _rust._new_message, ("google.protobuf.Any",), {"serialized": packed.SerializeToString()}
    )
    assert _rust._reduce_ex_by_path(packed, 4) == (any_pb2.Any, (), packed.__getstate__())
    message = proto.Relation.FromString(b"\x08\x01\xfa\x3e\x01x\xc3\x3e\x08\x05\xc4\x3e")
    assert _pyspark._proto_unknown_field_pairs(message) == [
        (b"\x08", b"\x01"), (b"\xfa\x3e", b"\x01x"), (b"\xc3\x3e", b"\x08\x05\xc4\x3e")
    ]
    assert _pyspark._proto_unknown_field_pairs(proto.Relation()) == []
    module = api_implementation._c_module
    hooked = module.SetAllowOversizeProtos
    assert hooked(False) == hooked._protobuf_function(False) and hooked._protobuf_function is not hooked
    saved = _rust._PROTOBUF.api_implementation
    _rust._PROTOBUF.api_implementation = types.SimpleNamespace()
    try:
        _rust._hook_depth_limit()
    finally:
        _rust._PROTOBUF.api_implementation = saved
    assert module.SetAllowOversizeProtos is hooked
    from google.protobuf import unknown_fields

    assert issubclass(unknown_fields.UnknownFieldSet._protobuf_class, unknown_fields.UnknownFieldSet)


def test_files_of_no_package_that_a_plugins_file_of_no_package_imports_keep_protobufs_classes(
    upb,
):
    # The files of no package are of no one package: a file of no package that a plugin's file of
    # no package imports keeps protobuf's classes, so that the modules of upb that import it take
    # its messages, as without the drop-in.
    shared = _add_plugin_file("tests/np/shared_np.proto", [], [], package="", message="SharedNp")
    other = _add_plugin_file(
        "tests/np/other_np.proto", [shared.name], [("c", ".SharedNp")], package="", message="OtherNp"
    )
    plugin_file = _add_plugin_file(
        "tests/np/plugin_np.proto",
        ["spark/connect/relations.proto", shared.name],
        [("input", ".spark.connect.Relation"), ("c", ".SharedNp")],
        package="",
        message="PluginNp",
    )
    shared_module, other_module = _generated_module(shared), _generated_module(other)
    try:
        plugin = _build_generated_module(plugin_file)["PluginNp"]
        shared_class, other_class = vars(shared_module)["SharedNp"], vars(other_module)["OtherNp"]
        assert not issubclass(shared_class, _pyspark.ProtoMessage)
        assert not issubclass(other_class, _pyspark.ProtoMessage)
        assert other_class(c=shared_class()).HasField("c")
        assert isinstance(other_class().c, shared_class)
        assert issubclass(plugin, _pyspark.ProtoMessage) and plugin(c=shared_class()).HasField("c")
    finally:
        for module in (shared_module, other_module):
            del sys.modules[module.__name__]


def _delta_like(prefix, order):
    """The generated modules of three files of the package ``tests.<prefix>``, as Delta Lake's:
    ``base`` (of ``Table``), ``commands`` (of ``Clone``, which imports only ``base``) and
    ``relations`` (of ``Scan``, which imports ``base`` and the protos of Spark Connect), built in
    ``order``, a permutation of their names, as the imports of an application build them: the
    modules by name."""
    package = f"tests.{prefix}"
    files = {
        "base": _add_plugin_file(f"tests/{prefix}/base.proto", [], [], package=package, message="Table"),
    }
    files["commands"] = _add_plugin_file(
        f"tests/{prefix}/commands.proto",
        [files["base"].name],
        [("table", f".{package}.Table")],
        package=package,
        message="Clone",
    )
    files["relations"] = _add_plugin_file(
        f"tests/{prefix}/relations.proto",
        ["spark/connect/relations.proto", files["base"].name],
        [("input", ".spark.connect.Relation"), ("table", f".{package}.Table")],
        package=package,
        message="Scan",
    )
    return {name: _generated_module(files[name]) for name in order}


def test_files_of_the_package_that_import_a_file_that_the_runtime_takes_over_get_its_classes(
    upb,
):
    # A file of the package of a plugin's file that imports a file that the runtime takes over but
    # no proto of Spark Connect (as Delta Lake's commands.proto would of base.proto alone) gets the
    # runtime's classes too, of its module built before (taken over of protobuf's symbol
    # database) or after, so that its messages take the messages of the file that the runtime took
    # over, and theirs, as upb's of one class of each type.
    from google.protobuf import message_factory

    for prefix, order in (
        ("dlbefore", ("base", "commands", "relations")),
        ("dlafter", ("base", "relations", "commands")),
    ):
        modules = _delta_like(prefix, order)
        try:
            table = vars(modules["base"])["Table"]
            clone = vars(modules["commands"])["Clone"]
            scan = vars(modules["relations"])["Scan"]
            for cls in (table, clone, scan):
                assert issubclass(cls, _pyspark.ProtoMessage), (prefix, cls)
                assert message_factory.GetMessageClass(cls.DESCRIPTOR) is cls
            assert clone(table=table()).HasField("table")
            assert clone(table=scan(table=table()).table).HasField("table")
            assert type(clone().table) is table and type(scan().table) is table
            copied = clone()
            copied.table.CopyFrom(scan(table=table()).table)
            assert copied.HasField("table")
        finally:
            for module in modules.values():
                del sys.modules[module.__name__]


def test_a_module_of_another_name_than_its_builders_is_found_in_its_package(upb):
    # A generated module of a name of another path than protoc's of its file (protoc's of
    # tests/mm/base.proto is tests.mm.base_pb2, which its builder names, imported as
    # tests.mm.proto.base_pb2) is found in the top-level package of the plugin's file that imports
    # it, so that the runtime takes it over, of the module name of upb's class; one that the runtime
    # does not find keeps protobuf's classes, of protobuf's lookups too, rather than the module and
    # the lookups giving two classes.
    from google.protobuf import message_factory

    base = _add_plugin_file("tests/mm/base.proto", [], [], package="tests.mm", message="Table")
    module = _generated_module(base, name="tests.mm.proto.base_pb2")
    plugin_file = _add_plugin_file(
        "tests/mm/relations.proto",
        ["spark/connect/relations.proto", base.name],
        [("input", ".spark.connect.Relation"), ("table", ".tests.mm.Table")],
        package="tests.mm",
        message="Scan",
    )
    lost = _add_plugin_file("tests/mm2/base.proto", [], [], package="tests.mm2", message="Table")
    hidden = _generated_module(lost, name="tests_unfound_mm2.base_pb2")
    lost_plugin = _add_plugin_file(
        "tests/mm2/relations.proto",
        ["spark/connect/relations.proto", lost.name],
        [("input", ".spark.connect.Relation"), ("table", ".tests.mm2.Table")],
        package="tests.mm2",
        message="Scan",
    )
    try:
        built = _build_generated_module(plugin_file)
        table = vars(module)["Table"]
        assert issubclass(table, _pyspark.ProtoMessage) and type(built["Scan"]().table) is table
        assert (table.__module__, message_factory.GetMessageClass(table.DESCRIPTOR)) == (
            "tests.mm.base_pb2",
            table,
        )
        kept = vars(hidden)["Table"]
        held = type(_build_generated_module(lost_plugin)["Scan"]().table)
        assert not issubclass(kept, _pyspark.ProtoMessage) and held is not kept
        assert message_factory.GetMessageClass(kept.DESCRIPTOR) is kept
    finally:
        for found in (module, hidden):
            del sys.modules[found.__name__]


def _unheld_importer_case(order):
    """What a file that the runtime cannot hold (of an extension range), which imports a file of
    the package of a plugin's file, built ``before`` the plugin's file or ``after``, and a file of
    another package that imports it, built after both, make of the classes of that file (see the
    test of a file that the runtime cannot hold below)."""
    from google.protobuf import message_factory, symbol_database

    package = f"tests.unheld{order}"
    prefix = package.replace(".", "/")
    common = _add_plugin_file(f"{prefix}/common.proto", [], [], package=package, message="Common")
    extendable = _add_plugin_file(
        f"{prefix}/extendable.proto",
        [common.name],
        [("c", f".{package}.Common")],
        syntax="proto2",
        package=package,
        message="Extendable",
        extendable=True,
    )
    other = _add_plugin_file(
        f"{prefix}/other.proto", [common.name], [("c", f".{package}.Common")],
        package=package + "x", message="Other",
    )
    plugin_file = _add_plugin_file(
        f"{prefix}/plugin.proto",
        ["spark/connect/relations.proto", common.name],
        [("input", ".spark.connect.Relation"), ("c", f".{package}.Common")],
        package=package,
        message="Plug",
    )
    common_module = _generated_module(common)
    protobufs = vars(common_module)["Common"]
    # A module of the package that star-imports the module of the file.
    star = types.ModuleType(package)
    star.Common = protobufs
    sys.modules[package] = star
    modules = {"common": common_module}
    try:
        if order == "before":
            modules["extendable"] = _generated_module(extendable)
        plugin = _build_generated_module(plugin_file)["Plug"]
        taken_over = vars(common_module)["Common"] is not protobufs
        if order == "after":
            modules["extendable"] = _generated_module(extendable)
        modules["other"] = _generated_module(other)
        common_class = vars(common_module)["Common"]
        extendable_class = vars(modules["extendable"])["Extendable"]
        other_class = vars(modules["other"])["Other"]
        return [
            taken_over,
            common_class is protobufs and star.Common is protobufs,
            [
                issubclass(c, _pyspark.ProtoMessage)
                for c in (common_class, extendable_class, other_class, plugin)
            ],
            extendable_class(c=common_class()).HasField("c"),
            other_class(c=common_class()).HasField("c"),
            plugin(c=common_class()).HasField("c"),
            message_factory.GetMessageClass(common_class.DESCRIPTOR) is common_class,
            symbol_database.Default().GetSymbol(f"{package}.Common") is common_class,
            type(pickle.loads(pickle.dumps(plugin(c=common_class()).c))) is common_class,
        ]
    finally:
        for module in [*modules.values(), star]:
            sys.modules.pop(module.__name__, None)


@pytest.mark.parametrize("order", ["before", "after"])
def test_a_file_that_the_runtime_cannot_hold_keeps_protobufs_classes_of_the_files_it_imports(
    upb, order
):
    # A file that the runtime cannot hold (of extension ranges) keeps protobuf's classes, whose
    # message fields take only protobuf's messages of the types of the files that it imports: a
    # file of the package of a plugin's file that it imports gets protobuf's classes back (in its
    # module, a module of its package that star-imports it and protobuf's lookups), whether it was
    # built before the plugin's file took the file over or after, so that the file's messages take
    # its messages, as without the drop-in, as do the plugin's (of their bytes) and those of a file
    # of another package that imports it, which keeps protobuf's classes. The plugin's message of
    # the type pickles as protobuf's. Of protobuf's pure-Python implementation too.
    expected = [order == "after", True, [False, False, False, True]] + [True] * 6
    assert _unheld_importer_case(order) == expected
    out = _run_script(
        f"""
        import json
        from tests.test_proto_rust_offline import _unheld_importer_case
        print(json.dumps(_unheld_importer_case({order!r})))
        """,
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python",
    )
    assert out == expected


def test_a_give_back_reaches_the_files_of_the_files_given_back(upb):
    # A file that the runtime cannot hold, built after a plugin's file took files of its package
    # over, gives protobuf's classes back to those that it imports and to those that they import
    # (once each, of their nested types too), and to one whose module was never built (of the
    # runtime's classes of no module), of upb's class of the type then: protobuf's lookups give
    # protobuf's classes, of which the file's messages take theirs. A name that the application
    # removed from a module stays removed, the runtime's classes of the types are of types that keep
    # protobuf's classes (their messages pickle as protobuf's), and the plugin's messages take
    # protobuf's.
    from google.protobuf import message_factory, symbol_database

    package = "tests.giveback"
    y = _add_plugin_file(
        "tests/giveback/y.proto", [], [("inner", f".{package}.Y.Inner")],
        package=package, message="Y", nested="Inner",
    )
    x = _add_plugin_file(
        "tests/giveback/x.proto", [y.name], [("y", f".{package}.Y")], package=package, message="X"
    )
    no_module = _add_plugin_file(
        "tests/giveback/nomodule.proto", [], [], package=package, message="NoModule"
    )
    fields = [("x", f".{package}.X"), ("y", f".{package}.Y"), ("n", f".{package}.NoModule")]
    plugin_file = _add_plugin_file(
        "tests/giveback/plugin.proto",
        ["spark/connect/relations.proto", x.name, y.name, no_module.name],
        [("input", ".spark.connect.Relation")] + fields,
        package=package,
        message="Plug",
    )
    extendable = _add_plugin_file(
        "tests/giveback/extendable.proto", [x.name, y.name, no_module.name], fields,
        syntax="proto2", package=package, message="Extendable", extendable=True,
    )
    y_module, x_module = _generated_module(y), _generated_module(x)
    protobufs_y, protobufs_x = vars(y_module)["Y"], vars(x_module)["X"]
    del vars(y_module)["Y"]
    modules = [y_module, x_module]
    try:
        plugin = _build_generated_module(plugin_file)["Plug"]
        assert _rust._hold_files(plugin_file, None)
        assert issubclass(vars(x_module)["X"], _pyspark.ProtoMessage)
        modules.append(_generated_module(extendable))
        assert vars(x_module)["X"] is protobufs_x and "Y" not in vars(y_module)
        database = symbol_database.Default()
        assert database.GetSymbol(f"{package}.Y") is protobufs_y
        assert database.GetSymbol(f"{package}.Y.Inner") is protobufs_y.Inner
        no_modules = message_factory.GetMessageClass(no_module.message_types_by_name["NoModule"])
        assert not issubclass(no_modules, _pyspark.ProtoMessage)
        message = vars(modules[-1])["Extendable"](
            x=protobufs_x(y=protobufs_y()), y=protobufs_y(inner={}), n=no_modules()
        )
        assert message.x.HasField("y") and message.y.HasField("inner") and message.HasField("n")
        assert plugin(x=protobufs_x(), y=protobufs_y()).HasField("x")
        inner = plugin(y={"inner": {}}).y.inner
        for _ in range(2):  # the second of the name of the nested type known
            assert type(pickle.loads(pickle.dumps(inner))) is protobufs_y.Inner
        assert not _rust._imports_spark_connect(x) and _rust._imports_spark_connect(plugin_file)
        assert _rust._declares_unheld(extendable) and _rust._declares_unheld(extendable)
    finally:
        for module in modules:
            sys.modules.pop(module.__name__, None)


def test_built_importers_are_held_of_one_read_and_one_of_no_module_gives_its_imports_back(upb):
    # The modules built before of the files that import a file that a plugin's file takes over are
    # held of one read of protobuf's symbol database, of which one that the hold of another held
    # (as a file of its package that it imports) is passed over. A file of such a module that the
    # runtime does not find keeps protobuf's classes, and gives protobuf's classes back to the file
    # that it imports, so that its messages take that file's messages.
    package = "tests.batch"
    base = _add_plugin_file("tests/batch/base.proto", [], [], package=package, message="Base")
    b = _add_plugin_file(
        "tests/batch/b.proto", [base.name], [("base", f".{package}.Base")], package=package,
        message="B",
    )
    a = _add_plugin_file(
        "tests/batch/a.proto", [b.name, base.name], [("b", f".{package}.B")], package=package,
        message="A",
    )
    plugin_file = _add_plugin_file(
        "tests/batch/plugin.proto",
        ["spark/connect/relations.proto", base.name],
        [("input", ".spark.connect.Relation"), ("base", f".{package}.Base")],
        package=package,
        message="Plug",
    )
    modules = [_generated_module(file) for file in (base, a, b)]  # of a before b
    try:
        _build_generated_module(plugin_file)
        base_class, a_class, b_class = (
            vars(module)[name] for module, name in zip(modules, ("Base", "A", "B"))
        )
        assert all(issubclass(c, _pyspark.ProtoMessage) for c in (base_class, a_class, b_class))
        assert a_class(b=b_class(base=base_class())).b.HasField("base")
    finally:
        for module in modules:
            sys.modules.pop(module.__name__, None)

    package = "tests.unfound"
    base = _add_plugin_file("tests/unfound/base.proto", [], [], package=package, message="Base")
    user = _add_plugin_file(
        "tests/unfound/user.proto", [base.name], [("base", f".{package}.Base")], package=package,
        message="User",
    )
    plugin_file = _add_plugin_file(
        "tests/unfound/plugin.proto",
        ["spark/connect/relations.proto", base.name],
        [("input", ".spark.connect.Relation"), ("base", f".{package}.Base")],
        package=package,
        message="Plug",
    )
    base_module = _generated_module(base)
    protobufs = vars(base_module)["Base"]
    user_class = _build_generated_module(user)["User"]  # of no module of sys.modules
    try:
        plugin = _build_generated_module(plugin_file)["Plug"]
        assert vars(base_module)["Base"] is protobufs and user.name in _rust._UNHELD
        assert user_class(base=protobufs()).HasField("base")
        assert plugin(base=protobufs()).HasField("base")
    finally:
        sys.modules.pop(base_module.__name__, None)


def test_the_runtimes_branches_of_protobufs_pure_python_implementation(upb):
    # Of protobuf's pure-Python implementation (of the runtime's flag of it, here, of upb's
    # protobuf): the parses follow its decoder's limit (decoder.SetRecursionLimit, which the runtime
    # wraps), a class that the metaclass makes out of a batch of registrations has its messages'
    # unknown fields (_unknown_fields) and registers with the Rust runtime at once, the class that
    # a descriptor of no class holds (_concrete_class) is the runtime's, and a give-back of its
    # file makes way for protobuf's class there.
    from google.protobuf.internal import decoder

    deep = proto.Relation(range={"end": 1})
    for _ in range(60):  # 120 levels of messages
        outer = proto.Relation()
        outer.limit.input.CopyFrom(deep)
        deep = outer
    data = deep.SerializeToString()
    saved, saved_limit = _rust._PROTOBUF, decoder._recursion_limit
    set_recursion_limit = decoder.SetRecursionLimit
    _rust._PROTOBUF = types.SimpleNamespace(**dict(vars(saved), python=True))
    try:
        _rust._hook_depth_limit()
        assert decoder.SetRecursionLimit._protobuf_function is set_recursion_limit
        assert _rust._python_depth_limit() == saved_limit
        assert _outcome(lambda: proto.Relation.FromString(data))[:2] == ("raise", "DecodeError")
        decoder.SetRecursionLimit(200)
        assert _rust._python_depth_limit() == 200 and proto.Relation.FromString(data) == deep

        file = _add_plugin_file("tests/pymode/fresh.proto", [], [], package="tests.pymode")
        _pyspark._proto_register_files([file.serialized_pb])
        descriptor = file.message_types_by_name["M"]
        cls = _rust.MessageMeta(
            "M", (_rust._MESSAGE_BASE,), {"DESCRIPTOR": descriptor, "__qualname__": "M"}
        )
        assert cls.FromString(b"\x08\x01")._unknown_fields == [(b"\x08", b"\x01")]

        class StandIn:
            """A descriptor of the pure-Python implementation, of no class (_concrete_class)."""

            nested_types = ()
            _concrete_class = None

        stand_in = StandIn()
        reduce = _rust._reduce_as_protobufs
        owned = type("Owned", (), {"DESCRIPTOR": stand_in, "__reduce__": reduce})
        _rust._own(owned)
        assert stand_in._concrete_class is owned and "__reduce__" not in vars(owned)
        protobufs = type("Protobufs", (), {})
        _rust._PROTOBUFS[stand_in] = protobufs
        file = types.SimpleNamespace(
            name="tests/pymode/stand_in.proto",
            message_types_by_name={"Owned": stand_in},
            dependencies=[],
        )
        _rust._HELD.add(file.name)
        _rust._release(file)
        assert stand_in._concrete_class is None and stand_in not in _rust._OWNED
        assert saved.database._classes.pop(stand_in) is protobufs
        assert vars(owned)["__reduce__"] is _rust._reduce_as_protobufs
        assert file.name in _rust._UNHELD and file.name not in _rust._HELD
    finally:
        _rust._PROTOBUF = saved
        decoder.SetRecursionLimit = set_recursion_limit
        set_recursion_limit(saved_limit)
        _rust._hook_depth_limit()
    assert _outcome(lambda: proto.Relation.FromString(data))[:2] == ("raise", "DecodeError")


def test_the_runtimes_hooks_pass_over_objects_and_entries_of_other_kinds(upb):
    # The hooks pass over what is not theirs: an entry of protobuf's symbol database of no
    # descriptor, a lookup of the class of an object of no file (of protobuf's error of it), an
    # object of no full name or one of the name of a type of the runtime but not its descriptor, a
    # module of sys.modules that is None, and a rebind of nothing. A batch of registrations in a
    # batch is the outer one, and one of no class registers none. A generated module of a file of
    # no messages is the file's; a file that another thread held as this one waited for the lock is
    # held; upb's limit after SetAllowOversizeProtos(True) is 65535 levels; the runtime's own
    # Timestamp methods assign of a datetime; and a class owns of no descriptors of protobuf.
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
    from google.protobuf.internal import api_implementation

    database = _rust._PROTOBUF.database._classes
    entry = object()
    database[entry] = type("Entry", (), {})
    try:
        assert all(file is not entry for file in _rust._built_importers())
        _rust._hold_built_modules()
    finally:
        del database[entry]
    stand_in = object()
    get_message_class = message_factory.GetMessageClass
    assert _outcome(lambda: get_message_class(stand_in))[:2] == _outcome(
        lambda: get_message_class._protobuf_function(stand_in)
    )[:2]
    assert _rust._class_of_descriptor(stand_in) is None
    of_the_name = types.SimpleNamespace(full_name="spark.connect.Relation")
    assert _rust._class_of_descriptor(of_the_name) is None
    sys.modules["tests_of_none"] = None  # type: ignore[assignment]
    try:
        assert _rust._modules_of_packages({"tests_of_none"}) == []
    finally:
        del sys.modules["tests_of_none"]
    _rust._restore_bindings({}, "tests_of_none")
    with _rust._registered_together():
        with _rust._registered_together():
            pass

    name = "tests/noms/enums.proto"
    try:
        file = descriptor_pool.Default().FindFileByName(name)
    except KeyError:
        enums = descriptor_pb2.FileDescriptorProto(name=name, package="tests.noms", syntax="proto3")
        enums.enum_type.add(name="E").value.add(name="E_ZERO", number=0)
        file = descriptor_pool.Default().AddSerializedFile(enums.SerializeToString())
    module = types.ModuleType("tests_noms_enums_pb2")
    module.DESCRIPTOR = file
    assert _rust._is_generated_module_of(module, file)

    plugin_file = _add_plugin_file(
        "tests/race/plugin.proto",
        ["spark/connect/relations.proto"],
        [("input", ".spark.connect.Relation")],
        package="tests.race",
    )

    class HeldMeanwhile:
        """The lock of the hooks, of another thread that holds the file meanwhile."""

        def __enter__(self):
            _rust._HELD.add(plugin_file.name)

        def __exit__(self, *exc_info):
            return False

    saved_lock = _rust._HOOK_LOCK
    _rust._HOOK_LOCK = HeldMeanwhile()
    try:
        assert _rust._hold_files(plugin_file, None)
    finally:
        _rust._HOOK_LOCK = saved_lock
        _rust._HELD.discard(plugin_file.name)

    set_allow_oversize_protos = api_implementation._c_module.SetAllowOversizeProtos
    before = _rust._upb_depth_limit()
    try:
        set_allow_oversize_protos(True)
        assert _rust._upb_depth_limit() == 65535
    finally:
        set_allow_oversize_protos(before == 65535)
    assert _rust._upb_depth_limit() == before

    stamp = proto.PipelineEvent().timestamp
    _rust._TimestampMethods._internal_assign(stamp, datetime.datetime(2020, 1, 2))
    assert stamp.seconds == 1577923200

    saved = _rust._USE_PROTOBUF_DESCRIPTORS
    _rust._USE_PROTOBUF_DESCRIPTORS = False
    try:
        _rust._own(type("Unowned", (), {}))
    finally:
        _rust._USE_PROTOBUF_DESCRIPTORS = saved


def test_the_attributes_of_names_that_a_class_has_are_upbs(upb):
    # The number of a field is the class's attribute of its name rather than a nested enum or a
    # value of a nested enum of that name, as upb gives the numbers of a read of any name of their
    # suffix, and a nested enum or a value of one of a name of the suffix of no field is no
    # attribute; a nested message, the DESCRIPTOR and a method of the base keep their names.
    from google.protobuf import descriptor_pb2, descriptor_pool

    field = descriptor_pb2.FieldDescriptorProto
    name = "tests/clash/clash.proto"
    file_proto = descriptor_pb2.FileDescriptorProto(
        name=name, package="tests.clash", dependency=["spark/connect/relations.proto"],
        syntax="proto3",
    )
    message = file_proto.message_type.add(name="Clash")
    enum = message.enum_type.add(name="DESCRIPTOR")
    for number, value in enumerate(["Clear", "X_FIELD_NUMBER", "BAR_FIELD_NUMBER"]):
        enum.value.add(name=value, number=number)
    message.enum_type.add(name="Z_FIELD_NUMBER").value.add(name="Z_ZERO", number=0)
    message.enum_type.add(name="W_FIELD_NUMBER").value.add(name="W_ZERO", number=0)
    message.nested_type.add(name="Y_FIELD_NUMBER")
    for field_name, number in (("x", 7), ("y", 8), ("z", 9)):
        message.field.add(
            name=field_name, number=number, type=field.TYPE_INT32, label=field.LABEL_OPTIONAL
        )
    message.field.add(
        name="input", number=1, type=field.TYPE_MESSAGE, label=field.LABEL_OPTIONAL,
        type_name=".spark.connect.Relation",
    )
    serialized = file_proto.SerializeToString()
    try:
        file = descriptor_pool.Default().FindFileByName(name)
    except KeyError:
        file = descriptor_pool.Default().AddSerializedFile(serialized)
    ours = _build_generated_module(file)["Clash"]
    assert issubclass(ours, _pyspark.ProtoMessage)

    def theirs():
        # A class of upb of no attribute read yet: upb binds the numbers of the fields once one
        # of their names is read, in place of what the class had of the names.
        return _upb_classes_of([(serialized, "tests.clash.clash_pb2")])["tests.clash.Clash"]

    for attribute in ("X_FIELD_NUMBER", "Z_FIELD_NUMBER", "INPUT_FIELD_NUMBER", "Z_ZERO"):
        assert getattr(ours, attribute) == getattr(theirs(), attribute)
    # Nor is a name of the suffix of no field (a nested enum or a value of one) an attribute.
    for attribute in ("BAR_FIELD_NUMBER", "W_FIELD_NUMBER"):
        assert not hasattr(ours, attribute) and not hasattr(theirs(), attribute)
    assert ours.X_FIELD_NUMBER == 7 and ours.Z_FIELD_NUMBER == 9
    assert ours.Y_FIELD_NUMBER.DESCRIPTOR.full_name == theirs().Y_FIELD_NUMBER.DESCRIPTOR.full_name
    assert ours.DESCRIPTOR is file.message_types_by_name["Clash"]
    assert type(ours.Clear).__name__ == type(theirs().Clear).__name__ == "method_descriptor"


def test_a_module_built_after_a_lookup_held_its_file_is_given_back(upb):
    # A plugin's file that a lookup holds (message_factory.GetMessageClass of its type of the
    # default pool) before the module of a file of its package that it imports builds gives that
    # module the runtime's class, then of the module's name (as upb's class is), so that a give-back
    # of the file (of a file that the runtime cannot hold that imports it) finds the module, which
    # gets protobuf's class back.
    from google.protobuf import message_factory

    package = "tests.looked"
    common = _add_plugin_file(
        "tests/looked/common.proto", [], [], package=package, message="Common"
    )
    plugin_file = _add_plugin_file(
        "tests/looked/plugin.proto",
        ["spark/connect/relations.proto", common.name],
        [("input", ".spark.connect.Relation"), ("c", f".{package}.Common")],
        package=package,
        message="Plug",
    )
    extendable = _add_plugin_file(
        "tests/looked/extendable.proto", [common.name], [("c", f".{package}.Common")],
        syntax="proto2", package=package, message="Extendable", extendable=True,
    )
    plugin = message_factory.GetMessageClass(plugin_file.message_types_by_name["Plug"])
    assert issubclass(plugin, _pyspark.ProtoMessage)
    modules = [_generated_module(common)]
    try:
        held = vars(modules[0])["Common"]
        assert issubclass(held, _pyspark.ProtoMessage)
        assert repr(held) == "<class 'tests.looked.common_pb2.Common'>"
        modules.append(_generated_module(extendable))
        given_back = vars(modules[0])["Common"]
        assert not issubclass(given_back, _pyspark.ProtoMessage)
        assert vars(modules[1])["Extendable"](c=given_back()).HasField("c")
        assert plugin(c=given_back()).HasField("c")
    finally:
        for module in modules:
            sys.modules.pop(module.__name__, None)


def test_a_class_attribute_that_fails_raises_its_error_and_not_the_attribute():
    # A message's read of an attribute of its class whose getter raises AttributeError (a property)
    # raises the error of the name, rather than giving the property object, of the attribute of
    # the class; an attribute of the metaclass (__name__) is the class's, and so is one that the
    # read of the class makes (__annotations__, which a class makes on its first read), as upb's.
    def failing(self):
        raise AttributeError("failing")

    def make():
        return type(
            "Failing",
            (_pyspark.ProtoMessage,),
            {
                "DESCRIPTOR": _own("spark.connect.Relation"),
                "__slots__": (),
                "failing": property(failing),
            },
        )

    message = make()()
    assert _outcome(lambda: message.failing)[:3] == ("raise", "AttributeError", "failing")
    assert not hasattr(message, "failing") and message.__name__ == "Failing"
    # Of a class that has not made it yet, as the read of the class gives it ({} of Python 3.10).
    instances = _outcome(lambda: make()().__annotations__)
    assert instances[:2] == _outcome(lambda: make().__annotations__)[:2]


def test_unknown_fields_of_protobufs_pure_python_messages_and_the_runtimes_compare(upb):
    # Of protobuf's pure-Python implementation, a message of the runtime has the unknown fields of
    # its messages (_unknown_fields, of the Rust runtime: the bytes of the tag and of the value of
    # each), so that protobuf's == of its message and one of the runtime compares them.
    out = _run_script(
        """
        import json
        from google.protobuf import timestamp_pb2
        import pyspark.sql.connect.proto as proto

        stamp = proto.PipelineEvent().timestamp
        stamp.MergeFromString(b"\\x18\\x01\\xfa\\x3e\\x01x")
        theirs = timestamp_pb2.Timestamp()
        theirs.MergeFromString(b"\\x18\\x01\\xfa\\x3e\\x01x")
        print(json.dumps([
            [[t.hex(), v.hex()] for t, v in stamp._unknown_fields],
            theirs == stamp, stamp == theirs, timestamp_pb2.Timestamp() == stamp,
        ]))
        """,
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python",
    )
    assert out == [[["18", "01"], ["fa3e", "0178"]], True, True, False]


def test_messages_of_from_string_and_copies_are_tracked_as_upbs(upb):
    # As upb's, the message of FromString and of copy.copy, as one of a call of the class, is
    # tracked by the garbage collector; the messages within it are not (see docs/compatibility.md).
    import gc

    data = proto.Relation(sql={"query": "q"}).SerializeToString()
    for cls in (proto.Relation, upb("spark.connect.Relation")):
        parsed = cls.FromString(data)
        assert gc.is_tracked(parsed) and gc.is_tracked(copy.copy(parsed)), cls
    assert not gc.is_tracked(proto.Relation.FromString(data).sql)


def test_the_class_of_a_type_of_another_package_pickles_as_protobufs(upb):
    # The class of the runtime of a type of a file that keeps protobuf's classes (of another
    # package) pickles as protobuf's class of the type, as its messages do; a class that protobuf's
    # lookups give pickles of its name.
    common = _add_plugin_file("tests/clsp/common.proto", [], [], package="tests.clsp", message="Common")
    plugin_file = _add_plugin_file(
        "tests/clsp/plugin.proto",
        ["spark/connect/relations.proto", common.name],
        [("input", ".spark.connect.Relation"), ("c", ".tests.clsp.Common")],
        package="tests.clsp_plugin",
    )
    module = _generated_module(common)
    try:
        held = type(_build_generated_module(plugin_file)["M"]().c)
        common_class = vars(module)["Common"]
        assert held is not common_class
        for protocol in (2, 4, 5):
            assert pickle.loads(pickle.dumps(held, protocol)) is common_class
        assert pickle.loads(pickle.dumps(proto.Relation)) is proto.Relation
        assert pickle.loads(pickle.dumps(proto.Expression.Literal)) is proto.Expression.Literal
    finally:
        del sys.modules[module.__name__]


def test_parses_follow_the_limit_of_protobufs_parsers_of_a_function_bound_before(upb):
    # A parse that reaches its limit asks protobuf for its limit again, so that a call of upb's
    # SetAllowOversizeProtos(True) of a reference to it taken before the package imports is seen;
    # of protobuf's pure-Python implementation, the parses follow its decoder.SetRecursionLimit,
    # of a reference taken before too.
    script = """
        import json
        {bind}
        import pyspark.sql.connect.proto as proto

        deep = proto.Relation(range={{"end": 1}})
        for _ in range(150):
            outer = proto.Relation()
            outer.limit.input.CopyFrom(deep)
            deep = outer
        data = deep.SerializeToString()

        def parse():
            try:
                return len(proto.Relation.FromString(data).SerializeToString()) == len(data)
            except Exception as e:
                return type(e).__name__

        out = [parse()]
        early({argument})
        out.append(parse())
        print(json.dumps(out))
        """
    upb_bind = "from google._upb._message import SetAllowOversizeProtos as early"
    assert _run_script(script.format(bind=upb_bind, argument="True")) == ["DecodeError", True]
    python_bind = "from google.protobuf.internal.decoder import SetRecursionLimit as early"
    assert _run_script(
        script.format(bind=python_bind, argument="1000"),
        PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python",
    ) == ["DecodeError", True]
    wrapped = script.replace("{bind}", "from google.protobuf.internal import decoder").replace(
        "early({argument})", "decoder.SetRecursionLimit(1000)"
    )
    assert _run_script(
        wrapped.replace("{{", "{").replace("}}", "}"), PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION="python"
    ) == ["DecodeError", True]


# The copies of the messages of a constructor (SPARK-62264): a message of the runtime given for a
# message field, a repeated message field or the value of a map of messages is copied natively, and
# its objects are made when Python reads them, of what upb's copy of its bytes would hold.


def _copy_sources(cls):
    """Messages of `cls` (the classes of a runtime, by full name), of every kind of field below
    them: a message of a message, a repeated message, a map of messages, a repeated scalar and a
    map of scalars field, and of a oneof."""
    relation = cls("spark.connect.Relation")()
    relation.read.data_source.format = "csv"
    relation.read.data_source.paths.extend(["a", "b"])
    relation.read.data_source.options["k"] = "v"
    relation.common.plan_id = 7
    expression = cls("spark.connect.Expression")()
    function = expression.unresolved_function
    function.function_name = "f"
    function.arguments.add().unresolved_attribute.unparsed_identifier = "x"
    function.arguments.add().literal.integer = 1
    sql = cls("spark.connect.SQL")(query="select 1")
    sql.named_arguments["n"].literal.string = "s"
    return relation, expression, sql


def _snapshot(*messages):
    """What a test compares of messages: their bytes and their text."""
    return [(m.SerializeToString(deterministic=True), str(m)) for m in messages]


def _independence_cases(cls):
    """The changes of constructor copies and of their sources, of each kind of field, and the
    messages after each, which upb's and the runtime's agree on: a copy keeps the fields that its
    source had when it was made, and its source keeps its own, whichever of them changes."""
    out = []
    relation, expression, sql = _copy_sources(cls)
    plan = cls("spark.connect.Plan")(root=relation)
    out.append(_snapshot(relation, plan))
    relation.read.data_source.paths.append("c")
    relation.read.data_source.options["k"] = "w"
    relation.common.plan_id = 8
    out.append(_snapshot(relation, plan))
    plan.root.read.data_source.paths[0] = "z"
    plan.root.read.data_source.options["j"] = "u"
    plan.root.common.ClearField("plan_id")
    plan.root.read.data_source.format = "json"
    out.append(_snapshot(relation, plan))
    # A repeated message field takes copies of its messages, each its own.
    references = cls("spark.connect.WithRelations")(references=[relation, relation])
    relation.read.data_source.paths.extend(["d"])
    references.references[0].read.data_source.paths.append("e")
    references.references[1].common.plan_id = 9
    out.append(_snapshot(relation, references))
    # The messages of a map of messages, and of a repeated field of a message held in a field.
    held = cls("spark.connect.SQL")(named_arguments={"a": expression, "b": expression})
    expression.unresolved_function.arguments[0].unresolved_attribute.unparsed_identifier = "y"
    held.named_arguments["a"].unresolved_function.arguments.add().literal.integer = 2
    held.named_arguments["b"].unresolved_function.function_name = "g"
    out.append(_snapshot(expression, held))
    # A member of a oneof: setting another member clears it, of the copy only.
    alias = cls("spark.connect.Expression")(unresolved_function=expression.unresolved_function)
    alias.literal.integer = 3
    out.append(_snapshot(expression, alias) + [alias.WhichOneof("expr_type")])
    expression.unresolved_function.ClearField("arguments")
    out.append(_snapshot(expression, alias, sql, held))
    # A copy of a copy is a copy of its own: the copies of the fields it shares with its source
    # are its own too.
    action = cls("spark.connect.MergeAction")(condition=expression, assignments=[
        cls("spark.connect.MergeAction.Assignment")(key=expression, value=expression)
    ])
    first = cls("spark.connect.Expression")(merge_action=action)
    second = cls("spark.connect.Expression")(merge_action=action)
    first.merge_action.assignments[0].key.unresolved_function.function_name = "first"
    action.condition.unresolved_function.function_name = "action"
    second.merge_action.condition.unresolved_function.arguments.add().literal.integer = 4
    out.append(_snapshot(action, first, second))
    command = cls("spark.connect.Command")(sql_command={"sql": "s", "input": relation})
    copied = cls("spark.connect.Command")(sql_command=command.sql_command)
    command.sql_command.input.read.data_source.paths.append("f")
    copied.sql_command.input.read.data_source.options.clear()
    out.append(_snapshot(command, copied))
    return out


def test_constructor_copies_are_independent_of_their_sources_as_upbs(upb):
    # A message given to a constructor is copied, as upb copies it by its bytes: a change of the
    # copy, of a message field, a nested message, a repeated field of messages or of scalars, a
    # map of messages or of scalars, or a oneof, is not one of its source, nor the other way, nor of
    # another copy of the same message or of the copy.
    assert _independence_cases(ours) == _independence_cases(upb)


def _identity_cases(cls):
    """Whether the objects that the fields of constructor copies give are the same object at each
    read, and not the objects of their sources."""
    relation, expression, sql = _copy_sources(cls)
    plan = cls("spark.connect.Plan")(root=relation)
    references = cls("spark.connect.WithRelations")(references=[relation])
    held = cls("spark.connect.SQL")(named_arguments={"a": expression})
    return {
        "field": plan.root is plan.root,
        "nested": plan.root.read.data_source is plan.root.read.data_source,
        "not the source": plan.root is not relation,
        "nested not the source": plan.root.read is not relation.read,
        "repeated scalars": plan.root.read.data_source.paths is plan.root.read.data_source.paths,
        "map of scalars": plan.root.read.data_source.options is plan.root.read.data_source.options,
        "container": references.references is references.references,
        "item": references.references[0] is references.references[0],
        "item not the source": references.references[0] is not relation,
        "map": held.named_arguments is held.named_arguments,
        "map value": held.named_arguments["a"] is held.named_arguments["a"],
        "map value not the source": held.named_arguments["a"] is not expression,
        "listed": [v for _, v in plan.ListFields()][0] is plan.root,
    }


def test_constructor_copies_keep_their_identity_as_upbs(upb):
    # The field of a constructor copy gives one object, at every read, as it does once it is set:
    # of a message, of its messages, of its containers, of their items and of the values of its
    # maps, and of ListFields. A weak reference of it lives as long as the message that holds it.
    import weakref

    mine = _identity_cases(ours)
    assert mine == _identity_cases(upb)
    assert all(mine.values()), mine
    relation, expression, sql = _copy_sources(ours)
    plan = ours("spark.connect.Plan")(root=relation)
    read = weakref.ref(plan.root.read)
    assert read() is plan.root.read
    del plan
    assert read() is None
    # As the copies of the runtime were made before, the objects of a copy are not tracked by
    # the garbage collector, and a message reports none of its own.
    import gc

    plan = ours("spark.connect.Plan")(root=relation)
    assert not gc.is_tracked(plan.root) and not gc.is_tracked(plan.root.read)
    assert gc.get_referents(plan) == [type(plan)]


def _operation_cases(cls):
    """The operations of upb's messages on messages of copies whose fields no object was made of
    yet, each on a copy of its own: what each gives, and the messages after it."""
    relation, expression, sql = _copy_sources(cls)
    unknown = cls("spark.connect.Relation").FromString(relation.SerializeToString() + _UNKNOWN)

    def plan():
        return cls("spark.connect.Plan")(root=unknown)

    def references():
        return cls("spark.connect.WithRelations")(references=[relation, unknown])

    def held():
        return cls("spark.connect.SQL")(named_arguments={"a": expression}, pos_arguments=[expression])

    import copy as copying

    operations = {
        "bytes": lambda m: m.SerializeToString(deterministic=True),
        "ByteSize": lambda m: m.ByteSize(),
        "str": str,
        "repr": repr,
        "==": lambda m: m == type(m).FromString(m.SerializeToString()),
        "!= of a change": lambda m: m != type(m)(),
        "ListFields": lambda m: [(f.name, _comparable(v)) for f, v in m.ListFields()],
        "HasField": lambda m: [m.HasField(f.name) for f in m.DESCRIPTOR.fields if f.has_presence],
        "deepcopy": lambda m: copying.deepcopy(m).SerializeToString(deterministic=True),
        "copy": lambda m: copying.copy(m).SerializeToString(deterministic=True),
        "CopyFrom": lambda m: _copied_from(m).SerializeToString(deterministic=True),
        "MergeFrom": lambda m: (m.MergeFrom(type(m)(**{})), m.MergeFrom(m))[1],
        "Clear": lambda m: m.Clear(),
        "DiscardUnknownFields": lambda m: m.DiscardUnknownFields(),
        "IsInitialized": lambda m: m.IsInitialized(),
        "ParseFromString": lambda m: m.ParseFromString(m.SerializeToString()),
        "MergeFromString": lambda m: m.MergeFromString(m.SerializeToString()),
    }
    out = {}
    for name, make in {"plan": plan, "references": references, "held": held}.items():
        for label, operation in operations.items():
            message = make()
            result = _outcome(lambda: operation(message))
            out[f"{name} {label}"] = (result, message.SerializeToString(deterministic=True))
    return out


def _copied_from(message):
    copied = type(message)()
    copied.CopyFrom(message)
    return copied


def test_operations_on_constructor_copies_are_upbs(upb):
    # The operations of a message read its constructor copies as they read their objects: its
    # bytes, size, text, equality, fields, presence, pickle, copies, merges, clears and parses.
    mine, theirs = _operation_cases(ours), _operation_cases(upb)
    for label in theirs:
        assert mine[label] == theirs[label], label
    # (upb's classes of the tests are of the names of the package's, so do not pickle.)
    relation, expression, sql = _copy_sources(ours)
    for message in (
        ours("spark.connect.Plan")(root=relation),
        ours("spark.connect.SQL")(named_arguments={"a": expression}, pos_arguments=[expression]),
    ):
        assert pickle.loads(pickle.dumps(message)) == message


def _chain(cls, additions):
    """The Expression of `additions` additions of 1 to a column of an origin, as PySpark's
    `c + 1 + 1 ...` of a Column of its origin builds it: of 2 * additions + 3 levels below it."""
    expression = cls("spark.connect.Expression")
    current = expression(unresolved_attribute={"unparsed_identifier": "id"})
    current.common.origin.python_origin.fragment = "col"
    for _ in range(additions):
        outer = expression()
        outer.unresolved_function.function_name = "+"
        outer.unresolved_function.arguments.add().CopyFrom(current)
        outer.unresolved_function.arguments.add().literal.integer = 1
        current = outer
    return current


def _merge_of(cls, deep, where):
    """Where PySpark's merge of `deep` at `where` (an assignment, the condition of a clause or the
    condition of the merge) fails, of its constructors (`_build_merge_action` and `merge()`):
    the constructor that raises, and its error, or None."""
    expression = cls("spark.connect.Expression")
    column = expression(unresolved_attribute={"unparsed_identifier": "c"})
    steps = []
    try:
        steps.append("Assignment")
        value = deep if where == "assignment" else column
        assignment = cls("spark.connect.MergeAction.Assignment")(key=column, value=value)
        steps.append("MergeAction")
        condition = deep if where == "clause condition" else None
        action = cls("spark.connect.MergeAction")(
            action_type=1, condition=condition, assignments=[assignment]
        )
        steps.append("Expression")
        clause = expression(merge_action=action)
        steps.append("MergeIntoTableCommand")
        merge_condition = deep if where == "merge condition" else column
        merge = cls("spark.connect.MergeIntoTableCommand")(
            target_table_name="t", merge_condition=merge_condition, match_actions=[clause]
        )
        steps.append("Command")
        command = cls("spark.connect.Command")(merge_into_table_command=merge)
        plan = cls("spark.connect.Plan")()
        plan.command.CopyFrom(command)
        return None, len(plan.SerializeToString())
    except Exception as e:  # noqa: BLE001
        return steps[-1], type(e).__name__, str(e)


def test_constructor_copies_fail_from_the_depths_of_upbs(upb):
    # As upb's copies of a constructor's message by its bytes: a copy of a message of more than
    # 100 levels below it fails, of upb's DecodeError, from the same depths as PySpark's merge, of
    # its constructors (of a chain of additions, of an origin): an assignment of 47 additions, a
    # clause condition of 48, the condition of the merge of 49; and copy.copy, as upb's of its
    # bytes. The copies that pass are of the same bytes.
    for additions in (46, 47, 48, 49, 100, 120):
        deep = {"ours": _chain(ours, additions), "upb": _chain(upb, additions)}
        for where in ("assignment", "clause condition", "merge condition"):
            mine = _merge_of(ours, deep["ours"], where)
            assert mine == _merge_of(upb, deep["upb"], where), (additions, where)
            fails = additions >= {"assignment": 47, "clause condition": 48, "merge condition": 49}[where]
            assert (mine[0] is not None) == fails, (additions, where, mine)
            if fails:
                assert mine[1] == "DecodeError", (additions, where, mine)
                assert mine[2].endswith(": Exceeded upb_DecodeOptions_MaxDepth"), mine
            if additions >= 100:
                # The first constructor fails, of the copy of the expression.
                assert mine[0] == {
                    "assignment": "Assignment",
                    "clause condition": "MergeAction",
                    "merge condition": "MergeIntoTableCommand",
                }[where], mine
        assert _outcome(lambda: copy.copy(deep["ours"])) == _outcome(
            lambda: copy.copy(deep["upb"])
        ), additions


# For a message type of the lockstep, fields of other types that take a message of it: a message
# field, a repeated message field, the value of a map of messages.
_COPY_HOLDERS = {
    "spark.connect.Relation": [
        ("spark.connect.Plan", "root", "single"),
        ("spark.connect.WithRelations", "references", "repeated"),
    ],
    "spark.connect.Expression": [
        ("spark.connect.Expression.Alias", "expr", "single"),
        ("spark.connect.Expression.UnresolvedFunction", "arguments", "repeated"),
        ("spark.connect.SQL", "named_arguments", "map"),
    ],
    "spark.connect.Command": [("spark.connect.Plan", "command", "single")],
    "spark.connect.Expression.Literal": [
        ("spark.connect.Expression", "literal", "single"),
        ("spark.connect.Expression.Literal.Array", "elements", "repeated"),
        ("spark.connect.SQL", "args", "map"),
    ],
    "spark.connect.MlParams": [("spark.connect.MlCommand.Fit", "params", "single")],
}


@pytest.mark.parametrize("seed", range(6))
def test_random_operations_on_constructor_copies_agree_with_upb(upb, seed):
    # Random operations on messages, on the copies that constructors make of them (of a message
    # field, a repeated field and a map of messages, of copies of the copies too), and then on
    # each again, of upb's and of the runtime's in step: the messages agree after each, so a
    # change of a copy or of its source is of that message only, as upb's.
    lockstep = _Lockstep(upb, seed)
    rng = lockstep.rng
    for full_name, holders in _COPY_HOLDERS.items():
        a, b = upb(full_name)(), ours(full_name)()
        log = []
        for _ in range(15):
            lockstep.step(a, b, log)
        for holder, field, shape in holders:

            def hold(cls, message):
                value = {"single": message, "repeated": [message, message]}.get(
                    shape, {"k": message, "j": message}
                )
                return cls(holder)(**{field: value})

            def held(m, at):
                value = getattr(m, field)
                if shape == "single":
                    return value
                return value[at] if shape == "repeated" else value["kj"[at]]

            pairs = [(full_name, a, b)]
            holders_of = [(hold(upb, a), hold(ours, b))]
            log.append(f"{holder}({field}=...)")
            for _ in range(24):
                kind = rng.random()
                at = rng.randrange(2)
                if kind < 0.35:
                    lockstep.step(a, b, log)
                elif kind < 0.85:
                    wa, wb = rng.choice(holders_of)
                    log.append(f"in {holder}.{field}[{at}]")
                    lockstep.step(held(wa, at), held(wb, at), log)
                else:
                    # A copy of a copy: of the message that a holder holds, which is copied again.
                    wa, wb = rng.choice(holders_of)
                    log.append(f"{holder}({field}={holder}.{field}[{at}])")
                    holders_of.append((hold(upb, held(wa, at)), hold(ours, held(wb, at))))
                for name, x, y in pairs:
                    lockstep.compare(name, x, y, log)
                for wa, wb in holders_of:
                    lockstep.compare(holder, wa, wb, log)


def _error_cases(cls):
    """What constructors raise of values of message fields beside the messages that they copy, in
    the order of their keyword arguments, and what they build of well-known types of values of
    other types (a Timestamp of a datetime, of `_internal_assign`)."""
    relation = cls("spark.connect.Relation")(range={"end": 1})
    expression = cls("spark.connect.Expression")(literal={"integer": 1})
    timestamp = cls("google.protobuf.Timestamp")(seconds=5, nanos=6)

    def init_again():
        plan = cls("spark.connect.Plan")(root=relation)
        plan.__init__(root={"range": {"start": 2}})
        return plan

    cases = {
        "a message of another type": lambda: cls("spark.connect.Plan")(root=expression),
        "an int": lambda: cls("spark.connect.Plan")(root=5),
        "a list of a message of another type": lambda: cls("spark.connect.WithRelations")(
            references=[relation, expression]
        ),
        "a list of an int": lambda: cls("spark.connect.WithRelations")(references=[relation, 5]),
        "a message for a list": lambda: cls("spark.connect.WithRelations")(references=relation),
        "a map value of another type": lambda: cls("spark.connect.SQL")(
            named_arguments={"a": relation}
        ),
        "a field of no name after a copy": lambda: cls("spark.connect.Plan")(root=relation, nope=1),
        "a copy after a field of no name": lambda: cls("spark.connect.Plan")(nope=1, root=relation),
        "a bad enum then a copy": lambda: cls("spark.connect.MergeAction")(
            action_type="NOPE", condition=expression
        ),
        "a copy then a bad enum": lambda: cls("spark.connect.MergeAction")(
            condition=expression, action_type="NOPE"
        ),
        "a copy then a bad copy": lambda: cls("spark.connect.MergeAction")(
            condition=expression, assignments=[expression]
        ),
        "a copy of a mapping of a copy": lambda: cls("spark.connect.Plan")(
            root={"limit": {"input": relation, "limit": 2}}
        ),
        "a Timestamp of a datetime": lambda: cls("spark.connect.PipelineEvent")(
            timestamp=datetime.datetime(2020, 1, 2, 3, 4, 5, 678000)
        ),
        "a Timestamp of a str": lambda: cls("spark.connect.PipelineEvent")(timestamp="x"),
        "a Timestamp of a Timestamp": lambda: cls("spark.connect.PipelineEvent")(
            timestamp=timestamp
        ),
        "a copy and a Timestamp of a datetime": lambda: cls("spark.connect.PipelineEvent")(
            timestamp=datetime.datetime(2021, 1, 1), message="m"
        ),
        "the constructor of a message of a field set": init_again,
    }
    return {label: _outcome(case) for label, case in cases.items()}


def test_constructors_raise_the_errors_of_upbs_beside_the_copies(upb):
    # The values of the message fields of a constructor that are not messages of the runtime of
    # the type take the paths that they took, of upb's errors, in the order of the keyword
    # arguments, and of the Timestamp of a datetime of `_internal_assign`.
    # A second constructor merges a mapping into the message of a field that the message holds,
    # as upb's (see test_merges_into_a_message_that_the_value_holds_are_upbs).
    mine, theirs = _error_cases(ours), _error_cases(upb)
    differ = {label for label in theirs if mine[label] != theirs[label]}
    assert differ == set(), {label: (mine[label], theirs[label]) for label in differ}


def _shared_cases(cls):
    """Copies of copies, which the runtime shares the native values of, changed one at a time:
    each change is of one message only, as upb's."""
    data = _relation_bytes(4, lambda level: True)
    source = cls("spark.connect.Relation").FromString(data)
    first = cls("spark.connect.Plan")(root=source)
    second = copy.deepcopy(first)
    third = cls("spark.connect.Plan")()
    third.CopyFrom(second)
    fourth = cls("spark.connect.Plan")(root=third.root)
    out = [_snapshot(source, first, second, third, fourth)]
    first.DiscardUnknownFields()
    out.append(_snapshot(source, first, second, third, fourth))
    second.root.limit.input.limit.limit = 9
    out.append(_snapshot(source, first, second, third, fourth))
    third.root.limit.input.DiscardUnknownFields()
    out.append(_snapshot(source, first, second, third, fourth))
    fourth.root.limit.input.limit.input.ClearField("limit")
    out.append(_snapshot(source, first, second, third, fourth))
    second.MergeFrom(fourth)
    out.append(_snapshot(source, first, second, third, fourth))
    return out


def test_copies_of_copies_change_alone_as_upbs(upb):
    # The copies of copies (a constructor's, CopyFrom's and copy.deepcopy's of a copy) are each
    # their own: discarding the unknown fields of one of them, or changing it, changes no other.
    assert _shared_cases(ours) == _shared_cases(upb)


def test_copies_are_of_the_classes_registered_when_they_were_made():
    # The objects that a copy of a constructor makes when its fields are read are of the class that
    # the copy would have made them of when it was made, the class registered of their type then,
    # even if a class of the type is registered since; a copy made after is of the new class.
    script = """
        import json

        import pyspark.sql.connect.proto as proto
        from pyspark import _pyspark
        from pyspark.sql.connect.proto import _rust

        relation = proto.Relation(limit={"limit": 1, "input": {"range": {"end": 2}}})
        before = proto.Plan(root=relation)
        listed = proto.WithRelations(references=[relation])
        held = proto.SQL(named_arguments={"a": {"literal": {"integer": 1}}})
        held_before = proto.Command(sql_command={"named_arguments": held.named_arguments})
        other = type(
            "Relation",
            (_pyspark.ProtoMessage, _rust._MessageBase),
            {"__slots__": (), "DESCRIPTOR": proto.Relation.DESCRIPTOR},
        )
        _pyspark._proto_register_classes({"spark.connect.Relation": other})
        try:
            after = proto.Plan(root=relation)
            out = [
                type(before.root) is proto.Relation,
                type(before.root.limit.input) is proto.Relation,
                type(listed.references[0]) is proto.Relation,
                type(after.root) is other,
                type(after.root.limit.input) is other,
            ]
        finally:
            _pyspark._proto_register_classes({"spark.connect.Relation": proto.Relation})
        again = proto.Plan(root=relation)
        out += [type(again.root) is proto.Relation, type(after.root) is other]
        print(json.dumps(out))
        """
    assert _run_script(script) == [True] * 7


def test_deep_copies_build_compare_and_drop_on_a_small_stack():
    # The copies of a message of tens of thousands of levels (copy.deepcopy, CopyFrom, a
    # constructor of a copy of it), their comparisons, encodings, discards of unknown fields,
    # reads of their fields to the bottom and their drops finish on a thread of a small stack, as
    # those of its objects do; a constructor's copy of it fails at upb's limit.
    script = textwrap.dedent(
        """
        import copy
        import json
        import threading

        import pyspark.sql.connect.proto as proto
        from pyspark import _pyspark
        from pyspark.sql.connect.proto._rust import DecodeError
        from tests.test_proto_rust_offline import _relation_bytes

        LEVELS = 30000
        results = []


        def run():
            data = _relation_bytes(LEVELS, lambda level: level % 7 == 0)
            deep = _pyspark._proto_parse_deep(proto.Relation, data)
            copied = copy.deepcopy(deep)
            results.append(copied == deep and copied.SerializeToString() == data)
            again = proto.Relation()
            again.CopyFrom(copied)
            results.append(again == copied and again.ByteSize() == len(data))
            try:
                proto.Plan(root=again)
            except DecodeError:
                results.append(True)
            copied.DiscardUnknownFields()
            kept = _relation_bytes(LEVELS, lambda level: level % 7 == 0 and 2 * level >= 64)
            results.append(copied.SerializeToString() == kept and again.SerializeToString() == data)
            walk = again
            for _ in range(LEVELS):
                walk = walk.limit.input
            # (The Relation at the bottom, of the level 30000, holds no unknown field.)
            results.append(walk.ByteSize() == 0 and not walk.HasField("limit"))
            del deep, copied, again, walk


        threading.stack_size(512 * 1024)
        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        print(json.dumps(results))
        """
    )
    assert _run_script(script) == [True] * 5


def _materialization_cases(cls):
    """Operations on constructor copies that make the objects of their fields as they change them
    (an unset child that a copy fills, a merge into an unset child, a parse into a copy, a second
    constructor of a copy, a cleared field, a oneof, the methods of their containers and maps, a
    merge of copies, `ListFields`, a Timestamp), and what upb's give: the bytes of the messages and
    the identity of the objects that they give."""
    relation = cls("spark.connect.Relation")(
        read={"data_source": {"format": "csv", "paths": ["a"], "options": {"k": "v"}}}
    )
    relation.limit.limit = 3

    def b(m):
        return m.SerializeToString(deterministic=True)

    def unset_child_copied_into():
        plan = cls("spark.connect.Plan")()
        root = plan.root
        plan.CopyFrom(cls("spark.connect.Plan")(root=relation))
        return [plan.root is root, b(plan), b(root), root.HasField("limit")]

    def merge_into_unset_child():
        plan = cls("spark.connect.Plan")()
        root = plan.root
        root.MergeFrom(cls("spark.connect.Plan")(root=relation).root)
        return [plan.root is root, b(plan)]

    def parse_into_copy():
        plan = cls("spark.connect.Plan")(root=relation)
        other = cls("spark.connect.Plan")(root={"limit": {"limit": 7, "input": {"range": {"end": 2}}}})
        plan.MergeFromString(other.SerializeToString())
        return [b(plan), plan.root.limit.input.range.end]

    def constructor_again():
        references = cls("spark.connect.WithRelations")(references=[relation])
        references.__init__(references=[relation, {"range": {"end": 1}}])
        first = references.references[0]
        references.__init__(references=[relation])
        return [len(references.references), references.references[0] is first, b(references)]

    def map_constructor_again():
        expression = cls("spark.connect.Expression")
        sql = cls("spark.connect.SQL")(named_arguments={"a": expression(literal={"integer": 1})})
        sql.__init__(named_arguments={"b": expression(literal={"integer": 2})})
        return [sorted(sql.named_arguments), b(sql)]

    def cleared():
        plan = cls("spark.connect.Plan")(root=relation)
        root = plan.root
        plan.ClearField("root")
        return [plan.HasField("root"), root.HasField("limit"), b(root), plan.root is root]

    def oneof():
        expression = cls("spark.connect.Expression")
        function = expression(unresolved_function={"function_name": "f"})
        other = expression(unresolved_function=function.unresolved_function)
        other.literal.integer = 1
        return [other.WhichOneof("expr_type"), other.HasField("unresolved_function"), b(other)]

    def repeated():
        references = cls("spark.connect.WithRelations")(references=[relation, relation])
        container = references.references
        container[0].limit.limit = 1
        del container[1]
        container.add().range.end = 5
        again = cls("spark.connect.WithRelations")(references=list(references.references))
        container.sort(key=lambda m: m.ByteSize())
        return [b(references), b(again), references.references is container]

    def maps():
        expression = cls("spark.connect.Expression")
        sql = cls("spark.connect.SQL")(
            named_arguments={"a": expression(literal={"integer": 1}), "b": expression(literal={})}
        )
        held = sql.named_arguments
        held["a"].literal.integer = 5
        del held["b"]
        held["c"].literal.boolean = True
        again = cls("spark.connect.SQL")(named_arguments=sql.named_arguments)
        return [b(sql), b(again), sorted(held), held is sql.named_arguments]

    def merges():
        target = cls("spark.connect.Plan")(root={"limit": {"input": {"range": {"end": 1}}}})
        target.MergeFrom(cls("spark.connect.Plan")(root=relation))
        references = cls("spark.connect.WithRelations")(references=[relation])
        references.MergeFrom(cls("spark.connect.WithRelations")(references=[relation, relation]))
        expression = cls("spark.connect.Expression")
        sql = cls("spark.connect.SQL")(named_arguments={"a": expression(literal={"integer": 1})})
        sql.MergeFrom(cls("spark.connect.SQL")(named_arguments={"b": expression(literal={})}))
        return [b(target), b(references), b(sql)]

    def listed():
        plan = cls("spark.connect.Plan")(root=relation)
        fields = plan.ListFields()
        return [[f.name for f, _ in fields], fields[0][1] is plan.root, b(fields[0][1])]

    def timestamps():
        event = cls("spark.connect.PipelineEvent")(
            timestamp=datetime.datetime(2020, 1, 2, 3, 4, 5, 678000)
        )
        again = cls("spark.connect.PipelineEvent")(timestamp=event.timestamp)
        event.timestamp.seconds = 1
        return [b(event), b(again), again.timestamp.ToJsonString()]

    cases = [
        unset_child_copied_into,
        merge_into_unset_child,
        parse_into_copy,
        constructor_again,
        map_constructor_again,
        cleared,
        oneof,
        repeated,
        maps,
        merges,
        listed,
        timestamps,
    ]
    return {case.__name__: _outcome(case) for case in cases}


def test_fields_of_constructor_copies_change_as_upbs(upb):
    # The operations that change a field of a constructor copy, or read it to change it, make its
    # object first and change that: of an unset child that CopyFrom fills, MergeFrom, a parse, a
    # second constructor of the message, ClearField, a member of a oneof, the methods of the
    # containers and the maps of a copy, and ListFields, as upb's change them.
    assert _materialization_cases(ours) == _materialization_cases(upb)


def _kind_copy_cases(cls):
    """Constructor copies, CopyFrom and copy.deepcopy of the messages of the plugin's files of
    proto2 and editions, of required fields, groups, delimited messages, closed enums and maps of
    messages, and what each reads of them: the required fields that they lack, their bytes, their
    text and their JSON."""
    from google.protobuf import json_format

    legacy, required = cls("thirdparty.legacy.Legacy"), cls("thirdparty.legacy.Required")
    featured, strict = cls("thirdparty.editions.Featured"), cls("thirdparty.editions.Strict")

    def source():
        message = legacy(id="x", colors=[1, 2], packed_colors=[3])
        message.required.next.next.SetInParent()
        message.many_required.add(a=1)
        message.many_required.add()
        message.item.x, message.item.label = 7, "l"
        message.entry.add(key=1).value.sql.query = "q"
        message.by_name["k"].count = 3
        message.defaults.SetInParent()
        message.tint = 2
        return message

    def copied():
        s = source()
        return legacy(
            id="y",
            required=s.required,
            many_required=list(s.many_required),
            item=s.item,
            entry=list(s.entry),
            by_name={"k": s.by_name["k"], "j": s.defaults},
            defaults=s.defaults,
        )

    def deep():
        return copy.deepcopy(source())

    def copied_from():
        target = legacy()
        target.CopyFrom(source())
        return target

    def editions():
        inner = featured(required="r", packed=[1, 2], expanded=[3, 4], implicit=0, explicit=0)
        inner.delimited.value = 5
        inner.delimited_values.add(value=6)
        return featured(
            required="s",
            delimited=inner.delimited,
            delimited_values=list(inner.delimited_values),
        ), strict(inner=inner.delimited), strict(inner=featured.Inner())

    out = []
    for make in (copied, deep, copied_from):
        out += [
            make().IsInitialized(),
            make().FindInitializationErrors(),
            _outcome(lambda: make().SerializeToString(deterministic=True)),
            make().SerializePartialToString(deterministic=True),
            _outcome(lambda: make().ByteSize()),
            str(make()),
            # (Of the order of its keys, as a map iterates them in another order of upb's.)
            json.loads(json_format.MessageToJson(make())),
            make() == legacy.FromString(make().SerializePartialToString()),
        ]
    for message in editions():
        out += [
            message.FindInitializationErrors(),
            message.SerializePartialToString(deterministic=True),
            str(message),
            json.loads(json_format.MessageToJson(message)),
        ]
    nested = required(a=1, next=required(next=required()))
    out += [
        nested.FindInitializationErrors(),
        _outcome(lambda: nested.SerializeToString(deterministic=True)),
    ]
    return out


def test_copies_of_messages_of_proto2_and_editions_are_upbs(kinds):
    # The copies of messages of proto2 and editions are read as upb's: the required fields that
    # they and their messages lack (also of LEGACY_REQUIRED), the EncodeError of their bytes, their
    # groups and delimited messages, their closed enums and maps of messages, their text and JSON.
    *_, upb = kinds
    assert _kind_copy_cases(ours) == _kind_copy_cases(upb)


# The merges and copies of a message into a message that the value holds (SPARK-62270): a message
# below the value (an unset child of it too), the value itself, or one above it. upb serializes the
# value first (`MergeFrom`), or reads it as it is when it reads it (a constructor, `add`, `append`,
# `extend`), so the merge reads what the value was then; upb's `CopyFrom` into a message below the
# value, and the copies into a map entry below it, end its process (of a recursion of its copy).

_ALIAS_NODE_FILE = "tests/alias/node.proto"


def _alias_node_file():
    """The serialized file of `tests.alias.Node`, a message of a plugin's file (which imports Spark
    Connect's relations.proto) that holds itself in each kind of field: a map, a repeated field, a
    message field, a oneof member, and its repeated scalars."""
    from google.protobuf import descriptor_pb2

    field = descriptor_pb2.FieldDescriptorProto
    file = descriptor_pb2.FileDescriptorProto(
        name=_ALIAS_NODE_FILE, package="tests.alias", dependency=["spark/connect/relations.proto"], syntax="proto3"
    )
    node = file.message_type.add(name="Node")
    entry = node.nested_type.add(name="ChildrenEntry")
    entry.options.map_entry = True
    entry.field.add(name="key", number=1, type=field.TYPE_STRING, label=field.LABEL_OPTIONAL)
    entry.field.add(
        name="value", number=2, type=field.TYPE_MESSAGE, label=field.LABEL_OPTIONAL, type_name=".tests.alias.Node"
    )
    message, repeated, optional = field.TYPE_MESSAGE, field.LABEL_REPEATED, field.LABEL_OPTIONAL
    node.field.add(name="children", number=1, type=message, label=repeated, type_name=".tests.alias.Node.ChildrenEntry")
    node.field.add(name="items", number=2, type=message, label=repeated, type_name=".tests.alias.Node")
    node.field.add(name="next", number=3, type=message, label=optional, type_name=".tests.alias.Node")
    node.field.add(name="v", number=4, type=field.TYPE_INT32, label=optional)
    node.field.add(name="input", number=5, type=message, label=optional, type_name=".spark.connect.Relation")
    node.oneof_decl.add(name="kind")
    node.field.add(name="left", number=6, type=message, label=optional, type_name=".tests.alias.Node", oneof_index=0)
    node.field.add(name="w", number=7, type=field.TYPE_INT32, label=optional, oneof_index=0)
    node.field.add(name="vs", number=8, type=field.TYPE_INT32, label=repeated)
    return file.SerializeToString()


def _alias_classes(which):
    """The lookup of the classes by full name of `which` ("ours" or "upb"), of `tests.alias.Node`
    too: the runtime's class of the module that protoc generates of its file, upb's of a pool of its
    own."""
    from google.protobuf import descriptor_pool

    serialized = _alias_node_file()
    if which == "upb":
        node = _upb_classes_of([(serialized, "tests.alias.node_pb2")])["tests.alias.Node"]
        classes = _upb_classes()
    else:
        pool = descriptor_pool.Default()
        try:
            added = pool.FindFileByName(_ALIAS_NODE_FILE)
        except KeyError:
            added = pool.AddSerializedFile(serialized)
        node = _build_generated_module(added)["Node"]
        classes = ours
    return lambda full_name: node if full_name == "tests.alias.Node" else classes(full_name)


def _hex(message):
    return message.SerializeToString(deterministic=True).hex()


def _alias_cases():
    """The cases of the merges and copies into a message that the value holds, each a function of
    the lookup of the classes of a runtime, of what it gives of the messages after (their bytes and
    the identity of the messages that Python holds)."""

    def deep(R, levels):
        r = R()
        m = r
        for _ in range(levels):
            m = m.limit.input
        m.range.end = 1
        return r, m

    def merge_into_stub_below(C):
        r = C("spark.connect.Relation")()
        r.limit.limit = 1
        t = r.limit.input
        t.MergeFrom(r)
        return [_hex(r), r.limit.input is t, _hex(t)]

    def merge_into_set_below(C):
        r = C("spark.connect.Relation")(limit={"limit": 1, "input": {"range": {"end": 2}}})
        t = r.limit.input
        t.MergeFrom(r)
        return [_hex(r), r.limit.input is t]

    def merge_into_stub_of_oneof_sibling(C):
        # Setting the unset child in its parent clears the member of the oneof that the value holds.
        r = C("spark.connect.Relation")(limit={"limit": 1})
        t = r.project.input
        t.MergeFrom(r)
        return [_hex(r), r.project.input is t, r.HasField("limit")]

    def merge_into_stub_of_stub(C):
        r = C("spark.connect.Relation")(limit={"limit": 1})
        t = r.limit.input.limit.input
        t.MergeFrom(r)
        return [_hex(r), r.limit.input.limit.input is t]

    def merge_into_below_repeated(C):
        d = C("spark.connect.DataType")(struct={"fields": [{"name": "a", "data_type": {"string": {}}}]})
        t = d.struct.fields[0].data_type
        t.MergeFrom(d)
        return [_hex(d), d.struct.fields[0].data_type is t]

    def merge_into_below_map(C):
        n = C("tests.alias.Node")(v=1)
        n.children["a"].v = 2
        child = n.children["a"]
        t = child.next
        t.MergeFrom(n)
        return [_hex(n), n.children["a"] is child, child.next is t]

    def merge_into_map_value_and_item(C):
        n = C("tests.alias.Node")(v=1, items=[{"v": 2}])
        child, item = n.children["a"], n.items[0]
        child.MergeFrom(n)
        item.MergeFrom(n)
        return [_hex(n), n.children["a"] is child, n.items[0] is item]

    def merge_into_oneof_member_of_value(C):
        n = C("tests.alias.Node")(left={"v": 1})
        t = n.left
        t.MergeFrom(n)
        return [_hex(n), n.left is t]

    def merge_into_stub_below_at_the_limit(C):
        # A value of the levels that a parse takes, whose unset child is one level below its
        # deepest message: upb reads the value before the child is set, so it takes it.
        out = []
        for levels in (48, 49, 50):
            r, m = deep(C("spark.connect.Relation"), levels)
            t = m.limit.input
            out.append([levels, _outcome(lambda: t.MergeFrom(r)), _hex(r)])
        return out

    def merge_from_below(C):
        r = C("spark.connect.Relation")(
            limit={"limit": 1, "input": {"limit": {"limit": 2, "input": {"range": {"end": 3}}}}}
        )
        r.MergeFrom(r.limit.input)
        n = C("tests.alias.Node")(v=1, items=[{"v": 2, "items": [{"v": 3}]}], next={"vs": [4], "next": {"v": 5}})
        n.children["k"].next.v = 6
        n.MergeFrom(n.items[0])
        n.MergeFrom(n.next)
        n.MergeFrom(n.children["k"])
        return [_hex(r), _hex(n)]

    def merge_from_itself(C):
        n = C("tests.alias.Node")(v=1, items=[{"v": 2}], vs=[3], next={"items": [{"v": 4}]})
        n.children["a"].v = 5
        n.MergeFrom(n)
        return _hex(n)

    def copy_from_below(C):
        r = C("spark.connect.Relation")(
            limit={"limit": 1, "input": {"limit": {"limit": 2, "input": {"range": {"end": 3}}}}}
        )
        inner = r.limit.input
        r.CopyFrom(inner)
        n = C("tests.alias.Node")(v=1, items=[{"v": 2, "items": [{"v": 3}]}])
        item = n.items[0]
        n.CopyFrom(item)
        return [_hex(r), _hex(inner), _hex(n), _hex(item)]

    def init_message_field_of_holder(C):
        r = C("spark.connect.Relation")(limit={"limit": 1})
        limit = r.limit
        limit.__init__(input=r)
        return [_hex(r), r.limit is limit]

    def init_stub_of_holder(C):
        # The unset child is set in its parent first, so the value holds it.
        r = C("spark.connect.Relation")(limit={"limit": 1})
        t = r.limit.input
        t.__init__(limit=r.limit)
        return [_hex(r), r.limit.input is t]

    def init_merges_into_the_field(C):
        R = C("spark.connect.Relation")
        r = R(limit={"limit": 1})
        r.__init__(limit={"input": {"range": {"end": 2}}})
        s = R(limit={"limit": 1})
        s.__init__(limit=R(limit={"input": {"range": {"end": 2}}}).limit)
        p = R(project={"expressions": [{"literal": {"integer": 1}}]})
        p.__init__(project=p.project)
        return [_hex(r), _hex(s), _hex(p)]

    def init_in_the_order_of_the_keywords(C):
        N = C("tests.alias.Node")
        n = N()
        n.next.__init__(v=7, next=n)
        m = N()
        m.next.__init__(next=m, v=7)
        o = N(v=1, items=[{"v": 2}])
        o.next.__init__(next=o, v=3)
        return [_hex(n), _hex(m), _hex(o)]

    def init_sets_an_unset_child(C):
        R = C("spark.connect.Relation")
        out = []
        for kwargs in ({"range": {"end": 1}}, {}, {"range": None}):
            r = R()
            t = r.limit.input
            t.__init__(**kwargs)
            out.append([_hex(r), r.limit.input is t, r.HasField("limit")])
        r = R()
        t = r.limit.input
        t.__init__()
        out.append([_hex(r), r.HasField("limit")])
        return out

    def init_mappings_of_holder(C):
        N = C("tests.alias.Node")
        n = N(v=1)
        n.next.__init__(next={"next": n})
        m = N(v=1)
        m.next.__init__(children={"a": {"next": m}})
        return [_hex(n), _hex(m)]

    def init_repeated_of_holder(C):
        N = C("tests.alias.Node")
        n = N(v=1)
        n.next.__init__(items=[n, n.next])
        m = N(v=1)
        m.__init__(items=[m, m])
        o = N(v=1)
        o.next.__init__(items=(x for x in (o, o.next)))
        return [_hex(n), _hex(m), _hex(o)]

    def add_of_holder(C):
        N = C("tests.alias.Node")
        n = N(v=1)
        first = n.next.items.add(v=2, next=n)
        m = N(v=1)
        second = m.next.items.add(next=m, v=2)
        return [_hex(n), n.next.items[0] is first, _hex(m), m.next.items[0] is second]

    def append_extend_insert_of_holder(C):
        N = C("tests.alias.Node")
        n = N(v=1)
        item = n.next.items.append(n)
        m = N(v=1)
        m.items.append(m)
        o = N(v=1)
        o.next.items.extend([o, o.next])
        p = N(v=1, items=[{"v": 2}])
        p.items.insert(0, p)
        d = C("spark.connect.DataType")(struct={"fields": [{"name": "a"}]})
        f = d.struct.fields[0]
        f.data_type.struct.fields.append(f)
        return [_hex(n), n.next.items[0] is item if item is not None else None, _hex(m), _hex(o), _hex(p), _hex(d)]

    def extend_and_merge_between_containers(C):
        N = C("tests.alias.Node")
        n = N(items=[{"v": 1}, {"v": 2}])
        n.items[0].items.extend(n.items)
        m = N(items=[{"v": 1}, {"v": 2}])
        m.items[0].items.MergeFrom(m.items)
        o = N(vs=[1, 2])
        o.vs.extend(o.vs)
        o.vs.MergeFrom(o.vs)
        return [_hex(n), _hex(m), list(o.vs)]

    def maps_of_holder(C):
        N = C("tests.alias.Node")
        n = N()
        n.children["a"].v = 1
        n.children.MergeFrom(n.children)
        m = N(v=1)
        m.children.get_or_create("a").MergeFrom(m)
        o = N(v=1)
        p = N(children={"a": o, "b": o.next})
        return [_hex(n), _hex(m), _hex(p), _hex(o)]

    def parses_of_its_own_bytes(C):
        r = C("spark.connect.Relation")(limit={"limit": 1})
        t = r.limit.input
        t.MergeFromString(r.SerializeToString())
        n = C("tests.alias.Node")(v=1, items=[{"v": 2}])
        n.MergeFromString(n.SerializeToString())
        s = C("spark.connect.Relation")(limit={"limit": 1})
        u = s.limit.input
        u.ParseFromString(s.SerializeToString())
        return [_hex(r), r.limit.input is t, _hex(n), _hex(s), s.limit.input is u]

    def init_of_too_deep_values(C):
        R = C("spark.connect.Relation")
        r, _ = deep(R, 60)
        plan = C("spark.connect.Plan")(root={"limit": {"limit": 9}})
        out = [_outcome(lambda: plan.__init__(root=r)), _hex(plan)]
        w = C("spark.connect.WithRelations")(references=[{"limit": {"limit": 9}}])
        out += [_outcome(lambda: w.__init__(references=[R(limit={"limit": 1}), r])), _hex(w)]
        return out

    cases = [
        merge_into_stub_below, merge_into_set_below, merge_into_stub_of_oneof_sibling, merge_into_stub_of_stub,
        merge_into_below_repeated, merge_into_below_map, merge_into_map_value_and_item,
        merge_into_oneof_member_of_value, merge_into_stub_below_at_the_limit, merge_from_below, merge_from_itself,
        copy_from_below, init_message_field_of_holder, init_stub_of_holder, init_merges_into_the_field,
        init_in_the_order_of_the_keywords, init_sets_an_unset_child, init_mappings_of_holder, init_repeated_of_holder,
        add_of_holder, append_extend_insert_of_holder, extend_and_merge_between_containers, maps_of_holder,
        parses_of_its_own_bytes, init_of_too_deep_values,
    ]
    return {case.__name__: case for case in cases}


def _alias_crash_cases():
    """The cases of which upb ends its process (of a recursion of its copy, or of a container that
    grows as it is read), each a function of the lookup of the classes of a runtime, and the copy
    that the runtime makes first (what a copy of the value taken before gives), of the same lookup."""

    def copy_into_stub_below(C):
        r = C("spark.connect.Relation")(limit={"limit": 1})
        t = r.limit.input
        t.CopyFrom(r)
        return [_hex(r), r.limit.input is t]

    def copy_into_set_below(C):
        r = C("spark.connect.Relation")(limit={"limit": 1, "input": {"range": {"end": 2}}})
        t = r.limit.input
        t.CopyFrom(r)
        return [_hex(r), r.limit.input is t]

    def copy_into_stub_of_oneof_sibling(C):
        r = C("spark.connect.Relation")(limit={"limit": 1})
        t = r.to_df.input
        t.CopyFrom(r)
        return [_hex(r), r.to_df.input is t]

    def copy_into_map_value(C):
        # The map has the entry (empty) once it is read, so the copy holds it.
        n = C("tests.alias.Node")(v=1)
        t = n.children["a"]
        t.CopyFrom(n)
        return [_hex(n), n.children["a"] is t]

    def copy_into_repeated_item(C):
        n = C("tests.alias.Node")(v=1, items=[{"v": 2}])
        t = n.items[0]
        t.CopyFrom(n)
        return [_hex(n), n.items[0] is t]

    def map_value_copy_from_holder(C):
        n = C("tests.alias.Node")(v=1)
        n.children["a"].next.CopyFrom(n)
        return _hex(n)

    def init_map_of_holder(C):
        # The value is copied before the map takes the entry.
        n = C("tests.alias.Node")(v=1)
        n.next.__init__(children={"a": n})
        return _hex(n)

    def map_merge_into_inner(C):
        n = C("tests.alias.Node")()
        n.children["a"].v = 1
        inner = n.children["a"].children
        inner.MergeFrom(n.children)
        return _hex(n)

    def extend_itself(C):
        # As a list's extend of itself: the messages that it has first.
        n = C("tests.alias.Node")(items=[{"v": 1}, {"v": 2}])
        n.items.extend(n.items)
        m = C("tests.alias.Node")(items=[{"v": 1}, {"v": 2}])
        m.items.MergeFrom(m.items)
        return [_hex(n), _hex(m)]

    def expected(C):
        R, N = C("spark.connect.Relation"), C("tests.alias.Node")
        return {
            "copy_into_stub_below": [_hex(R(limit={"limit": 1, "input": {"limit": {"limit": 1}}})), True],
            "copy_into_set_below": [
                _hex(R(limit={"limit": 1, "input": {"limit": {"limit": 1, "input": {"range": {"end": 2}}}}})),
                True,
            ],
            "copy_into_stub_of_oneof_sibling": [_hex(R(to_df={"input": {"limit": {"limit": 1}}})), True],
            "copy_into_map_value": [_hex(N(v=1, children={"a": {"v": 1, "children": {"a": {}}}})), True],
            "copy_into_repeated_item": [_hex(N(v=1, items=[{"v": 1, "items": [{"v": 2}]}])), True],
            "map_value_copy_from_holder": _hex(N(v=1, children={"a": {"next": {"v": 1, "children": {"a": {}}}}})),
            "init_map_of_holder": _hex(N(v=1, next={"children": {"a": {"v": 1, "next": {}}}})),
            "map_merge_into_inner": _hex(N(children={"a": {"v": 1, "children": {"a": {"v": 1}}}})),
            "extend_itself": [_hex(N(items=[{"v": 1}, {"v": 2}, {"v": 1}, {"v": 2}]))] * 2,
        }

    cases = [
        copy_into_stub_below, copy_into_set_below, copy_into_stub_of_oneof_sibling, copy_into_map_value,
        copy_into_repeated_item, map_value_copy_from_holder, init_map_of_holder, map_merge_into_inner, extend_itself,
    ]
    return {case.__name__: case for case in cases}, expected


def _run_alias_cases(which, names, crash=False):
    """The outcomes of the cases `names` of the runtime `which`, of what each raises too (a panic of
    the runtime as well)."""
    C = _alias_classes(which)
    cases = _alias_crash_cases()[0] if crash else _alias_cases()
    out = {}
    for name in names:
        try:
            out[name] = ("ok", cases[name](C))
        except BaseException as e:  # noqa: BLE001 (a PanicException of the runtime too)
            out[name] = ("raise", type(e).__name__, str(e)[:300])
    return out


def _capped_to_4_gb():
    """The limits of a process of a case that may recurse without bound (the `preexec_fn` of its
    `subprocess.run`): an address space of at most 4 GB, which it fails of, rather than the memory
    of the machine, and no core dump."""
    import resource

    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = 4_000_000 * 1024
    resource.setrlimit(resource.RLIMIT_AS, (cap if hard == resource.RLIM_INFINITY else min(cap, hard), hard))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _alias_process(which, names, crash=False):
    """The return code and the outcomes of the cases `names` of the runtime `which` run in a process
    of its own, of an address space of at most 4 GB and no core dump: a case that recursed without
    bound fails there, of the memory that it runs out of, rather than in this process."""
    script = textwrap.dedent(
        f"""
        import json
        from tests.test_proto_rust_offline import _run_alias_cases
        print(json.dumps(_run_alias_cases({which!r}, {list(names)!r}, crash={crash!r})))
        """
    )

    root = pathlib.Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
        cwd=str(root),
        preexec_fn=_capped_to_4_gb,
    )
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    return result.returncode, (json.loads(lines[-1]) if lines else None), result.stderr[-2000:]


def test_merges_into_a_message_that_the_value_holds_are_upbs(upb):
    # A merge into a message below its value (an unset child of it too), into the value itself or
    # from a message below it, and the constructors, `add`, `append`, `extend` and the maps of a
    # value that holds the message, are upb's, of the same messages: upb serializes the value of a
    # `MergeFrom` first, so the merge takes what the value was then, where the runtime's merge
    # read the value as the merge changed it, of a recursion without bound of a message below it.
    # A second `__init__` sets an unset child in its parent first and merges the values of its
    # message fields into the messages that the fields have, in the order of the keywords; `add`
    # appends the message first. Each runtime runs in a process of its own, of 4 GB.
    names = list(_alias_cases())
    code, mine, err = _alias_process("ours", names)
    assert code == 0 and mine is not None, err
    code, theirs, err = _alias_process("upb", names)
    assert code == 0 and theirs is not None, err
    assert {name for name in names if mine[name] != theirs[name]} == set(), {
        name: (mine[name], theirs[name]) for name in names if mine[name] != theirs[name]
    }
    assert all(outcome[0] == "ok" for outcome in mine.values()), mine


def test_copies_into_a_message_that_the_value_holds_copy_the_value_first(upb):
    # Where upb ends its process (its CopyFrom into a message below the value, the copies of a map's
    # constructor and MergeFrom into an entry below the value, and the extend of a repeated field
    # by itself, which upb reads as it grows), the runtime copies the value first: the copy of the
    # value as it was, as a CopyFrom of a copy taken before gives, and a repeated field extended by
    # itself takes the messages that it has first, as a list. upb runs each case in a process of its
    # own, which upb's crash ends (another result of upb's must be the runtime's).
    cases, expected = _alias_crash_cases()
    names = list(cases)
    code, mine, err = _alias_process("ours", names, crash=True)
    assert code == 0 and mine is not None, err
    assert mine == {name: ["ok", value] for name, value in expected(_alias_classes("ours")).items()}
    for name in names:
        code, theirs, err = _alias_process("upb", [name], crash=True)
        assert code < 0 or (theirs is not None and theirs[name] == mine[name]), (name, code, theirs, err)


def _reaching_cases(C):
    """Python code of a value of a constructor (an `__iter__`, a `__next__`, an `__index__`) that
    reaches the message under construction through the garbage collector's `get_objects`, and reads
    or changes it."""
    N = C("tests.alias.Node")
    marker = 424242

    def under_construction(before):
        # The new message of the marker, set first (upb's wrappers of its children are new too).
        found = [
            o
            for o in gc.get_objects()
            if type(o) is N and id(o) not in before and o.WhichOneof("kind") == "w" and o.w == marker
        ]
        assert len(found) == 1, len(found)
        return found[0]

    def existing():
        return {id(o) for o in gc.get_objects() if type(o) is N}

    def iter_changes_message():
        before, seen = existing(), []

        class It:
            def __iter__(self):
                m = under_construction(before)
                seen.append(_hex(m))
                m.items.add(v=8)
                m.vs.append(7)
                return iter([N(v=2)])

        return [_hex(N(w=marker, v=1, items=It(), vs=[3])), seen]

    def next_reads_message():
        before, seen = existing(), []

        def gen():
            for v in (2, 3):
                m = under_construction(before)
                seen.append([_hex(m), len(m.items)])
                yield N(v=v)

        return [_hex(N(w=marker, v=1, items=gen())), seen]

    def next_yields_message_itself():
        before = existing()

        def gen():
            yield N(v=2)
            yield under_construction(before)

        return _hex(N(w=marker, v=1, items=gen()))

    def index_changes_message():
        before, seen = existing(), []

        class Index:
            def __index__(self):
                m = under_construction(before)
                seen.append(_hex(m))
                m.vs.append(99)
                m.v = 5
                return 4

        return [_hex(N(w=marker, v=1, vs=[1, Index(), 3])), seen]

    def index_in_a_mapping():
        before, seen = existing(), []

        class Index:
            def __index__(self):
                m = under_construction(before)
                seen.append(_hex(m))
                m.next.next.v = 6
                return 4

        return [_hex(N(w=marker, v=1, next={"vs": [Index()], "v": 3})), seen]

    def index_clears_message():
        before = existing()

        class Index:
            def __index__(self):
                under_construction(before).Clear()
                return 4

        return _hex(N(w=marker, v=1, items=[N(v=2)], next={"v": Index()}))

    def index_after_a_list_of_messages():
        before = existing()

        class Index:
            def __index__(self):
                return len(under_construction(before).items)

        return _hex(N(w=marker, items=[N(v=2), N(v=3)], v=Index()))

    cases = [
        iter_changes_message, next_reads_message, next_yields_message_itself, index_changes_message,
        index_in_a_mapping, index_clears_message, index_after_a_list_of_messages,
    ]
    return {case.__name__: _outcome(case) for case in cases}


def test_code_of_a_value_that_reaches_the_message_under_construction_finds_upbs(upb):
    # The constructor holds no borrow of its message while the Python code of a value runs, and
    # sets the fields in upb's order: code that reaches the message under construction (of the
    # garbage collector, which tracks it, as upb's) finds the fields before it set and changes
    # it as upb's, where it raised the runtime's PanicException ("Already mutably borrowed").
    assert _reaching_cases(_alias_classes("ours")) == _reaching_cases(_alias_classes("upb"))


def _partial_cases(C):
    """What a second `__init__` (and `add`) leaves of a repeated field whose values fail midway, and
    of an unset child, as upb's."""
    N, R = C("tests.alias.Node"), C("spark.connect.Relation")

    def after(make, change):
        m = make()
        outcome = _outcome(lambda: change(m))
        return [outcome, _hex(m)]

    def gen_failing(*values):
        yield from values
        raise RuntimeError("midway")

    def unset_child(change):
        r = R()
        t = r.limit.input
        return [_outcome(lambda: change(t)), r.HasField("limit"), _hex(r)]

    return {
        "scalars, a bad value": after(lambda: N(vs=[0]), lambda m: m.__init__(vs=[1, 2, "x"])),
        "scalars, a failing iterator": after(lambda: N(vs=[0]), lambda m: m.__init__(vs=gen_failing(1, 2))),
        "scalars, not iterable": _outcome(lambda: N(vs=5)),
        "messages, a bad mapping": after(
            lambda: N(items=[{"v": 0}]), lambda m: m.__init__(items=[{"v": 1}, {"nope": 1}])
        ),
        "messages, a bad value of a mapping": after(
            lambda: N(items=[{"v": 0}]), lambda m: m.__init__(items=[{"v": 1}, {"v": 2, "w": "bad"}])
        ),
        "messages, an int": after(lambda: N(items=[{"v": 0}]), lambda m: m.__init__(items=[N(v=1), 5, N(v=3)])),
        "messages, another type": after(lambda: N(items=[{"v": 0}]), lambda m: m.__init__(items=[N(v=1), R()])),
        "messages, a failing iterator": after(
            lambda: N(items=[{"v": 0}]), lambda m: m.__init__(items=gen_failing(N(v=1), {"v": 2}))
        ),
        "add, a bad name": after(N, lambda m: m.items.add(v=1, nope=2)),
        "add, a bad value": after(lambda: N(items=[{"v": 7}]), lambda m: m.items.add(v=1, w="bad")),
        "a name of no field, of None": _outcome(lambda: R(nope=None)),
        "an unset child, of **{}": unset_child(lambda t: t.__init__(**{})),
        "an unset child, of None": unset_child(lambda t: t.__init__(range=None)),
        "an unset child, of no field": unset_child(lambda t: t.__init__(nope=1)),
        "an unset child, of no arguments": unset_child(lambda t: t.__init__()),
    }


def test_a_second_init_keeps_what_upbs_keeps_of_a_failure(upb):
    # A second `__init__` appends the values of a repeated message field in turn, so a failure
    # midway keeps those before it (and, of a value that is not a mapping, the new message that it
    # failed to merge into), and those of a repeated scalar field as the container's `extend`
    # (none, of a failure); `add` removes its message of a failure. A name of no field is upb's
    # ValueError of a value of None too, and keyword arguments (even none) set an unset child in its
    # parent first, as upb's `__init__` does.
    assert _partial_cases(_alias_classes("ours")) == _partial_cases(_alias_classes("upb"))



def _plan_of_an_any(count):
    """Plans of a relation of an `Any` of a known type (which protobuf's text format expands to its
    message), each the copy of a constructor, whose `root` is native until it is read."""
    inner = proto.Relation(range={"end": 1})
    rel = proto.Relation()
    rel.extension.Pack(inner)
    rel.common.plan_id = 3
    return rel, [proto.Plan(root=rel) for _ in range(count)]


def test_a_first_read_of_a_collection_of_the_text_format_finds_the_message():
    # The drop-in's writer of protobuf's text format (`_proto_message_to_string`, of the client's
    # plan logging) writes a native copy of the message, so a collection that its expansion of an
    # Any starts, whose callback reads the native field of the message for the first time, finds it
    # free, as it does of protobuf's `text_format` and of `str()`: the writer held the message
    # borrowed, of the runtime's PanicException ("Already borrowed"). (Without protobuf, of the
    # drop-in's writer and of `str()` only.)
    try:
        from google.protobuf import text_format
    except ImportError:
        text_format = None
    writers = [("_proto_message_to_string", _pyspark._proto_message_to_string), ("str", str)]
    if text_format is not None:
        writers.append(("text_format", text_format.MessageToString))
    out = {}
    for name, write in writers:
        _, (plan,) = _plan_of_an_any(1)
        state = {"ran": False}

        def callback(phase, info, plan=plan, state=state):
            if phase == "start" and not state["ran"]:
                state["ran"] = True
                try:
                    root = plan.root
                    state["same"] = root is plan.root
                except BaseException as e:  # noqa: BLE001 (a PanicException too)
                    state["error"] = f"{type(e).__name__}: {e}"

        gc.collect()
        gc.callbacks.append(callback)
        threshold = gc.get_threshold()
        gc.set_threshold(1)
        try:
            text = write(plan)
        finally:
            gc.set_threshold(*threshold)
            gc.callbacks.remove(callback)
        out[name] = [state.get("error"), state.get("same"), "plan_id: 3" in text, plan.root.common.plan_id]
    assert out["_proto_message_to_string"] == [None, True, True, 3], out
    assert out.get("text_format", out["_proto_message_to_string"]) == out["_proto_message_to_string"], out
    assert out["str"][0] is None and out["str"][2:] == [True, 3], out


def test_a_first_read_of_another_thread_during_the_text_format_finds_the_message():
    # The cross-thread form: a callback of the collections of the writer gives the GIL to another
    # thread, which reads the native field of each plan for the first time while the writer
    # writes it. Each read is whole, and the field gives one object from then on.
    rel, plans = _plan_of_an_any(100)
    state = {"current": None, "done": False}
    seen = {"ok": 0, "errors": {}}

    def callback(phase, info):
        if phase == "start":
            time.sleep(0.0002)

    def reader():
        last = None
        while not state["done"]:
            i = state["current"]
            if i is None or i == last:
                time.sleep(0)
                continue
            last = i
            try:
                plans[i].root
                seen["ok"] += 1
            except BaseException as e:  # noqa: BLE001 (a PanicException too)
                key = f"{type(e).__name__}: {e}"
                seen["errors"][key] = seen["errors"].get(key, 0) + 1

    thread = threading.Thread(target=reader)
    thread.start()
    gc.callbacks.append(callback)
    threshold = gc.get_threshold()
    gc.set_threshold(1)
    try:
        for i in range(len(plans)):
            state["current"] = i
            _pyspark._proto_message_to_string(plans[i], as_one_line=True)
    finally:
        gc.set_threshold(*threshold)
        gc.callbacks.remove(callback)
        state["done"] = True
        thread.join()
    assert seen["errors"] == {} and seen["ok"] > 0, seen
    expected = proto.Plan(root=rel).SerializeToString()
    assert all(p.root is p.root and p.root.common.plan_id == 3 for p in plans)
    assert all(p.SerializeToString() == expected for p in plans)


def test_unknown_field_pairs_of_a_collection_find_the_message():
    # The unknown field pairs (protobuf's `_unknown_fields` of its pure-Python messages) are made of
    # a copy of the bytes, so a collection that the list starts, whose callback reads the message,
    # finds it free.
    unknown = b"".join(_tag(900 + i, 0) + _varint(i) for i in range(200))
    plan = proto.Plan.FromString(proto.Plan(root=proto.Relation(range={"end": 1})).SerializeToString() + unknown)
    message = proto.Plan(root=plan.root)
    message.MergeFromString(unknown)
    state = {"errors": []}

    def callback(phase, info):
        if phase == "start":
            try:
                message.root
                message.SerializeToString()
            except BaseException as e:  # noqa: BLE001 (a PanicException too)
                state["errors"].append(type(e).__name__)

    gc.callbacks.append(callback)
    threshold = gc.get_threshold()
    gc.set_threshold(1)
    try:
        pairs = _pyspark._proto_unknown_field_pairs(message)
    finally:
        gc.set_threshold(*threshold)
        gc.callbacks.remove(callback)
    assert state["errors"] == [] and len(pairs) == 200
    assert b"".join(tag + value for tag, value in pairs) == unknown



def test_threads_that_read_change_and_copy_the_same_copies_at_once_see_them_whole():
    # Threads that read, change, replace and copy the same native copies at once, of a switch
    # interval of a microsecond, so that one takes the GIL at the checkpoints of the making of
    # the objects of a container of more than 1,024 messages: one container of the field, of
    # every message of it and of both additions, and each copy of its own.
    count = 3000
    element = proto.Expression.Literal(integer=1)
    source = proto.Expression.Literal.Array(elements=[element] * count)
    other = proto.Expression.Literal.Array(elements=[proto.Expression.Literal(string="o")] * 7)
    problems = []

    def holder():
        return proto.Expression.Literal(array=source)

    def run_threads(*targets):
        threads = [threading.Thread(target=target) for target in targets]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        for round_ in range(30):
            h = holder()
            seen = [None, None]

            def reader(i, h=h, seen=seen):
                seen[i] = h.array.elements

            run_threads(lambda: reader(0), lambda: reader(1))
            if seen[0] is not seen[1] or len(seen[0]) != count or seen[0] is not h.array.elements:
                problems.append(("two readers", round_))

            h = holder()

            def adder(value, h=h):
                h.array.elements.add().integer = value

            run_threads(lambda: adder(5), lambda: adder(6))
            values = [e.integer for e in h.array.elements]
            if len(values) != count + 2 or sorted(values[-2:]) != [5, 6] or len(source.elements) != count:
                problems.append(("two adders", round_, len(values)))

            h = holder()
            got = {}

            def counter(h=h, got=got):
                got["n"] = len(h.array.elements)

            run_threads(counter, lambda h=h: h.array.CopyFrom(other))
            if got["n"] not in (count, 7) or h.array.SerializeToString() != other.SerializeToString():
                problems.append(("a reader and a CopyFrom", round_, got["n"]))

            h = holder()
            made = {}

            def changer(h=h):
                h.array.elements[0].integer = 9

            def copier(h=h, made=made):
                made["copy"] = proto.Expression(literal=h)

            run_threads(changer, copier)
            first = made["copy"].literal.array.elements[0].integer
            made["copy"].literal.array.elements[1].integer = 4
            if first not in (1, 9) or h.array.elements[0].integer != 9 or h.array.elements[1].integer != 1:
                problems.append(("a change and a copy", round_, first))
    finally:
        sys.setswitchinterval(interval)
    assert problems == [] and len(source.elements) == count and source.elements[1].integer == 1, problems


def test_a_signal_during_the_first_read_of_a_copy_leaves_it_whole():
    # The first read of a repeated message field of a copy of many messages makes their objects
    # of checkpoints, every 1,024 messages: a signal there raises the exception of its handler
    # from the read, as from the other operations of many steps, and the message stays whole, its
    # field read again in full, of one object from then on.
    import signal

    count = 200_000
    source = proto.Expression.Literal.Array(elements=[proto.Expression.Literal(integer=1)] * count)
    data = source.SerializeToString()

    class Stop(Exception):
        pass

    def handler(signum, frame):
        raise Stop()

    previous = signal.signal(signal.SIGALRM, handler)
    outcomes = []
    try:
        for attempt in range(10):
            h = proto.Expression.Literal(array=source)
            array = h.array  # of one level: its elements stay native
            signal.setitimer(signal.ITIMER_REAL, 0.0005 + attempt * 0.0002)
            try:
                len(array.elements)
                outcomes.append("completed")
            except Stop:
                outcomes.append("interrupted")
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            again = array.elements
            assert len(again) == count and again is array.elements, attempt
            assert array.SerializeToString() == data and h == proto.Expression.Literal(array=source), attempt
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert "interrupted" in outcomes, outcomes


def test_copies_of_copies_across_a_registration_are_of_the_classes_of_their_copy():
    # A copy of a copy made after another class of its type is registered clones the native values
    # that it shares at the new generation, so its objects, of any level, are of the new class,
    # where those of the first copy stay of the class registered when it was made; of a message of
    # thousands of levels, on a thread of a small stack.
    script = """
        import copy
        import json
        import threading

        import pyspark.sql.connect.proto as proto
        from pyspark import _pyspark
        from pyspark.sql.connect.proto import _rust
        from tests.test_proto_rust_offline import _relation_bytes

        LEVELS = 3000
        results = {}


        def run():
            data = _relation_bytes(LEVELS, lambda level: level % 7 == 0)
            deep = _pyspark._proto_parse_deep(proto.Relation, data)
            first = copy.deepcopy(deep)
            other = type(
                "Relation",
                (_pyspark.ProtoMessage, _rust._MessageBase),
                {"__slots__": (), "DESCRIPTOR": proto.Relation.DESCRIPTOR},
            )
            _pyspark._proto_register_classes({"spark.connect.Relation": other})
            try:
                second = copy.deepcopy(first)
                third = proto.Relation()
                third.CopyFrom(first)
                results["copies"] = second.SerializeToString() == data and third == first
                walk = second
                for _ in range(LEVELS):
                    walk = walk.limit.input
                results["the bottom of the copy of the copy"] = type(walk) is other
                walk = first
                for _ in range(100):
                    walk = walk.limit.input
                results["the objects of the first copy"] = type(walk) is proto.Relation
            finally:
                _pyspark._proto_register_classes({"spark.connect.Relation": proto.Relation})
            results["after"] = first == deep and second == deep and first.ByteSize() == len(data)


        threading.stack_size(512 * 1024)
        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        print(json.dumps(results))
        """
    assert _run_script(script) == {
        "copies": True,
        "the bottom of the copy of the copy": True,
        "the objects of the first copy": True,
        "after": True,
    }


def _clear_field_of_a_member_cases(C):
    """ClearField of a member of a oneof that the message does not hold, of an unset child of it
    that Python holds or not."""
    R = C("spark.connect.Relation")

    def run(hold, change):
        a = R()
        if hold:
            held = a.crosstab
        else:
            a.crosstab  # noqa: B018 (read, and dropped at once)
            held = None
        change(a)
        a.ClearField("crosstab")
        return [_hex(a), a.WhichOneof("rel_type"), held is not None and held is a.crosstab]

    def below(read, keep):
        # The unset child of the member is held of what Python holds below it (`read` of it),
        # or not (`keep` false: read, and dropped at once).
        a = R()
        held = read(a.crosstab) if keep else None
        if not keep:
            read(a.crosstab)
        a.show_string.vertical = True
        a.ClearField("crosstab")
        return [_hex(a), a.WhichOneof("rel_type"), held is not None]

    return {
        "held, the other set by a read": _outcome(lambda: run(True, lambda a: setattr(a.show_string, "vertical", True))),
        "held, the other of CopyFrom": _outcome(
            lambda: run(True, lambda a: a.show_string.CopyFrom(C("spark.connect.ShowString")(num_rows=3)))
        ),
        "not held": _outcome(lambda: run(False, lambda a: setattr(a.show_string, "vertical", True))),
        "held, none set": _outcome(lambda: run(True, lambda a: None)),
        "held of its unset child": _outcome(lambda: below(lambda c: c.input, True)),
        "held of an unset child two levels below": _outcome(lambda: below(lambda c: c.input.limit, True)),
        "held of a container below": _outcome(lambda: below(lambda c: c.input.project.expressions, True)),
        "held of a container of scalars below": _outcome(
            lambda: below(lambda c: c.input.read.data_source.paths, True)
        ),
        "held of a map below": _outcome(lambda: below(lambda c: c.input.read.data_source.options, True)),
        "a container below dropped": _outcome(lambda: below(lambda c: c.input.project.expressions, False)),
        "an unset child below dropped": _outcome(lambda: below(lambda c: c.input.limit, False)),
    }


def test_clear_field_of_a_member_of_an_unset_child_held_clears_its_oneof_as_upbs(upb):
    # As upb's, whose ClearField sets an unset child of the field that Python holds in its message
    # first (its `EnsureReified`), so the member of the oneof that the message holds is cleared;
    # of one that Python no longer holds (upb holds its unset children weakly), it is not. upb's
    # unset children and their containers hold their parents, so Python holds the unset child of
    # what it holds below it too: an unset child of it, of any level, or a container of one.
    assert _clear_field_of_a_member_cases(ours) == _clear_field_of_a_member_cases(upb)



def _copy_into_unset_children_cases(C):
    """CopyFrom into a message whose unset children, of several levels, Python holds."""
    P, R = C("spark.connect.Plan"), C("spark.connect.Relation")

    def rel():
        return R(
            read={"data_source": {"format": "csv", "paths": ["a"], "options": {"k": "v"}}},
            common={"plan_id": 1},
        )

    def nested():
        p = P()
        stub = p.root
        deep = stub.read
        deeper = deep.data_source
        p.CopyFrom(P(root=rel()))
        deeper.format = "json"
        return [p.root is stub, p.root.read is deep, p.root.read.data_source is deeper, _hex(p)]

    def nested_changed_at_the_middle():
        p = P()
        deep = p.root.read
        p.CopyFrom(P(root=rel()))
        deep.data_source.format = "json"
        return [p.root.read is deep, _hex(p)]

    def unset_by_the_copy():
        # The copy sets `read`; the unset child of `limit` (of the same oneof) stays unset, and
        # setting it sets the member.
        p = P()
        stub = p.root
        other = stub.limit
        p.CopyFrom(P(root=rel()))
        before = _hex(p)
        other.limit = 5
        return [p.root is stub, before, _hex(p), p.root.HasField("read")]

    def copied_twice():
        p = P()
        deep = p.root.read
        p.CopyFrom(P(root=rel()))
        p.CopyFrom(P(root=R(read={"named_table": {"unparsed_identifier": "t"}})))
        deep.data_source.format = "json"
        return [p.root.read is deep, _hex(p), _hex(deep)]

    cases = [nested, nested_changed_at_the_middle, unset_by_the_copy, copied_twice]
    return {case.__name__: _outcome(case) for case in cases}


def test_copies_into_unset_children_of_many_levels_keep_what_they_copied_as_upbs(upb):
    # The unset children that Python holds of a message that a CopyFrom sets, of any level, are
    # the messages of the copy, as upb's are (of its `SyncSubobjs`): a change of one keeps what the
    # copy set, where a change of an unset child below the first level replaced the copied
    # message with the unset child, of the change only.
    assert _copy_into_unset_children_cases(ours) == _copy_into_unset_children_cases(upb)



def test_copies_of_many_levels_of_maps_grow_the_stack_at_their_entries():
    # A native copy grows its stack every 16 levels, a map entry a level of its own: of a chain of
    # maps whose entries are at the levels of a multiple of 16, the copy grows it at the entries,
    # which it did not, so a copy of many such levels on a small stack (a thread of 256 KiB)
    # overflowed it. Of a process of its own, of 4 GB. (The plugin's type of maps of itself needs
    # protobuf's descriptor pool and builder.)
    pytest.importorskip("google.protobuf")
    script = textwrap.dedent(
        """
        import copy, json, threading
        from tests.test_proto_rust_offline import _alias_classes
        N = _alias_classes("ours")("tests.alias.Node")
        out = {}
        def run():
            n = N()
            m = n.next  # the entries at the even levels, the messages at the odd ones
            for _ in range(20000):
                m = m.children["k"]
            m.v = 7
            data = n.SerializeToString()
            twin = copy.deepcopy(n)
            out["deepcopy"] = twin.SerializeToString() == data
            other = N()
            other.CopyFrom(n)
            out["CopyFrom"] = other.SerializeToString() == data
            del twin, other, n, m
        threading.stack_size(256 * 1024)
        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        print(json.dumps(out))
        """
    )

    root = pathlib.Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=600, cwd=str(root), preexec_fn=_capped_to_4_gb
    )
    assert result.returncode == 0, result.stderr[-2000:]
    assert json.loads(result.stdout.strip().splitlines()[-1]) == {"deepcopy": True, "CopyFrom": True}


def _first_reads_of_containers():
    """The first reads of a container of a message (a repeated message field, a map, a repeated
    scalar field), each of the message and of the read."""
    return {
        "repeated": (lambda m: m.project, lambda t: t.expressions),
        "map": (lambda m: m.read.data_source, lambda t: t.options),
        "repeated scalar": (lambda m: m.read.data_source, lambda t: t.paths),
    }


def test_a_collection_during_the_first_read_of_a_container_finds_the_message():
    # The first read of a container of a message makes the container, and the weak reference of
    # its owner, with no borrow of the message held: a collection that their allocation starts,
    # whose callback reads the message, finds it free, where it raised the runtime's
    # PanicException ("Already mutably borrowed").
    out = {}
    for case, (target_of, read) in _first_reads_of_containers().items():
        target = target_of(proto.Relation())
        state = {"ran": False, "error": None}

        def callback(phase, info, target=target, state=state):
            if phase == "start" and not state["ran"]:
                state["ran"] = True
                try:
                    target.ListFields()
                except BaseException as e:  # noqa: BLE001 (a PanicException too)
                    state["error"] = f"{type(e).__name__}: {e}"

        gc.collect()
        gc.callbacks.append(callback)
        threshold = gc.get_threshold()
        gc.set_threshold(1)
        try:
            container = read(target)
        finally:
            gc.set_threshold(*threshold)
            gc.callbacks.remove(callback)
        out[case] = [state["ran"], state["error"], container is read(target)]
    assert out == {case: [True, None, True] for case in _first_reads_of_containers()}, out


def test_another_thread_during_the_first_read_of_a_container_finds_the_message():
    # The cross-thread form: a callback of the collections of the first read gives the GIL to
    # another thread, which reads the message meanwhile, whole.
    out = {}
    for case, (target_of, read) in _first_reads_of_containers().items():
        targets = [target_of(proto.Relation()) for _ in range(40)]
        state = {"current": None, "done": False}
        seen = {"ok": 0, "errors": {}}

        def reader(targets=targets, state=state, seen=seen):
            last = None
            while not state["done"]:
                i = state["current"]
                if i is None or i == last:
                    time.sleep(0)
                    continue
                last = i
                try:
                    targets[i].ListFields()
                    seen["ok"] += 1
                except BaseException as e:  # noqa: BLE001 (a PanicException too)
                    key = f"{type(e).__name__}: {e}"
                    seen["errors"][key] = seen["errors"].get(key, 0) + 1

        def callback(phase, info):
            if phase == "start":
                time.sleep(0.0002)

        thread = threading.Thread(target=reader)
        thread.start()
        gc.callbacks.append(callback)
        threshold = gc.get_threshold()
        gc.set_threshold(1)
        try:
            for i, target in enumerate(targets):
                state["current"] = i
                read(target)
        finally:
            gc.set_threshold(*threshold)
            gc.callbacks.remove(callback)
            state["done"] = True
            thread.join()
        out[case] = seen["errors"]
        assert all(read(t) is read(t) for t in targets)
    assert out == {case: {} for case in _first_reads_of_containers()}, out


def _parse_scenarios(cls):
    """Parses into messages of the classes of ``cls``: into an unset child (of two levels too),
    into a message that gave out unset children of fields that the parse sets and does not set,
    into a message of a container and of a map, of no bytes and of bytes that fail; and reads,
    clears, merges and parses again of what a parse made, of the identities of what they give."""
    R, L, E = cls("spark.connect.Relation"), cls("spark.connect.Limit"), cls("spark.connect.Expression")
    S = cls("spark.connect.SQL")
    limit = R(limit={"limit": 5, "input": {"sql": {"query": "q"}}}).SerializeToString()
    unknown = R(limit={"limit": 5}).SerializeToString()[:-2] + _field(2, _UNKNOWN + _tag(2, 0) + _varint(5))

    def stub_parsed(method, data):
        r = R()
        s = r.limit
        try:
            getattr(s, method)(data)
            failure = None
        except Exception as e:  # noqa: BLE001
            failure = type(e).__name__
        return failure, r.HasField("limit"), r.limit is s, r.SerializeToString()

    def stub_of_a_stub_parsed():
        r = R()
        s = r.limit.input
        try:
            s.ParseFromString(R(sql={"query": "q"}).SerializeToString()[:-1])
        except Exception as e:  # noqa: BLE001
            failure = type(e).__name__
        return failure, r.HasField("limit"), r.limit.input is s, r.SerializeToString()

    def parent_parsed(method):
        r = R()
        x, y = r.limit.input, r.limit
        getattr(r, method)(limit)
        return r.limit is y, r.limit.input is x, x.sql.query, y.limit, r.SerializeToString()

    def unset_member_stays_unset():
        r = R()
        z = r.project
        r.MergeFromString(limit)
        z.SetInParent()
        return r.WhichOneof("rel_type"), r.HasField("limit"), r.SerializeToString()

    def container_item():
        p = R(project={})
        c = p.project.expressions.add()
        c.ParseFromString(E(literal={"integer": 3}).SerializeToString())
        return p.project.expressions[0] is c, p.SerializeToString()

    def map_value():
        q = S(query="x")
        v = q.named_arguments["k"]
        v.ParseFromString(E(literal={"string": "s"}).SerializeToString())
        return q.named_arguments["k"] is v, q.SerializeToString(deterministic=True)

    def reads():
        r = R.FromString(limit)
        a = r.limit
        listed = [v for _, v in r.ListFields()]
        return a is r.limit, listed[0] is a, a.input is r.limit.input, r.limit.input.sql.query

    def read_then_changed(change):
        r = R.FromString(limit)
        child = r.limit
        grandchild = child.input
        change(r, child)
        return child.SerializeToString(), grandchild.sql.query, r.SerializeToString(), r.limit is child

    def merged_into_a_descendant():
        r = R.FromString(limit)
        below = r.limit.input
        below.MergeFromString(r.SerializeToString())
        return below.WhichOneof("rel_type"), r.SerializeToString()

    def discard_unknown():
        r = R.FromString(unknown)
        r.DiscardUnknownFields()
        return r.SerializeToString(), r.limit.SerializeToString()

    def depth(levels):
        data = b""
        for _ in range(levels):
            data = _field(R.LIMIT_FIELD_NUMBER, _field(1, data))
        try:
            R.FromString(data)
            return None
        except Exception as e:  # noqa: BLE001
            return type(e).__name__, str(e)

    cases = {
        "an unset child, ParseFromString": lambda: stub_parsed("ParseFromString", L(limit=5).SerializeToString()),
        "an unset child, MergeFromString": lambda: stub_parsed("MergeFromString", L(limit=5).SerializeToString()),
        "an unset child, of no bytes": lambda: stub_parsed("ParseFromString", b""),
        "an unset child, MergeFromString of no bytes": lambda: stub_parsed("MergeFromString", b""),
        "an unset child, of a failure": lambda: stub_parsed("ParseFromString", L(limit=5).SerializeToString() + b"\xff"),
        "an unset child of an unset child, of a failure": stub_of_a_stub_parsed,
        "a message of unset children, ParseFromString": lambda: parent_parsed("ParseFromString"),
        "a message of unset children, MergeFromString": lambda: parent_parsed("MergeFromString"),
        "an unset child of a field that the parse does not set": unset_member_stays_unset,
        "a message of a container": container_item,
        "a message of a map": map_value,
        "reads of a parse": reads,
        "a read child, cleared": lambda: read_then_changed(lambda r, c: r.ClearField("limit")),
        "a read child, parsed again": lambda: read_then_changed(
            lambda r, c: r.ParseFromString(R(sql={"query": "z"}).SerializeToString())
        ),
        "a read child, merged into": lambda: read_then_changed(lambda r, c: c.MergeFromString(L(limit=7).SerializeToString())),
        "a merge into a descendant of the bytes of its ancestor": merged_into_a_descendant,
        "DiscardUnknownFields": discard_unknown,
        "100 levels": lambda: depth(50),
        "101 levels": lambda: depth(51),
    }
    return {name: _outcome(case) for name, case in cases.items()}


def test_parses_into_unset_children_and_held_messages_are_upbs(upb):
    # A parse into a message that holds no field reads its bytes natively, of no Python object for
    # each message (`codec::merge_with`), and gives what upb's gives: an unset child that it parses
    # into is set in its parent, of any level, an unset child that the message gave out of a field
    # that the parse sets is the parsed message (upb's `SyncSubobjs`), of every level, and one of a
    # field that it does not set stays the field's; a message of a container or a map is the
    # container's; a failure keeps what it read; and what the parse read gives one object of each
    # message from then on, which a clear, a merge and a parse again of the message keep, as upb's.
    # The levels of a parse are upb's, of its `DecodeError` of the limit.
    assert _parse_scenarios(ours) == _parse_scenarios(upb)


def _parse_again(lockstep, generator, a, b, log, depth=0):
    """A parse, of `MergeFromString`, of the bytes of a random message into the messages `a`
    (upb's) and `b`, or into a message of a message field of them (set or unset), as of each
    runtime alike. The objects that the lockstep keeps are released first, as upb's merge into a
    message of more than two unset children given out takes the parsed messages as some of them in
    an order of their addresses (its `SyncSubobjs` skips one after each it takes). Its bytes are
    whole: upb's parse that fails in a message that holds fields reads memory that it does not
    hold (see `failed_string_value` of `codec.rs`), and may end the process."""
    rng = lockstep.rng
    fields = [f for f in a.DESCRIPTOR.fields if f.type == f.TYPE_MESSAGE and not f.is_repeated]
    if fields and depth < 2 and rng.random() < 0.5:
        f = rng.choice(fields)
        log.append(f"in {f.name}")
        _parse_again(lockstep, generator, getattr(a, f.name), getattr(b, f.name), log, depth + 1)
        return
    source = lockstep.upb(a.DESCRIPTOR.full_name)()
    generator.fill(source, depth=rng.randint(0, 2))
    data = source.SerializeToString(deterministic=True)
    log.append(f"MergeFromString({data.hex()})")
    lockstep.kept.clear()
    lockstep.both(log[-1], a, b, lambda m: m.MergeFromString(data))


@pytest.mark.parametrize("seed", range(6))
def test_random_operations_on_parsed_messages_agree_with_upb(upb, seed):
    # The random operations of the lockstep on messages that FromString, ParseFromString and
    # MergeFromString parse natively (whose objects a read makes, of all the levels of a value of
    # few messages at once, else of one level), and parses again into them and into their
    # messages, agree with upb's, of the identities of the objects that they give too.
    lockstep = _Lockstep(upb, seed)
    generator = _Generator(upb, seed)
    rng = lockstep.rng
    for full_name in _LOCKSTEP_TYPES:
        for _ in range(3):
            source = upb(full_name)()
            generator.fill(source, depth=rng.randint(1, 3))
            data = source.SerializeToString(deterministic=True)
            how = rng.choice(["FromString", "ParseFromString", "MergeFromString"])
            if how == "FromString":
                a, b = upb(full_name).FromString(data), ours(full_name).FromString(data)
            else:
                a, b = upb(full_name)(), ours(full_name)()
                getattr(a, how)(data)
                getattr(b, how)(data)
            log = [f"{how}({data.hex()})"]
            lockstep.kept.clear()
            lockstep.compare(full_name, a, b, log)
            for _ in range(30):
                if rng.random() < 0.15:
                    _parse_again(lockstep, generator, a, b, log)
                else:
                    lockstep.step(a, b, log)
                lockstep.compare(full_name, a, b, log)


def _parsed_projects(count):
    """The bytes of a relation of a projection of `count` expressions of three levels each, whose
    projection is of `count * 7 + 3` messages (of its input and the range of the input)."""
    expression = proto.Expression(
        unresolved_function={
            "function_name": "+",
            "arguments": [{"unresolved_attribute": {"unparsed_identifier": "a"}}, {"literal": {"integer": 1}}],
        }
    )
    return proto.Relation(project={"input": {"range": {"end": 1}}, "expressions": [expression] * count}).SerializeToString()


def test_a_collection_during_the_first_read_of_a_parse_finds_the_message():
    # The first read of a field of a parse makes the objects of its value with no borrow of the
    # message held, of all its levels at once of a value of up to 256 messages (see
    # `native::WHOLE`), else of one level: a collection that their allocation starts, whose
    # callback reads the whole message, finds it free and reads it whole, and the field gives the
    # objects of the read from then on.
    out = {}
    for count in (2, 36, 37, 300):
        data = _parsed_projects(count)
        message = proto.Relation.FromString(data)
        state = {"ran": 0, "errors": []}

        def walk(m):
            for f, v in m.ListFields():
                if f.message_type is not None:
                    for x in (v if f.is_repeated else [v]):
                        walk(x)

        def callback(phase, info, message=message, state=state):
            if phase == "start" and state["ran"] < 50:
                state["ran"] += 1
                try:
                    walk(message)
                except BaseException as e:  # noqa: BLE001 (a PanicException too)
                    state["errors"].append(f"{type(e).__name__}: {e}")

        gc.collect()
        gc.callbacks.append(callback)
        threshold = gc.get_threshold()
        gc.set_threshold(1)
        try:
            project = message.project
            expressions = list(project.expressions)
            arguments = [list(e.unresolved_function.arguments) for e in expressions]
        finally:
            gc.set_threshold(*threshold)
            gc.callbacks.remove(callback)
        same = (
            message.project is project
            and all(x is y for x, y in zip(project.expressions, expressions))
            and all(
                all(x is y for x, y in zip(e.unresolved_function.arguments, args))
                for e, args in zip(project.expressions, arguments)
            )
        )
        out[count] = [state["ran"] > 0, state["errors"], same, message.SerializeToString() == data]
    assert out == {count: [True, [], True, True] for count in (2, 36, 37, 300)}, out


def test_a_signal_during_the_first_read_of_a_parse_leaves_it_whole():
    # As of a copy (see test_a_signal_during_the_first_read_of_a_copy_leaves_it_whole): the first
    # read of a repeated message field of a parse of many messages makes their objects of
    # checkpoints, so a signal there raises the exception of its handler from the read, and the
    # message stays whole, its field read again in full, of one object from then on.
    import signal

    count = 200_000
    source = proto.Expression.Literal.Array(elements=[proto.Expression.Literal(integer=1)] * count)
    data = proto.Expression.Literal(array=source).SerializeToString()

    class Stop(Exception):
        pass

    def handler(signum, frame):
        raise Stop()

    previous = signal.signal(signal.SIGALRM, handler)
    outcomes = []
    try:
        for attempt in range(10):
            h = proto.Expression.Literal.FromString(data)
            array = h.array  # of one level, of more than 256 messages: its elements stay native
            signal.setitimer(signal.ITIMER_REAL, 0.0005 + attempt * 0.0002)
            try:
                len(array.elements)
                outcomes.append("completed")
            except Stop:
                outcomes.append("interrupted")
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            again = array.elements
            assert len(again) == count and again is array.elements, attempt
            assert h.SerializeToString() == data, attempt
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
    assert "interrupted" in outcomes, outcomes
