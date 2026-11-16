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
import pathlib
import random
import subprocess
import sys
import textwrap

import pytest
from pyspark import _pyspark
from pyspark.sql.connect import proto
from pyspark.sql.connect.proto import _rust

# The classes of upb, by full name, that _upb_classes makes.
_UPB_CLASSES = {}


def _upb_classes():
    """The class of a message of upb by full name, as the modules that protoc generates of the
    protos make it (of its module and qualified names, as Apache Spark's do), or None without
    protobuf of upb. The classes are of a pool of their own, of the files of the protos and of
    protobuf's own of the well-known types, so that no class that protobuf made before of the
    default pool, of another name, stands in for one."""
    try:
        from google.protobuf import any_pb2, descriptor_pool, timestamp_pb2
        from google.protobuf.internal import api_implementation, builder
    except ImportError:
        return None
    if api_implementation.Type() != "upb":
        return None
    if not _UPB_CLASSES:
        pool = descriptor_pool.DescriptorPool()
        files = [(m.DESCRIPTOR.serialized_pb, m.__name__) for m in (any_pb2, timestamp_pb2)]
        files += [
            (_pyspark._proto_file_descriptor(name).serialized_pb, _rust._module_name(name, package))
            for name, package, _, _ in _pyspark._proto_files()
            if package == "spark.connect"
        ]

        def add(cls):
            _UPB_CLASSES[cls.DESCRIPTOR.full_name] = cls
            for nested in cls.DESCRIPTOR.nested_types:
                add(getattr(cls, nested.name))

        for serialized, module_name in files:
            file = pool.AddSerializedFile(serialized)
            namespace = {}
            builder.BuildMessageAndEnumDescriptors(file, namespace)
            builder.BuildTopDescriptorsAndMessages(file, module_name, namespace)
            for name in file.message_types_by_name:
                add(namespace[name])
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


_SEQUENCE_CASES = [
    ("abc", lambda c: [isinstance(c, t) for t in (abc.Sequence, abc.MutableSequence, abc.Mapping)]),
    ("len", len),
    ("list", list),
    ("reversed", lambda c: list(reversed(c))),
    ("[-1]", lambda c: c[-1]),
    ("[3]", lambda c: c[3]),
    ("[::-1]", lambda c: c[::-1]),
    ('["a"]', lambda c: c["a"]),
    ("== list", lambda c: c == list(c)),
    ("in", lambda c: c[0] in c),
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
    ("remove", lambda c: (c.remove(c[1]), len(c))),
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
        from pyspark.sql.connect.proto import _rust

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


def test_pipeline_analysis_context_of_the_runtime_is_the_vendored_ones(upb):
    # The context manager of the runtime adds the user context extension that the vendored one
    # adds with protobuf's Any, and removes it, also when its body raises. The vendored module,
    # as it is imported, has the runtime's (see pyspark._proto_hooks), so its file is run here.
    import importlib.util

    import pyspark.pipelines.add_pipeline_analysis_context as module

    assert module.add_pipeline_analysis_context is _rust.add_pipeline_analysis_context
    assert _rust.add_pipeline_analysis_context.__module__ == module.__name__
    spec = importlib.util.spec_from_file_location("_vendored_analysis_context", module.__file__)
    vendored_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vendored_module)
    vendored = vendored_module.add_pipeline_analysis_context
    assert vendored is not _rust.add_pipeline_analysis_context

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


def test_modules_that_protoc_generates_of_protos_importing_spark_connects_build(upb):
    # The files of the protos are in protobuf's default pool, as the generated modules add them,
    # so that a file of a plugin that imports them builds, as the module that protoc generates of
    # it adds it (Delta Lake's do). Its messages are protobuf's, which take a message of
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
