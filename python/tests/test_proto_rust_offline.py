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
"""The Rust protobuf runtime of the drop-in (``pyspark.sql.connect.proto._rust``) against upb,
the implementation of a ``protobuf`` install: the same operations on a message of each agree in
their bytes, text, fields, containers, presence, errors and descriptors. The last test runs the
runtime where ``protobuf`` cannot be imported, as the drop-in runs without it.

The comparisons need the generated modules of upb in use (protobuf installed, and the runtime not
asked for), and are skipped otherwise.
"""

import collections.abc as abc
import copy
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


@pytest.fixture(scope="module")
def upb():
    """The class of a message of upb, by full name."""
    if proto._implementation != "protobuf":
        pytest.skip("the generated modules of protobuf are not in use")
    from google.protobuf import symbol_database
    from google.protobuf.internal import api_implementation

    if api_implementation.Type() != "upb":
        pytest.skip("the runtime follows upb, the implementation of a protobuf install")
    return symbol_database.Default().GetSymbol


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
    # mappings, as protobuf's are. With upb installed, the bytes and the text are upb's.
    script = textwrap.dedent(
        """
        import collections.abc, copy, importlib.abc, json, pickle, sys

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "google.protobuf" or name.startswith("google.protobuf."):
                    raise ImportError("no protobuf")

        sys.meta_path.insert(0, Block())
        import numpy as np
        import pyspark.sql.connect.proto as proto
        from pyspark.sql.connect.proto import expressions_pb2
        from tests.test_proto_rust_offline import _text_cases

        assert proto._implementation == "rust"
        assert expressions_pb2.Expression is proto.Expression
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
    if proto._implementation == "protobuf":
        from google.protobuf import symbol_database
        from google.protobuf.internal import api_implementation

        if api_implementation.Type() == "upb":
            relation, context = _text_cases(symbol_database.Default().GetSymbol)
            relation.sql.pos_arguments.add().literal.specialized_array.doubles.values.extend(
                [1.5, 2.5]
            )
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
    # package's star imports of the generated modules give protobuf's.
    names = {}
    saved = {k: v for k, v in sys.modules.items() if k.startswith("pyspark.sql.connect.proto.")}
    try:
        _rust.install(names)
        assert sys.modules["pyspark.sql.connect.proto.base_pb2"].Plan is ours("spark.connect.Plan")
    finally:
        sys.modules.update(saved)
    assert names["Relation"] is ours("spark.connect.Relation")
    assert names["relations_pb2"].Relation is names["Relation"]
    real_names = {n for n in dir(proto) if not n.startswith("_") and n[0].isupper()}
    assert real_names <= set(names), sorted(real_names - set(names))[:5]
    assert not any(n.startswith("_rust") for n in names)


def test_pipeline_analysis_context_of_the_runtime_is_the_vendored_ones(upb):
    # The context manager of the runtime adds the user context extension that the vendored one
    # adds with protobuf's Any, and removes it, also when its body raises.
    from pyspark.pipelines.add_pipeline_analysis_context import (
        add_pipeline_analysis_context as vendored,
    )

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
