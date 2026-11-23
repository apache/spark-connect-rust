# SPDX-License-Identifier: Apache-2.0
"""The drop-in's types and Columns as official PySpark 4.3.0's, without a server: the JSON type
strings of `_parse_datatype_json_value`, the types that official PySpark pickled, the abstract
classes of the types (`DataType`, `AtomicType`, ...) as types of their own, and the errors of them
in the types of UDTFs, Java UDFs, tables and `createDataFrame`, the Python subclasses of the
concrete types, the metadata of fields, the pickle of a CalendarIntervalType, the parameters of the
types as attributes, the parameters of the methods of the types, the origins of the expressions
that Column methods send, the names of `Column.alias` and `Column.name`, and the user-set
attributes of a Column's copies. The expected values are official PySpark's, of the same calls."""

import base64
import copy
import copyreg
import datetime
import gc
import inspect
import json
import pickle
import re
from pathlib import Path

import pytest
from _mock_connect_server import mock_connect_session
from pyspark.errors import PySparkAttributeError, PySparkTypeError, PySparkValueError
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.connect.proto import base_pb2

# The JSON type strings that official PySpark parses, with the repr of its type: PySpark matches
# its patterns at the start of the string, so text after a match is ignored.
PARSED = [
    ("interval", "CalendarIntervalType()"),
    ("timestamp_ntz", "TimestampNTZType()"),
    ("char(3) collate UTF8_LCASE", "CharType(3)"),
    ("varchar(3)collate UTF8_LCASE", "VarcharType(3)"),
    ("char(3)xyz", "CharType(3)"),
    ("char( 3 )", "CharType(3)"),
    ("string collate a-b", "StringType('a')"),
    ("string collate UTF8_LCASE x", "StringType('UTF8_LCASE')"),
    ("string  collate   UTF8_LCASE", "StringType('UTF8_LCASE')"),
    ("interval dayx", "DayTimeIntervalType(0, 0)"),
    ("interval day to x", "DayTimeIntervalType(0, 0)"),
    ("interval day to secondx", "DayTimeIntervalType(0, 3)"),
    ("interval year to month", "YearMonthIntervalType(0, 1)"),
    ("interval month to year", "YearMonthIntervalType(1, 0)"),
    ("decimal(10,2)junk", "DecimalType(10,2)"),
    ("decimal( 10 , -2 )", "DecimalType(10,-2)"),
    ("time(3)x", "TimeType(3)"),
    ("geometry(OGC:CRS84)x", "GeometryType(4326)"),
    ("string", "StringType()"),
    ("void", "NullType()"),
    ("variant", "VariantType()"),
]

# The JSON type strings that official PySpark does not parse: CANNOT_PARSE_DATATYPE, also of the
# lenient forms of the Rust core (`null`, `calendarinterval`, `unparsed(...)`).
UNPARSED = [
    "null",
    "calendarinterval",
    "unparsed(foo)",
    "timestamp_ntz(9)",
    "timestamp_ltz(6)",
    "decimal(10)",
    "integer ",
    " integer",
]


@pytest.mark.parametrize("value, expected", PARSED, ids=[case for case, _ in PARSED])
def test_json_type_strings_parse_as_pyspark(value, expected):
    assert repr(T._parse_datatype_json_value(value)) == expected


@pytest.mark.parametrize("value", UNPARSED)
def test_json_type_strings_that_pyspark_does_not_parse(value):
    with pytest.raises(PySparkValueError) as error:
        T._parse_datatype_json_value(value)
    assert error.value.getCondition() == "CANNOT_PARSE_DATATYPE"
    assert error.value.getMessageParameters() == {"msg": value}


@pytest.mark.parametrize(
    "value, message",
    [
        (5, "expected string or bytes-like object, got 'int'"),
        (None, "expected string or bytes-like object, got 'NoneType'"),
        (True, "expected string or bytes-like object, got 'bool'"),
        ([1], "unhashable type: 'list'"),
    ],
)
def test_json_type_values_that_are_not_strings_raise_the_type_error_of_pyspark(value, message):
    # Of PySpark's lookup of the name, or of the match of its patterns.
    with pytest.raises(TypeError) as error:
        T._parse_datatype_json_value(value)
    assert str(error.value) == message


# Official PySpark 4.3.0's pickles of its types, at the protocols 0 to 5, with its repr, JSON,
# simpleString and attributes of each type, of scripts/capture_type_pickles_golden.py.
PYSPARK_PICKLES = json.loads(
    (Path(__file__).resolve().parents[2] / "tests/golden/type_pickles.json").read_text()
)


def _pysparks_pickle(key, protocol):
    return pickle.loads(base64.b64decode(PYSPARK_PICKLES[key][str(protocol)]))


def _attributes(t, case):
    """The repr of each attribute of PySpark's type of `case`, of its `__dict__`, and the repr of
    the attribute of the drop-in's type `t`."""
    expected = case["attributes"]
    return {k: repr(getattr(t, k)) for k in expected}, expected


# The types of parameters, which keep them in their `__dict__`, as PySpark's (SPARK-62251).
PARAMETER_TYPES = (
    T.DecimalType, T.StringType, T.CharType, T.VarcharType, T.TimeType, T.DayTimeIntervalType,
    T.YearMonthIntervalType, T.GeometryType, T.GeographyType,
)
# The types whose `__dict__` holds their parameters, as PySpark's: the types of parameters, and
# arrays, maps and fields (SPARK-62251).
DICT_TYPES = PARAMETER_TYPES + (T.ArrayType, T.MapType, T.StructField)


def _dict_of(t):
    """The repr of each entry of the `__dict__` of `t`, in its order."""
    return [(k, repr(v)) for k, v in vars(t).items()]


@pytest.mark.parametrize("protocol", [2, 3, 4, 5])
def test_types_that_pyspark_pickled_load_as_its_types(protocol):
    # PySpark's types are plain classes, which pickle as `copyreg.__newobj__(cls)` and their
    # `__dict__`: the drop-in's class makes a type of `cls.__new__(cls)`, without arguments, and
    # sets the attributes of the `__dict__`, of which the type then is, also in a struct, an
    # array or a map. A type of parameters, an array, a map and a field keep their parameters in
    # their `__dict__`, as PySpark's; a struct keeps its as its parameters. A type of PySpark's
    # `DataTypeSingleton` is the one instance, which `cls.__new__(cls)` gives, where PySpark's own
    # unpickling gives another object.
    from pyspark.ml.linalg import VectorUDT

    names = dict(vars(T), VectorUDT=VectorUDT)
    singletons = (
        T.NullType, T.BooleanType, T.ByteType, T.ShortType, T.IntegerType, T.LongType,
        T.FloatType, T.DoubleType, T.BinaryType, T.DateType, T.TimestampType,
        T.TimestampNTZType, T.CalendarIntervalType,
    )
    loaded = _pysparks_pickle("pickles", protocol)
    assert len(loaded) == len(PYSPARK_PICKLES["cases"])
    for t, case in zip(loaded, PYSPARK_PICKLES["cases"]):
        made = eval(case["type"], names)
        assert type(t) is type(made), case["type"]
        assert t == made and hash(t) == hash(made), case["type"]
        assert (t is made) is isinstance(made, singletons), case["type"]
        assert (repr(t), t.json(), t.simpleString()) == (
            case["repr"], case["json"], case["simpleString"]
        )
        got, expected = _attributes(t, case)
        assert got == expected, case["type"]
        if isinstance(t, DICT_TYPES):
            assert _dict_of(t) == list(case["attributes"].items()), case["type"]
        else:
            assert vars(t) == {}, case["type"]


@pytest.mark.parametrize("protocol", [0, 1])
def test_types_that_pyspark_pickled_below_protocol_2_do_not_load(protocol):
    # Below protocol 2, pickle makes PySpark's type of `copyreg._reconstructor(cls, object,
    # None)`, which calls `object.__new__(cls)`, and CPython refuses it for a class of a native
    # base, of which every type of the drop-in is: pickle's default protocol is above.
    message = r"^object\.__new__\(NullType\) is not safe, use NullType\.__new__\(\)$"
    with pytest.raises(TypeError, match=message):
        _pysparks_pickle("pickles", protocol)


@pytest.mark.parametrize("protocol", [2, 3, 4, 5])
def test_types_that_pyspark_pickled_after_an_edit_load_as_they_were(protocol):
    # The attributes as they were set, which no constructor makes: a `hasPrecisionInfo` of
    # another value and an attribute of the user, which is in the `__dict__`, a set parameter, and
    # the names and the `_needConversion` of a struct, which an edit of a field does not change.
    loaded = _pysparks_pickle("edited_pickles", protocol)
    assert set(loaded) == set(PYSPARK_PICKLES["edited"])
    for name, t in loaded.items():
        case = PYSPARK_PICKLES["edited"][name]
        assert (repr(t), t.simpleString()) == (case["repr"], case["simpleString"]), name
        got, expected = _attributes(t, case)
        assert got == expected, name
        assert t.json() == case["json"], name
        if isinstance(t, PARAMETER_TYPES):
            assert _dict_of(t) == list(case["attributes"].items()), name
    decimal = loaded["decimal_attributes"]
    assert vars(decimal) == {"precision": 10, "scale": 2, "hasPrecisionInfo": False, "tag": "x"}
    assert decimal != T.DecimalType(10, 2) and hash(decimal) == hash(T.DecimalType(10, 2))
    assert loaded["set_scale"] == T.DecimalType(10, 3)
    assert loaded["retyped_field"]._needSerializeAnyField is True
    # PySpark's JSON of a GeometryType of a set SRID is that of the CRS of its constructor, which
    # it keeps as `_crs`, so the type is equal to no GeometryType(0), whose `_crs` is another.
    assert loaded["set_srid"].json() == '"geometry(OGC:CRS84)"'
    assert loaded["set_srid"] != T.GeometryType(0)


# The abstract classes of PySpark's type hierarchy, with the name of each without `Type` in lower
# case, PySpark's `typeName()` of the class.
ABSTRACT_TYPES = [
    ("DataType", "data"),
    ("AtomicType", "atomic"),
    ("NumericType", "numeric"),
    ("IntegralType", "integral"),
    ("FractionalType", "fractional"),
    ("DatetimeType", "datetime"),
    ("AnyTimeType", "anytime"),
    ("SpatialType", "spatial"),
    ("AnsiIntervalType", "ansiinterval"),
]

# Official PySpark 4.3.0's pickles at the protocols 2 to 5 of `AtomicType()`, of its class, which
# `copyreg.__newobj__` makes, and of no state of its empty `__dict__`; the pickle at protocol 2 of
# each abstract class, of the name of the class; and the pickles of an `AtomicType()` of an
# attribute that was set, of its `__dict__`.
PYSPARKS_ATOMIC_TYPE_PICKLES = {
    2: b"\x80\x02cpyspark.sql.types\nAtomicType\nq\x00)\x81q\x01.",
    3: b"\x80\x03cpyspark.sql.types\nAtomicType\nq\x00)\x81q\x01.",
    4: b"\x80\x04\x95'\x00\x00\x00\x00\x00\x00\x00\x8c\x11pyspark.sql.types\x94\x8c\nAtomicType"
       b"\x94\x93\x94)\x81\x94.",
    5: b"\x80\x05\x95'\x00\x00\x00\x00\x00\x00\x00\x8c\x11pyspark.sql.types\x94\x8c\nAtomicType"
       b"\x94\x93\x94)\x81\x94.",
}
PYSPARKS_PROTOCOL_2_PICKLE = b"\x80\x02cpyspark.sql.types\n%s\nq\x00)\x81q\x01."
PYSPARKS_TAGGED_ATOMIC_TYPE_PICKLES = {
    2: b"\x80\x02cpyspark.sql.types\nAtomicType\nq\x00)\x81q\x01}q\x02X\x03\x00\x00\x00tagq\x03"
       b"X\x01\x00\x00\x00xq\x04sb.",
    5: b"\x80\x05\x955\x00\x00\x00\x00\x00\x00\x00\x8c\x11pyspark.sql.types\x94\x8c\nAtomicType"
       b"\x94\x93\x94)\x81\x94}\x94\x8c\x03tag\x94\x8c\x01x\x94sb.",
}


@pytest.mark.parametrize("name", [name for name, _ in ABSTRACT_TYPES])
def test_the_abstract_types_pickle_as_pysparks(name):
    # PySpark's abstract classes are plain classes, which pickle as their class and their
    # `__dict__`: the drop-in's pickle of the type at the protocols 2 and up is PySpark's own, so
    # each client loads the other's, and the one at the protocols 0 and 1, of
    # `copyreg.__newobj__(cls)` rather than PySpark's `copyreg._reconstructor`, loads in both
    # clients too. A copy is of the class, and IntegralType's is its one instance.
    cls = getattr(T, name)
    t = cls()
    assert pickle.dumps(t, protocol=2) == PYSPARKS_PROTOCOL_2_PICKLE % name.encode()
    assert type(pickle.loads(PYSPARKS_PROTOCOL_2_PICKLE % name.encode())) is cls
    for protocol in range(6):
        loaded = pickle.loads(pickle.dumps(t, protocol=protocol))
        assert type(loaded) is cls and loaded == t and vars(loaded) == {}, protocol
        assert (loaded is t) is (cls is T.IntegralType), protocol
    for copied in (copy.copy(t), copy.deepcopy(t)):
        assert type(copied) is cls and copied == t and (copied is t) is (cls is T.IntegralType)


def test_an_abstract_type_pickles_its_attributes_as_pysparks():
    # Of the protocols 2 to 5 of PySpark's pickle of `AtomicType()`, and of its pickle of the
    # `__dict__` of an attribute that was set, which the drop-in's pickle keeps too.
    for protocol, pickled in PYSPARKS_ATOMIC_TYPE_PICKLES.items():
        assert pickle.dumps(T.AtomicType(), protocol=protocol) == pickled
        assert pickle.loads(pickled) == T.AtomicType()
    tagged = T.AtomicType()
    tagged.tag = "x"
    for protocol, pickled in PYSPARKS_TAGGED_ATOMIC_TYPE_PICKLES.items():
        assert pickle.dumps(tagged, protocol=protocol) == pickled
        loaded = pickle.loads(pickled)
        assert type(loaded) is T.AtomicType and vars(loaded) == {"tag": "x"}
        assert loaded == tagged and loaded != T.AtomicType()
    assert vars(copy.deepcopy(tagged)) == {"tag": "x"}


# Official PySpark 4.3.0's pickles at the protocols 2 and 5 of `DecimalType(7, 3)` of an attribute
# `user` "u" that was set, of `DecimalType(10, 2)` of its `hasPrecisionInfo` set false, and of
# `IntegerType()` of an attribute `tag` "x": of `copyreg.__newobj__(cls)` and the `__dict__`.
PYSPARKS_PICKLES_OF_SET_ATTRIBUTES = {
    ("user", 2): b"\x80\x02cpyspark.sql.types\nDecimalType\nq\x00)\x81q\x01}q\x02(X\t\x00\x00\x00"
        b"precisionq\x03K\x07X\x05\x00\x00\x00scaleq\x04K\x03X\x10\x00\x00\x00hasPrecisionInfoq"
        b"\x05\x88X\x04\x00\x00\x00userq\x06X\x01\x00\x00\x00uq\x07ub.",
    ("user", 5): b"\x80\x05\x95d\x00\x00\x00\x00\x00\x00\x00\x8c\x11pyspark.sql.types\x94\x8c\x0b"
        b"DecimalType\x94\x93\x94)\x81\x94}\x94(\x8c\tprecision\x94K\x07\x8c\x05scale\x94K\x03\x8c"
        b"\x10hasPrecisionInfo\x94\x88\x8c\x04user\x94\x8c\x01u\x94ub.",
    ("precision_info", 2): b"\x80\x02cpyspark.sql.types\nDecimalType\nq\x00)\x81q\x01}q\x02(X\t"
        b"\x00\x00\x00precisionq\x03K\nX\x05\x00\x00\x00scaleq\x04K\x02X\x10\x00\x00\x00hasPrecisi"
        b"onInfoq\x05\x89ub.",
    ("precision_info", 5): b"\x80\x05\x95Y\x00\x00\x00\x00\x00\x00\x00\x8c\x11pyspark.sql.types"
        b"\x94\x8c\x0bDecimalType\x94\x93\x94)\x81\x94}\x94(\x8c\tprecision\x94K\n\x8c\x05scale"
        b"\x94K\x02\x8c\x10hasPrecisionInfo\x94\x89ub.",
    ("tag", 2): b"\x80\x02cpyspark.sql.types\nIntegerType\nq\x00)\x81q\x01}q\x02X\x03\x00\x00\x00"
        b"tagq\x03X\x01\x00\x00\x00xq\x04sb.",
    ("tag", 5): b"\x80\x05\x956\x00\x00\x00\x00\x00\x00\x00\x8c\x11pyspark.sql.types\x94\x8c\x0bI"
        b"ntegerType\x94\x93\x94)\x81\x94}\x94\x8c\x03tag\x94\x8c\x01x\x94sb.",
}


def test_types_of_set_attributes_pickle_and_copy_them_as_pysparks():
    # PySpark's types pickle, copy and deep-copy as their class and their `__dict__`, of the
    # attributes that a user set too, which their JSON does not keep, nor a `hasPrecisionInfo`
    # other than true, nor the `names` and `_needConversion` of a struct that an edit of its fields
    # makes differ: the drop-in's pickle of such a type of the class's own is PySpark's, byte for
    # byte, and its copies and pickles keep them, also in the types that hold it.
    user = T.DecimalType(7, 3)
    user.user = "u"
    precision_info = T.DecimalType(10, 2)
    precision_info.hasPrecisionInfo = False
    tag = T.IntegerType()
    tag.tag = "x"
    try:
        for (name, protocol), pickled in PYSPARKS_PICKLES_OF_SET_ATTRIBUTES.items():
            t = {"user": user, "precision_info": precision_info, "tag": tag}[name]
            assert pickle.dumps(t, protocol=protocol) == pickled
    finally:
        # The one instance of the singleton type is of every IntegerType() of the process.
        del tag.tag
    for t in (user, precision_info):
        copies = [copy.copy(t), copy.deepcopy(t)]
        copies += [pickle.loads(pickle.dumps(t, protocol)) for protocol in range(6)]
        for copied in copies:
            assert type(copied) is T.DecimalType and copied == t and vars(copied) == vars(t)
            assert copied.hasPrecisionInfo is t.hasPrecisionInfo
    field = T.StructField("f", user, True, {"k": "v"})
    field.tag = 1
    struct = T.StructType([field])
    struct.tag = 2
    for t, inner in [
        (T.ArrayType(user), lambda c: c.elementType),
        (T.MapType(T.StringType(), user), lambda c: c.valueType),
        (field, lambda c: c.dataType),
        (struct, lambda c: c.fields[0].dataType),
    ]:
        for copied in [copy.deepcopy(t)] + [pickle.loads(pickle.dumps(t, p)) for p in range(6)]:
            assert type(copied) is type(t) and copied == t and vars(copied) == vars(t)
            assert inner(copied).user == "u"
    assert pickle.loads(pickle.dumps(struct)).fields[0].tag == 1
    renamed = T.StructType([T.StructField("a", T.LongType())])
    renamed.fields[0].name = "z"
    for copied in (copy.copy(renamed), copy.deepcopy(renamed), pickle.loads(pickle.dumps(renamed))):
        assert (copied.names, copied.fields[0].name) == (["a"], "z") and copied == renamed
    retyped = T.StructType([T.StructField("a", T.LongType())])
    retyped.fields[0].dataType = T.TimestampType()
    for copied in (copy.copy(retyped), copy.deepcopy(retyped), pickle.loads(pickle.dumps(retyped))):
        assert (copied._needConversion, copied._needSerializeAnyField) == ([False], False)
        assert copied.fields[0].dataType == T.TimestampType() and copied == retyped


def test_calendar_interval_type_pickles_of_its_json_name():
    # As the JSON of the type, "interval", which official PySpark's parser reads too.
    t = T.CalendarIntervalType()
    assert t.__reduce__()[1] == ('"interval"',)
    assert pickle.loads(pickle.dumps(t)) == t


# Of a parameter that is set: the type, the parameter and its value, official PySpark's repr and
# JSON of the type after `setattr(t, name, value)`, and the constructor's type of the parameters.
SET_PARAMETERS = [
    ("DecimalType(10, 2)", "scale", 3, "DecimalType(10,3)", '"decimal(10,3)"',
     "DecimalType(10, 3)"),
    ("DecimalType(10, 2)", "precision", 12, "DecimalType(12,2)", '"decimal(12,2)"',
     "DecimalType(12, 2)"),
    ("StringType()", "collation", "UTF8_LCASE", "StringType('UTF8_LCASE')",
     '"string collate UTF8_LCASE"', "StringType('UTF8_LCASE')"),
    ("StringType('UTF8_LCASE')", "collation", "UTF8_BINARY", "StringType()", '"string"',
     "StringType()"),
    ("CharType(5)", "length", 6, "CharType(6)", '"char(6)"', "CharType(6)"),
    ("VarcharType(5)", "length", 6, "VarcharType(6)", '"varchar(6)"', "VarcharType(6)"),
    ("TimeType(3)", "precision", 4, "TimeType(4)", '"time(4)"', "TimeType(4)"),
    ("DayTimeIntervalType(0, 3)", "startField", 1, "DayTimeIntervalType(1, 3)",
     '"interval hour to second"', "DayTimeIntervalType(1, 3)"),
    ("DayTimeIntervalType(0, 3)", "endField", 1, "DayTimeIntervalType(0, 1)",
     '"interval day to hour"', "DayTimeIntervalType(0, 1)"),
    ("DayTimeIntervalType(0, 1)", "startField", 3, "DayTimeIntervalType(3, 1)",
     '"interval second to hour"', "DayTimeIntervalType(3, 1)"),
    ("DayTimeIntervalType(0, 3)", "startField", 1.0, "DayTimeIntervalType(1, 3)",
     '"interval hour to second"', "DayTimeIntervalType(1, 3)"),
    ("YearMonthIntervalType(0, 1)", "startField", 1, "YearMonthIntervalType(1, 1)",
     '"interval month"', "YearMonthIntervalType(1, 1)"),
    ("YearMonthIntervalType(0, 1)", "endField", 0, "YearMonthIntervalType(0, 0)",
     '"interval year"', "YearMonthIntervalType(0, 0)"),
]


@pytest.mark.parametrize(
    "made, name, value, shown, json_text, same",
    SET_PARAMETERS,
    ids=[f"{made}.{name}={value!r}" for made, name, value, *_ in SET_PARAMETERS],
)
def test_type_parameters_are_plain_attributes_as_pysparks(
    made, name, value, shown, json_text, same
):
    # PySpark's parameters are plain attributes: an assignment of one makes the type of it, as its
    # repr, JSON, simpleString, `==` and hash show, as of the constructor's type of the same
    # parameters, and the type is no longer equal to the one it was made as.
    t, same = eval(made, vars(T)), eval(same, vars(T))
    setattr(t, name, value)
    assert getattr(t, name) == value
    assert (repr(t), t.json()) == (shown, json_text)
    assert t.simpleString() == same.simpleString()
    assert t == same and hash(t) == hash(same)
    assert t != eval(made, vars(T))


def test_type_parameters_set_in_a_nested_type_are_of_the_containing_type_as_pysparks():
    # As PySpark's, whose struct, array and map read the attributes of their types: a parameter
    # set on a type in one of them is of it, also of the collations of a field in its metadata.
    t = T.StructType([
        T.StructField("a", T.DecimalType(10, 2)),
        T.StructField("b", T.ArrayType(T.StringType())),
        T.StructField("m", T.MapType(T.StringType(), T.TimeType(3))),
    ])
    t.fields[0].dataType.scale = 4
    t.fields[1].dataType.elementType.collation = "UTF8_LCASE"
    t["m"].dataType.valueType.precision = 5
    assert t.simpleString() == (
        "struct<a:decimal(10,4),b:array<string collate UTF8_LCASE>,m:map<string,time(5)>>"
    )
    assert t.json() == (
        '{"fields":[{"metadata":{},"name":"a","nullable":true,"type":"decimal(10,4)"},'
        '{"metadata":{"__COLLATIONS":{"b.element":"spark.UTF8_LCASE"}},"name":"b",'
        '"nullable":true,"type":{"containsNull":true,"elementType":"string","type":"array"}},'
        '{"metadata":{},"name":"m","nullable":true,"type":{"keyType":"string","type":"map",'
        '"valueContainsNull":true,"valueType":"time(5)"}}],"type":"struct"}'
    )
    assert t == T.StructType([
        T.StructField("a", T.DecimalType(10, 4)),
        T.StructField("b", T.ArrayType(T.StringType("UTF8_LCASE"))),
        T.StructField("m", T.MapType(T.StringType(), T.TimeType(5))),
    ])


def test_set_spatial_srids_and_precision_info_as_pysparks():
    # A set `srid` shows in the repr and simpleString of the type, as PySpark's; PySpark's JSON
    # stays that of the CRS of its constructor, which it keeps as `_crs` (`geometry(OGC:CRS84)`),
    # so that its type is equal to no GeometryType(0), as the drop-in's, of the same `__dict__`
    # (SPARK-62251). The mixed SRID -1, of which `GeographyType("ANY")` is, is an SRID too.
    geometry = T.GeometryType(4326)
    geometry.srid = 0
    assert (repr(geometry), geometry.simpleString()) == ("GeometryType(0)", "geometry(0)")
    assert geometry != T.GeometryType(0) and geometry.json() == '"geometry(OGC:CRS84)"'
    assert vars(geometry) == {"srid": 0, "_crs": "OGC:CRS84"}
    geography = T.GeographyType(4326)
    geography.srid = -1
    assert (repr(geography), geography.simpleString()) == ("GeographyType(ANY)", "geography(any)")
    assert geography != T.GeographyType("ANY")
    assert geography.json() == '"geography(OGC:CRS84, SPHERICAL)"'
    # `hasPrecisionInfo`, which PySpark's `__init__` sets true, is a plain attribute too, which
    # `==` compares, as PySpark's `__dict__`, and the repr and hash do not show.
    decimal = T.DecimalType(10, 2)
    decimal.hasPrecisionInfo = False
    assert decimal.hasPrecisionInfo is False and repr(decimal) == "DecimalType(10,2)"
    assert decimal != T.DecimalType(10, 2) and hash(decimal) == hash(T.DecimalType(10, 2))


def test_type_parameters_take_any_value_as_pysparks():
    # SPARK-62251: PySpark's attributes are entries of the `__dict__`, of any value, of which the
    # repr, JSON or simpleString fails later, as PySpark's: the TypeError of `%d` of a str, the
    # KeyError of a field that the type does not have. A plan of the type sets them in its proto,
    # which refuses a bool and an int out of the range of its field, as protobuf does. The texts
    # are official PySpark 4.3.0's.
    decimal, interval, geometry = T.DecimalType(10, 2), T.DayTimeIntervalType(), T.GeometryType(0)
    decimal.scale = "3"
    assert vars(decimal) == {"precision": 10, "scale": "3", "hasPrecisionInfo": True}
    for use in (repr, lambda t: t.simpleString(), lambda t: t.json()):
        with pytest.raises(TypeError, match="^%d format: a real number is required, not str$"):
            use(decimal)
    decimal.scale = 3.7
    assert (repr(decimal), decimal.json()) == ("DecimalType(10,3)", '"decimal(10,3)"')
    with pytest.raises(TypeError, match="^'float' object cannot be interpreted as an integer$"):
        F.col("v").cast(decimal)._expr.to_plan(None)
    decimal.scale = True
    with pytest.raises(TypeError) as e:
        F.col("v").cast(decimal)._expr.to_plan(None)
    assert str(e.value) == "Field spark.connect.DataType.Decimal.scale: Expected an int, got a boolean."
    string = T.StringType()
    string.collation = 5
    assert (repr(string), string.simpleString()) == ("StringType('5')", "string collate 5")
    with pytest.raises(AttributeError, match="^'int' object has no attribute 'startswith'$"):
        T.StructType([T.StructField("s", string)]).json()
    interval.startField = 7
    assert repr(interval) == "DayTimeIntervalType(7, 3)"
    with pytest.raises(KeyError, match="^7$"):
        interval.simpleString()
    interval.endField = None
    with pytest.raises(TypeError, match="^%d format: a real number is required, not NoneType$"):
        repr(interval)
    geometry.srid = 1234
    assert (repr(geometry), geometry.json()) == ("GeometryType(1234)", '"geometry(SRID:0)"')
    # The `"ANY"` of the constructor is the mixed SRID; a set "ANY" is the str itself.
    geometry.srid = "ANY"
    assert geometry.srid == "ANY" and geometry.simpleString() == "geometry(ANY)"


@pytest.mark.parametrize("name, stem", ABSTRACT_TYPES, ids=[name for name, _ in ABSTRACT_TYPES])
def test_the_abstract_types_are_types_of_their_classes_as_pysparks(name, stem):
    # As PySpark's plain classes, of the methods of its base DataType: the repr and str of the
    # class, the typeName, simpleString and jsonValue of the name of the class, no conversion of
    # the values, and the equality of the class and the `__dict__`. IntegralType is a singleton
    # of PySpark's DataTypeSingleton, whose `__call__` takes no arguments; `object.__new__` of
    # the others takes none either, and the class shows no parameters.
    cls = getattr(T, name)
    t, value = cls(), object()
    assert repr(t) == str(t) == f"{name}()"
    assert t.typeName() == cls.typeName() == t.simpleString() == t.jsonValue() == stem
    assert t.json() == f'"{stem}"'
    assert t.needConversion() is False
    assert t.toInternal(value) is value and t.fromInternal(value) is value
    assert t == cls() and not t != cls() and hash(t) == hash(cls()) == hash(f"{name}()")
    assert vars(t) == {}
    assert (t is cls()) is (cls is T.IntegralType)
    assert list(inspect.signature(cls).parameters) == []
    if cls is T.IntegralType:
        errors = [
            (lambda: cls(1), "takes 1 positional argument but 2 were given"),
            (lambda: cls(1, 2), "takes 1 positional argument but 3 were given"),
            (lambda: cls(x=1), "got an unexpected keyword argument 'x'"),
            (lambda: cls(1, x=1), "got an unexpected keyword argument 'x'"),
        ]
        errors = [(make, f"DataTypeSingleton.__call__() {message}") for make, message in errors]
    else:
        errors = [
            (make, f"{name}() takes no arguments")
            for make in (lambda: cls(1), lambda: cls(None, 2), lambda: cls(x=1),
                         lambda: cls.__new__(cls, 1))
        ]
    for make, message in errors:
        with pytest.raises(TypeError) as e:
            make()
        assert str(e.value) == message


def test_the_abstract_types_are_equal_of_their_classes_and_attributes_as_pysparks():
    # PySpark's `isinstance(other, self.__class__) and self.__dict__ == other.__dict__`, of which
    # `==` calls the `__eq__` of the subclass first, as Python does: so a type is equal to a type
    # of its class only, not of a subclass, nor of a base class. A direct `__eq__` of a base
    # class compares the `__dict__` of a subclass, which of a type of parameters holds them.
    for a, b in [
        (T.AtomicType(), T.DataType()),
        (T.DataType(), T.AtomicType()),
        (T.DataType(), T.IntegerType()),
        (T.IntegerType(), T.DataType()),
        (T.NumericType(), T.IntegralType()),
        (T.IntegralType(), T.ByteType()),
        (T.FractionalType(), T.DoubleType()),
        (T.DatetimeType(), T.DateType()),
        (T.AnyTimeType(), T.TimeType()),
        (T.SpatialType(), T.GeometryType(0)),
        (T.AnsiIntervalType(), T.DayTimeIntervalType()),
        (T.AtomicType(), T.StringType()),
        (T.DataType(), T.NullType()),
        (T.DataType(), T.StructType()),
        (T.DataType(), T.StructField("a", T.IntegerType())),
        (T.AtomicType(), 5),
    ]:
        assert a != b and not a == b, (a, b)
    assert [
        T.DataType().__eq__(T.AtomicType()),
        T.DataType().__eq__(T.IntegerType()),
        T.DataType().__eq__(T.VariantType()),
        T.NumericType().__eq__(T.IntegralType()),
        T.AtomicType().__eq__(T.StringType()),
        T.AtomicType().__eq__(T.DecimalType()),
        T.DataType().__eq__(T.StructField("a", T.IntegerType())),
        T.DataType().__ne__(T.AtomicType()),
    ] == [True, True, True, True, False, False, False, False]
    tagged, other = T.AtomicType(), T.AtomicType()
    tagged.tag = other.tag = "x"
    assert tagged == other and tagged != T.AtomicType()
    assert hash(tagged) == hash(T.AtomicType())


def test_types_of_the_abstract_types_are_pysparks():
    # A struct, array or map of a type of its class only names it by its typeName, in its
    # simpleString, JSON and tree string, as PySpark's, and converts no values of it.
    struct = T.StructType([
        T.StructField("a", T.DataType()),
        T.StructField("b", T.ArrayType(T.SpatialType()), False),
    ])
    assert repr(struct) == (
        "StructType([StructField('a', DataType(), True), "
        "StructField('b', ArrayType(SpatialType(), True), False)])"
    )
    assert struct.simpleString() == "struct<a:data,b:array<spatial>>"
    assert struct.json() == (
        '{"fields":[{"metadata":{},"name":"a","nullable":true,"type":"data"},'
        '{"metadata":{},"name":"b","nullable":false,"type":{"containsNull":true,'
        '"elementType":"spatial","type":"array"}}],"type":"struct"}'
    )
    assert struct.jsonValue() == {"type": "struct", "fields": [
        {"name": "a", "type": "data", "nullable": True, "metadata": {}},
        {"name": "b", "type": {"type": "array", "elementType": "spatial", "containsNull": True},
         "nullable": False, "metadata": {}},
    ]}
    assert struct.treeString() == (
        "root\n |-- a: data (nullable = true)\n |-- b: array (nullable = false)\n"
        " |    |-- element: spatial (containsNull = true)\n"
    )
    assert repr(struct.toNullable()) == (
        "StructType([StructField('a', DataType(), True), "
        "StructField('b', ArrayType(SpatialType(), True), True)])"
    )
    assert struct.needConversion() is True and struct.fieldNames() == ["a", "b"]
    assert struct.toInternal((1, [2])) == (1, [2]) and struct.fromInternal((1, [2])) == (1, [2])
    assert struct.add("c", T.AnyTimeType()).simpleString() == (
        "struct<a:data,b:array<spatial>,c:anytime>"
    )
    field = T.StructField("c", T.AnsiIntervalType())
    assert (repr(field), field.simpleString(), field.json()) == (
        "StructField('c', AnsiIntervalType(), True)",
        "c:ansiinterval",
        '{"metadata":{},"name":"c","nullable":true,"type":"ansiinterval"}',
    )
    map_type = T.MapType(T.StringType(), T.IntegralType())
    assert (repr(map_type), map_type.simpleString(), map_type.json()) == (
        "MapType(StringType(), IntegralType(), True)",
        "map<string,integral>",
        '{"keyType":"string","type":"map","valueContainsNull":true,"valueType":"integral"}',
    )
    assert map_type.valueType is T.IntegralType() and map_type.needConversion() is False
    assert map_type.toInternal({"a": 1}) == map_type.fromInternal({"a": 1}) == {"a": 1}
    array = T.ArrayType(T.NumericType(), False)
    assert (repr(array), array.simpleString(), array.json(), repr(array.toNullable())) == (
        "ArrayType(NumericType(), False)",
        "array<numeric>",
        '{"containsNull":false,"elementType":"numeric","type":"array"}',
        "ArrayType(NumericType(), True)",
    )
    assert array == T.ArrayType(T.NumericType(), False) and array != T.ArrayType(T.DataType(), False)
    assert hash(array) == hash(repr(array))


# Official PySpark 4.3.0's pickles at protocol 2 of types of the abstract types, of the `__dict__`
# of each type.
PYSPARKS_PICKLES_OF_ABSTRACT_TYPES = [
    ("MapType(StringType(), IntegralType())",
     b"\x80\x02cpyspark.sql.types\nMapType\nq\x00)\x81q\x01}q\x02(X\x07\x00\x00\x00keyTypeq\x03c"
     b"pyspark.sql.types\nStringType\nq\x04)\x81q\x05}q\x06X\t\x00\x00\x00collationq\x07X\x0b\x00"
     b"\x00\x00UTF8_BINARYq\x08sbX\t\x00\x00\x00valueTypeq\tcpyspark.sql.types\nIntegralType\nq"
     b"\n)\x81q\x0bX\x11\x00\x00\x00valueContainsNullq\x0c\x88ub."),
    ("ArrayType(DataType(), False)",
     b"\x80\x02cpyspark.sql.types\nArrayType\nq\x00)\x81q\x01}q\x02(X\x0b\x00\x00\x00elementType"
     b"q\x03cpyspark.sql.types\nDataType\nq\x04)\x81q\x05X\x0c\x00\x00\x00containsNullq\x06\x89u"
     b"b."),
    ("StructType([StructField('a', NumericType())])",
     b"\x80\x02cpyspark.sql.types\nStructType\nq\x00)\x81q\x01}q\x02(X\x06\x00\x00\x00fieldsq"
     b"\x03]q\x04cpyspark.sql.types\nStructField\nq\x05)\x81q\x06}q\x07(X\x04\x00\x00\x00nameq"
     b"\x08X\x01\x00\x00\x00aq\tX\x08\x00\x00\x00dataTypeq\ncpyspark.sql.types\nNumericType\nq"
     b"\x0b)\x81q\x0cX\x08\x00\x00\x00nullableq\r\x88X\x08\x00\x00\x00metadataq\x0e}q\x0fubaX"
     b"\x05\x00\x00\x00namesq\x10]q\x11h\taX\x0f\x00\x00\x00_needConversionq\x12]q\x13\x89aX\x16"
     b"\x00\x00\x00_needSerializeAnyFieldq\x14\x89ub."),
]


@pytest.mark.parametrize(
    "made, pickled",
    PYSPARKS_PICKLES_OF_ABSTRACT_TYPES,
    ids=[made for made, _ in PYSPARKS_PICKLES_OF_ABSTRACT_TYPES],
)
def test_types_of_the_abstract_types_pickle_as_pysparks(made, pickled):
    # PySpark's pickle of the `__dict__` of a struct, array or map of a type of its class only
    # loads as the same type. The drop-in's array and map pickle as PySpark's, of
    # `copyreg.__newobj__` of the class and the `__dict__`, and its struct of the types it is made
    # of, as of a user-defined type, since PySpark's `_parse_datatype_json_string` does not parse
    # the JSON of an abstract type (`"integral"`): both clients load them at every protocol.
    made = eval(made, vars(T))
    loaded = pickle.loads(pickled)
    assert type(loaded) is type(made) and loaded == made
    assert (repr(loaded), loaded.json()) == (repr(made), made.json())
    for protocol in range(6):
        copied = pickle.loads(pickle.dumps(made, protocol=protocol))
        assert type(copied) is type(made) and copied == made, protocol
        assert (repr(copied), copied.json()) == (repr(made), made.json()), protocol
    if isinstance(made, (T.ArrayType, T.MapType)):
        reduced = made.__reduce_ex__(2)
        assert reduced[:3] == (copyreg.__newobj__, (type(made),), vars(made))
    else:
        reduced, _ = made.__reduce__()
        assert reduced is type(made)


def test_pyspark_types_to_proto_types_refuses_values_that_are_not_types_as_pysparks():
    # As official PySpark's `pyspark_types_to_proto_types`, which converts a DataType only:
    # another value, a DDL string too, raises UNSUPPORTED_OPERATION of `f"data type {value}"`, so a
    # value whose str fails raises the error of its str. It sets the string of an UnparsedDataType
    # as protobuf converts it: bytes of UTF-8 too, and protobuf's TypeError of another value.
    from pyspark.sql.connect.types import UnparsedDataType, pyspark_types_to_proto_types

    class Unshown:
        def __str__(self):
            raise KeyError("no str")

    for value, operation in [
        (5, "data type 5"),
        ("int", "data type int"),
        (b"int", "data type b'int'"),
        (None, "data type None"),
    ]:
        with pytest.raises(PySparkValueError) as e:
            pyspark_types_to_proto_types(value)
        assert e.value.getCondition() == "UNSUPPORTED_OPERATION"
        assert e.value.getMessageParameters() == {"operation": operation}
    with pytest.raises(KeyError, match="no str"):
        pyspark_types_to_proto_types(Unshown())
    unparsed = pyspark_types_to_proto_types(UnparsedDataType(b"x int")).unparsed
    assert unparsed.data_type_string == "x int"
    with pytest.raises(TypeError, match="^bad argument type for built-in operation$"):
        pyspark_types_to_proto_types(UnparsedDataType(5))


def test_proto_schema_conversion_raises_pysparks_errors_of_types_that_it_does_not_convert():
    # As official PySpark 4.3.0's `proto_schema_to_pyspark_data_type`, whose last branch raises
    # UNSUPPORTED_OPERATION of `f"data type {schema}"`, of the proto text of the type, for a type
    # of no kind, such as `pb2.DataType()` or an element type that an array does not set, and of
    # a kind that it has no branch of, an unparsed type or a nanosecond-precision timestamp: in the
    # order of its conversion, of which a spatial type of an SRID that it does not take raises
    # ST_INVALID_SRID_VALUE first. A user-defined type is of its Python class, of no conversion of
    # its SQL type. The drop-in raised a PySparkRuntimeError of its own for a type of no kind,
    # and gave an UnparsedDataType.
    from pyspark.errors import IllegalArgumentException
    from pyspark.sql.connect import proto as pb2
    from pyspark.sql.connect.types import proto_schema_to_pyspark_data_type

    def struct(*builds):
        dt = pb2.DataType()
        dt.struct.SetInParent()
        for i, build in enumerate(builds):
            field = dt.struct.fields.add()
            field.name, field.nullable = f"f{i}", True
            if build is not None:
                build(field.data_type)
        return dt

    def unparsed(dt):
        dt.unparsed.data_type_string = 'a "b"'

    def nanos(dt):
        dt.timestamp_ntz_nanos.precision = 9

    def srid(dt):
        dt.geometry.srid = 2154

    array, map_ = pb2.DataType(), pb2.DataType()
    array.array.SetInParent()
    map_.map.key_type.string.SetInParent()
    for dt, text in [
        (pb2.DataType(), ""),
        (struct(None), ""),
        (struct(lambda dt: dt.long.SetInParent(), lambda dt: None), ""),
        (array, ""),
        (map_, ""),
        (struct(unparsed), 'unparsed {\n  data_type_string: "a \\"b\\""\n}\n'),
        (struct(nanos, None), "timestamp_ntz_nanos {\n  precision: 9\n}\n"),
        (struct(None, nanos), ""),
    ]:
        with pytest.raises(PySparkValueError) as e:
            proto_schema_to_pyspark_data_type(dt)
        assert e.value.getCondition() == "UNSUPPORTED_OPERATION"
        assert e.value.getMessageParameters() == {"operation": f"data type {text}"}
    with pytest.raises(IllegalArgumentException) as e:
        proto_schema_to_pyspark_data_type(struct(srid, None))
    assert e.value.getCondition() == "ST_INVALID_SRID_VALUE"
    for sql_type in (nanos, unparsed, lambda dt: dt.SetInParent()):
        udt = pb2.DataType()
        udt.udt.type, udt.udt.python_class = "udt", "pyspark.ml.linalg.VectorUDT"
        sql_type(udt.udt.sql_type)
        assert type(proto_schema_to_pyspark_data_type(udt)).__name__ == "VectorUDT"


def test_the_abstract_types_have_no_proto_nor_arrow_type_as_pysparks():
    # PySpark's `pyspark_types_to_proto_types` has no proto of a type of its class only, of which
    # it raises UNSUPPORTED_OPERATION, also in a struct, an array or a map. `cast` and `try_cast`
    # convert their type when the plan is built, as PySpark's: the Column renders the type's
    # `simpleString`, and its plan raises the error (see docs/compatibility.md).
    from pyspark.sql.connect.types import pyspark_types_to_proto_types

    cases = [
        (T.AtomicType(), "AtomicType()", "ATOMIC"),
        (T.ArrayType(T.DataType()), "DataType()", "ARRAY<DATA>"),
        (T.MapType(T.StringType(), T.IntegralType()), "IntegralType()", "MAP<STRING,INTEGRAL>"),
        (T.StructType([T.StructField("a", T.NumericType())]), "NumericType()", "STRUCT<A:NUMERIC>"),
    ]
    with mock_connect_session() as (spark, _):
        for t, shown, rendered in cases:
            with pytest.raises(PySparkValueError) as e:
                pyspark_types_to_proto_types(t)
            assert e.value.getCondition() == "UNSUPPORTED_OPERATION"
            assert e.value.getMessageParameters() == {"operation": f"data type {shown}"}
            for name, cast in (("CAST", F.col("a").cast), ("TRY_CAST", F.col("a").try_cast)):
                c = cast(t)
                assert repr(c) == f"Column<'{name}(a AS {rendered})'>"
                with pytest.raises(PySparkValueError) as e:
                    spark.range(1).select(c).schema
                assert e.value.getCondition() == "UNSUPPORTED_OPERATION"
                assert e.value.getMessageParameters() == {"operation": f"data type {shown}"}
    # Nor an Arrow type, of which PySpark's conversion of local rows raises its error, and the
    # plan of `DataFrame.to` raises UNSUPPORTED_OPERATION at the action, as PySpark's.
    from pyspark.errors import PySparkTypeError

    schema = T.StructType([T.StructField("a", T.AtomicType())])
    with mock_connect_session() as (spark, _):
        with pytest.raises(PySparkTypeError) as e:
            spark.createDataFrame([(1,)], schema)
        assert e.value.getCondition() == "UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION"
        assert e.value.getMessageParameters() == {"data_type": "AtomicType()"}
        with pytest.raises(PySparkValueError) as e:
            spark.range(1).to(schema).collect()
        assert e.value.getMessageParameters() == {"operation": "data type AtomicType()"}


class _AtomicSqlUDT(T.UserDefinedType):
    """A UserDefinedType whose SQL type holds a type of its class only."""

    @classmethod
    def sqlType(cls):
        return T.StructType([T.StructField("v", T.AtomicType())])

    @classmethod
    def module(cls):
        return __name__

    def serialize(self, obj):
        return (obj,)

    def deserialize(self, datum):
        return datum[0]


class _FieldSqlUDT(_AtomicSqlUDT):
    """A UserDefinedType whose SQL type holds a StructField as a type."""

    @classmethod
    def sqlType(cls):
        return T.ArrayType(T.StructField("x", T.IntegerType()))


def test_user_defined_types_of_sql_types_without_a_proto_raise_pysparks_errors():
    # Official PySpark's `pyspark_types_to_proto_types` converts the `sqlType()` of a
    # user-defined type, and so does its `to_arrow_type`: a type of its class only or a
    # StructField as a type in it raises the UNSUPPORTED_OPERATION of the plans of a cast, a UDF,
    # a Java UDF, a table, a UDTF and `DataFrame.to`, and the
    # UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION of local rows, where the drop-in sent the plans
    # for the server to refuse and raised a TypeError of the rows. No request is sent. A tree
    # string does not print the SQL type. The errors are official PySpark 4.3.0's.
    from pyspark.errors import PySparkTypeError
    from pyspark.sql.functions import udf, udtf

    class Fn:
        def eval(self):
            yield (None,)

    for udt, shown in [
        (_AtomicSqlUDT(), "AtomicType()"),
        (_FieldSqlUDT(), "StructField('x', IntegerType(), True)"),
    ]:
        operation = f"data type {shown}"
        schema = T.StructType([T.StructField("a", udt)])
        with mock_connect_session() as (spark, service):
            df = spark.range(1)
            for t in (udt, T.ArrayType(udt), T.MapType(T.StringType(), udt), schema):
                for cast in (F.col("id").cast, F.col("id").try_cast):
                    _raises_unsupported_operation(lambda: df.select(cast(t)).schema, operation)
            requests = [
                lambda: df.select(udf(lambda x: x, udt, useArrow=False)("id")).schema,
                lambda: spark.udf.registerJavaFunction("f", "x.Y", udt),
                lambda: spark.catalog.createTable("t", schema=schema),
                lambda: udtf(Fn, returnType=schema)().schema,
                lambda: df.to(schema).collect(),
            ]
            for request in requests:
                _raises_unsupported_operation(request, operation)
            with pytest.raises(PySparkTypeError) as e:
                spark.createDataFrame([(1,)], schema)
            assert e.value.getCondition() == "UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION"
            assert e.value.getMessageParameters() == {"data_type": shown}
            name = type(udt).__name__.lower()
            assert schema.treeString() == f"root\n |-- a: {name} (nullable = true)\n"
            assert service.analyze_requests == [] and service.execute_requests == []


class _NoSerializeUDT(T.UserDefinedType):
    """A UserDefinedType of a SQL type of its class only, which does not serialize."""

    @classmethod
    def sqlType(cls):
        return T.AtomicType()

    @classmethod
    def module(cls):
        return __name__


def test_local_rows_of_a_type_without_an_arrow_type_convert_their_values_first_as_pysparks():
    # As official PySpark's LocalDataToArrowConversion.convert, which runs the converters of the
    # columns, a UDT's `serialize` too, before `to_arrow_schema` raises the
    # UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION of a type of its class only or of a StructField
    # as a type, depth first with the types that Arrow has none of: the errors of the values come
    # first, where the drop-in raised that error before it converted them. The errors are
    # official PySpark 4.3.0's.
    from pyspark.errors import PySparkNotImplementedError, PySparkTypeError

    def struct(*fields):
        return T.StructType([T.StructField(name, t) for name, t in fields])

    atomic = ("a", T.AtomicType())
    with mock_connect_session() as (spark, service):
        with pytest.raises(PySparkNotImplementedError) as e:
            spark.createDataFrame([(1,)], struct(("a", _NoSerializeUDT())))
        assert e.value.getCondition() == "NOT_IMPLEMENTED"
        assert e.value.getMessageParameters() == {"feature": "toInternal()"}
        for data, schema in [
            ([("x", 1)], struct(("d", T.DecimalType()), atomic)),
            ([(1, "x")], struct(atomic, ("d", T.DecimalType()))),
            ([(1, 1)], struct(("b", T.BinaryType()), atomic)),
        ]:
            with pytest.raises(AssertionError, match="^$"):
                spark.createDataFrame(data, schema)
        not_null = T.StructType([T.StructField("a", T.AtomicType(), False)])
        with pytest.raises(PySparkValueError) as e:
            spark.createDataFrame([(None,)], not_null)
        assert str(e.value) == "input for AtomicType() must not be None"
        field = T.StructField("x", T.IntegerType())
        for data, schema, shown in [
            ([("x", 1)], struct(("c", T.CharType(3)), atomic), "CharType(3)"),
            ([(1, "x")], struct(atomic, ("c", T.CharType(3))), "AtomicType()"),
            ([1], T.AtomicType(), "AtomicType()"),
            (iter([(1,)]), struct(atomic), "AtomicType()"),
            ([(5,)], struct(("a", T.ArrayType(T.AtomicType()))), "AtomicType()"),
            ([(1,)], struct(("a", field)), "StructField('x', IntegerType(), True)"),
        ]:
            with pytest.raises(PySparkTypeError) as e:
                spark.createDataFrame(data, schema)
            assert e.value.getCondition() == "UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION", schema
            assert e.value.getMessageParameters() == {"data_type": shown}, schema
        assert service.analyze_requests == [] and service.execute_requests == []


# Types that official PySpark 4.3.0's `pyspark_types_to_proto_types` has no proto of, nested in a
# struct, and the operation of the UNSUPPORTED_OPERATION that it raises of each.
UNSENDABLE_TYPES = [
    (T.StructType([T.StructField("a", T.AtomicType())]), "data type AtomicType()"),
    (T.StructType([T.StructField("a", T.ArrayType(T.DataType()))]), "data type DataType()"),
    (
        T.StructType([T.StructField("a", T.MapType(T.StringType(), T.IntegralType()))]),
        "data type IntegralType()",
    ),
    (
        T.StructType([T.StructField("a", T.ArrayType(T.StructField("b", T.IntegerType())))]),
        "data type StructField('b', IntegerType(), True)",
    ),
    (T.AtomicType(), "data type AtomicType()"),
]


def _raises_unsupported_operation(call, operation):
    with pytest.raises(PySparkValueError) as e:
        call()
    assert e.value.getCondition() == "UNSUPPORTED_OPERATION"
    assert e.value.getMessageParameters() == {"operation": operation}


@pytest.mark.parametrize(
    "t, operation", UNSENDABLE_TYPES, ids=[t.simpleString() for t, _ in UNSENDABLE_TYPES]
)
def test_udtfs_java_udfs_and_tables_of_types_without_a_proto_raise_pysparks_errors(t, operation):
    # As official PySpark's `PythonUDTF.to_plan`, `JavaUDF` and `CreateTable` plan, which convert
    # the types with `pyspark_types_to_proto_types`: the DataFrame of a call of a UDTF raises its
    # UNSUPPORTED_OPERATION at each request, after the errors of its arguments; the registration
    # of a UDTF or of a Java UDF raises it at the call, and so does `createTable`, after the
    # errors of its name, path and description, and before those of its options. No request is
    # sent.
    from pyspark.sql.functions import udtf

    class Fn:
        def eval(self, x):
            yield (1,)

    with mock_connect_session() as (spark, service):
        sent = (len(service.analyze_requests), len(service.execute_requests))
        df = udtf(Fn, returnType=t)(F.lit(1))
        for request in (lambda: df.schema, df.collect, lambda: df.alias("x").columns):
            _raises_unsupported_operation(request, operation)
        with pytest.raises(UnicodeEncodeError):
            udtf(Fn, returnType=t)(F.lit("a\ud800")).schema
        _raises_unsupported_operation(
            lambda: spark.udtf.register("f", udtf(Fn, returnType=t)), operation
        )
        _raises_unsupported_operation(
            lambda: spark.udf.registerJavaFunction("f", "x.Y", t), operation
        )
        if isinstance(t, T.StructType):
            for create in (spark.catalog.createTable, spark.catalog.createExternalTable):
                _raises_unsupported_operation(lambda: create("t", schema=t), operation)
            _raises_unsupported_operation(
                lambda: spark.catalog.createTable("t", schema=t, k=1), operation
            )
            for name, path, description in [(1, None, None), ("t", 1, None), ("t", None, 1)]:
                with pytest.raises(TypeError, match="^bad argument type for built-in operation$"):
                    spark.catalog.createTable(
                        name, path=path, schema=t, description=description
                    )
        assert (len(service.analyze_requests), len(service.execute_requests)) == sent


class _OwnType(T.DataType):
    pass


class _OwnAtomicType(T.AtomicType):
    pass


def _local_data():
    """Data of each kind that `createDataFrame` takes, with rows and without: its key, and a
    function of it."""
    import numpy as np
    import pandas as pd
    import pyarrow as pa

    return {
        "list": lambda: [1],
        "empty list": lambda: [],
        "tuple": lambda: (1,),
        "empty tuple": lambda: (),
        "pandas": lambda: pd.DataFrame({"a": [1]}),
        "empty pandas": lambda: pd.DataFrame({"a": pd.Series([], dtype="int64")}),
        "numpy": lambda: np.array([1]),
        "empty numpy": lambda: np.array([], dtype="int64"),
        "pyarrow": lambda: pa.table({"a": [1]}),
        "empty pyarrow": lambda: pa.table({"a": pa.array([], type=pa.int64())}),
        "iterator": lambda: iter([1]),
    }


@pytest.mark.parametrize(
    "schema, arg_type", [(T.DataType(), "DataType"), (_OwnType(), "_OwnType")]
)
def test_create_dataframe_of_a_data_type_that_is_not_atomic_raises_pysparks_error(
    schema, arg_type
):
    # As official PySpark's `isinstance(schema, (AtomicType, StructType))`, of the class of the
    # schema: the base DataType() and a Python subclass of it are not expected, of any data, and
    # named by their class, where the drop-in named the class of the type that it kept them as.
    from pyspark.errors import PySparkTypeError

    with mock_connect_session() as (spark, _):
        for key, data in _local_data().items():
            with pytest.raises(PySparkTypeError) as e:
                spark.createDataFrame(data(), schema)
            assert e.value.getCondition() == "NOT_EXPECTED_TYPE", key
            assert e.value.getMessageParameters() == {
                "arg_name": "schema",
                "arg_type": arg_type,
                "expected_type": "list, None or StructType",
            }, key


@pytest.mark.parametrize(
    "schema, shown, json_text",
    [
        (T.AtomicType(), "AtomicType()", '"atomic"'),
        (T.IntegralType(), "IntegralType()", '"integral"'),
        (_OwnAtomicType(), "_OwnAtomicType()", '"_ownatomic"'),
    ],
)
def test_create_dataframe_of_a_type_of_its_class_only_is_pysparks(schema, shown, json_text):
    # As official PySpark, of an atomic type of its class only as the single type of the rows:
    # local rows of it raise UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION, and a pandas DataFrame
    # UNSUPPORTED_DATA_TYPE_FOR_ARROW, at the call; data of no rows, and a NumPy array or a
    # pyarrow Table, make a relation of no data of the JSON of the type, which the server fails
    # to parse (test_dropin_e2e.py), where the drop-in sent the JSON of the core's unparsed type.
    from pyspark.errors import PySparkTypeError

    errors = {
        "list": "UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION",
        "tuple": "UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION",
        "iterator": "UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION",
        "pandas": "UNSUPPORTED_DATA_TYPE_FOR_ARROW",
    }
    with mock_connect_session() as (spark, service):
        for key, data in _local_data().items():
            sent = (len(service.analyze_requests), len(service.execute_requests))
            if key in errors:
                with pytest.raises(PySparkTypeError) as e:
                    spark.createDataFrame(data(), schema)
                assert e.value.getCondition() == errors[key], key
                assert e.value.getMessageParameters() == {"data_type": shown}, key
                assert (len(service.analyze_requests), len(service.execute_requests)) == sent
                continue
            spark.createDataFrame(data(), schema).explain()
            request = base_pb2.AnalyzePlanRequest.FromString(service.analyze_requests[-1])
            relation = request.explain.plan.root.local_relation
            assert (relation.schema, relation.HasField("data")) == (json_text, False), key


def test_create_dataframe_of_no_rows_sends_the_json_of_its_schema_object_as_pysparks():
    # As official PySpark's createDataFrame of data of no rows, which sends the `json()` of the
    # schema object, an atomic one as it is: of a type of a Python subclass, of its methods, which
    # the server fails to parse (`"subinteger"`), and of the metadata dict of a field, of an int
    # of more than 64 bits as it is.
    import pandas as pd

    class SubIntegerType(T.IntegerType):
        pass

    class SchemaType(T.StructType):
        pass

    def struct(metadata, type_name, struct_name="struct"):
        return (f'{{"fields":[{{"metadata":{metadata},"name":"a","nullable":true,'
                f'"type":"{type_name}"}}],"type":"{struct_name}"}}')

    cases = [
        (SubIntegerType(), '"subinteger"'),
        (T.StructType([T.StructField("a", SubIntegerType())]), struct("{}", "subinteger")),
        (SchemaType([T.StructField("a", T.IntegerType())]), struct("{}", "integer", "schema")),
        (T.StructType([T.StructField("a", T.IntegerType(), True, {"k": 10**30})]),
         struct('{"k":1000000000000000000000000000000}', "integer")),
    ]
    with mock_connect_session() as (spark, service):
        for schema, sent in cases:
            for data in ([], (), pd.DataFrame({"a": []})):
                spark.createDataFrame(data, schema).explain()
                request = base_pb2.AnalyzePlanRequest.FromString(service.analyze_requests[-1])
                relation = request.explain.plan.root.local_relation
                assert (relation.schema, relation.HasField("data")) == (sent, False), data


def test_create_dataframe_of_rows_of_no_length_checks_the_schema_as_pysparks():
    # As official PySpark, whose conversion of local rows takes the rows of an iterator apart
    # first, and then checks the schema as of a list: a type of no proto nested in it raises
    # UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION, and a nullable flag that is not a bool the
    # AssertionError of its converter, where the drop-in checked neither of rows of no length.
    from pyspark.errors import PySparkTypeError

    with mock_connect_session() as (spark, _):
        for schema, shown in [
            (T.StructType([T.StructField("a", T.AtomicType())]), "AtomicType()"),
            (
                T.StructType([T.StructField("a", T.ArrayType(T.StructField("b", T.IntegerType())))]),
                "StructField('b', IntegerType(), True)",
            ),
        ]:
            with pytest.raises(PySparkTypeError) as e:
                spark.createDataFrame(iter([([1],)]), schema)
            assert e.value.getCondition() == "UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION"
            assert e.value.getMessageParameters() == {"data_type": shown}
        with pytest.raises(AssertionError):
            spark.createDataFrame(iter([(1,)]), T.StructType([T.StructField("a", T.IntegerType(), 1)]))
        with pytest.raises(IndexError, match="^list index out of range$"):
            spark.createDataFrame(iter([]), T.AtomicType())


def _same_named_classes():
    """Two classes of one name, so of one `typeName` and JSON, `x`, whose `simpleString` differ."""

    def make(n):
        class XType(T.DataType):
            def simpleString(self):
                return f"x{n}"

        return XType

    return make(1), make(2)


class _ScaleUDT(T.UserDefinedType):
    """A UserDefinedType of a parameter, whose JSON does not hold it."""

    def __init__(self, scale):
        self.scale = scale

    @classmethod
    def sqlType(cls):
        return T.DoubleType()

    @classmethod
    def module(cls):
        return __name__

    def serialize(self, obj):
        return obj * self.scale

    def deserialize(self, datum):
        return datum / self.scale

    def simpleString(self):
        return f"scale{self.scale}"


def test_nested_types_of_the_same_json_compare_by_their_objects_as_pysparks():
    # As official PySpark's `__dict__` comparison, which compares the type objects that a struct,
    # an array, a map or a field holds: two types of the same JSON, of one class and another
    # `__dict__`, of two classes of one name, or of a UserDefinedType of other parameters, make
    # types that are not equal, where the drop-in compared them by their JSON. Of the same
    # objects, or of equal ones, the types are equal.
    X1, X2 = _same_named_classes()
    tagged, other = T.AtomicType(), T.AtomicType()
    tagged.tag, other.tag = 1, 2
    for a, b in [(tagged, other), (X1(), X2()), (_ScaleUDT(2), _ScaleUDT(10))]:
        assert a.json() == b.json() and a != b
        for wrap in (
            T.ArrayType,
            lambda t: T.MapType(T.StringType(), t),
            lambda t: T.MapType(t, T.StringType()),
            lambda t: T.StructField("f", t),
            lambda t: T.StructType([T.StructField("f", t)]),
            lambda t: T.StructType([T.StructField("f", T.ArrayType(T.MapType(T.StringType(), t)))]),
        ):
            assert wrap(a) != wrap(b) and not wrap(a) == wrap(b)
            assert wrap(a) == wrap(a) and wrap(b) == wrap(b)
            assert hash(wrap(a)) == hash(wrap(b))
    same = T.AtomicType()
    same.tag = 1
    assert T.ArrayType(tagged) == T.ArrayType(same)
    assert T.StructType([T.StructField("a", _ScaleUDT(2)), T.StructField("b", _ScaleUDT(10))]) != (
        T.StructType([T.StructField("a", _ScaleUDT(10)), T.StructField("b", _ScaleUDT(2))])
    )


def test_nested_types_of_the_same_json_render_by_their_objects_as_pysparks():
    # As official PySpark, which names each type that a struct, an array or a map holds by its
    # object: its `simpleString`, its `typeName` in the tree string, and its object in a copy of
    # `toNullable`, which keeps the types that are not arrays, maps nor structs as they are. The
    # drop-in named them by the last object of their JSON that a type was made of.
    X1, X2 = _same_named_classes()
    x1, x2 = X1(), X2()
    struct = T.StructType([T.StructField("a", x1), T.StructField("b", x2, False)])
    assert struct.simpleString() == "struct<a:x1,b:x2>"
    assert T.MapType(x1, T.ArrayType(x2)).simpleString() == "map<x1,array<x2>>"
    assert T.StructType([T.StructField("a", x1)]).add("b", x2).simpleString() == "struct<a:x1,b:x2>"
    nullable = struct.toNullable()
    assert [f.dataType for f in nullable.fields] == [x1, x2]
    assert nullable.fields[0].dataType is x1 and nullable.fields[1].dataType is x2
    assert [f.nullable for f in nullable.fields] == [True, True]
    assert nullable.simpleString() == "struct<a:x1,b:x2>"
    array = T.ArrayType(T.MapType(T.StringType(), x1), False).toNullable()
    assert array.elementType.valueType is x1 and array.containsNull is True
    assert array.elementType.valueContainsNull is True
    udts = T.StructType([T.StructField("a", _ScaleUDT(2)), T.StructField("b", _ScaleUDT(10))])
    assert udts.simpleString() == "struct<a:scale2,b:scale10>"
    assert [f.dataType.scale for f in udts.toNullable().fields] == [2, 10]
    assert udts.toInternal((1.0, 1.0)) == (2.0, 10.0)

    class Alpha(T.AtomicType):
        @classmethod
        def typeName(cls):
            return "alpha"

        def jsonValue(self):
            return "same"

    class Beta(T.AtomicType):
        @classmethod
        def typeName(cls):
            return "beta"

        def jsonValue(self):
            return "same"

    tree = T.StructType([
        T.StructField("a", Alpha()),
        T.StructField("b", T.ArrayType(Beta())),
        T.StructField("c", T.MapType(Alpha(), Beta())),
    ])
    assert tree.treeString() == (
        "root\n |-- a: alpha (nullable = true)\n |-- b: array (nullable = true)\n"
        " |    |-- element: beta (containsNull = true)\n |-- c: map (nullable = true)\n"
        " |    |-- key: alpha\n |    |-- value: beta (valueContainsNull = true)\n"
    )
    assert tree.simpleString() == "struct<a:alpha,b:array<beta>,c:map<alpha,beta>>"


class _NoModuleUDT(T.UserDefinedType):
    """A UserDefinedType whose module() is not implemented, of a SQL type of no conversion."""

    @classmethod
    def sqlType(cls):
        return T.LongType()


class _NoSqlTypeUDT(T.UserDefinedType):
    """A UserDefinedType whose sqlType() is not implemented."""

    @classmethod
    def module(cls):
        return __name__


def test_types_of_a_user_defined_type_of_no_json_keep_it_as_pysparks():
    # As official PySpark's ArrayType, MapType, StructField and StructType, which keep the type
    # objects that they are given as they are: of a UserDefinedType whose module() or sqlType()
    # is not implemented, such as UserDefinedType() itself, they are made, also as a setter takes
    # it, and their repr, simpleString, needConversion, ==, hash, copies, pickles, tree string,
    # names, add and toNullable read the objects. Their json() and jsonValue() raise the error of
    # the type's jsonValue(), and toInternal and fromInternal that of its own methods. The
    # drop-in raised the error of its JSON when they were made. The values are official PySpark
    # 4.3.0's.
    from pyspark.errors import PySparkNotImplementedError

    def raises(call, feature):
        with pytest.raises(PySparkNotImplementedError) as e:
            call()
        assert e.value.getCondition() == "NOT_IMPLEMENTED"
        assert e.value.getMessageParameters() == {"feature": feature}

    for make, json_feature, to_feature, from_feature in [
        (T.UserDefinedType, "module()", "sqlType()", "sqlType()"),
        (_NoModuleUDT, "module()", "toInternal()", "fromInternal()"),
        (_NoSqlTypeUDT, "sqlType()", "sqlType()", "sqlType()"),
    ]:
        name = make.__name__
        cases = [
            (lambda: T.ArrayType(make()), f"ArrayType({name}(), True)", "array<udt>"),
            (lambda: T.ArrayType(make(), False), f"ArrayType({name}(), False)", "array<udt>"),
            (
                lambda: T.MapType(T.StringType(), make()),
                f"MapType(StringType(), {name}(), True)",
                "map<string,udt>",
            ),
            (
                lambda: T.MapType(make(), T.StringType()),
                f"MapType({name}(), StringType(), True)",
                "map<udt,string>",
            ),
            (lambda: T.StructField("a", make()), f"StructField('a', {name}(), True)", "a:udt"),
            (
                lambda: T.StructType([T.StructField("a", make())]),
                f"StructType([StructField('a', {name}(), True)])",
                "struct<a:udt>",
            ),
            (
                lambda: T.ArrayType(T.ArrayType(make())),
                f"ArrayType(ArrayType({name}(), True), True)",
                "array<array<udt>>",
            ),
        ]
        for build, shown, simple in cases:
            t = build()
            assert (repr(t), str(t), t.simpleString()) == (shown, shown, simple)
            assert t.needConversion() is True, shown
            assert t == build() and not t != build() and hash(t) == hash(build()), shown
            for copied in (copy.copy(t), copy.deepcopy(t), pickle.loads(pickle.dumps(t))):
                assert type(copied) is type(t) and copied == t and repr(copied) == shown
            raises(t.json, json_feature)
            raises(t.jsonValue, json_feature)
        raises(lambda: T.ArrayType(make()).toInternal([1]), to_feature)
        raises(lambda: T.ArrayType(make()).fromInternal([1]), from_feature)
        raises(lambda: T.StructType([T.StructField("a", make())]).toInternal((1,)), to_feature)
        struct = T.StructType([T.StructField("a", make()), T.StructField("b", T.ArrayType(make()))])
        assert struct.typeName() == "struct" and T.ArrayType(make()).typeName() == "array"
        assert (struct.names, struct.fieldNames(), len(struct)) == (["a", "b"], ["a", "b"], 2)
        assert struct["a"] is struct.fields[0] and struct.fields[0].dataType == make()
        assert struct.treeString() == (
            f"root\n |-- a: {name.lower()} (nullable = true)\n |-- b: array (nullable = true)\n"
            f" |    |-- element: {name.lower()} (containsNull = true)\n"
        )
        added = T.StructType([T.StructField("x", T.IntegerType())]).add("b", make())
        assert repr(added) == (
            f"StructType([StructField('x', IntegerType(), True), StructField('b', {name}(), True)])"
        )
        assert (added.names, added._needConversion) == (["x", "b"], [False, True])
        nullable = T.StructType([T.StructField("a", T.ArrayType(make(), False), False)])
        assert repr(nullable.toNullable()) == (
            f"StructType([StructField('a', ArrayType({name}(), True), True)])"
        )
        assert repr(T.MapType(T.StringType(), make(), False).toNullable()) == (
            f"MapType(StringType(), {name}(), True)"
        )
        array, field = T.ArrayType(T.IntegerType()), T.StructField("a", T.IntegerType())
        kv = T.MapType(T.StringType(), T.IntegerType())
        array.elementType, field.dataType, kv.keyType = make(), make(), make()
        assert (repr(array), repr(field), repr(kv)) == (
            f"ArrayType({name}(), True)",
            f"StructField('a', {name}(), True)",
            f"MapType({name}(), IntegerType(), True)",
        )
        raises(array.json, json_feature)


class _SubBooleanType(T.BooleanType):
    """A Python subclass of a type, whose `typeName()`, `_subboolean`, PySpark's methods read."""


def test_casts_show_the_simple_strings_of_their_type_objects_as_pysparks():
    # As official PySpark's CastExpression.__repr__, of the `simpleString()` of the type object in
    # upper case: of a user-defined type of its own simpleString(), also in an array, a map or a
    # struct, and of a type of a Python subclass, where the drop-in showed `UDT` and the
    # simpleString of the core type of the class. The plans are those of the types. A cast to a
    # type that does not lower now, such as an array of UserDefinedType(), is made, and each
    # request raises the error of its JSON. The strings are official PySpark 4.3.0's.
    from pyspark.errors import PySparkNotImplementedError

    scales = T.StructType([T.StructField("a", _ScaleUDT(2)), T.StructField("b", _ScaleUDT(10))])
    for data_type, shown in [
        (_ScaleUDT(2), "SCALE2"),
        (scales, "STRUCT<A:SCALE2,B:SCALE10>"),
        (T.ArrayType(_ScaleUDT(3)), "ARRAY<SCALE3>"),
        (T.MapType(T.StringType(), _ScaleUDT(4)), "MAP<STRING,SCALE4>"),
        (_SubBooleanType(), "_SUBBOOLEAN"),
        (T.ArrayType(_SubBooleanType()), "ARRAY<_SUBBOOLEAN>"),
        (T.DecimalType(10, 2), "DECIMAL(10,2)"),
        (T.ArrayType(T.UserDefinedType()), "ARRAY<UDT>"),
    ]:
        for name, cast in (("CAST", F.col("id").cast), ("TRY_CAST", F.col("id").try_cast)):
            column = cast(data_type)
            assert repr(column) == f"Column<'{name}(id AS {shown})'>"
            assert repr(column.alias("v")) == f"Column<'{name}(id AS {shown}) AS v'>"
    with mock_connect_session() as (spark, service):
        column = F.col("id").cast(T.ArrayType(T.UserDefinedType()))
        for request in (lambda: spark.range(1).select(column).schema,
                        spark.range(1).select(column).collect):
            with pytest.raises(PySparkNotImplementedError) as e:
                request()
            assert e.value.getMessageParameters() == {"feature": "module()"}
        assert service.analyze_requests == [] and service.execute_requests == []


def test_python_subclasses_of_the_abstract_types_are_made_as_pysparks():
    # As PySpark's plain classes, whose `object.__new__` takes the arguments that the `__init__`
    # of a subclass takes, such as that of a UserDefinedType of parameters, and raises them
    # otherwise. A subclass is a type of its class, whose `typeName` is the name of the class,
    # and whose state is its `__dict__`.
    class Q(T.DataType):
        pass

    class R(T.AtomicType):
        def __new__(cls, *args):
            return super().__new__(cls, *args)

    class S(T.NumericType):
        def __init__(self, x):
            self.x = x

    class U(T.DataType):
        def __init__(self, x):
            super().__init__(x)

    for make, message in [
        (lambda: Q(1), "Q() takes no arguments"),
        (lambda: T.UserDefinedType(1), "UserDefinedType() takes no arguments"),
        (lambda: T.VariantType(1), "VariantType() takes no arguments"),
        (lambda: R(1), "object.__new__() takes exactly one argument (the type to instantiate)"),
        (lambda: U(1), "object.__init__() takes exactly one argument (the instance to initialize)"),
        (lambda: S(), f"{S.__init__.__qualname__}() missing 1 required positional argument: 'x'"),
    ]:
        with pytest.raises(TypeError) as e:
            make()
        assert str(e.value) == message
    assert (Q.typeName(), Q().simpleString(), Q().json()) == ("", "", '""')
    assert repr(R()) == "R()" and vars(S(1)) == {"x": 1}
    assert S(1) == S(1) and S(1) != S(2) and S(1) != T.NumericType()
    # A Python subclass of IntegralType has the one instance of its own that PySpark's
    # DataTypeSingleton keeps of each class, and takes no arguments.
    class MyIntegral(T.IntegralType):
        pass

    assert MyIntegral() is MyIntegral() and MyIntegral() is not T.IntegralType()
    assert (repr(MyIntegral()), MyIntegral.typeName(), MyIntegral().json()) == (
        "MyIntegral()", "myinte", '"myinte"'
    )
    assert MyIntegral() != T.IntegralType() and T.IntegralType() != MyIntegral()
    with pytest.raises(TypeError) as e:
        MyIntegral(1)
    assert str(e.value) == (
        "DataTypeSingleton.__call__() takes 1 positional argument but 2 were given"
    )


# PySpark's subclass `class Sub<name>(<name>)` of each concrete type, made of the arguments (of
# the expressions of the strings): its repr, `typeName()`, `simpleString()` and `json()`, the name
# that a tree string writes of it, and whether it is of PySpark's `DataTypeSingleton`.
CONCRETE_SUBCLASSES = [
    ("NullType", (), "SubNullType()", "void", "void", '"void"', "void", True),
    ("BooleanType", (), "SubBooleanType()", "subboolean", "subboolean", '"subboolean"',
     "subboolean", True),
    ("ByteType", (), "SubByteType()", "subbyte", "tinyint", '"subbyte"', "subbyte", True),
    ("ShortType", (), "SubShortType()", "subshort", "smallint", '"subshort"', "subshort", True),
    ("IntegerType", (), "SubIntegerType()", "subinteger", "int", '"subinteger"', "subinteger",
     True),
    ("LongType", (), "SubLongType()", "sublong", "bigint", '"sublong"', "sublong", True),
    ("FloatType", (), "SubFloatType()", "subfloat", "subfloat", '"subfloat"', "subfloat", True),
    ("DoubleType", (), "SubDoubleType()", "subdouble", "subdouble", '"subdouble"', "subdouble",
     True),
    ("BinaryType", (), "SubBinaryType()", "subbinary", "subbinary", '"subbinary"', "subbinary",
     True),
    ("DateType", (), "SubDateType()", "subdate", "subdate", '"subdate"', "subdate", True),
    ("TimestampType", (), "SubTimestampType()", "subtimestamp", "subtimestamp",
     '"subtimestamp"', "subtimestamp", True),
    ("TimestampNTZType", (), "SubTimestampNTZType()", "timestamp_ntz", "timestamp_ntz",
     '"timestamp_ntz"', "timestamp_ntz", True),
    ("CalendarIntervalType", (), "SubCalendarIntervalType()", "interval", "interval",
     '"interval"', "interval", True),
    ("VariantType", (), "SubVariantType()", "subvariant", "subvariant", '"subvariant"',
     "subvariant", False),
    ("StringType", ("'UTF8_LCASE'",), "StringType('UTF8_LCASE')", "substring",
     "string collate UTF8_LCASE", '"string collate UTF8_LCASE"', "substring", False),
    ("CharType", (5,), "CharType(5)", "subchar", "char(5)", '"char(5)"', "char(5)", False),
    ("VarcharType", (5,), "VarcharType(5)", "subvarchar", "varchar(5)", '"varchar(5)"',
     "varchar(5)", False),
    ("TimeType", (3,), "TimeType(3)", "subtime", "time(3)", '"time(3)"', "time(3)", False),
    ("DecimalType", (10, 2), "DecimalType(10,2)", "subdecimal", "decimal(10,2)",
     '"decimal(10,2)"', "decimal(10,2)", False),
    ("DayTimeIntervalType", (1, 2), "SubDayTimeIntervalType(1, 2)", "subdaytimeinterval",
     "interval hour to minute", '"interval hour to minute"', "interval hour to minute", False),
    ("YearMonthIntervalType", (0, 0), "SubYearMonthIntervalType(0, 0)", "subyearmonthinterval",
     "interval year", '"interval year"', "interval year", False),
    ("GeometryType", (4326,), "GeometryType(4326)", "subgeometry", "geometry(4326)",
     '"geometry(OGC:CRS84)"', "subgeometry", False),
    ("GeographyType", (4326,), "GeographyType(4326)", "subgeography", "geography(4326)",
     '"geography(OGC:CRS84, SPHERICAL)"', "subgeography", False),
    ("ArrayType", ("IntegerType()", False), "ArrayType(IntegerType(), False)", "subarray",
     "array<int>", '{"containsNull":false,"elementType":"integer","type":"subarray"}', "subarray",
     False),
    ("MapType", ("StringType()", "IntegerType()", False),
     "MapType(StringType(), IntegerType(), False)", "submap", "map<string,int>",
     '{"keyType":"string","type":"submap","valueContainsNull":false,"valueType":"integer"}',
     "submap", False),
    ("StructType", ("[StructField('a', IntegerType())]",),
     "StructType([StructField('a', IntegerType(), True)])", "substruct", "struct<a:int>",
     '{"fields":[{"metadata":{},"name":"a","nullable":true,"type":"integer"}],'
     '"type":"substruct"}', "substruct", False),
    ("StructField", ("'a'", "IntegerType()", False), "StructField('a', IntegerType(), False)",
     None, "a:int", '{"metadata":{},"name":"a","nullable":false,"type":"integer"}', None, False),
]


@pytest.mark.parametrize(
    "name, args, shown, type_name, simple, json_text, tree, singleton",
    CONCRETE_SUBCLASSES,
    ids=[case[0] for case in CONCRETE_SUBCLASSES],
)
def test_python_subclasses_of_the_concrete_types_are_pysparks(
    name, args, shown, type_name, simple, json_text, tree, singleton
):
    # PySpark's types are plain classes, which a user subclasses: the methods of a class call the
    # methods of the object, so the `typeName()` of a subclass is the name of the subclass, of
    # which `simpleString()` and `jsonValue()` are where the class has none of its own, and the
    # `type` of the JSON of an array, a map and a struct; the repr of a type that names its class
    # names the subclass. A type of a subclass equals only a type of its class (or of a subclass
    # of it), of the same `__dict__`, and `==` calls the `__eq__` of the subclass first, so a type
    # of the base class and one of the subclass are not equal in either order. A subclass of a
    # singleton type is of PySpark's `DataTypeSingleton` too, of one instance of its own.
    base = getattr(T, name)
    sub = type("Sub" + name, (base,), {})
    made = [eval(arg, vars(T)) if isinstance(arg, str) else arg for arg in args]
    t, again, of_base = sub(*made), sub(*made), base(*made)
    assert type(t) is sub and isinstance(t, base) and isinstance(t, T.DataType)
    assert not isinstance(of_base, sub) and issubclass(sub, base)
    assert (repr(t), str(t), t.simpleString(), t.json()) == (shown, shown, simple, json_text)
    assert json.dumps(t.jsonValue(), separators=(",", ":"), sort_keys=True) == json_text
    if type_name is not None:
        assert (t.typeName(), sub.typeName()) == (type_name, type_name)
    assert t == again and not (t != again) and hash(t) == hash(again) == hash(shown)
    assert t != of_base and of_base != t and not (t == of_base) and not (of_base == t)
    assert (again is t) is singleton
    if singleton:
        assert sub() is not base()
        with pytest.raises(TypeError) as e:
            sub(1)
        assert str(e.value) == "DataTypeSingleton.__call__() takes 1 positional argument but 2 " \
                               "were given"
    for copied in (copy.copy(t), copy.deepcopy(t)):
        assert type(copied) is sub and copied == t and (copied is t) is singleton
    if tree is None:
        return
    # Nested in another type, it is the object that the type holds, which names it, writes it
    # and compares it of its own methods.
    struct = T.StructType([T.StructField("f", t)])
    assert struct.fields[0].dataType is t and T.ArrayType(t).elementType is t
    assert struct.treeString().splitlines()[1] == f" |-- f: {tree} (nullable = true)"
    assert struct.simpleString() == f"struct<f:{simple}>"
    # Of a collation, the JSON of a field holds the base class, as PySpark's `removeCollations`.
    written = "string" if name == "StringType" else t.jsonValue()
    assert json.loads(struct.json())["fields"][0]["type"] == written
    assert T.ArrayType(t).simpleString() == f"array<{simple}>"
    assert repr(T.MapType(T.StringType(), t)) == f"MapType(StringType(), {shown}, True)"
    assert T.ArrayType(t) == T.ArrayType(again) and T.ArrayType(t) != T.ArrayType(of_base)
    assert struct != T.StructType([T.StructField("f", of_base)])
    nullable = T.StructType([T.StructField("f", t, False)]).toNullable().fields[0].dataType
    assert (nullable is t) is not isinstance(t, (T.ArrayType, T.MapType, T.StructType))


# Official PySpark 4.3.0's cloudpickles of `class SubLong(LongType): pass` of `__main__`, which it
# sends by value, of the class and of its instance: they name the metaclass of the class,
# `pyspark.sql.types.DataTypeSingleton`.
PYSPARKS_CLOUDPICKLED_SINGLETON_SUBCLASS = (
    b"\x80\x05\x95\x02\x01\x00\x00\x00\x00\x00\x00\x8c\x1fpyspark.cloudpickle.cloudpickle\x94\x8c"
    b"\x14_make_skeleton_class\x94\x93\x94(\x8c\x11pyspark.sql.types\x94\x8c\x11DataTypeSingleton"
    b"\x94\x93\x94\x8c\x07SubLong\x94h\x03\x8c\x08LongType\x94\x93\x94\x85\x94}\x94\x8c\n__module_"
    b"_\x94\x8c\x08__main__\x94s\x8c 50b6ec2a58624c058b858a5d4aa44b73\x94Nt\x94R\x94h\x00\x8c\x0f_"
    b"class_setstate\x94\x93\x94h\x0f}\x94(\x8c\x07__doc__\x94N\x8c\n__module__\x94h\x0cu}\x94\x86"
    b"\x94\x86R0."
)
PYSPARKS_CLOUDPICKLED_SINGLETON_SUBCLASS_INSTANCE = (
    b"\x80\x05\x95\x17\x01\x00\x00\x00\x00\x00\x00\x8c\x1fpyspark.cloudpickle.cloudpickle\x94\x8c"
    b"\x14_make_skeleton_class\x94\x93\x94(\x8c\x11pyspark.sql.types\x94\x8c\x11DataTypeSingleton"
    b"\x94\x93\x94\x8c\x07SubLong\x94h\x03\x8c\x08LongType\x94\x93\x94\x85\x94}\x94\x8c\n__module_"
    b"_\x94\x8c\x08__main__\x94s\x8c 50b6ec2a58624c058b858a5d4aa44b73\x94Nt\x94R\x94h\x00\x8c\x0f_"
    b"class_setstate\x94\x93\x94h\x0f}\x94(\x8c\x07__doc__\x94N\x8c\n__module__\x94h\x0c\x8c\r__sl"
    b"otnames__\x94]\x94u}\x94\x86\x94\x86R0)\x81\x94."
)


def test_pysparks_cloudpickles_of_a_subclass_of_a_singleton_type_load_of_its_metaclass():
    # PySpark's singleton types are classes of its metaclass `DataTypeSingleton`, which its
    # cloudpickle of a subclass of one of `__main__` names: `pyspark.sql.types` has PySpark's
    # metaclass, of which such a class loads, of the native class as its base, and keeps the one
    # instance of the class, which its instance's pickle loads as. The native classes themselves
    # are of the metaclass `type`.
    assert type(T.LongType) is type and T.DataTypeSingleton.__bases__ == (type,)
    sub = pickle.loads(PYSPARKS_CLOUDPICKLED_SINGLETON_SUBCLASS)
    assert type(sub) is T.DataTypeSingleton and sub.__bases__ == (T.LongType,)
    assert (sub.__name__, sub.__module__) == ("SubLong", "__main__")
    t = sub()
    assert t is sub() and type(t) is sub and t is not T.LongType()
    assert T.DataTypeSingleton._instances[sub] is t
    assert (repr(t), t.json(), t.simpleString()) == ("SubLong()", '"sub"', "bigint")
    assert pickle.loads(PYSPARKS_CLOUDPICKLED_SINGLETON_SUBCLASS_INSTANCE) is t
    # A class of the metaclass is made as PySpark's, which takes no arguments for its instance.
    made = T.DataTypeSingleton("Made", (T.IntegerType,), {})
    assert made() is made() and isinstance(made(), T.IntegerType)
    with pytest.raises(TypeError) as e:
        made(1)
    assert str(e.value) == (
        "DataTypeSingleton.__call__() takes 1 positional argument but 2 were given"
    )


class _Money(T.DecimalType):
    """A decimal of a currency, a subclass of PySpark's plain class."""

    kind = "money"

    def __init__(self, currency, precision=12, scale=2):
        super().__init__(precision, scale)
        self.currency = currency

    def describe(self):
        return f"{self.currency}:{self.simpleString()}"


class _Tag(T.IntegerType):
    pass


class _Names(T.StructType):
    def __init__(self, *names):
        super().__init__([T.StructField(name, T.StringType()) for name in names])


class _Slotted(T.StringType):
    __slots__ = ("tag",)


# Official PySpark 4.3.0's pickles of `_Money("EUR")`, of `_Tag()` and of a `_Slotted("UTF8_LCASE")`
# of the slot `tag` "x", of their classes of this module, at the protocols 2 and 5.
PYSPARKS_SUBCLASS_PICKLES = {
    ("money", 2): b"\x80\x02ctest_dropin_types_columns_offline\n_Money\nq\x00)\x81q\x01}q\x02(X"
                  b"\t\x00\x00\x00precisionq\x03K\x0cX\x05\x00\x00\x00scaleq\x04K\x02X\x10\x00"
                  b"\x00\x00hasPrecisionInfoq\x05\x88X\x08\x00\x00\x00currencyq\x06X\x03\x00\x00"
                  b"\x00EURq\x07ub.",
    ("money", 5): b"\x80\x05\x95u\x00\x00\x00\x00\x00\x00\x00\x8c!test_dropin_types_columns_"
                  b"offline\x94\x8c\x06_Money\x94\x93\x94)\x81\x94}\x94(\x8c\tprecision\x94K"
                  b"\x0c\x8c\x05scale\x94K\x02\x8c\x10hasPrecisionInfo\x94\x88\x8c\x08currency"
                  b"\x94\x8c\x03EUR\x94ub.",
    ("tag", 2): b"\x80\x02ctest_dropin_types_columns_offline\n_Tag\nq\x00)\x81q\x01.",
    ("tag", 5): b"\x80\x05\x951\x00\x00\x00\x00\x00\x00\x00\x8c!test_dropin_types_columns_offline"
                b"\x94\x8c\x04_Tag\x94\x93\x94)\x81\x94.",
    ("slotted", 2): b"\x80\x02ctest_dropin_types_columns_offline\n_Slotted\nq\x00)\x81q\x01}q\x02X"
                    b"\t\x00\x00\x00collationq\x03X\n\x00\x00\x00UTF8_LCASEq\x04s}q\x05X\x03"
                    b"\x00\x00\x00tagq\x06X\x01\x00\x00\x00xq\x07s\x86q\x08b.",
    ("slotted", 5): b"\x80\x05\x95a\x00\x00\x00\x00\x00\x00\x00\x8c!test_dropin_types_columns_"
                    b"offline\x94\x8c\x08_Slotted\x94\x93\x94)\x81\x94}\x94\x8c\tcollation\x94"
                    b"\x8c\nUTF8_LCASE\x94s}\x94\x8c\x03tag\x94\x8c\x01x\x94s\x86\x94b.",
}


def test_python_subclasses_of_the_types_initialize_and_pickle_as_pysparks():
    # A subclass's `__init__` takes arguments of its own, which the class's `__new__` takes as
    # `object.__new__` does, and passes the parameters to the class's `__init__`; its class
    # attributes, methods and instance attributes are its own. A type of a subclass pickles at
    # every protocol from 2 as PySpark's plain class: of `copyreg.__newobj__(cls)` and of the
    # `__dict__` of PySpark's type, its parameters and the attributes that a user set, byte for
    # byte, so each client loads the other's, and copies of it keep its class and attributes.
    money = _Money("EUR")
    assert (repr(money), money.kind, money.describe(), money.currency) == (
        "DecimalType(12,2)", "money", "EUR:decimal(12,2)", "EUR"
    )
    assert _Money(currency="USD", scale=4) == _Money("USD", 12, 4) != _Money("USD")
    assert money == _Money("EUR") and money != _Money("USD") and money != T.DecimalType(12, 2)
    assert hash(money) == hash(T.DecimalType(12, 2))
    # The parameters are entries of the `__dict__`, as PySpark's (SPARK-62251).
    assert vars(money) == {"precision": 12, "scale": 2, "hasPrecisionInfo": True, "currency": "EUR"}
    state = {"precision": 12, "scale": 2, "hasPrecisionInfo": True, "currency": "EUR"}
    assert money.__reduce_ex__(2) == (copyreg.__newobj__, (_Money,), state, None, None)
    slotted = _Slotted("UTF8_LCASE")
    slotted.tag = "x"
    for (name, protocol), pickled in PYSPARKS_SUBCLASS_PICKLES.items():
        t = {"money": money, "tag": _Tag(), "slotted": slotted}[name]
        assert pickle.dumps(t, protocol=protocol) == pickled
        assert type(pickle.loads(pickled)) is type(t) and pickle.loads(pickled) == t
    # The slots of a subclass of `__slots__` are of the state of its pickles and copies.
    for loaded in (pickle.loads(pickle.dumps(slotted)), copy.copy(slotted), copy.deepcopy(slotted)):
        assert (type(loaded), loaded.tag, loaded.collation) == (_Slotted, "x", "UTF8_LCASE")
    for protocol in range(6):
        loaded = pickle.loads(pickle.dumps(money, protocol=protocol))
        assert type(loaded) is _Money and loaded == money and vars(loaded) == vars(money)
    for copied in (copy.copy(money), copy.deepcopy(money)):
        assert type(copied) is _Money and copied == money and copied.currency == "EUR"
    # A struct pickles its `fields`, `names` and `_needConversion`, and an array its element of
    # its own class.
    names = _Names("a", "b")
    assert names.names == ["a", "b"] and names.add("c", T.LongType()) is names
    assert names.fieldNames() == ["a", "b", "c"]
    for loaded in (pickle.loads(pickle.dumps(names)), copy.deepcopy(names)):
        assert type(loaded) is _Names and loaded == names and loaded.names == ["a", "b", "c"]
    assert copy.copy(names).fields is names.fields
    array = pickle.loads(pickle.dumps(T.ArrayType(_Money("USD"), False)))
    assert type(array.elementType) is _Money and array.elementType.currency == "USD"
    # A singleton type of a subclass is the one instance of its class, which `__new__` gives, so
    # its copies and pickles are that instance too, where PySpark's make another object.
    assert pickle.loads(pickle.dumps(_Tag())) is _Tag() and copy.deepcopy(_Tag()) is _Tag()


class _SlottedArray(T.ArrayType):
    __slots__ = ("tag",)


class _SlottedInteger(T.IntegerType):
    __slots__ = ("tag",)


class _SlottedStruct(T.StructType):
    __slots__ = ("tag",)


class _SlottedField(T.StructField):
    __slots__ = ("tag",)


class _SlottedDecimal(T.DecimalType):
    __slots__ = ("tag",)


class _SlottedGeography(T.GeographyType):
    __slots__ = ("tag",)


class _SlottedOwnState(T.DecimalType):
    __slots__ = ("tag",)

    def __getstate__(self):
        return (vars(self), {"tag": "y"})


class _EmptySlots(T.StringType):
    __slots__ = ()


class _SlottedOwnReduce(T.CharType):
    __slots__ = ("tag",)

    def __reduce__(self):
        return (T.CharType, (3,))


def test_python_subclasses_of_slots_pickle_at_the_protocols_of_pysparks():
    # SPARK-62251: as copyreg's `_reduce_ex` of PySpark's plain classes, a subclass of `__slots__`
    # of no `__getstate__` of its own raises at the protocols 0 and 1, of any type; one of its own
    # `__getstate__` or `__reduce__`, or of empty slots, pickles at every protocol, and from the
    # protocol 2 the slots are of the state. The text is official PySpark 4.3.0's.
    message = "^a class that defines __slots__ without defining __getstate__ cannot be pickled$"
    for make in (
        lambda: _SlottedArray(T.LongType()),
        lambda: _SlottedInteger(),
        lambda: _SlottedStruct([T.StructField("a", T.LongType())]),
        lambda: _SlottedField("a", T.LongType()),
        lambda: _SlottedDecimal(10, 2),
        lambda: _SlottedGeography(4326),
    ):
        for protocol in (0, 1):
            with pytest.raises(TypeError, match=message):
                pickle.dumps(make(), protocol)
        t = make()
        t.tag = "x"
        for protocol in (2, 5):
            loaded = pickle.loads(pickle.dumps(t, protocol))
            assert (type(loaded), loaded.tag, loaded == t) == (type(t), "x", True)
    own, empty, reduced = _SlottedOwnState(10, 2), _EmptySlots(), _SlottedOwnReduce(3)
    for protocol in range(6):
        assert pickle.loads(pickle.dumps(own, protocol)).tag == "y"
        assert pickle.loads(pickle.dumps(empty, protocol)) == empty
        assert type(pickle.loads(pickle.dumps(reduced, protocol))) is T.CharType


def test_the_dicts_of_the_types_of_parameters_are_their_state_as_pysparks():
    # SPARK-62251: the `__dict__` of a type of parameters is its state, as PySpark's: an entry
    # written there is the parameter of its repr, JSON, plan and `==`, also in the struct that
    # holds it, whose JSON writes the `jsonValue()` of a type of values that the core type does
    # not hold, of its own errors, and a type of other entries than those of its constructor
    # compares, copies and pickles of them. The texts are official PySpark 4.3.0's.
    decimal = T.DecimalType(10, 2)
    decimal.__dict__["scale"] = 4
    assert (repr(decimal), decimal.json()) == ("DecimalType(10,4)", '"decimal(10,4)"')
    assert decimal == T.DecimalType(10, 4)
    assert F.col("v").cast(decimal)._expr.to_plan(None).cast.type.decimal.scale == 4
    decimal.__dict__["precision"] = 7.9
    struct = T.StructType([T.StructField("d", decimal)])
    assert struct.json() == (
        '{"fields":[{"metadata":{},"name":"d","nullable":true,"type":"decimal(7,4)"}],'
        '"type":"struct"}'
    )
    assert struct.simpleString() == "struct<d:decimal(7,4)>"
    del decimal.__dict__["precision"]
    with pytest.raises(AttributeError, match="^'DecimalType' object has no attribute 'precision'$"):
        struct.json()
    geometry = T.GeometryType(0)
    geometry.tag = "t"
    assert vars(geometry) == {"srid": 0, "_crs": "SRID:0", "tag": "t"}
    assert geometry != T.GeometryType(0) and T.ArrayType(geometry) != T.ArrayType(T.GeometryType(0))
    for copied in (copy.copy(geometry), copy.deepcopy(geometry), pickle.loads(pickle.dumps(geometry))):
        assert vars(copied) == vars(geometry) and copied == geometry
    array = pickle.loads(pickle.dumps(T.ArrayType(geometry)))
    assert vars(array.elementType) == vars(geometry)


def test_python_subclasses_that_do_not_initialize_their_class_raise_pysparks_errors():
    # PySpark's subclass whose `__init__` does not call the `__init__` of its class has none of
    # its parameters, and raises the AttributeError of each where it reads it; so does the
    # drop-in's type of parameters, array, map and field, which keep them in their `__dict__`
    # (SPARK-62251). Singletons have no parameters.
    class Unset(T.StringType):
        def __init__(self, x):
            self.x = x

    class Array(T.ArrayType):
        def __init__(self):
            self.elementType = T.LongType()

    class Field(T.StructField):
        def __init__(self):
            pass

    assert vars(Unset(1)) == {"x": 1}
    for use in (repr, lambda t: t.collation, lambda t: t.simpleString(), lambda t: t.json()):
        with pytest.raises(AttributeError, match="^'Unset' object has no attribute 'collation'$"):
            use(Unset(1))
    assert Unset(1) == Unset(1) and Unset(1) != Unset(2)
    assert Array() == Array() and vars(Array()) == {"elementType": T.LongType()}
    for use in (repr, lambda t: t.containsNull, lambda t: t.json()):
        with pytest.raises(AttributeError, match="^'Array' object has no attribute 'containsNull'$"):
            use(Array())
    assert Array().simpleString() == "array<bigint>" and not Array().needConversion()
    for use, missing in [
        (repr, "name"),
        (lambda t: t.simpleString(), "name"),
        (lambda t: t.json(), "dataType"),
        (lambda t: t.needConversion(), "dataType"),
        (lambda t: T.StructType([t]), "name"),
    ]:
        with pytest.raises(AttributeError, match=f"^'Field' object has no attribute '{missing}'$"):
            use(Field())

    class Counted(T.IntegerType):
        made = 0

        def __init__(self):
            Counted.made += 1

    # The instance is the class's one, and Python calls its `__init__` at each call, where
    # PySpark's metaclass calls it on the first one.
    assert Counted() is Counted() and Counted.made == 2


def _cast_type_proto(t):
    """The proto of the type `t` of a cast, as a plan sends it."""
    return F.lit(None).cast(t)._expr.to_plan(None).cast.type


def test_arrays_maps_and_fields_keep_their_parameters_in_their_dict_as_pysparks():
    # PySpark's array, map and field are plain classes whose parameters are entries of their
    # `__dict__`, of any value, which their methods read: `vars()` holds them, an assignment of
    # any value is kept and the uses fail later as PySpark's do, and `==` compares the `__dict__`s.
    array = T.ArrayType(T.IntegerType())
    assert vars(array) == {"elementType": T.IntegerType(), "containsNull": True}
    field = T.StructField("a", T.LongType(), False, {"k": 1})
    assert list(vars(field)) == ["name", "dataType", "nullable", "metadata"]
    assert vars(T.MapType(T.StringType(), T.DateType(), 0)) == {
        "keyType": T.StringType(), "valueType": T.DateType(), "valueContainsNull": 0,
    }
    # A value that is not a type, kept as it is: each method reads it, of its own error.
    array.elementType = "x"
    assert repr(array) == "ArrayType(x, True)"
    with pytest.raises(AttributeError, match="^'str' object has no attribute 'simpleString'$"):
        array.simpleString()
    with pytest.raises(AttributeError, match="^'str' object has no attribute 'jsonValue'$"):
        array.json()
    with pytest.raises(AttributeError, match="^'str' object has no attribute 'needConversion'$"):
        T.StructType([T.StructField("f", array)])
    array.elementType = T.LongType()
    assert array == T.ArrayType(T.LongType()) and array.simpleString() == "array<bigint>"
    # A flag of another value: its JSON and repr are of the value, and the proto of a plan of it,
    # of protobuf's bool field, which takes an int and raises of another value.
    flagged = T.ArrayType(T.StringType(), None)
    assert repr(flagged) == "ArrayType(StringType(), None)"
    assert flagged.json() == '{"containsNull":null,"elementType":"string","type":"array"}'
    with pytest.raises(TypeError, match="^'NoneType' object cannot be interpreted as an integer$"):
        _cast_type_proto(flagged)
    assert _cast_type_proto(T.ArrayType(T.StringType(), 0)).array.contains_null is False
    # A name that is not a str: the struct's `names` hold it, `treeString` raises the error of its
    # `replace`, and the proto of a plan the TypeError of protobuf's string field, as PySpark's.
    renamed = T.StructField("a", T.IntegerType())
    renamed.name = 5
    struct = T.StructType([renamed])
    assert struct.names == [5] and struct.simpleString() == "struct<5:int>"
    assert struct.json() == (
        '{"fields":[{"metadata":{},"name":5,"nullable":true,"type":"integer"}],"type":"struct"}'
    )
    with pytest.raises(AttributeError, match="^'int' object has no attribute 'replace'$"):
        struct.treeString()
    with pytest.raises(TypeError, match="^bad argument type for built-in operation$"):
        _cast_type_proto(struct)
    # A raw write of the `__dict__` counts, as PySpark's methods read the attribute.
    field.__dict__["dataType"] = T.ArrayType(T.DateType())
    assert field.simpleString() == "a:array<date>"
    assert repr(field) == "StructField('a', ArrayType(DateType(), True), False)"
    del field.__dict__["nullable"]
    with pytest.raises(AttributeError, match="^'StructField' object has no attribute 'nullable'$"):
        repr(field)


def test_arrays_maps_and_fields_pickle_and_copy_of_their_dict_as_pysparks():
    # They pickle as PySpark's plain classes do, `copyreg.__newobj__` of the class and the
    # `__dict__`, also of values that the core types do not hold, and a copy shares the values.
    udt_like = T.MapType(T.StringType(), T.ArrayType(T.IntegerType(), 1))
    for t in [
        T.ArrayType(T.IntegerType()),
        udt_like,
        T.StructField("a", T.LongType(), "yes", []),
    ]:
        reduced = t.__reduce_ex__(2)
        assert reduced[:3] == (copyreg.__newobj__, (type(t),), vars(t))
        for protocol in range(6):
            loaded = pickle.loads(pickle.dumps(t, protocol))
            assert type(loaded) is type(t) and vars(loaded) == vars(t) and loaded == t
        copied = copy.copy(t)
        assert all(a is b for a, b in zip(vars(copied).values(), vars(t).values()))
    held = T.ArrayType(T.LongType())
    held.tag = [1]
    assert held != T.ArrayType(T.LongType()) and copy.copy(held).tag is held.tag


def test_subclasses_of_arrays_maps_and_fields_read_their_own_attributes_as_pysparks():
    # A subclass reads its parameters of its own attributes, as PySpark's methods read `self.<name>`,
    # also in the types that hold it: a property, and a class attribute that PySpark's `__init__`
    # shadows with the instance's own.
    class LongElement(T.ArrayType):
        @property
        def elementType(self):
            return T.LongType()

        @elementType.setter
        def elementType(self, value):
            self.__dict__["given"] = value

    class NotNullable(T.StructField):
        nullable = False

    array = LongElement(T.StringType())
    assert vars(array) == {"given": T.StringType(), "containsNull": True}
    assert repr(array) == "ArrayType(LongType(), True)" and array.simpleString() == "array<bigint>"
    assert T.StructType([T.StructField("a", array)]).simpleString() == "struct<a:array<bigint>>"
    assert _cast_type_proto(array).array.element_type.HasField("long")
    field = NotNullable("a", T.LongType())
    assert vars(field)["nullable"] is True and repr(field) == "StructField('a', LongType(), True)"
    assert T.StructType([field]).json() == (
        '{"fields":[{"metadata":{},"name":"a","nullable":true,"type":"long"}],"type":"struct"}'
    )


def test_the_lowering_of_fields_reads_each_entry_of_their_dict_as_pysparks():
    # A plan reads `field.name`, `dataType`, `nullable` and `metadata` as Python reads them, of any
    # key of the `__dict__` that equals the name of the attribute: of a key that is another str
    # object, of a key of a str subclass, and of a field of a key that is not a str too.
    class Key(str):
        pass

    field = T.StructField("a", T.LongType())
    struct = T.StructType([field])
    assert _cast_type_proto(struct).struct.fields[0].name == "a"
    # An equal key keeps the key of the dict and replaces its value.
    field.__dict__["".join(["na", "me"])] = "x"
    assert _cast_type_proto(struct).struct.fields[0].name == "x"
    # A key of another object, of the dict anew.
    for key in ("".join(["na", "me"]), Key("name")):
        del field.__dict__["name"]
        field.__dict__[key] = "y" + type(key).__name__
        assert _cast_type_proto(struct).struct.fields[0].name == "y" + type(key).__name__
    del field.__dict__["nullable"]
    field.__dict__["".join(["null", "able"])] = False
    lowered = _cast_type_proto(struct).struct.fields[0]
    assert lowered.name == "yKey" and lowered.nullable is False
    field.__dict__[1] = "not a str"
    del field.__dict__["dataType"]
    field.__dict__[Key("dataType")] = T.StringType()
    assert _cast_type_proto(struct).struct.fields[0].data_type.HasField("string")
    assert struct.simpleString() == "struct<yKey:string>"


def test_the_lowering_of_fields_reads_each_new_name_as_pysparks():
    # A name rebound to new strs many times, whose objects take the addresses of the freed ones,
    # lowers of each, as PySpark's plan reads `field.name`.
    field = T.StructField("a", T.LongType())
    struct = T.StructType([field])
    for i in range(2000):
        field.name = "".join(["n", str(i % 7), "x" * (i % 3)])
        if i % 97 == 0:
            gc.collect()
        assert _cast_type_proto(struct).struct.fields[0].name == field.name


def test_python_subclasses_of_two_branches_of_the_type_classes_raise():
    # PySpark's plain classes make a class of two types of other branches of their hierarchy. The
    # native state of a type is of one class and the classes it extends, so such a class raises
    # the TypeError that CPython raises of bases of conflicting layouts; the classes of one branch
    # make a class, and `__init_subclass__` takes no keyword arguments, as `object`'s.
    for bases in [
        (T.StringType, T.IntegerType),
        (T.ArrayType, T.MapType),
        (T.IntegralType, T.StringType),
    ]:
        with pytest.raises(TypeError) as e:
            type("Both", bases, {})
        assert str(e.value) == "multiple bases have instance lay-out conflict"
    integers = type("Integers", (T.IntegerType, T.NumericType), {})
    assert integers() is integers() and repr(integers()) == "Integers()"

    # A class whose `__init_subclass__` does not call that of DataType is made, and its
    # constructor raises.
    class Quiet(T.StringType):
        def __init_subclass__(cls, **kwargs):
            pass

    for bases in [(T.IntegralType, Quiet), (Quiet, T.IntegerType)]:
        with pytest.raises(TypeError) as e:
            type("Both", bases, {})()
        assert str(e.value) == "multiple bases have instance lay-out conflict"
    with pytest.raises(TypeError) as e:
        type("Keyed", (T.StringType,), {}, key=1)
    assert str(e.value) == "Keyed.__init_subclass__() takes no keyword arguments"


def test_python_subclasses_nested_in_types_run_their_methods_as_pysparks():
    # PySpark's array, map and struct call the methods of the type objects that they hold, and a
    # struct the methods of its fields, so the overrides of a subclass are of their JSON, simple
    # string, repr, tree string and conversions.
    class Commented(T.StructField):
        def jsonValue(self):
            value = super().jsonValue()
            value["metadata"] = {"comment": "c"}
            return value

        def simpleString(self):
            return "commented"

    class Tagged(T.IntegerType):
        @classmethod
        def typeName(cls):
            return "tagged"

    class Plain(T.IntegerType):
        def simpleString(self):
            return "plain"

        def jsonValue(self):
            return "long"

        def __repr__(self):
            return "Plain!"

    class Tens(T.IntegerType):
        def needConversion(self):
            return True

        def toInternal(self, obj):
            return None if obj is None else obj * 10

        def fromInternal(self, obj):
            return None if obj is None else obj // 10

    struct = T.StructType([
        Commented("a", T.LongType()),
        T.StructField("b", Tagged()),
        T.StructField("c", T.ArrayType(Plain())),
    ])
    assert json.loads(struct.json())["fields"] == [
        {"metadata": {"comment": "c"}, "name": "a", "nullable": True, "type": "long"},
        {"metadata": {}, "name": "b", "nullable": True, "type": "tagged"},
        {"metadata": {}, "name": "c", "nullable": True,
         "type": {"containsNull": True, "elementType": "long", "type": "array"}},
    ]
    assert struct.simpleString() == "struct<commented,b:int,c:array<plain>>"
    assert repr(struct) == (
        "StructType([StructField('a', LongType(), True), StructField('b', Tagged(), True), "
        "StructField('c', ArrayType(Plain!, True), True)])"
    )
    assert struct.treeString() == (
        "root\n |-- a: long (nullable = true)\n |-- b: tagged (nullable = true)\n"
        " |-- c: array (nullable = true)\n |    |-- element: p (containsNull = true)\n"
    )
    assert (Tagged().simpleString(), Tagged().json(), hash(Plain())) == ("int", '"tagged"',
                                                                         hash("Plain!"))
    assert T.MapType(Plain(), Tagged()).json() == (
        '{"keyType":"long","type":"map","valueContainsNull":true,"valueType":"tagged"}'
    )
    assert T.ArrayType(Tens()).needConversion() is True
    assert T.ArrayType(Tens()).toInternal([1, 2]) == [10, 20]
    assert T.MapType(T.StringType(), Tens()).toInternal({"x": 1}) == {"x": 10}
    tens = T.StructType([T.StructField("a", Tens()), T.StructField("b", T.ArrayType(Tens()))])
    assert tens._needConversion == [True, True] and tens.toInternal((3, [4])) == (30, [40])
    assert tens.fromInternal((30, [40])) == T.Row(a=3, b=[4])
    # In a field of collations, PySpark's JSON is of the base classes of its strings, arrays
    # and maps (`removeCollations`), and of the other types of subclasses.
    collated = T.StructField("f", T.MapType(T.StringType("UTF8_LCASE"), Tagged()))
    assert collated.json() == (
        '{"metadata":{"__COLLATIONS":{"f.key":"spark.UTF8_LCASE"}},"name":"f","nullable":true,'
        '"type":{"keyType":"string","type":"map","valueContainsNull":true,"valueType":"tagged"}}'
    )
    sub_string = type("SubString", (T.StringType,), {})("UTF8_LCASE")
    assert json.loads(T.StructField("f", T.ArrayType(sub_string)).json())["type"] == {
        "containsNull": True, "elementType": "string", "type": "array"}


def test_python_subclasses_that_override_the_reads_of_json_write_it_as_pysparks():
    # PySpark's `jsonValue` reads the attributes of a type as Python reads them: of a struct, the
    # fields as it iterates itself (`for f in self`, which is `iter(self.fields)`), of a field its
    # `name`, `dataType`, `nullable` and `metadata`, so a property of a subclass that overrides one,
    # and an iteration of its own, are of its JSON, at any depth, as PySpark 4.3.0's.
    class FieldStr(T.StructField):
        @property
        def dataType(self):
            return T.StringType()

        @dataType.setter
        def dataType(self, value):
            self.__dict__["given"] = value

    class FieldNotNull(T.StructField):
        @property
        def nullable(self):
            return False

        @nullable.setter
        def nullable(self, value):
            self.__dict__["given"] = value

    class FieldMeta(T.StructField):
        @property
        def metadata(self):
            return {"own": 1}

        @metadata.setter
        def metadata(self, value):
            self.__dict__["given"] = value

    class StructRevr(T.StructType):
        # Of the `typeName` of `struct`, the class name less its last four letters.
        def __iter__(self):
            return iter(list(self.fields)[::-1])

    long_type = T.LongType()
    struct = T.StructType([FieldStr("a", long_type, True, {"k": 1}), FieldNotNull("b", long_type),
                           FieldMeta("c", long_type)])
    assert json.loads(struct.json())["fields"] == [
        {"metadata": {"k": 1}, "name": "a", "nullable": True, "type": "string"},
        {"metadata": {}, "name": "b", "nullable": False, "type": "long"},
        {"metadata": {"own": 1}, "name": "c", "nullable": True, "type": "long"},
    ]
    assert struct.jsonValue() == json.loads(struct.json())
    assert FieldStr("a", long_type).json() == (
        '{"metadata":{},"name":"a","nullable":true,"type":"string"}')
    assert FieldStr("a", long_type).jsonValue() == {
        "name": "a", "type": "string", "nullable": True, "metadata": {}}
    nested = T.StructType([T.StructField("s", T.ArrayType(T.MapType(
        T.StringType(), T.StructType([FieldStr("a", long_type)]))))])
    assert json.loads(nested.json())["fields"][0]["type"] == {
        "containsNull": True, "type": "array", "elementType": {
            "keyType": "string", "type": "map", "valueContainsNull": True, "valueType": {
                "fields": [{"metadata": {}, "name": "a", "nullable": True, "type": "string"}],
                "type": "struct"}}}
    reversed_struct = StructRevr([T.StructField("a", long_type), T.StructField("b", T.StringType())])
    assert reversed_struct.json() == (
        '{"fields":[{"metadata":{},"name":"b","nullable":true,"type":"string"},'
        '{"metadata":{},"name":"a","nullable":true,"type":"long"}],"type":"struct"}')

    # PySpark's `__init__` assigns the fields through a property of the class, which keeps them of
    # its setter, and the native `__init__` does not (SPARK-62231), so the property reads none, or
    # raises the AttributeError of the attribute that the setter sets. The JSON of such a struct,
    # whose fields read otherwise than PySpark's `__init__` would have assigned them, is of the
    # fields that the native `__init__` was given, as PySpark's is.
    class StructFlds(T.StructType):
        @property
        def fields(self):
            return list(self.__dict__.get("_f", []))

        @fields.setter
        def fields(self, value):
            self.__dict__["_f"] = value

    class StructPrivate(T.StructType):
        @property
        def fields(self):
            return self._fields

        @fields.setter
        def fields(self, value):
            self._fields = value

    for cls, type_name in ((StructFlds, "struct"), (StructPrivate, "structpri")):
        assert cls([FieldStr("a", long_type)]).json() == (
            '{"fields":[{"metadata":{},"name":"a","nullable":true,"type":"string"}],'
            f'"type":"{type_name}"}}')


def test_patched_native_type_classes_are_of_their_json_as_pysparks(monkeypatch):
    # As PySpark's `jsonValue`, `simpleString` and `__repr__`, which call the methods of the
    # classes: of `StringType.isUTF8BinaryCollation` patched to return False, a string is written
    # of its collation, which a field holds in its metadata, and of a patched `__iter__` of the
    # struct, its fields are written in that order.
    monkeypatch.setattr(T.StringType, "isUTF8BinaryCollation", lambda self: False)
    assert T.StructType([T.StructField("a", T.StringType())]).json() == (
        '{"fields":[{"metadata":{"__COLLATIONS":{"a":"spark.UTF8_BINARY"}},"name":"a",'
        '"nullable":true,"type":"string collate UTF8_BINARY"}],"type":"struct"}')
    string = T.StringType()
    assert (repr(string), string.simpleString(), string.json()) == (
        "StringType('UTF8_BINARY')", "string collate UTF8_BINARY", '"string collate UTF8_BINARY"')
    assert T.ArrayType(T.StringType()).json() == (
        '{"containsNull":true,"elementType":"string collate UTF8_BINARY","type":"array"}')
    monkeypatch.undo()
    assert T.StructType([T.StructField("a", T.StringType())]).json() == (
        '{"fields":[{"metadata":{},"name":"a","nullable":true,"type":"string"}],"type":"struct"}')
    monkeypatch.setattr(T.StructType, "__iter__", lambda self: iter(list(self.fields)[::-1]))
    struct = T.StructType([T.StructField("a", T.LongType()), T.StructField("b", T.StringType())])
    assert struct.json() == (
        '{"fields":[{"metadata":{},"name":"b","nullable":true,"type":"string"},'
        '{"metadata":{},"name":"a","nullable":true,"type":"long"}],"type":"struct"}')


@pytest.mark.parametrize(
    "a, b, equal",
    [
        ({"k": 1}, {"k": 1.0}, True),
        ({"k": True}, {"k": 1}, True),
        ({"k": 2**64}, {"k": float(2**64)}, True),
        ({"a": 1, "b": 2}, {"b": 2, "a": 1}, True),
        ({"k": (1, 2)}, {"k": [1, 2]}, False),
        ({1: "a"}, {"1": "a"}, False),
        ({"k": 10**30}, {"k": 10**30 + 1}, False),
        ({"k": 10**30}, {"k": 1e30}, False),
        ({"k": -2**63 - 1}, {"k": float(-2**63 - 1)}, False),
        ({"k": {"x": [1]}}, {"k": {"x": (1,)}}, False),
    ],
    ids=["int-float", "bool-int", "2**64-float", "order", "tuple-list", "int-key", "big-int",
         "big-int-float", "below-i64", "nested-tuple"],
)
def test_field_metadata_compares_as_python_compares_the_dicts_as_pysparks(a, b, equal):
    # PySpark's fields compare their `__dict__`s, so their metadata dicts compare as Python
    # compares them, also in a struct and a nested type.
    fa, fb = T.StructField("a", T.LongType(), True, a), T.StructField("a", T.LongType(), True, b)
    assert (fa == fb, fa != fb) == (equal, not equal)
    assert (T.StructType([fa]) == T.StructType([fb])) is equal
    assert (T.ArrayType(T.StructType([fa])) == T.ArrayType(T.StructType([fb]))) is equal


def test_field_metadata_is_written_of_its_dict_as_pysparks():
    # PySpark's `jsonValue` holds the metadata dict of the field, which `json()` writes with
    # `json.dumps`: an int of any size as it is, and a float as Python writes it.
    for metadata, text in [
        ({"k": 10**30}, '{"k":1000000000000000000000000000000}'),
        ({"k": -2**63 - 1}, '{"k":-9223372036854775809}'),
        ({"k": [1e16, {"x": 1e-7}]}, '{"k":[1e+16,{"x":1e-07}]}'),
        ({"k": (1, 2)}, '{"k":[1,2]}'),
        ({1: "a"}, '{"1":"a"}'),
    ]:
        field = T.StructField("a", T.LongType(), True, metadata)
        assert field.jsonValue()["metadata"] is field.metadata
        sent = f'{{"metadata":{text},"name":"a","nullable":true,"type":"long"}}'
        assert field.json() == sent
        assert T.StructType([field]).json() == f'{{"fields":[{sent}],"type":"struct"}}'
        assert T.ArrayType(T.StructType([field])).json() == (
            f'{{"containsNull":true,"elementType":{{"fields":[{sent}],"type":"struct"}},'
            f'"type":"array"}}'
        )
    # A pickle of the type keeps the ints, of the dict.
    big = T.StructType([T.StructField("a", T.LongType(), True, {"k": 10**30})])
    assert pickle.loads(pickle.dumps(big)).fields[0].metadata == {"k": 10**30}


@pytest.mark.parametrize(
    "metadata",
    [{"k": (1, 2)}, {1: "a"}, {"k": 10**30}, {"b": 1, "a": 2}, {"k": {"n": (1,)}}],
    ids=["tuple", "int-key", "big-int", "key-order", "nested-tuple"],
)
def test_copies_pickles_and_nullable_types_keep_the_metadata_dicts_as_pysparks(metadata):
    # PySpark's types copy, deep-copy and pickle their `__dict__`, of the metadata dicts of their
    # fields as they are, which JSON does not keep (a tuple, a key that is not a str, the order of
    # the keys), and `toNullable` makes each field of `metadata=field.metadata`, the same dict. So
    # do the drop-in's, also of the types that hold such a struct: they pickle of the types they
    # are made of, which the Python workers of the server load as PySpark's `StructType(fields)`,
    # and a type of no metadata of its JSON.
    decimal = T.DecimalType(10, 2)
    field = T.StructField("a", decimal, False, metadata)
    struct = T.StructType([field, T.StructField("b", T.ArrayType(T.LongType(), False), False)])
    for t, inner in [
        (struct, lambda c: c),
        (T.ArrayType(struct, False), lambda c: c.elementType),
        (T.MapType(T.StringType(), struct, False), lambda c: c.valueType),
        (T.StructField("s", struct, False), lambda c: c.dataType),
        (T.StructType([T.StructField("s", struct, False)]), lambda c: c.fields[0].dataType),
    ]:
        copies = [copy.copy(t), copy.deepcopy(t)]
        copies += [pickle.loads(pickle.dumps(t, protocol)) for protocol in range(6)]
        for copied in copies:
            assert type(copied) is type(t) and copied == t, repr(t)
            copied_metadata = inner(copied).fields[0].metadata
            assert list(copied_metadata.items()) == list(metadata.items())
            assert repr(copied_metadata) == repr(metadata)
    assert struct.__reduce__()[0] is T.StructType
    plain = T.StructType([T.StructField("a", T.LongType())])
    assert plain.__reduce__()[0] is T._parse_datatype_json_string
    nullable = struct.toNullable()
    assert nullable == T.StructType([
        T.StructField("a", decimal, True, metadata),
        T.StructField("b", T.ArrayType(T.LongType(), True), True),
    ])
    assert nullable.fields[0].metadata is metadata and nullable.fields[0].dataType is decimal
    assert T.ArrayType(struct).toNullable().elementType.fields[0].metadata is metadata
    assert T.MapType(T.StringType(), struct).toNullable().valueType.fields[0].metadata is metadata


@pytest.mark.parametrize(
    "metadata",
    [
        {"k": 10**30},
        {"k": -(10**25), "u": 2**64 - 1},
        {"k": [1, 2**70], "f": 1e-07, "g": 1e16, "z": -0.0},
        {"b": 1, "a": {"d": 2, "c": [1.5, 10**20]}},
        {"k": "\xe9\u4e2d\U0001f600\x7f"},
        {"t": (1, 2)},
        {1: "a", "1": "b"},
        {"x": float("inf"), "y": float("-inf"), "n": float("nan")},
    ],
    ids=["big-int", "big-ints", "array-floats", "key-order", "non-ascii", "tuple", "int-keys",
         "non-finite"],
)
def test_field_metadata_is_sent_as_pysparks_json_dumps_of_its_dict(metadata):
    # PySpark's `pyspark_types_to_proto_types` sends the metadata of a field as `json.dumps` of its
    # dict: of its keys in their order, of the escapes of `ensure_ascii`, of an int of any size and
    # of NaN and Infinity, also in a struct of an array, a map or a struct. The core held JSON
    # values, which sent an int of more than 64 bits as a float, and refused a float that is not
    # finite when the field was made. The texts are official PySpark 4.3.0's.
    field = T.StructField("a", T.LongType(), True, metadata)
    struct = T.StructType([field])
    for data_type, fields in [
        (struct, lambda t: t.struct.fields),
        (T.ArrayType(struct), lambda t: t.array.element_type.struct.fields),
        (T.MapType(T.StringType(), struct), lambda t: t.map.value_type.struct.fields),
        (T.StructType([T.StructField("s", struct)]),
         lambda t: t.struct.fields[0].data_type.struct.fields),
    ]:
        for cast in (F.col("v").cast, F.col("v").try_cast):
            sent = fields(cast(data_type)._expr.to_plan(None).cast.type)[0]
            assert sent.metadata == json.dumps(metadata), repr(data_type)
    # Its JSON is `json.dumps` of the `jsonValue()` of the type, of the sorted keys, which keys of
    # an int and a str do not sort.
    try:
        text = json.dumps(struct.jsonValue(), separators=(",", ":"), sort_keys=True)
    except TypeError as error:
        with pytest.raises(TypeError, match=f"^{re.escape(str(error))}$"):
            struct.json()
        return
    assert struct.json() == text and text.isascii()
    assert T.ArrayType(struct).json() == (
        f'{{"containsNull":true,"elementType":{text},"type":"array"}}')


def test_field_metadata_of_big_ints_and_non_finite_floats_is_pysparks():
    # The texts that official PySpark 4.3.0 sends of these, which the server reads: the low 64
    # bits of the int, and a JSON error of a float that is not finite.
    big = T.StructField("a", T.LongType(), True, {"x": 10**30, "b": [1, 2**70]})
    inf = T.StructField("a", T.LongType(), True, {"x": float("inf"), "n": float("nan")})
    for field, sent, written in [
        (big, '{"x": 1000000000000000000000000000000, "b": [1, 1180591620717411303424]}',
         '{"b":[1,1180591620717411303424],"x":1000000000000000000000000000000}'),
        (inf, '{"x": Infinity, "n": NaN}', '{"n":NaN,"x":Infinity}'),
    ]:
        plan = F.col("v").cast(T.StructType([field]))._expr.to_plan(None)
        assert plan.cast.type.struct.fields[0].metadata == sent
        assert field.json() == (
            f'{{"metadata":{written},"name":"a","nullable":true,"type":"long"}}')
    assert inf.metadata["x"] == float("inf") and inf == inf


def test_fields_of_metadata_that_json_dumps_refuses_are_made_as_pysparks():
    # PySpark's StructField keeps any metadata dict, and only the uses that write its JSON raise
    # the error of `json.dumps` of it: `json()`, of the sorted keys of `json.dumps` of the
    # `jsonValue()` too, and the plans of the type, of `json.dumps(field.metadata)`, when a request
    # builds them. The drop-in raised the error of `json.dumps` when the field was made. The errors
    # are official PySpark 4.3.0's.
    circular = {}
    circular["self"] = circular
    set_error = (TypeError, "Object of type set is not JSON serializable")
    for metadata, json_error, plan_error in [
        ({"x": {1}}, set_error, set_error),
        ({(1,): 1, "b": 2}, (TypeError, "'<' not supported between instances of 'str' and 'tuple'"),
         (TypeError, "keys must be str, int, float, bool or None, not tuple")),
        ({1: "a", "b": 2}, (TypeError, "'<' not supported between instances of 'str' and 'int'"),
         None),
        (circular, (ValueError, "Circular reference detected"),
         (ValueError, "Circular reference detected")),
    ]:
        field = T.StructField("a", T.IntegerType(), True, metadata)
        struct = T.StructType([field, T.StructField("b", T.LongType())])
        plain = T.StructType([T.StructField("a", T.IntegerType()), T.StructField("b", T.LongType())])
        assert field.metadata is metadata and repr(struct) == repr(plain)
        assert struct == T.StructType([T.StructField("a", T.IntegerType(), True, metadata),
                                       T.StructField("b", T.LongType())])
        assert struct != plain and hash(struct) == hash(plain)
        assert struct.simpleString() == "struct<a:int,b:bigint>"
        assert struct.treeString() == plain.treeString()
        assert struct.jsonValue()["fields"][0]["metadata"] is metadata
        assert field.jsonValue()["metadata"] is metadata
        assert struct.toNullable().fields[0].metadata is metadata
        assert struct.fromInternal((1, 2)) == T.Row(a=1, b=2)
        assert copy.copy(struct).fields[0].metadata is metadata
        if metadata is not circular:
            assert pickle.loads(pickle.dumps(struct)).fields[0].metadata == metadata
        added = T.StructType().add("a", T.IntegerType(), True, metadata)
        assert added.fields[0].metadata is metadata
        settable = T.StructField("z", T.IntegerType())
        settable.metadata = metadata
        assert settable.metadata is metadata
        for typed in (struct, field, added, settable, T.ArrayType(struct),
                      T.MapType(T.StringType(), struct), T.StructField("s", struct)):
            with pytest.raises(json_error[0]) as error:
                typed.json()
            assert str(error.value) == json_error[1], repr(typed)
        # A cast of the type is made, and shows its simpleString, as PySpark's Column.
        column = F.col("v").cast(struct)
        assert repr(column) == "Column<'CAST(v AS STRUCT<A:INT,B:BIGINT>)'>"
        if plan_error is None:
            column._expr.to_plan(None)
            continue
        with pytest.raises(plan_error[0]) as error:
            column._expr.to_plan(None)
        assert str(error.value) == plan_error[1]
    # Each request of such a cast raises the error, before it sends anything.
    with mock_connect_session() as (spark, service):
        column = F.col("id").cast(T.StructType([T.StructField("a", T.LongType(), True, {"x": {1}})]))
        df = spark.range(1).select(column)
        for request in (lambda: df.schema, df.collect):
            with pytest.raises(TypeError) as error:
                request()
            assert str(error.value) == set_error[1]
        assert service.analyze_requests == [] and service.execute_requests == []


@pytest.mark.parametrize(
    "metadata, sent",
    [
        ([1, 2], "[1, 2]"),
        ("abc", '"abc"'),
        ((1, 2), "[1, 2]"),
        ([{"a": 1}, [2]], '[{"a": 1}, [2]]'),
        ('{"k": 1}', '"{\\"k\\": 1}"'),
        (5, "object of type 'int' has no len()"),
        (object(), "object of type 'object' has no len()"),
    ],
    ids=["list", "str", "tuple", "nested", "str-of-json", "int", "object"],
)
def test_fields_keep_metadata_that_is_not_a_dict_as_pysparks(metadata, sent):
    # SPARK-62230: PySpark's StructField keeps any metadata (`metadata or {}`), of which its
    # `jsonValue()`, `json()`, copies, pickles and `toNullable()` keep the object, and its plans send
    # `json.dumps` of it when it has a length (`len(field.metadata) > 0`), which the server refuses
    # (`ClassCastException`), or raise the `TypeError` of its length. The drop-in refused metadata
    # that is not a dict when the field was made. The texts are official PySpark 4.3.0's.
    field = T.StructField("a", T.LongType(), False, metadata)
    struct = T.StructType([field])
    assert field.metadata is metadata
    assert field == T.StructField("a", T.LongType(), False, metadata)
    assert field != T.StructField("a", T.LongType(), False)
    assert repr(field) == "StructField('a', LongType(), False)"
    assert field.jsonValue()["metadata"] is metadata
    assert struct.jsonValue()["fields"][0]["metadata"] is metadata
    assert struct.toNullable().fields[0].metadata is metadata
    assert copy.copy(field).metadata is metadata
    assert T.StructType().add("a", T.LongType(), True, metadata).fields[0].metadata is metadata
    settable = T.StructField("z", T.LongType())
    settable.metadata = metadata
    assert settable.metadata is metadata
    if type(metadata) is not object:
        written = json.dumps(struct.jsonValue(), separators=(",", ":"), sort_keys=True)
        assert struct.json() == written
        assert pickle.loads(pickle.dumps(struct)).fields[0].metadata == metadata
        assert copy.deepcopy(field).metadata == metadata
        # As PySpark's `fromJson`, which looks for the collations in the metadata.
        if isinstance(metadata, int):
            with pytest.raises(TypeError, match="^argument of type 'int' is not iterable$"):
                T.StructType.fromJson(json.loads(written))
        else:
            parsed = T.StructType.fromJson(json.loads(written))
            assert parsed.fields[0].metadata == json.loads(json.dumps(metadata))
    column = F.col("v").cast(struct)
    assert repr(column) == "Column<'CAST(v AS STRUCT<A:BIGINT>)'>"
    if sent.startswith("object of type"):
        with pytest.raises(TypeError) as error:
            column._expr.to_plan(None)
        assert str(error.value) == sent
    else:
        assert column._expr.to_plan(None).cast.type.struct.fields[0].metadata == sent


def test_fields_of_false_metadata_make_a_dict_and_keep_one_that_is_set_as_pysparks():
    # As PySpark's `metadata or {}`: a field of false metadata, of no length, has a new empty dict,
    # and one set after it is kept as it is, and is not sent; a falsy value set keeps the
    # `TypeError` of PySpark's unpacking of the collations of a string of the field in its JSON.
    for metadata in ([], "", 0, None, {}):
        field = T.StructField("a", T.LongType(), True, metadata)
        assert field.metadata == {} and field.metadata is not metadata
    field = T.StructField("a", T.LongType())
    field.metadata = []
    assert field.metadata == [] and pickle.loads(pickle.dumps(field)).metadata == []
    assert copy.deepcopy(field).metadata == [] and field.jsonValue()["metadata"] == []
    plan = F.col("v").cast(T.StructType([field]))._expr.to_plan(None)
    assert not plan.cast.type.struct.fields[0].HasField("metadata")
    collated = T.StructField("s", T.StringType("UTF8_LCASE"), True, [1, 2])
    with pytest.raises(TypeError, match="^'list' object is not a mapping$"):
        collated.jsonValue()
    with pytest.raises(TypeError, match="^'str' object is not a mapping$"):
        T.StructType([T.StructField("s", T.ArrayType(T.StringType("UTF8_LCASE")), True, "ab")]).json()


def _metadata_udt(metadata):
    """A UserDefinedType whose SQL type is a struct of a field of `metadata`."""

    class MetadataUDT(T.UserDefinedType):
        @classmethod
        def sqlType(cls):
            return T.StructType([T.StructField("x", T.DoubleType(), False, metadata)])

        @classmethod
        def module(cls):
            return "pyspark.sql.types"

        def serialize(self, obj):
            return (obj,)

        def deserialize(self, datum):
            return datum[0]

    return MetadataUDT()


def test_the_metadata_of_the_sql_type_of_a_user_defined_type_is_written_as_pysparks():
    # PySpark's `jsonValue()` of a UserDefinedType holds the `jsonValue()` of its `sqlType()`, of
    # the metadata dicts of its fields as they are, which `json()` writes with `json.dumps`: of the
    # keys sorted as Python sorts them, ints by their values, and of its errors. The drop-in wrote
    # the JSON of the core type of the UDT: the keys as strs, sorted as strs, `{}` of a dict that
    # `json.dumps` refuses, and no error of keys of an int and a str. The values are official
    # PySpark 4.3.0's.
    for metadata, written in [
        ({10: "a", 9: "b"}, '{"9":"b","10":"a"}'),
        ({"t": 74334700308107.625}, '{"t":74334700308107.62}'),
    ]:
        struct = T.StructType([T.StructField("u", _metadata_udt(metadata), True)])
        sql_type = struct.jsonValue()["fields"][0]["type"]["sqlType"]
        assert sql_type["fields"][0]["metadata"] is metadata
        assert f'"metadata":{written}' in struct.json()
        assert f'"metadata":{written}' in T.ArrayType(_metadata_udt(metadata)).json()
    for metadata, error in [
        ({1: "a", "b": 2}, "'<' not supported between instances of 'str' and 'int'"),
        ({"x": {1}}, "Object of type set is not JSON serializable"),
    ]:
        struct = T.StructType([T.StructField("u", _metadata_udt(metadata), True)])
        sql_type = struct.jsonValue()["fields"][0]["type"]["sqlType"]
        assert sql_type["fields"][0]["metadata"] is metadata
        with pytest.raises(TypeError, match=f"^{re.escape(error)}$"):
            struct.json()


def test_floats_are_written_of_the_shortest_digits_of_an_even_tie_as_pysparks():
    # As Python's `repr`, of two shortest digits as near the value, the even last one, in the
    # metadata that a plan sends and in the repr of a literal Column. The drop-in rounded the tie
    # up (`74334700308107.63`). The texts are official PySpark 4.3.0's.
    for value, text in [
        (74334700308107.625, "74334700308107.62"),
        (810452564081.40625, "810452564081.4062"),
        (-2226063173806980.25, "-2226063173806980.2"),
    ]:
        assert repr(F.lit(value)) == f"Column<'{text}'>"
        field = T.StructField("a", T.LongType(), True, {"t": value})
        plan = F.col("v").cast(T.StructType([field]))._expr.to_plan(None)
        assert plan.cast.type.struct.fields[0].metadata == f'{{"t": {text}}}'


def test_a_lowering_lowers_the_types_of_the_code_that_it_calls_as_they_are_elsewhere():
    # The JSON value of a type takes the metadata of its fields as they were given, but the code
    # that it calls, such as the `sqlType()` of a UserDefinedType, lowers its own types as it does
    # elsewhere: a cast made there to a struct of metadata that `json.dumps` refuses raises the
    # error at each request of it, as PySpark's, and nothing is sent. The drop-in's thread-wide
    # mode cast it to the struct of no metadata, which it sent.
    armed, captured = [], []
    refused = T.StructType([T.StructField("z", T.IntegerType(), True, {"bad": {1}})])

    class CapturingUDT(T.UserDefinedType):
        @classmethod
        def sqlType(cls):
            # The cast is made in the `sqlType()` that `jsonValue()` calls only.
            if armed and not captured:
                captured.append(F.lit(None).cast(refused))
            return T.DoubleType()

        @classmethod
        def module(cls):
            return "pyspark.sql.types"

    schema = T.StructType([T.StructField("u", CapturingUDT(), True, {"m": 1})])
    armed.append(True)
    schema.jsonValue()
    assert len(captured) == 1
    with mock_connect_session() as (spark, service):
        df = spark.range(1).select(captured[0].alias("c"))
        for request in (lambda: df.schema, df.collect, lambda: df.select("c").schema):
            with pytest.raises(TypeError, match="^Object of type set is not JSON serializable$"):
                request()
        assert service.analyze_requests == [] and service.execute_requests == []


def test_metadata_of_many_keys_compares_in_linear_time():
    # Each key of the metadata of a field is compared once, by a hash: a comparison of the keys of
    # one with each of the other took a second for 20,000 keys, and would take minutes for these.
    import time

    metadata = {f"key{i}": i for i in range(100_000)}
    reordered = dict(reversed(list(metadata.items())))
    a, b = T.StructField("a", T.IntegerType(), True, metadata), T.StructField(
        "a", T.IntegerType(), True, reordered)
    started = time.perf_counter()
    assert a == b and T.StructType([a]) == T.StructType([b])
    assert a != T.StructField("a", T.IntegerType(), True, {**metadata, "key0": -1})
    assert time.perf_counter() - started < 20


def test_json_escapes_the_characters_that_are_not_printable_ascii_as_pysparks():
    # As PySpark's `json()`, of the `ensure_ascii` of `json.dumps`: a character that is not
    # printable ASCII, also in a name, is a `\uXXXX` escape, of a surrogate pair above U+FFFF,
    # where the core wrote UTF-8.
    name = "\xe9\x7f\u4e2d\U0001f600"
    for data_type in (
        T.StructType([T.StructField(name, T.IntegerType())]),
        T.StructType([T.StructField("a", T.StringType(), True, {name: name})]),
        T.ArrayType(T.StructType([T.StructField(name, T.IntegerType())])),
    ):
        text = data_type.json()
        assert text.isascii() and json.loads(text) == data_type.jsonValue()
        assert text == json.dumps(data_type.jsonValue(), separators=(",", ":"), sort_keys=True)
    field = T.StructField(name, T.IntegerType())
    assert field.json() == json.dumps(field.jsonValue(), separators=(",", ":"), sort_keys=True)


def test_types_of_required_parameters_are_made_as_pysparks():
    # As PySpark's plain classes, whose `__new__` is `object.__new__`, of any arguments, and whose
    # `__init__` takes the parameters: a missing one is the TypeError of `__init__`, a type of
    # `cls.__new__(cls)` is made, as pickle makes it, `__init__` makes a type again, and
    # `inspect.signature` of a class shows the parameters of `__init__`.
    for make, message in [
        (T.ArrayType, "ArrayType.__init__() missing 1 required positional argument: 'elementType'"),
        (T.MapType, "MapType.__init__() missing 2 required positional arguments: 'keyType' and "
                    "'valueType'"),
        (lambda: T.MapType(T.IntegerType()),
         "MapType.__init__() missing 1 required positional argument: 'valueType'"),
        (T.StructField, "StructField.__init__() missing 2 required positional arguments: 'name' "
                        "and 'dataType'"),
        (T.CharType, "CharType.__init__() missing 1 required positional argument: 'length'"),
        (T.VarcharType, "VarcharType.__init__() missing 1 required positional argument: 'length'"),
        (T.GeometryType, "GeometryType.__init__() missing 1 required positional argument: 'srid'"),
        (T.GeographyType,
         "GeographyType.__init__() missing 1 required positional argument: 'srid'"),
        (lambda: T.ArrayType(T.IntegerType(), foo=1),
         "ArrayType.__init__() got an unexpected keyword argument 'foo'"),
    ]:
        with pytest.raises(TypeError) as e:
            make()
        assert str(e.value) == message
    empty = inspect.Parameter.empty
    for cls, parameters in [
        (T.ArrayType, [("elementType", empty), ("containsNull", True)]),
        (T.MapType, [("keyType", empty), ("valueType", empty), ("valueContainsNull", True)]),
        (T.StructField,
         [("name", empty), ("dataType", empty), ("nullable", True), ("metadata", None)]),
        (T.CharType, [("length", empty)]),
        (T.VarcharType, [("length", empty)]),
        (T.GeometryType, [("srid", empty)]),
        (T.GeographyType, [("srid", empty)]),
        # And so are the classes of parameters of defaults, whose `__init__` sets them.
        (T.StringType, [("collation", "UTF8_BINARY")]),
        (T.DecimalType, [("precision", 10), ("scale", 0)]),
        (T.TimeType, [("precision", 6)]),
        (T.DayTimeIntervalType, [("startField", None), ("endField", None)]),
        (T.YearMonthIntervalType, [("startField", None), ("endField", None)]),
        (T.StructType, [("fields", None)]),
    ]:
        assert type(cls.__new__(cls)) is cls
        signature = inspect.signature(cls).parameters.values()
        assert [(p.name, p.default) for p in signature] == parameters
    array, field = T.ArrayType(T.IntegerType()), T.StructField("a", T.IntegerType())
    array.__init__(T.LongType(), False)
    field.__init__("b", T.CharType(3), False, {"k": "v"})
    assert array == T.ArrayType(T.LongType(), False)
    assert field == T.StructField("b", T.CharType(3), False, {"k": "v"})
    string, decimal, struct = T.StringType(), T.DecimalType(), T.StructType()
    string.__init__("UTF8_LCASE")
    decimal.__init__(5, scale=2)
    struct.__init__([field])
    assert (string, decimal) == (T.StringType("UTF8_LCASE"), T.DecimalType(5, 2))
    assert struct == T.StructType([field]) and struct.names == ["b"]
    for make, message in [
        (lambda: T.DecimalType(foo=1), "DecimalType.__init__() got an unexpected keyword argument "
                                       "'foo'"),
        (lambda: T.StructType(fields=1, names=2), "StructType.__init__() got an unexpected "
                                                  "keyword argument 'names'"),
    ]:
        with pytest.raises(TypeError) as e:
            make()
        assert str(e.value) == message


def test_the_parameters_of_the_methods_of_the_types_are_pysparks():
    # The names and the defaults of the parameters of PySpark's methods, without its annotations,
    # which the keyword arguments of those names take. The class of a type without parameters
    # shows none, as PySpark's `DataTypeSingleton.__call__(cls)` and `object.__new__` do.
    empty = inspect.Parameter.empty

    def parameters(method):
        signature = inspect.signature(method).parameters.values()
        return [(p.name, p.default) for p in signature if p.name not in ("self", "cls")]

    for method, expected in [
        (T.StringType.collationProvider, [("collationName", empty)]),
        (T.DateType.toInternal, [("d", empty)]),
        (T.DateType.fromInternal, [("v", empty)]),
        (T.TimeType.toInternal, [("t", empty)]),
        (T.TimeType.fromInternal, [("nano", empty)]),
        (T.TimestampType.toInternal, [("dt", empty)]),
        (T.TimestampType.fromInternal, [("ts", empty)]),
        (T.TimestampNTZType.toInternal, [("dt", empty)]),
        (T.TimestampNTZType.fromInternal, [("ts", empty)]),
        (T.DayTimeIntervalType.toInternal, [("dt", empty)]),
        (T.DayTimeIntervalType.fromInternal, [("micros", empty)]),
        (T.GeographyType.toInternal, [("geography", empty)]),
        (T.GeographyType.fromInternal, [("obj", empty)]),
        (T.GeometryType.toInternal, [("geometry", empty)]),
        (T.GeometryType.fromInternal, [("obj", empty)]),
        (T.DataType.toInternal, [("obj", empty)]),
        (T.DataType.fromInternal, [("obj", empty)]),
        (T.ArrayType.fromJson, [("json", empty), ("fieldPath", ""), ("collationsMap", None)]),
        (T.MapType.fromJson, [("json", empty), ("fieldPath", ""), ("collationsMap", None)]),
        (T.StructType.add,
         [("field", empty), ("data_type", None), ("nullable", True), ("metadata", None)]),
        (T.StringType, [("collation", "UTF8_BINARY")]),
        (T.NullType, []),
        (T.IntegerType, []),
        (T.CalendarIntervalType, []),
        (T.VariantType, []),
        (T.UserDefinedType, []),
    ]:
        assert parameters(method) == expected, method
    assert T.StringType.collationProvider(collationName="UNICODE") == "icu"
    assert T.DateType().toInternal(d=datetime.date(1970, 1, 2)) == 1
    assert T.DateType().fromInternal(v=1) == datetime.date(1970, 1, 2)
    assert T.TimeType().toInternal(t=datetime.time(0, 0, 1)) == 1_000_000_000
    assert T.TimeType().fromInternal(nano=1_000_000_000) == datetime.time(0, 0, 1)
    assert T.TimestampNTZType().toInternal(dt=datetime.datetime(1970, 1, 1, 0, 0, 1)) == 1_000_000
    assert T.TimestampNTZType().fromInternal(ts=1_000_000) == datetime.datetime(1970, 1, 1, 0, 0, 1)
    assert T.DayTimeIntervalType().toInternal(dt=datetime.timedelta(seconds=1)) == 1_000_000
    assert T.DayTimeIntervalType().fromInternal(micros=1_000_000) == datetime.timedelta(seconds=1)
    assert T.GeometryType(0).toInternal(geometry=None) is None
    assert T.GeographyType(4326).toInternal(geography=None) is None


def test_a_types_hash_is_the_hash_of_its_repr_as_pysparks():
    # PySpark's `hash(str(self))`: of the flag as it was given, so equal types of a flag of 1
    # and of True hash apart.
    assert T.ArrayType(T.IntegerType(), 1) == T.ArrayType(T.IntegerType(), True)
    assert hash(T.ArrayType(T.IntegerType(), 1)) != hash(T.ArrayType(T.IntegerType(), True))
    for t in [
        T.IntegerType(),
        T.DecimalType(10, 2),
        T.MapType(T.StringType(), T.LongType(), False),
        T.StructType([T.StructField("a", T.StringType())]),
    ]:
        assert hash(t) == hash(str(t))


def _origins(message, path="plan"):
    """The python origins of the expressions of `message`, by their paths, as the probes of the
    review compared them with official PySpark's."""
    from google.protobuf.descriptor import FieldDescriptor

    out = []
    fields = [field.name for field in message.DESCRIPTOR.fields]
    if (
        "common" in fields
        and message.HasField("common")
        and message.common.HasField("origin")
        and message.common.origin.HasField("python_origin")
    ):
        out.append((path, message.common.origin.python_origin.fragment))
    for field, value in message.ListFields():
        if field.type != FieldDescriptor.TYPE_MESSAGE or field.name == "common":
            continue
        values = value if field.is_repeated else [value]
        for index, item in enumerate(values):
            label = field.name + (f"[{index}]" if field.is_repeated else "")
            if item.DESCRIPTOR.name == "UnresolvedFunction":
                label += f"({item.function_name})"
            out.extend(_origins(item, f"{path}.{label}"))
    return out


# The origins that official PySpark sends of the projection of each expression.
_E = "plan.root.project.expressions[0]"
ORIGINS = {
    # A CASE WHEN is a new `when` of no origin, made when the plan is built; its branches keep theirs.
    "when_otherwise": (
        lambda df: F.when(df.a == 1, 1).otherwise(0),
        [
            (f"{_E}.unresolved_function(when).arguments[0]", "__eq__"),
            (f"{_E}.unresolved_function(when).arguments[0].unresolved_function(==).arguments[1]", "__eq__"),
            (f"{_E}.unresolved_function(when).arguments[2]", "otherwise"),
        ],
    ),
    # `(self >= lowerBound) & (self <= upperBound)`, of the dunders' origins.
    "between": (
        lambda df: df.a.between(1, 2),
        [
            (_E, "__and__"),
            (f"{_E}.unresolved_function(and).arguments[0]", "__ge__"),
            (f"{_E}.unresolved_function(and).arguments[0].unresolved_function(>=).arguments[1]", "__ge__"),
            (f"{_E}.unresolved_function(and).arguments[1]", "__le__"),
            (f"{_E}.unresolved_function(and).arguments[1].unresolved_function(<=).arguments[1]", "__le__"),
        ],
    ),
    # `self[key]`.
    "getItem": (
        lambda df: df.arr.getItem(0),
        [(_E, "__getitem__"), (f"{_E}.unresolved_extract_value.extraction", "__getitem__")],
    ),
    # `desc` is `desc_nulls_last`, and `F.asc` is `asc`, which is `asc_nulls_first`, of the Column of
    # `col` of a name.
    "desc": (lambda df: df.a.desc(), [(_E, "desc_nulls_last")]),
    "F_asc": (
        lambda df: F.asc("a"),
        [(_E, "asc_nulls_first"), (f"{_E}.sort_order.child", "col")],
    ),
}


@pytest.mark.parametrize("name", list(ORIGINS))
def test_the_origins_of_column_methods_are_pysparks(name, monkeypatch):
    import pyspark.errors.utils as errors_utils

    make, expected = ORIGINS[name]
    # PySpark's process-wide cache of the debugging conf, which an earlier test may have set:
    # unset, it is read again from the conf of this session.
    monkeypatch.setattr(errors_utils, "_enable_debugging_cache", None)
    with mock_connect_session() as (spark, service):
        service.confs = {"spark.python.sql.dataFrameDebugging.enabled": "true"}
        df = spark.range(1).selectExpr("id as a", "array(id) as arr")
        before = len(service.execute_requests)
        try:
            df.select(make(df)).collect()
        except Exception:
            pass  # The mock's answer: only the request is compared.
        requests = [
            base_pb2.ExecutePlanRequest.FromString(raw)
            for raw in service.execute_requests[before:]
        ]
        origins = [origin for request in requests for origin in _origins(request.plan)]
    assert origins == expected


@pytest.mark.parametrize(
    "name, error",
    [
        (bytearray(b"q"), "bad argument type for built-in operation"),
        (None, "bad argument type for built-in operation"),
        (1, "bad argument type for built-in operation"),
    ],
)
def test_alias_names_are_pysparks_string_fields(name, error):
    # The name is a string field of the plan, which PySpark's ColumnAlias sets as it builds the
    # plan: a str, a bytes of its UTF-8, and protobuf's TypeError of any other value, which the
    # request raises, sending nothing; its __repr__ joins the names, which fails of a non-str.
    with mock_connect_session() as (spark, service):
        for method in ("alias", "name"):
            column = getattr(F.col("id"), method)(name)
            with pytest.raises(TypeError) as raised:
                repr(column)
            assert str(raised.value) == (
                f"sequence item 0: expected str instance, {type(name).__name__} found"
            )
            df = spark.range(1).select(column)
            before = len(service.execute_requests)
            with pytest.raises(TypeError) as raised:
                df.collect()
            assert str(raised.value) == error
            assert len(service.execute_requests) == before


def test_alias_metadata_fails_when_the_plan_is_built_as_pysparks():
    # `json.dumps` of the metadata of one name, and the metadata of no names, fail as the plan is
    # built; the metadata of several names, and other keywords, fail at the call, as PySpark's.
    with mock_connect_session() as (spark, _):
        df = spark.range(1)
        unserializable = F.col("id").alias("x", metadata={"k": {1}})
        no_names = F.col("id").alias(metadata={"k": "v"})
        assert repr(unserializable) == "Column<'id AS x'>"
        assert repr(no_names) == "Column<'id AS '>"
        with pytest.raises(TypeError, match="^Object of type set is not JSON serializable$"):
            df.select(unserializable).collect()
        with pytest.raises(PySparkValueError) as raised:
            df.select(no_names).collect()
        assert raised.value.getCondition() == "CANNOT_PROVIDE_METADATA"
        # The first error of the plan, in the order in which PySpark builds it: the name of an
        # alias before its child, and the metadata after it.
        nested = F.upper(F.col("id").alias(1)).alias("y", metadata={"k": {1}})
        with pytest.raises(TypeError, match="^bad argument type for built-in operation$"):
            df.select(nested).collect()
        with pytest.raises(PySparkValueError) as raised:
            F.col("id").alias("a", "b", metadata={"k": "v"})
        assert raised.value.getCondition() == "ONLY_ALLOWED_FOR_SINGLE_COLUMN"
        with pytest.raises(AssertionError, match="^Unexpected kwargs where passed"):
            F.col("id").alias("x", foo=1)


def test_a_column_of_an_alias_that_fails_copies_and_pickles_as_pysparks():
    # PySpark's Column copies and pickles its expression objects, and the copy fails as the
    # Column did when its plan is built: so does this client's, which pickles the errors.
    with mock_connect_session() as (spark, _):
        df = spark.range(1)
        column = F.upper(F.col("id").alias(1))
        for copied in (copy.copy(column), copy.deepcopy(column), pickle.loads(pickle.dumps(column))):
            with pytest.raises(TypeError, match="^sequence item 0: expected str instance, int found$"):
                repr(copied)
            with pytest.raises(TypeError, match="^bad argument type for built-in operation$"):
                df.select(copied).collect()
        no_names = pickle.loads(pickle.dumps(F.col("id").alias(metadata={"k": "v"})))
        with pytest.raises(PySparkValueError) as raised:
            df.select(no_names).collect()
        assert raised.value.getCondition() == "CANNOT_PROVIDE_METADATA"
        surrogate = pickle.loads(pickle.dumps(F.col("id").alias("\ud800")))
        with pytest.raises(UnicodeEncodeError, match="surrogates not allowed"):
            df.select(surrogate).collect()
        bad_bytes = pickle.loads(pickle.dumps(F.col("id").alias(b"\xff")))
        with pytest.raises(UnicodeDecodeError, match="invalid start byte"):
            df.select(bad_bytes).collect()


@pytest.mark.parametrize(
    "error, raised, message",
    [
        (("TypeError", "no", None), TypeError, "no"),
        (("ValueError", "no", None), ValueError, "no"),
        (("OverflowError", "no", None), OverflowError, "no"),
        (("AttributeError", "no", None), AttributeError, "no"),
        (("UnicodeEncodeError", "", "\ud800"), UnicodeEncodeError, "surrogates not allowed"),
        (("UnicodeDecodeError", "", b"\xff"), UnicodeDecodeError, "invalid start byte"),
        (("PySparkValueError", "CANNOT_PROVIDE_METADATA", {}), PySparkValueError, "single column"),
        (("PySparkAttributeError", "ATTRIBUTE_NOT_SUPPORTED", {"attr_name": "_expr"}),
         PySparkAttributeError, "Attribute `_expr` is not supported"),
        (("PySparkTypeError", "NOT_ITERABLE", {"objectName": "Column"}), PySparkTypeError,
         "Column is not iterable"),
    ],
)
def test_a_pickled_columns_errors_raise_on_a_worker_as_pysparks(error, raised, message):
    # The errors that a Python worker of the server raises of a pickled Column or DataFrame, of
    # PySpark's: a builtin error, the error of UTF-8 of a value, or the error of one of PySpark's
    # error classes, of its error class and parameters, as the plan raised it.
    from pyspark.sql.connect._column_reduce import raise_error

    with pytest.raises(raised, match=message) as caught:
        raise_error(error)
    assert type(caught.value) is raised


def test_alias_takes_bytes_as_pyspark():
    with mock_connect_session() as (spark, service):
        df = spark.range(1)
        before = len(service.execute_requests)
        try:
            df.select(F.col("id").alias(b"q")).collect()
        except Exception:
            pass
        (request,) = [
            base_pb2.ExecutePlanRequest.FromString(raw)
            for raw in service.execute_requests[before:]
        ]
    assert list(request.plan.root.project.expressions[0].alias.name) == ["q"]


def test_a_columns_attributes_are_kept_by_its_copies_as_pysparks():
    # As a plain object's: the attribute is the state of the reduce.
    c = F.col("id") + 1
    c.foo = 5
    assert copy.copy(c).foo == 5
    assert copy.deepcopy(c).foo == 5
    assert pickle.loads(pickle.dumps(c)).foo == 5
    # A Column of no attribute pickles as before.
    assert repr(pickle.loads(pickle.dumps(F.col("x") * 2))) == "Column<'*(x, 2)'>"
