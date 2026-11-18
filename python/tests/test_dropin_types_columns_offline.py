# SPDX-License-Identifier: Apache-2.0
"""The drop-in's types and Columns as official PySpark 4.3.0's, without a server: the JSON type
strings of `_parse_datatype_json_value`, the types that official PySpark pickled, the abstract
classes of the types (`DataType`, `AtomicType`, ...) as types of their own, and the errors of them
in the types of UDTFs, Java UDFs, tables and `createDataFrame`, the pickle of a
CalendarIntervalType, the parameters of the types as attributes, the parameters of the methods of
the types, the origins of the expressions that Column methods send, the names of `Column.alias`
and `Column.name`, and the user-set attributes of a Column's copies. The expected values are
official PySpark's, of the same calls."""

import base64
import copy
import datetime
import inspect
import json
import pickle
from pathlib import Path

import pytest
from _mock_connect_server import mock_connect_session
from pyspark.errors import PySparkValueError
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
    the attribute of the drop-in's type `t`. The `_crs` and `_alg` that PySpark's spatial types
    keep of their SRID are not attributes of the drop-in's."""
    expected = {k: v for k, v in case["attributes"].items() if k not in ("_crs", "_alg")}
    return {k: repr(getattr(t, k)) for k in expected}, expected


@pytest.mark.parametrize("protocol", [2, 3, 4, 5])
def test_types_that_pyspark_pickled_load_as_its_types(protocol):
    # PySpark's types are plain classes, which pickle as `copyreg.__newobj__(cls)` and their
    # `__dict__`: the drop-in's class makes a type of `cls.__new__(cls)`, without arguments, and
    # sets the attributes of the `__dict__`, of which the type then is, also in a struct, an
    # array or a map. They are its parameters, not entries of its own `__dict__`. A type of
    # PySpark's `DataTypeSingleton` is the one instance, which `cls.__new__(cls)` gives, where
    # PySpark's own unpickling gives another object.
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
        if name != "set_srid":
            assert t.json() == case["json"], name
    decimal = loaded["decimal_attributes"]
    assert vars(decimal) == {"tag": "x"}
    assert decimal != T.DecimalType(10, 2) and hash(decimal) == hash(T.DecimalType(10, 2))
    assert loaded["set_scale"] == T.DecimalType(10, 3)
    assert loaded["retyped_field"]._needSerializeAnyField is True
    # PySpark's JSON of a GeometryType of a set SRID is that of the CRS of its constructor, which
    # it keeps as `_crs`; the drop-in's type is of the SRID, so it is GeometryType(0), of its JSON.
    assert PYSPARK_PICKLES["edited"]["set_srid"]["json"] == '"geometry(OGC:CRS84)"'
    assert loaded["set_srid"] == T.GeometryType(0)
    assert loaded["set_srid"].json() == '"geometry(SRID:0)"'


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
    # so that its type is equal to no GeometryType(0), where the drop-in's type is of the SRID,
    # also of its JSON. The mixed SRID -1, of which `GeographyType("ANY")` is, is an SRID too.
    geometry = T.GeometryType(4326)
    geometry.srid = 0
    assert (repr(geometry), geometry.simpleString()) == ("GeometryType(0)", "geometry(0)")
    assert geometry == T.GeometryType(0) and geometry.json() == '"geometry(SRID:0)"'
    geography = T.GeographyType(4326)
    geography.srid = -1
    assert (repr(geography), geography.simpleString()) == ("GeographyType(ANY)", "geography(any)")
    assert geography == T.GeographyType("ANY")
    # `hasPrecisionInfo`, which PySpark's `__init__` sets true, is a plain attribute too, which
    # `==` compares, as PySpark's `__dict__`, and the repr and hash do not show.
    decimal = T.DecimalType(10, 2)
    decimal.hasPrecisionInfo = False
    assert decimal.hasPrecisionInfo is False and repr(decimal) == "DecimalType(10,2)"
    assert decimal != T.DecimalType(10, 2) and hash(decimal) == hash(T.DecimalType(10, 2))


def test_type_parameters_take_the_values_of_their_constructors():
    # PySpark's attributes take any value, of which its repr, JSON or simpleString may fail later
    # (the TypeError of `%d` of a str, the KeyError of a field that the type does not have). The
    # drop-in's core type holds the values that the constructor takes, so another raises the
    # constructor's error at the assignment, and the type stays as it was.
    from pyspark.errors import IllegalArgumentException, PySparkRuntimeError

    decimal, interval, geometry = T.DecimalType(10, 2), T.DayTimeIntervalType(), T.GeometryType(0)
    with pytest.raises(TypeError):
        decimal.scale = "3"
    with pytest.raises(TypeError):
        T.StringType().collation = 5
    for name, value, parameters in [
        ("startField", 7, {"start_field": "7", "end_field": "3"}),
        ("endField", None, {"start_field": "0", "end_field": "None"}),
    ]:
        with pytest.raises(PySparkRuntimeError) as e:
            setattr(interval, name, value)
        assert e.value.getCondition() == "INVALID_INTERVAL_CASTING"
        assert e.value.getMessageParameters() == parameters
    for t, srid in [(geometry, 1234), (T.GeographyType(4326), 0)]:
        with pytest.raises(IllegalArgumentException) as e:
            t.srid = srid
        assert e.value.getCondition() == "ST_INVALID_SRID_VALUE"
    assert (decimal, interval, geometry) == (
        T.DecimalType(10, 2), T.DayTimeIntervalType(), T.GeometryType(0)
    )
    # The `"ANY"` of the constructor is the mixed SRID, which PySpark's attribute keeps as a str.
    geometry.srid = "ANY"
    assert geometry.srid == -1 and geometry == T.GeometryType("ANY")


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
    # loads as the same type. The drop-in's own pickles it of the types it is made of, as of a
    # user-defined type, since PySpark's `_parse_datatype_json_string` does not parse the JSON
    # of an abstract type (`"integral"`): both clients load it at every protocol.
    made = eval(made, vars(T))
    loaded = pickle.loads(pickled)
    assert type(loaded) is type(made) and loaded == made
    assert (repr(loaded), loaded.json()) == (repr(made), made.json())
    for protocol in range(6):
        copied = pickle.loads(pickle.dumps(made, protocol=protocol))
        assert type(copied) is type(made) and copied == made, protocol
        assert (repr(copied), copied.json()) == (repr(made), made.json()), protocol
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
    # A native constructor returns an object that exists, or makes one of the class it is called
    # for, so a Python subclass of IntegralType cannot have the one instance of its own that
    # PySpark's DataTypeSingleton keeps of each class: it cannot be instantiated.
    class MyIntegral(T.IntegralType):
        pass

    with pytest.raises(TypeError, match=r"^cannot create 'MyIntegral' instances$"):
        MyIntegral()


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
    ]:
        assert type(cls.__new__(cls)) is cls
        signature = inspect.signature(cls).parameters.values()
        assert [(p.name, p.default) for p in signature] == parameters
    array, field = T.ArrayType(T.IntegerType()), T.StructField("a", T.IntegerType())
    array.__init__(T.LongType(), False)
    field.__init__("b", T.CharType(3), False, {"k": "v"})
    assert array == T.ArrayType(T.LongType(), False)
    assert field == T.StructField("b", T.CharType(3), False, {"k": "v"})


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
        (("UnicodeEncodeError", "", "\ud800"), UnicodeEncodeError, "surrogates not allowed"),
        (("UnicodeDecodeError", "", b"\xff"), UnicodeDecodeError, "invalid start byte"),
        (("PySparkValueError", "CANNOT_PROVIDE_METADATA", {}), PySparkValueError, "single column"),
    ],
)
def test_a_pickled_columns_errors_raise_on_a_worker_as_pysparks(error, raised, message):
    # The errors that a Python worker of the server raises of a pickled Column, of PySpark's.
    from pyspark.sql.connect._column_reduce import raise_error

    with pytest.raises(raised, match=message):
        raise_error(error)


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
