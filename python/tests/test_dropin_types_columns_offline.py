# SPDX-License-Identifier: Apache-2.0
"""The drop-in's types and Columns as official PySpark 4.3.0's, without a server: the JSON type
strings of `_parse_datatype_json_value`, the types that official PySpark pickled, the pickle of a
CalendarIntervalType, the parameters of the types as attributes, the origins of the expressions
that Column methods send, the names of `Column.alias` and `Column.name`, and the user-set
attributes of a Column's copies. The expected values are official PySpark's, of the same calls."""

import base64
import copy
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
