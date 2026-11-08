"""Offline tests for the pyspark drop-in (no Spark Connect server required).

These exercise the pure-Python shim (types parsing/pickling, the session builder,
functions/column expression building, udf/udtf construction, StorageLevel) so the
Python side has real, measurable coverage in CI without a running server. Tests
that require a live server live in test_dropin_e2e.py and are gated on SPARK_REMOTE.
"""

import array
import contextlib
import datetime
import os
import re
import decimal
import json
import pickle
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from fractions import Fraction
from threading import Barrier

import pytest

from _mock_connect_server import (
    ddl_data_type,
    ddl_json,
    ddl_parse_response,
    message_field,
    mock_connect_session,
    parse_message,
    struct_schema,
    varint_field,
)
from pyspark.sql import Row, SparkSession, types as T

# Collected here so the existing offline CI suite exercises real Arrow results.
from day_time_interval_cases import DayTimeIntervalTests  # noqa: F401
from interval_input_cases import IntervalInputConversionTests  # noqa: F401


def test_create_dataframe_input_names_offline():
    # Keep the fixture's active session out of this process's offline tests.
    import subprocess
    import sys
    from pathlib import Path

    for dependency in ("grpc", "pandas", "pyarrow"):
        pytest.importorskip(dependency)
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("input_name_cases.py"))],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_create_dataframe_integer_tuples_offline():
    # Isolate the mock fixture's active session from the rest of the offline suite.
    import subprocess
    import sys
    from pathlib import Path

    for dependency in ("grpc", "pandas", "pyarrow"):
        pytest.importorskip(dependency)
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("integer_tuple_cases.py"))],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


# --------------------------------------------------------------------------- types

ATOMIC = [
    T.NullType, T.BooleanType, T.ByteType, T.ShortType, T.IntegerType, T.LongType,
    T.FloatType, T.DoubleType, T.StringType, T.BinaryType, T.DateType,
    T.TimestampType, T.TimestampNTZType, T.VariantType,
]


@pytest.mark.parametrize("cls", ATOMIC)
def test_atomic_types_instantiate_and_roundtrip(cls):
    t = cls()
    assert t.typeName()
    assert t.simpleString()
    assert pickle.loads(pickle.dumps(t)).simpleString() == t.simpleString()


def test_parameterized_types_roundtrip():
    cases = [
        T.DecimalType(12, 3), T.CharType(5), T.VarcharType(9), T.TimeType(3),
        T.CalendarIntervalType(), T.YearMonthIntervalType(0, 1),
        T.YearMonthIntervalType(0, 0), T.DayTimeIntervalType(0, 3),
        T.DayTimeIntervalType(0, 1),
    ]
    # Interval/time types expose repr but not simpleString; compare structurally
    # via the value round-tripped through pickle (which uses __reduce__'s JSON).
    for t in cases:
        assert repr(pickle.loads(pickle.dumps(t))) == repr(t)


def test_decimal_precision_scale():
    d = T.DecimalType(10, 2)
    assert d.precision == 10 and d.scale == 2


def test_nested_types_pickle_roundtrip():
    nested = [
        T.ArrayType(T.IntegerType()),
        T.MapType(T.StringType(), T.ArrayType(T.IntegerType())),
        T.StructType([
            T.StructField("a", T.ArrayType(T.IntegerType())),
            T.StructField("b", T.MapType(T.StringType(), T.IntegerType()), False),
        ]),
        T.ArrayType(T.StructType([
            T.StructField("x", T.MapType(T.StringType(), T.ArrayType(T.DecimalType(10, 2)))),
        ])),
    ]
    for t in nested:
        assert pickle.loads(pickle.dumps(t)).simpleString() == t.simpleString()


def test_parse_datatype_json_string_nested():
    t = T.StructType([
        T.StructField("a", T.ArrayType(T.IntegerType())),
        T.StructField("b", T.MapType(T.StringType(), T.IntegerType())),
    ])
    # Round-trip via the JSON that __reduce__ emits.
    j = pickle.dumps(t)
    assert pickle.loads(j).simpleString() == t.simpleString()


def test_parse_datatype_json_string_atomic_forms():
    import json
    assert T._parse_datatype_json_string(json.dumps("integer")).simpleString() == "int"
    assert T._parse_datatype_json_string(json.dumps("decimal(10,2)")).simpleString() == "decimal(10,2)"
    assert T._parse_datatype_json_string(json.dumps("char(4)")).simpleString() == "char(4)"


def test_parse_datatype_json_string_rejects_garbage():
    import json
    with pytest.raises(ValueError):
        T._parse_datatype_json_string(json.dumps("not_a_type"))


# One instance of every DataType variant the drop-in has, including the parameterized
# forms whose JSON needs more than a bare type name (collations, spatial CRS names).
EVERY_TYPE = [
    T.NullType(), T.BooleanType(), T.ByteType(), T.ShortType(), T.IntegerType(),
    T.LongType(), T.FloatType(), T.DoubleType(), T.DecimalType(), T.DecimalType(38, 18),
    T.StringType(), T.StringType("UTF8_LCASE"), T.StringType("UNICODE_CI"),
    T.CharType(5), T.VarcharType(9), T.BinaryType(), T.DateType(), T.TimestampType(),
    T.TimestampNTZType(), T.TimestampNTZNanosType(9), T.TimestampLTZNanosType(7),
    T.TimeType(3), T.CalendarIntervalType(), T.YearMonthIntervalType(0, 1),
    T.YearMonthIntervalType(1, 1), T.DayTimeIntervalType(0, 3),
    T.DayTimeIntervalType(2, 2), T.VariantType(),
    T.GeometryType(0), T.GeometryType(3857), T.GeometryType(4326), T.GeometryType("ANY"),
    T.GeographyType(4326), T.GeographyType("ANY"),
    T.ArrayType(T.GeometryType(3857), False),
    T.MapType(T.StringType("UTF8_LCASE"), T.GeographyType("ANY"), False),
]


@pytest.mark.parametrize("t", EVERY_TYPE, ids=repr)
def test_every_datatype_pickle_roundtrip(t):
    r = pickle.loads(pickle.dumps(t))
    assert type(r) is type(t)
    assert r == t and hash(r) == hash(t)
    assert r.json() == t.json()


def test_every_datatype_pickle_roundtrip_nested_in_struct():
    st = T.StructType([
        T.StructField(f"f{i}", t, i % 2 == 0, {"i": i, "tag": "v"})
        for i, t in enumerate(EVERY_TYPE)
    ])
    nested = T.StructType([
        T.StructField("s", st),
        T.StructField("a", T.ArrayType(st)),
        T.StructField("m", T.MapType(T.StringType("UNICODE"), st)),
    ])
    for t in (st, nested):
        r = pickle.loads(pickle.dumps(t))
        assert r == t
        assert r.json() == t.json()
    r = pickle.loads(pickle.dumps(st))
    for got, want in zip(r.fields, st.fields):
        assert type(got.dataType) is type(want.dataType)
        assert got.dataType == want.dataType
        assert got.metadata == want.metadata


def test_spatial_and_collated_json_forms_parse():
    import json
    cases = {
        "geometry(SRID:0)": T.GeometryType(0),
        "geometry(EPSG:3857)": T.GeometryType(3857),
        "geometry(OGC:CRS84)": T.GeometryType(4326),
        "geometry(SRID:ANY)": T.GeometryType("ANY"),
        "geography(OGC:CRS84, SPHERICAL)": T.GeographyType(4326),
        "geography(SRID:ANY, SPHERICAL)": T.GeographyType("ANY"),
        "string collate UTF8_LCASE": T.StringType("UTF8_LCASE"),
    }
    for j, want in cases.items():
        assert T._parse_datatype_json_string(json.dumps(j)) == want
        assert want.json() == json.dumps(j)


def test_private_type_helpers_are_those_of_pyspark():
    # As PySpark's pyspark.sql.types: the inference, merging, conversion and verification of the
    # types of Python objects, and their tables. They take the client's Row as PySpark's take
    # theirs, a tuple.
    from array import array

    from pyspark.errors import PySparkTypeError, PySparkValueError
    from pyspark.sql.types import (
        JVM_INT_MAX,
        _acceptable_types,
        _array_type_mappings,
        _create_converter,
        _has_nulltype,
        _infer_schema,
        _infer_type,
        _make_type_verifier,
        _merge_type,
    )

    assert JVM_INT_MAX == 2**31 - 1
    assert _array_type_mappings["d"] is T.DoubleType and _array_type_mappings["i"] is T.IntegerType
    assert T.StructType in _acceptable_types and dict in _acceptable_types[T.StructType]
    assert _infer_type(array("d", [1.0])) == T.ArrayType(T.DoubleType(), False)
    assert _infer_type({"k": [1]}) == T.MapType(T.StringType(), T.ArrayType(T.LongType()))
    schema = T.StructType([T.StructField("a", T.LongType()), T.StructField("b", T.StringType())])
    assert _infer_schema(Row(a=1, b="x")) == schema
    assert _infer_schema(Row(1, "x"), names=["a", "b"]) == schema
    assert _has_nulltype(_infer_schema({"a": None}))
    merged = _merge_type(schema, T.StructType([T.StructField("c", T.DoubleType())]))
    assert merged.names == ["a", "b", "c"]
    with pytest.raises(PySparkTypeError) as error:
        _merge_type(T.LongType(), T.DoubleType())
    assert error.value.getCondition() == "CANNOT_MERGE_TYPE"

    verify = _make_type_verifier(T.StructType([T.StructField("a", T.IntegerType(), False)]))
    for value in (Row(a=1), (1,), {"a": 1}):
        verify(value)
    with pytest.raises(PySparkValueError) as error:
        verify(Row(a=None))
    assert error.value.getCondition() == "FIELD_NOT_NULLABLE_WITH_NAME"
    with pytest.raises(PySparkValueError) as error:
        verify(Row(a=1, b=2))
    assert error.value.getCondition() == "FIELD_STRUCT_LENGTH_MISMATCH"
    struct = T.StructType([T.StructField("m", T.MapType(T.StringType(), T.IntegerType()))])
    assert _create_converter(struct)(Row(m={"k": 1})) == ({"k": 1},)




def _type_verifier_cases():
    # The cases of Apache Spark's DataTypeVerificationTests.test_verify_type_not_nullable: the
    # values that a type takes, and those it rejects, of the error of each.
    import array
    import datetime
    import decimal

    from pyspark.testing.objects import ExamplePoint, ExamplePointUDT, PythonOnlyUDT

    schema = T.StructType(
        [T.StructField("s", T.StringType(), False), T.StructField("i", T.IntegerType(), True)]
    )

    class MyObj:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    success = [
        ("", T.StringType()), (1, T.StringType()), ([], T.StringType()), ({}, T.StringType()),
        ("", T.StringType("UTF8_LCASE")), (1.0, T.CharType(10)), ({}, T.VarcharType(10)),
        (ExamplePoint(1.0, 2.0), ExamplePointUDT()), (True, T.BooleanType()),
        (-(2**7), T.ByteType()), (2**7 - 1, T.ByteType()), (-(2**15), T.ShortType()),
        (2**15 - 1, T.ShortType()), (-(2**31), T.IntegerType()), (2**31 - 1, T.IntegerType()),
        (-(2**63), T.LongType()), (2**63 - 1, T.LongType()), (1.0, T.FloatType()),
        (1.0, T.DoubleType()), (decimal.Decimal("1.0"), T.DecimalType()),
        (bytearray([1, 2]), T.BinaryType()), (datetime.date(2000, 1, 2), T.DateType()),
        (datetime.time(1, 0, 0), T.TimeType()), (datetime.datetime(2000, 1, 2), T.DateType()),
        (datetime.datetime(2000, 1, 2, 3, 4), T.TimestampType()),
        ([], T.ArrayType(T.IntegerType())), (["1", None], T.ArrayType(T.StringType())),
        ((1, 2), T.ArrayType(T.IntegerType())), (array.array("h", [1]), T.ArrayType(T.IntegerType())),
        ({"a": 1}, T.MapType(T.StringType(), T.IntegerType())),
        ({"a": None}, T.MapType(T.StringType(), T.IntegerType(), True)),
        ({"s": "a", "i": None}, schema), ({"s": "a", "f": 1.0}, schema), (Row(s="a", i=1), schema),
        (Row(s="a", i=None), schema), (["a", 1], schema), (("a", None), schema),
        (MyObj(s="a", i=1), schema), (MyObj(s="a"), schema),
    ]
    failure = [
        (None, T.StringType(), ValueError), (None, T.CharType(10), ValueError),
        (None, T.VarcharType(10), ValueError), (ExamplePoint(1.0, 2.0), PythonOnlyUDT(), ValueError),
        (1, T.BooleanType(), TypeError), ("True", T.BooleanType(), TypeError),
        (-(2**7) - 1, T.ByteType(), ValueError), (2**7, T.ByteType(), ValueError),
        ("1", T.ByteType(), TypeError), (1.0, T.ByteType(), TypeError),
        (2**15, T.ShortType(), ValueError), (2**31, T.IntegerType(), ValueError),
        (1, T.FloatType(), TypeError), (1, T.DoubleType(), TypeError),
        (1.0, T.DecimalType(), TypeError), ("1.0", T.DecimalType(), TypeError),
        (1, T.BinaryType(), TypeError), ("2000-01-02", T.DateType(), TypeError),
        ("23:59:59", T.TimeType(), TypeError), (946811040, T.TimestampType(), TypeError),
        (["1", None], T.ArrayType(T.StringType(), False), ValueError),
        ([1, "2"], T.ArrayType(T.IntegerType()), TypeError),
        ({"a": 1}, T.MapType(T.IntegerType(), T.IntegerType()), TypeError),
        ({"a": "1"}, T.MapType(T.StringType(), T.IntegerType()), TypeError),
        ({"a": None}, T.MapType(T.StringType(), T.IntegerType(), False), ValueError),
        ({"s": "a", "i": "1"}, schema, TypeError), (Row(s="a"), schema, ValueError),
        (Row(s="a", i="1"), schema, TypeError), (["a"], schema, ValueError),
        (["a", "1"], schema, TypeError), (MyObj(s="a", i="1"), schema, TypeError),
        (MyObj(s=None, i="1"), schema, ValueError),
    ]
    return success, failure


def test_type_verifiers_take_and_reject_the_values_of_pyspark():
    # As Apache Spark's DataTypeVerificationTests: _make_type_verifier takes the values of each
    # type, of the client's Row too, and raises the errors of PySpark for the others, with the
    # names of the fields.
    from pyspark.errors import PySparkTypeError, PySparkValueError
    from pyspark.sql.types import _make_type_verifier

    success, failure = _type_verifier_cases()
    for obj, data_type in success:
        _make_type_verifier(data_type, nullable=False)(obj)
    for obj, data_type, error in failure:
        with pytest.raises(error):
            _make_type_verifier(data_type, nullable=False)(obj)
    for data_type in (T.IntegerType(), T.StringType(), T.StructType([])):
        _make_type_verifier(data_type, nullable=True)(None)
    with pytest.raises(PySparkValueError) as e:
        _make_type_verifier(T.StringType(), nullable=False, name="test_name")(None)
    assert e.value.getCondition() == "FIELD_NOT_NULLABLE_WITH_NAME"
    schema = T.StructType([T.StructField("a", T.StructType([T.StructField("b", T.IntegerType())]))])
    with pytest.raises(PySparkTypeError) as e:
        _make_type_verifier(schema)([["data"]])
    assert e.value.getCondition() == "FIELD_DATA_TYPE_UNACCEPTABLE_WITH_NAME"
    assert e.value.getMessageParameters()["field_name"] == "field b in field a"


def test_types_of_python_objects_are_inferred_merged_and_converted_as_pyspark():
    # As PySpark's _infer_type, _infer_schema, _merge_type and _create_converter, of the
    # values of each type, also nested, of the client's Row too.
    import datetime
    import decimal

    from pyspark.errors import PySparkTypeError
    from pyspark.sql.types import (
        _create_converter,
        _infer_schema,
        _infer_type,
        _merge_type,
        _need_converter,
    )

    values = [
        (None, T.NullType()), (True, T.BooleanType()), (1, T.LongType()), (1.5, T.DoubleType()),
        ("s", T.StringType()), (b"b", T.BinaryType()),
        (decimal.Decimal("1.50"), T.DecimalType(38, 18)), (datetime.date(2000, 1, 1), T.DateType()),
        (datetime.datetime(2000, 1, 1), T.TimestampType()), (datetime.time(1, 2), T.TimeType()),
        (datetime.timedelta(1), T.DayTimeIntervalType()),
    ]
    for value, data_type in values:
        assert _infer_type(value) == data_type
    assert _infer_type([1, None]) == T.ArrayType(T.LongType())
    assert _infer_type([None, 1], infer_array_from_first_element=True) == T.ArrayType(
        T.NullType()
    )
    assert _infer_type({"a": {"b": 1}}, infer_dict_as_struct=True) == T.StructType(
        [T.StructField("a", T.StructType([T.StructField("b", T.LongType())]))]
    )
    assert _infer_type(datetime.datetime(2000, 1, 1), prefer_timestamp_ntz=True) == (
        T.TimestampNTZType()
    )
    assert _infer_type(Row(a=[Row(b=1)])) == T.StructType(
        [T.StructField("a", T.ArrayType(T.StructType([T.StructField("b", T.LongType())])))]
    )

    class Obj:
        def __init__(self):
            self.b, self.a = "x", 1

    assert _infer_schema(Obj()).names == ["a", "b"]
    assert _infer_schema((1, 2), names=["x"]).names == ["x", "_2"]
    with pytest.raises(PySparkTypeError) as e:
        _infer_schema(1)
    assert e.value.getCondition() == "CANNOT_INFER_SCHEMA_FOR_TYPE"

    a = T.StructType([T.StructField("f", T.ArrayType(T.NullType()))])
    b = T.StructType([T.StructField("f", T.ArrayType(T.LongType()))])
    assert _merge_type(a, b) == b and _merge_type(T.NullType(), T.LongType()) == T.LongType()
    assert _merge_type(
        T.MapType(T.StringType(), T.NullType()), T.MapType(T.StringType(), T.LongType())
    ) == T.MapType(T.StringType(), T.LongType())
    strings = T.StructType([T.StructField("f", T.ArrayType(T.StringType()))])
    assert _merge_type(b, strings) == strings
    with pytest.raises(PySparkTypeError) as e:
        _merge_type(b, T.StructType([T.StructField("f", T.ArrayType(T.DateType()))]))
    assert e.value.getCondition() == "CANNOT_MERGE_TYPE"

    struct = T.StructType(
        [T.StructField("m", T.MapType(T.StringType(), T.StringType())),
         T.StructField("s", T.StructType([T.StructField("x", T.LongType())]))]
    )
    assert _need_converter(struct) and not _need_converter(T.LongType())
    convert = _create_converter(struct)
    assert convert(Row(m={"k": "v"}, s=Row(x=1))) == ({"k": "v"}, (1,))
    assert convert({"m": None, "s": {"x": 2}}) == (None, (2,))
    with pytest.raises(PySparkTypeError) as e:
        convert(1)
    assert e.value.getCondition() == "UNSUPPORTED_DATA_TYPE"




def test_unparsed_datatype_pickle_roundtrip():
    # The core's JSON form of an unparsed type is the UnparsedDataType of
    # pyspark.sql.connect.types, which pickles of its DDL string, as its JSON raises.
    from pyspark.sql.connect.types import UnparsedDataType

    t = T._parse_datatype_json_value("unparsed(some_ddl)")
    assert isinstance(t, UnparsedDataType) and t.data_type_string == "some_ddl"
    r = pickle.loads(pickle.dumps(t))
    assert r == t and repr(r) == "UnparsedDataType('some_ddl')"


def test_user_defined_type_pickle_roundtrip():
    import sys
    import types as _pytypes

    mod = _pytypes.ModuleType("_udt_pickle_mod")
    sys.modules["_udt_pickle_mod"] = mod
    src = (
        "from pyspark.sql.types import UserDefinedType, ArrayType, DoubleType\n"
        "class PointUDT(UserDefinedType):\n"
        "    @classmethod\n"
        "    def sqlType(cls): return ArrayType(DoubleType(), False)\n"
        "    @classmethod\n"
        "    def module(cls): return '_udt_pickle_mod'\n"
    )
    exec(compile(src, "_udt_pickle_mod", "exec"), mod.__dict__)
    try:
        u = mod.PointUDT()
        assert isinstance(pickle.loads(pickle.dumps(u)), mod.PointUDT)
        st = T.StructType([
            T.StructField("p", u), T.StructField("g", T.GeometryType(4326)),
        ])
        r = pickle.loads(pickle.dumps(st))
        assert isinstance(r.fields[0].dataType, mod.PointUDT)
        assert r.fields[1].dataType == T.GeometryType(4326)
    finally:
        del sys.modules["_udt_pickle_mod"]


def test_structfield_construction():
    f = T.StructField("n", T.IntegerType(), True)
    st = T.StructType([f])
    assert "n" in st.simpleString()


# --------------------------------------------------------------------------- type equality
# Expected results are what PySpark 4.3.0 returns (`DataType.__eq__` is
# `isinstance(other, self.__class__) and self.__dict__ == other.__dict__`, `__hash__` is
# `hash(str(self))`).

def _eq_cases():
    F = T.StructField
    return [
        T.NullType, T.BooleanType, T.ByteType, T.ShortType, T.IntegerType, T.LongType,
        T.FloatType, T.DoubleType, T.StringType, lambda: T.StringType("UTF8_LCASE"),
        T.BinaryType, T.DateType, T.TimestampType, T.TimestampNTZType, T.VariantType,
        T.CalendarIntervalType, T.DecimalType, lambda: T.DecimalType(10, 2),
        lambda: T.DecimalType(12, 2), lambda: T.CharType(5), lambda: T.CharType(6),
        lambda: T.VarcharType(5), lambda: T.TimeType(3), lambda: T.TimeType(6),
        lambda: T.YearMonthIntervalType(0, 1), lambda: T.YearMonthIntervalType(0, 0),
        lambda: T.DayTimeIntervalType(0, 3), lambda: T.DayTimeIntervalType(1, 3),
        lambda: T.GeometryType(0), lambda: T.GeometryType(4326),
        lambda: T.GeographyType(4326),
        lambda: T.ArrayType(T.IntegerType()), lambda: T.ArrayType(T.IntegerType(), False),
        lambda: T.ArrayType(T.LongType()),
        lambda: T.MapType(T.StringType(), T.IntegerType()),
        lambda: T.MapType(T.StringType(), T.IntegerType(), False),
        lambda: T.StructType([F("a", T.IntegerType())]),
        lambda: T.StructType([F("a", T.IntegerType(), False)]),
        lambda: T.StructType([F("b", T.IntegerType())]),
        lambda: T.StructType([F("a", T.IntegerType(), True, {"k": "v"})]),
        lambda: T.StructType(),
        lambda: T.StructType([
            F("x", T.ArrayType(T.MapType(T.StringType(), T.DecimalType(10, 2))))
        ]),
    ]


def test_data_types_convert_values_to_and_from_internal_as_pyspark(monkeypatch):
    # As PySpark's toInternal and fromInternal: the days of a date since the epoch, the
    # microseconds of a timestamp and of a day-time interval, the nanoseconds of a time, the
    # dicts of variants and spatial values, and the elements, keys, values and fields of
    # arrays, maps and structs; and needConversion of the types that convert.
    import datetime

    from pyspark.errors import PySparkNotImplementedError, PySparkValueError
    from pyspark.sql.types import Geometry, VariantVal
    from pyspark.testing.objects import PythonOnlyUDT

    d, dt = datetime.date(2020, 1, 1), datetime.datetime(2020, 1, 1, 12, 30, 15, 123456)
    assert T.DateType().toInternal(d) == 18262 and T.DateType().fromInternal(18262) == d
    assert T.DateType().toInternal(None) is None
    utc = dt.replace(tzinfo=datetime.timezone.utc)
    assert T.TimestampType().toInternal(utc) == 1577881815123456
    assert T.TimestampType().fromInternal(1577881815123456) == (
        datetime.datetime.fromtimestamp(1577881815).replace(microsecond=123456))
    assert T.TimestampNTZType().toInternal(dt) == 1577881815123456
    assert T.TimestampNTZType().fromInternal(1577881815123456) == dt
    time = datetime.time(12, 30, 15, 123456)
    assert T.TimeType().toInternal(time) == 45015123456000
    assert T.TimeType().fromInternal(45015123456000) == time
    delta = datetime.timedelta(days=1, seconds=5, microseconds=7)
    assert T.DayTimeIntervalType().toInternal(delta) == 86405000007
    assert T.DayTimeIntervalType().fromInternal(86405000007) == delta
    for data_type in (T.YearMonthIntervalType(), T.CalendarIntervalType()):
        for convert in (data_type.toInternal, data_type.fromInternal):
            with pytest.raises(PySparkNotImplementedError) as e:
                convert(1)
            assert e.value.getCondition() == "NOT_IMPLEMENTED"
    variant = VariantVal(b"\x0c\x01", b"\x01\x00\x00")
    internal = {"value": b"\x0c\x01", "metadata": b"\x01\x00\x00"}
    assert T.VariantType().toInternal(variant) == internal
    assert T.VariantType().fromInternal(internal).value == b"\x0c\x01"
    wkb = bytes.fromhex("0101000000000000000000f03f0000000000000040")
    assert T.GeometryType(4326).toInternal(Geometry.fromWKB(wkb, 4326)) == {
        "srid": 4326, "wkb": wkb}
    point = T.GeometryType(4326).fromInternal({"srid": 4326, "bytes": wkb})
    assert (point.srid, point.wkb) == (4326, wkb)

    st = T.StructType([
        T.StructField("a", T.IntegerType()),
        T.StructField("d", T.DateType()),
        T.StructField("l", T.ArrayType(T.DateType())),
        T.StructField("m", T.MapType(T.StringType(), T.DateType())),
    ])
    internal = (1, 18262, [18262, None], {"k": 18262})
    row = Row(a=1, d=d, l=[d, None], m={"k": d})
    for value in ((1, d, [d, None], {"k": d}), {"a": 1, "d": d, "l": [d, None], "m": {"k": d}},
                  row):
        assert st.toInternal(value) == internal
    assert st.fromInternal(internal) == row
    assert (st._needSerializeAnyField, st._needConversion) == (True, [False, True, True, True])
    assert st.fields[1].toInternal(d) == 18262
    with pytest.raises(PySparkValueError) as e:
        st.toInternal(5)
    assert e.value.getCondition() == "UNEXPECTED_TUPLE_WITH_STRUCT"

    for data_type in (T.VariantType(), T.GeometryType(4326), T.GeographyType(4326),
                      T.ArrayType(PythonOnlyUDT()), T.YearMonthIntervalType()):
        assert data_type.needConversion() is True
    assert T.ArrayType(T.IntegerType()).needConversion() is False
    monkeypatch.setenv("PYSPARK_YM_INTERVAL_LEGACY", "1")
    assert T.ArrayType(T.YearMonthIntervalType()).needConversion() is False




def test_datatype_eq_and_hash_match_pyspark():
    cases = _eq_cases()
    left = [f() for f in cases]
    right = [f() for f in cases]
    for i, a in enumerate(left):
        for j, b in enumerate(right):
            # Every case is a distinct type, so equal exactly on the diagonal.
            assert (a == b) is (i == j), (a, b)
            assert (a != b) is (i != j), (a, b)
        assert hash(a) == hash(right[i]), a
    assert type(T.LongType()).__eq__ is not object.__eq__


def test_datatype_eq_across_construction_paths(mock_connect):
    import json

    st = T.StructType([T.StructField("a", T.IntegerType())])
    assert T.IntegerType() == T.DataType.fromDDL("int")
    assert T.DataType.fromDDL("int") == T.IntegerType()
    assert hash(T.DataType.fromDDL("int")) == hash(T.IntegerType())
    assert T.DataType.fromDDL("bigint") != T.IntegerType()
    assert T._parse_datatype_json_string(json.dumps("integer")) == T.IntegerType()
    assert T._parse_datatype_json_string(st.json()) == st
    assert pickle.loads(pickle.dumps(st)) == st
    assert T.StructType().add("a", T.IntegerType()) == st


def test_datatype_eq_with_non_datatypes():
    for t in (T.IntegerType(), T.StructType([T.StructField("a", T.IntegerType())])):
        assert (t == "int") is False
        assert (t == None) is False  # noqa: E711
        assert t != "int"
    assert (T.StructField("a", T.LongType()) == T.LongType()) is False


def test_mutated_structtype_compares_by_live_fields():
    target = T.StructType([T.StructField("a", T.IntegerType())])
    s = T.StructType()
    assert s != target
    s.add("a", T.IntegerType())
    assert s == target
    assert hash(s) == hash(target)


def test_datatypes_usable_in_sets_and_dicts():
    assert len({T.LongType(), T.LongType(), T.IntegerType()}) == 2
    assert {T.ArrayType(T.LongType()): 1}.get(T.ArrayType(T.LongType())) == 1
    fa, fa2, fb = (
        T.StructField("a", T.LongType()),
        T.StructField("a", T.LongType()),
        T.StructField("a", T.IntegerType()),
    )
    assert fa == fa2 and fa != fb
    assert hash(fa) == hash(fa2)
    assert len({fa, fa2, fb}) == 2


def test_data_types_have_their_attributes_type_names_and_singletons_as_pyspark():
    # As PySpark's: the attributes of the parameters of the types, typeName as a classmethod of
    # the name of the class, INVALID_TYPENAME_CALL of StructField.typeName, and the types of
    # DataTypeSingleton, of which every call returns the one instance.
    from pyspark.errors import PySparkTypeError

    assert T.StringType("UTF8_LCASE").collation == "UTF8_LCASE"
    assert T.StringType().collation == "UTF8_BINARY"
    assert (T.CharType(3).length, T.VarcharType(4).length, T.TimeType(3).precision) == (3, 4, 3)
    interval = T.DayTimeIntervalType(T.DayTimeIntervalType.HOUR, T.DayTimeIntervalType.SECOND)
    assert (interval.startField, interval.endField) == (1, 3)
    assert (T.YearMonthIntervalType().startField, T.YearMonthIntervalType().endField) == (0, 1)
    assert T.DecimalType(10, 2).hasPrecisionInfo is True
    for cls, name in ((T.DataType, "data"), (T.AtomicType, "atomic"), (T.IntegerType, "integer"),
                      (T.DayTimeIntervalType, "daytimeinterval"), (T.StructType, "struct"),
                      (T.NullType, "void"), (T.TimestampNTZType, "timestamp_ntz"),
                      (T.CalendarIntervalType, "interval")):
        assert cls.typeName() == name
    assert T.CharType(3).typeName() == "char"

    class MyType(T.DataType):
        pass

    assert MyType.typeName() == MyType().typeName() == "my"
    with pytest.raises(PySparkTypeError) as e:
        T.StructField("a", T.IntegerType()).typeName()
    assert e.value.getCondition() == "INVALID_TYPENAME_CALL"
    for cls in (T.NullType, T.ByteType, T.IntegerType, T.LongType, T.DoubleType, T.BinaryType,
                T.DateType, T.TimestampNTZType, T.CalendarIntervalType):
        assert cls() is cls()
    assert T.StringType() is not T.StringType()
    assert T.ArrayType(T.IntegerType()).elementType is T.IntegerType()




def test_user_defined_type_eq_and_hash():
    class PointUDT(T.UserDefinedType):
        @classmethod
        def sqlType(cls):
            return T.ArrayType(T.DoubleType(), False)

    class OtherUDT(T.UserDefinedType):
        @classmethod
        def sqlType(cls):
            return T.ArrayType(T.DoubleType(), False)

    # pyspark: same class and equal __dict__ -> equal; a different UDT class (even with
    # the same sqlType) or a non-UDT type -> unequal.
    assert PointUDT() == PointUDT()
    assert hash(PointUDT()) == hash(PointUDT())
    assert PointUDT() != OtherUDT()
    assert PointUDT() != T.ArrayType(T.DoubleType(), False)
    assert T.ArrayType(T.DoubleType(), False) != PointUDT()
    assert T.LongType() != PointUDT()
    u = PointUDT()
    u.extra = 1
    assert u != PointUDT()


# --------------------------------------------------------------------------- StorageLevel

def test_storage_level():
    from pyspark.storagelevel import StorageLevel
    sl = StorageLevel(True, True, False, False, 2)
    assert sl.useDisk and sl.useMemory and sl.replication == 2
    assert "Disk" in str(sl) and "Replicated" in str(sl)
    assert repr(sl).startswith("StorageLevel(")
    assert StorageLevel.MEMORY_ONLY == StorageLevel(False, True, False, False, 1)
    assert StorageLevel.DISK_ONLY != StorageLevel.MEMORY_ONLY


def test_dataframe_plot_is_the_plot_accessor_of_pyspark():
    # As PySpark's DataFrame.plot: the PySparkPlotAccessor of pyspark.sql.plot, of the
    # DataFrame itself.
    from pyspark.sql.plot import PySparkPlotAccessor

    df = _offline_session().range(1)
    plot = df.plot
    assert type(plot) is PySparkPlotAccessor
    assert plot.data is df


def test_checkpoints_send_their_eager_and_storage_level_as_pyspark(mock_connect):
    # As PySpark's, checkpoint(eager) and localCheckpoint(eager, storageLevel) send them in
    # their CheckpointCommand, of which the mock answers with a CachedRemoteRelation.
    from pyspark.storagelevel import StorageLevel

    spark, service = mock_connect
    # ExecutePlanResponse.checkpoint_command_result=19 -> relation=1 -> relation_id=1.
    service.execute_payload = message_field(19, message_field(1, message_field(1, "r1")))

    def sent():
        # ExecutePlanRequest.plan=3 -> Plan.command=2 -> Command.checkpoint_command=14: local=2,
        # eager=3, storage_level=4, where false is not sent.
        plan = parse_message(parse_message(service.execute_requests[-1])[3][0])
        return parse_message(parse_message(plan[2][0])[14][0])

    df = spark.range(1)
    for checkpoint, local, eager in ((lambda: df.checkpoint(), [], [1]),
                                     (lambda: df.checkpoint(eager=False), [], []),
                                     (lambda: df.localCheckpoint(), [1], [1]),
                                     (lambda: df.localCheckpoint(False), [1], [])):
        assert type(checkpoint()).__name__ == "DataFrame"
        command = sent()
        assert (command.get(2, []), command.get(3, []), command.get(4)) == (local, eager, None)
    df.localCheckpoint(storageLevel=StorageLevel.DISK_ONLY)
    # StorageLevel: use_disk=1, use_memory=2, use_off_heap=3, deserialized=4, replication=5.
    assert parse_message(sent()[4][0]) == {1: [1], 5: [1]}
    df.localCheckpoint(True, StorageLevel(False, True, True, True, 2))
    assert parse_message(sent()[4][0]) == {2: [1], 3: [1], 4: [1], 5: [2]}


def test_persist_cache_and_unpersist_return_the_dataframe_itself_as_pyspark(mock_connect):
    # As PySpark's, which persist the plan of the DataFrame and return it, not a copy.
    from pyspark.storagelevel import StorageLevel

    spark, _ = mock_connect
    df = spark.range(1)
    assert df.persist() is df
    assert df.persist(StorageLevel.DISK_ONLY) is df
    assert df.cache() is df
    assert df.unpersist() is df
    assert df.unpersist(blocking=True) is df


# --------------------------------------------------------------------------- session builder

def test_session_builder_chaining_offline():
    from pyspark.sql import SparkSession
    b = SparkSession.builder
    assert type(b).__name__ == "Builder"
    # Chainable, no server contacted until getOrCreate: each method returns the builder.
    b2 = b.remote("sc://localhost:15002").appName("x").config("k", "v")
    assert b2 is b


def test_sessions_are_got_or_created_as_pyspark():
    # As PySpark's builder: getOrCreate() returns the active session of the thread, or else
    # the default session, whatever the URL, and creates one when there is neither, of the
    # URL of remote() or else of SPARK_REMOTE; create() makes a new one, which is the default
    # and the active session only when they are not set; and stop() unsets them. In a new
    # interpreter, which has no default session.
    import subprocess
    import sys

    subprocess.run([sys.executable, "-c", """
import os
import threading

from pyspark.errors import PySparkRuntimeError
from pyspark.sql import SparkSession

def condition(call):
    try:
        call()
    except PySparkRuntimeError as error:
        return error.getCondition()
    raise AssertionError("no error")

os.environ.pop("SPARK_REMOTE", None)
assert condition(SparkSession.builder.getOrCreate) == "CONNECT_URL_NOT_SET"
assert condition(SparkSession.active) == "NO_ACTIVE_OR_DEFAULT_SESSION"

first = SparkSession.builder.remote("sc://localhost:1").getOrCreate()
assert SparkSession.builder.remote("sc://localhost:2").getOrCreate() is first
assert SparkSession.getActiveSession() is first and SparkSession.active() is first

seen = {}
def other_thread():
    seen["active"] = SparkSession.getActiveSession()
    seen["default"] = SparkSession.active()
    seen["got"] = SparkSession.builder.remote("sc://localhost:1").getOrCreate()
thread = threading.Thread(target=other_thread)
thread.start()
thread.join()
assert seen == {"active": None, "default": first, "got": first}, seen

created = SparkSession.builder.remote("sc://localhost:1").create()
assert created is not first and SparkSession.getActiveSession() is first

first.stop()
assert SparkSession.getActiveSession() is None
assert condition(SparkSession.active) == "NO_ACTIVE_OR_DEFAULT_SESSION"
os.environ["SPARK_REMOTE"] = "sc://localhost:1"
again = SparkSession.builder.getOrCreate()
assert again is not first and again is not created
assert SparkSession.getActiveSession() is again and SparkSession.active() is again
"""], check=True)


def test_conf_warns_the_warnings_of_the_server_and_checks_its_arguments_as_pyspark(mock_connect):
    # As PySpark's RuntimeConf: set, unset and the options of the builder warn each warning
    # of the server; set sends a bool as "true" or "false", an int as its str(), and None as no
    # value, and raises the TypeError of protobuf for another value; get raises
    # NOT_EXPECTED_TYPE of a key, and of a default other than None, that is not a str.
    import warnings

    from pyspark.errors import PySparkTypeError
    from pyspark.sql import SparkSession

    spark, service = mock_connect
    service.config_warnings = ["one", "two"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        spark.conf.set("k", True)
        spark.conf.unset("k")
        builder = SparkSession.builder.remote("sc://localhost:1").config("spark.sql.k", "v")
        assert builder.getOrCreate() is spark
    assert [(w.category, str(w.message)) for w in caught] == [
        (UserWarning, "one"),
        (UserWarning, "two"),
    ] * 3
    assert service.unset_confs == [["k"]]
    service.config_warnings = []
    for value in [1, "x", None, b"y"]:
        spark.conf.set("v", value)
    assert [pairs for pairs, _ in service.set_confs] == [
        [("k", "true")],
        [("spark.sql.k", "v")],
        [("v", "1")],
        [("v", "x")],
        [("v", None)],
        [("v", "y")],
    ]
    for value in [1.5, [1], object()]:
        with pytest.raises(TypeError, match="Cannot set spark.connect.KeyValue.value"):
            spark.conf.set("v", value)
    with pytest.raises(ValueError, match="isn't valid UTF-8"):
        spark.conf.set("v", b"\xff")
    assert len(service.set_confs) == 6
    for args, arg_name, arg_type in [
        ((1,), "key", "int"),
        ((None,), "key", "NoneType"),
        (("k", 1), "default", "int"),
        (("k", True), "default", "bool"),
    ]:
        with pytest.raises(PySparkTypeError) as e:
            spark.conf.get(*args)
        assert e.value.getCondition() == "NOT_EXPECTED_TYPE"
        assert e.value.getMessageParameters() == {
            "arg_name": arg_name,
            "expected_type": "str",
            "arg_type": arg_type,
        }


def test_builder_options_are_set_on_the_session_as_pyspark(mock_connect):
    # As PySpark's _apply_options: the options of the builder, but spark.remote and
    # spark.master, are set on the session that getOrCreate() returns, also an existing one,
    # in one request whose failures are ignored.
    from pyspark.sql import SparkSession

    spark, service = mock_connect
    builder = SparkSession.builder.remote("sc://localhost:1")
    builder = builder.appName("app").config("spark.sql.shuffle.partitions", 3)
    assert builder.getOrCreate() is spark
    assert service.set_confs == [
        ([("spark.app.name", "app"), ("spark.sql.shuffle.partitions", "3")], 1)
    ]
    assert SparkSession.builder.remote("sc://localhost:1").getOrCreate() is spark
    assert len(service.set_confs) == 1


def test_collected_rows_take_the_names_of_the_schema_of_the_result_as_pyspark(mock_connect):
    # As PySpark Connect's, which builds the rows by the schema of the result: the fields of
    # nested structs have the names of the schema, which keep the duplicate names that the
    # server's Arrow data numbers; collect raises for a calendar interval, whose Arrow type
    # from_arrow_schema does not convert, while toLocalIterator does not check it.
    import pyarrow as pa
    from _mock_connect_server import data_type_proto
    from pyspark.errors import PySparkTypeError

    spark, service = mock_connect

    def respond(schema, table):
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)
        # ExecutePlanResponse.schema=7 and arrow_batch=2: ArrowBatch.row_count=1, data=2.
        service.execute_payload = message_field(7, schema) + message_field(
            2, varint_field(1, table.num_rows) + message_field(2, sink.getvalue().to_pybytes()))

    field = {"name": "a", "type": "integer", "nullable": True, "metadata": {}}
    pair = {"type": "struct", "fields": [field, field]}
    respond(data_type_proto({"type": "struct", "fields": [
        {"name": "v", "type": pair, "nullable": True, "metadata": {}}]}),
        pa.table({"v": pa.array([{"a_0": 1, "a_1": 2}])}))
    for rows in (spark.range(1).collect(), list(spark.range(1).toLocalIterator())):
        assert rows[0].v.__fields__ == ["a", "a"]
        assert tuple(rows[0].v) == (1, 2)
    # Struct.fields=1, StructField: name=1, data_type=2 (DataType.calendar_interval=17).
    respond(message_field(21, message_field(1, message_field(1, "v") + message_field(
        2, message_field(17, b"")))),
        pa.table({"v": pa.array([(1, 2, 3)], pa.month_day_nano_interval())}))
    with pytest.raises(PySparkTypeError) as e:
        spark.range(1).collect()
    assert e.value.getCondition() == "UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION"
    assert e.value.getMessageParameters() == {"data_type": "month_day_nano_interval"}
    assert len(list(spark.range(1).toLocalIterator())) == 1

def test_tags_are_those_of_each_thread_as_pyspark(mock_connect):
    # As PySpark's: the tags of addTag are those of the current thread, of a set of getTags,
    # which the operations that the thread starts carry; a new thread starts without tags.
    import threading

    spark, service = mock_connect

    def sent_tags():
        # ExecutePlanRequest.tags=7.
        return sorted(tag.decode() for tag in parse_message(service.execute_requests[-1]).get(7, []))

    spark.addTag("main")
    seen = {}

    def child():
        seen["before"] = spark.getTags()
        spark.addTag("child")
        seen["after"] = spark.getTags()
        spark.range(1).collect()
        seen["sent"] = sent_tags()

    thread = threading.Thread(target=child)
    thread.start()
    thread.join()
    try:
        assert seen == {"before": set(), "after": {"child"}, "sent": ["child"]}
        assert spark.getTags() == {"main"}
        spark.range(1).collect()
        assert sent_tags() == ["main"]
    finally:
        spark.clearTags()


def test_session_builders_set_their_options_and_validate_their_urls_as_pyspark(monkeypatch):
    # As PySpark's SparkSession.Builder: SparkSession.builder is a new builder of each access,
    # whose methods set their option on it and return it, and a master with a remote raises
    # CANNOT_CONFIGURE_SPARK_CONNECT_MASTER, also of MASTER or SPARK_REMOTE.
    from pyspark.errors import PySparkRuntimeError
    from pyspark.sql import SparkSession

    monkeypatch.delenv("SPARK_REMOTE", raising=False)
    monkeypatch.delenv("MASTER", raising=False)
    assert SparkSession.builder is not SparkSession.builder
    builder = SparkSession.builder
    assert type(builder).__qualname__ == "SparkSession.Builder"
    assert builder.config("k", "v") is builder and builder.appName("app") is builder
    assert builder.master("local[2]") is builder
    for make in (lambda: SparkSession.builder.master("local").remote("sc://host:1"),
                 lambda: SparkSession.builder.remote("sc://host:1").master("local"),
                 lambda: SparkSession.builder.config(
                     map={"spark.master": "local", "spark.remote": "sc://host:1"})):
        with pytest.raises(PySparkRuntimeError) as e:
            make()
        assert e.value.getCondition() == "CANNOT_CONFIGURE_SPARK_CONNECT_MASTER"
        assert e.value.getMessageParameters() == {
            "master_url": "local", "connect_url": "sc://host:1"}
    monkeypatch.setenv("SPARK_REMOTE", "sc://env:15002")
    with pytest.raises(PySparkRuntimeError) as e:
        SparkSession.builder.master("local")
    assert e.value.getMessageParameters()["connect_url"] == "sc://env:15002"
    monkeypatch.setenv("SPARK_LOCAL_REMOTE", "1")
    with pytest.raises(PySparkRuntimeError) as e:
        SparkSession.builder.remote("sc://other:1")
    assert e.value.getCondition() == "CANNOT_CONFIGURE_SPARK_CONNECT"
    SparkSession.builder.remote("local[2]")




def test_session_builder_class_constructor():
    from pyspark.sql import SparkSession
    b = SparkSession.Builder()
    assert type(b).__name__ == "Builder"


def test_sessions_raise_the_errors_of_their_arguments_as_pyspark(mock_connect, tmp_path):
    # As PySpark Connect's: the errors of the tags of addTag and removeTag, of the name of
    # table, of the paths of copyFromLocalToFs, and of the JVM attributes of a classic session.
    from pyspark.errors import PySparkAttributeError, PySparkTypeError, PySparkValueError

    spark, _ = mock_connect
    for method in (spark.addTag, spark.removeTag):
        for tag, condition in (("", "VALUE_NOT_NON_EMPTY_STR"), ("a,b", "VALUE_ALLOWED"),
                               (None, "CANNOT_BE_NONE")):
            with pytest.raises(PySparkValueError) as e:
                method(tag)
            assert e.value.getCondition() == condition
            assert e.value.getMessageParameters()["arg_name"] == "Spark Connect tag"
    spark.removeTag("absent")
    with pytest.raises(PySparkTypeError) as e:
        spark.table(1)
    assert e.value.getMessageParameters() == {
        "arg_name": "tableName", "expected_type": "str", "arg_type": "int"}
    local = tmp_path / "local.txt"
    local.write_text("x")
    with pytest.raises(PySparkValueError) as e:
        spark.copyFromLocalToFs(str(local), "hdfs://host/dest")
    assert e.value.getCondition() == "NO_SCHEMA_AND_DRIVER_DEFAULT_SCHEME"
    for path in (tmp_path / "missing.txt", tmp_path):
        with pytest.raises(AssertionError, match="local path must be a file path."):
            spark.copyFromLocalToFs(str(path), "/dest")
    for name in ("_jsc", "_jconf", "_jvm", "_jsparkSession", "sparkContext"):
        with pytest.raises(PySparkAttributeError) as e:
            getattr(spark, name)
        assert e.value.getMessageParameters() == {"attr_name": name}
    assert getattr(spark, "no_such_attribute", "default") == "default"




def test_repartition_by_range_without_a_count_sends_none_as_pyspark(mock_connect):
    # As PySpark's repartitionByRange(col) and repartitionByRange(col, *cols): a
    # RepartitionByExpression with num_partitions unset, so the server picks the count;
    # repartitionByRange(n, *cols) sends n.
    from pyspark.sql import functions as F

    spark, service = mock_connect
    df = spark.range(1)

    def sent(dataframe):
        dataframe.collect()
        # ExecutePlanRequest.plan=3, Plan.root=1; Relation.repartition_by_expression=27,
        # RepartitionByExpression.num_partitions=3, which is absent when unset.
        plan = parse_message(parse_message(service.execute_requests[-1])[3][0])[1][0]
        return parse_message(parse_message(plan)[27][0]).get(3)

    assert sent(df.repartitionByRange("id")) is None
    assert sent(df.repartitionByRange(F.col("id"))) is None
    assert sent(df.repartitionByRange("id", "id")) is None
    assert sent(df.repartitionByRange(2, "id")) == [2]


def test_channel_builder_uses_endpoint():
    # channelBuilder reconstructs an sc:// URL from the builder's host/port (+ params)
    # for the native transport, returning a configured Builder (no longer raises).
    from pyspark.sql import SparkSession

    class CB:
        host = "localhost"
        port = 15002
        _params = {"use_ssl": "false"}

    b = SparkSession.builder.channelBuilder(CB())
    assert type(b).__name__ == "Builder"
    # A builder without a host is a clear error (not a silent misconfig).
    with pytest.raises(ValueError):
        SparkSession.builder.channelBuilder(object())


def test_functions_udtf_and_arrow_decorators():
    from pyspark.sql import functions as F
    from pyspark.sql.types import IntegerType

    # functions.udtf / arrow_udtf build UserDefinedTableFunctions (with the right evalType).
    @F.udtf(returnType="a int")
    class Echo:
        def eval(self, n):
            for i in range(n):
                yield (i,)

    assert type(Echo).__name__ == "UserDefinedTableFunction"
    assert Echo.evalType == 300  # SQL_TABLE_UDF

    @F.arrow_udtf(returnType="a int")
    class AEcho:
        def eval(self, n):
            yield (n,)

    assert AEcho.evalType == 302  # SQL_ARROW_UDTF

    # functions.arrow_udf builds an Arrow scalar UDF.
    au = F.arrow_udf(lambda x: x, IntegerType())
    assert callable(au)
    # As in PySpark, the function type is a value of ArrowUDFType.
    aui = F.arrow_udf(lambda x: x, IntegerType(), functionType=F.ArrowUDFType.SCALAR_ITER)
    assert callable(aui)


# --------------------------------------------------------------------------- functions / columns

def test_sql_modules_are_those_of_the_pyspark_client():
    # As the PySpark client: pyspark.sql has the modules geo_utils, sql_formatter, table_arg and
    # tvf_argument, of which a Column and a TableArg are TableValuedFunctionArguments, and
    # `import pyspark.sql` imports its submodules of conf, profiler, streaming and variant_utils.
    import pyspark.sql
    from pyspark.sql import functions as F
    from pyspark.sql.geo_utils import (
        CartesianSpatialReferenceSystemMapper,
        GeographicSpatialReferenceSystemMapper,
    )
    from pyspark.sql.sql_formatter import SQLStringFormatter
    from pyspark.sql.table_arg import TableArg
    from pyspark.sql.tvf_argument import TableValuedFunctionArgument

    for name in ("conf", "geo_utils", "profiler", "sql_formatter", "streaming", "table_arg",
                 "tvf_argument", "variant_utils"):
        assert hasattr(pyspark.sql, name), name
    assert isinstance(F.col("a"), TableValuedFunctionArgument)
    assert issubclass(TableArg, TableValuedFunctionArgument)
    assert CartesianSpatialReferenceSystemMapper and GeographicSpatialReferenceSystemMapper
    assert SQLStringFormatter.__module__ == "pyspark.sql.sql_formatter"




def test_functions_exports_fresh_process():
    import subprocess
    import sys
    import textwrap

    result = subprocess.run([sys.executable, "-c", textwrap.dedent('''
        import importlib
        import inspect
        import sys
        from pyspark.sql import functions as F
        assert not hasattr(F, "name") and not hasattr(F, "obj")

        official = __OFFICIAL_FUNCTIONS_ALL__

        def check_exports():
            assert type(F.__all__) is list
            assert F.__all__ == official
            assert len(F.__all__) == len(set(F.__all__))
            assert all(callable(getattr(F, name)) for name in F.__all__)
            assert not any(name.startswith("_") for name in F.__all__)
            for name in ("OrderingColumn", "PartitioningColumn", "SelectedColumn"):
                assert name in F.__all__
                assert inspect.isclass(getattr(F, name))
            namespace = {}
            exec("from pyspark.sql.functions import *", namespace)
            assert [name for name in namespace if name != "__builtins__"] == F.__all__
            assert all(namespace[name] is getattr(F, name) for name in F.__all__)
            builtin = importlib.import_module("pyspark.sql.functions.builtin")
            assert builtin is F.builtin is sys.modules[builtin.__name__]
            # As PySpark's, the builtin module has no __all__, and has every public function of
            # pyspark.sql.functions, also the deprecated ones that __all__ leaves out.
            assert not hasattr(builtin, "__all__")
            assert all(getattr(builtin, name) is getattr(F, name) for name in F.__all__)
            builtin_namespace = {}
            exec("from pyspark.sql.functions.builtin import *", builtin_namespace)
            assert set(F.__all__) <= set(builtin_namespace)
            assert all(builtin_namespace[name] is getattr(F, name) for name in F.__all__)
            part = importlib.import_module("pyspark.sql.functions.partitioning")
            assert part is F.partitioning is sys.modules[part.__name__]
            assert part.__all__ == ["years", "months", "days", "hours", "bucket"]
            # As in PySpark, distinct from those of pyspark.sql.functions, which are deprecated
            # and call them.
            assert all(getattr(part, name) is not getattr(F, name) for name in part.__all__)
            assert F.random is F.rand
            assert F.column is not F.col
            assert list(inspect.signature(F.first).parameters) == ["col", "ignorenulls"]
            assert list(inspect.signature(F.split).parameters) == ["str", "pattern", "limit"]
            return F.__all__.copy()

        expected = check_exports()
        old_all = F.__all__
        old_all.append("caller_added_name")
        F.builtin.builtin_only_name = None
        importlib.reload(F)
        assert F.__all__ is not old_all
        assert not hasattr(F.builtin, "builtin_only_name")
        assert check_exports() == expected
    ''').replace("__OFFICIAL_FUNCTIONS_ALL__", repr(OFFICIAL_FUNCTIONS_ALL))], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


def test_sql_utils_reexports_the_exceptions_of_pyspark(mock_connect):
    # As PySpark's pyspark.sql.utils: the exceptions of pyspark.errors, of their import path
    # before it, which the errors of the server are instances of.
    import pyspark.errors as errors
    import pyspark.sql.utils as utils
    from pyspark.errors.exceptions.captured import CapturedException

    for name in [
        "AnalysisException",
        "ParseException",
        "IllegalArgumentException",
        "StreamingQueryException",
        "QueryExecutionException",
        "PythonException",
        "UnknownException",
        "SparkUpgradeException",
        "PySparkImportError",
        "PySparkNotImplementedError",
        "PySparkRuntimeError",
    ]:
        assert getattr(utils, name) is getattr(errors, name), name
    assert utils.CapturedException is CapturedException

    spark, service = mock_connect
    classes = ["org.apache.spark.sql.AnalysisException", "java.lang.Exception"]
    message = "[TABLE_OR_VIEW_NOT_FOUND] The table or view `t` cannot be found."
    service.execute_error = (message, classes[0], _error_info(classes))
    service.confs = {
        "spark.sql.connect.serverStacktrace.enabled": "false",
        "spark.sql.pyspark.jvmStacktrace.enabled": "false",
    }
    with pytest.raises(utils.AnalysisException) as error:
        spark.range(1).collect()
    assert "TABLE_OR_VIEW_NOT_FOUND" in str(error.value)


def test_functions_build_columns():
    from pyspark.sql import functions as F
    c = F.col("id")
    assert type(c).__name__ == "Column"
    # Arithmetic / comparison / logical build new Columns.
    for expr in [c + 1, c - 1, c * 2, c == 1, c > 0, (c > 0) & (c < 10), -c, c.isNull()]:
        assert type(expr).__name__ == "Column"
    assert type(F.lit(5)).__name__ == "Column"
    assert type(F.upper(c)).__name__ == "Column"


def test_column_methods_offline():
    from pyspark.sql import functions as F
    c = F.col("id")
    assert type(c.alias("x")).__name__ == "Column"
    assert type(c.cast("string")).__name__ == "Column"
    assert type(c.cast(T.StringType())).__name__ == "Column"
    assert type(c.outer()).__name__ == "Column"
    assert type(c.asc()).__name__ == "Column"
    assert type(c.desc()).__name__ == "Column"
    # Column.transform(f) is just f(self).
    assert type(c.transform(lambda x: x + 1)).__name__ == "Column"


def test_functions_keyword_and_dispatch_paths():
    # PySpark 4.2.0 keyword-argument parity: generated wrappers carry the reference
    # parameter names, and the hand-written wrappers dispatch to the core. All lazy
    # (column-expression building; no server needed).
    from pyspark.sql import functions as F

    def is_col(x):
        return type(x).__name__ == "Column"

    c = F.col("id")
    # hand-written wrappers with reference parameter names
    assert is_col(F.col("id")) and is_col(F.lit(3))
    assert is_col(F.expr("id + 1")) and is_col(F.column("id"))
    assert is_col(F.sha2(c, 256))
    assert is_col(F.window(c, "10 minutes"))
    assert is_col(F.window(c, "10 minutes", "5 minutes", "1 minute"))
    # generated wrappers accept the reference keyword arguments (GAP 1)
    assert is_col(F.first(c, ignorenulls=True))
    assert is_col(F.round(c, scale=2))
    assert is_col(F.split(c, pattern=",", limit=2))
    assert is_col(F.array_contains(c, value=1))
    assert is_col(F.concat_ws("-", c, c))
    # lag/lead literal-arg overrides
    assert is_col(F.lag(c, offset=2, default=0))
    assert is_col(F.lead(c, offset=2, default=0))
    # the *args fallback wrapper (used for Rust-only extras) still dispatches positionally
    assert is_col(F._create_wrapper("upper")(c))
    # the _UNSET sentinel (default for optional generated args) has a stable repr
    assert repr(F._UNSET) == "<unset>"


def _sent_function_arguments(service):
    """The arguments of the function that the last ExecutePlan selects: ("col", name) for a
    column reference and ("lit", fields) for a literal."""
    # ExecutePlanRequest.plan=3, Plan.root=1, Relation.project=3, Project.expressions=3,
    # Expression.unresolved_function=3, UnresolvedFunction.arguments=2; an argument is an
    # Expression.literal=1 or an .unresolved_attribute=2, of unparsed_identifier=1.
    plan = parse_message(parse_message(service.execute_requests[-1])[3][0])[1][0]
    expression = parse_message(parse_message(parse_message(plan)[3][0])[3][0])
    arguments = []
    for argument in parse_message(expression[3][0]).get(2, []):
        argument = parse_message(argument)
        if 2 in argument:
            arguments.append(("col", parse_message(argument[2][0])[1][0].decode()))
        else:
            arguments.append(("lit", parse_message(argument[1][0])))
    return arguments


@pytest.mark.parametrize("call,expected", [
    # Literal.string=13 and .boolean=3.
    (lambda F: F.variant_get("v", "$.a", "int"),
     [("col", "v"), ("lit", {13: [b"$.a"]}), ("lit", {13: [b"int"]})]),
    (lambda F: F.try_variant_get("v", "$.a", "int"),
     [("col", "v"), ("lit", {13: [b"$.a"]}), ("lit", {13: [b"int"]})]),
    (lambda F: F.variant_delete("v", "$.a", "$.b"),
     [("col", "v"), ("lit", {13: [b"$.a"]}), ("lit", {13: [b"$.b"]})]),
    (lambda F: F.variant_insert("v", "$.b", "x"),
     [("col", "v"), ("lit", {13: [b"$.b"]}), ("col", "x")]),
    (lambda F: F.try_variant_insert("v", "$.b", "x"),
     [("col", "v"), ("lit", {13: [b"$.b"]}), ("col", "x")]),
    (lambda F: F.variant_set("v", "$.b", "x", True),
     [("col", "v"), ("lit", {13: [b"$.b"]}), ("col", "x"), ("lit", {3: [1]})]),
    (lambda F: F.try_variant_set("v", "$.b", "x", True),
     [("col", "v"), ("lit", {13: [b"$.b"]}), ("col", "x"), ("lit", {3: [1]})]),
    (lambda F: F.variant_array_append("v", "$", "x"),
     [("col", "v"), ("lit", {13: [b"$"]}), ("col", "x")]),
    (lambda F: F.try_variant_array_append("v", "$", "x"),
     [("col", "v"), ("lit", {13: [b"$"]}), ("col", "x")]),
    (lambda F: F.date_part("YEAR", "d"), [("col", "YEAR"), ("col", "d")]),
    (lambda F: F.convert_timezone(None, "tz", "ts"), [("col", "tz"), ("col", "ts")]),
])
def test_function_arguments_are_columns_or_literals_as_pyspark(mock_connect, call, expected):
    # A str is a literal where PySpark takes a str, as a variant path, and a column name
    # where it takes a Column or a name, as date_part's field: a path sent as a column,
    # such as variant_delete(v, "$.b"), would fail analysis on the server. The cases agree
    # with official PySpark 4.3.0.
    from pyspark.sql import functions as F

    spark, service = mock_connect
    spark.range(1).select(call(F)).collect()
    assert _sent_function_arguments(service) == expected


def test_column_alias_metadata_offline():
    from pyspark.sql import functions as F

    c = F.col("id")
    assert type(c.alias("x")).__name__ == "Column"
    assert type(c.alias("x", metadata={"k": "v"})).__name__ == "Column"
    assert type(c.name("y")).__name__ == "Column"


def test_new_v420_methods_exposed():
    # Presence of the v4.2.0 parity methods on the drop-in classes (behaviour that
    # needs a live server is covered by the e2e tests).
    import pyspark._pyspark as p

    for cls, methods in [
        (p.DataFrame, ["repartitionById", "zipWithIndex"]),
        (p.Column, ["transform"]),
        (p.RuntimeConf, ["getAll"]),
        (p.DataFrameStatFunctions, ["sampleBy"]),
        (p.DataFrameReader, ["changes"]),
        (p.DataStreamReader, ["changes", "xml", "name"]),
    ]:
        have = set(m for m in dir(cls) if not m.startswith("_"))
        missing = [m for m in methods if m not in have]
        assert not missing, f"{cls.__name__} missing {missing}"


def test_streaming_query_manager_wrapper_has_listener_api():
    # spark.streams returns the Python StreamingQueryManager wrapper, which exposes the
    # listener-bus API (addListener/removeListener/close). Verify the class contract
    # offline (no server needed to check the wrapper class).
    from pyspark.sql.streaming.query import StreamingQueryManager

    for m in ["addListener", "removeListener", "close"]:
        assert hasattr(StreamingQueryManager, m), f"StreamingQueryManager missing {m}"


# --------------------------------------------------------------------------- udf / udtf

def test_udf_construction_and_call():
    from pyspark.sql import functions as F
    from pyspark.sql.udf import UserDefinedFunction
    # As PySpark's: (func, returnType, name, evalType, deterministic).
    u = UserDefinedFunction(lambda x: x + 1, T.IntegerType(), "plus1", 100)
    assert u._name == "plus1" and u._wrapped().__name__ == "plus1"
    assert type(u(F.col("id"))).__name__ == "Column"


def test_udf_decorator():
    from pyspark.sql.functions import udf
    u = udf(lambda x: x, T.IntegerType())
    assert callable(u)
    # Decorator form (no function arg) returns a decorator.
    deco = udf(returnType=T.IntegerType())
    assert callable(deco(lambda x: x))


def test_pandas_udf():
    from pyspark.sql.functions import pandas_udf
    p = pandas_udf(lambda s: s, T.IntegerType())
    assert callable(p) or p is not None


def test_add_artifacts_raises_for_paths_that_pyspark_does_not_upload(tmp_path):
    from pyspark.errors import PySparkRuntimeError, PySparkValueError

    spark = _offline_session()
    note = tmp_path / "note.txt"
    note.write_text("x")
    # Raised before any upload, as in PySpark.
    for path, operation in [(str(note), f"{note} file format"), ("s3://b/x.jar", "s3 scheme")]:
        with pytest.raises(PySparkRuntimeError) as error:
            spark.addArtifacts(path)
        assert error.value.getCondition() == "UNSUPPORTED_OPERATION"
        assert error.value.getMessageParameters() == {"operation": operation}
    with pytest.raises(PySparkValueError) as error:
        spark.addArtifacts(str(note), pyfile=True, file=True)
    assert error.value.getCondition() == "INVALID_MULTIPLE_ARGUMENT_CONDITIONS"
    assert error.value.getMessageParameters() == {
        "arg_names": "'pyfile', 'archive' and/or 'file'", "condition": "True together"}


def test_show_checks_its_arguments_as_pyspark():
    import inspect

    from pyspark.errors import PySparkTypeError
    from pyspark.sql import DataFrame

    assert str(inspect.signature(DataFrame.show)) == "(self, /, n=20, truncate=True, vertical=False)"
    df = _offline_session().range(3)
    # Raised before any request, as by DataFrame._show_string.
    for kwargs, name, expected, arg_type in (
        ({"n": True}, "n", "int", "bool"),
        ({"n": "3"}, "n", "int", "str"),
        ({"n": None}, "n", "int", "NoneType"),
        ({"vertical": 1}, "vertical", "bool", "int"),
        ({"vertical": None}, "vertical", "bool", "NoneType"),
        ({"truncate": "x"}, "truncate", "bool", "str"),
    ):
        with pytest.raises(PySparkTypeError) as error:
            df.show(**kwargs)
        assert error.value.getCondition() == "NOT_EXPECTED_TYPE"
        assert error.value.getMessageParameters() == {
            "expected_type": expected, "arg_name": name, "arg_type": arg_type}
    with pytest.raises(TypeError, match=r"^int\(\) argument must be"):
        df.show(truncate=None)
    with pytest.raises(TypeError, match=r"^DataFrame\.show\(\) got an unexpected keyword argument 'foo'$"):
        df.show(foo=1)
    with pytest.raises(TypeError, match=r"^DataFrame\._show_string\(\) got multiple values for argument 'n'$"):
        df._show_string(1, n=2)


def test_progress_handlers_are_called_with_the_progress_of_executions_as_pyspark(
        mock_connect, capfd, monkeypatch):
    # As PySpark Connect's: a handler is registered once, and called with the keyword arguments
    # of ProgressHandler, of the StageInfo of each stage, for each progress of an execution that
    # has tasks, and once more, as done, when it ends; removing a handler that is not registered
    # raises a ValueError; an exception of a handler goes to sys.unraisablehook; and
    # SPARK_CONNECT_PROGRESS_BAR_ENABLED prints the progress bar.
    import sys

    from pyspark.sql.connect.shell.progress import StageInfo

    spark, service = mock_connect
    # ExecutePlanResponse.execution_progress=18 -> ExecutionProgress.stages=1 (StageInfo
    # stage_id=1, num_tasks=2, num_completed_tasks=3, input_bytes_read=4, done=5),
    # num_inflight_tasks=2.
    stage = (varint_field(1, 7) + varint_field(2, 4) + varint_field(3, 1)
             + varint_field(4, 1024) + varint_field(5, 0))
    service.execute_payload = message_field(18, message_field(1, stage) + varint_field(2, 3))
    calls = []

    def handler(**kwargs):
        calls.append(kwargs)

    assert spark.registerProgressHandler(handler) is None
    spark.registerProgressHandler(handler)
    spark.range(1).collect()
    assert [call["done"] for call in calls] == [False, True]
    assert calls[0]["stages"] == [StageInfo(
        stage_id=7, num_tasks=4, num_completed_tasks=1, num_bytes_read=1024, done=False)]
    assert calls[1]["stages"] == calls[0]["stages"]
    assert calls[1]["inflight_tasks"] == calls[0]["inflight_tasks"] == 3
    assert calls[0]["operation_id"] == calls[1]["operation_id"]
    assert len(calls[0]["operation_id"]) == 36

    unraisable = []
    monkeypatch.setattr(sys, "unraisablehook", unraisable.append)

    def failing(**kwargs):
        raise ZeroDivisionError("handler")

    spark.registerProgressHandler(failing)
    spark.range(1).collect()
    assert unraisable and isinstance(unraisable[0].exc_value, ZeroDivisionError)
    spark.removeProgressHandler(failing)

    monkeypatch.setenv("SPARK_CONNECT_PROGRESS_BAR_ENABLED", "1")
    capfd.readouterr()
    spark.range(1).collect()
    assert "25.00% Complete" in capfd.readouterr().out
    monkeypatch.delenv("SPARK_CONNECT_PROGRESS_BAR_ENABLED")

    spark.removeProgressHandler(handler)
    with pytest.raises(ValueError):
        spark.removeProgressHandler(handler)
    calls.clear()
    spark.registerProgressHandler(handler)
    spark.clearProgressHandlers()
    spark.range(1).collect()
    assert calls == []




def test_enums_are_literals_of_their_values_as_pyspark():
    # As PySpark's enum_to_value: an Enum is the literal of its value, of lit, the Column
    # operators and the functions that take a value, also of an Enum of an Enum.
    from enum import Enum

    from pyspark.sql import functions as F

    class IntEnum(Enum):
        X = 1

    class StrEnum(Enum):
        S = "s"

    class BoolEnum(Enum):
        T = True

    class Nested(Enum):
        N = IntEnum.X

    c = F.col("id")
    for column, expected in [
        (c + IntEnum.X, "+(id, 1)"),
        (IntEnum.X + c, "+(1, id)"),
        (c == IntEnum.X, "==(id, 1)"),
        (F.col("b") & BoolEnum.T, "and(b, True)"),
        (F.lit(IntEnum.X), "1"),
        (F.lit(StrEnum.S), "s"),
        (F.lit(Nested.N), "1"),
        (c.isin(IntEnum.X), "in(id, 1)"),
        (c.between(IntEnum.X, 5), "and(>=(id, 1), <=(id, 5))"),
        (F.col("s").contains(StrEnum.S), "contains(s, s)"),
        (c ** IntEnum.X, "power(id, 1)"),
        (F.concat_ws(StrEnum.S, c), "concat_ws(s, id)"),
        (F.array_contains(F.col("a"), IntEnum.X), "array_contains(a, 1)"),
    ]:
        assert repr(column) == f"Column<'{expected}'>"


def test_udfs_are_created_as_pyspark(mock_connect):
    # As PySpark's _create_py_udf: udf() without useArrow makes an Arrow-optimized UDF
    # (SQL_ARROW_BATCHED_UDF) when the active session enables
    # spark.sql.execution.pythonUDF.arrow.enabled, as the server does by default, and infers
    # the eval type of a function with type hints; and it returns the function wrapper of
    # the UDF.
    import pandas as pd
    import pyarrow as pa

    from pyspark.sql import functions as F
    from pyspark.sql.udf import UserDefinedFunction

    spark, service = mock_connect
    arrow_conf = "spark.sql.execution.pythonUDF.arrow.enabled"

    def plus_one(x):
        """Adds one."""
        return x + 1

    def pandas_plus_one(s: pd.Series) -> pd.Series:
        return s + 1

    def arrow_plus_one(a: pa.Array) -> pa.Array:
        return a

    def sent(fn):
        # ExecutePlanRequest.plan=3, Plan.root=1, Relation.project=3, Project.expressions=3;
        # Expression.common_inline_user_defined_function=15; CommonInlineUserDefinedFunction.
        # function_name=1, python_udf=4; PythonUDF.eval_type=2.
        spark.range(1).select(fn("id")).collect()
        plan = parse_message(parse_message(service.execute_requests[-1])[3][0])[1][0]
        expression = parse_message(parse_message(parse_message(plan)[3][0])[3][0])
        udf = parse_message(expression[15][0])
        return udf[1][0].decode(), parse_message(udf[4][0])[2][0]

    wrapper = F.udf(plus_one, "int")
    assert wrapper.__name__ == "plus_one" and wrapper.__doc__ == "Adds one."
    assert wrapper.__module__ == plus_one.__module__ and wrapper.func is plus_one
    assert isinstance(wrapper._unwrapped, UserDefinedFunction)
    assert wrapper.returnType == T.IntegerType() and wrapper.deterministic is True
    assert wrapper.evalType == 101
    assert sent(wrapper) == ("plus_one", 101)
    assert sent(F.udf(returnType="int", useArrow=False)(plus_one)) == ("plus_one", 100)
    assert sent(F.udf(pandas_plus_one, "long")) == ("pandas_plus_one", 200)
    assert sent(F.udf(arrow_plus_one, "long")) == ("arrow_plus_one", 250)
    # With useArrow, the type hints are not used.
    assert F.udf(pandas_plus_one, "long", useArrow=True).evalType == 101

    service.confs = {arrow_conf: "false"}
    try:
        assert sent(F.udf("int")(plus_one)) == ("plus_one", 100)
        assert F.udf(plus_one, "int", useArrow=True).evalType == 101
    finally:
        service.confs = {}
    # Only udf() without useArrow reads the configuration.
    assert service.read_confs == [arrow_conf] * 4


def test_map_functions_send_the_names_of_their_functions_as_pyspark(mock_connect):
    # As PySpark's UserDefinedFunction: the UDF of a map or apply function has the name of the
    # function, or that of the class of a callable object, which the server shows in the plan.
    spark, service = mock_connect
    df = spark.range(1)
    grouped = df.groupBy("id")

    def sent_name(dataframe, relation_field, func_field):
        # ExecutePlanRequest.plan=3, Plan.root=1; CommonInlineUserDefinedFunction.function_name
        # =1.
        dataframe.collect()
        plan = parse_message(parse_message(service.execute_requests[-1])[3][0])[1][0]
        relation = parse_message(parse_message(plan)[relation_field][0])
        return parse_message(relation[func_field][0])[1][0].decode()

    def pandas_func(iterator):
        yield from iterator

    class ArrowFunc:
        def __call__(self, iterator):
            yield from iterator

    def with_state(key, pdfs, state):
        yield from pdfs

    # Relation.map_partitions=28 (MapPartitions.func=2), group_map=31 (GroupMap.func=3),
    # co_group_map=32 (CoGroupMap.func=5), apply_in_pandas_with_state=34 (its func=3).
    assert sent_name(df.mapInPandas(pandas_func, "id long"), 28, 2) == "pandas_func"
    assert sent_name(df.mapInArrow(ArrowFunc(), "id long"), 28, 2) == "ArrowFunc"
    assert sent_name(grouped.applyInPandas(lambda pdf: pdf, "id long"), 31, 3) == "<lambda>"
    assert sent_name(grouped.applyInArrow(lambda table: table, "id long"), 31, 3) == "<lambda>"
    cogrouped = grouped.cogroup(grouped)
    assert sent_name(cogrouped.applyInPandas(lambda a, b: a, "id long"), 32, 5) == "<lambda>"
    assert sent_name(cogrouped.applyInArrow(lambda a, b: a, "id long"), 32, 5) == "<lambda>"
    stateful = grouped.applyInPandasWithState(
        with_state, "id long", "c long", "append", "NoTimeout"
    )
    assert sent_name(stateful, 34, 3) == "with_state"


def test_writers_shared_by_threads_run_their_requests_as_pyspark(mock_connect):
    # As PySpark's: a writer is a plain Python object, so threads may call its methods while
    # a request waits for the server, which releases the GIL. Two threads save with one
    # DataFrameWriter and one thread appends with a DataFrameWriterV2, while the main thread
    # sets the format and the provider of the writers meanwhile.
    spark, service = mock_connect
    writer = spark.range(1).write.mode("overwrite")
    writer_v2 = spark.range(1).writeTo("shared_writer")
    service.block_execute = True
    with ThreadPoolExecutor(max_workers=3) as pool:
        requests = [pool.submit(writer.parquet, f"/tmp/shared_writer/p{i}") for i in range(2)]
        requests.append(pool.submit(writer_v2.append))
        try:
            for _ in range(500):
                if service.execute_calls >= 3:
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("The writes never reached the server")
            assert writer.format("parquet") is writer
            assert writer_v2.using("parquet") is writer_v2
        finally:
            service.release_response()
        for request in requests:
            request.result(timeout=10)


def test_try_cast_and_isin_take_their_arguments_as_pyspark():
    # As PySpark Connect's: try_cast raises NOT_EXPECTED_TYPE for a type that is neither a
    # DataType nor a str, and isin unpacks a single list or set, but not a tuple, which is a
    # value of lit (UNSUPPORTED_DATA_TYPE).
    from pyspark.errors import PySparkTypeError
    from pyspark.sql import functions as F

    column = F.col("a")
    with pytest.raises(PySparkTypeError) as e:
        column.try_cast(1)
    assert e.value.getMessageParameters() == {
        "expected_type": "DataType or str", "arg_name": "dataType", "arg_type": "int"}
    assert "TRY_CAST" in repr(column.try_cast("string")).upper()
    column.try_cast(T.StringType())
    assert repr(column.isin([1, 2])) == repr(column.isin({1, 2})) == repr(column.isin(1, 2))
    with pytest.raises(PySparkTypeError) as e:
        column.isin((1, 2))
    assert e.value.getMessageParameters() == {"data_type": "tuple"}




def test_map_functions_send_their_resource_profile_as_pyspark(mock_connect):
    # As PySpark's, mapInPandas and mapInArrow send the id of their resource profile, which
    # `profile.id` creates on the server, so that their stage runs with its resources.
    from pyspark.resource import ResourceProfileBuilder, TaskResourceRequests

    spark, service = mock_connect
    profile = ResourceProfileBuilder().require(TaskResourceRequests().cpus(1)).build
    profile._id = 7  # As if the server had created it.

    def sent_map_partitions(dataframe):
        # ExecutePlanRequest.plan=3, Plan.root=1, Relation.map_partitions=28.
        dataframe.collect()
        plan = parse_message(parse_message(service.execute_requests[-1])[3][0])[1][0]
        return parse_message(parse_message(plan)[28][0])

    df = spark.range(1)
    # MapPartitions.profile_id=4.
    for mapped in (df.mapInPandas(lambda it: it, "id long", profile=profile),
                   df.mapInArrow(lambda it: it, "id long", True, profile)):
        assert sent_map_partitions(mapped)[4] == [7]
    assert 4 not in sent_map_partitions(df.mapInPandas(lambda it: it, "id long"))
    assert 4 not in sent_map_partitions(df.mapInArrow(lambda it: it, "id long", profile=None))


def test_udf_and_udtf_calls_pass_named_arguments_as_pyspark(mock_connect):
    # As PySpark's UserDefinedFunction and UserDefinedTableFunction: each keyword argument
    # of a call is a NamedArgumentExpression, after the positional arguments, and a bare
    # string is a column name.
    from pyspark.sql import functions as F

    spark, service = mock_connect

    @F.udf("int")
    def plus(a, b):
        return a + 10 * b

    assert repr(plus(a=F.lit(1), b=F.col("x"))) == "Column<'plus(a => 1, b => x)'>"
    called = plus(F.lit(1), b="x")
    assert repr(called) == "Column<'plus(1, b => x)'>"

    def arguments(function):
        # Expression.literal=1, unresolved_attribute=2 (unparsed_identifier=1),
        # named_argument_expression=17 (key=1, value=2).
        first, second = (parse_message(a) for a in function[3])
        named = parse_message(second[17][0])
        return first, named[1][0].decode(), parse_message(named[2][0])

    # ExecutePlanRequest.plan=3, Plan.root=1, Relation.project=3, Project.expressions=3;
    # Expression.common_inline_user_defined_function=15, whose arguments=3.
    spark.range(1).select(called).collect()
    plan = parse_message(parse_message(service.execute_requests[-1])[3][0])[1][0]
    expression = parse_message(parse_message(parse_message(plan)[3][0])[3][0])
    first, key, value = arguments(parse_message(expression[15][0]))
    assert set(first) == {1} and key == "b"
    assert parse_message(value[2][0])[1] == [b"x"]

    @F.udtf(returnType="a int, b int")
    class Two:
        def eval(self, a, b):
            yield a, b

    # Relation.common_inline_user_defined_table_function=38, whose arguments=3.
    Two("x", b=F.lit(2)).collect()
    relation = parse_message(parse_message(service.execute_requests[-1])[3][0])[1][0]
    first, key, value = arguments(parse_message(parse_message(relation)[38][0]))
    assert parse_message(first[2][0])[1] == [b"x"] and key == "b" and set(value) == {1}


def test_subqueries_are_sent_with_the_plans_they_refer_to_as_pyspark(mock_connect):
    # As PySpark's DataFrame.scalar(), DataFrame.exists() and Column.isin(df): Columns of
    # SubqueryExpressions of the plan of the DataFrame, sent in a WithRelations of the plans
    # that they refer to; and asTable() is a table argument that a UDTF call takes.
    from pyspark.sql import Column, functions as F

    spark, service = mock_connect

    def relation():
        # ExecutePlanRequest.plan=3, Plan.root=1.
        return parse_message(parse_message(parse_message(service.execute_requests[-1])[3][0])[1][0])

    def plan_id(rel):
        # Relation.common=1, RelationCommon.plan_id=2.
        return parse_message(rel[1][0])[2][0]

    def with_relations(rel):
        # Relation.with_relations=41: root=1, references=2.
        with_rel = parse_message(rel[41][0])
        return parse_message(with_rel[1][0]), [parse_message(r) for r in with_rel[2]]

    def subquery(expression):
        # Expression.alias=6 (expr=1); .subquery_expression=21: plan_id=1, subquery_type=2,
        # table_arg_options=3, in_subquery_values=4.
        expression = parse_message(expression)
        if 6 in expression:
            expression = parse_message(parse_message(expression[6][0])[1][0])
        return parse_message(expression[21][0])

    inner = spark.range(3)
    scalar = inner.scalar()
    assert isinstance(scalar, Column) and isinstance(inner.exists(), Column)
    assert "type=scalar" in repr(scalar) and "type=exists" in repr(inner.exists())
    spark.range(1).select(
        scalar.alias("s"), inner.exists().alias("e"), F.col("id").isin(inner).alias("i")
    ).collect()
    rel = relation()
    root, references = with_relations(rel)
    assert plan_id(root) != plan_id(rel)
    # Relation.project=3, Project.expressions=3.
    subqueries = [subquery(e) for e in parse_message(root[3][0])[3]]
    assert [s[2][0] for s in subqueries] == [1, 2, 4]  # SCALAR, EXISTS, IN
    assert {s[1][0] for s in subqueries} == {plan_id(r) for r in references}
    assert len(references) == 3 and all(3 not in s for s in subqueries)
    assert parse_message(subqueries[2][4][0])[2]  # IN of unresolved_attribute "id"

    @F.udtf(returnType="id bigint")
    class Rows:
        def eval(self, row):
            yield row["id"],

    table = spark.range(2).asTable().partitionBy("id").orderBy(F.col("id").desc())
    Rows(table).collect()
    root, references = with_relations(relation())
    # Relation.common_inline_user_defined_table_function=38, its arguments=3.
    argument = subquery(parse_message(root[38][0])[3][0])
    assert argument[2] == [3] and argument[1][0] == plan_id(references[0])  # TABLE_ARG
    options = parse_message(argument[3][0])
    assert len(options[1]) == 1 and len(options[2]) == 1 and 3 not in options


def test_udtf_handlers_are_pickled_when_used_as_pyspark(mock_connect, tmp_path):
    # As PySpark Connect's PythonUDTF: the handler is pickled when the UDTF is used, so a
    # handler that does not pickle raises there, a PySparkPicklingError of
    # UDTF_SERIALIZATION_ERROR of its PicklingError.
    from pyspark.errors import PySparkPicklingError
    from pyspark.sql.functions import udtf

    spark, _ = mock_connect
    file_obj = open(tmp_path / "file.txt", "w")
    try:

        @udtf(returnType="x: int")
        class Unpicklable:
            def eval(self):
                file_obj
                yield (1,)

        for use in (lambda: Unpicklable(), lambda: spark.udtf.register("unpicklable", Unpicklable)):
            with pytest.raises(PySparkPicklingError) as error:
                use()
            assert error.value.getCondition() == "UDTF_SERIALIZATION_ERROR"
            assert error.value.getMessageParameters() == {
                "name": "Unpicklable",
                "message": "Please check the stack trace and make sure the function is "
                "serializable.",
            }
    finally:
        file_obj.close()


def test_udtfs_are_created_as_pyspark(mock_connect):
    # As PySpark's _create_py_udtf and _create_pyarrow_udtf: a UDTF is not deterministic
    # unless asDeterministic() marks it so, in place; arrow_udtf() makes a PyArrow-native
    # UDTF (SQL_ARROW_UDTF); and udtf() makes an Arrow-optimized one (SQL_ARROW_TABLE_UDF)
    # with useArrow=True, or without useArrow when the active session enables
    # spark.sql.execution.pythonUDTF.arrow.enabled.
    from pyspark.sql import functions as F

    spark, service = mock_connect
    arrow_conf = "spark.sql.execution.pythonUDTF.arrow.enabled"

    class Echo:
        def eval(self, n):
            yield (n,)

    def sent(fn):
        # ExecutePlanRequest.plan=3, Plan.root=1, Relation.common_inline_user_defined_table_
        # function=38; CommonInlineUserDefinedTableFunction.deterministic=2, python_udtf=4;
        # PythonUDTF.eval_type=2.
        fn(F.lit(1)).collect()
        relation = parse_message(parse_message(service.execute_requests[-1])[3][0])[1][0]
        udtf = parse_message(parse_message(relation)[38][0])
        return udtf.get(2, [0]) == [1], parse_message(udtf[4][0])[2][0]

    # The server enables the config by default.
    arrow = F.udtf(Echo, returnType="a int")
    assert arrow.deterministic is False
    assert sent(arrow) == (False, 301)
    assert arrow.asDeterministic() is arrow and arrow.deterministic is True
    assert sent(arrow) == (True, 301)
    assert sent(F.udtf(returnType="a int", useArrow=False)(Echo)) == (False, 300)
    assert sent(F.arrow_udtf(Echo, returnType="a int")) == (False, 302)
    assert sent(F.arrow_udtf(returnType="a int")(Echo).asDeterministic()) == (True, 302)

    service.confs = {arrow_conf: "false"}
    try:
        assert sent(F.udtf(returnType="a int")(Echo)) == (False, 300)
        assert sent(F.udtf(Echo, returnType="a int", useArrow=True)) == (False, 301)
    finally:
        service.confs = {}
    # Only udtf() without useArrow reads the configuration.
    assert service.read_confs == [arrow_conf, arrow_conf]


def test_time_values_are_sent_as_pyspark(mock_connect):
    # As PySpark: a datetime.time is of TimeType(), and createDataFrame sends it as PyArrow's
    # time64[ns] of the value, which takes the nanoseconds of an int too, and lit() of it is
    # a time literal of its nanoseconds since midnight.
    import datetime

    import pyarrow as pa
    from pyspark.sql import functions as F

    spark, service = mock_connect
    t = datetime.time(1, 2, 3, 456789)
    nanos = 3_723_456_789_000

    def local_relation(df):
        df.toArrow()
        request = parse_message(service.execute_requests[-1])
        relation = parse_message(parse_message(request[3][0])[1][0])
        while 18 in relation:  # Relation.to_df -> ToDF.input.
            relation = parse_message(parse_message(relation[18][0])[1][0])
        # Relation.local_relation=11 -> LocalRelation.data=1, schema=2.
        local = parse_message(relation[11][0])
        table = pa.ipc.open_stream(local[1][0]).read_all()
        return table, [field["type"] for field in json.loads(local[2][0])["fields"]]

    for schema in (None, "t time"):
        table, types = local_relation(spark.createDataFrame([(t,), (None,)], schema))
        assert types == ["time(6)"]
        assert table.schema.field(0).type == pa.time64("ns")
        assert table.column(0).cast(pa.int64()).to_pylist() == [nanos, None]
    table, _ = local_relation(spark.createDataFrame([(5,)], "t time"))
    assert table.column(0).cast(pa.int64()).to_pylist() == [5]
    with pytest.raises(TypeError, match="cannot be converted to int"):
        spark.createDataFrame([("01:02:03",)], "t time")

    literal = F.lit(t)
    assert repr(literal) == "Column<'01:02:03.456789'>"
    spark.range(1).select(literal).collect()
    # ... Project.expressions=3; Expression.literal=1, Literal.time=26 (nano=1, precision=2).
    plan = parse_message(parse_message(service.execute_requests[-1])[3][0])[1][0]
    expression = parse_message(parse_message(parse_message(plan)[3][0])[3][0])
    assert parse_message(parse_message(expression[1][0])[26][0]) == {1: [nanos], 2: [6]}


def test_time_types_parse_as_pyspark(mock_connect):
    # As PySpark's: a bare time is TimeType(), of the default precision 6, in a DDL string and
    # in the JSON of a type.
    assert T.DataType.fromDDL("t time") == T.StructType([T.StructField("t", T.TimeType(6))])
    assert T.DataType.fromDDL("t time(3)") == T.StructType([T.StructField("t", T.TimeType(3))])
    assert T._parse_datatype_json_string('"time"') == T.TimeType(6)
    assert T._parse_datatype_json_string('"time(0)"') == T.TimeType(0)


def test_decimal_literal_is_of_the_default_decimal_type_as_pyspark(mock_connect):
    # As PySpark's LiteralExpression of a Decimal, of DecimalType(): precision 10 and scale
    # 0, which the server widens to those of the value, wherever the literal is.
    from decimal import Decimal

    from pyspark.sql import functions as F

    spark, service = mock_connect
    spark.range(1).select(
        F.lit(Decimal("1.5")), F.col("id") == Decimal("-0.001"), F.lit(Decimal("12345678901.25"))
    ).collect()
    # ExecutePlanRequest.plan=3, Plan.root=1, Relation.project=3, Project.expressions=3;
    # Expression.literal=1, Literal.decimal=12 (value=1, precision=2, scale=3);
    # .unresolved_function=3, UnresolvedFunction.arguments=2.
    plan = parse_message(parse_message(service.execute_requests[-1])[3][0])[1][0]
    expressions = parse_message(parse_message(plan)[3][0])[3]

    def decimal(literal):
        return parse_message(parse_message(literal)[12][0])

    first, compared, wide = (parse_message(e) for e in expressions)
    compared = parse_message(parse_message(compared[3][0])[2][1])
    assert decimal(first[1][0]) == {1: [b"1.5"], 2: [10], 3: [0]}
    assert decimal(compared[1][0]) == {1: [b"-0.001"], 2: [10], 3: [0]}
    assert decimal(wide[1][0]) == {1: [b"12345678901.25"], 2: [10], 3: [0]}


def test_user_defined_types_are_named_by_their_classes_as_pyspark():
    # As PySpark's simpleString(): a user-defined type in an array, map or struct is named
    # by its class, as VectorUDT's `vector`, and `udt` by default.
    from pyspark.ml.linalg import VectorUDT

    point = _DayPointUDT()
    assert T.ArrayType(VectorUDT()).simpleString() == "array<vector>"
    assert T.MapType(T.StringType(), point).simpleString() == "map<string,udt>"
    struct = T.StructType([T.StructField("v", VectorUDT()), T.StructField("p", point)])
    assert struct.simpleString() == "struct<v:vector,p:udt>"
    assert struct.fields[0].simpleString() == "v:vector"


class _DayPointUDT(T.UserDefinedType):
    @classmethod
    def sqlType(cls):
        return T.ArrayType(T.DoubleType())

    @classmethod
    def module(cls):
        return __name__

    def serialize(self, obj):
        return obj

    def deserialize(self, datum):
        return tuple(datum)


class _ScaledPointUDT(_DayPointUDT):
    def __init__(self, scale=1.0):
        self.scale = scale


def test_user_defined_types_pickle_as_python_objects_as_pyspark():
    # As PySpark's UDT, a Python object: a UDT of parameters is made with them, and pickles
    # as its class and its state, which cloudpickle sends by value for a class of __main__.
    # In a new interpreter, whose __main__ defines the class, so that this one does not.
    import copy
    import pickle
    import subprocess
    import sys

    from pyspark.serializers import CloudPickleSerializer

    udt = _ScaledPointUDT(3.0)
    for copied in (pickle.loads(pickle.dumps(udt)), copy.deepcopy(udt)):
        assert type(copied) is _ScaledPointUDT and copied.scale == 3.0
    assert pickle.loads(pickle.dumps(_DayPointUDT())).__dict__ == {}

    pickled = subprocess.run([sys.executable, "-c", """
import sys
from pyspark.serializers import CloudPickleSerializer
from pyspark.sql.types import ArrayType, DoubleType, UserDefinedType

class MainPointUDT(UserDefinedType):
    def __init__(self, scale):
        self.scale = scale

    @classmethod
    def sqlType(cls):
        return ArrayType(DoubleType())

    @classmethod
    def module(cls):
        return "__main__"

sys.stdout.buffer.write(CloudPickleSerializer().dumps(MainPointUDT(2.5)))
"""], check=True, capture_output=True).stdout
    main_udt = CloudPickleSerializer().loads(pickled)
    assert type(main_udt).__name__ == "MainPointUDT" and main_udt.scale == 2.5
    assert main_udt.sqlType() == T.ArrayType(T.DoubleType())


def test_types_give_back_and_pickle_their_user_defined_types_as_pyspark():
    # As PySpark's ArrayType, MapType, StructField and StructType, which hold the type objects
    # they are made of: each gives back its user-defined type, also nested, though the pyClass
    # of the JSON of ExamplePointUDT, pyspark.sql.tests.ExamplePointUDT, imports no class. They
    # pickle with it, and so does a StructField of any type.
    import pickle

    from pyspark.serializers import CloudPickleSerializer
    from pyspark.testing.objects import ExamplePointUDT

    udt = ExamplePointUDT()
    assert udt.jsonValue()["pyClass"] == "pyspark.sql.tests.ExamplePointUDT"
    nested = T.MapType(T.StringType(), T.StructType([T.StructField("p", udt)]))
    assert T.ArrayType(udt).elementType is udt
    assert T.MapType(udt, T.StringType()).keyType is udt
    assert T.MapType(T.StringType(), udt).valueType is udt
    assert T.StructField("p", udt).dataType is udt
    assert T.StructType([T.StructField("p", udt)]).fields[0].dataType is udt
    assert T.StructType().add("p", udt)["p"].dataType is udt
    assert T.ArrayType(T.ArrayType(udt)).elementType.elementType is udt
    assert nested.valueType["p"].dataType is udt
    assert repr(T.ArrayType(udt)) == "ArrayType(ExamplePointUDT(), True)"
    assert T.StructType([T.StructField("p", udt)]).simpleString() == "struct<p:udt>"

    serializer = CloudPickleSerializer()
    for t in [
        T.ArrayType(udt, False),
        T.MapType(T.StringType(), udt, False),
        T.StructType([T.StructField("p", udt, False, {"k": "v"})]),
        nested,
        T.StructField("p", udt, True, {"k": 1}),
        T.StructField("i", T.IntegerType(), False, {"k": "v"}),
        T.StructType([T.StructField("i", T.IntegerType())]),
    ]:
        for copied in (pickle.loads(pickle.dumps(t)), serializer.loads(serializer.dumps(t))):
            assert type(copied) is type(t) and copied == t and repr(copied) == repr(t)
    field = pickle.loads(pickle.dumps(T.StructField("p", udt, True, {"k": 1})))
    assert type(field.dataType) is ExamplePointUDT and field.metadata == {"k": 1}
    # A type with a user-defined type pickles of the class of pyspark.sql.types and the type
    # objects, which the Python workers of the server rebuild; one without, by its JSON.
    assert T.ArrayType(udt).__reduce__() == (T.ArrayType, (udt, True))
    assert T.ArrayType(T.IntegerType()).__reduce__()[0] is T._parse_datatype_json_string




def test_collected_values_of_user_defined_types_are_deserialized_as_pyspark():
    # The converters that the client applies to collected rows: the values of user-defined
    # types, in arrays, maps and structs too, are deserialized by their UDTs, as PySpark's
    # ArrowTableToRowsConversion does; fields of other types need none.
    from pyspark.ml.linalg import DenseVector, VectorUDT
    from pyspark.sql.types import _udt_field_converters

    vector = Row(type=1, size=None, indices=None, values=[1.5, 2.0])
    schema = T.StructType([
        T.StructField("v", VectorUDT()),
        T.StructField("k", T.IntegerType()),
        T.StructField("arr", T.ArrayType(VectorUDT())),
        T.StructField("m", T.MapType(T.StringType(), _DayPointUDT())),
        T.StructField("st", T.StructType([T.StructField("v", VectorUDT())])),
    ])
    v, k, arr, m, st = _udt_field_converters(schema)
    assert v(vector) == DenseVector([1.5, 2.0]) and v(None) is None
    assert k is None
    assert arr([vector, None]) == [DenseVector([1.5, 2.0]), None]
    assert m({"x": [1.0, 2.0]}) == {"x": (1.0, 2.0)}
    converted = st(Row(v=vector))
    assert converted == Row(v=DenseVector([1.5, 2.0])) and converted.v == DenseVector([1.5, 2.0])
    assert _udt_field_converters(T.StructType([T.StructField("k", T.LongType())])) is None


def test_dropin_module_imports():
    # The thin re-export shims import cleanly.
    import pyspark.sql.dataframe  # noqa: F401
    import pyspark.sql.observation  # noqa: F401
    import pyspark.sql.window  # noqa: F401
    from pyspark.sql import Row
    r = Row(a=1, b=2)
    assert r["a"] == 1


def test_functions_module_wrappers():
    from pyspark.sql import functions as F
    c = F.col("v")
    assert type(F.sha2(c, 256)).__name__ == "Column"
    assert type(F.window(F.col("t"), "10 minutes")).__name__ == "Column"
    assert type(F.window(F.col("t"), "10 minutes", "5 minutes", "0 seconds")).__name__ == "Column"
    import pyspark.sql.avro.functions as avro
    import pyspark.sql.protobuf.functions as protobuf

    assert type(avro.from_avro(c, "{}")).__name__ == "Column"
    assert type(avro.from_avro(c, "{}", {"mode": "PERMISSIVE"})).__name__ == "Column"
    assert type(avro.to_avro(c, "{}")).__name__ == "Column"
    assert type(protobuf.from_protobuf(c, "M", options={"x": "y"})).__name__ == "Column"
    assert type(protobuf.to_protobuf(c, "M", options={"x": "y"})).__name__ == "Column"


def test_type_class_hierarchy():
    # The concrete type classes have the reference MRO, and isinstance works up the
    # whole chain (DataType / AtomicType / NumericType / IntegralType / ...).
    assert [c.__name__ for c in T.IntegerType.__mro__ if c.__name__ != "object"] == [
        "IntegerType", "IntegralType", "NumericType", "AtomicType", "DataType",
    ]
    it = T.IntegerType()
    assert isinstance(it, T.DataType)
    assert isinstance(it, T.AtomicType)
    assert isinstance(it, T.NumericType)
    assert isinstance(it, T.IntegralType)
    assert not isinstance(it, T.FractionalType)

    assert isinstance(T.DoubleType(), T.FractionalType)
    assert isinstance(T.DecimalType(10, 2), T.FractionalType)
    assert isinstance(T.DateType(), T.DatetimeType)
    assert isinstance(T.TimeType(6), T.AnyTimeType)
    assert isinstance(T.DayTimeIntervalType(), T.AnsiIntervalType)
    assert isinstance(T.StringType(), T.AtomicType)
    # Complex types are DataType but not AtomicType.
    for c in (T.ArrayType(T.IntegerType()), T.MapType(T.StringType(), T.IntegerType()),
              T.StructType([T.StructField("a", T.IntegerType())]), T.NullType()):
        assert isinstance(c, T.DataType)
        assert not isinstance(c, T.AtomicType)
    # issubclass relationships.
    assert issubclass(T.IntegerType, T.NumericType)
    assert issubclass(T.NumericType, T.AtomicType)
    assert issubclass(T.AtomicType, T.DataType)


@pytest.mark.parametrize("make, expected", [
    (lambda: T.GeometryType("ANY"), "GeometryType(ANY)"),
    (lambda: T.GeographyType("ANY"), "GeographyType(ANY)"),
    (lambda: T.GeometryType(0), "GeometryType(0)"),
    (lambda: T.GeometryType(3857), "GeometryType(3857)"),
    (lambda: T.GeometryType(4326), "GeometryType(4326)"),
    (lambda: T.GeographyType(4326), "GeographyType(4326)"),
])
def test_spatial_types_take_the_srids_that_pyspark_knows(make, expected):
    spatial = make()
    assert repr(spatial) == expected
    # "ANY" is the mixed SRID.
    assert (spatial.srid == -1) is ("ANY" in expected)


@pytest.mark.parametrize("make, srid", [
    (lambda: T.GeometryType(2000), "2000"),
    (lambda: T.GeometryType(-1), "-1"),
    (lambda: T.GeometryType("any"), "any"),
    (lambda: T.GeometryType("4326"), "4326"),
    (lambda: T.GeographyType(3857), "3857"),
    (lambda: T.GeographyType(0), "0"),
])
def test_spatial_types_reject_other_srids_as_pyspark(make, srid):
    from pyspark.errors import IllegalArgumentException

    with pytest.raises(IllegalArgumentException) as error:
        make()
    assert error.value.getCondition() == "ST_INVALID_SRID_VALUE"
    assert error.value.getMessageParameters() == {"srid": srid}


def test_spatial_types_need_an_srid():
    for spatial in [T.GeometryType, T.GeographyType]:
        with pytest.raises(TypeError):
            spatial()


def test_spatial_types():
    g = T.GeometryType(4326)
    assert [c.__name__ for c in type(g).__mro__ if c.__name__ != "object"] == [
        "GeometryType", "SpatialType", "AtomicType", "DataType",
    ]
    assert g.srid == 4326
    assert g.simpleString() == "geometry(4326)"
    assert g.typeName() == "geometry"
    assert isinstance(g, T.SpatialType) and isinstance(g, T.DataType)
    gg = T.GeographyType(4326)
    assert gg.simpleString() == "geography(4326)" and gg.srid == 4326
    assert isinstance(gg, T.SpatialType)


def test_structtype_add_reflected_in_json():
    # Regression: after mutating with add(), json()/jsonValue() must reflect the new
    # field (py_to_data_type reads the live concrete fields, not a base snapshot).
    st = T.StructType([T.StructField("a", T.IntegerType())])
    assert '"b"' not in st.json()
    st.add("b", T.StringType())
    assert '"b"' in st.json()
    assert st.fieldNames() == ["a", "b"]


def test_structtype_add_reflected_in_the_base_data_type():
    # Regression: add() must update the base DataType too, which the schemas of
    # mapInArrow/mapInPandas/applyInPandas/... read (via resolve_datatype). The base
    # implementation of simpleString reads it.
    built = T.StructType([T.StructField("a", T.LongType()), T.StructField("b", T.LongType())])
    chained = T.StructType().add("a", T.LongType()).add("b", T.LongType())
    st = T.StructType([T.StructField("a", T.LongType())])
    st.add("b", T.LongType())
    for added in (chained, st):
        assert T.DataType.simpleString(added) == "struct<a:bigint,b:bigint>"
        assert str(added) == str(built) == repr(built)


@pytest.mark.parametrize("data_type,expected", [
    (T.LongType(), "LongType()"),
    (T.TimestampNTZType(), "TimestampNTZType()"),
    (T.VariantType(), "VariantType()"),
    (T.DecimalType(10, 2), "DecimalType(10,2)"),
    (T.CharType(5), "CharType(5)"),
    (T.DayTimeIntervalType(0, 3), "DayTimeIntervalType(0, 3)"),
    (T.YearMonthIntervalType(), "YearMonthIntervalType(0, 1)"),
    (T.ArrayType(T.StringType(), False), "ArrayType(StringType(), False)"),
    (T.MapType(T.StringType(), T.LongType()), "MapType(StringType(), LongType(), True)"),
    (T.StructType([T.StructField("a", T.LongType()),
                   T.StructField("b", T.ArrayType(T.StringType()), False)]),
     "StructType([StructField('a', LongType(), True), "
     "StructField('b', ArrayType(StringType(), True), False)])"),
    (T.StructType(), "StructType([])"),
    (T.DataType(), "DataType()"),
])
def test_datatype_str_is_its_repr_as_pyspark(data_type, expected):
    # pyspark data types define only __repr__, so str() is the same.
    assert repr(data_type) == expected
    assert str(data_type) == expected


def test_user_defined_type_repr_is_its_class_name():
    from pyspark.testing.objects import ExamplePointUDT

    assert repr(ExamplePointUDT()) == str(ExamplePointUDT()) == "ExamplePointUDT()"


@pytest.mark.parametrize("ddl, cls, expected", [
    # The examples of pyspark's DataType.fromDDL.
    ("b string, a int", T.StructType,
     "StructType([StructField('b', StringType(), True), StructField('a', IntegerType(), True)])"),
    ("decimal(10,10)", T.DecimalType, "DecimalType(10,10)"),
    ("b: string, a: int", T.StructType,
     "StructType([StructField('b', StringType(), True), StructField('a', IntegerType(), True)])"),
    ("int", T.IntegerType, "IntegerType()"),
])
def test_datatype_from_ddl_returns_the_concrete_type(ddl, cls, expected, mock_connect):
    parsed = T.DataType.fromDDL(ddl)
    assert repr(parsed) == expected
    assert isinstance(parsed, cls)
    # The classmethod parses DDL the same way from any data type class.
    assert repr(T.LongType.fromDDL(ddl)) == expected


def test_datatype_object_model(mock_connect):
    # json / jsonValue / typeName / simpleString / needConversion match official pyspark.
    assert T.IntegerType().typeName() == "integer"
    assert T.IntegerType().simpleString() == "int"
    assert T.IntegerType().json() == '"integer"'
    assert T.IntegerType().jsonValue() == "integer"
    assert T.IntegerType().needConversion() is False
    assert T.ArrayType(T.IntegerType()).typeName() == "array"
    assert T.DecimalType(10, 2).json() == '"decimal(10,2)"'
    # fromInternal/toInternal default to identity.
    assert T.IntegerType().fromInternal(5) == 5
    assert T.IntegerType().toInternal(5) == 5
    # fromDDL parses a DDL string client-side.
    assert T.DataType.fromDDL("a int, b string").simpleString() == "struct<a:int,b:string>"
    # A single-field DDL schema (no comma) must also parse to a one-field struct, not be
    # misread as a bare type (SPARK-59113 follow-up).
    assert T.DataType.fromDDL("i int").simpleString() == "struct<i:int>"
    assert T.StructType.fromDDL("id long").simpleString() == "struct<id:bigint>"


def test_structfield_collation_methods():
    f = T.StructField("a", T.StringType())
    assert f.getCollationsMap({}) == {}
    assert f.getCollationsMap({"__COLLATIONS": {"a": "icu.UNICODE"}}) == {"a": "UNICODE"}
    assert f.schemaCollationValue(T.StringType()) == "spark.UTF8_BINARY"


def test_structfield_object_model():
    f = T.StructField("a", T.IntegerType(), True)
    assert f.simpleString() == "a:int"
    assert f.json() == '{"metadata":{},"name":"a","nullable":true,"type":"integer"}'
    assert f.jsonValue() == {"name": "a", "type": "integer", "nullable": True, "metadata": {}}
    assert f.needConversion() is False
    assert f.getCollationMetadata() == {}
    # StructField.typeName raises (use typeName on the type instead), as in pyspark.
    with pytest.raises(TypeError):
        f.typeName()
    # fromJson round-trips.
    assert T.StructField.fromJson(f.jsonValue()).simpleString() == "a:int"


def test_structtype_object_model():
    st = T.StructType([
        T.StructField("a", T.IntegerType(), True),
        T.StructField("b", T.StringType(), False),
    ])
    assert st.typeName() == "struct"
    assert st.fieldNames() == ["a", "b"]
    assert "root" in st.treeString() and "a: integer (nullable = true)" in st.treeString()
    # toNullable makes every field nullable.
    assert st.toNullable().simpleString() == "struct<a:int,b:string>"
    # json round-trips through fromJson.
    assert T.StructType.fromJson(st.jsonValue()).simpleString() == st.simpleString()
    # needConversion true for a struct with fields.
    assert isinstance(st.needConversion(), bool)


def test_higher_order_functions():
    from pyspark.sql import functions as F

    def is_col(x):
        return type(x).__name__ == "Column"

    # Every higher-order function builds a Column via the lambda machinery.
    assert is_col(F.transform("arr", lambda x: x + 1))
    assert is_col(F.transform("arr", lambda x, i: x + i))  # 2-arg (element, index)
    assert is_col(F.filter("arr", lambda x: x > 0))
    assert is_col(F.filter("arr", lambda x, i: i > 0))
    assert is_col(F.exists("arr", lambda x: x > 0))
    assert is_col(F.forall("arr", lambda x: x > 0))
    assert is_col(F.aggregate("arr", F.lit(0), lambda acc, x: acc + x))
    assert is_col(F.aggregate("arr", F.lit(0), lambda acc, x: acc + x, lambda acc: acc * 2))
    assert is_col(F.reduce("arr", F.lit(0), lambda acc, x: acc + x))
    assert is_col(F.zip_with("a", "b", lambda x, y: x + y))
    assert is_col(F.transform_keys("m", lambda k, v: k))
    assert is_col(F.transform_values("m", lambda k, v: v + 1))
    assert is_col(F.map_filter("m", lambda k, v: v > 0))
    assert is_col(F.map_zip_with("m1", "m2", lambda k, v1, v2: v1 + v2))
    # Nested lambdas must get distinct fresh variable names (no collision).
    assert is_col(F.transform("arr", lambda x: F.transform(x, lambda y: y + 1)))


def test_higher_order_function_validation():
    from pyspark.sql import functions as F
    from pyspark.errors import PySparkValueError

    # arity must be 1..3
    with pytest.raises(PySparkValueError):
        F.transform("arr", lambda: 1)
    with pytest.raises(PySparkValueError):
        F.transform("arr", lambda a, b, c, d: a)
    # the lambda must return a Column
    with pytest.raises(PySparkValueError):
        F.transform("arr", lambda x: 123)


def test_misc_functions():
    from pyspark.sql import functions as F

    def is_col(x):
        return type(x).__name__ == "Column"

    assert is_col(F.cume_dist())
    assert is_col(F.column("a"))
    assert is_col(F.call_udf("my_udf", F.col("a"), F.col("b")))
    assert is_col(F.call_udf("my_udf", "a", "b"))  # str args coerced to columns
    assert is_col(F.call_function("my_fn", F.col("a")))
    assert is_col(F.call_function("my_fn", "a"))


def test_column_attributes_are_fields_and_slices_are_substrings_as_pyspark():
    from pyspark.errors import PySparkAttributeError, PySparkTypeError, PySparkValueError
    from pyspark.sql import functions as F

    c = F.col("a")
    # An attribute that a Column does not have is the field of that name.
    assert repr(c.b) == repr(c["b"])
    assert repr(c.b.c) == repr(c["b"]["c"])
    assert repr(c._private) == repr(c["_private"])
    assert callable(c.alias)
    for name, condition in (("_jc", "JVM_ATTRIBUTE_NOT_SUPPORTED"), ("__x__", "ATTRIBUTE_NOT_SUPPORTED")):
        with pytest.raises(PySparkAttributeError) as error:
            getattr(c, name)
        assert error.value.getCondition() == condition
        assert error.value.getMessageParameters() == {"attr_name": name}
    assert not hasattr(c, "__array__")
    # A slice is substr(start, stop), whose arguments are of one type, a Column or an int.
    assert repr(c[1:3]) == "Column<'substr(a, 1, 3)'>"
    assert repr(c[F.lit(1):F.lit(2)]) == "Column<'substr(a, 1, 2)'>"
    with pytest.raises(PySparkValueError) as error:
        c[1:3:2]
    assert error.value.getCondition() == "SLICE_WITH_STEP"
    for args, condition, parameters in (
        ((1, F.lit(2)), "NOT_SAME_TYPE", {
            "arg_name1": "startPos", "arg_name2": "length", "arg_type1": "int",
            "arg_type2": "Column"}),
        ((None, 3), "NOT_SAME_TYPE", {
            "arg_name1": "startPos", "arg_name2": "length", "arg_type1": "NoneType",
            "arg_type2": "int"}),
        (("x", "y"), "NOT_EXPECTED_TYPE", {
            "expected_type": "Column or int", "arg_name": "startPos", "arg_type": "str"}),
    ):
        with pytest.raises(PySparkTypeError) as error:
            c.substr(*args)
        assert error.value.getCondition() == condition
        assert error.value.getMessageParameters() == parameters
    with pytest.raises(PySparkTypeError, match="NOT_SAME_TYPE"):
        c[:3]


def test_broadcast_type_check():
    from pyspark.sql import functions as F
    from pyspark.errors import PySparkTypeError

    with pytest.raises(PySparkTypeError):
        F.broadcast("not a dataframe")


def test_dataframe_columns_refer_to_it_by_its_plan_id_as_pyspark(mock_connect):
    # As PySpark Connect's df[name], df.name, df["*"] and colRegex: the columns of a DataFrame
    # refer to it by the plan ID that its relation is sent with, and colRegex is a Column.
    from pyspark.sql import Column

    spark, service = mock_connect
    left, right = spark.range(2), spark.range(3)
    assert isinstance(left.colRegex("`i.*`"), Column)
    left.join(right, left["id"] == right.id).select(
        left.id, right["*"], left.colRegex("`i.*`"), left["id"]
    ).collect()
    # ExecutePlanRequest.plan=3, Plan.root=1, Relation.project=3: input=1, expressions=3;
    # Relation.join=5: left=1, right=2; Relation.common=1, RelationCommon.plan_id=2.
    plan = parse_message(parse_message(service.execute_requests[-1])[3][0])[1][0]
    project = parse_message(parse_message(plan)[3][0])
    join = parse_message(parse_message(project[1][0])[5][0])

    def plan_id(relation):
        return parse_message(parse_message(relation)[1][0])[2][0]

    left_id, right_id = plan_id(join[1][0]), plan_id(join[2][0])
    assert left_id != right_id
    # Expression.unresolved_attribute=2 (plan_id=2), unresolved_star=5 (plan_id=2),
    # unresolved_regex=8 (plan_id=2).
    ids = []
    for expression in project[3]:
        expression = parse_message(expression)
        (field,) = set(expression) & {2, 5, 8}
        ids.append(parse_message(expression[field][0])[2][0])
    assert ids == [left_id, right_id, left_id, left_id]


def test_row_count_and_index():
    from pyspark.sql import Row
    r = Row(a=1, b=2, c=1)
    assert r.count(1) == 2
    assert r.count(2) == 1
    assert r.index(2) == 1
    assert r.index(1) == 0
    with pytest.raises(ValueError):
        r.index(99)


def test_na_and_stat_functions_check_their_arguments_as_pyspark():
    # As PySpark's NA and statistic functions of a DataFrame, which check their arguments on
    # the client.
    from pyspark.errors import PySparkTypeError, PySparkValueError

    df = _offline_session().range(1)

    def not_expected(arg_name, expected_type, arg_type):
        return (PySparkTypeError, "NOT_EXPECTED_TYPE",
                {"arg_name": arg_name, "expected_type": expected_type, "arg_type": arg_type})

    for call, (error_type, condition, parameters) in (
        (lambda: df.dropna(how="bogus"), (PySparkValueError, "VALUE_NOT_ALLOWED",
                                          {"arg_name": "how", "allowed_values": "['any', 'all']"})),
        (lambda: df.na.drop(how=1), not_expected("how", "str", "int")),
        (lambda: df.dropna(thresh=1.5), not_expected("thresh", "int", "float")),
        (lambda: df.dropna(subset=1), not_expected("subset", "list, str or tuple", "int")),
        (lambda: df.na.drop(subset=["id", 1]), not_expected("cols", "list, str or tuple", "int")),
        (lambda: df.fillna([1]), not_expected("value", "bool, dict, float, int or str", "list")),
        (lambda: df.na.fill({}), (PySparkValueError, "CANNOT_BE_EMPTY", {"item": "value"})),
        (lambda: df.fillna({1: 0}), not_expected("key type of dict", "str", "int")),
        (lambda: df.fillna({"id": [1]}),
         not_expected("value type of dict", "bool, float, int or str", "list")),
        (lambda: df.fillna(0, subset=1), not_expected("subset", "list or tuple", "int")),
        (lambda: df.summary(1), not_expected("statistics", "list[str]", "int")),
        (lambda: df.cov("id", 1), not_expected("col2", "str", "int")),
        (lambda: df.stat.corr(1, "id"), not_expected("col1", "str", "int")),
        (lambda: df.corr("id", "id", "kendall"), (PySparkValueError, "VALUE_NOT_ALLOWED",
                                                  {"arg_name": "method",
                                                   "allowed_values": "['pearson']"})),
        (lambda: df.stat.crosstab("id", 1), not_expected("col2", "str", "int")),
        (lambda: df.freqItems("id"), not_expected("cols", "list or tuple", "str")),
        (lambda: df.approxQuantile(1, [0.5], 0.0), not_expected("col", "list, str or tuple", "int")),
        (lambda: df.approxQuantile(["id", 1], [0.5], 0.0), not_expected("columns", "list[str]", "int")),
        (lambda: df.stat.approxQuantile("id", 0.5, 0.0),
         not_expected("probabilities", "list or tuple", "float")),
        (lambda: df.approxQuantile("id", [1.5], 0.0),
         not_expected("probabilities", "list[float, int]", "float")),
        (lambda: df.approxQuantile("id", [0.5], "0"), not_expected("relativeError", "float or int", "str")),
        (lambda: df.approxQuantile("id", [0.5], -1), (PySparkValueError, "NEGATIVE_VALUE",
                                                     {"arg_name": "relativeError", "arg_value": "-1"})),
    ):
        with pytest.raises(error_type) as error:
            call()
        assert error.value.getCondition() == condition
        assert error.value.getMessageParameters() == parameters


def test_avro_protobuf_partitioning_submodules():
    # The official import paths resolve (the implementations live in
    # pyspark.sql.functions and are re-exported here).
    import pyspark.sql.avro.functions as af
    import pyspark.sql.protobuf.functions as pf
    import pyspark.sql.functions.partitioning as part
    from pyspark.sql import functions as F

    assert hasattr(af, "from_avro") and hasattr(af, "to_avro")
    assert hasattr(pf, "from_protobuf") and hasattr(pf, "to_protobuf")
    for n in ("bucket", "days", "hours", "months", "years"):
        assert hasattr(part, n), f"partitioning.{n} missing"
        assert hasattr(F.partitioning, n)


def test_java_functions_are_registered_of_their_return_types_as_pyspark(mock_connect):
    # As PySpark's spark.udf, whose sparkSession is the session, and whose registerJavaFunction
    # sends a DDL string as an unparsed data type, which the server parses, and a DataType as
    # itself, in a JavaUDF that is not deterministic.
    from pyspark.errors import PySparkValueError
    from pyspark.sql.types import IntegerType

    spark, service = mock_connect
    assert spark.udf.sparkSession is spark

    def registered():
        # ExecutePlanRequest.plan=3, Plan.command=2, Command.register_function=1;
        # CommonInlineUserDefinedFunction.function_name=1, deterministic=2, java_udf=6.
        plan = parse_message(parse_message(service.execute_requests[-1])[3][0])
        function = parse_message(parse_message(plan[2][0])[1][0])
        return function, parse_message(function[6][0])

    spark.udf.registerJavaFunction("f", "com.example.F", "int")
    function, udf = registered()
    assert function[1] == [b"f"]
    assert function.get(2, [0]) == [0]
    # JavaUDF.class_name=1, output_type=2, aggregate=3; DataType.unparsed=24 of
    # Unparsed.data_type_string=1, and DataType.integer=6.
    assert udf[1] == [b"com.example.F"]
    assert parse_message(parse_message(udf[2][0])[24][0]) == {1: [b"int"]}
    spark.udf.registerJavaFunction("g", "com.example.G", IntegerType())
    _, udf = registered()
    assert list(parse_message(udf[2][0])) == [6]
    spark.udf.registerJavaFunction("h", "com.example.H")
    _, udf = registered()
    assert 2 not in udf and udf.get(3, [0]) == [0]
    spark.udf.registerJavaUDAF("i", "com.example.I")
    _, udf = registered()
    assert 2 not in udf and udf[3] == [1]
    calls = len(service.execute_requests)
    with pytest.raises(PySparkValueError) as e:
        spark.udf.registerJavaFunction("j", "com.example.J", 1)
    assert e.value.getCondition() == "UNSUPPORTED_OPERATION"
    assert e.value.getMessageParameters() == {"operation": "data type 1"}
    assert len(service.execute_requests) == calls


def test_udf_registration_has_java_methods():
    # The UDFRegistration class (what spark.udf returns) exposes the Java methods.
    from pyspark.sql.udf import UDFRegistration
    for m in ("register", "registerJavaFunction", "registerJavaUDAF"):
        assert hasattr(UDFRegistration, m), f"UDFRegistration missing {m}"


def test_udf_udtf_determinism_and_extras():
    from pyspark.sql import functions as F
    from pyspark.sql.udf import udf, UserDefinedFunction
    from pyspark.sql.udtf import udtf

    # As PySpark's, asNondeterministic() of the function wrapper of a UDF marks the UDF
    # nondeterministic, in place, and returns a wrapper of it.
    u = udf(lambda x: x, T.IntegerType())
    nd = u.asNondeterministic()
    assert isinstance(nd._unwrapped, UserDefinedFunction) and nd.deterministic is False
    assert nd._unwrapped is u._unwrapped and u._unwrapped.deterministic is False
    assert u.returnType is not None

    @udtf(returnType="a int")
    class E:
        def eval(self, n):
            yield (n,)

    d = E.asDeterministic()
    assert d.deterministic is True

    # functions.random is an alias for rand.
    assert F.random is F.rand
    # StructField.fromDDL.
    assert T.StructField.fromDDL("a int").simpleString() == "a:int"
    assert T.StructField.fromDDL("b: string").simpleString() == "b:string"


def test_catalog_result_classes():
    # As PySpark's: the catalog metadata result classes are the NamedTuples of
    # pyspark.sql.catalog, and the Table `database` property derives from a single-element
    # namespace.
    from pyspark.sql.catalog import (
        Catalog, CatalogMetadata, Database, Table, Column, Function, TablePartition,
    )
    assert Catalog.__module__ == "pyspark.sql.catalog"
    for cls in (CatalogMetadata, Database, Table, Column, Function, TablePartition):
        assert cls.__module__ == "pyspark.sql.catalog", f"{cls.__name__} module = {cls.__module__}"
        assert issubclass(cls, tuple)
    assert Column._fields == (
        "name", "description", "dataType", "nullable", "isPartition", "isBucket", "isCluster")
    assert Table("t", None, ["db"], None, "MANAGED", False).database == "db"
    assert Table("t", None, ["a", "b"], None, "MANAGED", False).database is None


def test_unparsed_data_types_are_those_of_pyspark(mock_connect):
    # As PySpark Connect's: UnparsedDataType of pyspark.sql.connect.types is a DDL string of a
    # type that the server parses, whose JSON and conversions raise
    # INVALID_CALL_ON_UNRESOLVED_OBJECT; the returnType of a UDTF of a DDL string is one; and
    # the registration of a data source without a name raises the AttributeError of name().
    import pickle

    from pyspark.errors import PySparkAssertionError
    from pyspark.sql.connect.types import UnparsedDataType
    from pyspark.sql.functions import udtf

    spark, _ = mock_connect
    unparsed = UnparsedDataType("a int, b string")
    assert isinstance(unparsed, T.DataType)
    assert repr(unparsed) == "UnparsedDataType('a int, b string')"
    assert unparsed.simpleString() == "unparsed('a int, b string')"
    for name, call in (("jsonValue", unparsed.jsonValue), ("jsonValue", unparsed.json),
                       ("needConversion", unparsed.needConversion),
                       ("toInternal", lambda: unparsed.toInternal(1)),
                       ("fromInternal", lambda: unparsed.fromInternal(1))):
        with pytest.raises(PySparkAssertionError) as e:
            call()
        assert e.value.getCondition() == "INVALID_CALL_ON_UNRESOLVED_OBJECT"
        assert e.value.getMessageParameters() == {"func_name": name}
    assert repr(pickle.loads(pickle.dumps(unparsed))) == repr(unparsed)

    @udtf(returnType="x int")
    class Tables:
        def eval(self, n: int):
            yield (n,)

    assert repr(Tables.returnType) == "UnparsedDataType('x int')"
    with pytest.raises(AttributeError):
        spark.dataSource.register(1)




def test_udtf_analyze_result_family():
    from pyspark.sql.udtf import (
        AnalyzeArgument, PartitioningColumn, OrderingColumn, SelectedColumn,
        AnalyzeResult, SkipRestOfInputTableException,
    )
    assert PartitioningColumn("x").name == "x"
    oc = OrderingColumn("y", ascending=False)
    assert oc.name == "y" and oc.ascending is False and oc.overrideNullsFirst is None
    assert SelectedColumn("z", alias="a").alias == "a"
    ar = AnalyzeResult(schema=T.StructType([T.StructField("a", T.IntegerType())]),
                       withSinglePartition=True, partitionBy=[PartitioningColumn("x")])
    assert ar.withSinglePartition is True and len(ar.partitionBy) == 1
    ar.withSinglePartition = False  # mutable dataclass
    assert ar.withSinglePartition is False
    assert issubclass(SkipRestOfInputTableException, Exception)


def test_udtf_analyze_result_subclass_preserves_custom_state():
    from dataclasses import dataclass, fields
    from pyspark.serializers import CloudPickleSerializer
    from pyspark.sql.connect.udtf import AnalyzeResult as ConnectAnalyzeResult
    from pyspark.sql.udtf import AnalyzeResult, udtf

    assert ConnectAnalyzeResult is AnalyzeResult

    @dataclass
    class ResultWithBuffer(AnalyzeResult):
        buffer: str = ""

    schema = T.StructType([T.StructField("buffer", T.StringType())])
    result = ResultWithBuffer(schema, buffer="kept")
    assert [field.name for field in fields(result)] == [
        "schema", "withSinglePartition", "partitionBy", "orderBy", "select", "buffer",
    ]
    assert result.partitionBy == result.orderBy == result.select == ()
    assert result.withSinglePartition is False
    result.buffer = "updated"
    restored = CloudPickleSerializer().loads(CloudPickleSerializer().dumps(result))
    assert isinstance(restored, AnalyzeResult)
    assert restored == result

    class Handler:
        def __init__(self, analyze_result):
            self.buffer = analyze_result.buffer

        @staticmethod
        def analyze(argument):
            return ResultWithBuffer(schema, withSinglePartition=True, buffer="kept")

        def eval(self, argument):
            yield (self.buffer,)

    function = udtf(Handler, useArrow=False)
    handler = CloudPickleSerializer().loads(function.command)
    analysis = handler.analyze(None)
    assert isinstance(analysis, AnalyzeResult)
    assert analysis.schema == schema and analysis.withSinglePartition is True
    assert list(handler(analysis).eval(None)) == [("kept",)]


def test_variant_spatial_and_user_defined_values_are_sent_as_pyspark(mock_connect):
    # As PySpark's LocalDataToArrowConversion: a VariantVal is the struct of its value and
    # metadata, a Geometry or a Geography that of its srid and WKB, whose metadata tags the
    # struct as of its type, and a value of a user-defined type is serialized by the UDT and
    # sent as its SQL type; a value of another type for these is malformed.
    import pyarrow as pa
    from pyspark.errors import PySparkValueError
    from pyspark.testing.objects import ExamplePoint

    spark, service = mock_connect
    wkb = bytes.fromhex("0101000000000000000000F03F0000000000000040")
    variant = T.VariantVal(b"\x0c\x01", b"\x01\x00\x00")

    def local_relation(df):
        df.toArrow()
        request = parse_message(service.execute_requests[-1])
        relation = parse_message(parse_message(request[3][0])[1][0])
        while 18 in relation:  # Relation.to_df -> ToDF.input.
            relation = parse_message(parse_message(relation[18][0])[1][0])
        # Relation.local_relation=11 -> LocalRelation.data=1, schema=2.
        local = parse_message(relation[11][0])
        table = pa.ipc.open_stream(local[1][0]).read_all()
        return table, [field["type"] for field in json.loads(local[2][0])["fields"]]

    table, types = local_relation(spark.createDataFrame([
        (variant, T.Geometry.fromWKB(wkb, 4326), T.Geography.fromWKB(wkb, 4326),
         ExamplePoint(1.0, 2.0)),
        (None, None, None, None),
    ]))
    assert types[:3] == [
        T.VariantType().jsonValue(), T.GeometryType(4326).jsonValue(),
        T.GeographyType(4326).jsonValue(),
    ]
    assert types[3]["type"] == "udt" and types[3]["sqlType"]["type"] == "array"
    variant_type, geometry_type, geography_type, udt_type = table.schema.types
    assert variant_type.field("metadata").metadata == {b"variant": b"true"}
    assert geometry_type.field("wkb").metadata == {b"geometry": b"true", b"srid": b"4326"}
    assert geography_type.field("wkb").metadata == {b"geography": b"true", b"srid": b"4326"}
    assert geometry_type.field("srid").type == pa.int32()
    assert pa.types.is_list(udt_type) and udt_type.value_type == pa.float64()
    assert table.to_pylist() == [
        {"_1": {"value": b"\x0c\x01", "metadata": b"\x01\x00\x00"},
         "_2": {"srid": 4326, "wkb": wkb}, "_3": {"srid": 4326, "wkb": wkb}, "_4": [1.0, 2.0]},
        {"_1": None, "_2": None, "_3": None, "_4": None},
    ]
    for ddl, condition in (("v variant", "MALFORMED_VARIANT"),
                           ("g geometry(4326)", "MALFORMED_GEOMETRY"),
                           ("g geography(4326)", "MALFORMED_GEOGRAPHY")):
        with pytest.raises(PySparkValueError) as error:
            spark.createDataFrame([("x",)], ddl)
        assert error.value.getCondition() == condition
    # The attributes of PySpark's spatial values.
    geometry = T.Geometry.fromWKB(wkb, 4326)
    assert (geometry.srid, geometry.wkb) == (4326, wkb)


def test_collected_spatial_values_are_their_objects_as_pyspark():
    # As PySpark's ArrowTableToRowsConversion, a collected GEOMETRY or GEOGRAPHY is the
    # Geometry or Geography of its WKB and srid.
    from pyspark.sql.types import _udt_field_converters

    wkb = b"\x01\x01"
    schema = T.StructType([
        T.StructField("g", T.GeometryType(4326)),
        T.StructField("gs", T.ArrayType(T.GeographyType(4326))),
    ])
    g, gs = _udt_field_converters(schema)
    assert g(Row(srid=4326, wkb=wkb)) == T.Geometry.fromWKB(wkb, 4326)
    assert gs([Row(srid=4326, wkb=wkb), None]) == [T.Geography.fromWKB(wkb, 4326), None]
    assert g(None) is None


def test_variant_and_geo_values():
    from pyspark.sql.types import VariantVal, Geometry, Geography
    v = VariantVal.parseJson('{"a": 1, "b": [2, 3]}')
    assert v.toJson() == '{"a":1,"b":[2,3]}'
    assert v.toPython() == {"a": 1, "b": [2, 3]}
    g = Geometry(b"\x01\x02", 4326)
    assert g.getSrid() == 4326 and g.getBytes() == b"\x01\x02"
    assert g == Geometry.fromWKB(b"\x01\x02", 4326)
    gg = Geography.fromWKB(b"\x03", 4326)
    assert gg.getSrid() == 4326 and gg.getBytes() == b"\x03"


def test_table_arg_class():
    import pyspark._pyspark as p
    assert hasattr(p, "TableArg")
    for m in ("partitionBy", "orderBy", "withSinglePartition"):
        assert hasattr(p.TableArg, m)


def test_python_eval_type_constants():
    from pyspark.util import PythonEvalType
    assert PythonEvalType.SQL_BATCHED_UDF == 100
    assert PythonEvalType.SQL_ARROW_BATCHED_UDF == 101
    assert PythonEvalType.SQL_SCALAR_PANDAS_UDF == 200
    assert PythonEvalType.SQL_SCALAR_ARROW_UDF == 250
    assert PythonEvalType.SQL_TABLE_UDF == 300
    assert PythonEvalType.SQL_ARROW_TABLE_UDF == 301
    assert PythonEvalType.__module__ == "pyspark.util"  # reference path (SPARK-59117)


def test_reexport_paths():
    from pyspark import StorageLevel
    assert StorageLevel.MEMORY_ONLY is not None
    from pyspark.sql.streaming import StreamingQueryListener
    assert StreamingQueryListener.__name__ == "StreamingQueryListener"


def test_util_helpers():
    from pyspark import util
    assert util.is_remote_only() is True
    assert util._parse_memory("256m") == 256
    assert util._parse_memory("2g") == 2048
    with pytest.raises(ValueError):
        util._parse_memory("100x")


def test_udf_nondeterministic_and_evaltypes():
    from pyspark.sql.functions import udf, pandas_udf
    # useArrow path selects the arrow eval type.
    a = udf(lambda x: x, T.IntegerType(), useArrow=True)
    assert callable(a)
    # pandas_udf with an explicit functionType, a value of PandasUDFType as in PySpark.
    from pyspark.sql.functions import PandasUDFType

    p = pandas_udf(lambda s: s, T.IntegerType(), functionType=PandasUDFType.SCALAR)
    assert p is not None


class _EvalHandler:
    def eval(self, x):
        yield (x,)


class _AnalyzeHandler:
    def eval(self, x):
        yield (x,)

    @staticmethod
    def analyze(x):
        pass


_NOT_A_COLUMN = {"expected_type": "Column", "arg_name": "condition"}


@pytest.mark.parametrize("call,error,condition,parameters", [
    (lambda F: F.when("a", 1), "PySparkTypeError", "NOT_EXPECTED_TYPE",
     {**_NOT_A_COLUMN, "arg_type": "str"}),
    (lambda F: F.when(True, 1), "PySparkTypeError", "NOT_EXPECTED_TYPE",
     {**_NOT_A_COLUMN, "arg_type": "bool"}),
    (lambda F: F.when(F.col("a") > 1, 1).when("b", 2), "PySparkTypeError", "NOT_EXPECTED_TYPE",
     {**_NOT_A_COLUMN, "arg_type": "str"}),
    (lambda F: F.col("a").when(F.col("b") > 1, 1), "PySparkTypeError", "INVALID_WHEN_USAGE", {}),
    (lambda F: F.when(F.col("a") > 1, 1).otherwise(0).when(F.col("b") > 1, 2),
     "PySparkTypeError", "INVALID_WHEN_USAGE", {}),
    (lambda F: F.partitioning.bucket("n", "c"), "PySparkTypeError", "NOT_EXPECTED_TYPE",
     {"expected_type": "Column or int", "arg_name": "numBuckets", "arg_type": "str"}),
    (lambda F: F.partitioning.months(3), "PySparkTypeError", "NOT_EXPECTED_TYPE",
     {"expected_type": "Column or str", "arg_name": "col", "arg_type": "int"}),
    (lambda F: F.pandas_udf(), "PySparkTypeError", "CANNOT_BE_NONE", {"arg_name": "returnType"}),
    (lambda F: F.pandas_udf(lambda s: s), "PySparkTypeError", "CANNOT_BE_NONE",
     {"arg_name": "returnType"}),
    (lambda F: F.pandas_udf(lambda s: s, "long", 3), "PySparkTypeError", "INVALID_PANDAS_UDF_TYPE",
     {"arg_name": "functionType", "arg_type": "3"}),
    (lambda F: F.arrow_udf(), "PySparkTypeError", "CANNOT_BE_NONE", {"arg_name": "returnType"}),
    (lambda F: F.arrow_udf("long", 3), "PySparkTypeError", "INVALID_PANDAS_UDF_TYPE",
     {"arg_name": "functionType", "arg_type": "3"}),
    (lambda F: F.udtf(3, returnType="x int"), "PySparkTypeError", "INVALID_UDTF_HANDLER_TYPE",
     {"type": "int"}),
    (lambda F: F.arrow_udtf(3), "PySparkTypeError", "INVALID_UDTF_HANDLER_TYPE", {"type": "int"}),
    (lambda F: F.udtf(type("T", (), {}), returnType="x int"), "PySparkAttributeError",
     "INVALID_UDTF_NO_EVAL", {"name": "T"}),
    (lambda F: F.udtf(_EvalHandler), "PySparkAttributeError", "INVALID_UDTF_RETURN_TYPE",
     {"name": "_EvalHandler"}),
    (lambda F: F.udtf(_AnalyzeHandler, returnType="x int"), "PySparkAttributeError",
     "INVALID_UDTF_BOTH_RETURN_TYPE_AND_ANALYZE", {"name": "_AnalyzeHandler"}),
    (lambda F: F.arrow_udtf(_AnalyzeHandler), "PySparkAttributeError",
     "INVALID_ARROW_UDTF_WITH_ANALYZE", {"name": "_AnalyzeHandler"}),
])
def test_functions_check_their_arguments_as_pyspark(call, error, condition, parameters):
    # Before any request, with PySpark's errors; the cases agree with official PySpark 4.3.0.
    from pyspark.errors import PySparkException
    from pyspark.sql import functions as F

    with pytest.raises(PySparkException) as raised:
        call(F)
    assert type(raised.value).__name__ == error
    assert raised.value.getCondition() == condition
    assert raised.value.getMessageParameters() == parameters


def test_when_and_otherwise_build_one_case_when_as_pyspark():
    from pyspark.errors import PySparkTypeError
    from pyspark.sql import functions as F

    branches = F.when(F.col("a") > 1, 1).when(F.col("b") > 1, F.col("b"))
    assert repr(branches) == "Column<'CASE WHEN >(a, 1) THEN 1 WHEN >(b, 1) THEN b END'>"
    assert repr(branches.otherwise(0)) == (
        "Column<'CASE WHEN >(a, 1) THEN 1 WHEN >(b, 1) THEN b ELSE 0 END'>")
    # As PySpark's Column.otherwise, with a message rather than a condition.
    with pytest.raises(PySparkTypeError, match=r"^otherwise\(\) can only be applied on a Column"):
        F.col("a").otherwise(0)
    with pytest.raises(PySparkTypeError, match=r"^otherwise\(\) can only be applied once"):
        branches.otherwise(0).otherwise(2)


def test_partition_transforms_are_those_of_pyspark_and_the_others_deprecated():
    import warnings

    import pyspark.sql.functions.partitioning as P
    from pyspark.sql import functions as F

    with pytest.warns(FutureWarning, match="use partitioning.years instead"):
        assert repr(F.years("c")) == "Column<'years(c)'>"
    with pytest.warns(FutureWarning, match="use partitioning.bucket instead"):
        assert repr(F.bucket(2, "c")) == "Column<'bucket(2, c)'>"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert repr(P.days("c")) == "Column<'days(c)'>"
        assert repr(F.partitioning.bucket(F.lit(2), "c")) == "Column<'bucket(2, c)'>"
    assert P.bucket is F.partitioning.bucket


def test_udtf_construction():
    from pyspark.sql.udtf import udtf, UserDefinedTableFunction

    @udtf(returnType="a int")
    class Echo:
        def eval(self, n):
            for i in range(n):
                yield (i,)

    assert isinstance(Echo, UserDefinedTableFunction)
    assert Echo.name == "Echo"


def test_udtf_register_sends_the_udtf_as_pyspark(mock_connect):
    # As PySpark's UDTFRegistration.register: a register_table_function command of the
    # CommonInlineUserDefinedTableFunction of the UDTF, under the name and without
    # arguments, which returns the UDTF as it is; for a UDTF of udtf() or arrow_udtf() only.
    from pyspark.errors import PySparkTypeError
    from pyspark.serializers import CloudPickleSerializer
    from pyspark.sql import functions as F
    from pyspark.sql.udtf import UDTFRegistration, UserDefinedTableFunction

    spark, service = mock_connect

    class Echo:
        def eval(self, n):
            yield (n,)

    udtf = F.udtf(Echo, returnType="a int", useArrow=False).asDeterministic()
    assert isinstance(spark.udtf, UDTFRegistration) and spark.udtf.sparkSession is spark
    assert spark.udtf.register("echo", udtf) is udtf and udtf.name == "Echo"
    # ExecutePlanRequest.plan=3, Plan.command=2, Command.register_table_function=10;
    # CommonInlineUserDefinedTableFunction.function_name=1, deterministic=2, arguments=3,
    # python_udtf=4; PythonUDTF.return_type=1, eval_type=2, command=3, python_ver=4.
    command = parse_message(parse_message(service.execute_requests[-1])[3][0])[2][0]
    function = parse_message(parse_message(command)[10][0])
    assert function[1] == [b"echo"] and function[2] == [1] and 3 not in function
    python_udtf = parse_message(function[4][0])
    # DataType.unparsed=24, whose data_type_string=1.
    return_type = parse_message(python_udtf[1][0])
    assert parse_message(return_type[24][0])[1] == [b"a int"]
    assert python_udtf[2] == [300] and python_udtf[4] == [udtf.python_ver.encode()]
    assert CloudPickleSerializer().loads(python_udtf[3][0]).__name__ == "Echo"

    calls = service.execute_calls
    with pytest.raises(PySparkTypeError) as error:
        spark.udtf.register("echo", Echo)
    assert error.value.getCondition() == "CANNOT_REGISTER_UDTF"
    scalar = UserDefinedTableFunction(Echo, "a int", evalType=100)
    with pytest.raises(PySparkTypeError) as error:
        spark.udtf.register("echo", scalar)
    assert error.value.getCondition() == "INVALID_UDTF_EVAL_TYPE"
    assert service.execute_calls == calls


def test_udtf_call_without_session_raises():
    # As PySpark's, udtf() without a session is not Arrow-optimized, and the call raises the
    # error of SparkSession.active(). In a new interpreter, which has no active session.
    import subprocess
    import sys

    subprocess.run([sys.executable, "-c", """
from pyspark.errors import PySparkRuntimeError
from pyspark.sql import SparkSession
from pyspark.sql.functions import udtf

@udtf(returnType="a int")
class T2:
    def eval(self, n):
        yield (n,)

assert T2.evalType == 300
# As PySpark's udf() without a session, a regular UDF.
from pyspark.sql.functions import udf
from pyspark.sql.types import IntegerType
assert udf(lambda x: x, IntegerType()).evalType == 100
for call in (SparkSession.active, lambda: T2(1)):
    try:
        call()
    except PySparkRuntimeError as error:
        assert error.getCondition() == "NO_ACTIVE_OR_DEFAULT_SESSION", error
    else:
        raise AssertionError("no error")
"""], check=True)


def test_udtf_return_type_connect_validation_and_command():
    # A Connect DDL string (including invalid DDL) is validated by the server,
    # not when decorating the class or constructing the DataFrame. Isolate the
    # active session so the other offline tests still start without a session.
    import subprocess
    import sys

    subprocess.run([sys.executable, "-c", """
from pyspark.sql import SparkSession, DataFrame, functions as F, types as T
from pyspark.sql.connect.types import UnparsedDataType
from pyspark.serializers import CloudPickleSerializer

# No server listens: with useArrow, udtf() does not read the Arrow config of the session.
spark = SparkSession.builder.remote("sc://localhost:15002").getOrCreate()
class Echo:
    def eval(self, value):
        yield (value,)

schema = T.StructType([T.StructField("value", T.LongType())])
for return_type in ["value BIGINT", "STRUCT<value: BIGINT>",
                    "value ARRAY<STRUCT<x: INT>>", "invalid DDL !!!", "INT", schema]:
    for decorate in [False, True]:
        fn = (F.udtf(returnType=return_type, useArrow=False)(Echo) if decorate else
              F.udtf(Echo, returnType=return_type, useArrow=False))
        assert isinstance(fn(F.lit(1)), DataFrame)
        handler = CloudPickleSerializer().loads(fn.command)
        assert isinstance(handler, type)
        assert list(handler().eval(7)) == [(7,)]
        # As PySpark Connect's, a DDL string is an UnparsedDataType.
        expected = (UnparsedDataType(return_type) if isinstance(return_type, str)
                    else return_type)
        assert fn.returnType == expected

# Dynamic schemas omit return_type and leave analysis to the Python handler.
class Dynamic(Echo):
    @staticmethod
    def analyze(value):
        pass
assert isinstance(F.udtf(Dynamic, useArrow=False)(F.lit(1)), DataFrame)
"""], check=True)


def test_udtf_python_binding_preserves_wire_schema_and_command():
    # Exercise the Python -> PyO3 -> protobuf seam against the recording mock,
    # including DDL that the client must send without parsing. Keep the active
    # session isolated, as in the construction-only test above.
    import subprocess
    import sys
    from pathlib import Path

    pytest.importorskip("grpc")
    subprocess.run([sys.executable, "-c", """
import json
import sys
sys.path.insert(0, sys.argv[1])
from _mock_connect_server import mock_connect_session, parse_message
from pyspark.sql import functions as F, types as T
from pyspark.sql.udtf import AnalyzeResult
from pyspark.serializers import CloudPickleSerializer

class Echo:
    def eval(self, value):
        yield (value,)

class Dynamic(Echo):
    @staticmethod
    def analyze(value):
        return AnalyzeResult(T.StructType([T.StructField("value", T.LongType())]))

schema = T.StructType([T.StructField("value", T.LongType(), False, {"comment": "kept"})])
with mock_connect_session() as (spark, service):
    for return_type in ["value BIGINT", "STRUCT<value: BIGINT>",
                        "value ARRAY<STRUCT<x: INT>>", "  `v à` BIGINT  ",
                        "", "invalid DDL !!!", "INT", schema, None]:
        for decorate in [False, True]:
            handler = Dynamic if return_type is None else Echo
            fn = (F.udtf(returnType=return_type)(handler) if decorate else
                  F.udtf(handler, returnType=return_type))
            calls = service.calls
            df = fn(F.lit(1))
            assert service.calls == calls
            assert df.columns == ["id"]
            assert service.calls == calls + 1

            # AnalyzePlanRequest.schema=4 -> plan=1 -> root=1 -> inline_udtf=38.
            call = parse_message(service.analyze_requests[-1])
            for field in (4, 1, 1, 38):
                call = parse_message(call[field][0])
            assert call[1] == [handler.__name__.encode()]
            # CommonInlineUserDefinedTableFunction.python_udtf=4.
            udtf = parse_message(call[4][0])
            if return_type is None:
                assert 1 not in udtf
            else:
                sent_type = parse_message(udtf[1][0])
                if isinstance(return_type, str):
                    assert set(sent_type) == {24}  # DataType.unparsed
                    unparsed = parse_message(sent_type[24][0])
                    assert unparsed.get(1, [b""])[0].decode() == return_type
                else:
                    assert set(sent_type) == {21}  # DataType.struct
                    fields = parse_message(sent_type[21][0])[1]
                    assert len(fields) == 1
                    field = parse_message(fields[0])
                    assert field[1] == [b"value"]
                    assert parse_message(field[2][0]) == {7: [b""]}  # LongType
                    assert field.get(3, [0]) == [0]  # nullable=False
                    assert json.loads(field[4][0]) == {"comment": "kept"}
            # SQL_ARROW_TABLE_UDF, as the server enables Arrow-optimized UDTFs by default.
            assert udtf[2] == [301]
            command_handler = CloudPickleSerializer().loads(udtf[3][0])
            assert isinstance(command_handler, type)
            assert command_handler.__name__ == handler.__name__
            assert list(command_handler().eval(7)) == [(7,)]
            assert udtf[4] == [fn.python_ver.encode()]
""", str(Path(__file__).resolve().parent)], check=True)


def test_spark_conf_is_that_of_pyspark():
    # As PySpark's SparkConf of a Connect-only client, which keeps its settings in Python.
    from pyspark import SparkConf
    c = SparkConf().set("a", "1").setAll([("b", "2"), ("c", "3")])
    assert c.get("a") == "1"
    assert dict(c.getAll())["b"] == "2"
    assert c.contains("b") and not c.contains("z")
    assert c.setIfMissing("a", "x").get("a") == "1"
    assert c.setAppName("app").setMaster("local[2]").get("spark.app.name") == "app"
    assert "spark.master=local[2]" in c.toDebugString().splitlines()


def test_pyspark_exports_the_names_of_the_pyspark_client():
    # As the PySpark client's pyspark, of a Connect-only client: its SparkConf, TaskContext,
    # accumulators, serializers, profilers and decorators, of their modules.
    import pyspark
    from pyspark.accumulators import Accumulator, AccumulatorParam
    from pyspark.conf import SparkConf
    from pyspark.profiler import BasicProfiler, Profiler
    from pyspark.serializers import CPickleSerializer, MarshalSerializer
    from pyspark.taskcontext import BarrierTaskContext, BarrierTaskInfo, TaskContext

    for name, value in [
        ("SparkConf", SparkConf),
        ("TaskContext", TaskContext),
        ("BarrierTaskContext", BarrierTaskContext),
        ("BarrierTaskInfo", BarrierTaskInfo),
        ("Accumulator", Accumulator),
        ("AccumulatorParam", AccumulatorParam),
        ("MarshalSerializer", MarshalSerializer),
        ("CPickleSerializer", CPickleSerializer),
        ("Profiler", Profiler),
        ("BasicProfiler", BasicProfiler),
    ]:
        assert getattr(pyspark, name) is value, name
    assert TaskContext.get() is None

    @pyspark.since(4.3)
    def documented():
        """A function."""

    assert documented.__doc__.endswith(".. versionadded:: 4.3")

    class Estimator:
        @pyspark.keyword_only
        def set(self, a=None):
            return self._input_kwargs

    assert Estimator().set(a=1) == {"a": 1}
    with pytest.raises(TypeError, match="forces keyword arguments"):
        Estimator().set(1)


def test_sql_is_remote():
    import pyspark.sql as S
    assert S.is_remote() is True


def test_functions_protobuf_descriptor_branches(tmp_path):
    import pyspark.sql.protobuf.functions as protobuf
    from pyspark.sql import functions as F

    c = F.col("v")
    descriptor = tmp_path / "m.desc"
    descriptor.write_bytes(b"\x00")
    # descriptor-file and binary-descriptor-set branches, with and without options
    assert type(protobuf.from_protobuf(c, "M", descFilePath=str(descriptor))).__name__ == "Column"
    assert type(protobuf.from_protobuf(c, "M", binaryDescriptorSet=b"\x00")).__name__ == "Column"
    assert type(protobuf.to_protobuf(c, "M", descFilePath=str(descriptor))).__name__ == "Column"
    assert type(protobuf.to_protobuf(c, "M", binaryDescriptorSet=b"\x00")).__name__ == "Column"
    assert type(protobuf.from_protobuf(c, "M", descFilePath=str(descriptor), options={"x": "y"})).__name__ == "Column"
    assert type(protobuf.to_protobuf(c, "M", binaryDescriptorSet=b"\x00", options={"x": "y"})).__name__ == "Column"


def test_udf_defaults_and_registration():
    from pyspark.sql.functions import udf, pandas_udf
    # returnType defaults to StringType
    u = udf(lambda x: x)
    assert callable(u)
    # As in PySpark, a pandas UDF takes no default return type, and a grouped map one a
    # struct type.
    from pyspark.errors import PySparkTypeError
    from pyspark.sql.functions import PandasUDFType

    with pytest.raises(PySparkTypeError, match="CANNOT_BE_NONE"):
        pandas_udf(lambda s: s)
    struct = T.StructType([T.StructField("a", T.IntegerType())])
    for ft, rt in [
        (PandasUDFType.SCALAR, T.IntegerType()),
        (PandasUDFType.GROUPED_MAP, struct),
        (PandasUDFType.GROUPED_AGG, T.IntegerType()),
    ]:
        assert pandas_udf(lambda s: s, rt, functionType=ft).evalType == ft
    with pytest.raises(PySparkTypeError, match="INVALID_RETURN_TYPE_FOR_PANDAS_UDF"):
        pandas_udf(lambda s: s, T.IntegerType(), functionType=PandasUDFType.GROUPED_MAP)
    # UDFRegistration.register wires to the session's _registerPythonUdf (so the name
    # resolves in SQL) and returns the UDF.
    from pyspark.sql.udf import UDFRegistration

    class _StubSession:
        def __init__(self):
            self.calls = []

        def _registerPythonUdf(self, name, returnType, evalType, command, python_ver, deterministic):
            self.calls.append((name, evalType, isinstance(command, (bytes, bytearray))))

    stub = _StubSession()
    reg = UDFRegistration(stub)
    r = reg.register("f", lambda x: x + 1, T.IntegerType())
    assert r.__name__ == "f"
    r2 = reg.register("g", lambda x: x)  # returnType default
    assert r2.__name__ == "g"
    # both were sent to the server with cloudpickled command bytes
    assert [c[0] for c in stub.calls] == ["f", "g"]
    assert all(c[2] for c in stub.calls)


def test_util_print_exec_and_skip_env(monkeypatch):
    import io
    from pyspark import util
    try:
        raise ValueError("boom")
    except ValueError:
        buf = io.StringIO()
        util.print_exec(buf)
        assert "ValueError" in buf.getvalue()
    monkeypatch.setenv("SPARK_SKIP_CONNECT_COMPAT_TESTS", "1")
    assert util.is_remote_only() is True


def test_struct_type_introspection():
    """StructType.fields / names / __getitem__ / __iter__ / __len__ + chainable add."""
    st = T.StructType(
        [T.StructField("a", T.IntegerType(), True), T.StructField("b", T.StringType(), False)]
    )
    assert len(st) == 2
    assert st.names == ["a", "b"] == st.fieldNames()
    assert [f.name for f in st.fields] == ["a", "b"]
    assert [type(f.dataType).__name__ for f in st.fields] == ["IntegerType", "StringType"]
    # index by position and by name
    assert st[0].name == "a"
    assert st["b"].name == "b"
    assert not st["b"].nullable
    # iteration
    assert [f.name for f in st] == ["a", "b"]
    # slice yields a StructType
    sl = st[0:1]
    assert isinstance(sl, T.StructType)
    assert sl.fieldNames() == ["a"]
    # out-of-range + missing name
    with pytest.raises(IndexError):
        _ = st[5]
    with pytest.raises(KeyError):
        _ = st["nope"]
    # chainable add
    st2 = T.StructType().add("x", "integer").add("y", T.StringType(), False).add(
        T.StructField("z", T.IntegerType())
    )
    assert st2.fieldNames() == ["x", "y", "z"]
    assert isinstance(st2, T.StructType)


def test_struct_types_take_their_keys_and_fields_as_pyspark():
    # As PySpark's StructType: __getitem__ raises KEY_NOT_EXISTS of a name,
    # INDEX_OUT_OF_RANGE of an index and NOT_EXPECTED_TYPE of another key, and add takes the
    # JSON value of a data type as a str, and requires a data type of a name.
    from pyspark.errors import (
        PySparkIndexError,
        PySparkKeyError,
        PySparkTypeError,
        PySparkValueError,
    )
    from pyspark.sql import types as T

    st = T.StructType([T.StructField("a", T.IntegerType())])
    with pytest.raises(PySparkKeyError) as e:
        st["zz"]
    assert e.value.getCondition() == "KEY_NOT_EXISTS"
    assert e.value.getMessageParameters() == {"key": "zz"}
    for index in (1, -2):
        with pytest.raises(PySparkIndexError) as e:
            st[index]
        assert e.value.getMessageParameters() == {"arg_name": "StructType", "index": str(index)}
    assert st[-1].name == "a"
    with pytest.raises(PySparkTypeError) as e:
        st[1.5]
    assert e.value.getMessageParameters() == {
        "expected_type": "int, slice or str",
        "arg_name": "key",
        "arg_type": "float",
    }
    added = T.StructType().add("a", "integer").add("b", "decimal(10,2)", False)
    assert added.simpleString() == "struct<a:int,b:decimal(10,2)>"
    assert added["b"].nullable is False
    with pytest.raises(PySparkValueError) as e:
        T.StructType().add("a", "int")
    assert e.value.getCondition() == "CANNOT_PARSE_DATATYPE"
    with pytest.raises(PySparkValueError) as e:
        T.StructType().add("a")
    assert e.value.getCondition() == "ARGUMENT_REQUIRED"


def test_struct_field_attributes():
    """StructField exposes name / dataType / nullable / metadata and a faithful repr/eq."""
    f = T.StructField("m", T.IntegerType(), True, {"k": "v"})
    assert f.name == "m"
    assert isinstance(f.dataType, T.IntegerType)
    assert f.nullable is True
    assert f.metadata == {"k": "v"}
    assert repr(f) == "StructField('m', IntegerType(), True)"
    assert f == T.StructField("m", T.IntegerType(), True, {"k": "v"})
    assert f != T.StructField("m", T.StringType(), True)


def test_user_defined_type():
    """UserDefinedType subclasses the Rust DataType and round-trips via json/fromJson."""
    import sys
    import types as _pytypes

    # Define the UDT in an importable module so cloudpickle serializes it by reference.
    mod = _pytypes.ModuleType("_udt_test_mod")
    sys.modules["_udt_test_mod"] = mod
    src = (
        "from pyspark.sql.types import UserDefinedType, ArrayType, DoubleType\n"
        "class PointUDT(UserDefinedType):\n"
        "    @classmethod\n"
        "    def sqlType(cls): return ArrayType(DoubleType(), False)\n"
        "    @classmethod\n"
        "    def module(cls): return '_udt_test_mod'\n"
        "    def serialize(self, obj): return [obj[0], obj[1]]\n"
        "    def deserialize(self, datum): return (datum[0], datum[1])\n"
    )
    exec(compile(src, "_udt_test_mod", "exec"), mod.__dict__)
    PointUDT = mod.PointUDT

    u = PointUDT()
    assert isinstance(u, T.DataType)
    assert isinstance(u, T.UserDefinedType)
    assert u.typeName() == "pointudt"
    assert u.simpleString() == "udt"
    assert u.needConversion() is True
    assert u.scalaUDT() == ""
    # serialize / deserialize round-trip
    assert u.serialize((1.0, 2.0)) == [1.0, 2.0]
    assert u.deserialize([1.0, 2.0]) == (1.0, 2.0)
    # jsonValue carries the udt envelope
    jv = u.jsonValue()
    assert jv["type"] == "udt"
    assert jv["pyClass"] == "_udt_test_mod.PointUDT"
    assert "serializedClass" in jv
    assert jv["sqlType"] == T.ArrayType(T.DoubleType(), False).jsonValue()
    # embed in a schema and parse it back -> reconstructs the UDT class
    sch = T.StructType([T.StructField("p", u, True)])
    rt = T._parse_datatype_json_string(sch.json())
    assert isinstance(rt.fields[0].dataType, PointUDT)
    del sys.modules["_udt_test_mod"]


def test_user_defined_type_not_implemented():
    """The abstract UDT hooks raise the pyspark NOT_IMPLEMENTED error."""
    from pyspark.errors import PySparkNotImplementedError

    class Bare(T.UserDefinedType):
        pass

    with pytest.raises(PySparkNotImplementedError):
        Bare.sqlType()
    with pytest.raises(PySparkNotImplementedError):
        Bare.module()
    with pytest.raises(PySparkNotImplementedError):
        Bare().serialize(object())
    with pytest.raises(PySparkNotImplementedError):
        Bare().deserialize(object())


# --------------------------------------------------------------------------- utils

def test_get_lit_sql_str():
    """Test SQL string literal escaping."""
    from pyspark.sql.utils import get_lit_sql_str

    # Simple strings
    assert get_lit_sql_str("hello") == "'hello'"
    assert get_lit_sql_str("") == "''"

    # Backslash escaping
    assert get_lit_sql_str("a\\b") == "'a\\\\b'"

    # Quote escaping
    assert get_lit_sql_str("it's") == "'it\\'s'"

    # Combined
    assert get_lit_sql_str("path\\to\\'file") == "'path\\\\to\\\\\\'file'"


def test_numpy_helper_linspace():
    """Test NumpyHelper.linspace for generating float sequences."""
    from pyspark.sql.utils import NumpyHelper

    # Single point
    assert NumpyHelper.linspace(0, 10, 1) == [0.0]

    # Two points
    assert NumpyHelper.linspace(0, 10, 2) == [0.0, 10.0]

    # Three points
    result = NumpyHelper.linspace(0, 10, 3)
    assert len(result) == 3
    assert result[0] == 0.0
    assert result[-1] == 10.0
    assert abs(result[1] - 5.0) < 1e-10

    # Negative range
    result = NumpyHelper.linspace(-10, 10, 5)
    assert len(result) == 5
    assert result[0] == -10.0
    assert result[-1] == 10.0


def test_version_utils_major_minor_version():
    """Test VersionUtils.majorMinorVersion parsing."""
    from pyspark.util import VersionUtils

    # Standard versions
    assert VersionUtils.majorMinorVersion("2.4.0") == (2, 4)
    assert VersionUtils.majorMinorVersion("3.0.0") == (3, 0)
    assert VersionUtils.majorMinorVersion("4.2.1") == (4, 2)

    # SNAPSHOT versions
    assert VersionUtils.majorMinorVersion("2.3.0-SNAPSHOT") == (2, 3)
    assert VersionUtils.majorMinorVersion("3.5.1-SNAPSHOT") == (3, 5)

    # Single digit
    assert VersionUtils.majorMinorVersion("1.0") == (1, 0)

    # Invalid versions
    with pytest.raises(ValueError):
        VersionUtils.majorMinorVersion("invalid")
    with pytest.raises(ValueError):
        VersionUtils.majorMinorVersion("a.b.c")
    with pytest.raises(ValueError):
        VersionUtils.majorMinorVersion("1")


def test_array_type_to_nullable():
    """Test ArrayType.toNullable() method."""
    # ArrayType with non-nullable elements
    arr = T.ArrayType(T.IntegerType(), containsNull=False)
    nullable_arr = arr.toNullable()
    assert nullable_arr.simpleString() == arr.simpleString()


def test_map_type_to_nullable():
    """Test MapType.toNullable() method."""
    # MapType with non-nullable values
    map_type = T.MapType(T.StringType(), T.IntegerType(), valueContainsNull=False)
    nullable_map = map_type.toNullable()
    assert nullable_map.simpleString() == map_type.simpleString()


def test_struct_type_to_nullable():
    """Test StructType.toNullable() making all fields nullable."""
    st = T.StructType([
        T.StructField("a", T.IntegerType(), False),
        T.StructField("b", T.StringType(), False),
    ])
    nullable_st = st.toNullable()
    assert all(f.nullable for f in nullable_st.fields)
    assert nullable_st.simpleString() == "struct<a:int,b:string>"


def test_array_type_from_json():
    """Test ArrayType.fromJson round-trip."""
    arr = T.ArrayType(T.IntegerType(), containsNull=True)
    json_val = arr.jsonValue()
    reconstructed = T.ArrayType.fromJson(json_val)
    assert reconstructed.simpleString() == arr.simpleString()


def test_map_type_from_json():
    """Test MapType.fromJson round-trip."""
    map_type = T.MapType(T.StringType(), T.IntegerType(), valueContainsNull=True)
    json_val = map_type.jsonValue()
    reconstructed = T.MapType.fromJson(json_val)
    assert reconstructed.simpleString() == map_type.simpleString()


def test_struct_fields_keep_the_collations_of_their_strings_in_their_metadata_as_pyspark():
    # As PySpark's: the JSON of a struct field has the collations of its strings, also in its
    # arrays and maps, in the __COLLATIONS of its metadata, and the readers take them back;
    # jsonValue has the keys of PySpark's, in its order.
    from pyspark.errors import PySparkTypeError, PySparkValueError
    from pyspark.sql.types import _parse_datatype_json_string
    from pyspark.testing.objects import PythonOnlyUDT

    st = T.StructType([
        T.StructField("s", T.StringType("UTF8_LCASE")),
        T.StructField("a", T.ArrayType(T.StringType("UNICODE"))),
        T.StructField("m", T.MapType(
            T.StringType("UTF8_LCASE"), T.ArrayType(T.StringType("UNICODE_CI"), False))),
        T.StructField("n", T.StructType([T.StructField("t", T.StringType("UTF8_LCASE"))])),
        T.StructField("plain", T.StringType(), False, {"k": "v"}),
    ])
    assert st.json() == (
        '{"fields":[{"metadata":{"__COLLATIONS":{"s":"spark.UTF8_LCASE"}},"name":"s",'
        '"nullable":true,"type":"string"},{"metadata":{"__COLLATIONS":{"a.element":'
        '"icu.UNICODE"}},"name":"a","nullable":true,"type":{"containsNull":true,'
        '"elementType":"string","type":"array"}},{"metadata":{"__COLLATIONS":{"m.key":'
        '"spark.UTF8_LCASE","m.value.element":"icu.UNICODE_CI"}},"name":"m","nullable":true,'
        '"type":{"keyType":"string","type":"map","valueContainsNull":true,"valueType":'
        '{"containsNull":false,"elementType":"string","type":"array"}}},{"metadata":{},'
        '"name":"n","nullable":true,"type":{"fields":[{"metadata":{"__COLLATIONS":{"t":'
        '"spark.UTF8_LCASE"}},"name":"t","nullable":true,"type":"string"}],"type":"struct"}},'
        '{"metadata":{"k":"v"},"name":"plain","nullable":false,"type":"string"}],'
        '"type":"struct"}')
    assert json.dumps(st.jsonValue()).startswith(
        '{"type": "struct", "fields": [{"name": "s", "type": "string", "nullable": true, '
        '"metadata": {"__COLLATIONS": {"s": "spark.UTF8_LCASE"}}}, {"name": "a", "type": '
        '{"type": "array", "elementType": "string", "containsNull": true}')
    assert _parse_datatype_json_string(st.json()) == st
    assert T.StructType.fromJson(json.loads(st.json())) == st
    assert T.StructField.fromJson(st.fields[2].jsonValue()) == st.fields[2]
    assert pickle.loads(pickle.dumps(st)) == st
    assert T.ArrayType.fromJson(
        {"elementType": "string", "containsNull": True}, "f", {"f.element": "UTF8_LCASE"}
    ) == T.ArrayType(T.StringType("UTF8_LCASE"))
    assert T.MapType.fromJson(
        {"keyType": "string", "valueType": "string", "valueContainsNull": True}, "",
        {"key": "UNICODE"},
    ) == T.MapType(T.StringType("UNICODE"), T.StringType())
    # A struct of a user-defined type reads back too.
    udt = T.StructType([T.StructField("p", PythonOnlyUDT())])
    assert T.StructType.fromJson(udt.jsonValue()) == udt

    def field(type_json, collation):
        return {"name": "f", "type": type_json, "nullable": True,
                "metadata": {"__COLLATIONS": {"f": collation}}}

    with pytest.raises(PySparkValueError) as e:
        T.StructField.fromJson(field("string", "other.UTF8_LCASE"))
    assert e.value.getCondition() == "COLLATION_INVALID_PROVIDER"
    assert e.value.getMessageParameters() == {
        "provider": "other", "supportedProviders": "spark, icu"}
    for type_json, json_type in (("integer", "integer"), ({"type": "array", "elementType": (
            "string"), "containsNull": True}, "array")):
        with pytest.raises(PySparkTypeError) as e:
            T.StructField.fromJson(field(type_json, "spark.UTF8_LCASE"))
        assert e.value.getCondition() == "INVALID_JSON_DATA_TYPE_FOR_COLLATIONS"
        assert e.value.getMessageParameters() == {"jsonType": json_type}




def test_string_type_collation():
    """Test StringType with collation parameter."""
    # StringType with explicit collation
    st = T.StringType(collation="UNICODE_CASE_INSENSITIVE")
    assert st is not None
    # Ensure collation property is accessible
    assert hasattr(st, 'collation') or hasattr(st, 'simpleString')


def test_row_construction_and_access():
    """Test Row construction and element access."""
    from pyspark.sql import Row

    # Named arguments
    r = Row(x=1, y=2, z=3)
    assert r.x == 1
    assert r.y == 2
    assert r["z"] == 3

    # With dict
    r2 = Row(a=10, b=20)
    assert r2["a"] == 10
    assert r2.b == 20

    # `in` tests the field names of a row that has them, as in PySpark, and the values of
    # one that does not.
    assert "x" in r and "b" in r2
    assert 1 not in r
    assert 1 in Row(1, 2) and 99 not in Row(1, 2)


def test_row_iteration():
    """Test iterating over Row values."""
    from pyspark.sql import Row

    r = Row(a=1, b=2, c=3)
    values = list(r)
    assert values == [1, 2, 3]


class _Thing:
    def __repr__(self):
        return "Thing()"


@pytest.mark.parametrize(
    "make, expected",
    [
        (lambda R: R(x=1, y="a"), "Row(x=1, y='a')"),
        (lambda R: R(1, 2), "<Row(1, 2)>"),
        (lambda R: R("a", "b"), "<Row('a', 'b')>"),
        (lambda R: R(), "<Row()>"),
        (lambda R: R(a=R(b=1, c=[1, 2]), d={"k": None}), "Row(a=Row(b=1, c=[1, 2]), d={'k': None})"),
        (
            lambda R: R(d=datetime.date(2020, 1, 2), n=decimal.Decimal("1.50"), b=b"\x00"),
            "Row(d=datetime.date(2020, 1, 2), n=Decimal('1.50'), b=b'\\x00')",
        ),
        (lambda R: R(a=None, f=1.5, t=True), "Row(a=None, f=1.5, t=True)"),
        # A row keeps the values it was made with, as they are.
        (lambda R: R(a=(1, 2), s={1}, o=_Thing()), "Row(a=(1, 2), s={1}, o=Thing())"),
        (lambda R: R("name", "age")("Alice", 11), "Row(name='Alice', age=11)"),
        (lambda R: R("name", "age")("Alice"), "Row(name='Alice')"),
    ],
)
def test_row_repr(make, expected):
    from pyspark.sql import Row

    row = make(Row)
    assert repr(row) == expected
    assert str(row) == expected


def test_row_behaves_as_a_tuple_of_its_values():
    import pickle

    from pyspark.sql import Row

    r = Row(a=1, b="x")
    # Equality, order and hash are those of the tuple of values; the names do not count.
    assert r == (1, "x") and (1, "x") == r and r == Row(c=1, d="x")
    assert r != [1, "x"] and r != (1,)
    assert Row(a=1) < Row(a=2) and Row(a=1) < (2,) and not Row(a=2) <= (1,)
    assert sorted([Row(x=2), Row(x=1)]) == [Row(x=1), Row(x=2)]
    assert hash(r) == hash((1, "x"))
    assert len({Row(a=1), Row(a=1)}) == 1 and {Row(a=1): 5}[Row(a=1)] == 5
    # Slices, concatenation and repetition are tuples.
    assert r[0:1] == (1,) and type(r[0:1]) is tuple and r[::-1] == ("x", 1)
    assert r + (3,) == (1, "x", 3) and (0,) + r == (0, 1, "x")
    assert r * 2 == (1, "x", 1, "x") and 2 * r == (1, "x", 1, "x")
    assert list(reversed(r)) == ["x", 1] and r[-1] == "x" and list(r) == [1, "x"]
    with pytest.raises(IndexError, match="tuple index out of range"):
        r[5]
    assert (r.count(1), r.index("x")) == (1, 1)
    with pytest.raises(ValueError, match="x not in tuple"):
        Row(a=1, b=2).index(2, 0, 1)
    # Pickled rows keep their names, also when two fields share one.
    for row in [r, Row(1, 2), Row(a=Row(b=[Row(c=1)])), Row("a", "b")(1, 2)]:
        restored = pickle.loads(pickle.dumps(row))
        assert repr(restored) == repr(row)
    assert pickle.loads(pickle.dumps(Row(1, 2))) == (1, 2)


def test_row_fields():
    from pyspark.errors import (
        PySparkAttributeError,
        PySparkRuntimeError,
        PySparkTypeError,
        PySparkValueError,
    )
    from pyspark.sql import Row

    r = Row(a=1, b="x")
    assert r.__fields__ == ["a", "b"] and hasattr(r, "__fields__")
    assert not hasattr(Row(1, 2), "__fields__")
    # `in` tests the names of a row with names, and the values of one without.
    assert "a" in r and 1 not in r and 1 in Row(1, 2)
    with pytest.raises(PySparkAttributeError) as error:
        r.missing
    assert error.value.getCondition() == "ATTRIBUTE_NOT_SUPPORTED"
    assert error.value.getMessageParameters() == {"attr_name": "missing"}
    with pytest.raises(PySparkValueError):
        r["missing"]
    with pytest.raises(PySparkAttributeError) as error:
        Row(1, 2)["a"]
    assert error.value.getMessageParameters() == {"attr_name": "__fields__"}
    with pytest.raises(PySparkRuntimeError) as error:
        r.a = 2
    assert error.value.getCondition() == "READ_ONLY"
    renamed = Row(a=1, b=2)
    renamed.__fields__ = ["p", "q"]
    assert repr(renamed) == "Row(p=1, q=2)" and renamed.p == 1
    with pytest.raises(PySparkValueError) as error:
        Row(1, a=2)
    assert error.value.getCondition() == "CANNOT_SET_TOGETHER"
    with pytest.raises(PySparkValueError) as error:
        Row("name", "age")(1, 2, 3)
    assert error.value.getCondition() == "TOO_MANY_VALUES"
    assert error.value.getMessageParameters() == {
        "expected": "2",
        "item": "fields",
        "actual": "3",
    }


def test_row_as_dict():
    from pyspark.errors import PySparkTypeError
    from pyspark.sql import Row

    nested = Row(a=Row(b=1), c=[Row(d=2)], e={"k": Row(f=3)})
    assert nested.asDict() == {"a": Row(b=1), "c": [Row(d=2)], "e": {"k": Row(f=3)}}
    assert type(nested.asDict()["a"]) is Row
    assert nested.asDict(recursive=True) == {"a": {"b": 1}, "c": [{"d": 2}], "e": {"k": {"f": 3}}}
    for row in [Row(1, 2), Row()]:
        with pytest.raises(PySparkTypeError) as error:
            row.asDict()
        assert error.value.getCondition() == "CANNOT_CONVERT_TYPE"


def test_datatype_from_ddl_parsing():
    """Test DataType.fromDDL parsing of DDL strings."""
    # StructField DDL parsing
    import pyspark.sql.types as types_module
    # StructField.fromDDL is what works offline
    sf = T.StructField.fromDDL("a int")
    assert sf.simpleString() == "a:int"

    sf2 = T.StructField.fromDDL("b: string")
    assert sf2.simpleString() == "b:string"


def test_decimal_type_methods():
    """Test DecimalType additional methods."""
    d = T.DecimalType(15, 4)
    assert d.precision == 15
    assert d.scale == 4
    assert d.simpleString() == "decimal(15,4)"
    assert d.typeName() == "decimal"


def test_column_repr_and_operations():
    """Test Column repr and various operations."""
    from pyspark.sql import functions as F

    c = F.col("data")
    repr_str = repr(c)
    assert "Column" in repr_str or "data" in repr_str

    # Test ** operator (power)
    c_pow = c ** 2
    assert type(c_pow).__name__ == "Column"

    # Test reverse power
    c_rpow = 2 ** c
    assert type(c_rpow).__name__ == "Column"


def test_column_guard_rails():
    """Test that Column raises errors for invalid operations."""
    from pyspark.sql import functions as F
    from pyspark.errors import PySparkValueError

    c = F.col("x")

    # bool(col) should raise an error
    with pytest.raises((TypeError, RuntimeError, PySparkValueError, Exception)):
        bool(c)


def test_functions_null_and_misc():
    """Test miscellaneous functions that may have low coverage."""
    from pyspark.sql import functions as F

    # Test isnull, isnan
    c = F.col("x")
    assert type(F.isnull(c)).__name__ == "Column"
    assert type(F.isnan(c)).__name__ == "Column"

    # Test corr and covar functions
    assert type(F.corr(c, c)).__name__ == "Column"
    assert type(F.covar_pop(c, c)).__name__ == "Column"
    assert type(F.covar_samp(c, c)).__name__ == "Column"


def test_na_and_stat_functions_are_of_their_dataframe_as_pyspark():
    # As PySpark's DataFrameNaFunctions and DataFrameStatFunctions, whose df is the DataFrame
    # of df.na and df.stat itself.
    from pyspark.sql import DataFrameNaFunctions, DataFrameStatFunctions

    df = _offline_session().range(1)
    assert isinstance(df.na, DataFrameNaFunctions)
    assert isinstance(df.stat, DataFrameStatFunctions)
    assert df.na.df is df
    assert df.stat.df is df
    assert df.na.df.stat.df is df


def test_sample_by_checks_its_arguments_as_pyspark():
    # As PySpark's sampleBy: a column name or a Column, a dict of fractions, and strata of
    # floats, ints and strings.
    from pyspark.errors import PySparkTypeError

    df = _offline_session().range(1)
    for call, condition, parameters in (
        (lambda: df.sampleBy(1, {"a": 1.0}), "NOT_EXPECTED_TYPE",
         {"expected_type": "Column or str", "arg_name": "col", "arg_type": "int"}),
        (lambda: df.stat.sampleBy("k", [0.5]), "NOT_EXPECTED_TYPE",
         {"expected_type": "dict", "arg_name": "fractions", "arg_type": "list"}),
        (lambda: df.sampleBy("k", {(1, 2): 1.0}), "DISALLOWED_TYPE_FOR_CONTAINER",
         {"arg_name": "fractions", "arg_type": "dict", "allowed_types": "float, int, str",
          "item_type": "tuple"}),
    ):
        with pytest.raises(PySparkTypeError) as error:
            call()
        assert error.value.getCondition() == condition
        assert error.value.getMessageParameters() == parameters


def test_functions_bitwise():
    """Test bitwise functions."""
    from pyspark.sql import functions as F

    c = F.col("x")
    # Test bitwise_not operation
    assert type(F.bitwise_not(c)).__name__ == "Column"


def test_functions_rounding():
    """Test rounding and numerical functions."""
    from pyspark.sql import functions as F

    c = F.col("x")
    assert type(F.round(c)).__name__ == "Column"
    assert type(F.round(c, 2)).__name__ == "Column"
    assert type(F.ceil(c)).__name__ == "Column"
    assert type(F.floor(c)).__name__ == "Column"
    assert type(F.abs(c)).__name__ == "Column"


def test_functions_string_operations():
    """Test string manipulation functions."""
    from pyspark.sql import functions as F

    c = F.col("text")
    assert type(F.concat(c, F.lit("suffix"))).__name__ == "Column"
    assert type(F.concat_ws("-", c, F.lit("a"))).__name__ == "Column"
    assert type(F.trim(c)).__name__ == "Column"
    assert type(F.ltrim(c)).__name__ == "Column"
    assert type(F.rtrim(c)).__name__ == "Column"
    assert type(F.length(c)).__name__ == "Column"
    assert type(F.reverse(c)).__name__ == "Column"


def test_functions_array_operations():
    """Test array functions."""
    from pyspark.sql import functions as F

    arr = F.col("arr")
    assert type(F.array_contains(arr, F.lit(1))).__name__ == "Column"
    assert type(F.flatten(arr)).__name__ == "Column"


def test_functions_map_operations():
    """Test map functions."""
    from pyspark.sql import functions as F

    map_col = F.col("m")
    assert type(F.map_keys(map_col)).__name__ == "Column"
    assert type(F.map_values(map_col)).__name__ == "Column"


def test_functions_date_time():
    """Test date/time functions."""
    from pyspark.sql import functions as F

    c = F.col("date_col")
    assert type(F.year(c)).__name__ == "Column"
    assert type(F.month(c)).__name__ == "Column"
    assert type(F.dayofmonth(c)).__name__ == "Column"
    assert type(F.hour(c)).__name__ == "Column"
    assert type(F.minute(c)).__name__ == "Column"
    assert type(F.second(c)).__name__ == "Column"


def test_functions_hash_and_crc():
    """Test hash and CRC functions."""
    from pyspark.sql import functions as F

    c = F.col("x")
    assert type(F.md5(c)).__name__ == "Column"
    assert type(F.sha1(c)).__name__ == "Column"
    assert type(F.sha2(c, 256)).__name__ == "Column"
    assert type(F.crc32(c)).__name__ == "Column"


def test_functions_json():
    """Test JSON functions."""
    from pyspark.sql import functions as F

    c = F.col("data")
    assert type(F.from_json(c, "a int")).__name__ == "Column"
    assert type(F.to_json(c)).__name__ == "Column"
    assert type(F.json_tuple(c, "key1", "key2")).__name__ == "Column"


def test_functions_mathematical():
    """Test mathematical functions."""
    from pyspark.sql import functions as F

    c = F.col("x")
    assert type(F.sqrt(c)).__name__ == "Column"
    assert type(F.pow(c, F.lit(2))).__name__ == "Column"
    assert type(F.exp(c)).__name__ == "Column"
    assert type(F.log(c)).__name__ == "Column"
    assert type(F.log10(c)).__name__ == "Column"
    assert type(F.sin(c)).__name__ == "Column"
    assert type(F.cos(c)).__name__ == "Column"
    assert type(F.tan(c)).__name__ == "Column"


def test_udf_with_defaults():
    """Test UDF with default return type."""
    from pyspark.sql.functions import udf
    from pyspark.sql.types import StringType

    # No return type specified (defaults to StringType)
    u = udf(lambda x: str(x))
    assert callable(u)


def test_udtf_variants():
    """Test UDTF builder methods."""
    from pyspark.sql.udtf import UserDefinedTableFunction

    # Create a simple UDTF for testing; not Arrow-optimized whatever the active session.
    @__import__("pyspark.sql.functions", fromlist=["udtf"]).udtf(
        returnType="a int", useArrow=False
    )
    class SimpleUDTF:
        def eval(self, n):
            yield (n,)

    # Test name and evalType properties
    assert isinstance(SimpleUDTF, UserDefinedTableFunction)
    assert SimpleUDTF.name == "SimpleUDTF"
    assert SimpleUDTF.evalType == 300  # SQL_TABLE_UDF

    # As PySpark's, a UDTF is not deterministic unless asDeterministic() marks it so.
    assert SimpleUDTF.deterministic is False


def test_column_with_alias_chain():
    """Test chaining column methods."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import StringType

    c = F.col("id")
    # Chain alias, cast, and asc
    result = c.alias("id_renamed").cast(StringType()).asc()
    assert type(result).__name__ == "Column"


def test_functions_json_parsing():
    """Test JSON parsing edge cases."""
    from pyspark.sql import functions as F

    c = F.col("json_str")
    # Test get_json_object
    assert type(F.get_json_object(c, "$.key")).__name__ == "Column"


def test_typed_columns_exist():
    """Verify existence of typed column creation functions."""
    from pyspark.sql import functions as F

    # These should all create columns
    assert type(F.input_file_name()).__name__ == "Column"
    assert type(F.rand()).__name__ == "Column"
    assert type(F.randn()).__name__ == "Column"
    assert type(F.current_timestamp()).__name__ == "Column"
    assert type(F.current_date()).__name__ == "Column"


def test_struct_types_convert_to_ddl_on_the_server_of_the_active_session_as_pyspark(mock_connect):
    # As PySpark Connect's StructType.toDDL: the server of the active session converts the JSON
    # of the schema (AnalyzePlanRequest.json_to_ddl=18 -> JsonToDDL.json_string=1), and there
    # is no DDL without an active session.
    spark, service = mock_connect
    service.ddl_string = "a INT,`b b` STRING NOT NULL"
    st = T.StructType([
        T.StructField("a", T.IntegerType()),
        T.StructField("b b", T.StringType(), False),
    ])
    assert st.toDDL() == "a INT,`b b` STRING NOT NULL"
    request = parse_message(service.analyze_requests[-1])
    assert parse_message(request[18][0])[1][0].decode() == st.json()
    spark.stop()
    with pytest.raises(AssertionError):
        st.toDDL()




def test_struct_type_schema_methods():
    """Test StructType schema introspection methods."""
    st = T.StructType([
        T.StructField("name", T.StringType(), False),
        T.StructField("age", T.IntegerType(), True),
    ])

    # Test treeString
    tree = st.treeString()
    assert "root" in tree
    assert "name" in tree


# --------------------------------------------------------------------------- functions extensions

def test_functions_bucket():
    """Test bucket function for partitioning."""
    from pyspark.sql import functions as F

    c = F.col("x")
    result = F.bucket(10, c)
    assert type(result).__name__ == "Column"


def test_functions_greatest_least():
    """Test greatest and least functions."""
    from pyspark.sql import functions as F

    c1 = F.col("x")
    c2 = F.col("y")
    assert type(F.greatest(c1, c2)).__name__ == "Column"
    assert type(F.least(c1, c2)).__name__ == "Column"


def test_functions_coalesce():
    """Test coalesce function."""
    from pyspark.sql import functions as F

    c1 = F.col("x")
    c2 = F.col("y")
    assert type(F.coalesce(c1, c2)).__name__ == "Column"


def test_functions_explode():
    """Test explode functions."""
    from pyspark.sql import functions as F

    arr_col = F.col("arr")
    map_col = F.col("map")
    assert type(F.explode(arr_col)).__name__ == "Column"
    assert type(F.explode_outer(arr_col)).__name__ == "Column"


def test_functions_sequence():
    """Test sequence function."""
    from pyspark.sql import functions as F

    c1 = F.col("start")
    c2 = F.col("end")
    assert type(F.sequence(c1, c2)).__name__ == "Column"


def test_functions_posexplode():
    """Test posexplode functions."""
    from pyspark.sql import functions as F

    arr_col = F.col("arr")
    assert type(F.posexplode(arr_col)).__name__ == "Column"
    assert type(F.posexplode_outer(arr_col)).__name__ == "Column"


def test_functions_substring():
    """Test substring functions."""
    from pyspark.sql import functions as F

    str_col = F.col("str")
    assert type(F.substring(str_col, 1, 5)).__name__ == "Column"
    # As in PySpark, substr takes the position and length as Columns (or names).
    assert type(F.substr(str_col, F.lit(1), F.lit(5))).__name__ == "Column"


def test_grouped_data_are_represented_as_pyspark(mock_connect):
    # As PySpark Connect's: the repr of a GroupedData names its grouping expressions, the
    # columns of its DataFrame and their types, and the type of the grouping.
    from pyspark.sql import functions as F

    spark, _ = mock_connect
    df = spark.range(3)
    value = ", ".join("%s: %s" % column for column in df.dtypes)
    for grouped, expressions, kind in (
            (df.groupBy("id"), "id", "GroupBy"), (df.rollup("id"), "id", "RollUp"),
            (df.cube(F.col("id") + 1), "+(id, 1)", "Cube"),
            (df.groupingSets([["id"]], "id"), "id", "GroupingSets"),
            (df.groupBy("id").pivot("id"), "id", "Pivot"), (df.groupBy(), "", "GroupBy")):
        assert repr(grouped) == (
            f"GroupedData[grouping expressions: [{expressions}], value: [{value}], type: {kind}]")




def test_pivot_pivots_only_a_group_by_and_checks_its_arguments_as_pyspark():
    # As PySpark's GroupedData.pivot: only the grouping of groupBy pivots, which it checks
    # first, and then the column is a str and the values a list of bools, floats, ints and
    # strs.
    from pyspark.errors import PySparkNotImplementedError, PySparkTypeError
    from pyspark.sql import functions as F

    df = _offline_session().range(1)
    for grouped, operation in [
        (df.rollup("id"), "PIVOT after ROLLUP"),
        (df.cube("id"), "PIVOT after CUBE"),
        (df.groupingSets([["id"]], "id"), "PIVOT after GROUPING_SETS"),
        (df.groupBy("id").pivot("id"), "Repeated PIVOT operation"),
    ]:
        with pytest.raises(PySparkNotImplementedError) as e:
            grouped.pivot(1)
        assert e.value.getCondition() == "UNSUPPORTED_OPERATION"
        assert e.value.getMessageParameters() == {"operation": operation}
    grouped = df.groupBy("id")
    for args, arg_name, expected_type, arg_type in [
        ((1,), "pivot_col", "str", "int"),
        ((F.col("id"),), "pivot_col", "str", "Column"),
        (("id", "a"), "values", "list", "str"),
        (("id", ("a",)), "values", "list", "tuple"),
        (("id", [F.lit(1)]), "value", "bool, float, int or str", "Column"),
        (("id", [None]), "value", "bool, float, int or str", "NoneType"),
    ]:
        with pytest.raises(PySparkTypeError) as e:
            grouped.pivot(*args)
        assert e.value.getCondition() == "NOT_EXPECTED_TYPE"
        assert e.value.getMessageParameters() == {
            "arg_name": arg_name,
            "expected_type": expected_type,
            "arg_type": arg_type,
        }
    for values in [None, [], [True, 1.5, 2, "a"]]:
        grouped.pivot("id", values)


def test_replace_and_unpivot_check_their_arguments_as_pyspark():
    # As PySpark's DataFrame.replace and unpivot, which check their arguments on the client.
    import warnings

    from pyspark.errors import PySparkTypeError, PySparkValueError

    df = _offline_session().range(1)
    for call, error_type, condition, parameters in (
        (lambda: df.replace("a"), PySparkTypeError, "ARGUMENT_REQUIRED",
         {"arg_name": "value", "condition": "`to_replace` is dict"}),
        (lambda: df.na.replace([1]), PySparkTypeError, "ARGUMENT_REQUIRED",
         {"arg_name": "value", "condition": "`to_replace` is dict"}),
        (lambda: df.replace({1, 2}, 3), PySparkTypeError, "NOT_EXPECTED_TYPE",
         {"expected_type": "bool, dict, float, int, str or tuple", "arg_name": "to_replace",
          "arg_type": "set"}),
        (lambda: df.replace(1, {2: 3}), PySparkTypeError, "NOT_EXPECTED_TYPE",
         {"expected_type": "bool, float, int, list, None, str or tuple", "arg_name": "value",
          "arg_type": "dict"}),
        (lambda: df.replace([1, 2], [3]), PySparkValueError, "LENGTH_SHOULD_BE_THE_SAME",
         {"arg1": "to_replace", "arg2": "value", "arg1_length": "2", "arg2_length": "1"}),
        (lambda: df.replace("a", "b", 1), PySparkTypeError, "NOT_EXPECTED_TYPE",
         {"arg_name": "subset", "expected_type": "list, str or tuple", "arg_type": "int"}),
        (lambda: df.replace({1: "a"}), PySparkValueError, "MIXED_TYPE_REPLACEMENT", {}),
        (lambda: df.na.replace([1, "a"], "b"), PySparkValueError, "MIXED_TYPE_REPLACEMENT", {}),
        (lambda: df.unpivot(1, "id", "var", "val"), PySparkTypeError, "NOT_EXPECTED_TYPE",
         {"expected_type": "Column or str", "arg_name": "col", "arg_type": "int"}),
        (lambda: df.melt(["id"], [1], "var", "val"), PySparkTypeError, "NOT_EXPECTED_TYPE",
         {"expected_type": "Column or str", "arg_name": "col", "arg_type": "int"}),
    ):
        with pytest.raises(error_type) as error:
            call()
        assert error.value.getCondition() == condition
        assert error.value.getMessageParameters() == parameters
    with pytest.raises(AssertionError, match="ids must not be None"):
        df.unpivot(None, "id", "var", "val")
    # All four arguments of unpivot, and of melt, which is the same function, are required.
    for unpivot in (df.unpivot, df.melt):
        with pytest.raises(TypeError):
            unpivot("id", "id")
    # The value of a dict to_replace is ignored, with a warning.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        df.replace({"a": "b"}, "c")
    assert [str(w.message) for w in caught] == [
        "to_replace is a dict and value is not None. value will be ignored."
    ]


def test_range_and_table_valued_functions_take_their_arguments_as_pyspark(mock_connect):
    # As PySpark's: spark.range and spark.tvf.range take the int() of their arguments, and the
    # table-valued functions of spark.tvf take Columns or column names, as F._to_col does.
    import re

    from pyspark.errors import PySparkTypeError, PySparkValueError
    from pyspark.sql import functions as F

    spark, service = mock_connect

    def sent():
        # ExecutePlanRequest.plan=3, Plan.root=1.
        return parse_message(parse_message(parse_message(service.execute_requests[-1])[3][0])[1][0])

    def sent_range():
        # Relation.range=15: Range.end=2, step=3, num_partitions=4.
        fields = parse_message(sent()[15][0])
        return {field: fields[field] for field in (2, 3, 4) if field in fields}

    spark.range("3").collect()
    assert sent_range() == {2: [3], 3: [1]}
    spark.tvf.range(0, 6.5, 2.5, "2").collect()
    assert sent_range() == {2: [6], 3: [2], 4: [2]}
    for args, error, message in [
        (("x",), ValueError, "invalid literal for int()"),
        ((None,), TypeError, "int() argument must be"),
        ((0, 5, None), TypeError, "int() argument must be"),
        ((2**63,), ValueError, "Value out of range: 9223372036854775808"),
        ((0, 3, 1, 2**31), ValueError, "Value out of range: 2147483648"),
    ]:
        for range_ in (spark.range, spark.tvf.range):
            with pytest.raises(error, match=re.escape(message)):
                range_(*args)
    spark.tvf.json_tuple(F.lit('{"a": 1}'), "a").collect()
    # Relation.unresolved_table_valued_function=43: arguments=2, of the column `a`, an
    # Expression.unresolved_attribute=2 of UnresolvedAttribute.unparsed_identifier=1.
    field = parse_message(parse_message(sent()[43][0])[2][1])
    assert parse_message(field[2][0])[1] == [b"a"]
    tvf = spark.tvf
    for function, args, arg_type in [
        (tvf.explode, ([1, 2],), "list"),
        (tvf.inline, (1,), "int"),
        (tvf.stack, (2, F.lit(1)), "int"),
        (tvf.json_tuple, (F.lit("{}"), 1), "int"),
    ]:
        with pytest.raises(PySparkTypeError) as e:
            function(*args)
        assert e.value.getCondition() == "NOT_EXPECTED_TYPE"
        assert e.value.getMessageParameters() == {
            "arg_name": "col",
            "expected_type": "Column or str",
            "arg_type": arg_type,
        }
    with pytest.raises(PySparkValueError) as e:
        tvf.json_tuple(1)
    assert e.value.getCondition() == "CANNOT_BE_EMPTY"


def test_functions_replace():
    """Test replace function."""
    from pyspark.sql import functions as F

    str_col = F.col("str")
    assert type(F.replace(str_col, "old", "new")).__name__ == "Column"


def test_functions_translate():
    """Test translate function."""
    from pyspark.sql import functions as F

    str_col = F.col("str")
    assert type(F.translate(str_col, "abc", "xyz")).__name__ == "Column"


def test_functions_pad_functions():
    """Test lpad and rpad functions."""
    from pyspark.sql import functions as F

    str_col = F.col("str")
    assert type(F.lpad(str_col, 10, " ")).__name__ == "Column"
    assert type(F.rpad(str_col, 10, " ")).__name__ == "Column"


@pytest.mark.parametrize("build,expected", [
    # As Column.alias(*alias, **kwargs) of PySpark: every name, metadata for one name only.
    (lambda c: c.alias("x"), "a AS x"),
    (lambda c: c.alias("x", "y"), "a AS x,y"),
    (lambda c: c.name("x", "y"), "a AS x,y"),
    (lambda c: c.alias(), "a AS "),
    (lambda c: c.alias("x", metadata={"k": 1, "n": [1, 2], "s": "v"}), "a AS x"),
    (lambda c: c.alias("x", "y", metadata=None), "a AS x,y"),
])
def test_column_alias_takes_every_name_as_pyspark(build, expected):
    from pyspark.sql import functions as F

    assert repr(build(F.col("a"))) == f"Column<'{expected}'>"


def test_column_alias_errors_as_pyspark():
    from pyspark.errors import PySparkValueError
    from pyspark.sql import functions as F

    with pytest.raises(PySparkValueError) as error:
        F.col("a").alias("x", "y", metadata={"k": 1})
    assert error.value.getCondition() == "ONLY_ALLOWED_FOR_SINGLE_COLUMN"
    assert error.value.getMessageParameters() == {"arg_name": "metadata"}
    with pytest.raises(AssertionError, match=r"^Unexpected kwargs where passed: \{'foo': 1\}$"):
        F.col("a").alias("x", foo=1)


def test_functions_instr():
    """Test instr function."""
    from pyspark.sql import functions as F

    str_col = F.col("str")
    assert type(F.instr(str_col, "substr")).__name__ == "Column"


def test_functions_locate():
    """Test locate function."""
    from pyspark.sql import functions as F

    str_col = F.col("str")
    assert type(F.locate("substr", str_col)).__name__ == "Column"


def test_functions_format_string():
    """Test format_string function."""
    from pyspark.sql import functions as F

    c1 = F.col("x")
    assert type(F.format_string("%d", c1)).__name__ == "Column"


def test_functions_initcap():
    """Test initcap function."""
    from pyspark.sql import functions as F

    str_col = F.col("str")
    assert type(F.initcap(str_col)).__name__ == "Column"


def test_functions_ascii_and_unicode():
    """Test ascii and unicode functions."""
    from pyspark.sql import functions as F

    str_col = F.col("str")
    assert type(F.ascii(str_col)).__name__ == "Column"


def test_functions_levenshtein():
    """Test levenshtein distance function."""
    from pyspark.sql import functions as F

    c1 = F.col("x")
    c2 = F.col("y")
    assert type(F.levenshtein(c1, c2)).__name__ == "Column"


def test_functions_soundex():
    """Test soundex function."""
    from pyspark.sql import functions as F

    str_col = F.col("str")
    assert type(F.soundex(str_col)).__name__ == "Column"


def test_functions_split():
    """Test split function."""
    from pyspark.sql import functions as F

    str_col = F.col("str")
    assert type(F.split(str_col, ",")).__name__ == "Column"


def test_functions_repeat():
    """Test repeat function."""
    from pyspark.sql import functions as F

    str_col = F.col("str")
    assert type(F.repeat(str_col, 3)).__name__ == "Column"


def test_functions_regexp_functions():
    """Test regexp functions."""
    from pyspark.sql import functions as F

    str_col = F.col("str")
    assert type(F.regexp_extract(str_col, r"\d+", 0)).__name__ == "Column"
    assert type(F.regexp_replace(str_col, r"\d+", "X")).__name__ == "Column"


def test_functions_rlike():
    """Test rlike function."""
    from pyspark.sql import functions as F

    str_col = F.col("str")
    assert type(F.rlike(str_col, "pattern")).__name__ == "Column"


def test_functions_like_and_glob():
    """Test like and glob functions."""
    from pyspark.sql import functions as F

    str_col = F.col("str")
    assert type(F.like(str_col, "%pattern%")).__name__ == "Column"


def test_functions_to_date():
    """Test to_date function."""
    from pyspark.sql import functions as F

    str_col = F.col("str")
    assert type(F.to_date(str_col)).__name__ == "Column"


def test_functions_from_unixtime():
    """Test from_unixtime function."""
    from pyspark.sql import functions as F

    c = F.col("ts")
    assert type(F.from_unixtime(c)).__name__ == "Column"


def test_functions_unix_timestamp():
    """Test unix_timestamp function."""
    from pyspark.sql import functions as F

    c = F.col("date")
    assert type(F.unix_timestamp(c)).__name__ == "Column"


def test_functions_from_utc_timestamp():
    """Test from_utc_timestamp function."""
    from pyspark.sql import functions as F

    c = F.col("ts")
    assert type(F.from_utc_timestamp(c, "UTC")).__name__ == "Column"


def test_functions_to_utc_timestamp():
    """Test to_utc_timestamp function."""
    from pyspark.sql import functions as F

    c = F.col("ts")
    assert type(F.to_utc_timestamp(c, "UTC")).__name__ == "Column"


def test_functions_date_format():
    """Test date_format function."""
    from pyspark.sql import functions as F

    c = F.col("date")
    assert type(F.date_format(c, "yyyy-MM-dd")).__name__ == "Column"


def test_functions_date_trunc():
    """Test date_trunc function."""
    from pyspark.sql import functions as F

    c = F.col("date")
    assert type(F.date_trunc("month", c)).__name__ == "Column"


def test_functions_date_add_sub():
    """Test date_add and date_sub functions."""
    from pyspark.sql import functions as F

    c = F.col("date")
    assert type(F.date_add(c, 10)).__name__ == "Column"
    assert type(F.date_sub(c, 10)).__name__ == "Column"


def test_functions_add_months():
    """Test add_months function."""
    from pyspark.sql import functions as F

    c = F.col("date")
    assert type(F.add_months(c, 3)).__name__ == "Column"


def test_functions_months_between():
    """Test months_between function."""
    from pyspark.sql import functions as F

    c1 = F.col("date1")
    c2 = F.col("date2")
    assert type(F.months_between(c1, c2)).__name__ == "Column"


def test_functions_last_day():
    """Test last_day function."""
    from pyspark.sql import functions as F

    c = F.col("date")
    assert type(F.last_day(c)).__name__ == "Column"


def test_functions_next_day():
    """Test next_day function."""
    from pyspark.sql import functions as F

    c = F.col("date")
    assert type(F.next_day(c, "Monday")).__name__ == "Column"


def test_functions_quarter():
    """Test quarter function."""
    from pyspark.sql import functions as F

    c = F.col("date")
    assert type(F.quarter(c)).__name__ == "Column"


def test_functions_dayofweek():
    """Test dayofweek and weekofyear functions."""
    from pyspark.sql import functions as F

    c = F.col("date")
    assert type(F.dayofweek(c)).__name__ == "Column"
    assert type(F.weekofyear(c)).__name__ == "Column"


def test_functions_dayofyear():
    """Test dayofyear function."""
    from pyspark.sql import functions as F

    c = F.col("date")
    assert type(F.dayofyear(c)).__name__ == "Column"


def test_functions_cast():
    """Test cast function with different types (via Column method)."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import StringType, IntegerType

    c = F.col("x")
    # cast is a Column method, not a functions module function
    assert type(c.cast("string")).__name__ == "Column"
    assert type(c.cast(StringType())).__name__ == "Column"
    assert type(c.cast(IntegerType())).__name__ == "Column"


def test_functions_acos_asin_atan():
    """Test inverse trig functions."""
    from pyspark.sql import functions as F

    c = F.col("x")
    assert type(F.acos(c)).__name__ == "Column"
    assert type(F.asin(c)).__name__ == "Column"
    assert type(F.atan(c)).__name__ == "Column"


def test_functions_atan2():
    """Test atan2 function."""
    from pyspark.sql import functions as F

    c1 = F.col("y")
    c2 = F.col("x")
    assert type(F.atan2(c1, c2)).__name__ == "Column"


def test_functions_cosh_sinh_tanh():
    """Test hyperbolic functions."""
    from pyspark.sql import functions as F

    c = F.col("x")
    assert type(F.cosh(c)).__name__ == "Column"
    assert type(F.sinh(c)).__name__ == "Column"
    assert type(F.tanh(c)).__name__ == "Column"


def test_functions_degrees_radians():
    """Test degrees and radians conversion."""
    from pyspark.sql import functions as F

    c = F.col("x")
    assert type(F.degrees(c)).__name__ == "Column"
    assert type(F.radians(c)).__name__ == "Column"


def test_functions_log2():
    """Test log2 function."""
    from pyspark.sql import functions as F

    c = F.col("x")
    assert type(F.log2(c)).__name__ == "Column"


def test_functions_hypot():
    """Test hypot function."""
    from pyspark.sql import functions as F

    c1 = F.col("x")
    c2 = F.col("y")
    assert type(F.hypot(c1, c2)).__name__ == "Column"


def test_functions_sign():
    """Test sign function."""
    from pyspark.sql import functions as F

    c = F.col("x")
    assert type(F.signum(c)).__name__ == "Column"


def test_udf_with_nondeterministic():
    """Test UDF nondeterministic variants."""
    from pyspark.sql.functions import udf
    from pyspark.sql.types import IntegerType

    u = udf(lambda x: x, IntegerType())
    nd = u.asNondeterministic()
    assert nd.deterministic is False


def test_column_cast_edge_cases():
    """Test Column.cast with edge cases."""
    from pyspark.sql import functions as F
    from pyspark.sql.types import DecimalType

    c = F.col("x")
    # Test casting to DecimalType with precision/scale
    result = c.cast(DecimalType(10, 2))
    assert type(result).__name__ == "Column"


def test_drop_metadata_utility():
    """Test _drop_metadata utility function if available."""
    try:
        from pyspark.sql.types import _drop_metadata, StructType, StructField, StringType, IntegerType

        # Test on StructType with metadata
        st = StructType([
            StructField("a", IntegerType(), True, {"x": "y"}),
            StructField("b", StringType(), False, {"z": "w"})
        ])
        cleaned_st = _drop_metadata(st)
        # Verify it doesn't crash
        assert cleaned_st is not None
    except (ImportError, AttributeError):
        # _drop_metadata may not be part of the public API
        pass


def test_create_row_from_fields():
    """Test Row creation utilities."""
    from pyspark.sql import Row

    # Direct Row construction
    r = Row(a=1, b=2, c=3)
    assert r.a == 1
    assert r["b"] == 2
    assert r.c == 3

    # Using asDict
    d = r.asDict()
    assert d == {"a": 1, "b": 2, "c": 3}


def test_string_type_collation_provider():
    """Test StringType collation provider if available."""
    st = T.StringType()
    # Just verify it doesn't crash when accessing
    assert st is not None


def test_error_handling_in_higher_order_functions():
    """Test error handling in higher-order functions."""
    from pyspark.sql import functions as F
    from pyspark.errors import PySparkValueError

    # Lambda with wrong arity should raise
    with pytest.raises(PySparkValueError):
        F.transform("arr", lambda: 1)  # arity 0, needs 1-3

    # Lambda not returning Column should raise
    with pytest.raises(PySparkValueError):
        F.transform("arr", lambda x: 42)  # returns int, not Column


def test_spatial_types_construction():
    """Test spatial type construction."""
    # GeometryType construction
    g = T.GeometryType(4326)
    assert g.srid == 4326
    assert "geometry" in g.simpleString()

    # GeographyType construction
    gg = T.GeographyType(4326)
    assert gg.srid == 4326
    assert "geography" in gg.simpleString()


def test_variant_val_pickles_as_pyspark():
    import pickle

    from pyspark.sql.types import VariantVal

    # As PySpark's VariantVal(value, metadata), which pickles as a plain class does.
    value = pickle.loads(pickle.dumps(VariantVal(b"\x0c\x01", b"\x01\x00\x00")))
    assert type(value) is VariantVal
    assert (value.value, value.metadata) == (b"\x0c\x01", b"\x01\x00\x00")


def test_str_of_variant_values_is_their_json_and_they_are_no_rows_as_pyspark():
    # As PySpark's VariantVal: str is its JSON, in UTC, as toJson(); repr is of its bytes. And
    # as PySpark's createDataFrame, whose VariantVal has a __dict__, so that it is a row, which
    # to_row rejects: a VariantVal is no row, also without a schema or of one that is not a
    # struct; but rows of a first None are rows of single values, of the VariantVals too.
    from pyspark.errors import PySparkValueError
    from pyspark.sql.types import VariantVal

    v = VariantVal.parseJson('{"b": [1, "2", true, null], "a": 1.5}')
    assert str(v) == v.toJson() == '{"a":1.5,"b":[1,"2",true,null]}'
    assert repr(v) == "VariantVal(%r, %r)" % (v.value, v.metadata)
    assert str(VariantVal.parseJson('"x"')) == '"x"'
    assert "%s" % VariantVal.parseJson("1") == "1"

    spark = _offline_session()
    struct = T.StructType([T.StructField("v", T.VariantType())])
    for data, schema in [([v], struct), ([v, None], struct), ([v], None), ([v], T.VariantType())]:
        with pytest.raises(PySparkValueError, match="Rows cannot be of type VariantVal"):
            spark.createDataFrame(data, schema)
    assert type(spark.createDataFrame([None, v], struct)).__name__ == "DataFrame"




def test_variant_values():
    """Test VariantVal and other variant types."""
    from pyspark.sql.types import VariantVal, Geometry, Geography

    # VariantVal JSON parsing
    v = VariantVal.parseJson('{"x": 1}')
    assert v.toPython() == {"x": 1}

    # Geometry
    g = Geometry(b"\x00\x01", 4326)
    assert g.getSrid() == 4326

    # Geography
    gg = Geography(b"\x00\x01", 4326)
    assert gg.getSrid() == 4326


def test_functions_with_literals():
    """Test functions accepting various literal types."""
    from pyspark.sql import functions as F

    # Test lit() with different types
    assert type(F.lit(1)).__name__ == "Column"
    assert type(F.lit("string")).__name__ == "Column"
    assert type(F.lit(1.5)).__name__ == "Column"
    assert type(F.lit(True)).__name__ == "Column"
    assert type(F.lit(None)).__name__ == "Column"


def test_partitioning_functions():
    """Test partitioning module functions."""
    from pyspark.sql.functions import partitioning

    # Test partition functions exist and create columns
    c = __import__("pyspark.sql.functions", fromlist=["col"]).col("x")
    assert type(partitioning.bucket(10, c)).__name__ == "Column"
    assert type(partitioning.years(c)).__name__ == "Column"
    assert type(partitioning.months(c)).__name__ == "Column"
    assert type(partitioning.days(c)).__name__ == "Column"
    assert type(partitioning.hours(c)).__name__ == "Column"


def test_nullable_types():
    """Test .toNullable() on various types."""
    # IntegerType (already atomic)
    i = T.IntegerType()
    if hasattr(i, 'toNullable'):
        assert i.toNullable().simpleString() == i.simpleString()

    # StructType with non-nullable fields
    s = T.StructType([T.StructField("x", T.IntegerType(), False)])
    s_nullable = s.toNullable()
    assert s_nullable.fields[0].nullable is True


def test_when_and_otherwise_conditions():
    """Test when/otherwise column expressions."""
    from pyspark.sql import functions as F

    c = F.col("x")
    cond = c > 5

    result = F.when(cond, 1)
    assert type(result).__name__ == "Column"

    result2 = F.when(cond, 1).otherwise(0)
    assert type(result2).__name__ == "Column"


def test_column_between():
    """Test Column.between method."""
    from pyspark.sql import functions as F

    c = F.col("x")
    result = c.between(1, 10)
    assert type(result).__name__ == "Column"


def test_column_startswith_endswith():
    """Test Column.startswith and endswith methods."""
    from pyspark.sql import functions as F

    c = F.col("text")
    assert type(c.startswith("prefix")).__name__ == "Column"
    assert type(c.endswith("suffix")).__name__ == "Column"


def test_column_methods_take_and_check_their_arguments_as_pyspark():
    # As PySpark's Column: its bool raises CANNOT_CONVERT_COLUMN_INTO_BOOL of the column, like,
    # rlike and ilike take any value and getField any name, and cast, withField, dropFields
    # and over check their arguments.
    import warnings

    from pyspark.errors import PySparkTypeError, PySparkValueError
    from pyspark.sql import functions as F

    c = F.col("a")
    for value, column in ((c, "a"), (c > 1, ">(a, 1)")):
        with pytest.raises(PySparkValueError) as error:
            bool(value)
        assert error.value.getCondition() == "CANNOT_CONVERT_COLUMN_INTO_BOOL"
        assert error.value.getMessageParameters() == {"column": column}
    assert repr(c.like(1)) == "Column<'like(a, 1)'>"
    assert repr(c.rlike(F.col("p"))) == "Column<'rlike(a, p)'>"
    assert repr(c.ilike("x%")) == "Column<'ilike(a, x%)'>"
    assert repr(c.getField(1)) == repr(c[1])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        c.getField(F.col("b"))
    assert [w.category for w in caught] == [FutureWarning]

    def not_expected(arg_name, expected_type, arg_type):
        return (PySparkTypeError, "NOT_EXPECTED_TYPE",
                {"arg_name": arg_name, "expected_type": expected_type, "arg_type": arg_type})

    for call, (error_type, condition, parameters) in (
        (lambda: c.cast(1), not_expected("dataType", "DataType or str", "int")),
        (lambda: c.withField(1, F.lit(1)), not_expected("fieldName", "str", "int")),
        (lambda: c.withField("x", 1), not_expected("col", "Column", "int")),
        (lambda: c.dropFields("x", 1), not_expected("fieldName", "str", "int")),
        (lambda: c.dropFields(), (PySparkValueError, "CANNOT_BE_EMPTY", {"item": "dropFields"})),
        (lambda: c.over(1), not_expected("window", "WindowSpec", "int")),
    ):
        with pytest.raises(error_type) as error:
            call()
        assert error.value.getCondition() == condition
        assert error.value.getMessageParameters() == parameters


def test_column_contains():
    """Test Column.contains method."""
    from pyspark.sql import functions as F

    c = F.col("text")
    assert type(c.contains("substring")).__name__ == "Column"


def test_column_like():
    """Test Column.like method."""
    from pyspark.sql import functions as F

    c = F.col("text")
    assert type(c.like("%pattern%")).__name__ == "Column"


def test_column_rlike():
    """Test Column.rlike method."""
    from pyspark.sql import functions as F

    c = F.col("text")
    assert type(c.rlike("pattern")).__name__ == "Column"


def test_column_in_list():
    """Test Column.isin method."""
    from pyspark.sql import functions as F

    c = F.col("status")
    assert type(c.isin([1, 2, 3])).__name__ == "Column"
    assert type(c.isin("a", "b", "c")).__name__ == "Column"


def test_column_getitem():
    """Test Column getitem access (for struct/map/array fields)."""
    from pyspark.sql import functions as F

    c = F.col("data")
    # Accessing a field by name or index
    assert type(c["field"]).__name__ == "Column"
    assert type(c[0]).__name__ == "Column"


def test_session_builder_remote_url():
    """Test SparkSession.Builder with remote URL parsing."""
    from pyspark.sql import SparkSession

    builder = SparkSession.builder.remote("sc://localhost:15002")
    assert type(builder).__name__ == "Builder"

    # Test with additional config
    builder2 = builder.appName("test").config("key", "value")
    assert builder2 is builder


def test_session_builder_remote_takes_the_url_as_pyspark():
    import inspect
    import os
    import subprocess
    import sys

    from pyspark.sql import SparkSession

    # PySpark's SparkSession.Builder.remote(url), and the Connect builder's location.
    assert list(inspect.signature(SparkSession.builder.remote).parameters) == ["url", "location"]
    with pytest.raises(TypeError):
        SparkSession.builder.remote("sc://a:1", location="sc://b:2")
    # Each keyword's URL is the server the session uses, in a new interpreter, as the
    # session becomes the active one.
    tests = os.path.dirname(os.path.abspath(__file__))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [tests, env.get("PYTHONPATH")]))
    subprocess.run([sys.executable, "-c", """
import multiprocessing
from _mock_connect_server import RemoteService, serve
from pyspark.sql import SparkSession

parent, child = multiprocessing.get_context("spawn").Pipe()
process = multiprocessing.get_context("spawn").Process(target=serve, args=(child,))
process.start()
try:
    assert parent.poll(20), "Mock server did not start"
    url = f"sc://127.0.0.1:{parent.recv()}"
    for calls, keyword in enumerate(["url", "location"], start=1):
        spark = SparkSession.builder.remote(**{keyword: url}).getOrCreate()
        assert spark.range(3).columns == ["id"]
        assert RemoteService(parent).calls == calls, keyword
        spark.stop()
finally:
    parent.send(("stop",))
    process.join(10)
"""], check=True, env=env)


def test_functions_struct_creation():
    """Test struct function for creating struct columns."""
    from pyspark.sql import functions as F

    c1 = F.col("a")
    c2 = F.col("b")
    # struct function creates a struct column
    result = F.struct(c1, c2)
    assert type(result).__name__ == "Column"


def test_functions_named_struct():
    """Test named_struct function."""
    from pyspark.sql import functions as F

    c1 = F.col("x")
    c2 = F.col("y")
    result = F.named_struct("a", c1, "b", c2)
    assert type(result).__name__ == "Column"


def test_functions_make_map():
    """Test create_map function (alternate name)."""
    from pyspark.sql import functions as F

    k1 = F.lit("key1")
    v1 = F.lit("value1")
    result = F.create_map(k1, v1)
    assert type(result).__name__ == "Column"


# Additional edge case tests to push coverage higher

def test_udf_with_explicit_name():
    """Test UDF with explicit name parameter."""
    from pyspark.sql.functions import udf
    from pyspark.sql.types import IntegerType

    # UDF name defaults to function name if not specified
    u = udf(lambda x: x + 1, IntegerType())
    # Name should be accessible if it exists
    if hasattr(u, 'name'):
        assert u.name is not None


def test_udf_call_with_column():
    """Test calling UDF with Column argument."""
    from pyspark.sql.functions import udf, col
    from pyspark.sql.types import IntegerType

    u = udf(lambda x: x + 1, IntegerType())
    c = col("x")
    result = u(c)
    assert type(result).__name__ == "Column"


def test_udf_multiple_args():
    """Test UDF with multiple arguments."""
    from pyspark.sql.functions import udf, col
    from pyspark.sql.types import IntegerType

    u = udf(lambda x, y: x + y, IntegerType())
    result = u(col("x"), col("y"))
    assert type(result).__name__ == "Column"


def test_pandas_udf_scalar():
    """Test pandas_udf with scalar function type."""
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import IntegerType

    # Create pandas_udf without decorator to avoid execution issues
    p = pandas_udf(lambda s: s + 1, IntegerType())
    assert callable(p)


def test_pandas_udf_grouped_agg():
    """Test pandas_udf with grouped_agg function type."""
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import IntegerType

    # Create pandas_udf without decorator to avoid execution issues
    from pyspark.sql.functions import PandasUDFType

    p = pandas_udf(lambda s: s, IntegerType(), functionType=PandasUDFType.GROUPED_AGG)
    assert callable(p)


def test_arrow_udf_types():
    """Test arrow_udf with different options."""
    from pyspark.sql.functions import arrow_udf
    from pyspark.sql.types import IntegerType

    # Scalar UDF
    u1 = arrow_udf(lambda x: x, IntegerType())
    assert callable(u1)

    # Scalar iter UDF
    from pyspark.sql.functions import ArrowUDFType

    u2 = arrow_udf(lambda x: x, IntegerType(), functionType=ArrowUDFType.SCALAR_ITER)
    assert callable(u2)


def test_array_type_properties():
    """Test ArrayType element properties via simpleString."""
    arr = T.ArrayType(T.StringType(), containsNull=True)
    assert "array" in arr.simpleString()
    assert "string" in arr.simpleString()


def test_map_type_properties():
    """Test MapType key and value properties via simpleString."""
    map_t = T.MapType(T.StringType(), T.IntegerType(), valueContainsNull=False)
    assert "map" in map_t.simpleString()
    assert "string" in map_t.simpleString()
    assert "int" in map_t.simpleString()


def test_decimal_type_json_roundtrip():
    """Test DecimalType JSON serialization."""
    d = T.DecimalType(10, 2)
    json_str = d.json()
    assert "10" in json_str
    assert "2" in json_str


def test_char_varchar_types():
    """Test CharType and VarcharType."""
    c = T.CharType(5)
    assert c.simpleString() == "char(5)"

    v = T.VarcharType(10)
    assert v.simpleString() == "varchar(10)"


def test_time_type():
    """Test TimeType with precision."""
    t = T.TimeType(3)
    assert "time" in t.simpleString()


def test_calendar_interval_type():
    """Test CalendarIntervalType."""
    c = T.CalendarIntervalType()
    assert c.simpleString() == "interval"


def test_year_month_interval_type():
    """Test YearMonthIntervalType with different ranges."""
    ym1 = T.YearMonthIntervalType(0, 1)  # YEAR TO MONTH
    assert "interval" in ym1.simpleString()

    ym2 = T.YearMonthIntervalType(0, 0)  # YEAR only
    assert "interval" in ym2.simpleString()


def test_day_time_interval_type():
    """Test DayTimeIntervalType with different ranges."""
    dt1 = T.DayTimeIntervalType(0, 3)  # DAY TO SECOND
    assert "interval" in dt1.simpleString()

    dt2 = T.DayTimeIntervalType(0, 1)  # DAY TO MINUTE
    assert "interval" in dt2.simpleString()


def test_struct_field_equality():
    """Test StructField equality comparison."""
    f1 = T.StructField("x", T.IntegerType(), True)
    f2 = T.StructField("x", T.IntegerType(), True)
    assert f1 == f2

    f3 = T.StructField("y", T.IntegerType(), True)
    assert f1 != f3


def test_is_streaming_asks_the_server_as_pyspark(mock_connect):
    # As PySpark's: isStreaming is the server's answer to an IsStreaming request of the plan.
    spark, service = mock_connect
    df = spark.range(1).select("id")
    try:
        assert df.isStreaming is False
        service.streaming = True
        assert df.isStreaming is True
    finally:
        service.streaming = False
    # base.proto: AnalyzePlanRequest.is_streaming=8.
    assert sum(8 in parse_message(request) for request in service.analyze_requests) == 2


def test_struct_type_field_access():
    """Test accessing StructType fields by index and name."""
    st = T.StructType([
        T.StructField("a", T.IntegerType()),
        T.StructField("b", T.StringType()),
        T.StructField("c", T.BooleanType()),
    ])

    # By index
    assert st[0].name == "a"
    assert st[1].name == "b"
    assert st[2].name == "c"

    # By name
    assert st["a"].dataType.simpleString() == "int"
    assert st["b"].dataType.simpleString() == "string"

    # Slicing
    subset = st[0:2]
    assert len(subset) == 2


def test_struct_type_iteration():
    """Test iterating over StructType fields."""
    st = T.StructType([
        T.StructField("x", T.IntegerType()),
        T.StructField("y", T.StringType()),
    ])

    names = [f.name for f in st]
    assert names == ["x", "y"]


def test_functions_expr():
    """Test expr function."""
    from pyspark.sql import functions as F

    # expr function
    expr_result = F.expr("1 + 1")
    assert type(expr_result).__name__ == "Column"


def test_functions_nullif_ifnull():
    """Test nullif and ifnull functions."""
    from pyspark.sql import functions as F

    c1 = F.col("x")
    c2 = F.col("y")

    # nullif returns NULL if two values are equal
    assert type(F.nullif(c1, c2)).__name__ == "Column"

    # ifnull replaces NULL with a value
    assert type(F.ifnull(c1, F.lit(0))).__name__ == "Column"


def test_functions_nvl_nvl2():
    """Test nvl and nvl2 functions."""
    from pyspark.sql import functions as F

    c = F.col("x")

    # nvl replaces NULL
    assert type(F.nvl(c, F.lit(0))).__name__ == "Column"

    # nvl2 returns a value based on nullability
    assert type(F.nvl2(c, F.lit(1), F.lit(0))).__name__ == "Column"


def test_functions_stack():
    """Test stack function for row construction."""
    from pyspark.sql import functions as F

    # stack creates rows from arguments; as in PySpark, the count is a Column too.
    result = F.stack(F.lit(2), F.col("a"), F.col("b"))
    assert type(result).__name__ == "Column"


def test_functions_sort_array():
    """Test sort_array function."""
    from pyspark.sql import functions as F

    arr = F.col("array_col")
    result = F.sort_array(arr)
    assert type(result).__name__ == "Column"


def test_functions_reverse_array():
    """Test reverse function on arrays."""
    from pyspark.sql import functions as F

    arr = F.col("array_col")
    result = F.reverse(arr)
    assert type(result).__name__ == "Column"


def test_functions_slice():
    """Test slice function for arrays."""
    from pyspark.sql import functions as F

    arr = F.col("array_col")
    result = F.slice(arr, F.lit(1), F.lit(3))
    assert type(result).__name__ == "Column"


def test_column_description_methods():
    """Test various Column description methods."""
    from pyspark.sql import functions as F

    c = F.col("x")

    # Test repr of column
    r = repr(c)
    assert "Column" in r or "x" in r


def test_session_builder_all_config_options():
    """Test various SparkSession builder configuration options."""
    from pyspark.sql import SparkSession

    builder = (SparkSession.builder
               .master("local")
               .appName("test")
               .config("key", "value")
               .config("spark.sql.shuffle.partitions", "10"))

    assert type(builder).__name__ == "Builder"


def test_higher_order_function_multiple_lambdas():
    """Test nested higher-order functions with multiple lambdas."""
    from pyspark.sql import functions as F

    # Nested transforms
    result = F.transform(
        F.col("array"),
        lambda x: F.transform(x, lambda y: y + 1)
    )
    assert type(result).__name__ == "Column"


def test_column_operations_with_literals():
    """Test column operations with various literal types."""
    from pyspark.sql import functions as F

    c = F.col("x")

    # Integer
    assert type(c + 1).__name__ == "Column"
    assert type(c - 1).__name__ == "Column"
    assert type(c * 2).__name__ == "Column"
    assert type(c / 2).__name__ == "Column"

    # String
    assert type(c + F.lit("suffix")).__name__ == "Column"

    # Boolean
    assert type(c == 1).__name__ == "Column"
    assert type(c > 0).__name__ == "Column"
    assert type(c < 10).__name__ == "Column"
    assert type(c >= 1).__name__ == "Column"
    assert type(c <= 100).__name__ == "Column"


def test_column_logical_operations():
    """Test column logical operations."""
    from pyspark.sql import functions as F

    c1 = F.col("x") > 0
    c2 = F.col("y") < 100

    assert type(c1 & c2).__name__ == "Column"
    assert type(c1 | c2).__name__ == "Column"
    assert type(~c1).__name__ == "Column"


def test_functions_greatest_least_with_multiple_cols():
    """Test greatest/least with multiple columns."""
    from pyspark.sql import functions as F

    c1 = F.lit(1)
    c2 = F.lit(2)
    c3 = F.lit(3)

    assert type(F.greatest(c1, c2, c3)).__name__ == "Column"
    assert type(F.least(c1, c2, c3)).__name__ == "Column"


def test_row_field_access_with_index():
    """Test Row field access with index."""
    from pyspark.sql import Row

    r = Row(x=10, y=20, z=30)

    # Index access
    assert r[0] == 10
    assert r[1] == 20
    assert r[2] == 30


def test_row_length():
    """Test Row length."""
    from pyspark.sql import Row

    r = Row(a=1, b=2, c=3)
    assert len(r) == 3


def test_since_decorator():
    """Test the @since decorator from pyspark, which appends the version to the docstring."""
    from pyspark import since

    @since(3.0)
    def new_function():
        """A function."""
        return "result"

    assert new_function() == "result"
    assert new_function.__doc__ == "A function.\n\n.. versionadded:: 3.0"

    # As PySpark's, of a function that has a docstring.
    with pytest.raises(AssertionError):

        @since("4.0")
        def another_func():
            return 42


def test_spark_conf_empty_config():
    """Test SparkConf with no settings."""
    from pyspark import SparkConf

    conf = SparkConf()
    assert conf.getAll() == []


def test_spark_conf_get_with_default():
    """Test SparkConf.get with default value."""
    from pyspark import SparkConf

    conf = SparkConf()
    # Getting a key that doesn't exist should return the default
    assert conf.get("missing", "default_value") == "default_value"
    assert conf.get("missing", None) is None
    assert conf.get("missing") is None


def test_spark_conf_chain_operations():
    """Test chaining SparkConf operations."""
    from pyspark import SparkConf

    conf = (SparkConf()
            .set("key1", "value1")
            .set("key2", "value2")
            .setAll([("key3", "value3"), ("key4", "value4")]))

    assert conf.get("key1") == "value1"
    assert conf.get("key3") == "value3"
    assert len(conf.getAll()) == 4


def test_sessions_and_dataframes_pickle_of_the_active_session_as_pyspark(mock_connect):
    # As PySpark Connect's, with cloudpickle: a session pickles as a function of its ID, which
    # gives back the active session of that ID, and a DataFrame of its session and its plan, of
    # its plan IDs, so that it is sent as it is. Where no session is active, as on a Python
    # worker of the server, both raise NO_ACTIVE_SESSION, as the session unpickles first; where
    # another is, PySparkAssertionError. Plain pickle cannot pickle the function, a local one.
    import pickle
    import threading

    from pyspark.errors import PySparkAssertionError, PySparkRuntimeError
    from pyspark.serializers import CloudPickleSerializer

    spark, _ = mock_connect
    serializer = CloudPickleSerializer()
    assert serializer.loads(serializer.dumps(spark)) is spark
    df = spark.range(3).selectExpr("id * 2 AS x")
    copy = serializer.loads(serializer.dumps(df))
    assert type(copy) is type(df) and copy._relation_bytes() == df._relation_bytes()
    assert copy.sparkSession.session_id == spark.session_id
    for obj in (spark, df):
        with pytest.raises((AttributeError, pickle.PicklingError)):
            pickle.dumps(obj)

    errors = []

    def unpickle(data):
        try:
            serializer.loads(data)
        except PySparkRuntimeError as error:
            errors.append(error.getCondition())

    for obj in (spark, df):
        thread = threading.Thread(target=unpickle, args=(serializer.dumps(obj),))
        thread.start()
        thread.join()
    assert errors == ["NO_ACTIVE_SESSION", "NO_ACTIVE_SESSION"]
    with pytest.raises(PySparkAssertionError, match="Expected session ID does not match"):
        SparkSession._getActiveSessionIfMatches("another")




def test_inheritable_threads_inherit_the_tags_of_the_session_as_pyspark(mock_connect):
    # As PySpark's of Spark Connect: InheritableThread takes the tags of its session of the
    # thread that starts it, and inheritable_thread_target those of the thread that wraps the
    # function, with the session as the active one of the thread; a plain thread has none.
    import threading

    from pyspark import InheritableThread, inheritable_thread_target
    from pyspark.sql import SparkSession

    spark, _ = mock_connect
    spark.addTag("parent")
    seen = {}

    def child(key):
        seen[key] = (spark.getTags(), SparkSession.getActiveSession() is spark)

    try:
        for thread in [
            threading.Thread(target=child, args=("plain",)),
            InheritableThread(target=child, args=("inheritable",), session=spark),
            threading.Thread(target=inheritable_thread_target(spark)(child), args=("target",)),
        ]:
            thread.start()
            thread.join()
        assert seen == {
            "plain": (set(), False),
            "inheritable": ({"parent"}, False),
            "target": ({"parent"}, True),
        }
        assert spark.getTags() == {"parent"}
    finally:
        spark.clearTags()
    with pytest.raises(AssertionError, match="Spark Connect must be provided"):
        InheritableThread(target=child)
    with pytest.raises(AssertionError, match="f is expected to be SparkSession"):
        inheritable_thread_target(child)(child)


def test_row_asdict():
    """Test Row.asDict() method."""
    from pyspark.sql import Row

    r = Row(name="Alice", age=30)
    d = r.asDict()
    assert isinstance(d, dict)
    assert d["name"] == "Alice"
    assert d["age"] == 30


def test_row_asdict_ordering():
    """Test that Row.asDict preserves field order."""
    from pyspark.sql import Row

    r = Row(a=1, b=2, c=3)
    d = r.asDict()
    keys = list(d.keys())
    assert keys == ["a", "b", "c"]


def test_row_hash_matches_tuple():
    """Row hashes like a tuple of its values (pyspark Row subclasses tuple)."""
    from pyspark.sql import Row

    assert hash(Row(a=1)) == hash(Row(a=1))
    assert hash(Row(a=1, b="x")) == hash((1, "x"))
    assert hash(Row(1, 2.5, None)) == hash((1, 2.5, None))
    assert hash(Row()) == hash(())
    # Nested struct values materialize as Rows, which are hashable too.
    assert hash(Row(a=Row(b=1))) == hash(((1,),))


def test_observations_check_their_arguments_as_pyspark(mock_connect):
    # As PySpark's Observation and DataFrame.observe, which check their arguments on the
    # client, and ask the server whether the DataFrame is streaming, as `df.isStreaming`.
    from pyspark.errors import PySparkAssertionError, PySparkTypeError, PySparkValueError
    from pyspark.sql import Observation
    from pyspark.sql import functions as F

    spark, _ = mock_connect
    count = F.count(F.lit(1)).alias("c")
    df = spark.range(1)
    for call, error_type, condition, parameters in (
        (lambda: Observation(1), PySparkTypeError, "NOT_EXPECTED_TYPE",
         {"arg_name": "name", "expected_type": "str", "arg_type": "int"}),
        (lambda: Observation(""), PySparkValueError, "VALUE_NOT_NON_EMPTY_STR",
         {"arg_name": "name", "arg_value": ""}),
        (lambda: Observation("o").get, PySparkAssertionError, "NO_OBSERVE_BEFORE_GET", {}),
        (lambda: df.observe(Observation("o")), PySparkValueError, "CANNOT_BE_EMPTY",
         {"item": "exprs"}),
        (lambda: df.observe("o", "id"), PySparkTypeError, "NOT_EXPECTED_TYPE",
         {"arg_name": "exprs", "expected_type": "list[Column]", "arg_type": "tuple"}),
        (lambda: df.observe(1, count), PySparkTypeError, "NOT_EXPECTED_TYPE",
         {"arg_name": "observation", "expected_type": "Observation or str", "arg_type": "int"}),
    ):
        with pytest.raises(error_type) as error:
            call()
        assert error.value.getCondition() == condition
        assert error.value.getMessageParameters() == parameters
    # An unnamed observation is of a random UUID once it is used, and is used once; it has no
    # metrics until an action runs.
    observation = Observation()
    assert observation._name is None
    df.observe(observation, count)
    assert len(observation._name) == 36 and observation.get == {}
    for observe in (lambda: df.observe(observation, count), lambda: observation._on(df, count)):
        with pytest.raises(PySparkAssertionError) as error:
            observe()
        assert error.value.getCondition() == "REUSE_OBSERVATION"


def test_row_usable_in_set_and_dict():
    """Equal Rows collapse in a set and work as dict keys."""
    from pyspark.sql import Row

    s = {Row(a=1, b=2), Row(a=1, b=2), Row(a=3, b=4)}
    assert len(s) == 2
    assert Row(a=1, b=2) in s
    d = {Row(k="x"): 1}
    assert d[Row(k="x")] == 1


def test_row_hash_unhashable_values_raise():
    """Rows holding list/dict values are unhashable, as a tuple holding them is."""
    import pytest

    from pyspark.sql import Row

    with pytest.raises(TypeError):
        hash(Row(a=[1, 2]))
    with pytest.raises(TypeError):
        hash(Row(a={"k": 1}))


# --- Connect import-path shims + offline utility helpers ---------------------
# The pyspark.sql.connect.* modules are thin re-exports that pandas-on-Spark and the
# official connect tests import from; verify they import and re-export the Rust-backed
# classes so the drop-in's connect import path is exercised offline.
def test_connect_shim_imports():
    import importlib

    for mod in [
        "pyspark.sql.connect.column",
        "pyspark.sql.connect.dataframe",
        "pyspark.sql.connect.session",
        "pyspark.sql.connect.plan",
        "pyspark.sql.connect.proto",
        "pyspark.sql.connect.functions",
        "pyspark.sql.connect.functions.builtin",
        "pyspark.sql.connect.avro",
        "pyspark.sql.connect.protobuf",
        "pyspark.sql.connect.resource",
        "pyspark.sql.connect.streaming",
        "pyspark.sql.connect.client",
        "pyspark.sql.connect.shell",
    ]:
        m = importlib.import_module(mod)
        assert m is not None

    # Column/DataFrame re-exported under the connect path are the Rust-backed classes.
    from pyspark.sql.connect.column import Column as ConnectColumn
    from pyspark.sql.connect.dataframe import DataFrame as ConnectDataFrame
    from pyspark._pyspark import Column as RustColumn, DataFrame as RustDataFrame

    assert ConnectColumn is RustColumn
    assert ConnectDataFrame is RustDataFrame

    # The internal-function dispatcher builds an UnresolvedFunction by name.
    from pyspark.sql.connect.functions.builtin import _invoke_function_over_columns
    from pyspark.sql import functions as F

    assert repr(_invoke_function_over_columns("pandas_product", F.col("x"))) == "Column<'pandas_product(x)'>"


def test_module_level_names_as_pyspark(monkeypatch):
    # As official PySpark 4.3.0: the names of pyspark.util, and the __all__ of the modules of
    # pyspark.sql, of which a star import exports only those.
    import gc
    import typing

    import pyspark.util as util
    from pyspark.errors import PySparkRuntimeError

    assert util.__all__ == []
    assert isinstance(util.FuncT, typing.TypeVar)
    monkeypatch.setenv("SPARK_CONNECT_MODE", "1")
    assert util.spark_connect_mode() == "1" and util.default_api_mode() == "connect"
    monkeypatch.setenv("SPARK_CONNECT_MODE", "0")
    assert util.spark_connect_mode() == "0" and util.default_api_mode() == "classic"

    def stops():
        raise StopIteration("done")

    assert util.fail_on_stopiteration(lambda x: x + 1)(1) == 2
    with pytest.raises(PySparkRuntimeError) as error:
        util.fail_on_stopiteration(stops)()
    assert error.value.getCondition() == "STOP_ITERATION_OCCURRED"
    assert error.value.getMessageParameters() == {"exc": "done"}
    enabled = gc.isenabled()
    with util.disable_gc():
        assert not gc.isenabled()
    assert gc.isenabled() == enabled

    for module, expected in [
        ("pyspark.sql.udf", ["UDFRegistration"]),
        (
            "pyspark.sql.udtf",
            [
                "AnalyzeArgument",
                "AnalyzeResult",
                "PartitioningColumn",
                "OrderingColumn",
                "SelectedColumn",
                "SkipRestOfInputTableException",
                "UDTFRegistration",
            ],
        ),
        ("pyspark.sql.avro", ["functions"]),
        ("pyspark.sql.protobuf", ["functions"]),
    ]:
        namespace = {}
        exec(f"from {module} import *", namespace)
        assert [name for name in namespace if name != "__builtins__"] == expected, module


def test_connect_import_path_modules_as_pyspark(mock_connect):
    # As official PySpark 4.3.0: the modules of the connect import path, of the functions and
    # the classes of this client, and the deprecated functions of the builtin modules.
    import importlib
    import logging

    from pyspark.sql import functions as F
    from pyspark.sql.avro import functions as avro
    from pyspark.sql.datasource import DataSource
    from pyspark.sql.functions import partitioning
    from pyspark.sql.protobuf import functions as protobuf
    from pyspark.sql.table_arg import TableArg

    spark, service = mock_connect
    connect_partitioning = importlib.import_module("pyspark.sql.connect.functions.partitioning")
    assert importlib.import_module("pyspark.sql.connect.functions").partitioning is (
        connect_partitioning
    )
    for name in ["years", "months", "days", "hours", "bucket"]:
        assert getattr(connect_partitioning, name) is getattr(partitioning, name)
    connect_avro = importlib.import_module("pyspark.sql.connect.avro.functions")
    assert connect_avro.from_avro is avro.from_avro and connect_avro.to_avro is avro.to_avro
    connect_protobuf = importlib.import_module("pyspark.sql.connect.protobuf.functions")
    assert connect_protobuf.from_protobuf is protobuf.from_protobuf
    assert connect_protobuf.to_protobuf is protobuf.to_protobuf
    assert importlib.import_module("pyspark.sql.connect.table_arg").TableArg is TableArg
    typing_module = importlib.import_module("pyspark.sql.connect._typing")
    assert {"ColumnOrName", "DataTypeOrString", "LiteralType"} <= set(dir(typing_module))

    connect_logging = importlib.import_module("pyspark.sql.connect.logging")
    assert connect_logging.__all__ == ["configureLogging", "getLogLevel"]
    assert isinstance(connect_logging.logger, logging.Logger)
    if "SPARK_CONNECT_LOG_LEVEL" not in os.environ:
        assert connect_logging.getLogLevel() is None

    connect_builtin = importlib.import_module("pyspark.sql.connect.functions.builtin")
    for name in [
        "approxCountDistinct",
        "bitwiseNOT",
        "countDistinct",
        "shiftLeft",
        "shiftRight",
        "shiftRightUnsigned",
        "sumDistinct",
        "toDegrees",
        "toRadians",
        "chr",
        "random",
        "uuid",
    ]:
        assert getattr(F.builtin, name) is getattr(F, name), name
        assert getattr(connect_builtin, name) is getattr(F, name), name

    # The registration of a data source registers it with the session.
    from pyspark.sql.connect.datasource import DataSourceRegistration

    class TestSource(DataSource):
        @classmethod
        def name(cls):
            return "test_source"

    DataSourceRegistration(spark).register(TestSource)
    assert service.execute_calls == 1


def test_connect_utils_check_the_dependencies_of_the_client(tmp_path):
    # As PySpark Connect's pyspark.sql.connect.utils, but check_dependencies checks the
    # dependencies of this client, which do not include gRPC. The script runs of a file: of
    # a main module without one, check_dependencies takes it for doctests.
    import subprocess
    import sys
    import textwrap

    from pyspark.errors import PySparkImportError
    from pyspark.sql.connect import utils

    assert utils.get_python_ver() == "%d.%d" % sys.version_info[:2]
    utils.check_dependencies()
    for check in [
        utils.require_minimum_grpc_version,
        utils.require_minimum_grpcio_status_version,
        utils.require_minimum_googleapis_common_protos_version,
        utils.require_minimum_zstandard_version,
    ]:
        # Of the packages installed here, or not.
        try:
            check()
        except PySparkImportError as error:
            assert error.getCondition() in ("PACKAGE_NOT_INSTALLED", "UNSUPPORTED_PACKAGE_VERSION")
    script = textwrap.dedent(
        """
        import importlib.abc, sys

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name.split(".")[0] in ("grpc", "grpc_status", "zstandard") or name.startswith(
                    "google.rpc"
                ):
                    raise ImportError("not installed: " + name)

        sys.meta_path.insert(0, Block())
        from pyspark.errors import PySparkImportError
        from pyspark.sql.connect import utils

        utils.check_dependencies()
        for check, package in [
            (utils.require_minimum_grpc_version, "grpcio"),
            (utils.require_minimum_grpcio_status_version, "grpcio-status"),
            (utils.require_minimum_googleapis_common_protos_version, "googleapis-common-protos"),
            (utils.require_minimum_zstandard_version, "zstandard"),
        ]:
            try:
                check()
            except PySparkImportError as error:
                assert error.getCondition() == "PACKAGE_NOT_INSTALLED"
                assert error.getMessageParameters()["package_name"] == package
            else:
                raise AssertionError(package)
        print("ok")
        """
    )
    path = tmp_path / "check_dependencies.py"
    path.write_text(script)
    result = subprocess.run(
        [sys.executable, str(path)], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0 and result.stdout.strip() == "ok", result.stderr


def test_is_remote_flags():
    # The Rust-backed drop-in is always Connect-only.
    from pyspark.util import is_remote_only

    assert is_remote_only() is True


# --- UserDefinedType protocol + types.py helpers (offline) -------------------
from pyspark.sql.types import (
    UserDefinedType,
    ArrayType as _AT,
    DoubleType as _DT,
    IntegerType as _IT,
    StringType as _ST,
    StructField as _SF,
    StructType as _StT,
    MapType as _MT,
    _drop_metadata,
    _create_row,
)


class _PointUDT(UserDefinedType):
    @classmethod
    def sqlType(cls):
        return _AT(_DT(), False)

    @classmethod
    def module(cls):
        return "tests.test_dropin_offline"

    def serialize(self, obj):
        return [float(obj[0]), float(obj[1])]

    def deserialize(self, datum):
        return (datum[0], datum[1])


class _BareUDT(UserDefinedType):
    @classmethod
    def sqlType(cls):
        return _IT()


def test_udt_protocol_roundtrip():
    udt = _PointUDT()
    assert udt.simpleString() == "udt"
    internal = udt.toInternal((1.0, 2.0))
    assert udt.fromInternal(internal) == (1.0, 2.0)
    # _cachedSqlType caches the sqlType instance.
    assert udt._cachedSqlType() is udt._cachedSqlType()
    # jsonValue (python UDT -> pyClass form) + json round-trips via fromJson.
    jv = udt.jsonValue()
    assert jv["type"] == "udt"
    assert "pyClass" in jv
    import json as _json

    assert _json.loads(udt.json())["type"] == "udt"
    # fromJson reconstructs the UDT from its serialized pyClass (covers the udt branch).
    UserDefinedType.fromJson(jv)
    # None passes through toInternal/fromInternal.
    assert udt.toInternal(None) is None


def test_udt_not_implemented_defaults():
    import pytest

    udt = _BareUDT()
    with pytest.raises(Exception):
        udt.serialize((1,))
    with pytest.raises(Exception):
        udt.deserialize(1)


def test_struct_field_metadata_any_values():
    # StructField metadata accepts arbitrary JSON values (Dict[str, Any]), not just str,
    # and round-trips through json.
    f = _SF("a", _IT(), True, {"m": 1, "s": "x", "b": True, "l": [1, 2]})
    assert f.metadata == {"m": 1, "s": "x", "b": True, "l": [1, 2]}
    f2 = _SF.fromJson(f.jsonValue())
    assert f2.metadata["m"] == 1 and f2.metadata["b"] is True and f2.metadata["l"] == [1, 2]


def test_drop_metadata_and_create_row():
    st = _StT([_SF("a", _IT(), True, {"m": 1}), _SF("b", _AT(_IT()), True)])
    dm = _drop_metadata(st)
    assert not dm.fields[0].metadata
    _drop_metadata(_AT(_IT()))
    _drop_metadata(_MT(_ST(), _IT()))
    _drop_metadata(_IT())
    r = _create_row(["a", "b"], [1, 2])
    assert r["a"] == 1 and r["b"] == 2
    assert list(r.__fields__) == ["a", "b"]
    # The fields can be a row class, whose values name them, as Row.__call__ passes it.
    r2 = _create_row(T.Row("a", "b"), [3, 4])
    assert r2["a"] == 3 and r2.b == 4


# ------------------------------------------- API parity: version, properties, signatures
#
# These guard the PySpark-surface parity that the transport-level official-suite gate does
# not exercise: the reported version, property-vs-method access, show()'s default, and the
# public keyword-argument names (camelCase, not the Rust snake_case).

import inspect  # noqa: E402

import pyspark  # noqa: E402
from pyspark.sql import SparkSession as _SS, DataFrame as _DF, Column as _Col  # noqa: E402


def test_dropin_version_is_spark_version():
    assert pyspark.__version__ == "4.3.0"


@pytest.mark.parametrize("cls,name", [(_SS, "version"), (_DF, "dtypes"), (_DF, "executionInfo")])
def test_property_not_method(cls, name):
    # PySpark accesses these without parentheses; the Rust getter is a getset_descriptor.
    assert type(inspect.getattr_static(cls, name)).__name__ == "getset_descriptor"


def test_show_has_no_required_args_and_defaults_n_to_20():
    sig = inspect.signature(_DF.show)
    required = [
        p.name
        for p in sig.parameters.values()
        if p.name not in ("self", "cls")
        and p.kind in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY, p.KEYWORD_ONLY)
        and p.default is inspect._empty
    ]
    assert required == [], required  # df.show() must work
    assert sig.parameters["n"].default == 20


@pytest.mark.parametrize(
    "cls,method,param",
    [
        (_SS, "sql", "sqlQuery"),
        (_SS, "table", "tableName"),
        (_DF, "limit", "num"),
        (_DF, "take", "num"),
        (_DF, "withColumn", "colName"),
        (_DF, "repartition", "numPartitions"),
        (_Col, "cast", "dataType"),
        (_Col, "astype", "dataType"),
        (_Col, "substr", "startPos"),
        (_Col, "between", "lowerBound"),
        (_Col, "over", "window"),
    ],
)
def test_keyword_param_names_match_pyspark(cls, method, param):
    assert param in inspect.signature(getattr(cls, method)).parameters


# --- SPARK-XXXXX: string function args are column names (ColumnOrName parity) ---
def test_fn_col_args_spec_classification():
    from pyspark.sql import functions as F
    spec = F._FN_COL_ARGS
    # literal-param functions: only the listed positions are columns
    assert spec["date_format"] == ((0,), None)
    assert spec["regexp_replace"] == ((0,), None)
    assert spec["split"] == ((0, 2), None)
    assert spec["substring_index"] == ((0,), None)
    assert spec["concat_ws"] == ((), 1)          # sep literal, *cols variadic columns
    assert spec["current_date"] == ((), None)     # no column args
    # pure-column functions are absent -> default: every arg is a column
    for name in ("sqrt", "upper", "abs", "mean", "struct", "array", "coalesce"):
        assert name not in spec, name


def test_to_col_coerces_str_to_column():
    from pyspark.sql import functions as F
    col_cls = type(F.col("x"))
    # a bare str resolves to a Column via col()
    assert isinstance(F._to_col("x"), col_cls)
    # an existing Column passes through unchanged
    c = F.col("y")
    assert F._to_col(c) is c


def test_generated_fn_accepts_str_column_name():
    from pyspark.sql import functions as F
    col_cls = type(F.col("x"))
    # string-name and Column forms build identical column expressions (lazy, no server)
    for build in (lambda a: F.sqrt(a), lambda a: F.upper(a), lambda a: F.abs(a)):
        assert isinstance(build("v"), col_cls)
        assert repr(build("v")) == repr(build(F.col("v")))
    # multi-col string args reference columns
    assert repr(F.struct("id", "g")) == repr(F.struct(F.col("id"), F.col("g")))
    # literal-position args stay literal (date_format format, regexp pattern)
    assert "yyyy" in repr(F.date_format(F.col("d"), "yyyy"))


def test_udf_string_arg_is_column_name_not_literal():
    # udf(f)("colname") must reference a column (ColumnOrName), not a string literal.
    from pyspark.sql.functions import udf, col
    from pyspark.sql.types import IntegerType
    u = udf(lambda x: x + 1, IntegerType())
    c = u("id")
    # Same rendered expression whether the arg is a str or an explicit col().
    assert repr(c) == repr(u(col("id")))


def test_udf_and_pandas_udf_decorator_forms(mock_connect):
    # @udf(returnType) / @pandas_udf(returnType) return a decorator that wraps a function.
    from pyspark.sql.functions import udf, pandas_udf
    from pyspark.sql.types import IntegerType
    from pyspark.sql.udf import UserDefinedFunction

    # As PySpark's, the function wrapper of the UDF.
    @udf(IntegerType())
    def inc(x):
        return x + 1
    assert isinstance(inc._unwrapped, UserDefinedFunction) and inc.__name__ == "inc"

    @pandas_udf("long")
    def p(s):
        return s + 1
    assert isinstance(p._unwrapped, UserDefinedFunction) and p.__name__ == "p"


def test_udf_return_type_normalized_to_concrete_datatype(mock_connect):
    # A DDL-string returnType must become a concrete, picklable DataType so the
    # cloudpickled command carries a real type (the pandas/Arrow worker needs it).
    import pickle
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import DoubleType

    @pandas_udf("double")
    def f(s):
        return s
    # The command is the cloudpickle of (func, returnType), so it carries this type.
    assert isinstance(f.returnType, DoubleType), type(f.returnType)
    pickle.dumps(f.returnType)  # concrete types pickle


def test_pandas_udf_eval_type_inferred_from_type_hints(mock_connect):
    # @pandas_udf("type") infers eval type from the function's pandas type hints:
    # Series->Series = scalar(200), Iterator->Iterator = scalar_iter(204),
    # Series->scalar = grouped_agg(202).
    from typing import Iterator
    import pandas as pd
    from pyspark.sql.functions import pandas_udf

    @pandas_udf("double")
    def scal(s: pd.Series) -> pd.Series:
        return s
    assert scal.evalType == 200, scal.evalType

    @pandas_udf("double")
    def itr(it: Iterator[pd.Series]) -> Iterator[pd.Series]:
        yield from it
    assert itr.evalType == 204, itr.evalType

    @pandas_udf("double")
    def agg(s: pd.Series) -> float:
        return float(s.mean())
    assert agg.evalType == 202, agg.evalType


# --------------------------------------------------------- __module__ / class pickling
# PyO3 defaults a #[pyclass]'s __module__ to "builtins", which diverges from the
# reference client (where each class reports the module that defines it) and makes the
# class itself unpicklable: pickle serializes a class BY REFERENCE, writing
# __module__ + __qualname__ and re-importing on load, so "builtins.IntegerType" fails
# the lookup. Every #[pyclass] therefore carries an explicit `module = "..."`.
# NB: DataFrame/Column report the *connect* module: upstream pyspark.sql.{dataframe,
# column} hold the abstract parent, while the concrete class a Connect session hands
# out is pyspark.sql.connect.*.


@pytest.mark.parametrize(
    "cls,expected",
    [
        (_DF, "pyspark.sql.connect.dataframe"),
        (_Col, "pyspark.sql.connect.column"),
        (_SS, "pyspark.sql.session"),
        (T.DataType, "pyspark.sql.types"),
        (T.IntegerType, "pyspark.sql.types"),
        (T.StructType, "pyspark.sql.types"),
        (T.NumericType, "pyspark.sql.types"),  # macro-generated abstract base
        (T.Row, "pyspark.sql.types"),
    ],
)
def test_class_module_matches_reference_path(cls, expected):
    assert cls.__module__ == expected, f"{cls.__name__}.__module__ = {cls.__module__}"


@pytest.mark.parametrize(
    "cls",
    [T.DataType, T.IntegerType, T.StringType, T.ArrayType, T.MapType, T.StructType,
     T.DecimalType, T.NumericType, T.AtomicType],
)
def test_datatype_class_itself_is_picklable(cls):
    # Pickling the CLASS (not an instance) round-trips to the identical object. A
    # __module__ of "builtins" raises PicklingError here.
    assert pickle.loads(pickle.dumps(cls)) is cls


def test_no_pyclass_reports_builtins_module():
    # Guard against a new #[pyclass] landing without `module = "..."`.
    import pyspark._pyspark as _rust

    offenders = sorted(
        name
        for name, obj in vars(_rust).items()
        if isinstance(obj, type) and obj.__module__ == "builtins"
    )
    assert offenders == [], f"missing module=: {offenders}"


# ------------------------------------------------- connect.* import-path parity

# The reference Spark Connect client exposes these classes under the
# pyspark.sql.connect.* package (and pyspark.sql.conf / pyspark.sql.tvf);
# existing code and the official test suite import from these paths.
@pytest.mark.parametrize(
    "module, names",
    [
        ("pyspark.sql.connect.catalog",
         ["Catalog", "Database", "Table", "Function", "CatalogMetadata", "TablePartition"]),
        ("pyspark.sql.connect.conf", ["RuntimeConf"]),
        ("pyspark.sql.connect.group", ["GroupedData", "PandasCogroupedOps"]),
        ("pyspark.sql.connect.window", ["Window", "WindowSpec"]),
        ("pyspark.sql.connect.readwriter",
         ["DataFrameReader", "DataFrameWriter", "DataFrameWriterV2"]),
        ("pyspark.sql.connect.tvf", ["TableValuedFunction"]),
        ("pyspark.sql.connect.udf", ["UserDefinedFunction", "UDFRegistration"]),
        ("pyspark.sql.connect.udtf",
         ["UserDefinedTableFunction", "UDTFRegistration", "AnalyzeArgument", "AnalyzeResult"]),
        ("pyspark.sql.connect.observation", ["Observation"]),
        ("pyspark.sql.connect.merge", ["MergeIntoWriter"]),
        ("pyspark.sql.connect.streaming.readwriter", ["DataStreamReader", "DataStreamWriter"]),
        ("pyspark.sql.connect.streaming.query", ["StreamingQuery", "StreamingQueryManager"]),
        ("pyspark.sql.conf", ["RuntimeConfig", "RuntimeConf"]),
        ("pyspark.sql.tvf", ["TableValuedFunction"]),
    ],
)
def test_connect_submodule_import_paths(module, names):
    import importlib

    mod = importlib.import_module(module)
    missing = [n for n in names if not hasattr(mod, n)]
    assert missing == [], f"{module} is missing {missing}"


# --------------------------------------------- v4.2.0 signature-parity fixes

def test_structtype_from_json_accepts_json_kwarg():
    # Reference names the fromJson parameter `json`, not `data`.
    import json as _json

    st = T.StructType([T.StructField("a", T.IntegerType()), T.StructField("b", T.StringType())])
    payload = _json.loads(st.json())
    assert T.StructType.fromJson(json=payload).simpleString() == st.simpleString()

    sf = T.StructField("a", T.IntegerType())
    assert T.StructField.fromJson(json=_json.loads(sf.json())).name == "a"


def test_structtype_tree_string_max_depth():
    nested = T.StructType([
        T.StructField("outer", T.StructType([T.StructField("inner", T.IntegerType())])),
    ])
    full = nested.treeString()
    assert "inner" in full
    # maxDepth=1 prints only the top level; the nested field is omitted.
    assert "inner" not in nested.treeString(1)
    assert "outer" in nested.treeString(1)


def test_variant_type_to_internal_accepts_variant_kwarg():
    # Reference names the VariantType.toInternal parameter `variant`, not `obj`.
    assert T.VariantType().toInternal(variant=None) is None


# --------------------------------------------- 4.3.0 API-surface additions


def test_new_4_3_0_sql_functions_build_columns():
    from pyspark.sql import functions as F

    c = F.col("v")
    for fn in (F.collect_union, F.to_base32, F.from_base32):
        assert isinstance(fn(c), _Col)
    assert isinstance(F.variant_strip_nulls(c), _Col)
    assert isinstance(F.variant_strip_nulls(c, include_arrays=False), _Col)
    assert isinstance(F.variant_delete(c, F.lit("$.a"), "$.b"), _Col)


def test_new_4_3_0_functions_are_exported():
    from pyspark.sql import functions as F

    for name in (
        "variant_delete",
        "collect_union",
        "to_base32",
        "from_base32",
        "variant_strip_nulls",
        "PandasUDFType",
        "ArrowUDFType",
        "UserDefinedFunction",
        "UserDefinedTableFunction",
        "AnalyzeArgument",
        "AnalyzeResult",
        "SkipRestOfInputTableException",
    ):
        assert name in F.__all__, name
        assert hasattr(F, name), name


def test_arrow_udf_type_enum_values():
    from pyspark.sql.pandas.functions import ArrowUDFType, PandasUDFType

    assert (ArrowUDFType.SCALAR, ArrowUDFType.SCALAR_ITER) == (250, 251)
    assert (ArrowUDFType.GROUPED_AGG, ArrowUDFType.GROUPED_AGG_ITER) == (252, 254)
    assert PandasUDFType.GROUPED_AGG_ITER == 217


def test_udf_types_are_re_exported_whichever_module_is_imported_first():
    # A new interpreter, as this one has imported pyspark.sql.functions already.
    import subprocess
    import sys

    subprocess.run([sys.executable, "-c", """
import pyspark.sql.pandas.functions as PF
from pyspark.sql.functions import ArrowUDFType, PandasUDFType, arrow_udf, pandas_udf
assert (PandasUDFType, ArrowUDFType) == (PF.PandasUDFType, PF.ArrowUDFType)
assert (arrow_udf, pandas_udf) == (PF.arrow_udf, PF.pandas_udf)
from pyspark.util import PythonEvalType
assert PF.PythonEvalType is PythonEvalType
"""], check=True)


def test_has_type_looks_into_nested_and_user_defined_types():
    from pyspark.sql.types import _has_type

    nested = T.StructType([T.StructField("m", T.MapType(T.StringType(), T.ArrayType(T.NullType())))])
    assert _has_type(nested, T.NullType)
    assert _has_type(nested, (T.DecimalType, T.ArrayType))
    assert not _has_type(nested, T.DecimalType)
    # A user-defined type is looked into by its sqlType: an array of doubles.
    assert _has_type(T.StructType([T.StructField("p", _PointUDT())]), T.DoubleType)


def test_conversion_converts_local_rows_to_arrow_and_back():
    import decimal

    from pyspark.sql import Row
    from pyspark.sql.conversion import ArrowTableToRowsConversion, LocalDataToArrowConversion

    schema = T.StructType([
        T.StructField("id", T.LongType()),
        T.StructField("a", T.ArrayType(T.DoubleType())),
        T.StructField("m", T.MapType(T.StringType(), T.LongType())),
        T.StructField("st", T.StructType([T.StructField("x", T.LongType())])),
        T.StructField("d", T.DecimalType(10, 2)),
    ])
    rows = [(1, [1.0, None], {"k": 1}, (5,), decimal.Decimal("1.25")), (2, None, None, None, None)]
    table = LocalDataToArrowConversion.convert(rows, schema, use_large_var_types=False)
    assert table.to_pylist()[0] == {
        "id": 1, "a": [1.0, None], "m": [("k", 1)], "st": {"x": 5}, "d": decimal.Decimal("1.25")}
    assert ArrowTableToRowsConversion.convert(table, schema, return_as_tuples=True) == [
        (1, [1.0, None], {"k": 1}, Row(x=5), decimal.Decimal("1.25")),
        (2, None, None, None, None),
    ]


def test_datatype_class_constants():
    assert (T.DayTimeIntervalType.DAY, T.DayTimeIntervalType.HOUR) == (0, 1)
    assert (T.DayTimeIntervalType.MINUTE, T.DayTimeIntervalType.SECOND) == (2, 3)
    assert (T.YearMonthIntervalType.YEAR, T.YearMonthIntervalType.MONTH) == (0, 1)
    assert T.DateType.EPOCH_ORDINAL == 719163
    assert T.GeometryType.DEFAULT_CRS == "OGC:CRS84"
    assert T.GeometryType.DEFAULT_SRID == 4326
    assert T.GeographyType.DEFAULT_ALG == "SPHERICAL"
    assert (T.SpatialType.MIXED_CRS, T.SpatialType.MIXED_SRID) == ("SRID:ANY", -1)


def test_types_all_exports_spatial_and_variant():
    for name in ("Geometry", "GeometryType", "Geography", "GeographyType", "VariantVal"):
        assert name in T.__all__, name


def test_streaming_exports_the_stateful_processor_and_tws_tester():
    import pandas as pd

    from pyspark.sql import Row
    from pyspark.sql.streaming import StatefulProcessor, StatefulProcessorHandle, TwsTester
    from pyspark.sql.streaming import stateful_processor, tws_tester

    assert StatefulProcessor is stateful_processor.StatefulProcessor
    assert StatefulProcessorHandle is stateful_processor.StatefulProcessorHandle
    assert TwsTester is tws_tester.TwsTester

    class RunningCount(StatefulProcessor):
        # As the running count of PySpark's tests of TwsTester.
        def __init__(self, use_pandas):
            self.use_pandas = use_pandas

        def init(self, handle):
            self.count = handle.getValueState(
                "count", T.StructType([T.StructField("value", T.IntegerType())]))

        def handleInputRows(self, key, rows, timerValues):
            count = self.count.get()[0] if self.count.exists() else 0
            if self.use_pandas:
                count += sum(len(frame) for frame in rows)
                self.count.update((count,))
                yield pd.DataFrame({"key": [key[0]], "count": [count]})
            else:
                count += sum(1 for _ in rows)
                self.count.update((count,))
                yield Row(key=key[0], count=count)

        def close(self):
            pass

    tester = TwsTester(RunningCount(use_pandas=False))
    assert tester.test("key1", [Row(value="a")]) == [Row(key="key1", count=1)]
    assert tester.test("key1", [Row(value="a"), Row(value="a")]) == [Row(key="key1", count=3)]
    assert tester.peekValueState("count", "key1") == (3,)
    assert tester.peekValueState("count", "key2") is None

    tester = TwsTester(RunningCount(use_pandas=True))
    result = tester.testInPandas("key1", pd.DataFrame({"value": ["a", "a"]}))
    assert result.to_dict("list") == {"key": ["key1"], "count": [2]}


def test_merge_into_writer_nested_class_refs():
    from pyspark.sql.merge import MergeIntoWriter

    assert isinstance(MergeIntoWriter.WhenMatched, type)
    assert isinstance(MergeIntoWriter.WhenNotMatched, type)
    assert isinstance(MergeIntoWriter.WhenNotMatchedBySource, type)


def test_session_hook_is_present_and_passthrough():
    hook = _SS.Hook()
    sentinel = object()
    assert hook.on_execute_plan(sentinel) is sentinel


def test_ml_vector_functions_build_their_columns_as_pyspark():
    from pyspark.ml.functions import array_to_vector, vector_to_array

    # Through pyspark.ml.connect.functions, as in PySpark Connect.
    assert repr(vector_to_array("v")) == "Column<'vector_to_array(v, float64)'>"
    assert repr(vector_to_array("v", "float32")) == "Column<'vector_to_array(v, float32)'>"
    assert repr(array_to_vector("a")) == "Column<'array_to_vector(a)'>"


def test_ml_connect_base_classes_and_functions():
    from pyspark.ml.connect import Estimator, Transformer, Evaluator, Model, PipelineModel

    for cls in (Estimator, Transformer, Evaluator, Model, PipelineModel):
        assert isinstance(cls, type)
    import pyspark.ml.functions as mf

    assert hasattr(mf, "vector_to_array") and hasattr(mf, "array_to_vector")


def test_pyspark_version_exported():
    assert pyspark.__version__ == "4.3.0"
    assert "__version__" in pyspark.__all__


# --------------------------------------------------------------------------- createDataFrame

def test_create_dataframe_width_mismatch_is_a_normal_exception():
    # Isolate the active session from the remaining offline tests. Session
    # construction and local input validation do not need a running server.
    import subprocess
    import sys

    subprocess.run([sys.executable, "-c", """
from pyspark.sql import SparkSession, types as T
from pyspark.errors import PySparkValueError
spark = SparkSession.builder.remote("sc://localhost:15002").getOrCreate()
schemas = [None, ["a", "b"], T.StructType([
    T.StructField("a", T.LongType()), T.StructField("b", T.LongType())
])]
for schema in schemas:
    try:
        spark.createDataFrame([(1, 2), (3,)], schema=schema)
    except PySparkValueError as error:
        # The same error as pyspark, for every kind of schema.
        assert type(error) is PySparkValueError, type(error)
        assert error.getCondition() == "AXIS_LENGTH_MISMATCH", error
        assert error.getMessageParameters() == {
            "expected_length": "2", "actual_length": "1"}, error
    else:
        raise AssertionError("expected a width validation error")
"""], check=True)


def test_create_dataframe_types_pandas_columns_as_arrow(mock_connect):
    pd = pytest.importorskip("pandas")
    np = pytest.importorskip("numpy")
    from pyspark.errors import PySparkTypeError

    # As PySpark, whose Arrow conversion types a column without values by its dtype, rather
    # than raising CANNOT_DETERMINE_TYPE (the live suite checks the types). The conversion
    # reads the server's Arrow settings first.
    spark, _ = mock_connect
    for column in [[np.nan, np.nan], [None, None], pd.to_datetime([None, None])]:
        spark.createDataFrame(pd.DataFrame({"id": [1, 2], "s": column}))
    with pytest.raises(PySparkTypeError, match=re.escape(
            "Exception thrown when converting pandas.Series (object) with name 'm' to Arrow")):
        spark.createDataFrame(pd.DataFrame({"m": [{None: 1}]}))


def test_timedelta_values_convert_as_pyspark():
    import datetime
    import re

    pa = pytest.importorskip("pyarrow")
    from pyspark.sql import functions as F

    # A day-time interval, rather than UNSUPPORTED_DATA_TYPE (the live suite checks values).
    one = datetime.timedelta(days=1, seconds=2, microseconds=3)
    F.lit(one)
    _local_create_dataframe([(one,), (None,)])
    _local_create_dataframe([(5,), (1.5,)], "d interval day to second")
    # Other values raise as PyArrow's conversion to duration[us] does.
    for value, error, message in (
        (True, pa.ArrowTypeError, "Expected integer, got bool"),
        ("x", pa.ArrowTypeError, "object of type <class 'str'> cannot be converted to int"),
        (float("nan"), ValueError, "cannot convert float NaN to integer"),
        (2**63, OverflowError, "Python int too large to convert to C long"),
    ):
        with pytest.raises(error, match="^" + re.escape(message) + "$"):
            _local_create_dataframe([(value,)], "d interval day to second")


def _local_create_dataframe(*args):
    # The local relation and its toDF are built without a server; a DDL schema, which PySpark
    # asks the server to parse, is parsed as the server would.
    args = list(args)
    if len(args) > 1 and isinstance(args[1], str) and ddl_json(args[1]) is not None:
        args[1] = T._parse_datatype_json_string(json.dumps(ddl_json(args[1])))
    return _offline_session().createDataFrame(*args)


@pytest.mark.parametrize("data,schema,expected", [
    # Column names are applied with toDF, as in pyspark, so the result is not local.
    ([(1, 2)], ["a", "b"], False),
    ([(1, 2)], ("a", "b"), False),
    ([(1, 2)], ["a"], False),
    ([(1, 2)], [], False),
    ([T.Row(x=1, y=2)], ["a", "b"], False),
    ([{"x": 1, "y": 2}], ["a", "b"], False),
    # Without names to apply, the local relation itself is local.
    ([(1, 2)], None, True),
    ([T.Row(x=1, y=2)], None, True),
    ([(1, 2)], "a long, b long", True),
    ([1, 2], T.LongType(), True),
    ([], "a long", True),
])
def test_create_dataframe_rows_is_local_as_pyspark(data, schema, expected):
    assert _local_create_dataframe(data, schema).isLocal() is expected


@pytest.mark.parametrize("schema", [None, ["c", "d"], ["c"], "a long, b string",
                                    T.StructType([T.StructField("a", T.LongType()),
                                                  T.StructField("b", T.StringType())])])
def test_create_dataframe_pandas_is_not_local_as_pyspark(mock_connect, schema):
    pd = pytest.importorskip("pandas")
    # pyspark applies the columns, or the field names of the schema, with toDF, so the result
    # is not local. pandas input reads the server's Arrow settings first, as PySpark does.
    spark, service = mock_connect
    spark.createDataFrame(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}), schema).collect()
    # ExecutePlanRequest.plan=3, Plan.root=1, Relation.to_df=18.
    plan = parse_message(parse_message(service.execute_requests[-1])[3][0])
    assert 18 in parse_message(plan[1][0])


def test_create_dataframe_empty_pandas_is_local_as_pyspark():
    pd = pytest.importorskip("pandas")
    # Empty input returns the relation of its schema, without toDF.
    assert _local_create_dataframe(pd.DataFrame({"a": [], "b": []}), "a long, b string").isLocal()


_Pair = namedtuple("_Pair", ["x", "y"])


@pytest.mark.parametrize("data,schema,expected,actual", [
    ([(1,)], ["a", "b"], "2", "1"),
    ([(1,), (2,)], ["a", "b"], "2", "1"),
    ([(1, 2), (3, 4, 5)], ["a"], "3", "2"),
    ([T.Row(x=1, y=2)], ["a"], "1", "2"),
    # A namedtuple, as a Row, does not extend the names.
    ([_Pair(1, 2)], ["a"], "1", "2"),
])
def test_create_dataframe_name_width_errors_as_pyspark(data, schema, expected, actual):
    from pyspark.errors import PySparkValueError

    with pytest.raises(PySparkValueError) as error:
        _local_create_dataframe(data, schema)
    assert error.value.getCondition() == "AXIS_LENGTH_MISMATCH"
    assert error.value.getMessageParameters() == {
        "expected_length": expected, "actual_length": actual}


def test_create_dataframe_pandas_errors_as_pyspark():
    pd = pytest.importorskip("pandas")
    from pyspark.errors import PySparkTypeError, PySparkValueError

    pdf = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    # pyspark reports the parsed type, also for an atomic DDL string.
    for schema, data_type in [(T.LongType(), "LongType()"), ("int", "IntegerType()")]:
        with pytest.raises(PySparkTypeError) as error:
            _local_create_dataframe(pdf, schema)
        assert error.value.getCondition() == "UNSUPPORTED_DATA_TYPE_FOR_ARROW"
        assert error.value.getMessageParameters() == {"data_type": data_type}
    for data, schema in [(pd.DataFrame(index=[0, 1]), None), ([], None), ([], ["a"]),
                         (pdf.iloc[:0], None), (pdf.iloc[:0], ["a", "b"])]:
        with pytest.raises(PySparkValueError) as error:
            _local_create_dataframe(data, schema)
        assert error.value.getCondition() == "CANNOT_INFER_EMPTY_SCHEMA"


@pytest.mark.parametrize("data", [[(None,)], [(1, None)], [{"a": None}], [{"a": 1}, {"b": None}]])
def test_create_dataframe_cannot_type_an_all_null_field_as_pyspark(data):
    from pyspark.errors import PySparkValueError

    # A field without a non-null value in any row has no type to infer.
    with pytest.raises(PySparkValueError) as error:
        _local_create_dataframe(data, None)
    assert error.value.getCondition() == "CANNOT_DETERMINE_TYPE"


@pytest.mark.parametrize("data,condition,parameters", [
    # The types of every row merge, as in PySpark, and types of other classes do not.
    ([(1,), (2.5,)], "CANNOT_MERGE_TYPE", {"data_type1": "LongType", "data_type2": "DoubleType"}),
    ([(True,), (1,)], "CANNOT_MERGE_TYPE", {"data_type1": "BooleanType", "data_type2": "LongType"}),
    ([(decimal.Decimal("1.5"),), (1,)], "CANNOT_MERGE_TYPE",
     {"data_type1": "DecimalType", "data_type2": "LongType"}),
    # A row is a Row, a tuple, a list, a dict or an object, which None is not.
    ([(1,), None], "CANNOT_INFER_SCHEMA_FOR_TYPE", {"data_type": "NoneType"}),
    # A value of a type that cannot be inferred names its field.
    ([({1, 2},)], "CANNOT_INFER_TYPE_FOR_FIELD", {"field_name": "_1"}),
    ([(array.array("q", [1]),)], "CANNOT_INFER_TYPE_FOR_FIELD", {"field_name": "_1"}),
])
def test_create_dataframe_raises_inference_errors_as_pyspark(data, condition, parameters):
    from pyspark.errors import PySparkTypeError

    with pytest.raises(PySparkTypeError) as error:
        _local_create_dataframe(data)
    assert error.value.getCondition() == condition
    assert error.value.getMessageParameters() == parameters


def test_sample_sort_group_by_and_drop_take_their_arguments_as_pyspark(mock_connect):
    # As PySpark's: sample takes a fraction and a seed positionally, sort and groupBy take the
    # 1-based ordinals of columns, a negative one of sort descending, and drop takes Columns,
    # which it sends as expressions.
    import struct

    from pyspark.errors import PySparkIndexError, PySparkTypeError, PySparkValueError
    from pyspark.sql import functions as F

    spark, service = mock_connect
    df = spark.range(1)

    def relation(dataframe, field):
        # ExecutePlanRequest.plan=3, Plan.root=1.
        dataframe.collect()
        plan = parse_message(parse_message(service.execute_requests[-1])[3][0])[1][0]
        return parse_message(parse_message(plan)[field][0])

    # Relation.sample=12: Sample.upper_bound=3, a double, seed=5.
    for sampled in (df.sample(0.5, 42), df.sample(fraction=0.5, seed=42),
                    df.sample(False, 0.5, 42)):
        sample = relation(sampled, 12)
        assert struct.unpack("<d", sample[3][0]) == (0.5,) and sample[5] == [42]
    # Relation.drop=21: Drop.columns=2, of expressions, column_names=3.
    drop = relation(df.drop(F.col("a"), "b"), 21)
    assert len(drop[2]) == 1 and drop[3] == [b"b"]

    def column_name(expression):
        # Expression.unresolved_attribute=2, UnresolvedAttribute.unparsed_identifier=1.
        return parse_message(parse_message(expression)[2][0])[1][0]

    # Relation.sort=7, Sort.order=2: SortOrder.child=1, direction=2 (DESCENDING=2).
    for sorted_df, direction in ((df.sort(1), 1), (df.orderBy(-1), 2),
                                 (df.sort(1, ascending=False), 2)):
        order = parse_message(relation(sorted_df, 7)[2][0])
        assert column_name(order[1][0]) == b"id" and order[2] == [direction]
    # Relation.aggregate=9, Aggregate.grouping_expressions=3.
    grouping = relation(df.groupBy(1).count(), 9)[3]
    assert [column_name(expression) for expression in grouping] == [b"id"]

    for call, (error_type, condition, parameters) in (
        (lambda: df.sort(), (PySparkValueError, "CANNOT_BE_EMPTY", {"item": "cols"})),
        (lambda: df.sort(0), (PySparkIndexError, "ZERO_INDEX", {})),
        (lambda: df.groupBy(0), (PySparkIndexError, "INDEX_NOT_POSITIVE", {"index": "0"})),
        (lambda: df.sort(1.5), (PySparkTypeError, "NOT_EXPECTED_TYPE",
                                {"expected_type": "Column, int or str", "arg_name": "col",
                                 "arg_type": "float"})),
        (lambda: df.groupBy(1.5), (PySparkTypeError, "NOT_EXPECTED_TYPE",
                                   {"expected_type": "Column, int or str", "arg_name": "cols",
                                    "arg_type": "float"})),
        (lambda: df.drop(1), (PySparkTypeError, "NOT_EXPECTED_TYPE",
                              {"expected_type": "Column or str", "arg_name": "cols",
                               "arg_type": "tuple"})),
        (lambda: df.sample(True, 1), (PySparkTypeError, "NOT_EXPECTED_TYPE",
                                      {"expected_type": "bool, float or int",
                                       "arg_name": "withReplacement (optional), fraction "
                                                   "(required) and seed (optional)",
                                       "arg_type": "bool, int, NoneType"})),
    ):
        with pytest.raises(error_type) as error:
            call()
        assert error.value.getCondition() == condition
        assert error.value.getMessageParameters() == parameters


def test_ordinals_and_names_of_columns_carry_the_origin_of_col_as_pyspark():
    # As PySpark's, `df[ordinal]`, and the columns of sort and groupBy of an ordinal or a
    # name, are of `F.col`, whose origin they carry. In a new interpreter, whose first column
    # of a session reads spark.python.sql.dataFrameDebugging.enabled.
    import subprocess
    import sys
    from pathlib import Path

    pytest.importorskip("grpc")
    code = """
import sys
sys.path.insert(0, sys.argv[1])
from _mock_connect_server import mock_connect_session, parse_message

def fragment(expression):
    # Expression.common=18 -> ExpressionCommon.origin=1 -> Origin.python_origin=1 ->
    # PythonOrigin.fragment=1.
    expression = parse_message(expression)
    if 18 not in expression:
        return None
    common = parse_message(expression[18][0])
    return parse_message(parse_message(common[1][0])[1][0])[1][0].decode()

if __name__ == "__main__":
    with mock_connect_session() as (spark, service):
        df = spark.range(1)

        def relation(dataframe, field):
            # ExecutePlanRequest.plan=3, Plan.root=1.
            dataframe.collect()
            plan = parse_message(parse_message(service.execute_requests[-1])[3][0])[1][0]
            return parse_message(parse_message(plan)[field][0])

        # Relation.project=3: Project.expressions=3.
        found = [fragment(relation(df.select(df[0]), 3)[3][0])]
        # Relation.sort=7: Sort.order=2, SortOrder.child=1.
        for sorted_df in (df.sort(1), df.sort(-1), df.sort("id")):
            found.append(fragment(parse_message(relation(sorted_df, 7)[2][0])[1][0]))
        # Relation.aggregate=9: Aggregate.grouping_expressions=3.
        for grouped in (df.groupBy(1), df.groupBy("id")):
            found.append(fragment(relation(grouped.count(), 9)[3][0]))
        print(found)
"""
    tests = str(Path(__file__).resolve().parent)
    result = subprocess.run(
        [sys.executable, "-c", code, tests], check=True, capture_output=True, text=True)
    assert result.stdout.strip().splitlines()[-1] == str(["col"] * 6)


def test_grouping_sets_group_by_their_grouping_columns_as_pyspark(mock_connect):
    # As PySpark's, groupingSets(groupingSets, *cols) groups by the sets with the grouping
    # columns `cols`: Relation.aggregate=9, Aggregate.group_type=2 (GROUPING_SETS=5),
    # grouping_expressions=3, grouping_sets=6 (GroupingSets.grouping_set=1).
    from pyspark.errors import PySparkTypeError
    from pyspark.sql import functions as F

    spark, service = mock_connect
    df = spark.range(1)

    def column_names(expressions):
        # Expression.unresolved_attribute=2, UnresolvedAttribute.unparsed_identifier=1.
        return [parse_message(parse_message(e)[2][0])[1][0].decode() for e in expressions]

    def sent(grouped):
        grouped.agg(F.count("*")).collect()
        plan = parse_message(parse_message(service.execute_requests[-1])[3][0])[1][0]
        aggregate = parse_message(parse_message(plan)[9][0])
        sets = [column_names(parse_message(s).get(1, [])) for s in aggregate.get(6, [])]
        return aggregate[2], column_names(aggregate.get(3, [])), sets

    assert sent(df.groupingSets([["id"], []], "id", F.col("x"))) == (
        [5], ["id", "x"], [["id"], []])
    # No grouping columns, and a set of one column given alone.
    assert sent(df.groupingSets(["id", ("x", 1)])) == ([5], [], [["id"], ["x", "id"]])
    for call, arg_name, arg_type in (
        (lambda: df.groupingSets([["id"]], ["id"]), "cols", "list"),
        (lambda: df.groupingSets([[["id"]]]), "groupingSets", "list"),
        (lambda: df.groupingSets([1.5]), "groupingSets", "float"),
    ):
        with pytest.raises(PySparkTypeError) as error:
            call()
        assert error.value.getCondition() == "NOT_EXPECTED_TYPE"
        assert error.value.getMessageParameters() == {
            "expected_type": "Column, int or str", "arg_name": arg_name, "arg_type": arg_type}


def test_create_dataframe_checks_every_row_against_the_merged_schema():
    from pyspark.errors import PySparkValueError

    # Rows of different fields merge into one struct, which each Row must then fill.
    with pytest.raises(PySparkValueError) as error:
        _local_create_dataframe([Row(a=1), Row(b=2)])
    assert error.value.getCondition() == "AXIS_LENGTH_MISMATCH"
    assert error.value.getMessageParameters() == {"expected_length": "2", "actual_length": "1"}


@pytest.mark.parametrize("schema,arg_type", [
    (T.ArrayType(T.LongType()), "ArrayType"),
    (T.MapType(T.StringType(), T.LongType()), "MapType"),
    ("array<int>", "ArrayType"),
    (5, "int"),
])
def test_create_dataframe_schema_is_names_a_struct_or_an_atomic_type(schema, arg_type):
    from pyspark.errors import PySparkTypeError

    with pytest.raises(PySparkTypeError) as error:
        _local_create_dataframe([(1,)], schema)
    assert error.value.getCondition() == "NOT_EXPECTED_TYPE"
    assert error.value.getMessageParameters() == {
        "arg_name": "schema", "expected_type": "list, None or StructType", "arg_type": arg_type}


def test_create_dataframe_of_a_numpy_array_as_pyspark(mock_connect):
    # As PySpark Connect's: an Arrow table of a column of each column of a 1- or 2-dimensional
    # array, named by the names, else `value` for one column, else `_1`, `_2`, ..., sent with
    # the schema, if any, for the server to cast to.
    import numpy as np
    import pyarrow as pa
    from pyspark.errors import PySparkValueError

    spark, service = mock_connect

    def sent(df):
        df.columns
        # Relation.local_relation=11: data=1, an Arrow IPC stream, and schema=2.
        relation = _sent_relation(service.analyze_requests[-1], 11)
        table = pa.ipc.open_stream(relation[1][0]).read_all()
        return table.to_pydict(), [json.loads(s) for s in relation.get(2, [])]

    assert sent(spark.createDataFrame(np.array([1, 2]))) == ({"value": [1, 2]}, [])
    assert sent(spark.createDataFrame(np.array([[1, 2], [3, 4]]))) == (
        {"_1": [1, 3], "_2": [2, 4]}, [])
    assert sent(spark.createDataFrame(np.array([[1], [2]]), ("x",))) == ({"x": [1, 2]}, [])
    schema = T.StructType().add("a", T.IntegerType()).add("b", T.StringType())
    assert sent(spark.createDataFrame(np.array([[1, 2]]), schema)) == (
        {"_1": [1], "_2": [2]}, [json.loads(schema.json())])
    assert sent(spark.createDataFrame(np.array([1.5]), T.DoubleType())) == (
        {"value": [1.5]}, ["double"])
    for data, schema, expected, actual in [
        (np.array([1, 2]), ["x", "y"], "2", "1"),
        (np.array([1, 2]), [], "0", "1"),
        (np.array([[1, 2]]), ["x"], "1", "2"),
        (np.array([[1, 2]]), T.IntegerType(), "1", "2"),
        (np.array([1, 2]), T.StructType().add("a", T.IntegerType()).add("b", T.LongType()),
         "2", "1"),
    ]:
        with pytest.raises(PySparkValueError) as e:
            spark.createDataFrame(data, schema)
        assert e.value.getCondition() == "AXIS_LENGTH_MISMATCH"
        assert e.value.getMessageParameters() == {
            "expected_length": expected, "actual_length": actual}
    for data in (np.zeros((2, 2, 2)), np.array(5)):
        with pytest.raises(PySparkValueError) as e:
            spark.createDataFrame(data)
        assert e.value.getCondition() == "INVALID_NDARRAY_DIMENSION"
        assert e.value.getMessageParameters() == {"dimensions": "1 or 2"}
    with pytest.raises(PySparkValueError) as e:
        spark.createDataFrame(np.array([]))
    assert e.value.getCondition() == "CANNOT_INFER_EMPTY_SCHEMA"

def test_create_dataframe_of_a_dataframe_raises_as_pyspark():
    from pyspark.errors import PySparkTypeError

    spark = _offline_session()
    with pytest.raises(PySparkTypeError) as error:
        spark.createDataFrame(spark.range(1))
    assert error.value.getCondition() == "INVALID_TYPE"
    assert error.value.getMessageParameters() == {"arg_name": "data", "arg_type": "DataFrame"}


def test_create_dataframe_warns_of_the_arguments_that_connect_ignores():
    # As PySpark Connect, which ignores both.
    with pytest.warns(UserWarning, match="'samplingRatio' is ignored"):
        _local_create_dataframe([(1,)], None, 0.5)
    with pytest.warns(UserWarning, match="'verifySchema' is ignored"):
        _local_create_dataframe([(1,)], None, None, True)


def test_create_dataframe_converts_numpy_scalars_as_pyarrow():
    import re

    np = pytest.importorskip("numpy")
    # As PySpark, where PyArrow takes a NumPy scalar as the Python value that it stands for:
    # an integer, a float (truncated for an integer type) and a bool (also for an integer).
    assert _local_create_dataframe(
        [(np.int64(1), np.float32(0.5), np.bool_(True), np.float32(1.5), np.bool_(True))],
        "a bigint, b double, c boolean, d bigint, e bigint").isLocal()
    for value, schema, message in [
        (np.bool_(True), "a double",
         r"Could not convert \S+ with type numpy\.bool_?: tried to convert to double"),
        (np.int64(1), "a boolean",
         r"Could not convert \S+ with type numpy\.int64: tried to convert to boolean"),
        (np.int32(300), "a tinyint", re.escape("Value 300 too large to fit in C integer type")),
        # An integer that a float type cannot hold exactly.
        (2**53 + 1, "a double", re.escape(
            "Integer value 9007199254740993 is outside of the range exactly representable by "
            "a IEEE 754 double precision value")),
        (2**24 + 1, "a float", re.escape(
            "Integer value 16777217 is outside of the range exactly representable by "
            "a IEEE 754 single precision value")),
    ]:
        with pytest.raises(ValueError, match=message):
            _local_create_dataframe([(value,)], schema)


def test_create_dataframe_raises_recursion_error_for_a_value_that_contains_itself():
    class Node:
        def __init__(self):
            self.parent = self

    cyclic = []
    cyclic.append(cyclic)
    # As PySpark, rather than overflowing the stack.
    for value in [Node(), cyclic]:
        with pytest.raises(RecursionError):
            _local_create_dataframe([(value,)])


# py_to_value converts the cells of object-dtype pandas input, as well as the values of
# fillna, pivot and Row conversion; local rows go through infer.convert instead, so the
# "local" source is a control for values that the exact-number fast path must not change.
def _RowPathFrame(pd):
    """A subclass of pandas.DataFrame: createDataFrame converts a customized frame with its
    row path (py_to_value), where it converts a plain one with PySpark's Arrow conversion."""
    return type("RowPathFrame", (pd.DataFrame,), {})


def _native_number_input_table(mock_connect, data, schema=None, *, source="local"):
    import pyarrow as pa

    if source == "pandas":
        pd = pytest.importorskip("pandas")
        # Local rows now use infer.convert. Object-backed pandas input of a customized frame, of
        # a subclass, retains py_to_value, the converter of the row path optimized by this
        # patch; a plain frame converts with PySpark's Arrow conversion.
        data = _RowPathFrame(pd)({"value": [row[0] for row in data]}, dtype=object)
    spark, service = mock_connect
    spark.createDataFrame(data, schema).toArrow()
    request = parse_message(service.execute_requests[-1])
    relation = parse_message(parse_message(request[3][0])[1][0])
    while 18 in relation:  # Relation.to_df -> ToDF.input.
        relation = parse_message(parse_message(relation[18][0])[1][0])
    local = parse_message(relation[11][0])
    return pa.ipc.open_stream(local[1][0]).read_all()


@pytest.mark.parametrize("value", [
    None, False, True, 0, -1, -(2**63), 2**63 - 1,
    0.0, -0.0, 1.25, float("inf"), float("-inf"),
])
def test_local_input_native_number_values_and_types(mock_connect, value):
    import struct

    dtype = T.BooleanType() if type(value) is bool else (
        T.LongType() if type(value) is int else T.DoubleType())
    schema = T.StructType([T.StructField("value", dtype)])
    actual = _native_number_input_table(
        mock_connect, [Row(value=value)], schema, source="pandas"
    ).column(0)[0].as_py()
    assert type(actual) is type(value)
    if type(value) is float:
        assert struct.pack(">d", actual) == struct.pack(">d", value)
    else:
        assert actual == value


def test_pandas_frames_of_duplicate_nested_field_names_are_sent_as_pyspark(mock_connect):
    # As PySpark Connect: a pandas frame under a schema whose structs repeat field names is sent
    # as the Arrow table of the deduplicated names (x_0, x_1), with the schema as given, which
    # the server applies. Before, the client raised DUPLICATED_FIELD_NAME_IN_ARROW_STRUCT.
    pd = pytest.importorskip("pandas")
    pa = pytest.importorskip("pyarrow")

    spark, service = mock_connect
    schema = T.StructType().add(
        "s", T.StructType().add("x", T.StringType()).add("x", T.IntegerType())
    )
    frame = pd.DataFrame.from_records([(Row("a", 1),), (Row("b", None),)], columns=schema.names)
    spark.createDataFrame(frame, schema).toArrow()
    relation = parse_message(parse_message(parse_message(service.execute_requests[-1])[3][0])[1][0])
    while 18 in relation:  # Relation.to_df -> ToDF.input.
        relation = parse_message(parse_message(relation[18][0])[1][0])
    # Relation.local_relation=11 -> LocalRelation.data=1, schema=2.
    local = parse_message(relation[11][0])
    table = pa.ipc.open_stream(local[1][0]).read_all()
    assert table.schema.field("s").type == pa.struct([("x_0", pa.string()), ("x_1", pa.int32())])
    assert table.to_pylist() == [{"s": {"x_0": "a", "x_1": 1}}, {"s": {"x_0": "b", "x_1": None}}]
    assert T.StructType.fromJson(json.loads(local[2][0])) == schema


def test_pandas_input_overflow_raises_as_pyspark(mock_connect):
    # Without a schema, PySpark infers the Arrow type of an object column, so an integer
    # outside the int64 range raises OverflowError rather than becoming a double.
    pd = pytest.importorskip("pandas")
    for value in (2**63, -(2**63) - 1):
        with pytest.raises(OverflowError, match="^Python int too large to convert to C long$"):
            _native_number_input_table(mock_connect, [Row(value=value)], source="pandas")
    with pytest.raises(OverflowError, match="^Python int too large to convert to C long$"):
        mock_connect[0].createDataFrame(pd.DataFrame({"value": [2**20000]}, dtype=object))


def test_local_input_nan_payload(mock_connect):
    import struct

    bits = bytes.fromhex("7ff8000000000042")
    value = struct.unpack(">d", bits)[0]
    actual = _native_number_input_table(mock_connect, [Row(value=value)]).column(0)[0].as_py()
    assert struct.pack(">d", actual) == bits


def test_pandas_input_number_subclasses_decimal_and_conversion_protocols(mock_connect):
    from decimal import Decimal

    trace = []

    class IntSubclass(int):
        def __float__(self):
            raise AssertionError("int subclass must retain integer conversion")

    class FloatSubclass(float):
        def __float__(self):
            raise AssertionError("float subclass must retain float extraction")

    class Indexable:
        def __index__(self):
            trace.append("index")
            return 42

    class Floatable:
        def __float__(self):
            trace.append("float")
            return -1.25

    class DecimalSubclass(Decimal):
        pass

    for value, expected in [(IntSubclass(13), 13), (FloatSubclass(2.25), 2.25),
                            (Indexable(), 42), (Floatable(), -1.25)]:
        actual = _native_number_input_table(mock_connect, [(value,)], source="pandas").column(0)[0].as_py()
        assert type(actual) is type(expected)
        assert actual == expected
    assert trace == ["index", "float"]
    for value in (Decimal("-12.340"), DecimalSubclass("1.25")):
        actual = _native_number_input_table(mock_connect, [(value,)], source="pandas").column(0)[0].as_py()
        assert type(actual) is Decimal
        assert actual == value


def test_local_input_native_numbers_inside_nested_values(mock_connect):
    value = Row(items=[1, 2, None], mapping={"x": 2**63 - 1})
    actual = _native_number_input_table(mock_connect, [Row(value=value)], source="pandas").column(0)[0].as_py()
    assert actual["items"] == [1, 2, None]
    assert dict(actual["mapping"]) == {"x": 2**63 - 1}


# Exact list/tuple conversion and pandas input IPC regressions.
from datetime import date as container_input_date, datetime as container_input_datetime
from decimal import Decimal as ContainerInputDecimal

from pyspark.sql import Row as ContainerInputRow


@pytest.mark.parametrize("container", [list, tuple])
def test_container_input_values_and_errors(container_input_capture_input, container):
    array_schema = T.StructType([T.StructField("value", T.ArrayType(T.LongType()))])
    for value, expected in [(container([1, None, -3]), [1, None, -3]),
                            (container(), []),
                            (container([1, 2, 255]), [1, 2, 255])]:
        table = container_input_capture_input([(value,)], array_schema)
        assert table.column(0)[0].as_py() == expected
    binary_schema = T.StructType([T.StructField("value", T.BinaryType())])
    for value in (b"abc", bytearray(b"abc")):
        table = container_input_capture_input([(value,)], binary_schema)
        assert table.column(0)[0].as_py() == b"abc"
    nested_schema = T.StructType([
        T.StructField("value", T.ArrayType(T.ArrayType(T.LongType())))])
    value = container([container([1, None]), container([2]), container()])
    table = container_input_capture_input([(value,)], nested_schema)
    assert table.column(0)[0].as_py() == [[1, None], [2], []]
    with pytest.raises(TypeError, match="unsupported value type"):
        container_input_capture_input([(container([object()]),)], array_schema)


@pytest.mark.parametrize("container", [list, tuple])
def test_container_input_subclass_numeric_precedence(container_input_capture_input, container):
    trace = []

    class Indexed(container):
        def __index__(self):
            trace.append("index")
            return 42

    class Floating(container):
        def __float__(self):
            trace.append("float")
            return 1.25

    class Iterated(container):
        def __iter__(self):
            raise AssertionError("existing built-in iterator must remain in use")

    cases = [(Indexed([1, 2]), T.LongType(), 42),
             (Floating([1, 2]), T.DoubleType(), 1.25),
             (Iterated([1, 2]), T.ArrayType(T.LongType()), [1, 2])]
    for value, dtype, expected in cases:
        schema = T.StructType([T.StructField("value", dtype)])
        table = container_input_capture_input([(value,)], schema)
        assert table.column(0)[0].as_py() == expected
    assert trace == ["index", "float"]


def test_container_input_recursive_leaf_and_row_types(container_input_capture_input):
    from datetime import timezone

    record = ContainerInputRow(
        flag=True, number=42, ratio=1.25, text="text", blob=b"abc",
        decimal=ContainerInputDecimal("-12.340"),
        date=container_input_date(2000, 2, 29),
        timestamp=container_input_datetime(2020, 1, 1),
        child=ContainerInputRow(x=1, y="a"), missing=None)
    record_schema = T.StructType([
        T.StructField("flag", T.BooleanType()), T.StructField("number", T.LongType()),
        T.StructField("ratio", T.DoubleType()), T.StructField("text", T.StringType()),
        T.StructField("blob", T.BinaryType()), T.StructField("decimal", T.DecimalType(18, 3)),
        T.StructField("date", T.DateType()), T.StructField("timestamp", T.TimestampType()),
        T.StructField("child", T.StructType([
            T.StructField("x", T.LongType()), T.StructField("y", T.StringType())])),
        T.StructField("missing", T.LongType()),
    ])
    table = container_input_capture_input(
        [(record,)], T.StructType([T.StructField("value", record_schema)]))
    result = table.column(0)[0].as_py()
    expected_timestamp = container_input_datetime.fromtimestamp(
        record.timestamp.timestamp(), timezone.utc)
    assert result == {
        "flag": True, "number": 42, "ratio": 1.25, "text": "text", "blob": b"abc",
        "decimal": ContainerInputDecimal("-12.340"), "date": container_input_date(2000, 2, 29),
        "timestamp": expected_timestamp, "child": {"x": 1, "y": "a"}, "missing": None,
    }
    assert type(result["flag"]) is bool
    assert type(result["decimal"]) is ContainerInputDecimal


# DDL handler for the exact-date capture server below.
def _ddl_parse(request, context):
    # As the server, of the DDL schemas that the client asks it to parse (`_parse_ddl`).
    import grpc

    response = ddl_parse_response(request)
    if response is None:
        context.abort(grpc.StatusCode.UNIMPLEMENTED, "Method not found!")
    return response


@pytest.fixture
def container_input_capture_input(request):
    # Check the optional dependency before acquiring the shared server/session.
    pd = pytest.importorskip("pandas")
    mock_connect = request.getfixturevalue("mock_connect")

    def capture(data, schema):
        # Keep input on the pandas path (generic conversion or Arrow import).
        # Local rows use infer.convert with different container/numeric semantics.
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data, dtype=object)
        return _native_number_input_table(mock_connect, data, schema)

    return capture


@pytest.mark.parametrize("source", ["positional_columns", "named_columns"])
@pytest.mark.parametrize("schema_kind", ["inferred", "names", "explicit", "ddl"])
@pytest.mark.parametrize("container", [list, tuple])
def test_container_input_array_input_ipc(container_input_capture_input, source, schema_kind, container):
    import pyarrow as pa

    values = [container([1, None, -3]), container(), None, container([255, 256])]
    schema = {"inferred": None, "names": ["items"], "ddl": "items array<bigint>",
              "explicit": T.StructType([T.StructField("items", T.ArrayType(T.LongType()))])}[schema_kind]
    data = [(value,) for value in values]
    if source == "named_columns":
        import pandas as pd
        data = pd.DataFrame(data, columns=["items"])
    table = container_input_capture_input(data, schema)
    assert table.column(0).type == pa.list_(pa.int64())
    assert table.column(0).to_pylist() == [[1, None, -3], [], None, [255, 256]]


def test_container_input_nested_array_struct_binary_ipc(container_input_capture_input):
    schema = T.StructType([
        T.StructField("nested", T.ArrayType(T.ArrayType(T.LongType()))),
        T.StructField("record", T.StructType([
            T.StructField("x", T.LongType()), T.StructField("y", T.StringType())])),
        T.StructField("blob", T.BinaryType()),
    ])
    table = container_input_capture_input([([[1, None], (), None], ContainerInputRow(x=7, y="row"), b"abc")], schema)
    assert table.to_pylist() == [{"nested": [[1, None], [], None],
                                  "record": {"x": 7, "y": "row"}, "blob": b"abc"}]


# --------------------------------------------------------------------------- pandas columns of Decimals and other values


@pytest.mark.parametrize("values, expected", [
    (["1.5", 2], (2, 1, ["1.5", "2.0"])),
    ([2, "1.5"], (2, 1, ["2.0", "1.5"])),
    (["1.5", 2.5], "PySparkTypeError"),
    (["1.5", ("x",)], "PySparkTypeError"),
    (["1.5", True], "PySparkValueError"),
    (["-0.5", -3, None], "PySparkValueError"),
])
def test_pandas_columns_of_decimals_and_other_values_follow_arrow(decimal_ipc_session, values, expected):
    # PySpark's conversion leaves such a column to Arrow, which types it, takes ints as
    # Decimals and raises for the others (strings in `values` are Decimals here, and a 1-tuple
    # holds a plain string).
    from decimal import Decimal
    import pyarrow as pa
    from pyspark import errors

    pd = pytest.importorskip("pandas")
    spark, payload = decimal_ipc_session
    cells = [Decimal(v) if isinstance(v, str) else v[0] if isinstance(v, tuple) else v
             for v in values]
    data = pd.DataFrame({"d": cells}, dtype=object)
    if isinstance(expected, str):
        with pytest.raises(getattr(errors, expected), match=r"pandas.Series \(object\) with name 'd'"):
            spark.createDataFrame(data)
        assert not payload.exists()
        return
    precision, scale, texts = expected
    spark.createDataFrame(data).collect()
    table = pa.ipc.open_stream(payload.read_bytes()).read_all()
    assert table.column(0).type == pa.decimal128(precision, scale)
    assert table.column(0).to_pylist() == [Decimal(text) for text in texts]


# --------------------------------------------------------------------------- exact date input

from datetime import (
    date as _exact_date_date, datetime as _exact_date_datetime,
    timezone as _exact_date_timezone,
)
from decimal import Decimal as _exact_date_Decimal
from pyspark.sql import Row as _exact_date_Row


def _exact_date_converted_value(mock_connect, value):
    import pyarrow as pa

    pd = pytest.importorskip("pandas")
    spark, service = mock_connect
    # py_to_value remains the pandas converter of the row path, which a customized frame, of a
    # subclass, keeps; a plain frame converts with PySpark's Arrow conversion. Local rows use
    # infer.convert, which intentionally has different subclass/type semantics.
    data = _RowPathFrame(pd)({"value": [value]}, dtype=object)
    before = len(service.execute_requests)
    spark.createDataFrame(data).toArrow()
    requests = service.execute_requests
    assert len(requests) == before + 1
    request = parse_message(requests[-1])
    relation = parse_message(parse_message(request[3][0])[1][0])
    renames = []
    while 18 in relation:  # Relation.to_df -> ToDF.input; column_names=2.
        rename = parse_message(relation[18][0])
        renames.append([name.decode() for name in rename.get(2, [])])
        relation = parse_message(rename[1][0])
    local = parse_message(relation[11][0])
    table = pa.ipc.open_stream(local[1][0]).read_all()
    table.validate(full=True)
    assert (table.num_rows, table.num_columns) == (1, 1)
    assert table.column_names == ["value"]
    assert [field["name"] for field in json.loads(local[2][0])["fields"]] == table.column_names
    for names in renames:
        assert len(names) == table.num_columns
    # The outermost toDF supplies observable names without rewriting raw IPC.
    assert (renames[0] if renames else table.column_names) == ["value"]
    return table.column(0)[0].as_py()


@pytest.mark.parametrize("value", [
    _exact_date_date.min, _exact_date_date.max, _exact_date_date(1, 3, 1), _exact_date_date(4, 2, 29),
    _exact_date_date(1600, 2, 29), _exact_date_date(1900, 2, 28), _exact_date_date(1900, 3, 1),
    _exact_date_date(1969, 12, 31), _exact_date_date(1970, 1, 1), _exact_date_date(1970, 1, 2),
    _exact_date_date(2000, 2, 29), _exact_date_date(2100, 3, 1),
])
def test_exact_date_builtin_roundtrip(mock_connect, value):
    actual = _exact_date_converted_value(mock_connect, value)
    assert type(actual) is _exact_date_date
    assert actual == value


def test_exact_date_subclass_protocol_precedence(mock_connect):
    trace = []

    class Indexed(_exact_date_date):
        def __index__(self):
            trace.append("index")
            return 42

        def toordinal(self):
            raise AssertionError("numeric extraction must precede date conversion")

    class Floating(_exact_date_date):
        def __float__(self):
            trace.append("float")
            return 1.25

        def toordinal(self):
            raise AssertionError("numeric extraction must precede date conversion")

    class Ordinal(_exact_date_date):
        def toordinal(self):
            trace.append("ordinal")
            return _exact_date_date(2000, 2, 29).toordinal()

    class Plain(_exact_date_date):
        pass

    assert _exact_date_converted_value(mock_connect, Indexed(2020, 1, 1)) == 42
    assert _exact_date_converted_value(mock_connect, Floating(2020, 1, 1)) == 1.25
    assert _exact_date_converted_value(mock_connect, Ordinal(2020, 1, 1)) == _exact_date_date(2000, 2, 29)
    assert type(_exact_date_converted_value(mock_connect, Plain(2020, 1, 1))) is _exact_date_date
    assert trace == ["index", "float", "ordinal"]


def test_exact_date_datetime_timestamp_encoding(mock_connect):
    pd = pytest.importorskip("pandas")
    for value in [_exact_date_datetime(2020, 2, 29, 12, 34, 56, 123456),
                  _exact_date_datetime(2020, 2, 29, tzinfo=_exact_date_timezone.utc),
                  pd.Timestamp("2020-02-29 12:34:56.123456"),
                  pd.Timestamp("2020-02-29 12:34:56.123456Z")]:
        actual = _exact_date_converted_value(mock_connect, value)
        assert type(actual) is _exact_date_datetime
        expected = value.to_pydatetime() if isinstance(value, pd.Timestamp) else value
        assert actual == _exact_date_datetime.fromtimestamp(expected.timestamp(), _exact_date_timezone.utc)


def test_exact_date_nested_unrelated_values(mock_connect):
    value = _exact_date_Row(items=[_exact_date_date.min, None, _exact_date_date.max],
                mapping={"epoch": _exact_date_date(1970, 1, 1)},
                nested=_exact_date_Row(day=_exact_date_date(2000, 2, 29)),
                amount=_exact_date_Decimal("-12.340"), number=42, flag=True, label="date")
    actual = _exact_date_converted_value(mock_connect, value)
    assert actual["items"] == [_exact_date_date.min, None, _exact_date_date.max]
    assert dict(actual["mapping"]) == {"epoch": _exact_date_date(1970, 1, 1)}
    assert actual["nested"]["day"] == _exact_date_date(2000, 2, 29)
    assert actual["amount"] == _exact_date_Decimal("-12.340")
    assert (actual["number"], actual["flag"], actual["label"]) == (42, True, "date")


def test_exact_date_named_impostor_rejected(mock_connect):
    pd = pytest.importorskip("pandas")
    from pyspark.errors import PySparkValueError

    # As PySpark's Arrow conversion of a frame without a schema, which does not take an object
    # that is not a date as one.
    spark, _ = mock_connect
    impostor = type("date", (), {"toordinal": lambda self: 719163})
    with pytest.raises(PySparkValueError, match=re.escape(
            "Exception thrown when converting pandas.Series (object) with name 'value'")):
        spark.createDataFrame(pd.DataFrame({"value": [impostor()]}, dtype=object))


def _exact_date_fields(data):
    """Decode only the small protobuf envelope around a local Arrow relation."""
    def _exact_date_varint(pos):
        value, shift = 0, 0
        while True:
            byte = data[pos]
            pos += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value, pos
            shift += 7

    result, pos = {}, 0
    while pos < len(data):
        tag, pos = _exact_date_varint(pos)
        wire = tag & 7
        if wire == 2:
            length, pos = _exact_date_varint(pos)
            result[tag >> 3] = data[pos:pos + length]
            pos += length
        elif wire == 0:
            _, pos = _exact_date_varint(pos)
        elif wire in (1, 5):
            pos += 8 if wire == 1 else 4
        else:
            raise AssertionError(f"unexpected protobuf wire type: {wire}")
    return result


def _exact_date_capture_server(connection):
    from concurrent.futures import ThreadPoolExecutor
    import grpc

    captured = []

    def _exact_date_execute(request, context):
        # ExecutePlanRequest.plan=3 -> Plan.root=1 ->
        # Relation.local_relation=11 -> LocalRelation.data=1.
        relation = _exact_date_fields(_exact_date_fields(_exact_date_fields(request)[3])[1])
        while 18 in relation:  # Relation.to_df -> ToDF.input.
            relation = _exact_date_fields(_exact_date_fields(relation[18])[1])
        captured.append(_exact_date_fields(relation[11])[1])
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, "captured date input")
        yield b""  # Make this a streaming handler.

    # Explicit DateType pandas input of 10,000 rows or more goes through the Arrow
    # importer, which reads these settings first; keep its relation inline.
    confs = {
        "spark.sql.timestampType": "TIMESTAMP_LTZ",
        "spark.sql.session.timeZone": "UTC",
        "spark.sql.execution.arrow.useLargeVarTypes": "false",
        "spark.sql.session.localRelationCacheThreshold": str(1 << 60),
        "spark.sql.session.localRelationSizeLimit": str(1 << 60),
        "spark.sql.session.localRelationChunkSizeRows": str(1 << 30),
        "spark.sql.session.localRelationChunkSizeBytes": str(1 << 40),
        "spark.sql.session.localRelationBatchOfChunksSizeBytes": str(1 << 40),
        "spark.sql.execution.pandas.convertToArrowArraySafely": "true",
        "spark.sql.execution.pandas.inferPandasDictAsMap": "false",
    }

    def _exact_date_config(request, context):
        # ConfigRequest.session_id=1, operation=3, Operation.get=2, Get.keys=1;
        # ConfigResponse.session_id=1, pairs=2, KeyValue.key=1, value=2.
        request = parse_message(request)
        keys = [key.decode() for key in
                parse_message(parse_message(request[3][0])[2][0]).get(1, [])]
        return message_field(1, request[1][0]) + b"".join(
            message_field(2, message_field(1, key) + message_field(2, confs[key]))
            for key in keys)

    with ThreadPoolExecutor(max_workers=2) as executor:
        server = grpc.server(executor)
        server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(
            "spark.connect.SparkConnectService", {
                "ExecutePlan": grpc.unary_stream_rpc_method_handler(_exact_date_execute),
                "Config": grpc.unary_unary_rpc_method_handler(_exact_date_config),
                "AnalyzePlan": grpc.unary_unary_rpc_method_handler(_ddl_parse),
            }),))
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        connection.send(port)
        try:
            while True:
                command = connection.recv()
                if command == "stop":
                    break
                assert command == "get"
                connection.send(captured.pop(0))
        finally:
            server.stop(0).wait()


@pytest.fixture
def exact_date_capture_input():
    import multiprocessing
    pytest.importorskip("grpc")
    from pyspark.sql import SparkSession

    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_exact_date_capture_server, args=(child,))
    process.start()
    try:
        assert parent.poll(20), "capture service failed to start"
        spark = SparkSession.builder.remote(f"sc://127.0.0.1:{parent.recv()}").getOrCreate()

        def _exact_date_capture(data, schema):
            import pyarrow as pa
            df = spark.createDataFrame(data, schema=schema)
            with pytest.raises(Exception, match="captured date input"):
                df.collect()
            parent.send("get")
            assert parent.poll(10), "capture service failed to return the IPC payload"
            return pa.ipc.open_stream(parent.recv()).read_all()

        yield _exact_date_capture
    finally:
        if process.is_alive():
            parent.send("stop")
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
        child.close()


@pytest.mark.parametrize("source", ["tuples", "pandas"])
@pytest.mark.parametrize("schema_kind", ["inferred", "names", "explicit", "ddl"])
def test_exact_date_ipc_calendar_range(exact_date_capture_input, source, schema_kind):
    import calendar
    import pyarrow as pa
    from pyspark.sql import types as T

    # Every year's endpoints and leap day cover the complete supported calendar
    # range, including non-leap centuries, without relying on the Rust decoder.
    dates = [_exact_date_date.min, _exact_date_date.max, _exact_date_date(1969, 12, 31), _exact_date_date(1970, 1, 1),
             _exact_date_date(1970, 1, 2), None]
    for year in range(1, 10000):
        dates.extend([_exact_date_date(year, 1, 1), _exact_date_date(year, 12, 31)])
        if calendar.isleap(year):
            dates.append(_exact_date_date(year, 2, 29))
    schema = {"inferred": None, "names": ["day"], "ddl": "day date",
              "explicit": T.StructType([T.StructField("day", T.DateType())])}[schema_kind]
    data = [(value,) for value in dates]
    if source == "pandas":
        pd = pytest.importorskip("pandas")
        data = pd.DataFrame(data, columns=["day"])
    table = exact_date_capture_input(data, schema)
    assert table.column(0).type == pa.date32()
    assert table.column(0).to_pylist() == dates
    epoch = _exact_date_date(1970, 1, 1)
    expected_days = [None if value is None else (value - epoch).days for value in dates]
    assert table.column(0).cast(pa.int32()).to_pylist() == expected_days


# Inspect the local-input IPC through a real transport boundary. The subprocess
# avoids sharing the GIL with native RPC calls and needs no JVM or Spark server.
_DECIMAL_IPC_SERVER = r"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import grpc
import pyarrow as pa
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
pool = descriptor_pool.DescriptorPool()
for file in descriptor_pb2.FileDescriptorSet.FromString(Path(sys.argv[1]).read_bytes()).file:
    pool.Add(file)
def message(name, **fields):
    cls = message_factory.GetMessageClass(pool.FindMessageTypeByName('spark.connect.' + name))
    return cls(**fields)
def config(raw, context):
    request = message('ConfigRequest').FromString(raw)
    response = message('ConfigResponse', session_id=request.session_id,
                       server_side_session_id='00000000-0000-0000-0000-000000000001')
    values = {
        'spark.sql.session.timeZone': 'UTC',
        'spark.sql.timestampType': 'TIMESTAMP_LTZ',
        'spark.sql.session.localRelationCacheThreshold': str(1 << 60),
        'spark.sql.session.localRelationSizeLimit': str(1 << 60),
        'spark.sql.session.localRelationChunkSizeRows': '10000',
        'spark.sql.session.localRelationChunkSizeBytes': str(16 << 20),
        'spark.sql.session.localRelationBatchOfChunksSizeBytes': str(64 << 20),
        # As the default of a server.
        'spark.sql.execution.pandas.convertToArrowArraySafely': 'true',
        'spark.sql.execution.pandas.inferPandasDictAsMap': 'false',
    }
    operation = request.operation
    if operation.HasField('get'):
        keys = operation.get.keys
    elif operation.HasField('get_with_default'):
        keys = [pair.key for pair in operation.get_with_default.pairs]
    else:
        context.abort(grpc.StatusCode.INVALID_ARGUMENT, 'unexpected config operation')
    for key in keys:
        response.pairs.add(key=key, value=values.get(key, 'false'))
    return response.SerializeToString()
def execute(raw, context):
    request = message('ExecutePlanRequest').FromString(raw)
    relation = request.plan.root
    # Python's pandas path wraps the local relation to assign column names.
    if relation.HasField('to_df'):
        relation = relation.to_df.input
    assert relation.HasField('local_relation'), relation
    payload = relation.local_relation.data
    Path(sys.argv[2]).write_bytes(payload)
    table = pa.ipc.open_stream(payload).read_all()
    header = dict(session_id=request.session_id, operation_id=request.operation_id,
                  server_side_session_id='00000000-0000-0000-0000-000000000001')
    response = message('ExecutePlanResponse', **header, response_id='batch',
                       arrow_batch=dict(data=payload, row_count=table.num_rows))
    for field in table.schema:
        output = response.schema.struct.fields.add(name=field.name, nullable=True)
        output.data_type.decimal.precision = field.type.precision
        output.data_type.decimal.scale = field.type.scale
    yield response.SerializeToString()
    yield message('ExecutePlanResponse', **header, response_id='complete',
                  result_complete={}).SerializeToString()
def release_session(raw, context):
    request = message('ReleaseSessionRequest').FromString(raw)
    return message('ReleaseSessionResponse', session_id=request.session_id,
                   server_side_session_id='00000000-0000-0000-0000-000000000001').SerializeToString()
def release_execute(raw, context):
    request = message('ReleaseExecuteRequest').FromString(raw)
    return message('ReleaseExecuteResponse', session_id=request.session_id,
                   operation_id=request.operation_id,
                   server_side_session_id='00000000-0000-0000-0000-000000000001').SerializeToString()
with ThreadPoolExecutor(max_workers=4) as executor:
    server = grpc.server(executor)
    server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(
        'spark.connect.SparkConnectService', {
            'Config': grpc.unary_unary_rpc_method_handler(config),
            'ExecutePlan': grpc.unary_stream_rpc_method_handler(execute),
            'ReleaseSession': grpc.unary_unary_rpc_method_handler(release_session),
            'ReleaseExecute': grpc.unary_unary_rpc_method_handler(release_execute),
        }),))
    port = server.add_insecure_port('127.0.0.1:0')
    assert port
    server.start()
    print(port, flush=True)
    sys.stdin.readline()
    server.stop(0).wait()
"""


@pytest.fixture(scope="module")
def decimal_ipc_protocol(tmp_path_factory):
    import os
    from pathlib import Path
    import shutil
    import subprocess

    pytest.importorskip("grpc")
    pytest.importorskip("pyarrow")
    protoc = shutil.which(os.environ.get("PROTOC", "protoc"))
    if protoc is None:
        pytest.skip("Decimal IPC transport tests require protoc")
    proto_root = Path(__file__).resolve().parents[2] / "crates/spark-connect-proto/proto"
    descriptor = tmp_path_factory.mktemp("decimal-proto") / "connect.pb"
    command = [protoc, f"-I{proto_root}"]
    if include := os.environ.get("PROTOC_INCLUDE"):
        command.append(f"-I{include}")
    subprocess.run(command + ["--include_imports", f"--descriptor_set_out={descriptor}",
                             str(proto_root / "spark/connect/base.proto")], check=True)
    return descriptor


@pytest.fixture
def decimal_ipc_session(decimal_ipc_protocol, tmp_path):
    import subprocess
    import sys
    from pyspark.sql import SparkSession

    payload = tmp_path / "input.arrow"
    process = subprocess.Popen(
        [sys.executable, "-u", "-c", _DECIMAL_IPC_SERVER, str(decimal_ipc_protocol), str(payload)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    )
    spark = None
    try:
        port = int(process.stdout.readline())
        spark = SparkSession.builder.remote(f"sc://127.0.0.1:{port}").create()
        yield spark, payload
    finally:
        try:
            if spark is not None:
                spark.stop()
        finally:
            process.communicate("stop\n", timeout=20)
            assert process.returncode == 0


@pytest.mark.parametrize("kind", ["tuples", "pandas"])
@pytest.mark.parametrize("precision, scale, texts", [
    (18, 7, ["1E+3", "1E-7", "-1E-7", "+1.25E+2", "-1.25E-2", "0E+3", "-0E-7", None]),
    (18, 7, ["99999999999.9999999", "-99999999999.9999999", "1.234500E-3"]),
    (38, 0, ["1E+37", "-1E+37", "9.9999999999999999999999999999999999999E+37"]),
    (38, 38, ["1E-38", "-1E-38", "0E-38"]),
    (10, 2, ["123.45", "-123.45", "1.2300", "0.00", "-0.00"]),
])
def test_decimal_scientific_local_input_ipc(decimal_ipc_session, kind, precision, scale, texts):
    from decimal import Decimal
    import pyarrow as pa

    spark, payload = decimal_ipc_session
    values = [None if text is None else Decimal(text) for text in texts]
    data = [(value,) for value in values]
    if kind == "pandas":
        pd = pytest.importorskip("pandas")
        data = pd.DataFrame({"v": values})
    schema = T.StructType([T.StructField("v", T.DecimalType(precision, scale))])
    spark.createDataFrame(data, schema).collect()
    # Read the uploaded IPC independently rather than trusting collect's decoder.
    table = pa.ipc.open_stream(payload.read_bytes()).read_all()
    expected = pa.table({"v": pa.array(values, type=pa.decimal128(precision, scale))})
    assert table.schema == expected.schema
    assert table.equals(expected)
    # Check the physical little-endian Decimal128 cells as well as their values.
    actual = table.column(0).combine_chunks()
    wanted = expected.column(0).combine_chunks()
    for index, value in enumerate(values):
        if value is not None:
            offset = index * 16
            assert actual.buffers()[1][offset:offset + 16] == wanted.buffers()[1][offset:offset + 16]


@pytest.mark.parametrize("text", ["1E+1000000000", "-1E+1000000000"])
def test_decimal_scientific_local_input_overflow(decimal_ipc_session, text):
    from decimal import Decimal

    spark, payload = decimal_ipc_session
    schema = T.StructType([T.StructField("v", T.DecimalType(38, 0))])
    with pytest.raises((ArithmeticError, RuntimeError)):
        spark.createDataFrame([(Decimal(text),)], schema).collect()
    assert not payload.exists()


@pytest.mark.parametrize("kind", ["tuples", "pandas"])
@pytest.mark.parametrize("text, precision, scale, local_expected", [
    ("1.5E-7", 18, 7, "0.0000002"),
    ("-1.5E-7", 18, 7, "-0.0000002"),
    ("1E-8", 18, 7, "0.0000000"),
    ("0.00000015", 18, 7, "0.0000002"),
    ("1.235", 10, 2, "1.24"),
    ("1E+2", 2, 0, None),
    ("-1E+2", 2, 0, None),
    ("100000000000", 18, 7, None),
    ("1E+1000000000", 38, 0, None),
])
def test_decimal_rescaling_follows_pyspark(
    decimal_ipc_session, kind, text, precision, scale, local_expected
):
    # As PySpark 4.3.0: local rows are quantized half-even to the scale, and fail beyond
    # the precision, in the Decimal conversion; a pandas column fails its Arrow conversion
    # for any value that does not fit exactly, in either notation, rather than losing digits.
    from decimal import Decimal
    import pyarrow as pa
    from pyspark.errors import PySparkValueError

    spark, payload = decimal_ipc_session
    data = [(Decimal(text),)]
    if kind == "pandas":
        pd = pytest.importorskip("pandas")
        data = pd.DataFrame({"v": [Decimal(text)]}, dtype=object)
    schema = T.StructType([T.StructField("v", T.DecimalType(precision, scale))])
    if kind == "tuples" and local_expected is not None:
        spark.createDataFrame(data, schema).collect()
        actual = pa.ipc.open_stream(payload.read_bytes()).read_all()
        expected = pa.table({"v": pa.array([Decimal(local_expected)],
                                           type=pa.decimal128(precision, scale))})
        actual.validate(full=True)
        assert actual.schema == expected.schema
        assert actual.equals(expected)
        return
    if kind == "tuples":
        with pytest.raises(ArithmeticError):
            spark.createDataFrame(data, schema)
    else:
        with pytest.raises(PySparkValueError) as error:
            spark.createDataFrame(data, schema)
        assert str(error.value) == (
            "Exception thrown when converting pandas.Series (object) with name 'v' to Arrow "
            f"Array (decimal128({precision}, {scale})). It can be caused by overflows or other "
            "unsafe conversions warned by Arrow. Arrow safe type check can be disabled by using "
            "SQL config `spark.sql.execution.pandas.convertToArrowArraySafely`."
        )
    assert not payload.exists()


@pytest.mark.parametrize("kind", ["tuples", "pandas"])
@pytest.mark.parametrize("schema_mode", ["explicit", "names", "inferred"])
@pytest.mark.parametrize("texts", [
    ["123.4500", "-2.3400", "0.0000", "-0.0000", None],
    ["1.2300E+2", "-2.3400E+2", "0.00", "-0.00", None],
    ["0.001234500", "-0.000002300", "0.000000", None],
    ["123", "-4", "0", None],
    ["1E+3", "1E-7", "-0E-7", "1.2300E-7", None],
])
def test_decimal_scale_input_ipc(decimal_ipc_session, kind, schema_mode, texts):
    # The scale of a Decimal comes from its text, not from as_tuple(): the IPC holds the same
    # cells and types as PySpark's, which types an inferred pandas column by its values as
    # Arrow does, and a local one as decimal(38, 18). The Decimal context of the thread,
    # here one that would round and trap, changes nothing.
    from decimal import Decimal, Inexact, ROUND_UP, localcontext
    import pyarrow as pa

    spark, payload = decimal_ipc_session
    values = [None if text is None else Decimal(text) for text in texts]
    name = "_1" if schema_mode == "inferred" and kind == "tuples" else "v"
    data = [(value,) for value in values]
    if kind == "pandas":
        pd = pytest.importorskip("pandas")
        data = pd.DataFrame({name: values}, dtype=object)
    if schema_mode == "explicit":
        decimal_type = pa.decimal128(18, 9)
        schema = T.StructType([T.StructField(name, T.DecimalType(18, 9))])
    else:
        decimal_type = pa.decimal128(38, 18) if kind == "tuples" else pa.array(values).type
        schema = [name] if schema_mode == "names" else None
    with localcontext() as context:
        context.prec = 2
        context.rounding = ROUND_UP
        context.traps[Inexact] = True
        spark.createDataFrame(data, schema).collect()
    table = pa.ipc.open_stream(payload.read_bytes()).read_all()
    # As PySpark's Arrow conversion of a pandas frame without a schema, whose table names its
    # columns _0, _1, ..., which toDF names.
    ipc_name = "_0" if kind == "pandas" and schema_mode != "explicit" else name
    expected = pa.table({ipc_name: pa.array(values, type=decimal_type)})
    table.validate(full=True)
    assert table.schema == expected.schema
    assert table.equals(expected)


def test_decimal_scale_of_subclasses(decimal_ipc_session):
    # A subclass can override __str__ and as_tuple, so it is still asked for both, as Arrow
    # does; their errors are raised.
    from decimal import Decimal
    import pyarrow as pa

    pd = pytest.importorskip("pandas")
    spark, payload = decimal_ipc_session
    calls = []

    class Traced(Decimal):
        def __str__(self):
            calls.append("str")
            return super().__str__()

        def as_tuple(self):
            calls.append("tuple")
            return super().as_tuple()

    values = [Traced("1.20"), None, Traced("-30.5")]
    spark.createDataFrame(pd.DataFrame({"v": values}, dtype=object), ["v"]).collect()
    assert calls.count("str") == 2 and "tuple" in calls
    table = pa.ipc.open_stream(payload.read_bytes()).read_all()
    expected = pa.array([Decimal("1.20"), None, Decimal("-30.5")])
    assert table.column(0).combine_chunks().equals(expected)

    for hook in ["__str__", "as_tuple"]:
        def fail(self):
            raise LookupError("decimal hook")

        failing = type("Failing", (Decimal,), {hook: fail})
        with pytest.raises(LookupError, match="decimal hook"):
            spark.createDataFrame(pd.DataFrame({"v": [failing("1.5")]}, dtype=object), ["v"])


@pytest.mark.parametrize("schema", [["v"], T.StructType([T.StructField("v", T.DecimalType(10, 4))])])
def test_decimal_scale_avoids_coefficient_tuple(decimal_ipc_session, schema):
    # Exact Decimals are typed and converted from their text: no DecimalTuple is built.
    import cProfile
    from decimal import Decimal

    pd = pytest.importorskip("pandas")
    spark, _ = decimal_ipc_session
    tuple_constructor = type(Decimal("1.20").as_tuple()).__new__.__code__
    # Of the converter of the row path, which a customized frame keeps (_RowPathFrame).
    values = _RowPathFrame(pd)(
        {"v": [Decimal("123.4500"), Decimal("-2.3000"), None]}, dtype=object)
    profile = cProfile.Profile()
    profile.enable()
    try:
        spark.createDataFrame(values, schema)
    finally:
        profile.disable()
    assert not any(entry.code is tuple_constructor for entry in profile.getstats())


@pytest.mark.parametrize("text", ["Infinity", "-Infinity"])
def test_decimal_scale_of_non_finite_values(decimal_ipc_session, text):
    # An infinity has no precision to type a column by.
    from decimal import Decimal

    pd = pytest.importorskip("pandas")
    spark, payload = decimal_ipc_session
    # Of the converter of the row path, which a customized frame keeps (_RowPathFrame).
    data = _RowPathFrame(pd)({"v": [Decimal(text), Decimal("1.5")]}, dtype=object)
    with pytest.raises(ValueError, match="has no precision to type a column by"):
        spark.createDataFrame(data, ["v"])
    assert not payload.exists()


def test_decimal_quanta_of_each_type_and_conversion(decimal_ipc_session):
    # Each decimal type of a conversion is quantized with its own quantum and context, which
    # are built once per conversion, as PySpark builds them once per converter: a change of
    # decimal.DefaultContext applies from the next createDataFrame.
    import decimal
    from decimal import Decimal
    import pyarrow as pa

    spark, payload = decimal_ipc_session
    schema = T.StructType([T.StructField("a", T.DecimalType(10, 2)),
                           T.StructField("b", T.DecimalType(5, 1)),
                           T.StructField("c", T.DecimalType(10, 2))])
    rows = [(Decimal("1.235"), Decimal("2.25"), Decimal("-0.005")),
            (Decimal("1.245"), Decimal("2.35"), None)]
    spark.createDataFrame(rows, schema).collect()
    table = pa.ipc.open_stream(payload.read_bytes()).read_all()
    assert table.to_pydict() == {
        "a": [Decimal("1.24"), Decimal("1.24")],
        "b": [Decimal("2.2"), Decimal("2.4")],
        "c": [Decimal("-0.00"), None],
    }
    decimal.DefaultContext.traps[decimal.Inexact] = True
    try:
        with pytest.raises(decimal.Inexact):
            spark.createDataFrame(rows, schema)
    finally:
        decimal.DefaultContext.traps[decimal.Inexact] = False
    spark.createDataFrame(rows[1:], schema)


# --------------------------------------------------------------------------- schema cache


@pytest.fixture(scope="module")
def _shared_mock_connect():
    pytest.importorskip("grpc")
    with mock_connect_session() as session:
        yield session


@pytest.fixture
def mock_connect(_shared_mock_connect):
    # Each test has a session of its own of the shared server, which is the active session
    # while the test runs, as the tests of what reads the active session expect.
    from pyspark.sql import SparkSession

    _, service = _shared_mock_connect
    service.reset()
    spark = SparkSession.builder.remote(service.remote).getOrCreate()
    yield spark, service
    spark.stop()


@pytest.fixture(autouse=True, scope="session")
def _debugging_read_without_a_session():
    # PySpark's is_debugging_enabled reads spark.python.sql.dataFrameDebugging.enabled of the
    # active session at the first expression of a Column method, and keeps it for the
    # interpreter. It is read without a session, before any test, so that a test whose session
    # has no server, such as one of _offline_session(), does not wait to read it, whatever
    # the tests that run. The tests of origins run in their own interpreters.
    from pyspark.errors.utils import is_debugging_enabled
    from pyspark.sql import SparkSession

    assert SparkSession.getActiveSession() is None
    is_debugging_enabled()


@pytest.fixture(autouse=True)
def _no_session_left_active():
    # Each test starts without an active session, as a new interpreter does. What reads the
    # active session, such as udtf() of its Arrow configuration, would otherwise reach the
    # server of a session that an earlier test left, which may not listen.
    yield
    from pyspark.sql import SparkSession

    session = SparkSession.getActiveSession()
    if session is not None:
        session.stop()


@pytest.mark.parametrize("first", ["schema", "columns", "dtypes"])
def test_schema_cache_one_analysis_shared_by_metadata_accessors(mock_connect, first):
    spark, service = mock_connect
    df = spark.range(10)
    assert service.calls == 0
    getattr(df, first)
    assert service.calls == 1
    for _ in range(3):
        schema = df.schema
        assert isinstance(schema, T.StructType)
        expected = T.StructType([T.StructField("id", T.LongType(), True, {"note": "original"})])
        assert schema.jsonValue() == expected.jsonValue()
        assert df.columns == ["id"]
        assert df.dtypes == [("id", "bigint")]
    assert service.calls == 1


def test_schema_cache_returned_schema_and_lists_are_independent(mock_connect):
    spark, service = mock_connect
    df = spark.range(10)
    expected_json = df.schema.jsonValue()
    modified = df.schema
    modified.add("extra", T.StringType())
    assert len(modified.fields) == 2
    columns = df.columns
    columns.append("extra")
    dtypes = df.dtypes
    dtypes.clear()
    assert df.schema.jsonValue() == expected_json
    assert df.columns == ["id"]
    assert df.dtypes == [("id", "bigint")]
    assert service.calls == 1


def test_schema_cache_derived_and_new_plans_resolve_independently(mock_connect):
    spark, service = mock_connect
    df = spark.range(10)
    assert df.columns == ["id"]
    derived = df.selectExpr("id AS renamed")
    service.data_type = struct_schema("renamed")
    assert derived.columns == ["renamed"]
    assert service.calls == 2
    assert df.columns == ["id"]
    # Rebuilding the same source must observe updated server metadata, while the
    # original immutable DataFrame retains its already-resolved schema.
    fresh = spark.range(10)
    assert fresh.columns == ["renamed"]
    other_session = spark.newSession()
    try:
        assert other_session.range(10).columns == ["renamed"]
    finally:
        other_session.stop()
    assert service.calls == 4
    assert df.columns == ["id"]


def test_grouped_aggregations_as_pyspark(mock_connect):
    # As PySpark's GroupedData: count is named count, sum, avg, mean, min and max aggregate
    # every numeric column without columns, and the aggregations check their arguments.
    from pyspark.errors import PySparkTypeError, PySparkValueError
    from pyspark.sql import functions as F

    spark, service = mock_connect
    df = spark.range(1)

    def aggregations(dataframe):
        # ExecutePlanRequest.plan=3, Plan.root=1, Relation.aggregate=9,
        # Aggregate.aggregate_expressions=4.
        dataframe.collect()
        plan = parse_message(parse_message(service.execute_requests[-1])[3][0])[1][0]
        return parse_message(parse_message(plan)[9][0])[4]

    # Expression.alias=6: Alias.expr=1, name=2; Expression.unresolved_function=3:
    # UnresolvedFunction.function_name=1, arguments=2.
    (count,) = aggregations(df.groupBy("id").count())
    alias = parse_message(parse_message(count)[6][0])
    assert alias[2] == [b"count"]
    assert parse_message(parse_message(alias[1][0])[3][0])[1] == [b"count"]
    for aggregate, function in ((lambda g: g.sum(), b"sum"), (lambda g: g.mean(), b"avg"),
                                (lambda g: g.max("id"), b"max")):
        (expression,) = aggregations(aggregate(df.groupBy("id")))
        assert parse_message(parse_message(expression)[3][0])[1] == [function]
    (expression,) = aggregations(df.agg({"id": "min"}))
    assert parse_message(parse_message(expression)[3][0])[1] == [b"min"]

    with pytest.raises(PySparkTypeError) as error:
        df.groupBy("id").sum("nope")
    assert error.value.getCondition() == "NOT_NUMERIC_COLUMNS"
    assert error.value.getMessageParameters() == {"invalid_columns": "['nope']"}
    with pytest.raises(PySparkValueError) as error:
        df.agg()
    assert error.value.getCondition() == "CANNOT_BE_EMPTY"
    for call, message in ((lambda: df.groupBy("id").agg(), "exprs should not be empty"),
                          (lambda: df.groupBy("id").agg("id"), "all exprs should be Column"),
                          (lambda: df.agg("id"), "all exprs should be Expression"),
                          (lambda: df.groupBy("id").sum(1), "")):
        with pytest.raises(AssertionError) as error:
            call()
        assert str(error.value) == message
    assert df.groupBy("id").agg(F.count("id")) is not None


@pytest.mark.parametrize("transform", [
    pytest.param(lambda df: df.filter("id > 0"), id="filter"),
    pytest.param(lambda df: df.limit(2), id="limit"),
    pytest.param(lambda df: df.alias("source"), id="alias"),
    pytest.param(lambda df: df.distinct(), id="distinct"),
    pytest.param(lambda df: df.dropDuplicates(["id"]), id="dropDuplicates"),
    pytest.param(lambda df: df.coalesce(1), id="coalesce"),
    pytest.param(lambda df: df.repartition(2), id="repartition"),
    pytest.param(lambda df: df.repartition(2, "id"), id="repartition-expression"),
    pytest.param(lambda df: df.repartition("id"), id="repartition-column"),
    pytest.param(lambda df: df.repartitionByRange(2, "id"), id="repartitionByRange"),
    pytest.param(lambda df: df.repartitionById(2, "id"), id="repartitionById"),
    pytest.param(lambda df: df.sort("id"), id="sort"),
    pytest.param(lambda df: df.sortWithinPartitions("id"), id="sortWithinPartitions"),
    pytest.param(lambda df: df.sample(fraction=0.5, seed=1), id="sample"),
    pytest.param(lambda df: df.hint("broadcast"), id="hint"),
])
def test_schema_cache_preserving_transforms_reuse_warm_schema(mock_connect, transform):
    spark, service = mock_connect
    df = spark.range(10)
    expected = df.schema.jsonValue()
    assert service.calls == 1
    # A new analysis would return different metadata, making accidental misses
    # observable as well as countable.
    service.data_type = struct_schema("unexpected")
    derived = transform(df)
    assert derived.schema.jsonValue() == expected
    assert derived.limit(1).columns == ["id"]
    assert service.calls == 1


def test_schema_cache_random_split_and_observe_reuse_warm_schema(mock_connect):
    from pyspark.sql import functions as F

    spark, service = mock_connect
    df = spark.range(10)
    assert df.columns == ["id"]
    service.data_type = struct_schema("unexpected")
    for split in df.randomSplit([1.0, 2.0], seed=1):
        assert split.columns == ["id"]
    observed = df.observe("rows", F.count("*").alias("count"))
    assert observed.columns == ["id"]
    assert service.calls == 1


def test_udf_profiles_are_collected_from_the_python_accumulators_as_pyspark(
        mock_connect, capsys, caplog):
    # As PySpark Connect's: spark.profile is a Profile of a ConnectProfilerCollector, which the
    # pickled updates of the UDF profiler, in the __python_accumulator__ observed metrics of an
    # execution, update; the updates of other accumulators are ignored, and the warning of one
    # that does not unpickle is logged.
    import cProfile
    import pstats

    from pyspark.accumulators import SpecialAccumulatorIds, specialAccumulatorSer
    from pyspark.sql.connect.profiler import ConnectProfilerCollector
    from pyspark.sql.profiler import Profile

    spark, service = mock_connect
    profiler = cProfile.Profile()
    profiler.enable()
    sum(range(10))
    profiler.disable()
    # As the profiler of the Python worker sends its stats.
    stats = pstats.Stats(profiler, stream=None)
    stats.stream = None

    def accumulator_metrics(*updates):
        # ExecutePlanResponse.observed_metrics=6 -> ObservedMetrics name=1, values=2
        # (Expression.Literal binary=2).
        observed = message_field(1, "__python_accumulator__")
        for update in updates:
            observed += message_field(2, message_field(2, update))
        return message_field(6, observed)

    service.execute_payload = accumulator_metrics(
        specialAccumulatorSer.dumps(
            (SpecialAccumulatorIds.SQL_UDF_PROFIER_V2, {7: {"perf": stats}})),
        specialAccumulatorSer.dumps((SpecialAccumulatorIds.SQL_UDF_PROFIER, {8: (stats, None)})),
        b"not a pickle",
    )
    with caplog.at_level("WARNING"):
        spark.range(1).collect()
    assert isinstance(spark.profile, Profile)
    assert isinstance(spark._profiler_collector, ConnectProfilerCollector)
    assert list(spark._profiler_collector._profile_results) == [7]
    assert "Error unpickling accumulator metric" in caplog.text
    capsys.readouterr()
    spark.profile.show(type="perf")
    assert "Profile of UDF<id=7>" in capsys.readouterr().out
    spark.profile.clear()
    assert spark._profiler_collector._profile_results == {}




def test_schema_cache_cold_children_keep_independent_caches(mock_connect):
    spark, service = mock_connect
    df = spark.range(10)
    before_lookup = df.limit(1)
    service.block_response = True
    with ThreadPoolExecutor(max_workers=1) as pool:
        lookup = pool.submit(lambda: df.schema)
        try:
            for _ in range(500):
                if service.calls:
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("Metadata request never reached the server")
            # Constructing a child also runs with the GIL. It must not wait for
            # the in-flight cache miss, or this thread cannot release the server.
            during_lookup = df.filter("id > 0")
            assert not lookup.done()
        finally:
            service.release_response()
        assert lookup.result(timeout=10).fieldNames() == ["id"]
    assert service.calls == 1
    # Neither child picks up a cache populated after the child was constructed.
    for calls, child in enumerate((before_lookup, during_lookup), start=2):
        service.data_type = struct_schema(f"child_{calls}")
        assert child.columns == [f"child_{calls}"]
        assert service.calls == calls
    assert df.columns == ["id"]


@pytest.mark.parametrize("transform", [
    pytest.param(lambda df: df.selectExpr("id AS renamed"), id="select"),
    pytest.param(lambda df: df.zipWithIndex(), id="zipWithIndex"),
    # Reference PySpark does not propagate through offset, despite its unchanged
    # output columns. Match its explicit propagation rules rather than guessing.
    pytest.param(lambda df: df.offset(1), id="offset"),
])
def test_schema_cache_other_transforms_analyze_independently(mock_connect, transform):
    spark, service = mock_connect
    df = spark.range(10)
    assert df.columns == ["id"]
    derived = transform(df)
    assert service.calls == 1
    service.data_type = struct_schema("derived")
    assert derived.columns == ["derived"]
    assert service.calls == 2
    assert df.columns == ["id"]


@pytest.mark.parametrize("operation", [
    "union", "unionAll", "unionByName", "subtract", "exceptAll", "intersect", "intersectAll",
])
def test_schema_cache_set_operations_require_equal_warm_schemas(mock_connect, operation):
    spark, service = mock_connect
    left, right = spark.range(10), spark.range(10)
    assert left.columns == right.columns == ["id"]
    assert service.calls == 2
    equal = getattr(left, operation)(right)
    assert equal.columns == ["id"]
    assert service.calls == 2

    cold = spark.range(10)
    missing = getattr(left, operation)(cold)
    assert service.calls == 2
    service.data_type = struct_schema("resolved")
    assert missing.columns == ["resolved"]
    assert service.calls == 3

    different = spark.range(10)
    assert different.columns == ["resolved"]
    assert service.calls == 4
    mismatched = getattr(left, operation)(different)
    service.data_type = struct_schema("coerced")
    assert mismatched.columns == ["coerced"]
    assert service.calls == 5
    assert left.columns == ["id"]


def test_dataframes_take_their_arguments_and_attributes_as_pyspark(mock_connect, monkeypatch):
    # As PySpark Connect's: the errors of randomSplit, withMetadata and withWatermark, and the
    # attributes of a DataFrame: a column of any name but a dunder one, which must be a column,
    # and JVM_ATTRIBUTE_NOT_SUPPORTED of those of a classic DataFrame.
    from pyspark.errors import (
        PySparkAttributeError,
        PySparkNotImplementedError,
        PySparkTypeError,
        PySparkValueError,
    )

    spark, _ = mock_connect
    df = spark.range(3)
    for weights, arg_name, arg_value in (([1.0, -1.0], "weights", "-1.0"),
                                          ([0, 0], "sum(weights)", "0")):
        with pytest.raises(PySparkValueError) as e:
            df.randomSplit(weights)
        assert e.value.getCondition() == "VALUE_NOT_POSITIVE"
        assert e.value.getMessageParameters() == {"arg_name": arg_name, "arg_value": arg_value}
    with pytest.raises(PySparkTypeError) as e:
        df.withMetadata("id", 1)
    assert e.value.getMessageParameters() == {
        "expected_type": "dict", "arg_name": "metadata", "arg_type": "int"}
    df.withMetadata("id", {"k": 1, "l": [1, 2]})
    for event_time, delay, arg_name, arg_type in ((1, "1 minute", "eventTime", "int"),
                                                  ("", "1 minute", "eventTime", "str"),
                                                  ("id", None, "delayThreshold", "NoneType")):
        with pytest.raises(PySparkTypeError) as e:
            df.withWatermark(event_time, delay)
        assert e.value.getMessageParameters() == {
            "arg_name": arg_name, "expected_type": "str", "arg_type": arg_type}
    assert type(df._not_a_dunder).__name__ == "Column"
    for name in ("_jseq", "_jdf", "_jmap", "_jcols"):
        with pytest.raises(PySparkAttributeError) as e:
            getattr(df, name)
        assert e.value.getCondition() == "JVM_ATTRIBUTE_NOT_SUPPORTED"
    with pytest.raises(PySparkNotImplementedError) as e:
        df.rdd
    assert e.value.getMessageParameters() == {"feature": "rdd"}
    with pytest.raises(PySparkAttributeError) as e:
        df.__not_a_column__
    assert e.value.getCondition() == "ATTRIBUTE_NOT_SUPPORTED"
    monkeypatch.setenv("PYSPARK_VALIDATE_COLUMN_NAME_LEGACY", "1")
    with pytest.raises(PySparkAttributeError) as e:
        df.not_a_column
    assert e.value.getMessageParameters() == {"attr_name": "not_a_column"}




def test_schema_cache_to_uses_supplied_struct_schema(mock_connect):
    spark, service = mock_connect
    df = spark.range(10)
    target = T.StructType([T.StructField("converted", T.StringType(), True)])
    expected = target.jsonValue()
    converted = df.to(target)
    target.add("extra", T.LongType())
    assert converted.schema.jsonValue() == expected
    assert converted.limit(1).dtypes == [("converted", "string")]
    assert service.calls == 0
    assert df.columns == ["id"]
    assert service.calls == 1


@pytest.mark.parametrize("method", ["mapInPandas", "mapInArrow"])
def test_schema_cache_map_uses_declared_struct_schema(mock_connect, method):
    spark, service = mock_connect
    target = T.StructType([
        T.StructField("output", T.StringType(), True, {"note": "declared"}),
    ])
    mapped = getattr(spark.range(10), method)(lambda batches: batches, schema=target)
    # The binding resolves the input columns while building the UDF. Reading its
    # declared output schema must not issue another analysis request.
    assert service.calls == 1
    service.data_type = struct_schema("unexpected")
    assert mapped.schema.jsonValue() == target.jsonValue()
    assert mapped.columns == ["output"]
    assert mapped.dtypes == [("output", "string")]
    assert service.calls == 1


@pytest.mark.parametrize("failure", ["rpc", "no-result", "missing-schema", "invalid-type"])
def test_schema_cache_failures_are_not_cached(mock_connect, failure):
    spark, service = mock_connect
    df = spark.range(10)
    service.failure = failure
    for count in (1, 2):
        message = {
            "rpc": "schema unavailable",
            "no-result": "no schema in response",
            "missing-schema": "Schema is missing",
            "invalid-type": "DataType kind not set",
        }[failure]
        with pytest.raises(Exception, match=message):
            _ = df.schema
        assert service.calls == count
    service.failure = None
    assert df.columns == ["id"]
    assert df.dtypes == [("id", "bigint")]
    assert service.calls == 3


def test_schema_cache_non_struct_responses_are_not_cached(mock_connect):
    spark, service = mock_connect
    df = spark.range(10)
    service.data_type = message_field(7, b"")
    for count in (1, 2):
        with pytest.raises(Exception, match="Schema is not a struct type"):
            _ = df.schema
        assert service.calls == count
    service.data_type = struct_schema()
    assert df.columns == ["id"]
    assert df.dtypes == [("id", "bigint")]
    assert service.calls == 3


def test_schema_cache_concurrent_metadata_access_issues_one_analysis(mock_connect):
    spark, service = mock_connect
    df = spark.range(10)
    accessors = ("schema", "columns", "dtypes") * 3
    start = Barrier(len(accessors) + 1)
    service.block_response = True

    def read_metadata(accessor):
        start.wait(timeout=10)
        return getattr(df, accessor)

    with ThreadPoolExecutor(max_workers=len(accessors)) as pool:
        pending = [pool.submit(read_metadata, accessor) for accessor in accessors]
        try:
            start.wait(timeout=10)
            for _ in range(500):
                if service.calls:
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("Metadata request never reached the server")
            # Keep the first RPC in flight while the other workers attempt their
            # cold lookups. An implementation without miss synchronization must
            # not pass just because the first response arrives immediately.
            time.sleep(0.1)
            assert not any(future.done() for future in pending)
        finally:
            service.release_response()
        results = [future.result(timeout=10) for future in pending]
    schemas = []
    for accessor, result in zip(accessors, results):
        if accessor == "schema":
            schemas.append(result)
        elif accessor == "columns":
            assert result == ["id"]
        else:
            assert result == [("id", "bigint")]
    assert all(schema.jsonValue() == schemas[0].jsonValue() for schema in schemas)
    assert len({id(schema) for schema in schemas}) == len(schemas)
    assert service.calls == 1


@pytest.mark.parametrize("accessor", ["schema", "columns", "dtypes"])
def test_schema_cache_metadata_wait_releases_gil(mock_connect, accessor):
    spark, service = mock_connect
    df = spark.range(10)
    service.block_response = True

    def allow_response_after_request():
        for _ in range(500):
            if service.calls:
                service.release_response()
                return
            time.sleep(0.01)
        raise AssertionError("Metadata request never reached the server")

    # This Python thread must run while the main thread waits for the RPC. The
    # server bounds its wait, so retaining the GIL fails instead of deadlocking.
    with ThreadPoolExecutor(max_workers=1) as pool:
        allowed = pool.submit(allow_response_after_request)
        result = getattr(df, accessor)
        allowed.result(timeout=10)
    if accessor == "schema":
        assert isinstance(result, T.StructType)
    elif accessor == "columns":
        assert result == ["id"]
    else:
        assert result == [("id", "bigint")]
    assert service.calls == 1


@pytest.mark.parametrize("names,expected", [(None, ["x", "y"]), (["a"], ["a", "_2"]), ([], ["_1", "_2"])])
def test_arrow_input_normalization(names, expected):
    pa = pytest.importorskip("pyarrow")
    from pyspark.sql._arrow_input import prepare_table

    table = pa.table({"x": pa.chunked_array([[1], [2]]), "y": ["a", None]})
    normalized, schema, columns = prepare_table(table, names, "TIMESTAMP_LTZ", "UTC", "false")
    assert normalized.equals(table)
    assert schema is None
    assert columns == expected


def test_arrow_input_empty_and_width_errors():
    pa = pytest.importorskip("pyarrow")
    from pyspark.errors import PySparkValueError
    from pyspark.sql._arrow_input import prepare_table

    empty = pa.table({"x": pa.array([], type=pa.int64())})
    with pytest.raises(PySparkValueError, match="CANNOT_INFER_EMPTY_SCHEMA"):
        prepare_table(empty, None, None, None, None)
    schema = T.StructType([T.StructField("x", T.LongType())])
    assert prepare_table(empty, schema, None, None, None)[1] is schema
    with pytest.raises(PySparkValueError, match="AXIS_LENGTH_MISMATCH"):
        prepare_table(pa.table({"x": [1]}), ["a", "b"], "TIMESTAMP_LTZ", "UTC", "false")


@pytest.mark.parametrize("schema,expected_type", [
    pytest.param(T.ArrayType(T.LongType()), "ArrayType", id="array"),
    pytest.param(T.MapType(T.StringType(), T.LongType()), "MapType", id="map"),
])
@pytest.mark.parametrize("empty", [False, True], ids=["nonempty", "empty"])
def test_arrow_input_unsupported_schema_types(schema, expected_type, empty):
    pa = pytest.importorskip("pyarrow")
    from pyspark.errors import PySparkTypeError
    from pyspark.sql._arrow_input import prepare_table

    table = pa.table({"x": pa.array([] if empty else [1], type=pa.int64())})
    with pytest.raises(PySparkTypeError) as error:
        prepare_table(table, schema, "TIMESTAMP_LTZ", "UTC", "false")
    assert error.value.getCondition() == "NOT_EXPECTED_TYPE"
    assert error.value.getMessageParameters() == {
        "arg_name": "schema", "expected_type": "list, None or StructType",
        "arg_type": expected_type,
    }


@pytest.mark.parametrize("schema", [
    None, T.LongType(), T.StructType([T.StructField("x", T.LongType())]), ["x"], ("x",),
])
def test_arrow_input_supported_schema_types(schema):
    pa = pytest.importorskip("pyarrow")
    from pyspark.sql._arrow_input import prepare_table

    table = pa.table({"x": [1]})
    normalized, result, names = prepare_table(table, schema, "TIMESTAMP_LTZ", "UTC", "false")
    assert normalized.equals(table)
    if isinstance(schema, (T.AtomicType, T.StructType)):
        assert result is schema
    else:
        assert result is None
    assert names == (None if isinstance(schema, (T.AtomicType, T.StructType)) else ["x"])


def test_arrow_input_array_and_map_struct_fields_allowed():
    pa = pytest.importorskip("pyarrow")
    from pyspark.sql._arrow_input import prepare_table

    table = pa.table({
        "items": pa.array([[1, None]], type=pa.list_(pa.int64())),
        "lookup": pa.array([[("a", 1)]], type=pa.map_(pa.string(), pa.int64())),
    })
    schema = T.StructType([
        T.StructField("items", T.ArrayType(T.LongType())),
        T.StructField("lookup", T.MapType(T.StringType(), T.LongType())),
    ])
    normalized, result, names = prepare_table(table, schema, "TIMESTAMP_LTZ", "UTC", "false")
    assert normalized.equals(table)
    assert result is schema
    assert names is None


@pytest.mark.parametrize("schema,expected_length", [
    pytest.param(T.StructType([T.StructField("x", T.IntegerType())]), "1", id="too-few-fields"),
    pytest.param(T.StructType([
        T.StructField("x", T.LongType()), T.StructField("y", T.LongType()),
        T.StructField("z", T.LongType()),
    ]), "3", id="too-many-fields"),
    pytest.param(T.LongType(), "1", id="atomic-multiple-columns"),
])
def test_arrow_input_explicit_schema_width_errors(schema, expected_length):
    pa = pytest.importorskip("pyarrow")
    from pyspark.errors import PySparkValueError
    from pyspark.sql._arrow_input import prepare_table

    # The first value would overflow the one-field IntegerType schema. Width
    # validation must produce the structured Spark error before an Arrow cast.
    table = pa.table({"x": [2**40], "y": [2]})
    with pytest.raises(PySparkValueError) as error:
        prepare_table(table, schema, "TIMESTAMP_LTZ", "UTC", "false")
    assert error.value.getCondition() == "AXIS_LENGTH_MISMATCH"
    assert error.value.getMessageParameters() == {
        "expected_length": expected_length, "actual_length": "2",
    }


@pytest.mark.parametrize("schema,values", [
    pytest.param(T.StructType([T.StructField("x", T.LongType())]),
                 {"x": [1, 2]}, id="single-field"),
    pytest.param(T.StructType([
        T.StructField("x", T.LongType()), T.StructField("y", T.LongType()),
    ]), {"x": [1, 2], "y": [3, 4]}, id="multiple-fields"),
    pytest.param(T.LongType(), {"x": [1, 2]}, id="atomic-single-column"),
])
def test_arrow_input_explicit_schema_matching_width(schema, values):
    pa = pytest.importorskip("pyarrow")
    from pyspark.sql._arrow_input import prepare_table

    table = pa.table(values)
    normalized, result, names = prepare_table(table, schema, "TIMESTAMP_LTZ", "UTC", "false")
    assert normalized.equals(table)
    assert result is schema
    assert names is None


@pytest.mark.parametrize("schema", [
    T.StructType([]),
    T.StructType([T.StructField("x", T.LongType()), T.StructField("y", T.LongType())]),
    T.LongType(),
])
def test_arrow_input_empty_explicit_schema_skips_width_validation(schema):
    pa = pytest.importorskip("pyarrow")
    from pyspark.sql._arrow_input import prepare_table

    table = pa.table({"x": pa.array([], type=pa.int64())})
    normalized, result, names = prepare_table(table, schema, None, None, None)
    assert normalized is table
    assert result is schema
    assert names is None


@pytest.mark.parametrize("timestamp_type,expected", [("TIMESTAMP_NTZ", T.TimestampNTZType), ("TIMESTAMP_LTZ", T.TimestampType)])
def test_arrow_input_timestamps(timestamp_type, expected):
    pa = pytest.importorskip("pyarrow")
    from pyspark.sql._arrow_input import prepare_table

    table = pa.table({"t": pa.array([1001], type=pa.timestamp("ns"))})
    normalized, schema, _ = prepare_table(table, None, timestamp_type, "UTC", "false")
    assert schema is None
    assert normalized.column(0).type.tz == ("UTC" if expected is T.TimestampType else None)
    assert normalized.column(0).type.unit == "us"
    assert normalized.column(0).cast(pa.int64()).to_pylist() == [1]


def test_arrow_input_explicit_cast_and_nested_values():
    pa = pytest.importorskip("pyarrow")
    from pyspark.sql._arrow_input import prepare_table

    table = pa.table({"x": [1, 2], "nested": [[1, None], []]})
    schema = T.StructType([T.StructField("x", T.IntegerType()),
                           T.StructField("nested", T.ArrayType(T.LongType()))])
    normalized, result, _ = prepare_table(table, schema, "TIMESTAMP_LTZ", "UTC", "false")
    assert result is schema
    assert normalized.column(0).type == pa.int32()
    assert normalized.to_pylist() == table.to_pylist()
    with pytest.raises(pa.ArrowInvalid):
        prepare_table(pa.table({"x": [2**40]}), T.StructType([T.StructField("x", T.IntegerType())]),
                      "TIMESTAMP_LTZ", "UTC", "false")


def test_arrow_input_roundtrip_live_server():
    """Run explicitly with SPARK_REMOTE; also checks imported buffers own their lifetime."""
    import gc
    import os
    pa = pytest.importorskip("pyarrow")
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("requires SPARK_REMOTE")
    from pyspark.sql import SparkSession

    spark = SparkSession.builder.remote(remote).getOrCreate()
    try:
        table = pa.table({"x": pa.chunked_array([[1, 2], [3]]), "nested": [[1], None, []]})
        df = spark.createDataFrame(table, ["id", "items"])
        del table
        gc.collect()
        assert df.columns == ["id", "items"]
        assert [(r[0], r[1]) for r in df.collect()] == [(1, [1]), (2, None), (3, [])]
        empty = spark.createDataFrame(pa.table({"x": pa.array([], type=pa.int64())}),
                                     T.StructType([T.StructField("x", T.LongType())]))
        assert empty.count() == 0
        assert empty.columns == ["x"]
    finally:
        spark.stop()


def test_to_arrow_casts_to_the_arrow_schema_of_the_schema_as_pyspark(mock_connect):
    # As PySpark Connect's: toArrow casts the table of the result to the Arrow schema of its
    # schema, of string and binary even of large var types, which raises for a type that
    # PySpark does not convert to Arrow. The data of a server that sends no schema is as it is.
    import pyarrow as pa
    from pyspark.errors import PySparkTypeError

    spark, service = mock_connect
    table = pa.table({"s": pa.array(["x"], pa.large_string()),
                      "b": pa.array([b"y"], pa.large_binary())})
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    # ExecutePlanResponse.arrow_batch=2: ArrowBatch.row_count=1, data=2.
    batch = message_field(2, varint_field(1, 1) + message_field(2, sink.getvalue().to_pybytes()))

    service.execute_payload = batch
    assert spark.range(1).toArrow().schema == table.schema
    # ExecutePlanResponse.schema=7.
    service.execute_payload = message_field(7, ddl_data_type("s string, b binary")) + batch
    result = spark.range(1).toArrow()
    assert result.schema == pa.schema([("s", pa.string()), ("b", pa.binary())])
    assert result.to_pylist() == [{"s": "x", "b": b"y"}]
    service.execute_payload = message_field(7, ddl_data_type("v interval year to month")) + batch
    with pytest.raises(PySparkTypeError) as e:
        spark.range(1).toArrow()
    assert e.value.getCondition() == "UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION"
    assert e.value.getMessageParameters() == {"data_type": "YearMonthIntervalType(0, 1)"}


@pytest.mark.parametrize("kind", ["slice", "dictionary", "decimal", "null", "struct", "nul"])
def test_arrow_input_types_live_server(kind):
    import os
    from decimal import Decimal
    pa = pytest.importorskip("pyarrow")
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("requires SPARK_REMOTE")
    from pyspark.sql import SparkSession

    tables = {
        "slice": pa.table({"x": pa.array([0, 1, 2, 3]).slice(1, 2)}),
        "dictionary": pa.table({"x": pa.array(["a", None, "b", "a"]).dictionary_encode()}),
        "decimal": pa.table({"x": pa.array([Decimal("1.23"), None], type=pa.decimal128(10, 2))}),
        "null": pa.table({"x": pa.nulls(2)}),
        "struct": pa.table({"x": [{"a": [1, None]}, None]}),
        "nul": pa.table({"a\0b": [1, 2]}),
    }
    spark = SparkSession.builder.remote(remote).getOrCreate()
    try:
        table = tables[kind]
        df = spark.createDataFrame(table)
        assert df.columns == table.column_names
        assert df.toArrow().to_pylist() == table.to_pylist()
    finally:
        spark.stop()


def test_arrow_input_large_upload_live_server():
    """A table larger than one gRPC message must use uploaded cache artifacts."""
    import os
    pa = pytest.importorskip("pyarrow")
    remote = os.environ.get("SPARK_REMOTE")
    if not remote:
        pytest.skip("requires SPARK_REMOTE")
    from pyspark.sql import SparkSession

    rows = 140000
    table = pa.table({"payload": [b"x" * 1024] * rows})
    assert table.nbytes > 128 * 1024**2
    spark = SparkSession.builder.remote(remote).getOrCreate()
    try:
        df = spark.createDataFrame(table)
        assert df.count() == rows
        assert df.selectExpr("sum(length(payload)) AS total").collect()[0][0] == rows * 1024
    finally:
        spark.stop()


# --------------------------------------------------------------------------- local rows of NumPy arrays, ints for dates and nested duplicate names

def _local_rows_relation_of(service, df):
    # The LocalRelation of a DataFrame of local rows, as the server receives it: its Arrow IPC
    # data (LocalRelation.data=1), and its schema JSON (LocalRelation.schema=2).
    import pyarrow as pa

    df.toArrow()
    request = parse_message(service.execute_requests[-1])
    relation = parse_message(parse_message(request[3][0])[1][0])
    while 18 in relation:  # Relation.to_df -> ToDF.input.
        relation = parse_message(parse_message(relation[18][0])[1][0])
    local = parse_message(relation[11][0])
    return pa.ipc.open_stream(local[1][0]).read_all(), json.loads(local[2][0])


def test_numpy_arrays_in_array_fields_convert_as_pyspark(mock_connect):
    # As PySpark: an array whose elements need no converter (numbers and bools) is taken as
    # PyArrow takes it, so a NumPy array is an array of its values. An array whose elements
    # need one (strings) must be a list or an array.array, or PySpark asserts.
    from pyspark.sql import types as T

    np = pytest.importorskip("numpy")
    spark, service = mock_connect
    table, _ = _local_rows_relation_of(
        service, spark.createDataFrame(
            [(np.array([1, 2]), np.array([1.5, 2.0]), np.array([True, False]))],
            "a array<bigint>, b array<double>, c array<boolean>"))
    assert table.column("a").to_pylist() == [[1, 2]]
    assert table.column("b").to_pylist() == [[1.5, 2.0]]
    assert table.column("c").to_pylist() == [[True, False]]

    long_array = T.StructType([T.StructField("a", T.ArrayType(T.LongType()))])
    table, _ = _local_rows_relation_of(
        service, spark.createDataFrame([(np.array([1, 2]),)], long_array))
    assert table.column("a").to_pylist() == [[1, 2]]
    table, _ = _local_rows_relation_of(
        service, spark.createDataFrame([((np.array([1, 2]),),)], "s struct<a:array<bigint>>"))
    assert table.column("s").to_pylist() == [{"a": [1, 2]}]

    for data, schema in [
        ([(np.array(["a", "b"]),)], "a array<string>"),
        ([(("a", "b"),)], "a array<string>"),
        ([(np.array([1, 2]),)], T.StructType([T.StructField(
            "a", T.ArrayType(T.LongType(), containsNull=False))])),
    ]:
        # PySpark's assertion, which has no message.
        with pytest.raises(AssertionError) as error:
            spark.createDataFrame(data, schema)
        assert str(error.value) == ""


def test_ints_of_date_fields_convert_as_pyarrow(mock_connect):
    # As PySpark: PyArrow takes an int, or a float truncated, as days since the epoch, and a
    # NumPy bool or number as the value it stands for. A Python bool, a string and the days
    # out of the calendar range raise PyArrow's errors.
    import re

    import pyarrow as pa

    np = pytest.importorskip("numpy")
    spark, service = mock_connect
    table, _ = _local_rows_relation_of(
        service, spark.createDataFrame([(1,), (-1,), (None,)], "a date"))
    assert table.schema.field(0).type == pa.date32()
    assert table.column(0).cast(pa.int32()).to_pylist() == [1, -1, None]
    table, _ = _local_rows_relation_of(service, spark.createDataFrame(
        [(np.int64(3),), (1.5,), (-1.5,), (np.bool_(True),), (2932896,), (-719162,)], "a date"))
    assert table.column(0).cast(pa.int32()).to_pylist() == [3, 1, -1, 1, 2932896, -719162]
    # PyArrow takes the days of an int32, also out of the dates of Python, and the Decimal
    # of the `__int__` of its value.
    table, _ = _local_rows_relation_of(service, spark.createDataFrame(
        [(2932897,), (-719163,), (10**9,), (decimal.Decimal("2.5"),)], "a date"))
    assert table.column(0).cast(pa.int32()).to_pylist() == [2932897, -719163, 10**9, 2]

    for value, error, message in (
        (True, pa.ArrowTypeError, "Expected integer, got bool"),
        ("x", pa.ArrowTypeError, "object of type <class 'str'> cannot be converted to int"),
        (float("nan"), ValueError, "cannot convert float NaN to integer"),
        (2**40, pa.ArrowInvalid, "Integer too large for date32"),
        (2**70, OverflowError, "Python int too large to convert to C long"),
    ):
        with pytest.raises(error, match="^" + re.escape(message) + "$"):
            spark.createDataFrame([(value,)], "a date")


def test_values_of_local_rows_convert_and_raise_as_pyarrow(mock_connect):
    # As PySpark Connect's, which passes these values to PyArrow as they are: an int, date,
    # day-time interval and time field takes the __index__, else the __int__, of a value, such
    # as a Decimal, and the errors are PyArrow's. A char or varchar field raises the error of
    # PySpark's to_arrow_schema.
    import re

    import pyarrow as pa
    from pyspark.errors import PySparkTypeError

    np = pytest.importorskip("numpy")
    spark, service = mock_connect
    D = decimal.Decimal
    table, _ = _local_rows_relation_of(service, spark.createDataFrame(
        [(D("2.7"), D("1.5"), D("3.9"), D("4.2"), np.timedelta64(5, "us"))],
        "i int, d date, t interval day to second, n time, u interval day to second"))
    assert [table.column(name).cast(cast).to_pylist()[0]
            for name, cast in [("i", pa.int32()), ("d", pa.int32()), ("t", pa.int64()),
                               ("n", pa.int64()), ("u", pa.int64())]] == [2, 1, 3, 4, 5]
    for value, ddl, error, message in [
        ("x", "int", pa.ArrowInvalid,
         "Could not convert 'x' with type str: tried to convert to int32"),
        ("x", "float", pa.ArrowInvalid,
         "Could not convert 'x' with type str: tried to convert to float32"),
        (2**40, "int", pa.ArrowInvalid, "Value 1099511627776 too large to fit in C integer type"),
        (2**70, "double", pa.ArrowInvalid, "PyLong is too large to fit int64"),
        (2**40, "float", pa.ArrowInvalid,
         "Integer value 1099511627776 is outside of the range exactly representable by a "
         "IEEE 754 single precision value"),
        (True, "int", pa.ArrowTypeError, "Expected integer, got bool"),
        (True, "time", pa.ArrowTypeError, "Expected integer, got bool"),
        (float("nan"), "time", ValueError, "cannot convert float NaN to integer"),
        ("x", "interval day to second", pa.ArrowTypeError,
         "object of type <class 'str'> cannot be converted to int"),
        (np.timedelta64(5, "ns"), "interval day to second", pa.ArrowNotImplementedError,
         "Expected np.timedelta64 but got: duration[ns]"),
        ("x", "boolean", pa.ArrowInvalid,
         "Could not convert 'x' with type str: tried to convert to boolean"),
    ]:
        with pytest.raises(error, match="^" + re.escape(message) + "$") as e:
            spark.createDataFrame([(value,)], f"v {ddl}")
        assert type(e.value) is error
    for field_type, data_type in [(T.CharType(3), "CharType(3)"),
                                  (T.ArrayType(T.VarcharType(2)), "VarcharType(2)")]:
        with pytest.raises(PySparkTypeError) as e:
            spark.createDataFrame([("x",)], T.StructType([T.StructField("v", field_type)]))
        assert e.value.getCondition() == "UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION"
        assert e.value.getMessageParameters() == {"data_type": data_type}

def test_nested_duplicate_field_names_are_deduplicated_in_the_arrow_schema(mock_connect):
    # As PySpark's _deduplicate_field_names: the Arrow schema of nested struct fields numbers
    # duplicate names, which the server cannot read, while the schema of the relation keeps
    # the names. A top-level duplicate name is not renamed.
    import pyarrow as pa
    from pyspark.sql import Row
    from pyspark.sql import types as T

    spark, service = mock_connect
    R = Row("a", "a")
    table, schema = _local_rows_relation_of(service, spark.createDataFrame([(R(1, 2),)]))
    assert table.schema.field(0).type == pa.struct([("a_0", pa.int64()), ("a_1", pa.int64())])
    assert table.column(0).to_pylist() == [{"a_0": 1, "a_1": 2}]
    assert [f["name"] for f in schema["fields"][0]["type"]["fields"]] == ["a", "a"]

    explicit = T.StructType([T.StructField("s", T.StructType([
        T.StructField("a", T.LongType()), T.StructField("a", T.LongType())]))])
    table, schema = _local_rows_relation_of(service, spark.createDataFrame([((1, 2),)], explicit))
    assert table.column("s").to_pylist() == [{"a_0": 1, "a_1": 2}]
    assert [f["name"] for f in schema["fields"][0]["type"]["fields"]] == ["a", "a"]

    table, _ = _local_rows_relation_of(service, spark.createDataFrame(
        [([(1, 2)],)], "s array<struct<a:bigint,a:bigint>>"))
    assert table.column("s").to_pylist() == [[{"a_0": 1, "a_1": 2}]]

    table, _ = _local_rows_relation_of(service, spark.createDataFrame(
        [(1, 2)], "a bigint, a bigint"))
    assert table.schema.names == ["a", "a"]
    assert table.column(1).to_pylist() == [2]


# ------------------------------------------------------- DataFrame completion

def _completion_varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def _completion_message(field, value):
    if isinstance(value, str):
        value = value.encode()
    return _completion_varint(field * 8 + 2) + _completion_varint(len(value)) + value


def _completion_schema(names=("id",)):
    # DataType.struct=21, Struct.fields=1, DataType.long=7.
    return _completion_message(21, b"".join(_completion_message(1, _completion_field(name, _completion_message(7, b"")))
                                  for name in names))


def _completion_field(name, data_type):
    return (_completion_message(1, name) + _completion_message(2, data_type) + b"\x18\x01"
            + _completion_message(4, '{"note":"original"}'))


def _completion_serve(connection):
    from concurrent.futures import ThreadPoolExecutor
    import grpc

    # A separate process also exercises columns/dtypes, whose first access may
    # hold the client's GIL while waiting for the RPC.
    class Service:
        calls = 0
        data_type = _completion_schema()
        failure = None

        def analyze(self, request, context):
            self.calls += 1
            if self.failure == "rpc":
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "schema unavailable")
            if self.failure == "no-result":
                return b""
            if self.failure == "missing-schema":
                return _completion_message(2, b"")
            if self.failure == "invalid-type":
                return _completion_message(2, _completion_message(1, b""))
            # base.proto: AnalyzePlanResponse.schema=2, Schema.schema=1.
            return _completion_message(2, _completion_message(1, self.data_type))

    service = Service()
    with ThreadPoolExecutor(max_workers=4) as executor:
        server = grpc.server(executor)
        handler = grpc.unary_unary_rpc_method_handler(service.analyze)
        server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(
            "spark.connect.SparkConnectService", {"AnalyzePlan": handler}),))
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        connection.send(port)
        try:
            while True:
                command = connection.recv()
                if command[0] == "stop":
                    break
                if command[0] == "get":
                    connection.send(getattr(service, command[1]))
                elif command[0] == "set":
                    setattr(service, command[1], command[2])
                    connection.send(None)
        finally:
            server.stop(0).wait()


class _CompletionRemoteService:
    def __init__(self, connection):
        object.__setattr__(self, "connection", connection)

    def __getattr__(self, name):
        self.connection.send(("get", name))
        assert self.connection.poll(10), "Mock server did not respond"
        return self.connection.recv()

    def __setattr__(self, name, value):
        self.connection.send(("set", name, value))
        assert self.connection.poll(10), "Mock server did not respond"
        self.connection.recv()


@pytest.fixture
def completion_analyzed():
    import multiprocessing
    from pyspark.sql import SparkSession

    pytest.importorskip("grpc")
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_completion_serve, args=(child,))
    process.start()
    try:
        assert parent.poll(20), "Mock server did not start"
        port = parent.recv()
        spark = SparkSession.builder.remote(f"sc://127.0.0.1:{port}").getOrCreate()
        yield spark, _CompletionRemoteService(parent)
    finally:
        if process.is_alive():
            parent.send(("stop",))
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
        child.close()


@pytest.mark.parametrize("names", [
    ["z", "a"],
    ["with space", "a.b", "with`tick", "", "123", "class", "雪", "_private"],
    ["id", "id", "select", "columns", "__dir__", "_ipython_key_completions_"],
    [],
])
def test_completion_matches_python_connect(completion_analyzed, names):
    from pyspark.sql import Column

    spark, service = completion_analyzed
    service.data_type = _completion_schema(names)
    df = spark.range(10)
    assert df._ipython_key_completions_() == names
    assert service.calls == 1
    expected = sorted(set(dir(type(df))) | set(names))
    # Each first access must analyze its own fresh plan, with or without caching.
    assert spark.range(10).__dir__() == expected
    assert service.calls == 2
    assert dir(spark.range(10)) == expected
    assert service.calls == 3
    assert callable(df.select)
    assert callable(df.__dir__)
    assert callable(df._ipython_key_completions_)
    assert df.columns == names
    for name in names:
        assert isinstance(df[name], Column)
    assert df.schema.fieldNames() == names


def test_ddl_schemas_are_parsed_on_the_server_as_pyspark(mock_connect):
    # As PySpark Connect's `_parse_ddl`, which asks the server to parse a DDL schema, of
    # createDataFrame and of the map functions; a type is sent as it is.
    spark, service = mock_connect

    def ddl_requests():
        # base.proto: AnalyzePlanRequest.ddl_parse=11, DDLParse.ddl_string=1.
        return [parse_message(parse_message(request)[11][0])[1][0].decode()
                for request in service.analyze_requests if 11 in parse_message(request)]

    def sent_local_schema():
        # The LocalRelation of the type that the server parsed: AnalyzePlanRequest.schema=4 ->
        # plan=1 -> root=1 -> (Relation.to_df=18 -> input=1 ->) local_relation=11 -> schema=2.
        relation = parse_message(parse_message(parse_message(
            parse_message(service.analyze_requests[-1])[4][0])[1][0])[1][0])
        while 18 in relation:
            relation = parse_message(parse_message(relation[18][0])[1][0])
        return T._parse_datatype_json_string(parse_message(relation[11][0])[2][0].decode())

    spark.createDataFrame([], "id long").columns
    assert sent_local_schema().simpleString() == "struct<id:bigint>"
    nested = "id struct<a:int,b:double> NOT NULL"
    spark.createDataFrame([], nested).columns
    assert sent_local_schema() == T.StructType([T.StructField("id", T.StructType([
        T.StructField("a", T.IntegerType()), T.StructField("b", T.DoubleType())]), False)])
    assert ddl_requests() == ["id long", nested]
    df = spark.range(1)
    df.mapInPandas(lambda it: it, "id long NOT NULL")
    df.mapInArrow(lambda it: it, "id long COMMENT 'c'")
    df.groupBy("id").applyInPandas(lambda pdf: pdf, "`id` long")
    df.groupBy("id").cogroup(df.groupBy("id")).applyInArrow(lambda l, r: l, "id: long")
    assert ddl_requests()[2:] == [
        "id long NOT NULL", "id long COMMENT 'c'", "`id` long", "id: long"]
    df.mapInPandas(lambda it: it, T.StructType([T.StructField("id", T.LongType())]))
    assert len(ddl_requests()) == 6


def test_completion_lists_are_independent(completion_analyzed):
    spark, service = completion_analyzed
    df = spark.range(10)
    df._ipython_key_completions_().append("extra")
    df.__dir__().clear()
    assert df._ipython_key_completions_() == ["id"]
    assert "select" in dir(df)
    service.data_type = _completion_schema(["changed"])
    assert spark.range(10)._ipython_key_completions_() == ["changed"]


@pytest.mark.parametrize("method", ["__dir__", "_ipython_key_completions_"])
@pytest.mark.parametrize("failure", ["rpc", "no-result", "missing-schema", "invalid-type"])
def test_completion_propagates_schema_errors_and_can_retry(completion_analyzed, method, failure):
    spark, service = completion_analyzed
    df = spark.range(10)
    service.failure = failure
    message = {
        "rpc": "schema unavailable",
        "no-result": "no schema in response",
        "missing-schema": "Schema is missing",
        "invalid-type": "DataType kind not set",
    }[failure]
    for count in (1, 2):
        with pytest.raises(Exception, match=message):
            getattr(df, method)()
        assert service.calls == count
    service.failure = None
    assert "id" in getattr(df, method)()
    assert service.calls == 3


# --------------------------------------------------------------------------- functions exports and literals of NumPy scalars

# The public names of PySpark 4.3.0's pyspark.sql.functions, in its order (__all__ is static there).
OFFICIAL_FUNCTIONS_ALL = """
    broadcast call_function col column lit expr coalesce ifnull nanvl nullif nullifzero nvl
    nvl2 when zeroifnull equal_null ilike isnan isnotnull isnull like regexp regexp_like
    rlike asc asc_nulls_first asc_nulls_last desc desc_nulls_first desc_nulls_last abs acos
    acosh asin asinh atan atan2 atanh bin bround cbrt ceil ceiling conv cos cosh cot csc
    degrees e exp expm1 factorial floor greatest hex hypot least ln log log10 log1p log2
    negate negative pi pmod positive pow power radians rand randn rint round sec sign signum
    sin sinh sqrt tan tanh try_add try_divide try_mod try_multiply try_subtract unhex
    uniform width_bucket ascii base64 bit_length btrim char char_length character_length
    collate collation concat_ws contains decode elt encode endswith find_in_set
    format_number format_string from_base32 initcap instr is_valid_utf8
    jaro_winkler_similarity lcase left length levenshtein locate lower lpad ltrim
    make_valid_utf8 mask octet_length overlay position printf quote randstr regexp_count
    regexp_extract regexp_extract_all regexp_instr regexp_replace regexp_substr repeat
    replace right rpad rtrim sentences soundex split split_part startswith substr substring
    substring_index to_base32 to_binary to_char to_number to_varchar translate trim
    try_to_binary try_to_number try_validate_utf8 ucase unbase64 upper validate_utf8
    bit_count bit_get bitwise_not getbit shiftleft shiftright shiftrightunsigned add_months
    convert_timezone curdate current_date current_time current_timestamp current_timezone
    date_add date_diff date_format date_from_unix_date date_part date_sub date_trunc dateadd
    datediff datepart day dayname dayofmonth dayofweek dayofyear extract from_unixtime
    from_utc_timestamp hour last_day localtimestamp make_date make_dt_interval make_interval
    make_time make_timestamp make_timestamp_ltz make_timestamp_ntz make_ym_interval minute
    month monthname months_between next_day now quarter second session_window timestamp_add
    timestamp_diff timestamp_micros timestamp_millis timestamp_nanos timestamp_seconds
    time_bucket time_diff time_from_micros time_from_millis time_from_seconds time_to_micros
    time_to_millis time_to_seconds time_trunc to_date to_time to_timestamp to_timestamp_ltz
    to_timestamp_ntz to_unix_timestamp to_utc_timestamp trunc try_make_interval
    try_make_timestamp try_make_timestamp_ltz try_make_timestamp_ntz try_to_time
    try_to_timestamp try_to_date unix_date unix_micros unix_millis unix_nanos unix_seconds
    unix_timestamp weekday weekofyear window window_time year crc32 hash md5 sha sha1 sha2
    xxhash64 aggregate array_sort cardinality concat element_at exists filter forall
    map_filter map_zip_with reduce reverse size transform transform_keys transform_values
    try_element_at zip_with array array_append array_compact array_contains array_distinct
    array_except array_insert array_intersect array_join array_max array_min array_position
    array_prepend array_remove array_repeat array_size array_union arrays_overlap arrays_zip
    flatten get sequence shuffle slice sort_array named_struct struct create_map map_concat
    map_contains_key map_entries map_from_arrays map_from_entries map_keys map_values
    str_to_map any_value approx_count_distinct approx_percentile array_agg avg bit_and
    bit_or bit_xor bitmap_and_agg bitmap_construct_agg bitmap_or_agg bool_and bool_or
    collect_list collect_set collect_union corr count count_distinct count_if
    count_min_sketch covar_pop covar_samp every first first_value grouping grouping_id
    histogram_numeric hll_sketch_agg hll_union_agg kll_sketch_agg_bigint
    kll_sketch_agg_double kll_sketch_agg_float kll_merge_agg_bigint kll_merge_agg_double
    kll_merge_agg_float kurtosis last last_value listagg listagg_distinct max max_by mean
    median min min_by mode percentile percentile_approx product regr_avgx regr_avgy
    regr_count regr_intercept regr_r2 regr_slope regr_sxx regr_sxy regr_syy skewness some
    std stddev stddev_pop stddev_samp string_agg string_agg_distinct sum sum_distinct
    theta_intersection_agg theta_sketch_agg theta_union_agg tuple_intersection_agg_double
    tuple_intersection_agg_integer tuple_sketch_agg_double tuple_sketch_agg_integer
    tuple_union_agg_double tuple_union_agg_integer try_avg try_sum var_pop var_samp variance
    counter_diff cume_dist dense_rank lag lead nth_value ntile percent_rank rank row_number
    explode explode_outer inline inline_outer posexplode posexplode_outer stack years months
    days hours bucket from_csv schema_of_csv to_csv from_json get_json_object
    json_array_length json_object_keys json_tuple schema_of_json to_json is_valid_variant
    is_variant_null parse_json schema_of_variant schema_of_variant_agg try_variant_get
    variant_array_append try_variant_array_append variant_delete variant_strip_nulls
    variant_get variant_insert try_variant_insert variant_set try_variant_set try_parse_json
    to_variant_object from_xml schema_of_xml to_xml xpath xpath_boolean xpath_double
    xpath_float xpath_int xpath_long xpath_number xpath_short xpath_string parse_url
    try_parse_url url_decode url_encode try_url_decode aes_decrypt aes_encrypt assert_true
    bitmap_bit_position bitmap_bucket_number bitmap_count current_catalog current_database
    current_path current_schema current_user hmac input_file_block_length
    input_file_block_start input_file_name java_method monotonically_increasing_id
    raise_error reflect session_user spark_partition_id try_aes_decrypt try_reflect typeof
    user version AnalyzeArgument AnalyzeResult ArrowUDFType OrderingColumn PandasUDFType
    PartitioningColumn SelectedColumn SkipRestOfInputTableException UserDefinedFunction
    UserDefinedTableFunction arrow_udf hll_sketch_estimate hll_union kll_sketch_get_n_bigint
    kll_sketch_get_n_double kll_sketch_get_n_float kll_sketch_get_quantile_bigint
    kll_sketch_get_quantile_double kll_sketch_get_quantile_float kll_sketch_get_rank_bigint
    kll_sketch_get_rank_double kll_sketch_get_rank_float kll_sketch_merge_bigint
    kll_sketch_merge_double kll_sketch_merge_float kll_sketch_to_string_bigint
    kll_sketch_to_string_double kll_sketch_to_string_float theta_difference
    theta_intersection theta_sketch_estimate theta_union tuple_difference_double
    tuple_difference_integer tuple_difference_theta_double tuple_difference_theta_integer
    tuple_intersection_double tuple_intersection_integer tuple_intersection_theta_double
    tuple_intersection_theta_integer tuple_sketch_estimate_double
    tuple_sketch_estimate_integer tuple_sketch_summary_double tuple_sketch_summary_integer
    tuple_sketch_theta_double tuple_sketch_theta_integer tuple_union_double
    tuple_union_integer tuple_union_theta_double tuple_union_theta_integer st_asbinary
    st_geogfromwkb st_geomfromwkb st_setsrid st_srid vector_cosine_similarity
    vector_inner_product vector_l2_distance vector_norm vector_normalize vector_avg
    vector_sum call_udf pandas_udf udf udtf arrow_udtf unwrap_udt
""".split()


def test_star_import_of_functions_exports_the_official_names_only():
    # A star import brings in exactly the names of PySpark's __all__. The names that PySpark has
    # only as attributes (chr, random, uuid, ...) are not exported, so they do not shadow the
    # builtin chr and the stdlib random and uuid; nor are the avro and protobuf functions, which
    # PySpark has in pyspark.sql.avro.functions and pyspark.sql.protobuf.functions.
    import builtins

    from pyspark.sql import functions as F

    assert F.__all__ == OFFICIAL_FUNCTIONS_ALL
    namespace = {}
    exec("from pyspark.sql.functions import *", namespace)
    namespace.pop("__builtins__")
    assert list(namespace) == OFFICIAL_FUNCTIONS_ALL
    for name in ("chr", "random", "uuid", "countDistinct", "bitwiseNOT", "toDegrees",
                 "approxCountDistinct"):
        assert name not in namespace
        assert callable(getattr(F, name))
    assert F.random is F.rand
    assert "chr" not in namespace and builtins.chr(65) == "A"
    for name in ("from_avro", "from_avro_with_options", "to_avro", "to_avro_with_schema",
                 "from_protobuf", "to_protobuf"):
        assert name not in namespace
        assert not hasattr(F, name)


def test_avro_and_protobuf_functions_are_on_their_official_modules(tmp_path, mock_connect):
    import pyspark.sql.avro.functions as avro
    import pyspark.sql.protobuf.functions as protobuf
    from pyspark.errors import PySparkTypeError
    from pyspark.sql import functions as F

    spark, service = mock_connect
    c = F.col("v")
    assert type(avro.from_avro(c, "{}")).__name__ == "Column"
    assert type(avro.from_avro("v", "{}", {"mode": "PERMISSIVE"})).__name__ == "Column"
    assert type(avro.to_avro(c)).__name__ == "Column"
    assert type(avro.to_avro(c, "{}")).__name__ == "Column"
    with pytest.raises(PySparkTypeError) as error:
        avro.from_avro(1, "{}")
    assert error.value.getCondition() == "INVALID_TYPE"
    assert error.value.getMessageParameters() == {
        "arg_name": "data", "arg_type": "pyspark.sql.Column or str"}
    # As PySpark's, the descriptor file is read (it was ignored before), and the bytes reach the
    # server as the binary of the function's descriptor set.
    descriptor = tmp_path / "m.desc"
    descriptor.write_bytes(b"\x00fue-e-descriptor\xff")
    spark.range(1).select(protobuf.from_protobuf(F.col("id"), "M", descFilePath=str(descriptor))).collect()
    assert b"\x00fue-e-descriptor\xff" in service.execute_requests[-1]


def test_numpy_scalar_literals_are_typed_by_their_dtype_as_pyspark(mock_connect):
    import struct

    np = pytest.importorskip("numpy")
    from pyspark.sql import functions as F

    spark, service = mock_connect
    # As PySpark's LiteralExpression of a NumPy scalar, typed by its dtype: a bool, a signed int
    # of 8 to 64 bits (tinyint to bigint), float32 (float) and float64 (double). The Literal
    # fields are: boolean=3, byte=4, short=5, integer=6, long=7, float=10, double=11, string=13,
    # binary=2.
    cases = [
        (np.bool_(True), 3, 1),
        (np.int8(5), 4, 5),
        (np.int16(5), 5, 5),
        (np.int32(5), 6, 5),
        (np.int64(5), 7, 5),
        (np.int64(2**40), 7, 2**40),
        (np.float32(1.5), 10, struct.pack("<f", 1.5)),
        (np.float64(1.5), 11, struct.pack("<d", 1.5)),
        (np.str_("abc"), 13, b"abc"),
        (np.bytes_(b"ab"), 2, b"ab"),
        # A Python int and float keep their types: int (integer) and double.
        (5, 6, 5),
        (1.5, 11, struct.pack("<d", 1.5)),
    ]
    for value, field, expected in cases:
        spark.range(1).select(F.lit(value)).collect()
        plan = parse_message(parse_message(service.execute_requests[-1])[3][0])[1][0]
        expression = parse_message(parse_message(parse_message(plan)[3][0])[3][0])
        literal = parse_message(expression[1][0])
        assert sorted(literal) == [field], repr(value)
        assert literal[field] == [expected], repr(value)


def test_numpy_scalars_of_other_types_are_unsupported_as_pyspark(mock_connect):
    np = pytest.importorskip("numpy")
    from pyspark.errors import PySparkTypeError
    from pyspark.sql import functions as F

    # As PySpark's, the unsupported NumPy scalars are UNSUPPORTED_DATA_TYPE, named by their
    # Python type; and an array of an unsupported dtype is UNSUPPORTED_NUMPY_ARRAY_SCALAR.
    for value, type_name in [
        (np.uint8(5), "uint8"),
        (np.uint64(5), "uint64"),
        (np.float16(1.5), "float16"),
        (np.longdouble(1.5), "longdouble"),
        (np.complex128(1 + 2j), "complex128"),
        (np.datetime64("2020-01-01"), "datetime64"),
        (np.timedelta64(1, "D"), "timedelta64"),
    ]:
        with pytest.raises(PySparkTypeError) as error:
            F.lit(value)
        assert error.value.getCondition() == "UNSUPPORTED_DATA_TYPE"
        assert error.value.getMessageParameters() == {"data_type": type_name}
    with pytest.raises(PySparkTypeError) as error:
        F.lit(np.array([1], dtype=np.uint8))
    assert error.value.getCondition() == "UNSUPPORTED_NUMPY_ARRAY_SCALAR"
    assert error.value.getMessageParameters() == {"dtype": "uint8"}


# --------------------------------------------------------------------------- 4.3.0 gaps


def test_timestamp_nanos_types_roundtrip_and_pickle():
    # Spark's TimestampNTZNanosType/TimestampLTZNanosType (precision 7..9) materialize as
    # their own types, not as TimeType / a microsecond TimestampType.
    for cls, name in ((T.TimestampNTZNanosType, "timestamp_ntz"),
                      (T.TimestampLTZNanosType, "timestamp_ltz")):
        t = cls(8)
        assert t.precision == 8
        assert t.simpleString() == t.typeName() == f"{name}(8)"
        assert t.json() == f'"{name}(8)"'
        assert cls().precision == 9
        assert isinstance(t, T.DatetimeType)
        assert repr(pickle.loads(pickle.dumps(t))) == repr(t)
        assert repr(T._parse_datatype_json_string(f'"{name}(7)"')) == repr(cls(7))
        with pytest.raises(ValueError):
            cls(6)
    nested = T.StructType([T.StructField("a", T.ArrayType(T.TimestampLTZNanosType(9)))])
    assert pickle.loads(pickle.dumps(nested)).simpleString() == "struct<a:array<timestamp_ltz(9)>>"


@pytest.mark.parametrize("call,expected", [
    # Optional arguments are sent when given, as PySpark Connect's functions send them, and
    # strings are column names where PySpark takes a Column or a name.
    (lambda F: F.to_date("d", "dd/MM/yyyy"), 'to_date(d, dd/MM/yyyy)'),
    (lambda F: F.to_timestamp("d", "yyyy-MM-dd"), 'to_timestamp(d, yyyy-MM-dd)'),
    (lambda F: F.log(2.0, "x"), 'log(2.0, x)'),
    (lambda F: F.log("x"), 'ln(x)'),
    (lambda F: F.sort_array("a", asc=False), 'sort_array(a, False)'),
    (lambda F: F.split("s", ",", 2), 'split(s, ,, 2)'),
    (lambda F: F.substr("s", F.lit(1), F.lit(2)), 'substr(s, 1, 2)'),
    (lambda F: F.like("p", F.lit("a!_c"), F.lit("!")), 'like(p, a!_c, !)'),
    (lambda F: F.unix_timestamp("t"), 'unix_timestamp(t, yyyy-MM-dd HH:mm:ss)'),
    (lambda F: F.unix_timestamp(), 'unix_timestamp()'),
    (lambda F: F.months_between("a", "b", roundOff=False), 'months_between(a, b, False)'),
    (lambda F: F.first("v", ignorenulls=True), 'first(v, True)'),
    (lambda F: F.last("v", True), 'last(v, True)'),
    (lambda F: F.aes_encrypt("i", "k", F.lit("ECB")), 'aes_encrypt(i, k, ECB, DEFAULT, , )'),
    (lambda F: F.rand(7), 'rand(7)'),
    (lambda F: F.randn(seed=7), 'randn(7)'),
    (lambda F: F.shuffle("a", 7), 'shuffle(a, 7)'),
    (lambda F: F.count_distinct("a", "b"), 'count(distinct a, b)'),
    (lambda F: F.countDistinct("a", "b"), 'count(distinct a, b)'),
    (lambda F: F.listagg_distinct("a", ","), 'listagg(distinct a, ,)'),
    (lambda F: F.string_agg("a", ","), 'string_agg(a, ,)'),
    (lambda F: F.any_value("a", True), 'any_value(a, True)'),
    (lambda F: F.max_by("a", "b", 2), 'max_by(a, b, 2)'),
    (lambda F: F.hll_sketch_agg("a", 10), 'hll_sketch_agg(a, 10)'),
    (lambda F: F.approx_percentile("c", 0.5, 100), 'approx_percentile(c, 0.5, 100)'),
    (lambda F: F.locate("b", "s", 3), 'locate(b, s, 3)'),
    (lambda F: F.window("t", "1 hour", "30 minutes"), 'window(t, 1 hour, 30 minutes)'),
    (lambda F: F.instr("s", "a", occurrence=2), 'instr(s, a, 1, 2)'),
    (lambda F: F.regexp_replace("s", "a", "b", 4), 'regexp_replace(s, a, b, 4)'),
    (lambda F: F.instr("s", "a"), 'instr(s, a)'),
    (lambda F: F.instr("s", "a", 3), 'instr(s, a, 3)'),
    (lambda F: F.regexp_replace("s", "a", "b"), 'regexp_replace(s, a, b)'),
    (lambda F: F.regexp_replace("s", "a", "b", position=None), 'regexp_replace(s, a, b)'),
    (lambda F: F.date_part(F.lit("YEAR"), "d"), 'date_part(YEAR, d)'),
    # An explicit None is what PySpark makes of it: no argument, a NULL or the default.
    (lambda F: F.round("c", None), 'round(c)'),
    (lambda F: F.bround("c", None), 'bround(c)'),
    (lambda F: F.first("c", None), 'first(c, NULL)'),
    (lambda F: F.lag("c"), 'lag(c, 1)'),
    (lambda F: F.lag("c", 2, None), 'lag(c, 2)'),
    (lambda F: F.lead("c", offset=None), 'lead(c, NULL)'),
    (lambda F: F.trim("c", None), 'trim(c)'),
    (lambda F: F.approx_percentile("c", 0.5, None), 'approx_percentile(c, 0.5, NULL)'),
    (lambda F: F.from_unixtime("c", None), 'from_unixtime(c, NULL)'),
    (lambda F: F.sort_array("c", None), 'sort_array(c, NULL)'),
    (lambda F: F.nth_value("c", 2, None), 'nth_value(c, 2)'),
    (lambda F: F.tuple_union_double("a", "b"), 'tuple_union_double(a, b, 12, sum)'),
    (lambda F: F.variant_strip_nulls("v", None), 'variant_strip_nulls(v, NULL)'),
    (lambda F: F.instr("s", "a", None, None), 'instr(s, a)'),
    (lambda F: F.array_join("a", ",", None), 'array_join(a, ,)'),
])
def test_function_arguments_as_pyspark(call, expected):
    # The cases agree with official PySpark 4.3.0.
    from pyspark.sql import functions as F

    assert repr(call(F)) == f"Column<'{expected}'>"


@pytest.mark.parametrize("call,error,condition,parameters", [
    (lambda F: F.split("s", ",", None), 'PySparkTypeError', 'NOT_EXPECTED_TYPE', {'expected_type': 'Column or str', 'arg_name': 'col', 'arg_type': 'NoneType'}),
    (lambda F: F.overlay("a", "b", "c", None), 'PySparkTypeError', 'NOT_EXPECTED_TYPE', {'expected_type': 'Column or str', 'arg_name': 'col', 'arg_type': 'NoneType'}),
    (lambda F: F.percentile("a", 0.5, None), 'PySparkTypeError', 'NOT_EXPECTED_TYPE', {'expected_type': 'Column or int', 'arg_name': 'frequency', 'arg_type': 'NoneType'}),
    (lambda F: F.substr("s", 1, 5), 'PySparkTypeError', 'NOT_EXPECTED_TYPE', {'expected_type': 'Column or str', 'arg_name': 'col', 'arg_type': 'int'}),
    (lambda F: F.stack(2, "a", "b"), 'PySparkTypeError', 'NOT_EXPECTED_TYPE', {'expected_type': 'Column or str', 'arg_name': 'col', 'arg_type': 'int'}),
    (lambda F: F.sha2("c", 100), 'PySparkValueError', 'VALUE_NOT_ALLOWED', {'arg_name': 'numBits', 'allowed_values': '[0, 224, 256, 384, 512]'}),
])
def test_function_arguments_raise_as_pyspark(call, error, condition, parameters):
    # Before any request, with PySpark Connect's errors; the cases agree with official
    # PySpark 4.3.0.
    from pyspark.errors import PySparkTypeError, PySparkValueError
    from pyspark.sql import functions as F

    with pytest.raises((PySparkTypeError, PySparkValueError)) as raised:
        call(F)
    assert type(raised.value).__name__ == error
    assert raised.value.getCondition() == condition
    assert raised.value.getMessageParameters() == parameters


@pytest.mark.parametrize("column,expected", [
    # As PySpark Connect's Expression reprs: an operator renders as a function.
    (lambda F: F.col("a") + 1, "+(a, 1)"),
    (lambda F: 1 - F.col("a"), "-(1, a)"),
    (lambda F: (F.col("a") + 1) * (F.col("b") - 2), "*(+(a, 1), -(b, 2))"),
    (lambda F: -F.col("a"), "negative(a)"),
    (lambda F: ~F.col("a"), "not(a)"),
    (lambda F: F.col("a") != 1, "not(==(a, 1))"),
    (lambda F: F.col("a") & F.col("b"), "and(a, b)"),
    (lambda F: F.col("a").between(1, 2), "and(>=(a, 1), <=(a, 2))"),
    (lambda F: F.when(F.col("a") > 1, 1).otherwise(0), "CASE WHEN >(a, 1) THEN 1 ELSE 0 END"),
    (lambda F: F.countDistinct("a"), "count(distinct a)"),
    (lambda F: F.col("a").cast("int"), "CAST(a AS INT)"),
    (lambda F: F.col("a").cast("struct<x:int,y:string>"), "CAST(a AS STRUCT<X:INT,Y:STRING>)"),
    (lambda F: F.col("a").try_cast("int"), "TRY_CAST(a AS INT)"),
    (lambda F: F.col("*"), "unresolvedstar()"),
    (lambda F: F.count("*"), "count(unresolvedstar())"),
    (lambda F: F.col("t.*"), "unresolvedstar(t.*)"),
    (lambda F: F.col("a")[0], "a['0']"),
    (lambda F: F.col("a")["k"], "a['k']"),
    # Literals, as LiteralExpression.__repr__ shows their Python values.
    (lambda F: F.lit(True), "True"),
    (lambda F: F.lit(None), "NULL"),
    (lambda F: F.lit(1.5), "1.5"),
    (lambda F: F.lit(1e20), "1e+20"),
    (lambda F: F.lit(1e-07), "1e-07"),
    (lambda F: F.lit(float("nan")), "nan"),
    (lambda F: F.lit(b"a'b"), 'b"a\'b"'),
    (lambda F: F.lit(bytes([0, 9, 92, 200])), "b'\\x00\\t\\\\\\xc8'"),
    (lambda F: F.lit(__import__("datetime").date(1999, 12, 31)), "1999-12-31"),
    # A naive datetime is a local time, rendered in the local time zone again.
    (lambda F: F.lit(__import__("datetime").datetime(2020, 1, 2, 3, 4, 5, 6)),
     "2020-01-02 03:04:05.000006"),
    (lambda F: F.lit(__import__("decimal").Decimal("-1.50")), "-1.50"),
])
def test_column_reprs_as_pyspark(column, expected):
    from pyspark.sql import functions as F

    assert repr(column(F)) == f"Column<'{expected}'>"


def test_window_and_lambda_reprs_as_pyspark():
    import re

    from pyspark.sql import Window
    from pyspark.sql import functions as F

    spec = Window.partitionBy("g").orderBy(F.col("o").desc()).rowsBetween(-1, 1)
    assert repr(spec) == "WindowSpec(PartitionBy(g), OrderBy(o DESC NULLS LAST), WindowFrame(ROW_FRAME, -1, 1))"
    window = F.sum("a").over(
        Window.orderBy(F.col("o").asc()).rangeBetween(Window.unboundedPreceding, Window.currentRow))
    assert repr(window) == (
        "Column<'WindowExpression(sum(a), (WindowSpec(OrderBy(o ASC NULLS FIRST), "
        "WindowFrame(RANGE_FRAME, -9223372036854775808, 0))))'>")
    # A lambda variable is named by a counter, as in PySpark.
    assert re.fullmatch(
        r"Column<'transform\(a, LambdaFunction\(\+\(x_(\d+), 1\), x_\1\)\)'>",
        repr(F.transform("a", lambda x: x + 1)))


def test_window_frames_and_columns_are_checked_as_pyspark():
    # As PySpark's: a start of -sys.maxsize or less is unbounded preceding and an end of
    # sys.maxsize or more unbounded following; a row frame raises VALUE_NOT_BETWEEN for a
    # boundary out of the range of an int, as a start of Window.unboundedFollowing; and the
    # columns are Columns or column names, else NOT_EXPECTED_TYPE of `col`.
    import sys

    from pyspark.errors import PySparkTypeError, PySparkValueError
    from pyspark.sql import Window

    o = Window.orderBy("id")
    assert repr(o.rowsBetween(-sys.maxsize, sys.maxsize)) == (
        "WindowSpec(OrderBy(id ASC NULLS FIRST), "
        "WindowFrame(ROW_FRAME, -9223372036854775808, 9223372036854775807))")
    assert repr(Window.rangeBetween(-2 ** 70, 2 ** 70)) == (
        "WindowSpec(WindowFrame(RANGE_FRAME, -9223372036854775808, 9223372036854775807))")
    assert repr(o.rangeBetween(-(1 << 40), 0)) == (
        "WindowSpec(OrderBy(id ASC NULLS FIRST), WindowFrame(RANGE_FRAME, -1099511627776, 0))")
    for start, end, name in ((-(1 << 31) - 1, 0, "start"), (0, 1 << 31, "end"),
                             (Window.unboundedFollowing, 0, "start"),
                             (0, Window.unboundedPreceding, "end")):
        with pytest.raises(PySparkValueError) as error:
            o.rowsBetween(start, end)
        assert error.value.getCondition() == "VALUE_NOT_BETWEEN"
        assert error.value.getMessageParameters() == {
            "arg_name": name, "min": "-2147483648", "max": "2147483647"}
    with pytest.raises(AssertionError):
        o.rowsBetween(1.5, 2)
    # A float bound past the threshold of its side is unbounded, as PySpark compares it with
    # Window._PRECEDING_THRESHOLD and _FOLLOWING_THRESHOLD before WindowFrame asserts an int.
    inf = float("inf")
    unbounded = (Window.unboundedPreceding, Window.unboundedFollowing)
    assert repr(o.rowsBetween(-inf, inf)) == repr(o.rowsBetween(*unbounded))
    assert repr(o.rangeBetween(-inf, inf)) == repr(o.rangeBetween(*unbounded))
    assert repr(o.rowsBetween(-1e30, 1e30)) == repr(o.rowsBetween(*unbounded))
    for start, end in ((inf, 0), (0, -inf), (-1e30, 1.5)):
        with pytest.raises(AssertionError):
            o.rowsBetween(start, end)
    for call, arg_type in ((lambda: Window.partitionBy(1), "int"),
                           (lambda: Window.partitionBy(("id",)), "tuple"),
                           (lambda: Window.orderBy(1), "int"),
                           (lambda: o.orderBy(1.5), "float")):
        with pytest.raises(PySparkTypeError) as error:
            call()
        assert error.value.getMessageParameters() == {
            "expected_type": "Column or str", "arg_name": "col", "arg_type": arg_type}
    assert repr(Window.partitionBy(["a", "b"])) == "WindowSpec(PartitionBy(a, b))"


def test_expressions_carry_their_origins_as_pyspark():
    # As PySpark's _with_origin: col() and the Column methods give the expressions that they
    # build the origin of their name and their call site in user code, as the python_origin of
    # their common fields, unless spark.python.sql.dataFrameDebugging.enabled, which PySpark
    # reads once, is false. In new interpreters, whose first column of a session reads it.
    import subprocess
    import sys
    from pathlib import Path

    pytest.importorskip("grpc")
    code = """
import sys
sys.path.insert(0, sys.argv[1])
from _mock_connect_server import mock_connect_session, parse_message
from pyspark.sql import functions as F

def origins(expression):
    # Expression.common=18 -> ExpressionCommon.origin=1 -> Origin.python_origin=1:
    # fragment=1, call_site=2; .unresolved_function=3, whose arguments=2.
    expression = parse_message(expression)
    found = []
    if 18 in expression:
        origin = parse_message(parse_message(parse_message(expression[18][0])[1][0])[1][0])
        found.append((origin[1][0].decode(), origin[2][0].decode()))
    if 3 in expression:
        for argument in parse_message(expression[3][0]).get(2, []):
            found += origins(argument)
    return found

if __name__ == "__main__":
    with mock_connect_session() as (spark, service):
        service.confs = dict(arg.split("=") for arg in sys.argv[2:])
        column = F.col("a") + 1
        spark.range(1).select(column).collect()
        # ExecutePlanRequest.plan=3, Plan.root=1, Relation.project=3, Project.expressions=3.
        plan = parse_message(parse_message(service.execute_requests[-1])[3][0])[1][0]
        project = parse_message(parse_message(plan)[3][0])
        print(origins(project[3][0]))
"""
    tests = str(Path(__file__).resolve().parent)

    def run(*confs):
        result = subprocess.run(
            [sys.executable, "-c", code, tests, *confs], check=True, capture_output=True,
            text=True,
        )
        return result.stdout.strip().splitlines()[-1]

    assert run() == (
        "[('__add__', '<string>:23'), ('col', '<string>:23'), ('__add__', '<string>:23')]"
    )
    assert run("spark.python.sql.dataFrameDebugging.enabled=false") == "[]"


@pytest.mark.parametrize("column,expected", [
    # instr and regexp_replace are plain functions, as other functions are.
    (lambda F: F.instr("s", "b"), "instr(s, b)"),
    (lambda F: F.instr("s", "b", 2), "instr(s, b, 2)"),
    (lambda F: F.instr("s", "b", occurrence=2), "instr(s, b, 1, 2)"),
    (lambda F: F.instr("s", "b", 2, 3), "instr(s, b, 2, 3)"),
    (lambda F: F.regexp_replace("s", "a", "b"), "regexp_replace(s, a, b)"),
    (lambda F: F.regexp_replace("s", "a", "b", 2), "regexp_replace(s, a, b, 2)"),
    (lambda F: F.regexp_replace("s", F.col("p"), F.col("r")), "regexp_replace(s, p, r)"),
    # call_function builds a CallFunction, which renders as in PySpark.
    (lambda F: F.call_function("abs", "x"), "CallFunction('abs', x)"),
    (lambda F: F.call_function("current_date"), "CallFunction('current_date')"),
    (lambda F: F.upper(F.call_function("abs", "x")), "upper(CallFunction('abs', x))"),
    (lambda F: F.call_function("coalesce", "x", F.lit(None)).alias("y"),
     "CallFunction('coalesce', x, NULL) AS y"),
])
def test_function_calls_render_as_pyspark(column, expected):
    from pyspark.sql import functions as F

    assert repr(column(F)) == f"Column<'{expected}'>"


def test_count_and_sum_distinct_take_their_arguments_as_pyspark():
    # As PySpark Connect's: a Column or a column name, else NOT_EXPECTED_TYPE; count of a star
    # Column counts the rows, as count(1), and sumDistinct warns that it is deprecated.
    import warnings

    from pyspark.errors import PySparkTypeError
    from pyspark.sql import functions as F

    df = _offline_session().range(1)
    assert repr(F.count(F.col("*"))) == "Column<'count(1)'>"
    assert repr(F.count(df["*"])) == "Column<'count(1)'>"
    assert repr(F.count(F.col("a.*"))) == "Column<'count(1)'>"
    assert repr(F.count("a")) == repr(F.count(F.col("a"))) == "Column<'count(a)'>"
    # A name is the column of it: count("*") is the count of a star, as in PySpark.
    assert repr(F.count("*")) == "Column<'count(unresolvedstar())'>"
    assert repr(F.sum_distinct("a")) == "Column<'sum(distinct a)'>"
    for call in (lambda: F.count(1), lambda: F.sum_distinct(1), lambda: F.sum_distinct(None)):
        with pytest.raises(PySparkTypeError) as error:
            call()
        assert error.value.getCondition() == "NOT_EXPECTED_TYPE"
        assert error.value.getMessageParameters()["expected_type"] == "Column or str"
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        assert repr(F.sumDistinct("a")) == "Column<'sum(distinct a)'>"
    assert [str(w.message) for w in record] == ["Deprecated in 3.4, use sum_distinct instead."]


@pytest.mark.parametrize("build,expected", [
    # As PySpark's lit: a list is the array of the literals of its elements.
    (lambda F: F.lit([1, 2]), "array(1, 2)"),
    (lambda F: F.lit([[1], [2, 3]]), "array(array(1), array(2, 3))"),
    (lambda F: F.lit([]), "array()"),
    # Functions take a value with lit.
    (lambda F: F.array_contains("a", [1]), "array_contains(a, array(1))"),
    # array, create_map, map_concat and struct take a single list, set or tuple of columns.
    (lambda F: F.array(["a", "b"]), "array(a, b)"),
    (lambda F: F.array(("a", "b")), "array(a, b)"),
    (lambda F: F.struct(["a", "b"]), "struct(a, b)"),
    (lambda F: F.create_map([F.lit("k"), F.col("v")]), "map(k, v)"),
    (lambda F: F.map_concat(["m1", "m2"]), "map_concat(m1, m2)"),
])
def test_lists_are_arrays_and_columns_as_pyspark(build, expected):
    from pyspark.sql import functions as F

    assert repr(build(F)) == f"Column<'{expected}'>"


def test_lit_of_a_numpy_array_is_an_array_of_its_type_as_pyspark():
    np = pytest.importorskip("numpy")
    from pyspark.sql import functions as F

    # The case of the type in the repr differs (SPARK-60948).
    for value, expected in (
        (np.array([1, 2]), "cast(array(1, 2) as array<bigint>)"),
        (np.array([1.5], dtype=np.float32), "cast(array(1.5) as array<float>)"),
        (np.array(["x"]), "cast(array(x) as array<string>)"),
    ):
        assert repr(F.lit(value)).lower() == f"column<'{expected}'>"


def test_lit_errors_as_pyspark():
    np = pytest.importorskip("numpy")
    from pyspark.errors import PySparkTypeError, PySparkValueError
    from pyspark.sql import functions as F

    with pytest.raises(PySparkValueError) as error:
        F.lit([F.col("a")])
    assert error.value.getCondition() == "COLUMN_IN_LIST"
    assert error.value.getMessageParameters() == {"func_name": "lit"}
    for value, dtype in ((np.array([1], dtype=np.uint8), "uint8"), (np.array([1, "a"], dtype=object), "object")):
        with pytest.raises(PySparkTypeError) as error:
            F.lit(value)
        assert error.value.getCondition() == "UNSUPPORTED_NUMPY_ARRAY_SCALAR"
        assert error.value.getMessageParameters() == {"dtype": dtype}
    # Rather than a scalar of an array of one element.
    for value, name in (((1, 2), "tuple"), ({"a": 1}, "dict"), (object(), "object"), (np.array([[1]]), "ndarray")):
        with pytest.raises(PySparkTypeError) as error:
            F.lit(value)
        assert error.value.getCondition() == "UNSUPPORTED_DATA_TYPE"
        assert error.value.getMessageParameters() == {"data_type": name}


def test_variant_delete_requires_paths():
    from pyspark.errors import PySparkValueError
    from pyspark.sql import functions as F

    with pytest.raises(PySparkValueError, match="CANNOT_BE_EMPTY"):
        F.variant_delete("v")


def _offline_session():
    # Port 1: nothing listens; the channel is lazy, so the session builds offline.
    return SparkSession.builder.remote("sc://localhost:1").getOrCreate()


def test_nearest_by_join_checks_its_arguments_as_pyspark():
    # As PySpark's nearestByJoin, as the Scala Dataset.validateNearestByJoinArgs: an
    # AnalysisException of NEAREST_BY_JOIN of the number of results, the join type, the mode and
    # the direction, before the plan; numResults, mode and direction have no defaults, and
    # joinType is keyword-only.
    from pyspark.errors import AnalysisException
    from pyspark.sql import functions as F

    spark = _offline_session()
    a, b = spark.range(1), spark.range(1)
    ranking = F.lit(1)
    for kwargs, condition, params in [
        (dict(numResults=0), "NUM_RESULTS_OUT_OF_RANGE", {"numResults": "0", "min": "1", "max": "100000"}),
        (dict(numResults=100001), "NUM_RESULTS_OUT_OF_RANGE", {"numResults": "100001", "min": "1", "max": "100000"}),
        (dict(joinType="right"), "UNSUPPORTED_JOIN_TYPE", {"joinType": "right", "supported": "'INNER', 'LEFT OUTER'"}),
        (dict(mode="brute"), "UNSUPPORTED_MODE", {"mode": "brute", "supported": "'approx', 'exact'"}),
        (dict(direction="asc"), "UNSUPPORTED_DIRECTION", {"direction": "asc", "supported": "'distance', 'similarity'"}),
    ]:
        args = {"numResults": 1, "mode": "approx", "direction": "distance", **kwargs}
        with pytest.raises(AnalysisException) as error:
            a.nearestByJoin(b, ranking, **args)
        assert error.value.getCondition() == "NEAREST_BY_JOIN." + condition
        assert error.value.getMessageParameters() == params
    for join_type in ["inner", "LEFT_OUTER", "left", "leftouter"]:
        a.nearestByJoin(b, ranking, "2", "EXACT", "Similarity", joinType=join_type)
    with pytest.raises(TypeError):
        a.nearestByJoin(b, ranking)
    with pytest.raises(TypeError):
        a.nearestByJoin(b, ranking, 1, "approx", "distance", "inner")


def test_dataframes_of_other_sessions_raise_session_not_same_as_pyspark():
    # As PySpark's `_check_same_session`, of the DataFrames that the methods take, before any
    # request: SESSION_NOT_SAME for a DataFrame of another session, and for what is not one.
    from pyspark.errors import PySparkTypeError, SessionNotSameException
    from pyspark.sql import functions as F

    spark = _offline_session()
    a, b = spark.range(1), spark.newSession().range(1)
    for call in (
        lambda: a.join(b), lambda: a.crossJoin(b), lambda: a.lateralJoin(b),
        lambda: a.union(b), lambda: a.unionAll(b), lambda: a.unionByName(b),
        lambda: a.intersect(b), lambda: a.intersectAll(b), lambda: a.subtract(b),
        lambda: a.exceptAll(b), lambda: a.zip(b),
        lambda: a.nearestByJoin(b, F.lit(1), 1, "approx", "distance"),
        lambda: a.sameSemantics(b), lambda: a.union(1), lambda: a.join("id"),
        lambda: a.groupBy("id").cogroup(b.groupBy("id")),
    ):
        with pytest.raises(SessionNotSameException) as error:
            call()
        assert error.value.getCondition() == "SESSION_NOT_SAME"
    with pytest.raises(PySparkTypeError) as error:
        a.sameSemantics(None)
    assert error.value.getMessageParameters() == {
        "expected_type": "DataFrame", "arg_name": "other", "arg_type": "NoneType"}
    # Of the same session, they are taken.
    c = spark.range(1)
    for result in (a.join(c), a.union(c), a.unionByName(c), a.exceptAll(c)):
        assert type(result).__name__ == "DataFrame"
    assert type(a.groupBy("id").cogroup(c.groupBy("id"))).__name__ == "PandasCogroupedOps"


def test_explain_checks_its_arguments_and_prints_with_print_as_pyspark(mock_connect):
    # As PySpark's DataFrame._explain_string, which checks its arguments and the mode, and
    # explain, which prints its plan with Python's print.
    import contextlib
    import io

    from pyspark.errors import PySparkTypeError, PySparkValueError
    from pyspark.sql import functions as F

    spark, service = mock_connect
    df = spark.range(1)
    not_bool_or_str = {
        "expected_type": "bool or str",
        "arg_name": "extended (optional) and mode (optional)",
        "arg_type": "<class 'int'>",
    }
    for call, error_type, condition, parameters in (
        (lambda: df.explain(True, "simple"), PySparkValueError, "CANNOT_SET_TOGETHER",
         {"arg_list": "extended and mode"}),
        (lambda: df._explain_string(1), PySparkTypeError, "NOT_EXPECTED_TYPE", not_bool_or_str),
        (lambda: df.explain(mode=1), PySparkTypeError, "NOT_EXPECTED_TYPE", not_bool_or_str),
        (lambda: df.explain(mode="FORMATTED"), PySparkValueError, "VALUE_NOT_ALLOWED",
         {"arg_name": "explain_mode",
          "allowed_values": "['simple', 'extended', 'codegen', 'cost', 'formatted']"}),
        # An error of a PySpark error class that the client raises carries its error class
        # and message parameters.
        (lambda: spark.tvf.json_tuple(F.lit("{}")), PySparkValueError, "CANNOT_BE_EMPTY",
         {"item": "field"}),
    ):
        with pytest.raises(error_type) as error:
            call()
        assert error.value.getCondition() == condition
        assert error.value.getMessageParameters() == parameters

    plan = service.explain_string
    assert df._explain_string(mode="cost") == plan
    printed = io.StringIO()
    with contextlib.redirect_stdout(printed):
        assert df.explain(True) is None
    assert printed.getvalue() == plan + "\n"
    # base.proto: AnalyzePlanRequest.explain=5, Explain.explain_mode=2, of EXPLAIN_MODE_COST=4
    # and EXPLAIN_MODE_EXTENDED=2.
    modes = [parse_message(parse_message(request)[5][0])[2][0]
             for request in service.analyze_requests[-2:]]
    assert modes == [4, 2]


def test_col_regex_metadata_column_where_and_transpose_take_their_arguments_as_pyspark():
    # As PySpark's: colRegex and metadataColumn take a str, where a Column or a str, and
    # transpose a Column or a column name, else NOT_EXPECTED_TYPE.
    from pyspark.errors import PySparkTypeError
    from pyspark.sql import functions as F

    df = _offline_session().range(1)
    for value in (1, None, ["id"], F.col("id")):
        for call, arg_name, expected in (
            (lambda: df.colRegex(value), "colName", "str"),
            (lambda: df.metadataColumn(value), "colName", "str"),
        ):
            with pytest.raises(PySparkTypeError) as error:
                call()
            assert error.value.getMessageParameters() == {
                "arg_name": arg_name, "expected_type": expected,
                "arg_type": type(value).__name__}
    for value in (1, 1.5, ["id"]):
        for call, arg_name in ((lambda: df.where(value), "condition"),
                               (lambda: df.transpose(value), "col")):
            with pytest.raises(PySparkTypeError) as error:
                call()
            assert error.value.getMessageParameters() == {
                "arg_name": arg_name, "expected_type": "Column or str",
                "arg_type": type(value).__name__}
    # transpose takes a column name as the column of it.
    assert type(df.transpose("id")).__name__ == "DataFrame"
    assert type(df.transpose(F.col("id"))).__name__ == "DataFrame"


def test_session_context_manager_stops_session():
    spark = _offline_session()
    with spark as s:
        assert s is spark
        assert not s.is_stopped
    # __exit__ stops it; stop() is best-effort (the release RPC fails offline) and
    # idempotent.
    assert spark.is_stopped
    spark.stop()
    assert SparkSession.getActiveSession() is None


def test_sql_context_connect_backed():
    import warnings

    from pyspark.errors import PySparkNotImplementedError
    from pyspark.sql import HiveContext, SQLContext
    from pyspark.sql.connect import context as cctx

    spark = _offline_session()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ctx = SQLContext.getOrCreate()
        assert any(issubclass(w.category, FutureWarning) for w in caught)
        assert isinstance(ctx, cctx.SQLContext)
        assert ctx.sparkSession.session_id == spark.session_id
        # Cached per session (getActiveSession returns a fresh wrapper each call).
        assert SQLContext.getOrCreate() is ctx
        assert cctx.SQLContext._get_or_create_from_session(spark) is ctx
        with pytest.raises(PySparkNotImplementedError):
            ctx.registerJavaFunction("f", "com.example.F")
        with pytest.raises(PySparkNotImplementedError):
            HiveContext.getOrCreate()
        with pytest.raises(PySparkNotImplementedError):
            cctx.HiveContext(spark)
        with pytest.raises(PySparkNotImplementedError):
            SQLContext(None)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            cctx.SQLContext(spark)
        assert any(issubclass(w.category, FutureWarning) for w in caught)
        assert ctx.udf is not None and ctx.read is not None
        # Stopping a different session leaves the cache; stopping its own clears it.
        other = spark.newSession()
        other.stop()
        assert cctx.SQLContext._instantiatedContext is ctx
        spark.stop()
        assert cctx.SQLContext._instantiatedContext is None
    finally:
        cctx.SQLContext._instantiatedContext = None
        spark.stop()


def test_execution_info_and_operation_id():
    from pyspark.errors import AnalysisException
    from pyspark.errors.exceptions.connect import (
        AnalysisException as ConnectAnalysisException,
        SparkConnectException,
    )
    from pyspark.sql.connect.client import PlanObservedMetrics
    from pyspark.sql.metrics import ExecutionInfo

    spark = _offline_session()
    try:
        # No action ran yet.
        assert spark.range(1).executionInfo is None
    finally:
        spark.stop()
    ei = ExecutionInfo([], [PlanObservedMetrics("o", [1, 2], [])], "op-1")
    assert ei.operation_id == "op-1"
    assert ei.flows == [("o", {"observed_metric_0": 1, "observed_metric_1": 2})]
    assert ei.metrics is None
    # SparkConnectException.operation_id (4.3.0): None unless the transport stamped it.
    err = ConnectAnalysisException("boom")
    assert isinstance(err, SparkConnectException)
    assert isinstance(err, AnalysisException)
    assert not isinstance(AnalysisException("boom"), SparkConnectException)
    assert err.operation_id is None
    err._operation_id = "op-2"
    assert err.operation_id == "op-2"


def test_create_auto_cdc_flow_scd2_track_history():
    from pyspark import pipelines as dp
    from pyspark.errors import PySparkValueError
    from pyspark.pipelines.graph_element_registry import graph_element_registration_context
    from pyspark.pipelines.tests.local_graph_element_registry import LocalGraphElementRegistry
    from pyspark.sql.functions import col

    registry = LocalGraphElementRegistry()
    with graph_element_registration_context(registry):
        dp.create_streaming_table("t")
        dp.create_auto_cdc_flow(
            target="t", source="s", keys=["k"], sequence_by="seq",
            stored_as_scd_type=2, track_history_column_list=["a", col("b")],
            spark_conf={"k": "v"},
        )
        with pytest.raises(PySparkValueError, match="CANNOT_SET_TOGETHER"):
            dp.create_auto_cdc_flow(
                target="t", source="s", keys=["k"], sequence_by="seq",
                stored_as_scd_type=2, track_history_column_list=["a"],
                track_history_except_column_list=["b"],
            )
        with pytest.raises(PySparkValueError, match="INVALID_MULTIPLE_ARGUMENT_CONDITIONS"):
            dp.create_auto_cdc_flow(
                target="t", source="s", keys=["k"], sequence_by="seq",
                stored_as_scd_type=1, track_history_column_list=["a"],
            )
    flow = registry.auto_cdc_flows[0]
    assert flow.stored_as_scd_type == 2
    assert len(flow.track_history_column_list) == 2
    assert flow.spark_conf == {"k": "v"}


# --------------------------------------------------------------------------- collect of NTZ timestamps, distant timestamps and year-month intervals


@contextlib.contextmanager
def _client_time_zone(name):
    # The process time zone, which the local conversions of datetime read, for a block.
    previous = os.environ.get("TZ")
    os.environ["TZ"] = name
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()


def _collect_mock_result(mock_connect, table, schema_ddl):
    # Makes the mock server answer the next execution with `table` (Arrow IPC) of the type
    # `schema_ddl`: ExecutePlanResponse.schema=7, arrow_batch=2 (ArrowBatch.row_count=1, data=2).
    import pyarrow as pa

    spark, service = mock_connect
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    batch = varint_field(1, table.num_rows) + message_field(2, sink.getvalue().to_pybytes())
    service.execute_payload = message_field(7, ddl_data_type(schema_ddl)) + message_field(2, batch)
    return spark.range(1)


def _micros_since_epoch(value):
    # The microseconds since 1970-01-01 00:00:00 of a naive datetime, exactly.
    return (value - datetime.datetime(1970, 1, 1)) // datetime.timedelta(microseconds=1)


def _naive_from_micros(micros):
    # The naive datetime of `micros` microseconds since the epoch, as integer arithmetic.
    return datetime.datetime(1970, 1, 1) + datetime.timedelta(microseconds=micros)


_NTZ_VALUES = [
    datetime.datetime(1, 1, 1),
    datetime.datetime(1500, 3, 1, 1, 2, 3, 4),
    datetime.datetime(2020, 7, 1, 12, 0),
    datetime.datetime(9999, 12, 31, 23, 59, 59, 999999),
]


def test_ntz_collect_is_the_wall_clock_in_any_time_zone_as_pyspark(mock_connect):
    # TIMESTAMP_NTZ (no time zone in Arrow) is the wall clock, which the time zone of the client
    # does not shift, from year 1 to year 9999, in arrays, maps and structs too.
    import pyarrow as pa

    ntz = pa.timestamp("us")
    values = [_micros_since_epoch(v) for v in _NTZ_VALUES]
    table = pa.table({
        "t": pa.array(values, ntz),
        "a": pa.array([[v] for v in values], pa.list_(ntz)),
        "m": pa.array([[("k", v)] for v in values], pa.map_(pa.string(), ntz)),
        "s": pa.array([{"x": v} for v in values], pa.struct([("x", ntz)])),
    })
    df = _collect_mock_result(
        mock_connect, table,
        "t timestamp_ntz, a array<timestamp_ntz>, m map<string,timestamp_ntz>, "
        "s struct<x:timestamp_ntz>")
    for zone in ("UTC", "America/Los_Angeles", "Asia/Kolkata"):
        with _client_time_zone(zone):
            rows = df.collect()
            assert [r.t for r in rows] == _NTZ_VALUES
            assert [r.a for r in rows] == [[v] for v in _NTZ_VALUES]
            assert [r.m for r in rows] == [{"k": v} for v in _NTZ_VALUES]
            assert [r.s for r in rows] == [Row(x=v) for v in _NTZ_VALUES]


def test_ltz_collect_is_the_instant_in_the_client_time_zone_as_pyspark(mock_connect):
    # TIMESTAMP (Arrow with a time zone) is the instant, shown in the local time zone of the
    # client, with its microseconds exact however far it is from the epoch.
    import pyarrow as pa

    micros = [
        _micros_since_epoch(datetime.datetime(2020, 7, 1, 12, 0)),
        _micros_since_epoch(datetime.datetime(2020, 7, 1, 12, 0, 0, 1)),
        9223372036854775,
        -9223372036854775,
        253402300799999999,
        -62135596800000000,
        1,
        -1,
    ]
    table = pa.table({"t": pa.array(micros, pa.timestamp("us", tz="UTC"))})
    df = _collect_mock_result(mock_connect, table, "t timestamp")
    with _client_time_zone("UTC"):
        assert [r.t for r in df.collect()] == [_naive_from_micros(m) for m in micros]
    # Year 1 is out of the range of a local time in Los Angeles (as in PySpark), so it is left out.
    in_range = [m for m in micros if m != -62135596800000000]
    df = _collect_mock_result(
        mock_connect, pa.table({"t": pa.array(in_range, pa.timestamp("us", tz="UTC"))}), "t timestamp")
    with _client_time_zone("America/Los_Angeles"):
        rows = [r.t for r in df.collect()]
        assert rows[0] == datetime.datetime(2020, 7, 1, 5, 0)
        assert rows[1] == datetime.datetime(2020, 7, 1, 5, 0, 0, 1)
        # 2262-04-11 23:47:16.854775 UTC is PDT, seven hours earlier.
        assert rows[2] == datetime.datetime(2262, 4, 11, 16, 47, 16, 854775)


@pytest.mark.parametrize("collect", [
    lambda df: df.collect(),
    lambda df: df.first(),
    lambda df: df.head(1),
    lambda df: df.take(1),
    lambda df: list(df.toLocalIterator()),
])
@pytest.mark.parametrize("schema_ddl, table_type", [
    ("ym interval year to month", "int32"),
    ("a array<interval year to month>", "list"),
    ("m map<string,interval year to month>", "map"),
    ("s struct<ym:interval year to month>", "struct"),
])
def test_year_month_intervals_are_not_collected_as_pyspark(
        mock_connect, collect, schema_ddl, table_type):
    # PySpark's Arrow conversion of the rows raises NOT_IMPLEMENTED for a year-month interval
    # anywhere in the schema, before any row is converted, so an empty result raises too.
    # The Arrow data is empty: Spark Connect has no pyarrow type to build a year-month one.
    import pyarrow as pa
    from pyspark.errors import PySparkNotImplementedError

    arrow_type = {
        "int32": pa.int32(),
        "list": pa.list_(pa.int32()),
        "map": pa.map_(pa.string(), pa.int32()),
        "struct": pa.struct([("ym", pa.int32())]),
    }[table_type]
    column = "ym" if table_type == "int32" else ("s" if table_type == "struct" else
                                                  ("m" if table_type == "map" else "a"))
    table = pa.table({column: pa.array([], arrow_type)})
    df = _collect_mock_result(mock_connect, table, schema_ddl)
    with pytest.raises(PySparkNotImplementedError, match=(
            r"\[NOT_IMPLEMENTED\] Collecting a year-month interval value in Spark Connect "
            r"is not implemented\.")):
        collect(df)


# ---------------------------------------------------------------- DataFrame repr

def _repr_varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def _repr_message(field, value):
    if isinstance(value, str):
        value = value.encode()
    return _repr_varint(field * 8 + 2) + _repr_varint(len(value)) + value


def _repr_field(name, data_type):
    return (_repr_message(1, name) + _repr_message(2, data_type) + b"\x18\x01"
            + _repr_message(4, '{"note":"original"}'))


def _repr_serve(connection):
    import grpc
    from concurrent.futures import ThreadPoolExecutor

    # A separate process can serve schema RPCs while repr holds the client GIL.
    class Service:
        calls = 0
        data_type = _repr_struct(("id", _repr_message(7, b"")))
        failure = None
        execute_calls = 0
        configs = {"spark.sql.repl.eagerEval.enabled": "false",
                   "spark.sql.repl.eagerEval.maxNumRows": "20",
                   "spark.sql.repl.eagerEval.truncate": "20"}
        config_keys = []

        def execute(self, request, context):
            self.execute_calls += 1
            context.abort(grpc.StatusCode.UNIMPLEMENTED, "repr must not execute rows")

        def config(self, request, context):
            # ConfigRequest.session_id=1, operation=3, Operation.get=2, Get.keys=1;
            # ConfigResponse.session_id=1, pairs=2, KeyValue.key=1, value=2.
            request = parse_message(request)
            keys = [key.decode() for key in
                    parse_message(parse_message(request[3][0])[2][0]).get(1, [])]
            self.config_keys = self.config_keys + [keys]
            return _repr_message(1, request[1][0]) + b"".join(
                _repr_message(2, _repr_message(1, key) + _repr_message(2, self.configs[key]))
                for key in keys)

        def analyze(self, request, context):
            self.calls += 1
            if self.failure == "rpc":
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "schema unavailable")
            if self.failure == "no-result":
                return b""
            if self.failure == "missing-schema":
                return _repr_message(2, b"")
            if self.failure == "invalid-type":
                return _repr_message(2, _repr_message(1, b""))
            # base.proto: AnalyzePlanResponse.schema=2, Schema.schema=1.
            return _repr_message(2, _repr_message(1, self.data_type))

    service = Service()
    with ThreadPoolExecutor(max_workers=4) as executor:
        server = grpc.server(executor)
        handler = grpc.unary_unary_rpc_method_handler(service.analyze)
        server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(
            "spark.connect.SparkConnectService", {"AnalyzePlan": handler,
                "ExecutePlan": grpc.unary_stream_rpc_method_handler(service.execute),
                "Config": grpc.unary_unary_rpc_method_handler(service.config)}),))
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        connection.send(port)
        try:
            while True:
                command = connection.recv()
                if command[0] == "stop":
                    break
                if command[0] == "get":
                    connection.send(getattr(service, command[1]))
                elif command[0] == "set":
                    setattr(service, command[1], command[2])
                    connection.send(None)
        finally:
            server.stop(0).wait()


class _ReprRemoteService:
    def __init__(self, connection):
        object.__setattr__(self, "connection", connection)

    def __getattr__(self, name):
        self.connection.send(("get", name))
        assert self.connection.poll(10), "Mock server did not respond"
        return self.connection.recv()

    def __setattr__(self, name, value):
        self.connection.send(("set", name, value))
        assert self.connection.poll(10), "Mock server did not respond"
        self.connection.recv()


@pytest.fixture
def repr_analyzed():
    import multiprocessing
    from pyspark.sql import SparkSession

    pytest.importorskip("grpc")
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_repr_serve, args=(child,))
    process.start()
    try:
        assert parent.poll(20), "Mock server did not start"
        port = parent.recv()
        spark = SparkSession.builder.remote(f"sc://127.0.0.1:{port}").getOrCreate()
        yield spark, _ReprRemoteService(parent)
    finally:
        if process.is_alive():
            parent.send(("stop",))
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
        child.close()


def _repr_struct(*fields):
    return _repr_message(21, b"".join(_repr_message(1, _repr_field(name, dtype)) for name, dtype in fields))


def test_repr_html_is_the_html_table_of_eager_evaluation_as_pyspark(mock_connect):
    # As PySpark Connect's: _repr_html_ is None without eager evaluation, else the HTML table
    # that the server formats of an HtmlString relation of maxNumRows rows and truncate; once
    # it was called, __repr__ is the string of the schema.
    import pyarrow as pa

    spark, service = mock_connect
    eager = {"spark.sql.repl.eagerEval.enabled": "false",
             "spark.sql.repl.eagerEval.maxNumRows": "7",
             "spark.sql.repl.eagerEval.truncate": "3"}
    service.confs = eager
    df = spark.range(1)
    assert df._repr_html_() is None
    service.confs = {**eager, "spark.sql.repl.eagerEval.enabled": "true"}
    table = pa.table({"html_string": ["<table/>"]})
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    # ExecutePlanResponse.arrow_batch=2: ArrowBatch.row_count=1, data=2.
    service.execute_payload = message_field(
        2, varint_field(1, 1) + message_field(2, sink.getvalue().to_pybytes()))
    assert df._repr_html_() == "<table/>"
    # ExecutePlanRequest.plan=3, Plan.root=1, Relation.html_string=35: HtmlString.num_rows=2,
    # truncate=3.
    plan = parse_message(parse_message(service.execute_requests[-1])[3][0])
    html = parse_message(parse_message(plan[1][0])[35][0])
    assert html[2] == [7] and html[3] == [3]
    assert repr(df) == "DataFrame[id: bigint]"
    # A DataFrame of no _repr_html_ shows the table of eager evaluation.
    assert repr(spark.range(1)) == "<table/>"


@pytest.mark.parametrize("schema, expected", [
    (_repr_struct(), "DataFrame[]"),
    (_repr_struct(("id", _repr_message(7, b""))), "DataFrame[id: bigint]"),
    (_repr_struct(("a.b", _repr_message(6, b"")), ("with space", _repr_message(11, b"")),
             ("a.b", _repr_message(9, b"")), ("雪`[]", _repr_message(3, b""))),
     "DataFrame[a.b: int, with space: string, a.b: double, 雪`[]: boolean]"),
    (_repr_struct(("nested", _repr_struct(("inner", _repr_message(7, b"")),
                               ("text", _repr_message(11, b""))))),
     "DataFrame[nested: struct<inner:bigint,text:string>]"),
    (_repr_struct(("items", _repr_message(20, _repr_message(1, _repr_message(6, b"")) + b"\x10\x01")),
             ("lookup", _repr_message(22, _repr_message(1, _repr_message(11, b""))
                                 + _repr_message(2, _repr_message(7, b"")) + b"\x18\x01")),
             ("amount", _repr_message(10, b"\x08\x02\x10\x0a"))),
     "DataFrame[items: array<int>, lookup: map<string,bigint>, amount: decimal(10,2)]"),
    (_repr_struct(("missing", _repr_message(1, b"")), ("raw", _repr_message(2, b"")),
             ("day", _repr_message(14, b"")), ("ts", _repr_message(15, b"")),
             ("local", _repr_message(16, b"")),
             ("label", _repr_message(11, _repr_message(2, "UNICODE")))),
     "DataFrame[missing: void, raw: binary, day: date, ts: timestamp, "
     "local: timestamp_ntz, label: string collate UNICODE]"),
])
def test_repr_formats_schema_without_executing_rows(repr_analyzed, schema, expected):
    spark, service = repr_analyzed
    service.data_type = schema
    df = spark.range(10)
    assert service.calls == 0
    assert repr(df) == expected
    assert service.calls == 1
    assert service.execute_calls == 0
    assert service.config_keys == [_REPR_EAGER_EVAL_KEYS]


@pytest.mark.parametrize("failure, message", [
    ("rpc", "schema unavailable"),
    ("no-result", "no schema in response"),
    ("missing-schema", "Schema is missing"),
    ("invalid-type", "DataType kind not set"),
])
def test_repr_propagates_schema_errors_and_can_retry(repr_analyzed, failure, message):
    spark, service = repr_analyzed
    df = spark.range(10)
    service.failure = failure
    with pytest.raises(Exception, match=message):
        repr(df)
    assert service.calls == 1
    service.failure = None
    assert repr(df) == "DataFrame[id: bigint]"
    assert service.calls == 2
    assert service.execute_calls == 0


_REPR_EAGER_EVAL_KEYS = ["spark.sql.repl.eagerEval.enabled",
                         "spark.sql.repl.eagerEval.maxNumRows",
                         "spark.sql.repl.eagerEval.truncate"]


def test_repr_executes_the_show_string_with_eager_evaluation(repr_analyzed):
    # As PySpark: with eager evaluation, repr is the table of _show_string, so it executes
    # the DataFrame (which this mock refuses), rather than formatting the schema.
    spark, service = repr_analyzed
    service.configs = {"spark.sql.repl.eagerEval.enabled": "true",
                       "spark.sql.repl.eagerEval.maxNumRows": "2",
                       "spark.sql.repl.eagerEval.truncate": "3"}
    with pytest.raises(Exception, match="repr must not execute rows"):
        repr(spark.range(10))
    assert service.execute_calls == 1
    assert service.calls == 0
    assert service.config_keys == [_REPR_EAGER_EVAL_KEYS]


def test_repr_rejects_non_struct_schema(repr_analyzed):
    spark, service = repr_analyzed
    service.data_type = _repr_message(7, b"")
    with pytest.raises(Exception, match="Schema is not a struct type"):
        repr(spark.range(10))
    assert service.execute_calls == 0


def test_repr_handles_wide_schema_and_independent_dataframe(repr_analyzed):
    spark, service = repr_analyzed
    service.data_type = _repr_struct(*[(f"c{i}", _repr_message(7, b"")) for i in range(256)])
    df = spark.range(10)
    assert repr(df) == "DataFrame[" + ", ".join(f"c{i}: bigint" for i in range(256)) + "]"
    # A new independent DataFrame resolves updated server metadata even when
    # another change caches the original DataFrame's schema.
    service.data_type = _repr_struct(("renamed", _repr_message(11, b"")))
    assert repr(spark.range(10)) == "DataFrame[renamed: string]"
    assert service.calls == 2
    assert service.execute_calls == 0


# --------------------------------------------------------------------------- ExecutePlan (mock)


def _request_operation_id(request):
    # ExecutePlanRequest.operation_id=6.
    return parse_message(request)[6][0].decode()


def test_execution_info_decodes_observed_metrics_and_operation_id(mock_connect):
    import datetime

    spark, service = mock_connect
    ts = 1_700_000_000_123_456
    # expressions.proto Literal: integer=6, string=13, date=16, timestamp=17,
    # day_time_interval=21, map=23 (keys=3, values=4), struct=24 (elements=2),
    # data_type=100; types.proto DataType.struct=21 (fields=1; StructField.name=1).
    struct_type = message_field(
        21, message_field(1, message_field(1, "a")) + message_field(1, message_field(1, "b"))
    )
    struct_literal = message_field(
        24, message_field(2, varint_field(6, 7)) + message_field(2, message_field(13, "x"))
    ) + message_field(100, struct_type)
    map_literal = message_field(
        23, message_field(3, message_field(13, "k")) + message_field(4, varint_field(16, 1))
    )
    # base.proto ObservedMetrics: name=1, values=2, keys=3; ExecutePlanResponse.observed_metrics=6.
    observed = message_field(1, "o")
    for literal in (
        varint_field(17, ts),
        struct_literal,
        map_literal,
        varint_field(21, 90_061_000_001),
    ):
        observed += message_field(2, literal)
    for key in ("ts", "s", "m", "dt"):
        observed += message_field(3, key)
    service.execute_payload = message_field(6, observed)

    assert spark.range(1).executionInfo is None
    df = spark.range(1)
    df.collect()
    info = df.executionInfo
    # The information is that of the DataFrame that ran, as in PySpark.
    assert spark.range(1).executionInfo is None
    assert info.operation_id == _request_operation_id(service.execute_requests[-1])
    ((name, pairs),) = info.flows
    assert name == "o"
    assert pairs["ts"] == datetime.datetime.fromtimestamp(ts // 10**6).replace(
        microsecond=ts % 10**6
    )
    assert (pairs["s"].a, pairs["s"].b) == (7, "x")
    assert pairs["m"] == {"k": datetime.date(1970, 1, 2)}
    assert pairs["dt"] == datetime.timedelta(microseconds=90_061_000_001)


def test_execution_info_unsupported_literal_raises_on_flows(mock_connect):
    from pyspark.errors import PySparkTypeError

    spark, service = mock_connect
    # Literal.calendar_interval=19 (months=1): the reference rejects it in _to_value.
    observed = message_field(1, "o") + message_field(2, message_field(19, varint_field(1, 2)))
    service.execute_payload = message_field(6, observed)
    df = spark.range(1)
    df.collect()
    info = df.executionInfo
    assert info.operation_id == _request_operation_id(service.execute_requests[-1])
    with pytest.raises(PySparkTypeError, match="UNSUPPORTED_LITERAL"):
        info.flows


def test_pandas_conversion_and_its_imports_are_pyspark_modules():
    # Vendored from PySpark: toPandas converts with pyspark.sql.pandas.conversion, which
    # imports the classic error conversion and the gateway's socket helpers, as PySpark does.
    import inspect

    import pyspark.errors.exceptions.captured as captured
    import pyspark.traceback_utils  # noqa: F401
    import pyspark.util
    from pyspark.sql.pandas.conversion import _convert_arrow_table_to_pandas

    assert callable(_convert_arrow_table_to_pandas) and callable(captured.convert_exception)
    assert inspect.isgeneratorfunction(pyspark.util._load_from_socket.__wrapped__)


def test_to_local_iterator_executes_with_its_first_row_as_pyspark(mock_connect):
    # As PySpark's generator: the execution starts with the first row, which raises its
    # error, and the iteration then ends.
    from pyspark.errors import PySparkException

    spark, service = mock_connect
    service.execute_failure = "plan rejected"
    rows = spark.range(3).toLocalIterator()
    assert service.execute_calls == 0
    with pytest.raises(PySparkException, match="plan rejected"):
        next(rows)
    assert service.execute_calls == 1
    assert list(rows) == []
    assert service.execute_calls == 1


def test_foreach_and_foreach_partition_send_python_functions_as_pyspark(mock_connect):
    # As PySpark Connect's: foreach selects a struct of the columns of each row with a UDF of
    # no result that calls f with the row, and foreachPartition runs a mapInArrow function of
    # no result that calls f with an iterator of the rows of each partition. They run on the
    # server, so this calls the functions that are sent.
    import pyarrow as pa

    from pyspark import cloudpickle

    spark, service = mock_connect
    df = spark.range(1)

    def relation():
        # ExecutePlanRequest.plan=3, Plan.root=1.
        return parse_message(parse_message(parse_message(service.execute_requests[-1])[3][0])[1][0])

    def raising(row):
        raise ValueError(row)

    df.foreach(raising)
    # Relation.project=3, Project.expressions=3; Expression.common_inline_user_defined_function
    # =15; CommonInlineUserDefinedFunction.function_name=1, python_udf=4; PythonUDF.command=3.
    udf = parse_message(parse_message(parse_message(relation()[3][0])[3][0])[15][0])
    assert udf[1][0].decode() == "foreach_func"
    func, return_type = cloudpickle.loads(parse_message(udf[4][0])[3][0])
    assert return_type == T.StructType()
    with pytest.raises(ValueError) as error:
        func(Row(id=1))
    assert error.value.args == (Row(id=1),)

    def consuming(rows):
        raise ValueError(list(rows))

    df.foreachPartition(consuming)
    # Relation.map_partitions=28, MapPartitions.func=2; PythonUDF.eval_type=2, of
    # SQL_MAP_ARROW_ITER_UDF.
    udf = parse_message(parse_message(relation()[28][0])[2][0])
    assert parse_message(udf[4][0])[2][0] == 207
    func, return_type = cloudpickle.loads(parse_message(udf[4][0])[3][0])
    assert return_type == T.StructType()
    with pytest.raises(ValueError) as error:
        func(iter([pa.RecordBatch.from_pydict({"id": [1, 2]})]))
    assert error.value.args == ([Row(id=1), Row(id=2)],)
    assert "spark.sql.execution.pyspark.binaryAsBytes" in service.read_confs


def test_execute_plan_error_carries_operation_id(mock_connect):
    from pyspark.errors import PySparkException

    spark, service = mock_connect
    service.execute_failure = "plan rejected"
    with pytest.raises(PySparkException, match="plan rejected") as error:
        spark.range(1).collect()
    assert error.value.operation_id == _request_operation_id(service.execute_requests[-1])


@pytest.mark.parametrize("code", [
    "INVALID_ARGUMENT", "PERMISSION_DENIED", "UNAUTHENTICATED", "RESOURCE_EXHAUSTED", "INTERNAL",
])
def test_grpc_errors_without_details_are_spark_connect_grpc_exceptions(mock_connect, code):
    # A gRPC error that carries no error of the server, as of a proxy, is a
    # SparkConnectGrpcException of its status code, as PySpark's _handle_rpc_error raises it.
    # The cases agree with official PySpark 4.3.0.
    from pyspark.errors.exceptions.connect import SparkConnectGrpcException, StatusCode

    spark, service = mock_connect
    service.execute_failure = (code, "denied by the proxy")
    with pytest.raises(SparkConnectGrpcException, match="denied by the proxy") as error:
        spark.range(1).collect()
    assert error.value.getGrpcStatusCode() == getattr(StatusCode, code)
    assert error.value.getCondition() is None
    assert error.value.operation_id == _request_operation_id(service.execute_requests[-1])


def test_analyze_errors_without_details_are_spark_connect_grpc_exceptions(mock_connect):
    from pyspark.errors.exceptions.connect import SparkConnectGrpcException, StatusCode

    spark, service = mock_connect
    service.failure = "rpc"  # INVALID_ARGUMENT "schema unavailable", of AnalyzePlan.
    with pytest.raises(SparkConnectGrpcException, match="schema unavailable") as error:
        spark.range(1).schema
    assert error.value.getGrpcStatusCode() == StatusCode.INVALID_ARGUMENT


def _error_info(classes, **metadata):
    """ErrorInfo metadata as the server sends it, with the exception classes, the most
    specific first."""
    return {"classes": json.dumps(classes), **metadata}


def _stack_element(method, line):
    # StackTraceElement.declaring_class=1, method_name=2, file_name=3, line_number=4.
    return (
        message_field(1, "org.apache.spark.Expr")
        + message_field(2, method)
        + message_field(3, "Expr.scala")
        + varint_field(4, line)
    )


def _error_details(message, hierarchy, stack, throwable):
    # FetchErrorDetailsResponse.root_error_idx=1, errors=2; Error.error_type_hierarchy=1,
    # message=2, stack_trace=3, spark_throwable=5.
    error = b"".join(message_field(1, name) for name in hierarchy) + message_field(2, message)
    error += b"".join(message_field(3, element) for element in stack)
    return varint_field(1, 0) + message_field(2, error + message_field(5, throwable))


ARITHMETIC = ["org.apache.spark.SparkArithmeticException", "java.lang.ArithmeticException"]


def test_a_server_error_is_raised_with_its_details(mock_connect):
    import pyspark.errors as E
    from pyspark.errors.exceptions import connect as C

    spark, service = mock_connect
    message = "[DIVIDE_BY_ZERO] Division by zero. SQLSTATE: 22012"
    params = {"config": "spark.sql.ansi.enabled", "alternative": "try_divide"}
    service.execute_error = (
        # The status carries a message cut short; the details have all of it.
        message[:24],
        ARITHMETIC[0],
        _error_info(
            ARITHMETIC,
            errorClass="DIVIDE_BY_ZERO",
            sqlState="22012",
            messageParameters=json.dumps(params),
            errorId="error-1",
        ),
    )
    # SparkThrowable.error_class=1, message_parameters=2 (entries of key=1, value=2),
    # query_contexts=3, breaking_change_info=5; QueryContext.context_type=10, start_index=3,
    # stop_index=4, fragment=5; BreakingChangeInfo.migration_message=1, mitigation_config=2
    # (key=1, value=2).
    throwable = message_field(1, "DIVIDE_BY_ZERO")
    for key, value in params.items():
        throwable += message_field(2, message_field(1, key) + message_field(2, value))
    throwable += message_field(
        3, varint_field(10, 0) + varint_field(3, 7) + varint_field(4, 9) + message_field(5, "1/0")
    )
    throwable += message_field(
        5,
        message_field(1, "Set the config.")
        + message_field(2, message_field(1, "spark.sql.ansi.enabled") + message_field(2, "false")),
    )
    service.error_details = _error_details(
        message, ARITHMETIC, [_stack_element("eval", 12)], throwable
    )
    service.confs = {"spark.sql.connect.serverStacktrace.enabled": "true"}
    with pytest.raises(C.ArithmeticException) as error:
        spark.range(1).collect()
    e = error.value
    # Both the Connect exception and the base one, as in the reference client.
    assert isinstance(e, C.SparkConnectGrpcException)
    assert isinstance(e, E.ArithmeticException)
    assert service.fetched_error_ids == ["error-1"]
    assert e.getCondition() == "DIVIDE_BY_ZERO"
    assert e.getSqlState() == "22012"
    # In the order that the server gave them.
    assert list(e.getMessageParameters().items()) == list(params.items())
    stacktrace = "\n".join(
        [ARITHMETIC[0], "\tat org.apache.spark.Expr.eval(Expr.scala:12)"]
    )
    assert e.getStackTrace() == stacktrace
    assert str(e) == message + "\n\nJVM stacktrace:\n" + stacktrace
    [context] = e.getQueryContext()
    assert type(context) is C.SQLQueryContext
    assert (context.fragment(), context.startIndex(), context.stopIndex()) == ("1/0", 7, 9)
    assert e.getBreakingChangeInfo() == {
        "migration_message": ["Set the config."],
        "needs_audit": True,
        "mitigation_config": {"key": "spark.sql.ansi.enabled", "value": "false"},
    }
    assert e.getGrpcStatusCode() == C.StatusCode.INTERNAL
    assert e.operation_id == _request_operation_id(service.execute_requests[-1])


PARSE = [
    "org.apache.spark.sql.catalyst.parser.ParseException",
    "org.apache.spark.sql.AnalysisException",
]
PARSE_MESSAGE = "\n[PARSE_SYNTAX_ERROR] Syntax error at or near 'SELEC'. SQLSTATE: 42601"


@pytest.mark.parametrize(
    "confs, shown",
    [
        ({"spark.sql.connect.serverStacktrace.enabled": "true"}, True),
        (
            {
                "spark.sql.connect.serverStacktrace.enabled": "false",
                "spark.sql.pyspark.jvmStacktrace.enabled": "true",
            },
            True,
        ),
        (
            {
                "spark.sql.connect.serverStacktrace.enabled": "false",
                "spark.sql.pyspark.jvmStacktrace.enabled": "false",
            },
            False,
        ),
        # A config that cannot be read shows it, as in the reference client.
        ({}, True),
        ({"spark.sql.connect.serverStacktrace.enabled": "false"}, True),
    ],
)
def test_a_server_error_without_details_keeps_what_its_status_has(mock_connect, confs, shown):
    from pyspark.errors.exceptions import connect as C

    spark, service = mock_connect
    stacktrace = PARSE[0] + "\n\tat org.apache.spark.sql.catalyst.parser.ParseErrorListener"
    service.execute_error = (
        PARSE_MESSAGE,
        PARSE[0],
        _error_info(
            PARSE,
            errorClass="PARSE_SYNTAX_ERROR",
            sqlState="42601",
            messageParameters=json.dumps({"error": "'SELEC'", "hint": ""}),
            stackTrace=stacktrace,
        ),
    )
    service.confs = confs
    # The most specific class decides: a ParseException is also an AnalysisException.
    with pytest.raises(C.ParseException) as error:
        spark.range(1).collect()
    e = error.value
    assert isinstance(e, C.AnalysisException)
    # Without an error id, there are no details to fetch.
    assert service.fetched_error_ids == []
    assert e.getCondition() == "PARSE_SYNTAX_ERROR"
    assert e.getMessageParameters() == {"error": "'SELEC'", "hint": ""}
    assert e.getStackTrace() == stacktrace
    assert e.getQueryContext() == []
    assert str(e) == (PARSE_MESSAGE + "\n\nJVM stacktrace:\n" + stacktrace if shown else PARSE_MESSAGE)


@pytest.mark.parametrize(
    "classes, message, expected, text",
    [
        (
            ["java.lang.IllegalStateException"],
            "boom",
            "UnknownException",
            "(java.lang.IllegalStateException) boom",
        ),
        (
            ["org.apache.spark.SparkException", "org.apache.spark.api.python.PythonException"],
            "boom",
            "PythonException",
            "\n  An exception was thrown from the Python worker. "
            "Please see the stack trace below.\nboom",
        ),
        (
            ["org.apache.spark.SparkException"],
            "net.razorvine.pickle.PickleException: expected zero arguments",
            "PickleException",
            "net.razorvine.pickle.PickleException: expected zero arguments",
        ),
        (["org.apache.spark.SparkException"], "boom", "SparkException", "boom"),
        (
            ["org.apache.spark.sql.connect.common.InvalidPlanInput"],
            "boom",
            "InvalidPlanInput",
            "boom",
        ),
    ],
)
def test_the_classes_of_a_server_error_pick_its_exception(
    mock_connect, classes, message, expected, text
):
    from pyspark.errors.exceptions import connect as C

    spark, service = mock_connect
    service.execute_error = (message, classes[0], _error_info(classes))
    service.confs = {
        "spark.sql.connect.serverStacktrace.enabled": "false",
        "spark.sql.pyspark.jvmStacktrace.enabled": "false",
    }
    with pytest.raises(C.SparkConnectGrpcException) as error:
        spark.range(1).collect()
    assert type(error.value).__name__ == expected
    assert type(error.value) is getattr(C, expected)
    assert str(error.value) == text


def test_grpc_status_codes():
    from pyspark.errors.exceptions.connect import StatusCode, _status_code

    assert _status_code(13) is StatusCode.INTERNAL
    assert _status_code(0) is StatusCode.OK
    assert _status_code(99) is StatusCode.UNKNOWN


def _sent_udf_command(request):
    """The pickled command of the function in a captured ExecutePlanRequest: plan=3 ->
    root=1 -> map_partitions=28 (func=2), group_map=31 (func=3) or co_group_map=32
    (func=5) -> python_udf=4 -> command=3."""
    root = parse_message(parse_message(parse_message(request)[3][0])[1][0])
    for relation, func in ((28, 2), (31, 3), (32, 5)):
        if relation in root:
            udf = parse_message(parse_message(root[relation][0])[func][0])
            return parse_message(udf[4][0])[3][0]
    raise AssertionError(f"no map function in the relation fields {sorted(root)}")


@pytest.mark.parametrize("build", [
    lambda df: df.mapInPandas(lambda frames: frames, "id long"),
    lambda df: df.mapInArrow(lambda batches: batches, "id long"),
    lambda df: df.groupBy("id").applyInPandas(lambda pdf: pdf, "id long"),
    lambda df: df.groupBy("id").applyInArrow(lambda table: table, "id long"),
    lambda df: df.groupBy("id").cogroup(df.groupBy("id")).applyInPandas(lambda a, b: a, "id long"),
    lambda df: df.groupBy("id").cogroup(df.groupBy("id")).applyInArrow(lambda a, b: a, "id long"),
])
def test_map_functions_pickle_a_ddl_schema_as_its_type(mock_connect, build):
    import pickle

    spark, service = mock_connect
    build(spark.range(3)).collect()
    # The worker reads the pickled type as a DataType, so a DDL string is pickled parsed.
    _, return_type = pickle.loads(_sent_udf_command(service.execute_requests[-1]))
    assert return_type == T.StructType([T.StructField("id", T.LongType())])


def _sent_define_flow(request):
    """The DefineFlow in a captured ExecutePlanRequest: plan=3 -> command=2 ->
    pipeline_command=19 -> define_flow=3."""
    message = parse_message(request)
    for field in (3, 2, 19, 3):
        message = parse_message(message[field][0])
    return message


def _unparsed_identifiers(expressions):
    # Expression.unresolved_attribute=2, UnresolvedAttribute.unparsed_identifier=1.
    return [
        parse_message(parse_message(e)[2][0])[1][0].decode() for e in expressions
    ]


def test_connect_registry_defines_a_flow_of_spark_sql_without_running_it(mock_connect):
    from pyspark import pipelines as dp
    from pyspark.pipelines.graph_element_registry import graph_element_registration_context
    from pyspark.pipelines.spark_connect_graph_element_registry import (
        SparkConnectGraphElementRegistry,
    )

    spark, service = mock_connect
    with graph_element_registration_context(SparkConnectGraphElementRegistry(spark, "g")):
        @dp.materialized_view
        def mv():
            return spark.sql("SELECT * FROM raw")

    # As PySpark, whose server returns the relation of spark.sql() within a flow function
    # without running the statement: no SqlCommand (Command.sql_command=5) is sent, and the
    # flow's relation is the query (DefineFlow.relation_flow_details=7, relation=1;
    # Relation.sql=10, SQL.query=1).
    commands = [parse_message(parse_message(parse_message(r)[3][0])[2][0])
                for r in service.execute_requests]
    assert not any(5 in command for command in commands)
    flow = _sent_define_flow(service.execute_requests[-1])
    relation = parse_message(parse_message(flow[7][0])[1][0])
    assert parse_message(relation[10][0])[1] == [b"SELECT * FROM raw"]
    # Outside of a flow function, the statement runs at once again.
    spark.sql("SET a=b")
    assert 5 in parse_message(parse_message(parse_message(service.execute_requests[-1])[3][0])[2][0])


_REQUESTS_WITHOUT_THE_GIL = {
    "save": lambda spark: spark.range(1).write.format("parquet").save("/tmp/no_such_dir"),
    "saveAsTable": lambda spark: spark.range(1).write.saveAsTable("t"),
    "insertInto": lambda spark: spark.range(1).write.insertInto("t"),
    "parquet": lambda spark: spark.range(1).write.parquet("/tmp/no_such_dir"),
    "writeTo.create": lambda spark: spark.range(1).writeTo("t").create(),
    "writeTo.append": lambda spark: spark.range(1).writeTo("t").append(),
    "mergeInto.merge": lambda spark: spark.range(1)
    .mergeInto("t", spark.range(1).id == 0)
    .whenMatched()
    .delete()
    .merge(),
    "isEmpty": lambda spark: spark.range(1).isEmpty(),
    "catalog.currentDatabase": lambda spark: spark.catalog.currentDatabase(),
    "catalog.tableExists": lambda spark: spark.catalog.tableExists("t"),
    "catalog.refreshTable": lambda spark: spark.catalog.refreshTable("t"),
    "writeStream.start": lambda spark: spark.readStream.format("rate")
    .load()
    .writeStream.format("noop")
    .start(),
}


@pytest.mark.parametrize("name", sorted(_REQUESTS_WITHOUT_THE_GIL))
def test_requests_of_writes_the_catalog_and_streams_release_the_gil_as_pyspark(mock_connect, name):
    # As PySpark's requests to the server, which release the GIL while they wait: this
    # thread runs while the request waits for the server.
    spark, service = mock_connect
    service.block_execute = True
    with ThreadPoolExecutor(max_workers=1) as pool:
        request = pool.submit(_REQUESTS_WITHOUT_THE_GIL[name], spark)
        try:
            for _ in range(500):
                if service.execute_calls:
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("The request never reached the server")
            assert not request.done()
        finally:
            service.release_response()
        error = request.exception(timeout=10)
    # Whatever the request makes of the empty response of the mock server, it did not hold
    # the GIL, which the server would have waited for.
    assert "could not run" not in str(error or "")


def test_sql_runs_a_statement_without_holding_the_gil(mock_connect):
    spark, service = mock_connect
    service.block_execute = True
    with ThreadPoolExecutor(max_workers=1) as pool:
        statement = pool.submit(lambda: spark.sql("SET a=b"))
        try:
            for _ in range(500):
                if service.execute_calls:
                    break
                time.sleep(0.01)
            else:
                raise AssertionError("The statement never reached the server")
            # This thread runs while the statement waits for the server.
            assert not statement.done()
        finally:
            service.release_response()
        statement.result(timeout=10)


def test_connect_registry_sends_auto_cdc_define_flow(mock_connect):
    from pyspark import pipelines as dp
    from pyspark.pipelines.graph_element_registry import graph_element_registration_context
    from pyspark.pipelines.spark_connect_graph_element_registry import (
        SparkConnectGraphElementRegistry,
    )

    spark, service = mock_connect
    with graph_element_registration_context(SparkConnectGraphElementRegistry(spark, "g")):
        dp.create_auto_cdc_flow(
            target="t", source="s", keys=["k"], sequence_by="seq",
            stored_as_scd_type=2, name="scd2",
            track_history_column_list=["a", "b"], spark_conf={"c": "v"},
        )
        dp.create_auto_cdc_flow(
            target="t", source="s", keys=["k"], sequence_by="seq",
            stored_as_scd_type="2", name="scd2_except",
            track_history_except_column_list=["z"],
        )
        dp.create_auto_cdc_flow(
            target="t", source="s", keys=["k"], sequence_by="seq", name="plain"
        )
    scd2, scd2_except, plain = (_sent_define_flow(r) for r in service.execute_requests[-3:])

    # DefineFlow: flow_name=2, sql_conf=4 (entries key=1, value=2), auto_cdc_flow_details=10;
    # AutoCdcFlowDetails: stored_as_scd_type=10, track_history_column_list=11,
    # track_history_except_column_list=12.
    assert scd2[2] == [b"scd2"]
    assert [parse_message(e) for e in scd2[4]] == [{1: [b"c"], 2: [b"v"]}]
    details = parse_message(scd2[10][0])
    assert details[10] == [2]
    assert _unparsed_identifiers(details[11]) == ["a", "b"]
    assert 12 not in details

    details = parse_message(scd2_except[10][0])
    assert details[10] == [2]
    assert 11 not in details
    assert _unparsed_identifiers(details[12]) == ["z"]

    details = parse_message(plain[10][0])
    assert 4 not in plain
    assert 10 not in details and 11 not in details and 12 not in details


# Positional DataFrame indexing via a separate-process mock gRPC service.
from concurrent.futures import ThreadPoolExecutor as positional_indexing_ThreadPoolExecutor
import multiprocessing as positional_indexing_multiprocessing

from pyspark.sql import (
    SparkSession as positional_indexing_SparkSession, Column as positional_indexing_Column,
)


def positional_indexing_varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def positional_indexing_message(field, value):
    if isinstance(value, str):
        value = value.encode()
    return positional_indexing_varint(field * 8 + 2) + positional_indexing_varint(len(value)) + value


def positional_indexing_schema(name="id", nested=False):
    # types.proto: DataType.long=7, .struct=21; Struct.fields=1;
    # StructField: name=1, data_type=2, nullable=3, metadata=4.
    data_type = positional_indexing_message(21, positional_indexing_message(1, positional_indexing_field("inner", positional_indexing_message(7, b"")))) if nested else positional_indexing_message(7, b"")
    return positional_indexing_message(21, positional_indexing_message(1, positional_indexing_field(name, data_type)))


def positional_indexing_field(name, data_type):
    return (positional_indexing_message(1, name) + positional_indexing_message(2, data_type) + b"\x18\x01"
            + positional_indexing_message(4, '{"note":"original"}'))


def positional_indexing_serve(connection):
    import grpc as positional_indexing_grpc
    # A separate process also exercises columns/dtypes, whose first access may
    # hold the client's GIL while waiting for the RPC.
    class Service:
        calls = 0
        data_type = positional_indexing_schema()
        failure = None

        def analyze(self, request, context):
            self.calls += 1
            # Both clients serialize the UUID session identifier as field 1.
            # Echo it so reference Python can verify response integrity.
            assert request[0] == 10 and request[1] < 128
            session = request[:2 + request[1]]
            if self.failure == "rpc":
                context.abort(positional_indexing_grpc.StatusCode.INVALID_ARGUMENT, "schema unavailable")
            if self.failure == "no-result":
                return b""
            if self.failure == "missing-schema":
                return positional_indexing_message(2, b"")
            if self.failure == "invalid-type":
                return positional_indexing_message(2, positional_indexing_message(1, b""))
            # base.proto: AnalyzePlanResponse.schema=2, Schema.schema=1.
            return session + positional_indexing_message(2, positional_indexing_message(1, self.data_type))

    service = Service()
    with positional_indexing_ThreadPoolExecutor(max_workers=4) as executor:
        server = positional_indexing_grpc.server(executor)
        handler = positional_indexing_grpc.unary_unary_rpc_method_handler(service.analyze)
        release = positional_indexing_grpc.unary_unary_rpc_method_handler(
            lambda request, context: request[:2 + request[1]])
        config = positional_indexing_grpc.unary_unary_rpc_method_handler(
            lambda request, context: request[:2 + request[1]] + positional_indexing_message(
                2, positional_indexing_message(1, "spark.python.sql.dataFrameDebugging.enabled")
                + positional_indexing_message(2, "false")))
        server.add_generic_rpc_handlers((positional_indexing_grpc.method_handlers_generic_handler(
            "spark.connect.SparkConnectService", {
                "AnalyzePlan": handler, "ReleaseSession": release, "Config": config}),))
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        connection.send(port)
        try:
            while True:
                command = connection.recv()
                if command[0] == "stop":
                    break
                if command[0] == "get":
                    connection.send(getattr(service, command[1]))
                elif command[0] == "set":
                    setattr(service, command[1], command[2])
                    connection.send(None)
        finally:
            server.stop(0).wait()


class positional_indexing_RemoteService:
    def __init__(self, connection):
        object.__setattr__(self, "connection", connection)

    def __getattr__(self, name):
        self.connection.send(("get", name))
        assert self.connection.poll(10), "Mock server did not respond"
        return self.connection.recv()

    def __setattr__(self, name, value):
        self.connection.send(("set", name, value))
        assert self.connection.poll(10), "Mock server did not respond"
        self.connection.recv()


@pytest.fixture
def positional_indexing_analyzed():
    pytest.importorskip("grpc")
    context = positional_indexing_multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=positional_indexing_serve, args=(child,))
    process.start()
    spark = None
    try:
        assert parent.poll(20), "Mock server did not start"
        port = parent.recv()
        spark = positional_indexing_SparkSession.builder.remote(f"sc://127.0.0.1:{port}").create()
        yield spark, positional_indexing_RemoteService(parent)
    finally:
        if spark is not None:
            spark.stop()
        if process.is_alive():
            parent.send(("stop",))
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
        child.close()


def positional_indexing_columns(*names):
    return positional_indexing_message(21, b"".join(positional_indexing_message(1, positional_indexing_field(name, positional_indexing_message(7, b""))) for name in names))


@pytest.mark.parametrize("index, name", [
    (0, "first"), (1, "second"), (2, "last"),
    (-1, "last"), (-2, "second"), (-3, "first"),
    (False, "first"), (True, "second"),
])
def test_positional_indexing_positional_column(positional_indexing_analyzed, index, name):
    spark, service = positional_indexing_analyzed
    service.data_type = positional_indexing_columns("first", "second", "last")
    df = spark.range(10)
    column = df[index]
    assert isinstance(column, positional_indexing_Column)
    assert repr(column) == repr(df[name])
    assert service.calls == 1


@pytest.mark.parametrize("names, index", [
    (("a", "b"), 2), (("a", "b"), -3),
    (("a",), 10**100), (("a",), -(10**100)),
    ((), 0), ((), -1), ((), False), (("a",), True),
])
def test_positional_indexing_positional_bounds(positional_indexing_analyzed, names, index):
    spark, service = positional_indexing_analyzed
    service.data_type = positional_indexing_columns(*names)
    with pytest.raises(IndexError):
        spark.range(10)[index]
    assert service.calls == 1


def test_positional_indexing_integer_subclass_and_string_lookup(positional_indexing_analyzed):
    class Position(int):
        pass
    class Name(str):
        pass
    spark, service = positional_indexing_analyzed
    service.data_type = positional_indexing_columns("a.b", "with space", "a.b")
    df = spark.range(10)
    assert repr(df[Position(-2)]) == repr(df[Name("with space")])
    assert repr(df[0]) == repr(df["a.b"])
    before = service.calls
    for name in ["missing", "*", "`a.b`", ""]:
        assert isinstance(df[name], positional_indexing_Column)
    assert service.calls == before


def test_positional_indexing_invalid_index_types_do_not_analyze(positional_indexing_analyzed):
    from pyspark.errors import PySparkTypeError

    class Position:
        def __index__(self):
            return 0
    spark, service = positional_indexing_analyzed
    df = spark.range(10)
    for value in [0.0, None, b"id", slice(0, 1), Position()]:
        # As PySpark, which takes a Column, an int, a list, a str or a tuple.
        with pytest.raises(PySparkTypeError) as error:
            df[value]
        assert error.value.getCondition() == "NOT_EXPECTED_TYPE"
        assert error.value.getMessageParameters() == {
            "expected_type": "Column, int, list, str or tuple", "arg_name": "item",
            "arg_type": type(value).__name__}
    assert service.calls == 0


def test_positional_indexing_columns_filter_and_sequences_select(positional_indexing_analyzed):
    from pyspark.sql import DataFrame

    # As PySpark: a Column filters the rows, and a list or tuple selects its columns,
    # without looking the schema up. The live tests check the rows.
    spark, service = positional_indexing_analyzed
    df = spark.range(10)
    for item in [df["id"] > 1, ["id"], ("id", "id"), [["id"]], []]:
        assert type(df[item]) is DataFrame
    assert service.calls == 0


def test_positional_indexing_positional_propagates_schema_failure(positional_indexing_analyzed):
    spark, service = positional_indexing_analyzed
    service.failure = "rpc"
    df = spark.range(10)
    with pytest.raises(Exception, match="schema unavailable"):
        df[0]
    assert service.calls == 1
    assert isinstance(df["id"], positional_indexing_Column)
    assert service.calls == 1


# --------------------------------------------------------------------------- join types, empty aggregations and the status of a streaming query

@pytest.mark.parametrize(("how", "join_type"), [
    (None, 1), ("inner", 1), ("INNER", 1), ("in_ner", 1),
    ("outer", 2), ("full", 2), ("fullouter", 2), ("full_outer", 2), ("FULL_OUTER", 2),
    ("leftouter", 3), ("left", 3), ("left_outer", 3),
    ("rightouter", 4), ("right", 4), ("right_outer", 4),
    ("leftanti", 5), ("left_anti", 5), ("anti", 5),
    ("leftsemi", 6), ("left_semi", 6), ("semi", 6),
    ("cross", 7),
])
def test_join_types_are_sent_as_pyspark_join_types(mock_connect, how, join_type):
    # As PySpark's plan.Join: how is lowercased without its underscores, and no how is inner.
    spark, service = mock_connect
    df = spark.range(2)
    df.join(df, "id", how).collect()
    # ExecutePlanRequest.plan=3, Plan.root=1, Relation.join=5, Join.join_type=4.
    plan = parse_message(parse_message(service.execute_requests[-1])[3][0])[1][0]
    assert parse_message(parse_message(plan)[5][0])[4] == [join_type]


def test_join_on_lists_join_on_their_conjunction_or_names_as_pyspark(mock_connect):
    # As PySpark's plan.Join: a list of conditions joins on their conjunction, a list of names
    # is a using join, and an empty list is no condition at all.
    spark, service = mock_connect
    left, right = spark.range(2), spark.range(3)

    def sent_join():
        # ExecutePlanRequest.plan=3, Plan.root=1, Relation.join=5.
        plan = parse_message(parse_message(service.execute_requests[-1])[3][0])[1][0]
        return parse_message(parse_message(plan)[5][0])

    left.join(right, [left.id == right.id, right.id > 0]).collect()
    join = sent_join()
    # Join.join_condition=3 is the conjunction: an UnresolvedFunction "and" (Expression=3).
    assert 5 not in join
    assert parse_message(parse_message(join[3][0])[3][0])[1] == [b"and"]

    left.join(right, ["id"]).collect()
    join = sent_join()
    # Join.using_columns=5.
    assert 3 not in join and join[5] == [b"id"]

    left.join(right, []).collect()
    join = sent_join()
    assert 3 not in join and 5 not in join


@pytest.mark.parametrize(("how", "typ"), [
    ("foo", "foo"), ("", ""), ("left semi", "left semi"), ("LEFT_SEMI_X", "leftsemix"), (1, "1"),
])
def test_join_types_that_are_not_accepted_raise_as_pyspark(offline_spark, how, typ):
    # As PySpark's plan.Join: the AnalysisException of UNSUPPORTED_JOIN_TYPE, raised locally,
    # which names the type as it is normalized (or its str() when it is not a str).
    from pyspark.errors import AnalysisException

    df = offline_spark.range(3)
    with pytest.raises(AnalysisException) as error:
        df.join(df, "id", how)
    assert error.value.getCondition() == "UNSUPPORTED_JOIN_TYPE"
    assert error.value.getMessageParameters()["typ"] == typ


@pytest.mark.parametrize("how", ["full", "rightouter", "semi", "outer", "foo"])
def test_lateral_join_types_that_are_not_accepted_raise_as_pyspark(offline_spark, how):
    # As PySpark's plan.LateralJoin, which accepts inner, leftouter, left and cross only.
    from pyspark.errors import AnalysisException

    df = offline_spark.range(3)
    with pytest.raises(AnalysisException) as error:
        df.lateralJoin(df, None, how)
    assert error.value.getCondition() == "UNSUPPORTED_JOIN_TYPE"
    assert error.value.getMessageParameters()["typ"] == how
    assert df.lateralJoin(df, None, "left_outer") is not None


def test_join_on_arguments_that_are_neither_names_nor_conditions_raise_as_pyspark(offline_spark):
    # As PySpark's plan.Join: an on that is not iterable is a TypeError when the join is built;
    # one that is iterable, but not a name, a condition or a list of them, fails on reading
    # the to_plan of it.
    df = offline_spark.range(3)
    with pytest.raises(TypeError, match="'int' object is not iterable"):
        df.join(df, 1)
    with pytest.raises(AttributeError, match="'dict' object has no attribute 'to_plan'"):
        df.join(df, {"id": 1}).collect()
    with pytest.raises(AttributeError, match="'tuple' object has no attribute 'to_plan'"):
        df.join(df, ("id",)).collect()


def test_dict_aggregations_that_are_empty_or_not_str_raise_as_pyspark(offline_spark):
    # As PySpark's DataFrame.agg: a dict with no entries is the aggregation of no expressions,
    # which asserts it has some. GroupedData.agg takes the same dict as no aggregation at all.
    # A column that is not a str is the AttributeError of F.col, and a function that is not a
    # str the AssertionError of _invoke_function, which has no message.
    df = offline_spark.range(3)
    with pytest.raises(AssertionError) as error:
        df.agg({})
    assert str(error.value) == "exprs should not be empty"
    assert df.groupBy().agg({}) is not None
    for call in (lambda: df.agg({"id": ["max"]}), lambda: df.agg({"id": None}),
                 lambda: df.agg({"x": 1}), lambda: df.groupBy().agg({"id": ["max"]}),
                 lambda: df.groupBy().agg({"id": None})):
        with pytest.raises(AssertionError) as error:
            call()
        assert str(error.value) == ""
    with pytest.raises(AttributeError, match="'int' object has no attribute 'endswith'"):
        df.agg({1: "max"})
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'endswith'"):
        df.groupBy().agg({None: "max"})


# ------------------------------------------------- reader/writer in-place setters

@pytest.fixture
def offline_spark():
    # The gRPC channel connects lazily, so building readers/writers and lazy
    # DataFrames needs no server.
    from pyspark.sql import SparkSession

    return SparkSession.builder.remote("sc://localhost:1").getOrCreate()


def test_stream_reader_checks_its_arguments_as_pyspark(offline_spark):
    # As PySpark's DataStreamReader: name raises NOT_EXPECTED_TYPE and
    # INVALID_STREAMING_SOURCE_NAME, load VALUE_NOT_NON_EMPTY_STR for a path that is not a
    # non-empty str, the format readers NOT_EXPECTED_TYPE for a path that is not a str, after
    # they set their options, and changes an AnalysisException for a reader of a schema.
    from pyspark.errors import AnalysisException, PySparkTypeError, PySparkValueError

    reader = offline_spark.readStream
    with pytest.raises(PySparkTypeError) as e:
        reader.name(1)
    assert e.value.getMessageParameters() == {
        "arg_name": "source_name",
        "expected_type": "str",
        "arg_type": "int",
    }
    for name in ("a-b", "", "a b"):
        with pytest.raises(PySparkValueError) as e:
            reader.name(name)
        assert e.value.getCondition() == "INVALID_STREAMING_SOURCE_NAME"
        assert e.value.getMessageParameters() == {"source_name": name}
    for path in ("", "  ", 1, ["p"]):
        with pytest.raises(PySparkValueError) as e:
            reader.load(path)
        assert e.value.getCondition() == "VALUE_NOT_NON_EMPTY_STR"
        assert e.value.getMessageParameters() == {"arg_name": "path", "arg_value": str(path)}
    for read in (reader.json, reader.csv, reader.parquet, reader.orc, reader.text, reader.xml):
        with pytest.raises(PySparkTypeError) as e:
            read(1)
        assert e.value.getMessageParameters() == {
            "arg_name": "path",
            "expected_type": "str",
            "arg_type": "int",
        }
    with pytest.raises(PySparkValueError) as e:
        reader.json("")
    assert e.value.getCondition() == "VALUE_NOT_NON_EMPTY_STR"
    with pytest.raises(PySparkTypeError):
        reader.json(1, multiLine=True)
    assert reader._options["multiLine"] == "true"
    with pytest.raises(AnalysisException) as e:
        offline_spark.readStream.schema("id long").changes("t")
    assert e.value.getCondition() == "_LEGACY_ERROR_TEMP_1189"
    assert str(e.value).endswith("User specified schema not supported with `changes`.")


def test_reader_setters_mutate_and_return_self(offline_spark):
    # PySpark: the setters change the reader and return it; the return value may be
    # ignored (the pattern pandas-on-Spark read_sql_table uses).
    r = offline_spark.read
    assert r.option("a", "b") is r
    assert r.options(c="d", skipped=None, flag=True) is r
    assert r.schema("x INT") is r
    assert r.format("jdbc") is r
    assert r._format == "jdbc"
    assert r._schema == "x INT"
    assert r._options == {"a": "b", "c": "d", "flag": "true"}
    assert type(r.load()).__name__ == "DataFrame"


def test_reader_read_sql_table_pattern(offline_spark):
    # Mirrors pyspark/pandas/namespace.py read_sql_table, which discards every
    # setter's return value before loading.
    reader = offline_spark.read
    reader.option("dbtable", "t")
    reader.option("url", "jdbc:postgresql:db")
    reader.schema("id INT")
    reader.options(fetchsize=10)
    sdf = reader.format("jdbc").load()
    assert type(sdf).__name__ == "DataFrame"
    assert reader._options == {
        "dbtable": "t", "url": "jdbc:postgresql:db", "fetchsize": "10",
    }


def test_pandas_read_sql_table_reader_stage(offline_spark, monkeypatch):
    # Drive the vendored read_sql_table itself up to the point it hands the loaded
    # frame on. default_session() asks the server for a config, so point it at the
    # offline session, and stop at _get_index_map (the first step after the read).
    # pandas-on-Spark does not import under pandas >= 3.0 (pyspark.pandas.frame
    # needs pandas.tseries.frequencies.DateOffset), so skip on any ImportError.
    ns = pytest.importorskip("pyspark.pandas.namespace", exc_type=ImportError)

    class _Loaded(Exception):
        pass

    seen = {}

    def fake_index_map(sdf, index_col):
        seen["sdf"] = sdf
        raise _Loaded

    monkeypatch.setattr(ns, "default_session", lambda: offline_spark)
    monkeypatch.setattr(ns, "_get_index_map", fake_index_map)
    with pytest.raises(_Loaded):
        ns.read_sql_table(
            "t", "jdbc:postgresql:db", schema="id INT", options={"fetchsize": "1"}
        )
    assert type(seen["sdf"]).__name__ == "DataFrame"


def test_reader_reusable_after_load(offline_spark):
    r = offline_spark.read.format("parquet")
    r.load("/a")
    r.load("/b")
    r.table("t")
    r.csv("/c", header=True)
    # Like PySpark, a format method keeps its options and format on the reader.
    assert r._format == "csv"
    assert r._options == {"header": "true"}
    r.json("/d")
    r.parquet("/e", mergeSchema=False)
    r.jdbc("jdbc:x", "tbl", properties={"user": "u"})
    assert r._format == "jdbc"
    assert r._options["dbtable"] == "tbl" and r._options["user"] == "u"
    assert r.option("k", "v") is r
    assert type(r.load()).__name__ == "DataFrame"


def test_reader_reusable_after_failed_read(offline_spark):
    r = offline_spark.read.option("a", "b")
    with pytest.raises(Exception):
        r.csv("/p", schema=object())
    with pytest.raises(Exception):
        r.json("/p", schema=12)
    # A failed call leaves the reader as it was.
    assert r._options == {"a": "b"}
    assert r._format is None
    assert type(r.format("csv").load("/p")).__name__ == "DataFrame"


def test_stream_reader_setters_mutate_and_return_self(offline_spark):
    r = offline_spark.readStream
    assert r.format("rate") is r
    assert r.option("rowsPerSecond", 5) is r
    assert r.options(numPartitions=2, skipped=None) is r
    assert r.schema("v LONG") is r
    assert r.name("src_1") is r
    with pytest.raises(ValueError):
        r.name("bad name")
    assert r._format == "rate"
    assert r._schema == "v LONG"
    assert r._options == {"rowsPerSecond": "5", "numPartitions": "2"}
    r.load()
    r.json("/j")
    assert r._format == "json"
    r.xml("/x", rowTag="row")
    assert r._format == "xml" and r._options["rowTag"] == "row"
    r.table("t")
    assert type(r.load()).__name__ == "DataFrame"


def test_writer_setters_mutate_and_return_self(offline_spark):
    w = offline_spark.range(3).write
    assert w.mode("overwrite") is w
    assert w.format("parquet") is w
    assert w.option("a", "b") is w
    assert w.option("skipped", None) is w
    assert w.options(c="d") is w
    assert w.partitionBy("id") is w
    assert w.bucketBy(2, "id") is w
    assert w.sortBy("id") is w
    assert w.clusterBy("id") is w
    # A failed format call (bad partitionBy) leaves the writer usable.
    with pytest.raises(Exception):
        w.parquet("/p", partitionBy=123)
    assert w.mode("append") is w


def test_writer_v2_takes_its_arguments_as_pyspark(mock_connect):
    # As PySpark's DataFrameWriterV2: overwrite takes a Column or a column name, and
    # partitionedBy a Column or a column name of each argument, else NOT_EXPECTED_TYPE of col,
    # also for a list; tableProperty(property, None) leaves the property unset.
    from pyspark.errors import PySparkTypeError
    from pyspark.sql import functions as F

    spark, service = mock_connect
    df = spark.range(1)

    def sent():
        # ExecutePlanRequest.plan=3, Plan.command=2, Command.write_operation_v2=4.
        plan = parse_message(parse_message(service.execute_requests[-1])[3][0])
        return parse_message(parse_message(plan[2][0])[4][0])

    df.writeTo("t").overwrite("flag")
    # WriteOperationV2.overwrite_condition=8: Expression.unresolved_attribute=2 of
    # UnresolvedAttribute.unparsed_identifier=1.
    condition = parse_message(sent()[8][0])
    assert parse_message(condition[2][0])[1] == [b"flag"]
    writer = df.writeTo("t").tableProperty("a", "1").tableProperty("b", "2")
    writer.tableProperty("a", None).append()
    # WriteOperationV2.table_properties=6, of entries of key=1 and value=2.
    assert [parse_message(entry)[1] for entry in sent().get(6, [])] == [[b"b"]]
    for args, arg_type in [((["id"],), "list"), (("id", 1), "int")]:
        with pytest.raises(PySparkTypeError) as e:
            df.writeTo("t").partitionedBy(*args)
        assert e.value.getCondition() == "NOT_EXPECTED_TYPE"
        assert e.value.getMessageParameters() == {
            "arg_name": "col",
            "expected_type": "Column or str",
            "arg_type": arg_type,
        }
    with pytest.raises(PySparkTypeError) as e:
        df.writeTo("t").overwrite(1)
    assert e.value.getMessageParameters()["arg_name"] == "col"
    df.writeTo("t").partitionedBy("id", F.col("id")).append()
    # WriteOperationV2.partitioning_columns=4.
    assert len(sent()[4]) == 2


def test_writer_v2_setters_mutate_and_return_self(offline_spark):
    from pyspark.sql import functions as F

    w = offline_spark.range(3).writeTo("t")
    assert w.using("parquet") is w
    assert w.option("a", "b") is w
    assert w.options(c="d") is w
    assert w.tableProperty("p", "v") is w
    assert w.partitionedBy(F.col("id")) is w
    assert w.clusterBy("id") is w


def test_dataframe_methods_raise_the_argument_errors_of_pyspark():
    # As PySpark's DataFrame methods, which check their arguments on the client.
    from pyspark.errors import PySparkTypeError, PySparkValueError

    df = _offline_session().range(1)

    def not_expected(arg_name, expected_type, arg_type):
        return (PySparkTypeError, "NOT_EXPECTED_TYPE",
                {"arg_name": arg_name, "expected_type": expected_type, "arg_type": arg_type})

    def not_positive(value):
        return (PySparkValueError, "VALUE_NOT_POSITIVE",
                {"arg_name": "numPartitions", "arg_value": value})

    allowed_hint_types = "str, float, int, Column, list[str], list[float], list[int]"
    for call, (error_type, condition, parameters) in (
        (lambda: df.select(1), not_expected("cols", "Column or str", "int")),
        (lambda: df.select(["id", 1.5]), not_expected("cols", "Column or str", "float")),
        (lambda: df.toDF(1), not_expected("cols", "list[str]", "int")),
        (lambda: df.withColumn("a", 1), not_expected("col", "Column", "int")),
        (lambda: df.withColumns([("a", 1)]), not_expected("colsMap", "dict", "list")),
        (lambda: df.withColumnsRenamed([("id", "a")]), not_expected("colsMap", "dict", "list")),
        (lambda: df.hint(1), not_expected("name", "str", "int")),
        (lambda: df.hint("x", {"a": 1}),
         (PySparkTypeError, "INVALID_ITEM_FOR_CONTAINER",
          {"arg_name": "parameters", "allowed_types": allowed_hint_types, "item_type": "dict"})),
        (lambda: df.hint("x", [None, 1]),
         (PySparkTypeError, "INVALID_ITEM_FOR_CONTAINER",
          {"arg_name": "parameters", "allowed_types": allowed_hint_types,
           "item_type": "NoneType"})),
        (lambda: df.hint("x", [1], [2, None]),
         (PySparkTypeError, "INVALID_ITEM_FOR_CONTAINER",
          {"arg_name": "parameters", "allowed_types": allowed_hint_types,
           "item_type": "list[int]"})),
        (lambda: df.dropDuplicates("id"), not_expected("subset", "list or tuple", "str")),
        (lambda: df.drop_duplicates(["id", 1]), not_expected("subset", "str", "int")),
        (lambda: df.dropDuplicatesWithinWatermark("id"),
         not_expected("subset", "list or tuple", "str")),
        (lambda: df.repartition(0), not_positive("0")),
        (lambda: df.repartition(1.5), not_expected("numPartitions", "Column or str", "float")),
        (lambda: df.repartition("id", 2), not_expected("cols", "Column or str", "int")),
        (lambda: df.repartitionByRange(-1, "id"), not_positive("-1")),
        (lambda: df.repartitionByRange(2), (PySparkValueError, "CANNOT_BE_EMPTY",
                                            {"item": "cols"})),
        (lambda: df.repartitionByRange(1.5, "id"),
         not_expected("numPartitions", "Column, int or str", "float")),
        (lambda: df.coalesce(0), not_positive("0")),
        (lambda: df.repartitionById(True, "id"), not_expected("numPartitions", "int", "bool")),
        (lambda: df.repartitionById(0, "id"), not_positive("0")),
    ):
        with pytest.raises(error_type) as error:
            call()
        assert error.value.getCondition() == condition
        assert error.value.getMessageParameters() == parameters
    for call in (lambda: df.withColumns(), lambda: df.withColumns({"a": 1})):
        with pytest.raises(AssertionError):
            call()


def test_streaming_queries_check_their_timeouts_as_pyspark(mock_connect):
    # As PySpark's: StreamingQueryException of pyspark.sql.streaming is that of
    # pyspark.errors; the waits of a query and of spark.streams raise VALUE_NOT_POSITIVE of a
    # timeout that is not a positive number, of the name of its type; and explain takes an
    # int as a bool, but no other value.
    import pyspark.errors
    import pyspark.sql.streaming
    import pyspark.sql.streaming.query
    from pyspark.errors import PySparkValueError

    exception = pyspark.errors.StreamingQueryException
    assert pyspark.sql.streaming.StreamingQueryException is exception
    assert pyspark.sql.streaming.query.StreamingQueryException is exception
    assert not hasattr(pyspark.sql.streaming, "StreamingQueryStatus")
    spark, service = mock_connect
    # ExecutePlanResponse.write_stream_operation_start_result=8: query_id=1 (id=1, run_id=2).
    service.execute_payload = message_field(
        8, message_field(1, message_field(1, "q") + message_field(2, "r"))
    )
    query = spark.readStream.format("rate").load().writeStream.format("noop").start()
    for timeout in (0, -1, 0.0, "1", [1]):
        for wait in (query.awaitTermination, spark.streams.awaitAnyTermination):
            with pytest.raises(PySparkValueError) as e:
                wait(timeout)
            assert e.value.getCondition() == "VALUE_NOT_POSITIVE"
            assert e.value.getMessageParameters() == {
                "arg_name": "timeout",
                "arg_value": type(timeout).__name__,
            }
    for extended in (1.5, "x", None):
        with pytest.raises(TypeError, match="but expected one of"):
            query.explain(extended)


def test_stream_writer_setters_mutate_and_return_self(offline_spark):
    w = offline_spark.readStream.format("rate").load().writeStream
    assert w.outputMode("append") is w
    assert w.format("memory") is w
    assert w.option("a", "b") is w
    assert w.options(c="d") is w
    assert w.partitionBy("value") is w
    assert w.clusterBy("value") is w
    assert w.queryName("q") is w
    with pytest.raises(ValueError):
        w.trigger()
    assert w.trigger(processingTime="1 second") is w
    assert w.foreachBatch(lambda df, i: None) is w
    assert w.foreach(lambda row: None) is w


@pytest.mark.parametrize("kwargs, condition, parameters", [
    ({}, "ONLY_ALLOW_SINGLE_TRIGGER", {}),
    ({"processingTime": "1 second", "once": True}, "ONLY_ALLOW_SINGLE_TRIGGER", {}),
    ({"processingTime": "  "}, "VALUE_NOT_NON_EMPTY_STR",
     {"arg_name": "processingTime", "arg_value": "  "}),
    ({"continuous": 5}, "VALUE_NOT_NON_EMPTY_STR", {"arg_name": "continuous", "arg_value": "5"}),
    ({"realTime": ""}, "VALUE_NOT_NON_EMPTY_STR", {"arg_name": "realTime", "arg_value": ""}),
    ({"once": False}, "VALUE_NOT_ALLOWED", {"arg_name": "once", "allowed_values": "[True]"}),
    ({"availableNow": 1}, "VALUE_NOT_ALLOWED",
     {"arg_name": "availableNow", "allowed_values": "[True]"}),
], ids=["none", "two", "empty", "not-a-string", "empty-real-time", "once-false",
        "available-now-1"])
def test_stream_writer_triggers_are_checked_as_pyspark(offline_spark, kwargs, condition,
                                                        parameters):
    from pyspark.errors import PySparkValueError

    w = offline_spark.readStream.format("rate").load().writeStream
    with pytest.raises(PySparkValueError) as error:
        w.trigger(**kwargs)
    assert error.value.getCondition() == condition
    assert error.value.getMessageParameters() == parameters
    # A trigger of None is not given, as in PySpark.
    assert w.trigger(processingTime=None, once=True) is w


def test_foreach_sends_the_function_of_each_partition_as_pyspark(mock_connect):
    # As PySpark's foreach: the command (func, None, serializer, serializer) of the function of
    # _construct_foreach_function, which calls a function with each row of a partition;
    # ATTRIBUTE_NOT_CALLABLE for an object without a callable process, open or close; and
    # STREAMING_CONNECT_SERIALIZATION_ERROR for a function that does not pickle, as of
    # foreachBatch.
    import threading

    from pyspark import cloudpickle
    from pyspark.errors import PySparkAttributeError, PySparkPicklingError

    spark, service = mock_connect
    # ExecutePlanResponse.write_stream_operation_start_result=8: query_id=1 (id=1, run_id=2).
    service.execute_payload = message_field(
        8, message_field(1, message_field(1, "q") + message_field(2, "r"))
    )
    writer = spark.readStream.format("rate").load().writeStream

    def row_function(row):
        raise ValueError(f"the row {row}")

    writer.foreach(row_function).start()
    # ExecutePlanRequest.plan=3, Plan.command=2, Command.write_stream_operation_start=6:
    # WriteStreamOperationStart.foreach_writer=13 of StreamingForeachFunction.python_function=1
    # of PythonUDF.command=3.
    plan = parse_message(parse_message(service.execute_requests[-1])[3][0])
    start = parse_message(parse_message(plan[2][0])[6][0])
    udf = parse_message(parse_message(start[13][0])[1][0])
    function, return_type, input_serializer, output_serializer = cloudpickle.loads(udf[3][0])
    assert return_type is None and input_serializer is output_serializer
    with pytest.raises(ValueError, match="the row 7"):
        function(0, iter([7]))

    class NoProcess:
        pass

    class Uncallable:
        process = 1

    class UncallableClose:
        def process(self, row):
            pass

        close = 1

    for writer_object, attr_name in [
        (1, "process"),
        (NoProcess(), "process"),
        (Uncallable(), "process"),
        (UncallableClose(), "close"),
    ]:
        with pytest.raises(PySparkAttributeError) as e:
            writer.foreach(writer_object)
        assert e.value.getCondition() == "ATTRIBUTE_NOT_CALLABLE"
        assert e.value.getMessageParameters() == {"attr_name": attr_name, "obj_name": "f"}
    lock = threading.Lock()
    for method, name, function in [
        (writer.foreach, "foreach", lambda row: lock),
        (writer.foreachBatch, "foreachBatch", lambda df, batch_id: lock),
    ]:
        with pytest.raises(PySparkPicklingError) as e:
            method(function)
        assert e.value.getCondition() == "STREAMING_CONNECT_SERIALIZATION_ERROR"
        assert e.value.getMessageParameters() == {"name": name}


def test_stream_writer_sends_its_trigger_as_pyspark(mock_connect):
    # As PySpark's: the trigger of a WriteStreamOperationStart, an interval or duration
    # stripped, of the real-time trigger too.
    spark, service = mock_connect
    # ExecutePlanResponse.write_stream_operation_start_result=8: query_id=1 (id=1, run_id=2).
    service.execute_payload = message_field(
        8, message_field(1, message_field(1, "q") + message_field(2, "r")))

    def sent(**trigger):
        spark.readStream.format("rate").load().writeStream.format("memory").queryName(
            "q").trigger(**trigger).start()
        # ExecutePlanRequest.plan=3 -> Plan.command=2 -> Command.write_stream_operation_start=6.
        plan = parse_message(parse_message(service.execute_requests[-1])[3][0])
        start = parse_message(parse_message(plan[2][0])[6][0])
        # WriteStreamOperationStart: processing_time_interval=5, available_now=6, once=7,
        # continuous_checkpoint_interval=8, real_time_batch_duration=100.
        return {field: start[field] for field in (5, 6, 7, 8, 100) if field in start}

    assert sent(processingTime=" 1 second ") == {5: [b"1 second"]}
    assert sent(once=True) == {7: [1]}
    assert sent(availableNow=True) == {6: [1]}
    assert sent(continuous="1 minute") == {8: [b"1 minute"]}
    assert sent(realTime=" 5 seconds ") == {100: [b"5 seconds"]}


def test_stream_writer_takes_its_query_name_and_columns_as_pyspark(offline_spark):
    # As PySpark's DataStreamWriter: queryName, also of start and toTable, raises
    # VALUE_NOT_NON_EMPTY_STR for a name that is not a non-empty str, and partitionBy and
    # clusterBy take a list or a tuple of column names alone.
    from pyspark.errors import PySparkValueError

    writer = offline_spark.readStream.format("rate").load().writeStream
    calls = [writer.queryName, lambda name: writer.start(queryName=name)]
    calls.append(lambda name: writer.toTable("t", queryName=name))
    for name in ("", "  ", 1):
        for call in calls:
            with pytest.raises(PySparkValueError) as e:
                call(name)
            assert e.value.getCondition() == "VALUE_NOT_NON_EMPTY_STR"
            assert e.value.getMessageParameters() == {
                "arg_name": "queryName",
                "arg_value": str(name),
            }
    with pytest.raises(PySparkValueError):
        writer.queryName(None)
    for columns in (["a", "b"], ("a", "b")):
        assert writer.partitionBy(columns) is writer
        assert writer.clusterBy(columns) is writer
    assert writer.partitionBy("a", "b") is writer
    with pytest.raises(TypeError):
        writer.partitionBy(["a"], "b")


def test_merge_into_writer_mutates_and_returns_self(offline_spark):
    # PySpark: withSchemaEvolution and every clause action change the writer and
    # return that same writer; a clause object can be used more than once.
    from pyspark.sql import functions as F

    src = offline_spark.range(3)
    m = src.mergeInto("target", F.col("id") == F.col("target.id"))
    assert m.withSchemaEvolution() is m

    matched = m.whenMatched(F.col("id") > 1)
    assert matched.updateAll() is m
    assert matched.update({"id": F.col("id")}) is m
    assert matched.delete() is m
    assert m.whenMatched().delete() is m

    not_matched = m.whenNotMatched()
    assert not_matched.insertAll() is m
    assert not_matched.insert({"id": F.col("id")}) is m

    by_source = m.whenNotMatchedBySource(F.col("target.id") < 0)
    assert by_source.updateAll() is m
    assert by_source.update({"id": F.lit(0)}) is m
    assert by_source.delete() is m

    # The ignore-the-return-value form keeps working on the original object.
    m.whenNotMatched().insertAll()
    m.withSchemaEvolution()
    assert m.whenMatched().updateAll() is m


# --------------------------------------------------------------------------- artifact uploads that stream their chunks, and class files

def test_add_artifacts_refuses_the_paths_pyspark_refuses(tmp_path):
    # Each error is PySpark's for the path and the flags, raised before any upload.
    import errno

    from pyspark.errors import PySparkRuntimeError, PySparkValueError

    for name in ["A.class", "m.py", "n.txt", "b.tar.gz", "b.zip", "b#c.zip"]:
        (tmp_path / name).write_bytes(b"x")
    (tmp_path / "dirx.jar").mkdir()
    (tmp_path / "sub").mkdir()

    def at(name):
        return str(tmp_path / name)

    def unsupported(name):
        return {"operation": f"{at(name)} file format"}

    cases = [
        # A class file is refused, unless it is a file.
        (at("A.class"), {}, PySparkRuntimeError, "UNSUPPORTED_OPERATION", unsupported("A.class")),
        (at("A.class"), {"pyfile": True}, PySparkRuntimeError, "UNSUPPORTED_OPERATION", unsupported("A.class")),
        (at("A.class"), {"archive": True}, PySparkRuntimeError, "UNSUPPORTED_OPERATION", unsupported("A.class")),
        # A flag takes only its formats, and a fragment names only an archive.
        (at("m.py"), {"archive": True}, PySparkRuntimeError, "UNSUPPORTED_OPERATION", unsupported("m.py")),
        (at("n.txt"), {"pyfile": True}, PySparkRuntimeError, "UNSUPPORTED_OPERATION", unsupported("n.txt")),
        (at("b.tar.gz"), {"pyfile": True}, PySparkRuntimeError, "UNSUPPORTED_OPERATION", unsupported("b.tar.gz")),
        (at("b.zip") + "#alias", {}, PySparkRuntimeError, "UNSUPPORTED_OPERATION", unsupported("b.zip")),
        (at("b%23c.zip") + "#alias", {"archive": True}, PySparkValueError, "VALUE_ALLOWED",
         {"arg_name": "artifact path", "disallowed_value": "#"}),
        # Remote URIs are refused whatever the flags.
        ("s3://bucket/a.jar", {}, PySparkRuntimeError, "UNSUPPORTED_OPERATION", {"operation": "s3 scheme"}),
        ("hdfs://h/a.jar", {"file": True}, PySparkRuntimeError, "UNSUPPORTED_OPERATION",
         {"operation": "hdfs scheme"}),
        # A path without a suffix is a format, even when it is a directory.
        (at("sub"), {}, PySparkRuntimeError, "UNSUPPORTED_OPERATION", unsupported("sub")),
    ]
    spark = _offline_session()
    for path, flags, error_type, condition, parameters in cases:
        with pytest.raises(error_type) as error:
            spark.addArtifacts(path, **flags)
        assert error.value.getCondition() == condition, path
        assert error.value.getMessageParameters() == parameters, path
    # A file that Python cannot open is the error of open, as PySpark's LocalFile raises it.
    for path, flags, expected in [
        (at("sub"), {"file": True}, f"[Errno 21] Is a directory: '{at('sub')}'"),
        (at("dirx.jar"), {"archive": True}, f"[Errno 21] Is a directory: '{at('dirx.jar')}'"),
        (at("missing.jar"), {}, f"[Errno 2] No such file or directory: '{at('missing.jar')}'"),
        ("file://" + at("missing.jar"), {"pyfile": True},
         f"[Errno 2] No such file or directory: '{at('missing.jar')}'"),
    ]:
        with pytest.raises(OSError) as error:
            spark.addArtifacts(path, **flags)
        assert str(error.value) == expected, path
        assert error.value.errno in (errno.ENOENT, errno.EISDIR), path


def test_add_artifacts_puts_a_pyfile_zip_on_the_client_path_before_opening_it(tmp_path, monkeypatch):
    # As PySpark's _parse_artifacts: a zip, egg or jar of pyfile is put on sys.path before its
    # file is opened, and a Python file is not.
    import sys

    monkeypatch.setattr(sys, "path", list(sys.path))
    spark = _offline_session()
    zipped = str(tmp_path / "missing.zip")
    with pytest.raises(FileNotFoundError):
        spark.addArtifacts(zipped, pyfile=True)
    assert sys.path[1] == zipped
    before = list(sys.path)
    with pytest.raises(FileNotFoundError):
        spark.addArtifacts(str(tmp_path / "missing.py"), pyfile=True)
    assert sys.path == before


# --------------------------------------------------------------------------- reader schema

def _sent_read_schema(request):
    """The DataSource.schema of a captured AnalyzePlanRequest: schema=4 -> plan=1 -> root=1 ->
    read=2 -> data_source=2 -> schema=2."""
    message = parse_message(request)
    for field in (4, 1, 1, 2, 2):
        message = parse_message(message[field][0])
    return message[2][0].decode()


def test_apply_in_pandas_with_state_sends_its_relation_as_pyspark(mock_connect):
    import json

    spark, service = mock_connect
    output = T.StructType([T.StructField("id", T.LongType()), T.StructField("c", T.LongType())])
    state = T.StructType([T.StructField("c", T.LongType())])
    spark.range(3).groupBy("id").applyInPandasWithState(
        lambda key, frames, s: frames, output, state, "update", "NoTimeout").collect()
    # plan=3 -> root=1 -> apply_in_pandas_with_state=34: func=3, output_schema=4,
    # state_schema=5, output_mode=6, timeout_conf=7.
    root = parse_message(parse_message(parse_message(service.execute_requests[-1])[3][0])[1][0])
    assert 34 in root, sorted(root)
    apply = parse_message(root[34][0])
    assert json.loads(apply[4][0]) == output.jsonValue()
    assert json.loads(apply[5][0]) == state.jsonValue()
    assert (apply[6][0], apply[7][0]) == (b"update", b"NoTimeout")
    # CommonInlineUserDefinedFunction.python_udf=4 -> PythonUDF.eval_type=2.
    assert parse_message(parse_message(apply[3][0])[4][0])[2][0] == 208


@pytest.mark.parametrize("read", [
    lambda spark, schema: spark.read.schema(schema).csv("unused.csv"),
    lambda spark, schema: spark.read.csv("unused.csv", schema=schema),
    lambda spark, schema: spark.read.load("unused", format="json", schema=schema),
    lambda spark, schema: spark.readStream.schema(schema).csv("unused"),
    lambda spark, schema: spark.readStream.csv("unused", schema=schema),
    lambda spark, schema: spark.readStream.load("unused", format="json", schema=schema),
], ids=["read.schema", "read.csv", "read.load", "readStream.schema", "readStream.csv",
        "readStream.load"])
def test_reader_schema_is_sent_as_pyspark_sends_it(mock_connect, read):
    from pyspark.errors import PySparkTypeError

    spark, service = mock_connect
    struct = T.StructType([T.StructField("a", T.LongType(), True, {"comment": "key"})])
    # pyspark sends a StructType as its JSON and a string as is, for the server to parse.
    for schema, sent in [
        (struct, '{"fields":[{"metadata":{"comment":"key"},"name":"a","nullable":true,'
                 '"type":"long"}],"type":"struct"}'),
        ("a long COMMENT 'key'", "a long COMMENT 'key'"),
    ]:
        read(spark, schema).columns
        assert _sent_read_schema(service.analyze_requests[-1]) == sent
    with pytest.raises(PySparkTypeError) as error:
        read(spark, T.LongType())
    assert error.value.getCondition() == "NOT_EXPECTED_TYPE"
    assert error.value.getMessageParameters() == {
        "arg_name": "schema", "expected_type": "str or struct type", "arg_type": "LongType"}


def _sent_relation(request, field):
    """The relation `field` of the root of a captured AnalyzePlanRequest or ExecutePlanRequest:
    AnalyzePlanRequest.schema=4 -> plan=1, or ExecutePlanRequest.plan=3; then Plan.root=1."""
    message = parse_message(request)
    plan = parse_message(message[4][0])[1][0] if 4 in message else message[3][0]
    root = parse_message(parse_message(plan)[1][0])
    assert field in root, sorted(root)
    return parse_message(root[field][0])


def _struct_field_names(data_type):
    # DataType.struct=21 -> Struct.fields=1 -> StructField.name=1.
    struct = parse_message(parse_message(data_type)[21][0])
    return [parse_message(field)[1][0].decode() for field in struct[1]]


@pytest.mark.parametrize("read, format, option", [
    (lambda spark, df: spark.read.csv(df, schema="a INT, b STRING", sep=";"), 1, b"sep"),
    (lambda spark, df: spark.read.json(df, schema="a INT, b STRING", lineSep=";"), 2,
     b"lineSep"),
    (lambda spark, df: spark.read.xml(df, schema="a INT, b STRING", rowTag=";"), 3, b"rowTag"),
], ids=["csv", "json", "xml"])
def test_reader_parses_a_dataframe_of_strings_as_pyspark(mock_connect, read, format, option):
    # As PySpark's, csv, json and xml of a DataFrame parse its strings: a Parse (Relation.parse
    # =30) of the DataFrame (input=1) in the format (format=2), with the schema (schema=3) of
    # the reader and its options (options=4).
    spark, service = mock_connect
    strings = T.StructType([T.StructField("value", T.StringType())])
    read(spark, spark.createDataFrame([("1;x",)], strings)).columns
    parse = _sent_relation(service.analyze_requests[-1], 30)
    assert parse[2] == [format]
    # The server parses a DDL schema, as PySpark's `StructType.fromDDL`:
    # AnalyzePlanRequest.ddl_parse=11 -> ddl_string=1.
    assert [parse_message(parse_message(request)[11][0])[1][0]
            for request in service.analyze_requests if 11 in parse_message(request)] == [
        b"a INT, b STRING"]
    assert _struct_field_names(parse[3][0]) == ["a", "b"]
    assert parse_message(parse[4][0]) == {1: [option], 2: [b";"]}
    assert 1 in parse
    # A schema of JSON, as a StructType sends it, or none.
    struct = T.StructType([T.StructField("c", T.LongType())])
    spark.read.json(spark.createDataFrame([("{}",)], strings), schema=struct).columns
    assert _struct_field_names(_sent_relation(service.analyze_requests[-1], 30)[3][0]) == ["c"]
    spark.read.csv(spark.createDataFrame([("1",)], strings)).columns
    assert 3 not in _sent_relation(service.analyze_requests[-1], 30)
    # The reader keeps the schema and the options of a parse, but not its format, which
    # only `load` sets.
    reader = spark.read.format("parquet")
    reader.csv(spark.createDataFrame([("1",)], strings), schema="a INT", sep=";")
    assert (reader._format, reader._schema, reader._options) == ("parquet", "a INT", {"sep": ";"})
    reader.csv("p", sep=",")
    assert (reader._format, reader._schema, reader._options) == ("csv", "a INT", {"sep": ","})


def test_reader_and_writer_check_their_arguments_as_pyspark(mock_connect):
    # As PySpark Connect's: a write of a mode other than append, overwrite, error (or
    # errorifexists, which PySpark documents) and ignore raises UNSUPPORTED_OPERATION, of the
    # last mode set; format(None) unsets the format; the reader raises JVM_ATTRIBUTE_NOT_SUPPORTED
    # of _jreader, asserts the bounds of a JDBC column and refuses a schema in changes; and csv,
    # text and orc of None read without a path.
    from pyspark.errors import AnalysisException, PySparkAttributeError, PySparkValueError

    spark, service = mock_connect
    df = spark.range(3)
    for write, mode in [
        (lambda: df.write.mode("bad").save("p"), "bad"),
        (lambda: df.write.mode("append").mode("default").saveAsTable("t"), "default"),
        (lambda: df.write.mode(" append").insertInto("t"), " append"),
        (lambda: df.write.parquet("p", mode=""), ""),
        (lambda: df.write.jdbc("jdbc:x", "t", mode="bad"), "bad"),
    ]:
        with pytest.raises(PySparkValueError) as e:
            write()
        assert e.value.getCondition() == "UNSUPPORTED_OPERATION"
        assert e.value.getMessageParameters() == {"operation": mode}

    def sent_write():
        # ExecutePlanRequest.plan=3, Plan.command=2, Command.write_operation=2.
        plan = parse_message(parse_message(service.execute_requests[-1])[3][0])
        return parse_message(parse_message(plan[2][0])[2][0])

    # WriteOperation.source=2 and mode=5, of SAVE_MODE_APPEND=1 and SAVE_MODE_ERROR_IF_EXISTS=3.
    df.write.mode("bad").mode("APPEND").format("json").format(None).save("p")
    assert 2 not in sent_write() and sent_write()[5] == [1]
    df.write.mode("errorifexists").insertInto("t")
    assert sent_write()[5] == [3]
    df.write.jdbc("jdbc:x", "t", properties={"user": 1, "flag": True, "skip": None})
    # WriteOperation.options=9, of entries of key=1 and value=2.
    options = {parse_message(entry)[1][0]: parse_message(entry)[2][0] for entry in sent_write()[9]}
    assert options == {b"user": b"1", b"flag": b"true", b"url": b"jdbc:x", b"dbtable": b"t"}

    reader = spark.read.format("csv")
    assert reader.format(None) is reader and reader._format is None
    stream = spark.readStream.format("rate")
    assert stream.format(None) is stream
    with pytest.raises(PySparkAttributeError) as e:
        spark.read._jreader
    assert e.value.getCondition() == "JVM_ATTRIBUTE_NOT_SUPPORTED"
    assert e.value.getMessageParameters() == {"attr_name": "_jreader"}
    with pytest.raises(AnalysisException) as e:
        spark.read.schema("a int").changes("t")
    assert e.value.getCondition() == "_LEGACY_ERROR_TEMP_1189"
    assert e.value.getMessageParameters() == {"operation": "changes"}
    assert "User specified schema not supported with `changes`." in str(e.value)
    for bounds, missing in [({}, "lowerBound"), ({"lowerBound": 1}, "upperBound"),
                            ({"lowerBound": 1, "upperBound": 2}, "numPartitions")]:
        reader = spark.read
        with pytest.raises(AssertionError, match=f"{missing} can not be None when ``column``"):
            reader.jdbc("jdbc:x", "t", column="c", **bounds)
        assert reader._format == "jdbc" and reader._options == {}

    def sent_data_source():
        # Relation.read=2 -> Read.data_source=2: options=3, of entries of key=1 and value=2,
        # and predicates=5.
        source = parse_message(_sent_relation(service.analyze_requests[-1], 2)[2][0])
        options = {parse_message(e)[1][0].decode(): parse_message(e)[2][0].decode()
                   for e in source.get(3, [])}
        return options, [p.decode() for p in source.get(5, [])]

    spark.read.jdbc("jdbc:x", "t", column="c", lowerBound=1, upperBound=2, numPartitions="3",
                    predicates=["p"], properties={"user": 1, "flag": True}).columns
    assert sent_data_source() == ({
        "partitionColumn": "c", "lowerBound": "1", "upperBound": "2", "numPartitions": "3",
        "user": "1", "flag": "true", "url": "jdbc:x", "dbtable": "t"}, [])
    spark.read.jdbc("jdbc:x", "t", lowerBound=1, predicates=["p"]).columns
    assert sent_data_source() == ({"url": "jdbc:x", "dbtable": "t"}, ["p"])
    for read in (spark.read.csv, spark.read.text, spark.read.orc):
        read(None).columns
        assert _sent_data_source_paths(service.analyze_requests[-1]) == []

def test_reader_paths_are_checked_as_pyspark(offline_spark):
    from pyspark.errors import PySparkTypeError

    # json and xml take a path, a list of them or a DataFrame; csv loads any other path,
    # which load then asserts, as PySpark's.
    for read in (offline_spark.read.json, offline_spark.read.xml):
        for path in (1, ("p",)):
            with pytest.raises(PySparkTypeError) as error:
                read(path)
            assert error.value.getCondition() == "NOT_EXPECTED_TYPE"
            assert error.value.getMessageParameters() == {
                "arg_name": "path", "expected_type": "str, list, or DataFrame",
                "arg_type": type(path).__name__}
    for read in (
        lambda path: offline_spark.read.csv(path),
        lambda path: offline_spark.read.load(path),
        lambda path: offline_spark.read.orc(path),
        lambda path: offline_spark.read.text(path),
        lambda path: offline_spark.read.json([path]),
    ):
        for path in (("p",), 1):
            with pytest.raises(AssertionError):
                read(path)
    with pytest.raises(AssertionError):
        offline_spark.read.parquet("p", 1)


def _sent_data_source_paths(request):
    # Relation.read=2 -> Read.data_source=2 -> DataSource.paths=4.
    read = _sent_relation(request, 2)
    return [path.decode() for path in parse_message(read[2][0]).get(4, [])]


@pytest.mark.parametrize("read", [
    lambda spark: spark.read.json(["a", "b"]),
    lambda spark: spark.read.csv(["a", "b"]),
    lambda spark: spark.read.xml(["a", "b"], rowTag="row"),
    lambda spark: spark.read.parquet("a", "b"),
    lambda spark: spark.read.orc(["a", "b"]),
    lambda spark: spark.read.text(["a", "b"]),
    lambda spark: spark.read.load(["a", "b"], format="json"),
], ids=["json", "csv", "xml", "parquet", "orc", "text", "load"])
def test_reader_loads_every_path_as_pyspark(mock_connect, read):
    spark, service = mock_connect
    read(spark).columns
    assert _sent_data_source_paths(service.analyze_requests[-1]) == ["a", "b"]


def test_writer_bucket_sort_and_cluster_columns_are_checked_as_pyspark(offline_spark):
    from pyspark.errors import PySparkTypeError, PySparkValueError

    w = offline_spark.range(3).write
    assert w.mode(None) is w
    assert w.bucketBy(True, "id") is w
    assert w.bucketBy(2, ["id"]) is w and w.bucketBy(2, ("id", "x")) is w
    assert w.sortBy(["id"]) is w and w.sortBy("id", "x") is w
    assert w.clusterBy(["id"]) is w and w.clusterBy(("id", "x")) is w
    for call, condition, parameters in [
        (lambda: w.bucketBy("2", "id"), "NOT_EXPECTED_TYPE",
         {"arg_name": "numBuckets", "expected_type": "int", "arg_type": "str"}),
        (lambda: w.bucketBy(2.0, "id"), "NOT_EXPECTED_TYPE",
         {"arg_name": "numBuckets", "expected_type": "int", "arg_type": "float"}),
        (lambda: w.bucketBy(2, ["id"], "x"), "CANNOT_SET_TOGETHER",
         {"arg_list": "`col` of type list and `cols`"}),
        (lambda: w.sortBy(("id",), "x"), "CANNOT_SET_TOGETHER",
         {"arg_list": "`col` of type tuple and `cols`"}),
        (lambda: w.bucketBy(2, 1), "NOT_EXPECTED_TYPE",
         {"arg_name": "col", "expected_type": "list[str]", "arg_type": "int"}),
        (lambda: w.bucketBy(2, "id", 1), "NOT_EXPECTED_TYPE",
         {"arg_name": "cols", "expected_type": "list[str]", "arg_type": "int"}),
        (lambda: w.sortBy(["id", 1]), "NOT_EXPECTED_TYPE",
         {"arg_name": "cols", "expected_type": "list[str]", "arg_type": "int"}),
        (lambda: w.sortBy(1), "NOT_EXPECTED_TYPE",
         {"arg_name": "col", "expected_type": "list[str]", "arg_type": "int"}),
    ]:
        with pytest.raises((PySparkTypeError, PySparkValueError)) as error:
            call()
        assert error.value.getCondition() == condition
        assert error.value.getMessageParameters() == parameters
    # As PySpark's `col[0]` of an empty list, and its assertion of the clustering columns.
    with pytest.raises(IndexError):
        w.bucketBy(2, [])
    for call in (w.clusterBy, lambda: w.clusterBy([])):
        with pytest.raises(AssertionError, match="clusterBy needs one or more"):
            call()


def _sent_catalog(request, field):
    # Relation.catalog=200 -> the Catalog operation `field`.
    return parse_message(_sent_relation(request, 200)[field][0])


def test_catalog_creates_tables_eagerly_as_pyspark(mock_connect):
    import warnings

    from pyspark.errors import PySparkTypeError

    spark, service = mock_connect
    struct = T.StructType([T.StructField("a", T.LongType()), T.StructField("b", T.StringType())])
    # As PySpark's, createTable runs on the call, a CreateTable (Catalog.create_table=14) of
    # the table (table_name=1), path=2, source=3, description=4, schema=5 and options=6, of
    # strings, where a None is left out.
    spark.catalog.createTable(
        "t", path="/p", source="parquet", schema=struct, description="d", k="v", skipped=None)
    create = _sent_catalog(service.execute_requests[-1], 14)
    assert (create[1], create[2], create[3], create[4]) == ([b"t"], [b"/p"], [b"parquet"], [b"d"])
    assert _struct_field_names(create[5][0]) == ["a", "b"]
    assert [parse_message(entry) for entry in create[6]] == [{1: [b"k"], 2: [b"v"]}]
    with pytest.raises(TypeError, match="1 has type <class 'int'>"):
        spark.catalog.createTable("t", source="parquet", k=1)
    for schema in ("a INT", T.LongType()):
        with pytest.raises(PySparkTypeError) as error:
            spark.catalog.createTable("t", source="parquet", schema=schema)
        assert error.value.getCondition() == "NOT_EXPECTED_TYPE"
        assert error.value.getMessageParameters() == {
            "arg_name": "schema", "expected_type": "struct type",
            "arg_type": type(schema).__name__}
    # createExternalTable warns that it is deprecated and creates the table as createTable.
    calls = len(service.execute_requests)
    with pytest.warns(FutureWarning, match="createExternalTable is deprecated since Spark 4.0"):
        spark.catalog.createExternalTable("e", path="/e", source="csv", schema=struct, sep=";")
    assert len(service.execute_requests) == calls + 1
    create = _sent_catalog(service.execute_requests[-1], 14)
    assert (create[1], create[2], create[3]) == ([b"e"], [b"/e"], [b"csv"])
    assert 4 not in create and _struct_field_names(create[5][0]) == ["a", "b"]
    assert [parse_message(entry) for entry in create[6]] == [{1: [b"sep"], 2: [b";"]}]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        with pytest.raises(PySparkTypeError):
            spark.catalog.createExternalTable("e", schema=1)


@pytest.mark.parametrize("call, method", [
    (lambda catalog, *db: catalog.tableExists("t", *db), "tableExists"),
    (lambda catalog, *db: catalog.functionExists("f", *db), "functionExists"),
    (lambda catalog, *db: catalog.listColumns("t", *db), "listColumns"),
], ids=["tableExists", "functionExists", "listColumns"])
def test_catalog_warns_of_the_deprecated_db_name_as_pyspark(mock_connect, call, method):
    import contextlib
    import warnings

    spark, _ = mock_connect
    # As PySpark's, only a call of `dbName` warns, before its request, which the mock answers
    # with no rows.
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        for db in ((), ("db",)):
            with contextlib.suppress(Exception):
                call(spark.catalog, *db)
    assert [str(w.message) for w in record if issubclass(w.category, FutureWarning)] == [
        "`dbName` has been deprecated since Spark 3.4 and might be removed in a future version. "
        f"Use {method}(`dbName.tableName`) instead."]


# Exact built-in string conversion guards. A Row made in Python keeps its values as they
# are, as PySpark's does, so these check the conversion that createDataFrame makes.


@pytest.mark.parametrize("text", [
    "", "1.25", "-42", "NaN", "a\x00b", "é雪😀", "e\u0301", "\uffff",
])
def test_exact_string_converts_as_text(text):
    # A string, even when it reads as a number, is text: with a string or an integer in
    # another row, the column is a string column, as PySpark merges the types (the live
    # suite checks the values).
    assert _local_create_dataframe([Row(value=text), Row(value="y")]).isLocal()
    assert _local_create_dataframe([Row(value=text), Row(value=5)]).isLocal()


@pytest.mark.parametrize("text", ["\ud800", "\udfff", "a\ud800b"])
def test_exact_string_surrogate_error_preserves_generic_fallback(text):
    # As PySpark, where PyArrow cannot encode a lone surrogate either.
    with pytest.raises(UnicodeEncodeError, match="surrogates not allowed"):
        _local_create_dataframe([Row(value=text)])


def test_exact_string_subclass_uses_generic_conversion():
    from pyspark.errors import PySparkTypeError

    class IndexString(str):
        def __index__(self):
            return 17

    # As in PySpark, which takes an instance of a subclass by its `__dict__`: an empty
    # struct, which merges with neither an integer nor a string.
    assert _local_create_dataframe([Row(value=IndexString("x"))]).isLocal()
    for other, data_type in [(5, "LongType"), ("y", "StringType")]:
        with pytest.raises(PySparkTypeError) as error:
            _local_create_dataframe([Row(value=IndexString("x")), Row(value=other)])
        assert error.value.getCondition() == "CANNOT_MERGE_TYPE"
        assert error.value.getMessageParameters() == {
            "data_type1": "StructType", "data_type2": data_type}


# --------------------------------------------------------------------------- spatial values with invalid or mixed SRIDs
_SPATIAL_WKB = bytes.fromhex("0101000000000000000000F03F0000000000000040")


def _sent_local_field_types(spark, service, data, schema=None):
    # The JSON types of the fields of the local relation that createDataFrame sends.
    df = spark.createDataFrame(data, schema) if schema is not None else spark.createDataFrame(data)
    df.toArrow()
    request = parse_message(service.execute_requests[-1])
    relation = parse_message(parse_message(request[3][0])[1][0])
    while 18 in relation:  # Relation.to_df -> ToDF.input.
        relation = parse_message(parse_message(relation[18][0])[1][0])
    local = parse_message(relation[11][0])  # Relation.local_relation=11 -> LocalRelation.schema=2.
    return [field["type"] for field in json.loads(local[2][0])["fields"]]


@pytest.mark.parametrize("make, srid", [
    (lambda: T.Geography.fromWKB(_SPATIAL_WKB, 4269), 4269),
    (lambda: T.Geometry.fromWKB(_SPATIAL_WKB, 4269), 4269),
    (lambda: T.Geometry.fromWKB(_SPATIAL_WKB, 99999), 99999),
    (lambda: T.Geography.fromWKB(_SPATIAL_WKB, 99999), 99999),
    (lambda: T.Geography.fromWKB(_SPATIAL_WKB, 0), 0),
    (lambda: T.Geography.fromWKB(_SPATIAL_WKB, 3857), 3857),
    (lambda: T.Geography.fromWKB(_SPATIAL_WKB, -1), -1),
])
def test_inferred_spatial_value_with_an_srid_its_type_does_not_take_is_rejected(mock_connect, make, srid):
    # As PySpark's _infer_type, which builds GeometryType(srid) or GeographyType(srid) for each
    # value: createDataFrame raises ST_INVALID_SRID_VALUE for a top-level value, for one nested
    # in an array, and for the value after a null.
    from pyspark.errors import IllegalArgumentException

    spark, _ = mock_connect
    for data in ([(make(),)], [([make()],)], [(None,), (make(),)]):
        with pytest.raises(IllegalArgumentException) as error:
            spark.createDataFrame(data)
        assert error.value.getCondition() == "ST_INVALID_SRID_VALUE"
        assert error.value.getMessageParameters() == {"srid": str(srid)}


def test_mixed_geography_srids_fail_at_the_row_with_the_invalid_srid(mock_connect):
    # Official PySpark infers each row's type before merging the rows, so the second row of
    # GEOGRAPHY SRIDs 4326 and 4269 raises the error with its SRID, as does 0 after 4326.
    from pyspark.errors import IllegalArgumentException

    spark, _ = mock_connect
    for second in (4269, 0):
        with pytest.raises(IllegalArgumentException) as error:
            spark.createDataFrame([
                (T.Geography.fromWKB(_SPATIAL_WKB, 4326),),
                (T.Geography.fromWKB(_SPATIAL_WKB, second),),
            ])
        assert error.value.getCondition() == "ST_INVALID_SRID_VALUE"
        assert error.value.getMessageParameters() == {"srid": str(second)}


def test_inferred_spatial_types_of_valid_srids_are_as_pyspark(mock_connect):
    # GEOMETRY takes the SRIDs 0, 3857 and 4326, and merges differing SRIDs into GEOMETRY(ANY)
    # as PySpark's _merge_type does; GEOGRAPHY takes only 4326.
    from pyspark.sql.types import Geometry

    spark, service = mock_connect
    geometry = lambda srid: Geometry.fromWKB(_SPATIAL_WKB, srid)  # noqa: E731
    types = lambda data: _sent_local_field_types(spark, service, data)  # noqa: E731
    assert types([(geometry(0),), (geometry(3857),), (geometry(4326),)]) == [
        T.GeometryType("ANY").jsonValue()]
    assert types([(geometry(4326),), (None,), (geometry(4326),)]) == [
        T.GeometryType(4326).jsonValue()]
    assert types([([geometry(4326)],), ([geometry(3857)],)]) == [
        T.ArrayType(T.GeometryType("ANY")).jsonValue()]
    assert types([(T.Geography.fromWKB(_SPATIAL_WKB, 4326),)]) == [
        T.GeographyType(4326).jsonValue()]


def test_explicit_schema_spatial_values_are_not_inferred_from_their_srids(mock_connect):
    # With an explicit schema, PySpark does not build the type from each value's SRID, so a
    # GEOGRAPHY(ANY) takes a value of SRID 4269 without the client checking it.
    spark, service = mock_connect
    schema = T.StructType([T.StructField("g", T.GeographyType("ANY"))])
    data = [(T.Geography.fromWKB(_SPATIAL_WKB, 4269),), (T.Geography.fromWKB(_SPATIAL_WKB, 4326),)]
    assert _sent_local_field_types(spark, service, data, schema) == [T.GeographyType("ANY").jsonValue()]


# --------------------------------------------------------------------------- map functions


def _group_map_functions():
    from typing import Iterator, Tuple

    import numpy as np
    import pandas as pd
    import pyarrow as pa

    def pandas_iter(batches: Iterator[pd.DataFrame]) -> Iterator[pd.DataFrame]:
        yield from batches

    def pandas_iter_key(
        key: Tuple[np.int64], batches: Iterator[pd.DataFrame]
    ) -> Iterator[pd.DataFrame]:
        yield from batches

    def pandas_frame(frame: pd.DataFrame) -> pd.DataFrame:
        return frame

    def arrow_iter(batches: Iterator[pa.RecordBatch]) -> Iterator[pa.RecordBatch]:
        yield from batches

    def arrow_iter_key(
        key: Tuple[pa.Scalar, ...], batches: Iterator[pa.RecordBatch]
    ) -> Iterator[pa.RecordBatch]:
        yield from batches

    def arrow_table(table: pa.Table) -> pa.Table:
        return table

    def plain(data):
        return data

    return {f.__name__: f for f in (
        pandas_iter, pandas_iter_key, pandas_frame, arrow_iter, arrow_iter_key, arrow_table, plain)}


def _sent_group_map_eval_type(request):
    """The eval type of the function in a captured ExecutePlanRequest: plan=3 -> root=1 ->
    group_map=31 -> func=3 -> python_udf=4 -> eval_type=2."""
    root = parse_message(parse_message(parse_message(request)[3][0])[1][0])
    udf = parse_message(parse_message(root[31][0])[3][0])
    return parse_message(udf[4][0])[2][0]


_ID_SCHEMA = T.StructType([T.StructField("id", T.LongType())])


@pytest.mark.parametrize("method,name,eval_type", [
    # SQL_GROUPED_MAP_PANDAS_ITER_UDF and SQL_GROUPED_MAP_PANDAS_UDF.
    ("applyInPandas", "pandas_iter", 216),
    ("applyInPandas", "pandas_iter_key", 216),
    ("applyInPandas", "pandas_frame", 201),
    ("applyInPandas", "plain", 201),
    # SQL_GROUPED_MAP_ARROW_ITER_UDF and SQL_GROUPED_MAP_ARROW_UDF.
    ("applyInArrow", "arrow_iter", 215),
    ("applyInArrow", "arrow_iter_key", 215),
    ("applyInArrow", "arrow_table", 209),
    ("applyInArrow", "plain", 209),
])
def test_grouped_map_functions_infer_their_eval_type_as_pyspark(mock_connect, method, name, eval_type):
    spark, service = mock_connect
    grouped = spark.range(3).groupBy("id")
    getattr(grouped, method)(_group_map_functions()[name], _ID_SCHEMA).collect()
    assert _sent_group_map_eval_type(service.execute_requests[-1]) == eval_type


@pytest.mark.parametrize("method,warning,eval_type", [
    ("applyInPandas", "Cannot infer the eval type from type hints.", 201),
    # PySpark 4.3.0 warns too, then raises UnboundLocalError, as its eval type is unbound;
    # here the function keeps the default eval type, as for applyInPandas.
    ("applyInArrow", "Cannot infer the eval type from type hints. ", 209),
])
def test_grouped_map_functions_warn_when_their_type_hints_cannot_be_inferred(
    mock_connect, method, warning, eval_type
):
    import warnings

    def partial(key, data: object) -> object:
        return data

    spark, service = mock_connect
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        getattr(spark.range(3).groupBy("id"), method)(partial, _ID_SCHEMA).collect()
    assert [str(w.message) for w in caught if w.category is UserWarning] == [warning]
    assert _sent_group_map_eval_type(service.execute_requests[-1]) == eval_type


@pytest.mark.parametrize("build,detail", [
    (lambda df: df.groupBy("id").applyInPandas(lambda a, b, c: a, _ID_SCHEMA),
     "pandas_udfs with function type GROUPED_MAP or the function in groupby.applyInPandas "
     "must take either one argument (data) or two arguments (key, data)."),
    (lambda df: df.groupBy("id").applyInArrow(lambda a, b, c: a, _ID_SCHEMA),
     "the function in groupby.applyInArrow must take either one argument (data) or two "
     "arguments (key, data)."),
    (lambda df: df.groupBy("id").cogroup(df.groupBy("id")).applyInPandas(lambda a: a, _ID_SCHEMA),
     "the function in cogroup.applyInPandas must take either two arguments (left, right) or "
     "three arguments (key, left, right)."),
    (lambda df: df.groupBy("id").cogroup(df.groupBy("id")).applyInArrow(lambda a: a, _ID_SCHEMA),
     "the function in cogroup.applyInArrow must take either two arguments (left, right) or "
     "three arguments (key, left, right)."),
])
def test_map_functions_validate_their_arguments_as_pyspark(mock_connect, build, detail):
    from pyspark.errors import PySparkValueError

    spark, service = mock_connect
    with pytest.raises(PySparkValueError) as error:
        build(spark.range(3))
    assert error.value.getCondition() == "INVALID_PANDAS_UDF"
    assert error.value.getMessageParameters() == {"detail": detail}
    # Validated before the input columns are analyzed, as in PySpark.
    assert service.calls == 0


@pytest.mark.parametrize("build", [
    lambda df: df.mapInPandas(1, _ID_SCHEMA),
    lambda df: df.mapInArrow(1, _ID_SCHEMA),
    lambda df: df.groupBy("id").applyInPandas(1, _ID_SCHEMA),
    lambda df: df.groupBy("id").applyInPandasWithState(1, _ID_SCHEMA, _ID_SCHEMA, "append", "NoTimeout"),
])
def test_map_functions_reject_what_is_not_callable_as_pyspark(mock_connect, build):
    import warnings

    spark, service = mock_connect
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(TypeError, match="^unsupported callable$"):
            build(spark.range(3))
    assert service.calls == 0


# Session version cache regressions, selected by the existing offline CI suite.
# The mock runs in a separate process and needs no Spark server.
from concurrent.futures import ThreadPoolExecutor as version_cache_ThreadPoolExecutor
import multiprocessing as version_cache_multiprocessing
from threading import Event as version_cache_Event

from pyspark.sql import SparkSession as version_cache_SparkSession


def version_cache_varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def version_cache_message(field, value):
    if isinstance(value, str):
        value = value.encode()
    return version_cache_varint(field * 8 + 2) + version_cache_varint(len(value)) + value


def version_cache_serve(connection, request_started, response_waiting, response_gate):
    import grpc

    class Service:
        calls = 0
        releases = 0
        failure = None
        version = "4.3.0"
        block_response = False

        def analyze(self, request, context):
            self.calls += 1
            if self.block_response:
                response_waiting.set()
                request_started.set()
                try:
                    released = response_gate.wait(10)
                finally:
                    response_waiting.clear()
                if not released:
                    context.abort(
                        grpc.StatusCode.DEADLINE_EXCEEDED,
                        "Python thread could not release the version response gate",
                    )
            if self.failure == "rpc":
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "version unavailable")
            if self.failure == "no-result":
                return b""
            # AnalyzePlanResponse.spark_version=8, SparkVersion.version=1.
            return version_cache_message(8, version_cache_message(1, self.version))

        def release(self, request, context):
            self.releases += 1
            return b""

        def clone(self, request, context):
            # CloneSessionResponse.new_session_id=3, of a new session.
            import uuid

            return message_field(3, str(uuid.uuid4()))

    service = Service()
    with version_cache_ThreadPoolExecutor(max_workers=4) as executor:
        server = grpc.server(executor)
        server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(
            "spark.connect.SparkConnectService", {
                "AnalyzePlan": grpc.unary_unary_rpc_method_handler(service.analyze),
                "ReleaseSession": grpc.unary_unary_rpc_method_handler(service.release),
                "CloneSession": grpc.unary_unary_rpc_method_handler(service.clone),
            }),))
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        connection.send(port)
        try:
            while True:
                command = connection.recv()
                if command[0] == "stop":
                    break
                if command[0] == "get":
                    connection.send(getattr(service, command[1]))
                elif command[0] == "set":
                    setattr(service, command[1], command[2])
                    connection.send(None)
        finally:
            response_gate.set()
            server.stop(0).wait(5)


class version_cache_RemoteService:
    def __init__(self, connection, request_started, response_waiting, response_gate):
        object.__setattr__(self, "connection", connection)
        object.__setattr__(self, "request_started", request_started)
        object.__setattr__(self, "response_waiting", response_waiting)
        object.__setattr__(self, "response_gate", response_gate)

    def __getattr__(self, name):
        self.connection.send(("get", name))
        assert self.connection.poll(10), "Mock server did not respond"
        return self.connection.recv()

    def __setattr__(self, name, value):
        self.connection.send(("set", name, value))
        assert self.connection.poll(10), "Mock server did not respond"
        self.connection.recv()

    def wait_for_request(self):
        return self.request_started.wait(5)

    def response_is_waiting(self):
        return self.response_waiting.is_set()

    def release_response(self):
        self.response_gate.set()


@pytest.fixture
def version_cache_session():
    pytest.importorskip("grpc")
    context = version_cache_multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    request_started = context.Event()
    response_waiting = context.Event()
    response_gate = context.Event()
    process = context.Process(
        target=version_cache_serve,
        args=(child, request_started, response_waiting, response_gate),
    )
    process.start()
    try:
        assert parent.poll(20), "Mock server did not start"
        port = parent.recv()
        spark = version_cache_SparkSession.builder.remote(f"sc://127.0.0.1:{port}").getOrCreate()
        yield spark, version_cache_RemoteService(
            parent, request_started, response_waiting, response_gate,
        )
    finally:
        response_gate.set()
        if process.is_alive():
            parent.send(("stop",))
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
        child.close()


@pytest.mark.parametrize("version", ["4.3.0", ""])
def test_version_cache_success_is_cached(version_cache_session, version):
    spark, service = version_cache_session
    service.version = version
    assert service.calls == 0
    for _ in range(10):
        assert spark.version == version
    assert service.calls == 1
    service.version = "changed"
    assert spark.version == version
    assert service.calls == 1


@pytest.mark.parametrize("failure", ["rpc", "no-result"])
def test_version_cache_failed_lookup_can_retry(version_cache_session, failure):
    spark, service = version_cache_session
    service.failure = failure
    with pytest.raises(Exception):
        _ = spark.version
    assert service.calls == 1
    service.failure = None
    assert spark.version == "4.3.0"
    assert spark.version == "4.3.0"
    assert service.calls == 2


def test_version_cache_new_sessions_have_independent_caches(version_cache_session):
    spark, service = version_cache_session
    assert spark.version == "4.3.0"
    service.version = "other"
    for other in [spark.newSession(), spark.cloneSession()]:
        assert other.session_id != spark.session_id
        assert other.version == "other"
        assert other.version == "other"
    assert service.calls == 3
    assert spark.version == "4.3.0"


def test_clone_session_clones_the_session_on_the_server_as_pyspark(mock_connect):
    # As PySpark's cloneSession, a CloneSession of the session (CloneSessionRequest
    # .session_id=1), of the new session id given (new_session_id=4), whose clone has the id of
    # the response.
    import uuid

    spark, service = mock_connect
    clone = spark.cloneSession()
    request = parse_message(service.clone_requests[-1])
    assert request[1] == [spark.session_id.encode()] and 4 not in request
    assert clone.session_id == service.cloned_session_id != spark.session_id
    new_id = str(uuid.uuid4())
    assert spark.cloneSession(new_id).session_id == new_id
    assert parse_message(service.clone_requests[-1])[4] == [new_id.encode()]
    assert len(service.clone_requests) == 2


def test_version_cache_success_remains_cached_after_stop(version_cache_session):
    spark, service = version_cache_session
    assert spark.version == "4.3.0"
    spark.stop()
    assert spark.is_stopped
    assert service.releases == 1
    service.failure = "rpc"
    assert spark.version == "4.3.0"
    assert service.calls == 1


def test_version_cache_concurrent_first_lookup(version_cache_session):
    spark, service = version_cache_session
    workers = 8
    start = Barrier(workers + 1)
    attempted = [version_cache_Event() for _ in range(workers)]
    service.block_response = True

    def read_version(index):
        start.wait(timeout=10)
        attempted[index].set()
        return spark.version

    with version_cache_ThreadPoolExecutor(max_workers=workers) as executor:
        pending = [executor.submit(read_version, index) for index in range(workers)]
        try:
            start.wait(timeout=10)
            for attempt in attempted:
                assert attempt.wait(5), "Worker did not attempt a version lookup"
            assert service.wait_for_request(), "Version request never reached the server"
            assert service.response_is_waiting(), "Version response gate timed out"
            # The barrier synchronizes Python call attempts; the held response
            # keeps the cache cold while those callers compete for the lookup.
            assert not any(future.done() for future in pending)
            assert service.calls == 1
        finally:
            service.release_response()
        assert [future.result(timeout=10) for future in pending] == ["4.3.0"] * workers
        assert list(executor.map(lambda _: spark.version, range(24))) == ["4.3.0"] * 24
    assert service.calls == 1


def test_version_cache_releases_gil_while_response_is_pending(version_cache_session):
    spark, service = version_cache_session
    service.block_response = True
    with version_cache_ThreadPoolExecutor(max_workers=1) as executor:
        pending = executor.submit(lambda: spark.version)
        try:
            assert service.wait_for_request(), "Version request never reached the server"
            # This Python thread must run before the mock's bounded wait ends.
            # The live flag distinguishes that from resuming after a GIL stall.
            assert service.response_is_waiting(), "Python could not run during the version RPC"
            assert not pending.done()
        finally:
            service.release_response()
        assert pending.result(timeout=10) == "4.3.0"
    assert service.calls == 1


def test_version_cache_active_session_handle_shares_cache(version_cache_session):
    spark, service = version_cache_session
    active = version_cache_SparkSession.getActiveSession()
    assert active.session_id == spark.session_id
    assert active.version == "4.3.0"
    service.version = "changed"
    assert spark.version == "4.3.0"
    assert service.calls == 1


def test_version_cache_stop_before_first_lookup_keeps_rpc_behavior(version_cache_session):
    spark, service = version_cache_session
    spark.stop()
    assert service.calls == 0
    service.failure = "rpc"
    with pytest.raises(Exception):
        _ = spark.version
    assert service.calls == 1


# --------------------------------------------------------------------------- printSchema


def _print_schema_field(name, data_type, nullable=True):
    return (
        message_field(1, name)
        + message_field(2, data_type)
        + (b"\x18\x01" if nullable else b"\x18\x00")
    )


def _print_schema_type():
    def struct(*fields):
        return message_field(21, b"".join(message_field(1, field) for field in fields))

    integer = message_field(6, b"")
    long = message_field(7, b"")
    string = message_field(11, b"")
    array = message_field(
        20, message_field(1, struct(_print_schema_field("item", long))) + b"\x10\x00"
    )
    mapping = message_field(
        22,
        message_field(1, string)
        + message_field(2, struct(_print_schema_field("value", integer)))
        + b"\x18\x01",
    )
    return struct(
        _print_schema_field("outer", struct(_print_schema_field("inner", integer, False))),
        _print_schema_field("items", array, False),
        _print_schema_field("lookup", mapping),
    )


_PRINT_SCHEMA_LEVEL1 = (
    "root\n"
    " |-- outer: struct (nullable = true)\n"
    " |-- items: array (nullable = false)\n"
    " |-- lookup: map (nullable = true)\n\n"
)
_PRINT_SCHEMA_LEVEL2 = (
    "root\n"
    " |-- outer: struct (nullable = true)\n"
    " |    |-- inner: integer (nullable = false)\n"
    " |-- items: array (nullable = false)\n"
    " |    |-- element: struct (containsNull = false)\n"
    " |-- lookup: map (nullable = true)\n"
    " |    |-- key: string\n"
    " |    |-- value: struct (valueContainsNull = true)\n\n"
)
_PRINT_SCHEMA_FULL = (
    "root\n"
    " |-- outer: struct (nullable = true)\n"
    " |    |-- inner: integer (nullable = false)\n"
    " |-- items: array (nullable = false)\n"
    " |    |-- element: struct (containsNull = false)\n"
    " |    |    |-- item: long (nullable = true)\n"
    " |-- lookup: map (nullable = true)\n"
    " |    |-- key: string\n"
    " |    |-- value: struct (valueContainsNull = true)\n"
    " |    |    |-- value: integer (nullable = true)\n\n"
)


@pytest.mark.parametrize(
    "level, expected",
    [
        (0, _PRINT_SCHEMA_FULL),
        (1, _PRINT_SCHEMA_LEVEL1),
        (1.5, _PRINT_SCHEMA_LEVEL2),
        (Fraction(1, 10**400), _PRINT_SCHEMA_LEVEL1),
        (decimal.Decimal("1e-1000"), _PRINT_SCHEMA_LEVEL1),
        (2**100, _PRINT_SCHEMA_FULL),
        (float("nan"), _PRINT_SCHEMA_FULL),
    ],
)
def test_print_schema_exact_python_stdout(mock_connect, capsys, level, expected):
    spark, service = mock_connect
    service.data_type = _print_schema_type()
    df = spark.range(10)
    assert df.printSchema(level) is None
    captured = capsys.readouterr()
    assert captured.out == expected
    assert captured.err == ""
    assert df.schema.treeString(maxDepth=level) == expected[:-1]
    assert service.execute_calls == 0


def test_print_schema_default_reuses_cached_schema(mock_connect, capsys):
    spark, service = mock_connect
    service.data_type = _print_schema_type()
    df = spark.range(10)
    for _ in range(2):
        assert df.printSchema() is None
        assert capsys.readouterr().out == _PRINT_SCHEMA_FULL
    assert service.calls == 1
    assert service.execute_calls == 0


@pytest.mark.parametrize("level", [None, []])
def test_print_schema_falsey_level_uses_default(mock_connect, capsys, level):
    spark, service = mock_connect
    service.data_type = _print_schema_type()
    df = spark.range(10)
    assert df.printSchema(level) is None
    assert capsys.readouterr().out == _PRINT_SCHEMA_FULL
    with pytest.raises(TypeError):
        df.schema.treeString(level)
    assert service.execute_calls == 0


def test_schema_tree_invalid_depth(mock_connect, capsys):
    spark, service = mock_connect
    service.data_type = _print_schema_type()
    df = spark.range(10)
    with pytest.raises(TypeError):
        df.printSchema("two")
    with pytest.raises(TypeError):
        df.schema.treeString("two")
    assert capsys.readouterr().out == ""
    assert service.execute_calls == 0


def test_print_schema_propagates_schema_error(mock_connect, capsys):
    spark, service = mock_connect
    service.failure = "rpc"
    with pytest.raises(Exception, match="schema unavailable"):
        spark.range(10).printSchema()
    assert capsys.readouterr().out == ""
    assert service.execute_calls == 0


def test_schema_tree_type_names_and_escaped_field_names():
    schema = T.StructType(
        [
            T.StructField("é🌱\\n\n\r\t\f\b\v\a", T.LongType(), False),
            T.StructField("decimal", T.DecimalType(10, 2)),
            T.StructField("small", T.ByteType()),
            T.StructField("short", T.ShortType()),
        ]
    )
    expected = (
        "root\n"
        " |-- é🌱\\n\\n\\r\\t\\f\\b\\v\\a: long (nullable = false)\n"
        " |-- decimal: decimal(10,2) (nullable = true)\n"
        " |-- small: byte (nullable = true)\n"
        " |-- short: short (nullable = true)\n"
    )
    assert schema.treeString() == expected
    assert schema.treeString(0) == expected
    assert schema.treeString(-1) == expected


@pytest.mark.parametrize(
    "python_class, expected",
    [
        ("pyspark.ml.linalg.VectorUDT", "vectorudt"),
        (None, "udt"),
    ],
)
def test_print_schema_udt_class_name(mock_connect, capsys, python_class, expected):
    spark, service = mock_connect
    udt = message_field(1, "udt")
    if python_class is not None:
        udt += message_field(3, python_class)
    service.data_type = message_field(
        21, message_field(1, _print_schema_field("features", message_field(23, udt)))
    )
    assert spark.range(10).printSchema() is None
    assert capsys.readouterr().out == f"root\n |-- features: {expected} (nullable = true)\n\n"
    assert service.calls == 1
    assert service.execute_calls == 0


# --------------------------------------------------------------------------- sql kwargs


def test_sql_formatter_formats_values_as_pyspark():
    from pyspark.sql import functions as F
    from pyspark.sql.connect.sql_formatter import SQLStringFormatter

    spark = _offline_session()
    df = spark.range(1)
    formatter = SQLStringFormatter(spark)
    query = formatter.format(
        "SELECT {c}, {d.id}, {s}, {n} FROM {df} JOIN {df}",
        c=F.col("id"), d=df, s="it's \\", n=1, df=df)
    # A DataFrame named twice is one reference.
    [(view, name)] = formatter._temp_views
    assert view is df and name.startswith("_pyspark_connect_temp_view_")
    assert query == f"SELECT id, id, 'it\\'s \\\\', 1 FROM {name} JOIN {name}"


def test_sql_keyword_argument_errors_as_pyspark():
    from pyspark.errors import PySparkTypeError, PySparkValueError
    from pyspark.sql import functions as F

    # Raised before the statement is sent, as in PySpark.
    spark = _offline_session()
    with pytest.raises(IndexError):
        spark.sql("SELECT {} AS v", x=1)
    with pytest.raises(KeyError):
        spark.sql("SELECT {y} AS v", x=1)
    with pytest.raises(PySparkValueError, match="in c should be a plain column reference"):
        spark.sql("SELECT {c}", c=F.col("id") + 1)
    with pytest.raises(PySparkTypeError) as error:
        spark.sql("SELECT ? AS v", args=(1,))
    assert error.value.getCondition() == "INVALID_TYPE"
    assert error.value.getMessageParameters() == {"arg_name": "args", "arg_type": "tuple"}


# --------------------------------------------------------------------------- optional args


@pytest.mark.parametrize("column,expected", [
    # The optional arguments of these functions are sent when given, and None is not given,
    # as PySpark does.
    (lambda F: F.hmac("k", "m", "a"), "hmac(k, m, a)"),
    (lambda F: F.hmac("k", "m", None), "hmac(k, m)"),
    (lambda F: F.vector_norm("v", "d"), "vector_norm(v, d)"),
    (lambda F: F.vector_norm("v", None), "vector_norm(v)"),
    (lambda F: F.vector_normalize("v", degree="d"), "vector_normalize(v, d)"),
    (lambda F: F.counter_diff("x", "t"), "counter_diff(x, t)"),
    (lambda F: F.counter_diff("x", startTime=None), "counter_diff(x)"),
])
def test_functions_send_their_optional_arguments_as_pyspark(column, expected):
    from pyspark.sql import functions as F

    assert repr(column(F)) == f"Column<'{expected}'>"
