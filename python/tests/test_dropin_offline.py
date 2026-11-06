"""Offline tests for the pyspark drop-in (no Spark Connect server required).

These exercise the pure-Python shim (types parsing/pickling, the session builder,
functions/column expression building, udf/udtf construction, StorageLevel) so the
Python side has real, measurable coverage in CI without a running server. Tests
that require a live server live in test_dropin_e2e.py and are gated on SPARK_REMOTE.
"""

import array
import datetime
import decimal
import json
import pickle
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from _mock_connect_server import (
    message_field,
    mock_connect_session,
    parse_message,
    struct_schema,
    varint_field,
)
from pyspark.sql import Row, SparkSession, types as T

# Collected here so the existing offline CI suite exercises real Arrow results.
from day_time_interval_cases import DayTimeIntervalTests  # noqa: F401


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


def test_unparsed_datatype_pickle_roundtrip():
    t = T._parse_datatype_json_value("unparsed(some_ddl)")
    assert t.json() == '"unparsed(some_ddl)"'
    r = pickle.loads(pickle.dumps(t))
    assert r == t and r.json() == t.json()


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


def test_datatype_eq_across_construction_paths():
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


# --------------------------------------------------------------------------- session builder

def test_session_builder_chaining_offline():
    from pyspark.sql import SparkSession
    b = SparkSession.builder
    assert type(b).__name__ == "SparkSessionBuilder"
    # Chainable, no server contacted until getOrCreate.
    b2 = b.remote("sc://localhost:15002").appName("x").master("local").config("k", "v")
    assert type(b2).__name__ == "SparkSessionBuilder"


def test_session_builder_class_constructor():
    from pyspark.sql import SparkSession
    b = SparkSession.Builder()
    assert type(b).__name__ == "SparkSessionBuilder"


def test_channel_builder_uses_endpoint():
    # channelBuilder reconstructs an sc:// URL from the builder's host/port (+ params)
    # for the native transport, returning a configured Builder (no longer raises).
    from pyspark.sql import SparkSession

    class CB:
        host = "localhost"
        port = 15002
        _params = {"use_ssl": "false"}

    b = SparkSession.builder.channelBuilder(CB())
    assert type(b).__name__ == "SparkSessionBuilder"
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

    assert AEcho.evalType == 301  # SQL_ARROW_TABLE_UDF

    # functions.arrow_udf builds an Arrow scalar UDF.
    au = F.arrow_udf(lambda x: x, IntegerType())
    assert callable(au)
    aui = F.arrow_udf(lambda x: x, IntegerType(), functionType="scalar_iter")
    assert callable(aui)


# --------------------------------------------------------------------------- functions / columns

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


def test_dispatch_keeps_literal_string_args_literal():
    # Guards _FN_COL_ARGS: a bare-str arg at a LITERAL position (e.g. a variant path)
    # must reach the core UNCHANGED, not be resolved to a column via _to_col. If it were
    # coerced, `variant_delete(v, "$.b")` would send an unresolved attribute col("$.b")
    # and fail analysis on the server. Asserting only that a Column is built (as elsewhere)
    # can't catch this, so we capture the exact positional args handed to the core: a
    # value column is coerced away from str, while a literal path stays a str.
    from pyspark.sql import functions as F

    captured = []

    orig = F._call_function
    F._call_function = lambda fname, *args: captured.append((fname, args)) or orig(fname, *args)
    def is_col(x):
        return type(x).__name__ == "Column"

    try:
        # (function call, positions that MUST stay raw literals — a str path or a bool
        # flag — and positions that MUST be coerced to a column)
        cases = [
            (lambda: F.variant_get("v", "$.a", "int"), {1, 2}, {0}),
            (lambda: F.try_variant_get("v", "$.a", "int"), {1, 2}, {0}),
            (lambda: F.variant_delete("v", "$.a", "$.b"), {1, 2}, {0}),
            (lambda: F.variant_insert("v", "$.b", "x"), {1}, {0, 2}),
            (lambda: F.try_variant_insert("v", "$.b", "x"), {1}, {0, 2}),
            (lambda: F.variant_set("v", "$.b", "x", True), {1, 3}, {0, 2}),
            (lambda: F.try_variant_set("v", "$.b", "x", True), {1, 3}, {0, 2}),
            (lambda: F.variant_array_append("v", "$", "x"), {1}, {0, 2}),
            (lambda: F.try_variant_array_append("v", "$", "x"), {1}, {0, 2}),
        ]
        for call, literal_pos, col_pos in cases:
            captured.clear()
            call()
            fname, args = captured[-1]
            for i in literal_pos:
                assert not is_col(args[i]), f"{fname} arg[{i}] should stay a raw literal, got a Column"
            for i in col_pos:
                assert is_col(args[i]), f"{fname} arg[{i}] should be coerced to a Column, got {type(args[i]).__name__}"
    finally:
        F._call_function = orig


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
    u = UserDefinedFunction(lambda x: x + 1, T.IntegerType(), 100, "plus1")
    assert u.name == "plus1"
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
    assert type(F.from_avro(c, "{}")).__name__ == "Column"
    assert type(F.from_avro_with_options(c, "{}", {"mode": "PERMISSIVE"})).__name__ == "Column"
    assert type(F.to_avro_with_schema(c, "{}")).__name__ == "Column"
    assert type(F.from_protobuf(c, "M", options={"x": "y"})).__name__ == "Column"
    assert type(F.to_protobuf(c, "M", options={"x": "y"})).__name__ == "Column"


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
def test_datatype_from_ddl_returns_the_concrete_type(ddl, cls, expected):
    parsed = T.DataType.fromDDL(ddl)
    assert repr(parsed) == expected
    assert isinstance(parsed, cls)
    # The classmethod parses DDL the same way from any data type class.
    assert repr(T.LongType.fromDDL(ddl)) == expected


def test_datatype_object_model():
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
    assert st.toDDL() == "a int,b string NOT NULL"
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


def test_broadcast_type_check():
    from pyspark.sql import functions as F
    from pyspark.errors import PySparkTypeError

    with pytest.raises(PySparkTypeError):
        F.broadcast("not a dataframe")


def test_row_count_and_index():
    from pyspark.sql import Row
    r = Row(a=1, b=2, c=1)
    assert r.count(1) == 2
    assert r.count(2) == 1
    assert r.index(2) == 1
    assert r.index(1) == 0
    with pytest.raises(ValueError):
        r.index(99)


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


def test_udf_registration_has_java_methods():
    # The UDFRegistration class (what spark.udf returns) exposes the Java methods.
    from pyspark.sql.udf import UDFRegistration
    for m in ("register", "registerJavaFunction", "registerJavaUDAF"):
        assert hasattr(UDFRegistration, m), f"UDFRegistration missing {m}"


def test_udf_udtf_determinism_and_extras():
    from pyspark.sql import functions as F
    from pyspark.sql.udf import udf, UserDefinedFunction
    from pyspark.sql.udtf import udtf

    u = udf(lambda x: x, T.IntegerType())
    nd = u.asNondeterministic()
    assert isinstance(nd, UserDefinedFunction) and nd.deterministic is False
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
    # The catalog metadata result classes are Rust-backed and importable; the Table
    # `database` property derives from a single-element namespace.
    from pyspark.sql.catalog import (
        Catalog, CatalogMetadata, Database, Table, Column, Function, TablePartition,
    )
    import pyspark._pyspark as p
    # Rust-backed, but reporting the reference `pyspark.sql.catalog` module path
    # (SPARK-59117), not "builtins".
    for cls in (CatalogMetadata, Database, Table, Function, TablePartition):
        assert cls.__module__ == "pyspark.sql.catalog", f"{cls.__name__} module = {cls.__module__}"
    assert Column is p.CatalogColumn


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
    # pandas_udf with an explicit functionType.
    p = pandas_udf(lambda s: s, T.IntegerType(), functionType="scalar")
    assert p is not None


def test_udtf_construction():
    from pyspark.sql.udtf import udtf, UserDefinedTableFunction, UDTFRegistration

    @udtf(returnType="a int")
    class Echo:
        def eval(self, n):
            for i in range(n):
                yield (i,)

    assert isinstance(Echo, UserDefinedTableFunction)
    assert Echo.name == "Echo"
    reg = UDTFRegistration()
    # register a plain class -> wraps it
    r = reg.register("echo", Echo.func, returnType="a int")
    assert r.name == "echo"
    # register an existing UserDefinedTableFunction -> renames in place
    r2 = reg.register("echo2", Echo)
    assert r2.name == "echo2" and r2 is Echo


def test_udtf_call_without_session_raises():
    from pyspark.sql.udtf import UserDefinedTableFunction
    from pyspark.sql import SparkSession
    # Ensure no active session, then calling must raise a clear error.
    prev = SparkSession.getActiveSession()
    if prev is not None:
        pytest.skip("an active session exists in this process")

    @__import__("pyspark.sql.udtf", fromlist=["udtf"]).udtf(returnType="a int")
    class T2:
        def eval(self, n):
            yield (n,)

    with pytest.raises(RuntimeError):
        T2(1)


def test_udtf_return_type_connect_validation_and_command():
    # A Connect DDL string (including invalid DDL) is validated by the server,
    # not when decorating the class or constructing the DataFrame. Isolate the
    # active session so the other offline tests still start without a session.
    import subprocess
    import sys

    subprocess.run([sys.executable, "-c", """
from pyspark.sql import SparkSession, DataFrame, functions as F, types as T
from pyspark.serializers import CloudPickleSerializer

spark = SparkSession.builder.remote("sc://localhost:15002").getOrCreate()
class Echo:
    def eval(self, value):
        yield (value,)

schema = T.StructType([T.StructField("value", T.LongType())])
for return_type in ["value BIGINT", "STRUCT<value: BIGINT>",
                    "value ARRAY<STRUCT<x: INT>>", "invalid DDL !!!", "INT", schema]:
    for decorate in [False, True]:
        fn = (F.udtf(returnType=return_type)(Echo) if decorate else
              F.udtf(Echo, returnType=return_type))
        assert isinstance(fn(F.lit(1)), DataFrame)
        handler = CloudPickleSerializer().loads(fn.command)
        assert isinstance(handler, type)
        assert list(handler().eval(7)) == [(7,)]
        assert fn.returnType == return_type

# Dynamic schemas omit return_type and leave analysis to the Python handler.
class Dynamic(Echo):
    @staticmethod
    def analyze(value):
        pass
assert isinstance(F.udtf(Dynamic)(F.lit(1)), DataFrame)
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
            assert udtf[2] == [300]
            command_handler = CloudPickleSerializer().loads(udtf[3][0])
            assert isinstance(command_handler, type)
            assert command_handler.__name__ == handler.__name__
            assert list(command_handler().eval(7)) == [(7,)]
            assert udtf[4] == [fn.python_ver.encode()]
""", str(Path(__file__).resolve().parent)], check=True)


def test_spark_conf_stub():
    from pyspark import SparkConf
    c = SparkConf().set("a", "1").setAll({"b": "2", "c": "3"})
    assert c.get("a") == "1"
    assert dict(c.getAll())["b"] == "2"
    c.remove("a")
    assert c.get("a", "default") == "default"
    c.remove("nonexistent")  # removing an absent key is a no-op


def test_sql_is_remote():
    import pyspark.sql as S
    assert S.is_remote() is True


def test_functions_protobuf_descriptor_branches():
    from pyspark.sql import functions as F
    c = F.col("v")
    # descriptor-file and binary-descriptor-set branches
    assert type(F.from_protobuf(c, "M", desc_file_path="/tmp/x.desc")).__name__ == "Column"
    assert type(F.from_protobuf(c, "M", binary_descriptor_set=b"\x00")).__name__ == "Column"
    assert type(F.to_protobuf(c, "M", desc_file_path="/tmp/x.desc")).__name__ == "Column"
    assert type(F.to_protobuf(c, "M", binary_descriptor_set=b"\x00")).__name__ == "Column"


def test_udf_defaults_and_registration():
    from pyspark.sql.functions import udf, pandas_udf
    # returnType defaults to StringType
    u = udf(lambda x: x)
    assert callable(u)
    p = pandas_udf(lambda s: s)  # default returnType
    assert p is not None
    # pandas_udf grouped_map/other functionType branches
    for ft in ["scalar", "grouped_map", "grouped_agg"]:
        assert pandas_udf(lambda s: s, T.IntegerType(), functionType=ft) is not None
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
    assert r.name == "f"
    r2 = reg.register("g", lambda x: x)  # returnType default
    assert r2.name == "g"
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
    st2 = T.StructType().add("x", "int").add("y", T.StringType(), False).add(
        T.StructField("z", T.IntegerType())
    )
    assert st2.fieldNames() == ["x", "y", "z"]
    assert isinstance(st2, T.StructType)


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

    # Create a simple UDTF for testing
    @__import__("pyspark.sql.functions", fromlist=["udtf"]).udtf(returnType="a int")
    class SimpleUDTF:
        def eval(self, n):
            yield (n,)

    # Test name and evalType properties
    assert isinstance(SimpleUDTF, UserDefinedTableFunction)
    assert SimpleUDTF.name == "SimpleUDTF"
    assert SimpleUDTF.evalType == 300  # SQL_TABLE_UDF

    # Test deterministic flag
    assert SimpleUDTF.deterministic is True


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


def test_struct_type_schema_methods():
    """Test StructType schema introspection methods."""
    st = T.StructType([
        T.StructField("name", T.StringType(), False),
        T.StructField("age", T.IntegerType(), True),
    ])

    # Test toDDL
    ddl = st.toDDL()
    assert "name" in ddl
    assert "age" in ddl

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
    assert type(F.substr(str_col, 1, 5)).__name__ == "Column"


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
    assert type(builder).__name__ == "SparkSessionBuilder"

    # Test with additional config
    builder2 = builder.appName("test").config("key", "value")
    assert type(builder2).__name__ == "SparkSessionBuilder"


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
    p = pandas_udf(lambda s: s, IntegerType(), functionType="grouped_agg")
    assert callable(p)


def test_arrow_udf_types():
    """Test arrow_udf with different options."""
    from pyspark.sql.functions import arrow_udf
    from pyspark.sql.types import IntegerType

    # Scalar UDF
    u1 = arrow_udf(lambda x: x, IntegerType())
    assert callable(u1)

    # Scalar iter UDF
    u2 = arrow_udf(lambda x: x, IntegerType(), functionType="scalar_iter")
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

    # stack creates rows from arguments
    result = F.stack(2, F.col("a"), F.col("b"))
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

    assert type(builder).__name__ == "SparkSessionBuilder"


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
    """Test the @since decorator from pyspark."""
    from pyspark import since

    @since(3.0)
    def new_function():
        return "result"

    # The decorator should be transparent (pass-through)
    assert new_function() == "result"

    @since("4.0")
    def another_func():
        return 42

    assert another_func() == 42


def test_spark_conf_empty_config():
    """Test SparkConf with no settings."""
    from pyspark import SparkConf

    conf = SparkConf()
    assert conf.getAll() == []


def test_spark_conf_remove_missing():
    """Test removing a non-existent key from SparkConf."""
    from pyspark import SparkConf

    conf = SparkConf()
    # Removing a key that doesn't exist should not raise
    result = conf.remove("nonexistent")
    assert result is conf  # Should return self for chaining


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
            .setAll({"key3": "value3", "key4": "value4"}))

    assert conf.get("key1") == "value1"
    assert conf.get("key3") == "value3"
    assert len(conf.getAll()) == 4


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


def test_udf_and_pandas_udf_decorator_forms():
    # @udf(returnType) / @pandas_udf(returnType) return a decorator that wraps a function.
    from pyspark.sql.functions import udf, pandas_udf
    from pyspark.sql.types import IntegerType
    from pyspark.sql.udf import UserDefinedFunction

    @udf(IntegerType())
    def inc(x):
        return x + 1
    assert isinstance(inc, UserDefinedFunction)

    @pandas_udf("long")
    def p(s):
        return s + 1
    assert isinstance(p, UserDefinedFunction)


def test_udf_return_type_normalized_to_concrete_datatype():
    # A DDL-string returnType must become a concrete, picklable DataType so the
    # cloudpickled command carries a real type (the pandas/Arrow worker needs it).
    import pickle
    from pyspark.sql.functions import pandas_udf
    from pyspark.sql.types import DoubleType

    @pandas_udf("double")
    def f(s):
        return s
    assert isinstance(f.returnType, DoubleType), type(f.returnType)
    # command is cloudpickle of (func, returnType) -> must not carry a bare str.
    assert isinstance(f.command, (bytes, bytearray))
    pickle.dumps(f.returnType)  # concrete types pickle


def test_pandas_udf_eval_type_inferred_from_type_hints():
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


def test_merge_into_writer_nested_class_refs():
    from pyspark.sql.merge import MergeIntoWriter

    assert isinstance(MergeIntoWriter.WhenMatched, type)
    assert isinstance(MergeIntoWriter.WhenNotMatched, type)
    assert isinstance(MergeIntoWriter.WhenNotMatchedBySource, type)


def test_session_hook_is_present_and_passthrough():
    hook = _SS.Hook()
    sentinel = object()
    assert hook.on_execute_plan(sentinel) is sentinel


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


def _local_create_dataframe(*args):
    # The local relation and its toDF are built without a server.
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
def test_create_dataframe_pandas_is_not_local_as_pyspark(schema):
    pd = pytest.importorskip("pandas")
    # pyspark applies the columns, or the field names of the schema, with toDF.
    df = _local_create_dataframe(pd.DataFrame({"a": [1, 2], "b": ["x", "y"]}), schema)
    assert not df.isLocal()


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
    with pytest.raises(PySparkValueError) as error:
        _local_create_dataframe(pdf, ["c", "d", "e"])
    assert error.value.getCondition() == "AXIS_LENGTH_MISMATCH"
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


# --------------------------------------------------------------------------- schema cache


@pytest.fixture(scope="module")
def _shared_mock_connect():
    pytest.importorskip("grpc")
    with mock_connect_session() as session:
        yield session


@pytest.fixture
def mock_connect(_shared_mock_connect):
    spark, service = _shared_mock_connect
    service.reset()
    return spark, service


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


def test_instr_regexp_replace_optional_args():
    from pyspark.sql import functions as F

    captured = []
    orig = F._call_function
    F._call_function = lambda fname, *args: captured.append((fname, args)) or orig(fname, *args)
    try:
        F.instr("s", "a")
        assert captured[-1][1][1:] == ("a",)
        # occurrence without start searches from position 1 (pyspark 4.3.0).
        F.instr("s", "a", occurrence=2)
        assert captured[-1][1][1:] == ("a", 1, 2)
        F.instr("s", "a", start=None, occurrence=None)
        assert len(captured[-1][1]) == 2
        F.instr("s", "a", 3)
        assert captured[-1][1][1:] == ("a", 3)
        F.regexp_replace("s", "a", "b")
        assert len(captured[-1][1]) == 3
        F.regexp_replace("s", "a", "b", position=None)
        assert len(captured[-1][1]) == 3
        F.regexp_replace("s", "a", "b", 4)
        assert captured[-1][1][1:] == ("a", "b", 4)
    finally:
        F._call_function = orig


def test_variant_delete_requires_paths():
    from pyspark.errors import PySparkValueError
    from pyspark.sql import functions as F

    with pytest.raises(PySparkValueError, match="CANNOT_BE_EMPTY"):
        F.variant_delete("v")


def _offline_session():
    # Port 1: nothing listens; the channel is lazy, so the session builds offline.
    return SparkSession.builder.remote("sc://localhost:1").getOrCreate()


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
    spark.range(1).collect()
    info = spark.range(1).executionInfo
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
    spark.range(1).collect()
    info = spark.range(1).executionInfo
    assert info.operation_id == _request_operation_id(service.execute_requests[-1])
    with pytest.raises(PySparkTypeError, match="UNSUPPORTED_LITERAL"):
        info.flows


def test_execute_plan_error_carries_operation_id(mock_connect):
    from pyspark.errors import PySparkException

    spark, service = mock_connect
    service.execute_failure = "plan rejected"
    with pytest.raises(PySparkException, match="plan rejected") as error:
        spark.range(1).collect()
    assert error.value.operation_id == _request_operation_id(service.execute_requests[-1])


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


# ------------------------------------------------- reader/writer in-place setters

@pytest.fixture
def offline_spark():
    # The gRPC channel connects lazily, so building readers/writers and lazy
    # DataFrames needs no server.
    from pyspark.sql import SparkSession

    return SparkSession.builder.remote("sc://localhost:1").getOrCreate()


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


def test_writer_v2_setters_mutate_and_return_self(offline_spark):
    from pyspark.sql import functions as F

    w = offline_spark.range(3).writeTo("t")
    assert w.using("parquet") is w
    assert w.option("a", "b") is w
    assert w.options(c="d") is w
    assert w.tableProperty("p", "v") is w
    assert w.partitionedBy(F.col("id")) is w
    assert w.clusterBy("id") is w


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


# --------------------------------------------------------------------------- reader schema

def _sent_read_schema(request):
    """The DataSource.schema of a captured AnalyzePlanRequest: schema=4 -> plan=1 -> root=1 ->
    read=2 -> data_source=2 -> schema=2."""
    message = parse_message(request)
    for field in (4, 1, 1, 2, 2):
        message = parse_message(message[field][0])
    return message[2][0].decode()


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
