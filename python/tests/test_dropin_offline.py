"""Offline tests for the pyspark drop-in (no Spark Connect server required).

These exercise the pure-Python shim (types parsing/pickling, the session builder,
functions/column expression building, udf/udtf construction, StorageLevel) so the
Python side has real, measurable coverage in CI without a running server. Tests
that require a live server live in test_dropin_e2e.py and are gated on SPARK_REMOTE.
"""

import pickle

import pytest

from pyspark.sql import types as T


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


def test_structfield_construction():
    f = T.StructField("n", T.IntegerType(), True)
    st = T.StructType([f])
    assert "n" in st.simpleString()


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


def test_channel_builder_unsupported():
    from pyspark.sql import SparkSession
    with pytest.raises(NotImplementedError):
        SparkSession.builder.channelBuilder(object())


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
    r = reg.register("echo", Echo)
    assert r.name == "echo"
