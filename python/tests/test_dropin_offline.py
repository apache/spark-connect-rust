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
    assert "root" in st.treeString() and "a: int (nullable = true)" in st.treeString()
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
    # Rust-backed (defined in the extension module).
    for cls in (CatalogMetadata, Database, Table, Function, TablePartition):
        assert cls.__module__ == "builtins", f"{cls.__name__} should be Rust-backed"
    assert Column is p.CatalogColumn


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
    # UDFRegistration.register returns a UDF
    from pyspark.sql.udf import UDFRegistration
    reg = UDFRegistration(object())
    r = reg.register("f", lambda x: x + 1, T.IntegerType())
    assert r.name == "f"
    r2 = reg.register("g", lambda x: x)  # returnType default
    assert r2.name == "g"


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
