# SPDX-License-Identifier: Apache-2.0
"""The drop-in's types and Columns as official PySpark 4.3.0's, without a server: the JSON type
strings of `_parse_datatype_json_value`, the pickle of a CalendarIntervalType, the origins of the
expressions that Column methods send, the names of `Column.alias` and `Column.name`, and the
user-set attributes of a Column's copies. The expected values are official PySpark's, of the same
calls."""

import copy
import pickle

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


def test_calendar_interval_type_pickles_of_its_json_name():
    # As the JSON of the type, "interval", which official PySpark's parser reads too.
    t = T.CalendarIntervalType()
    assert t.__reduce__()[1] == ('"interval"',)
    assert pickle.loads(pickle.dumps(t)) == t


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
