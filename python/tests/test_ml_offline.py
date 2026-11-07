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
"""
pyspark.ml on this client, without a Spark Connect server: the messages that the vendored
pyspark.ml builds, of the literals, the types, the plans and the client of this client, as those
of official PySpark 4.3.0, with a mock server where a command is sent. The expected values are
those of the reference client.
"""

import datetime
import decimal
import enum
import gc
import subprocess
import sys
import textwrap
import weakref

import numpy as np
import pytest

from _mock_connect_server import mock_connect_session

from pyspark.loose_version import LooseVersion

# The generated pyspark.sql.connect.proto, of Apache Spark v4.3.0, needs protobuf >= 6.33.5.
_protobuf = pytest.importorskip("google.protobuf")
if LooseVersion(_protobuf.__version__) < LooseVersion("6.33.5"):
    pytest.skip("pyspark.sql.connect.proto needs protobuf >= 6.33.5", allow_module_level=True)

import pyspark.sql.connect.proto as pb2  # noqa: E402

from google.protobuf import text_format  # noqa: E402

from pyspark.errors import PySparkTypeError, PySparkValueError  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql import functions as F  # noqa: E402
from pyspark.sql import types as T  # noqa: E402


@pytest.fixture(scope="module")
def _ml_mock_server():
    pytest.importorskip("grpc")
    with mock_connect_session() as session:
        yield session


@pytest.fixture
def ml_mock(_ml_mock_server):
    # A session of its own of the shared server, which is the active session while the test
    # runs, as pyspark.ml reads it.
    _, service = _ml_mock_server
    service.reset()
    spark = SparkSession.builder.remote(service.remote).getOrCreate()
    yield spark, service
    spark.stop()


@pytest.fixture(autouse=True, scope="module")
def _debugging_read_without_a_session_of_ml():
    # PySpark's is_debugging_enabled reads spark.python.sql.dataFrameDebugging.enabled of the
    # active session at the first expression of a Column method, and keeps it: it is read
    # without a session, as test_dropin_offline reads it, before a mock server is there.
    from pyspark.errors.utils import is_debugging_enabled

    if SparkSession.getActiveSession() is None:
        is_debugging_enabled()


@pytest.fixture(autouse=True)
def _no_session_left_active_of_ml():
    yield
    session = SparkSession.getActiveSession()
    if session is not None:
        session.stop()


def _text(message):
    return text_format.MessageToString(message, as_one_line=True)


def _ml_result(**fields):
    # The ExecutePlanResponse of an MlCommandResult, which the mock server sends of a command.
    return pb2.ExecutePlanResponse(ml_command_result=pb2.MlCommandResult(**fields)).SerializeToString()


def _operator_info(ref_id):
    return _ml_result(
        operator_info=pb2.MlCommandResult.MlOperatorInfo(obj_ref=pb2.ObjectRef(id=ref_id))
    )


def _commands(service):
    # The commands of the ExecutePlanRequests that the mock server received.
    return [pb2.ExecutePlanRequest.FromString(r).plan.command for r in service.execute_requests]


class _Color(enum.Enum):
    RED = 1


_UTF8 = 'string { collation: "UTF8_BINARY" }'


@pytest.mark.parametrize(
    "value, expected",
    [
        (None, "literal { null { null { } } }"),
        (1, "literal { integer: 1 }"),
        (2**31, "literal { long: 2147483648 }"),
        (1.5, "literal { double: 1.5 }"),
        ("a", 'literal { string: "a" }'),
        (True, "literal { boolean: true }"),
        (b"ab", 'literal { binary: "ab" }'),
        (decimal.Decimal("1.5"), 'literal { decimal { value: "1.5" precision: 10 scale: 0 } }'),
        (datetime.date(2020, 1, 2), "literal { date: 18263 }"),
        (datetime.time(1, 2, 3, 4), "literal { time { nano: 3723000004000 precision: 6 } }"),
        (datetime.timedelta(days=1, microseconds=3), "literal { day_time_interval: 86400000003 }"),
        (np.int64(3), "literal { long: 3 }"),
        (np.int8(3), "literal { byte: 3 }"),
        (np.float32(1.5), "literal { float: 1.5 }"),
        (np.bool_(True), "literal { boolean: true }"),
        (_Color.RED, "literal { integer: 1 }"),
        (
            [1.0, 2.0],
            "literal { array { element_type { double { } } elements { double: 1.0 } "
            "elements { double: 2.0 } } }",
        ),
        (
            ["a"],
            f'literal {{ array {{ element_type {{ {_UTF8} }} elements {{ string: "a" }} }} }}',
        ),
        (
            [1, None],
            "literal { array { element_type { integer { } } elements { integer: 1 } "
            "elements { null { integer { } } } } }",
        ),
        (
            [[1.0, 2.0], [3.0]],
            "literal { array { element_type { array { element_type { double { } } "
            "contains_null: true } } elements { array { element_type { double { } } "
            "elements { double: 1.0 } elements { double: 2.0 } } } elements { array { "
            "element_type { double { } } elements { double: 3.0 } } } } }",
        ),
    ],
)
def test_ml_literals_of_inferred_types_as_pyspark(value, expected):
    # As official PySpark 4.3.0's LiteralExpression._from_value, of which pyspark.ml sends the
    # values of its params: the type of a value, of a list that of its first element.
    from pyspark.sql.connect.expressions import LiteralExpression

    assert _text(LiteralExpression._from_value(value).to_plan(None)) == expected


@pytest.mark.parametrize(
    "value, data_type, expected",
    [
        (None, T.IntegerType(), "literal { null { integer { } } }"),
        (
            None,
            T.ArrayType(T.StringType()),
            f"literal {{ null {{ array {{ element_type {{ {_UTF8} }} contains_null: true }} }} }}",
        ),
        (1, T.LongType(), "literal { long: 1 }"),
        (1, T.ShortType(), "literal { short: 1 }"),
        (1.0, T.FloatType(), "literal { float: 1.0 }"),
        ([], T.ArrayType(T.StringType()), f"literal {{ array {{ element_type {{ {_UTF8} }} }} }}"),
        (
            [1.0, None],
            T.ArrayType(T.DoubleType()),
            "literal { array { element_type { double { } } elements { double: 1.0 } "
            "elements { null { double { } } } } }",
        ),
        (
            decimal.Decimal("1.25"),
            T.DecimalType(5, 2),
            'literal { decimal { value: "1.25" precision: 5 scale: 2 } }',
        ),
        (
            datetime.datetime(2020, 1, 2),
            T.TimestampNTZType(),
            "literal { timestamp_ntz: 1577923200000000 }",
        ),
        (
            datetime.time(1, 2),
            T.TimeType(),
            "literal { time { nano: 3720000000000 precision: 6 } }",
        ),
        (np.int32(1), T.IntegerType(), "literal { integer: 1 }"),
    ],
)
def test_ml_literals_of_given_types_as_pyspark(value, data_type, expected):
    # As official PySpark 4.3.0's LiteralExpression(value, dataType), of which pyspark.ml sends
    # the arguments of the types it gives, such as an empty list of strings.
    from pyspark.sql.connect.expressions import LiteralExpression

    assert _text(LiteralExpression(value=value, dataType=data_type).to_plan(None)) == expected


def test_ml_literals_raise_the_errors_of_pyspark():
    # As official PySpark 4.3.0: a value of no literal type, an int out of the range of a long,
    # a list of no element type, and a value of another type than the given one.
    from pyspark.sql.connect.expressions import LiteralExpression

    for value, name in [((1, 2), "tuple"), ({"a": 1}, "dict"), (object(), "object")]:
        with pytest.raises(PySparkTypeError) as error:
            LiteralExpression._from_value(value)
        assert error.value.getCondition() == "UNSUPPORTED_DATA_TYPE"
        assert error.value.getMessageParameters() == {"data_type": name}
    with pytest.raises(PySparkValueError) as error:
        LiteralExpression._from_value(2**63)
    assert error.value.getCondition() == "VALUE_NOT_BETWEEN"
    for value in [[], [None]]:
        with pytest.raises(PySparkTypeError) as error:
            LiteralExpression._from_value(value)
        assert error.value.getCondition() == "CANNOT_INFER_ARRAY_ELEMENT_TYPE"
    for value, data_type in [
        (1, T.StringType()),
        ("a", T.IntegerType()),
        (300, T.ByteType()),
        (1, T.BooleanType()),
        ({"a": 1}, T.MapType(T.StringType(), T.IntegerType())),
    ]:
        with pytest.raises(AssertionError):
            LiteralExpression(value, data_type)
    with pytest.raises(AssertionError):
        LiteralExpression._from_value([1, "a"]).to_plan(None)


def test_ml_literal_values_as_pyspark():
    # As official PySpark 4.3.0's LiteralExpression._to_value, of which pyspark.ml reads the
    # values of the results of the server.
    from pyspark.sql.connect.expressions import LiteralExpression

    for value in [
        None,
        1,
        2**40,
        1.5,
        "a",
        True,
        b"ab",
        decimal.Decimal("1.5"),
        datetime.date(2020, 1, 2),
        datetime.time(1, 2, 3, 4),
        datetime.timedelta(days=1, microseconds=3),
        [1.0, 2.0],
        [["a"], ["b", "c"]],
    ]:
        literal = LiteralExpression._from_value(value).to_plan(None).literal
        assert LiteralExpression._to_value(literal) == value
    assert type(LiteralExpression._to_value(pb2.Expression.Literal(binary=b"x"))) is bytes

    struct_type = T.StructType([T.StructField("a", T.IntegerType()), T.StructField("b", T.StringType())])
    struct = pb2.Expression.Literal()
    struct.struct.struct_type.CopyFrom(_data_type_proto(struct_type))
    struct.struct.elements.append(pb2.Expression.Literal(integer=1))
    struct.struct.elements.append(pb2.Expression.Literal(string="x"))
    row = LiteralExpression._to_value(struct)
    assert row == T.Row(a=1, b="x") and row.asDict() == {"a": 1, "b": "x"}

    mapping = pb2.Expression.Literal()
    mapping.map.key_type.CopyFrom(_data_type_proto(T.StringType()))
    mapping.map.value_type.CopyFrom(_data_type_proto(T.LongType()))
    mapping.map.keys.append(pb2.Expression.Literal(string="k"))
    mapping.map.values.append(pb2.Expression.Literal(long=2))
    assert LiteralExpression._to_value(mapping) == {"k": 2}

    ntz = pb2.Expression.Literal(timestamp_ntz=1577923200000000)
    assert LiteralExpression._to_value(ntz) == datetime.datetime(2020, 1, 2)
    with pytest.raises(PySparkTypeError) as error:
        LiteralExpression._to_value(pb2.Expression.Literal())
    assert error.value.getCondition() == "UNSUPPORTED_LITERAL"


def test_ml_literal_reprs_as_pyspark():
    from pyspark.sql.connect.expressions import Expression, LiteralExpression

    assert repr(LiteralExpression(None, T.IntegerType())) == "NULL"
    assert repr(LiteralExpression(1, T.LongType())) == "1"
    assert repr(LiteralExpression(datetime.date(2020, 1, 2), T.DateType())) == "2020-01-02"
    assert repr(LiteralExpression(datetime.time(1, 2), T.TimeType())) == "01:02:00.000000"
    assert (
        repr(LiteralExpression(datetime.datetime(2020, 1, 2, 3), T.TimestampNTZType()))
        == "2020-01-02 03:00:00.000000"
    )
    assert repr(LiteralExpression(datetime.timedelta(seconds=90), T.DayTimeIntervalType())) == (
        "P0DT0H1M30S"
    )
    literal = LiteralExpression._from_value(1)
    assert isinstance(literal, Expression)
    assert literal.children == [] and literal.origin is None
    seen = []
    literal.foreach(seen.append)
    assert seen == [literal]


def _data_type_proto(data_type):
    from pyspark.sql.connect.types import pyspark_types_to_proto_types

    return pyspark_types_to_proto_types(data_type)


@pytest.mark.parametrize(
    "data_type, expected",
    [
        (T.NullType(), "null { }"),
        (T.StringType(), _UTF8),
        (T.StringType("UTF8_LCASE"), 'string { collation: "UTF8_LCASE" }'),
        (T.CharType(3), "char { length: 3 }"),
        (T.DecimalType(5, 2), "decimal { scale: 2 precision: 5 }"),
        (T.TimeType(3), "time { precision: 3 }"),
        (T.DayTimeIntervalType(0, 1), "day_time_interval { start_field: 0 end_field: 1 }"),
        (
            T.ArrayType(T.StringType(), False),
            f"array {{ element_type {{ {_UTF8} }} }}",
        ),
        (
            T.MapType(T.IntegerType(), T.DoubleType()),
            "map { key_type { integer { } } value_type { double { } } value_contains_null: true }",
        ),
        (
            T.StructType([T.StructField("a", T.IntegerType()), T.StructField("b", T.LongType(), False)]),
            'struct { fields { name: "a" data_type { integer { } } nullable: true } '
            'fields { name: "b" data_type { long { } } } }',
        ),
    ],
)
def test_ml_data_types_are_those_of_pyspark(data_type, expected):
    # As official PySpark 4.3.0's pyspark_types_to_proto_types and
    # proto_schema_to_pyspark_data_type, of which pyspark.ml sends and reads types.
    from pyspark.sql.connect.types import proto_schema_to_pyspark_data_type

    message = _data_type_proto(data_type)
    assert isinstance(message, pb2.DataType)
    assert _text(message) == expected
    assert proto_schema_to_pyspark_data_type(message) == data_type


@pytest.mark.parametrize(
    "build, expected",
    [
        (
            lambda P: P.Read("db.t", {"k": "v"}, True),
            'read { named_table { unparsed_identifier: "db.t" options { key: "k" value: "v" } } '
            "is_streaming: true }",
        ),
        (lambda P: P.Range(0, 3, 1, 2), "range { start: 0 end: 3 step: 1 num_partitions: 2 }"),
        (lambda P: P.Range(5, 9, 2), "range { start: 5 end: 9 step: 2 }"),
        (
            lambda P: P.SQL("SELECT ?", args=[F.lit(1)]),
            'sql { query: "SELECT ?" pos_arguments { literal { integer: 1 } } }',
        ),
        (
            lambda P: P.SQL("SELECT :a", named_args={"a": F.lit("x")}),
            'sql { query: "SELECT :a" named_arguments { key: "a" value { literal { string: "x" } } } }',
        ),
        (
            lambda P: P.SQL("SELECT * FROM v", views=[P.SubqueryAlias(P.Range(0, 1, 1), "v")]),
            'with_relations { root { sql { query: "SELECT * FROM v" } } references { subquery_alias '
            '{ input { range { start: 0 end: 1 step: 1 } } alias: "v" } } }',
        ),
        (
            lambda P: P.SubqueryAlias(P.Read("t"), "x"),
            'subquery_alias { input { read { named_table { unparsed_identifier: "t" } } } alias: "x" }',
        ),
    ],
)
def test_connect_plans_build_the_relations_of_pyspark(build, expected):
    # As official PySpark 4.3.0's plans of pyspark.sql.connect.plan, of which
    # pyspark.testing.connectutils builds DataFrames: their relations, of their own plan IDs.
    import re

    import pyspark.sql.connect.plan as P

    plan = build(P)
    relation = plan.plan(None)
    assert relation.common.plan_id == plan._plan_id
    assert re.sub(r"common \{ plan_id: \d+ \} ?", "", _text(relation)) == expected


def test_connect_sql_plans_as_pyspark():
    # As official PySpark 4.3.0: the command of a SQL plan, its print, and its arguments, which
    # are Columns.
    import pyspark.sql.connect.plan as P

    command = P.SQL("SELECT 1").command(None)
    assert command.sql_command.input.sql.query == "SELECT 1"
    assert P.Read("t").print(2) == "  <Read table_name=t>\n"
    with pytest.raises(AssertionError):
        P.SQL("SELECT ?", args=[1])
    with pytest.raises(AssertionError):
        P.SQL("SELECT 1", views=[P.Range(0, 1, 1)])


def test_column_to_plan_as_pyspark():
    # As PySpark Connect's Column.to_plan: the spark.connect.Expression of the column.
    from pyspark.sql import functions as F

    assert _text(F.col("a").to_plan(None)) == (
        'unresolved_attribute { unparsed_identifier: "a" is_metadata_column: false }'
    )
    column = F.lit(1) + F.col("b")
    assert column.to_plan(None) == column._expr.to_plan(None)
    assert column.to_plan(None).unresolved_function.function_name == "+"


def test_connect_type_helpers_as_pyspark():
    # As official PySpark 4.3.0's helpers of pyspark.sql.connect.types.
    from pyspark.sql.connect.types import (
        parse_attr_name,
        verify_col_name,
        verify_numeric_col_name,
    )

    assert parse_attr_name("a.`b.c`.d") == ["a", "b.c", "d"]
    assert parse_attr_name("`a``b`") == ["a`b"]
    assert parse_attr_name("a..b") is None and parse_attr_name("`a`b") is None
    assert parse_attr_name("a.") is None and parse_attr_name("`a") is None
    schema = T.StructType(
        [
            T.StructField("n", T.IntegerType()),
            T.StructField("s", T.StringType()),
            T.StructField(
                "st", T.StructType([T.StructField("x", T.DoubleType()), T.StructField("y", T.StringType())])
            ),
            T.StructField("arr", T.ArrayType(T.StructType([T.StructField("z", T.LongType())]))),
        ]
    )
    assert verify_col_name("st.x", schema) and verify_col_name("arr.z", schema)
    assert not verify_col_name("st.w", schema) and not verify_col_name("a..b", schema)
    assert verify_numeric_col_name("n", schema) and verify_numeric_col_name("st.x", schema)
    assert verify_numeric_col_name("arr.z", schema)
    assert not verify_numeric_col_name("s", schema) and not verify_numeric_col_name("st.y", schema)


def test_connect_conversions_as_pyspark(ml_mock):
    # As official PySpark 4.3.0's pyspark.sql.connect.conversion: the storage levels of their
    # messages, and the DataFrame of a relation cached on the server, which removes the relation
    # from the server once it is collected.
    from pyspark.sql.connect.conversion import (
        proto_to_remote_cached_dataframe,
        proto_to_storage_level,
        storage_level_to_proto,
    )
    from pyspark.storagelevel import StorageLevel

    spark, service = ml_mock
    level = StorageLevel(True, True, False, False, 2)
    message = storage_level_to_proto(level)
    assert message == pb2.StorageLevel(
        use_disk=True, use_memory=True, use_off_heap=False, deserialized=False, replication=2
    )
    assert proto_to_storage_level(message) == level

    df = proto_to_remote_cached_dataframe(pb2.CachedRemoteRelation(relation_id="r1"))
    assert df._plan.plan(spark.client).cached_remote_relation.relation_id == "r1"
    assert service.execute_requests == []
    del df
    gc.collect()
    (command,) = _commands(service)
    assert command.remove_cached_remote_relation_command.relation.relation_id == "r1"


def test_ml_vector_udt_is_sent_as_pyspark():
    from pyspark.ml.linalg import VectorUDT
    from pyspark.sql.connect.types import proto_schema_to_pyspark_data_type

    message = _data_type_proto(VectorUDT())
    assert message.udt.jvm_class == "org.apache.spark.ml.linalg.VectorUDT"
    assert message.udt.python_class == "pyspark.ml.linalg.VectorUDT"
    assert isinstance(proto_schema_to_pyspark_data_type(message), VectorUDT)


def test_user_context_extensions_are_sent_as_pyspark(ml_mock):
    # As PySpark Connect's SparkConnectClient: the user context extensions of the current thread
    # and the global ones, of the thread_local of the client, which the requests of the
    # executions and the analyses of a thread carry, the global ones first.
    import threading

    from google.protobuf import any_pb2

    spark, service = ml_mock
    client = spark.client

    def packed(graph_id):
        extension = any_pb2.Any()
        extension.Pack(pb2.PipelineAnalysisContext(dataflow_graph_id=graph_id))
        return extension

    local = client.add_threadlocal_user_context_extension(packed("local"))
    shared = client.add_global_user_context_extension(packed("global"))
    assert local.startswith("threadlocal_") and shared.startswith("global_")
    assert client.thread_local.user_context_extensions == [(local, packed("local"))]
    assert client.global_user_context_extensions == [(shared, packed("global"))]

    command = pb2.Command()
    command.ml_command.clean_cache.SetInParent()
    client.execute_command(command)
    request = pb2.ExecutePlanRequest.FromString(service.execute_requests[-1])
    assert list(request.user_context.extensions) == [packed("global"), packed("local")]
    assert spark.range(1).schema is not None
    analysis = pb2.AnalyzePlanRequest.FromString(service.analyze_requests[-1])
    assert list(analysis.user_context.extensions) == [packed("global"), packed("local")]

    # Another thread has the global extensions only.
    seen = []

    def other():
        seen.append(client.thread_local.user_context_extensions)
        client.execute_command(command)

    thread = threading.Thread(target=other)
    thread.start()
    thread.join()
    assert seen == [[]]
    request = pb2.ExecutePlanRequest.FromString(service.execute_requests[-1])
    assert list(request.user_context.extensions) == [packed("global")]

    client.remove_user_context_extension(local)
    assert client.thread_local.user_context_extensions == []
    client.clear_user_context_extensions()
    assert client.global_user_context_extensions == []
    client.execute_command(command)
    request = pb2.ExecutePlanRequest.FromString(service.execute_requests[-1])
    assert list(request.user_context.extensions) == []


def test_pipeline_analysis_context_is_an_extension_of_its_block_as_pyspark(ml_mock):
    # As official PySpark 4.3.0's add_pipeline_analysis_context: the PipelineAnalysisContext of
    # the requests of its block, of the current thread.
    from pyspark.pipelines.add_pipeline_analysis_context import add_pipeline_analysis_context

    spark, service = ml_mock
    command = pb2.Command()
    command.ml_command.clean_cache.SetInParent()
    with add_pipeline_analysis_context(spark, "graph-1", "flow-1"):
        spark.client.execute_command(command)
    spark.client.execute_command(command)
    inside, after = [pb2.ExecutePlanRequest.FromString(r) for r in service.execute_requests]
    (extension,) = inside.user_context.extensions
    context = pb2.PipelineAnalysisContext()
    assert extension.Unpack(context)
    assert (context.dataflow_graph_id, context.flow_name) == ("graph-1", "flow-1")
    assert list(after.user_context.extensions) == []


def test_ml_session_client_is_one_object_of_the_session(ml_mock):
    # As PySpark Connect's SparkSession.client: one object of each session, of its session ID.
    from pyspark.sql.connect.client import SparkConnectClient

    spark, service = ml_mock
    client = spark.client
    assert isinstance(client, SparkConnectClient)
    assert spark.client is client
    assert client._server_session_id == spark.session_id
    other = SparkSession.builder.remote(service.remote).create()
    try:
        assert other.client is not client
        assert other.client._server_session_id == other.session_id
    finally:
        other.stop()


def test_ml_execute_command_returns_the_results_of_pyspark(ml_mock):
    # As PySpark Connect's SparkSession.client.execute_command: no data, the results of the
    # responses by their fields, and no execution information.
    spark, service = ml_mock
    command = pb2.Command()
    command.ml_command.clean_cache.SetInParent()
    service.execute_payload = _operator_info("ref-1")
    data, properties, info = spark.client.execute_command(command)
    assert data is None and info is None
    assert list(properties) == ["ml_command_result"]
    assert properties["ml_command_result"].operator_info.obj_ref.id == "ref-1"
    request = pb2.ExecutePlanRequest.FromString(service.execute_requests[-1])
    assert request.session_id == spark.session_id
    assert request.plan.command == command

    # The result of a SQL command is its relation, as SparkConnectClient's.
    relation = pb2.Relation(sql=pb2.SQL(query="SELECT 1"))
    service.execute_payload = pb2.ExecutePlanResponse(
        sql_command_result=pb2.ExecutePlanResponse.SqlCommandResult(relation=relation)
    ).SerializeToString()
    _, properties, _ = spark.client.execute_command(command)
    assert properties == {"sql_command_result": relation}


def test_ml_cache_commands_are_those_of_pyspark(ml_mock):
    # As PySpark Connect's _delete_ml_cache and _cleanup_ml_cache: the IDs that the server
    # deleted, and none of no IDs or of an error.
    spark, service = ml_mock
    client = spark.client
    service.execute_payload = _operator_info("a,b")
    assert client._delete_ml_cache(["a", "b"], evict_only=True) == ["a", "b"]
    (command,) = _commands(service)
    assert [ref.id for ref in command.ml_command.delete.obj_refs] == ["a", "b"]
    assert command.ml_command.delete.evict_only is True
    assert client._delete_ml_cache([]) == []
    assert len(service.execute_requests) == 1
    client._cleanup_ml_cache()
    assert _commands(service)[-1].ml_command.HasField("clean_cache")
    service.execute_failure = "the server failed"
    assert client._delete_ml_cache(["c"]) == []
    client._cleanup_ml_cache()


def test_ml_fit_and_transform_send_the_messages_of_pyspark(ml_mock):
    # As official PySpark 4.3.0: fit sends an MlCommand.Fit of the estimator, its params and
    # the relation of the dataset, of which the server returns the model, and the transform of
    # the model is an ml_relation of the dataset that later transformations take as input.
    from pyspark.ml.feature import CountVectorizer

    spark, service = ml_mock
    service.execute_payload = _operator_info("model-1")
    df = spark.range(1).selectExpr("array('a', 'b') AS words")
    estimator = CountVectorizer(vocabSize=5, minDF=0.5, inputCol="words", outputCol="counts")
    model = estimator.fit(df)

    (command,) = _commands(service)
    fit = command.ml_command.fit
    assert fit.estimator.name == "org.apache.spark.ml.feature.CountVectorizer"
    assert fit.estimator.uid == estimator.uid
    assert fit.estimator.type == pb2.MlOperator.OPERATOR_TYPE_ESTIMATOR
    assert dict(fit.params.params) == {
        "vocabSize": pb2.Expression.Literal(integer=5),
        "minDF": pb2.Expression.Literal(double=0.5),
        "inputCol": pb2.Expression.Literal(string="words"),
        "outputCol": pb2.Expression.Literal(string="counts"),
    }
    assert fit.dataset == df._plan.plan(spark.client)
    assert fit.dataset.HasField("project")

    assert str(model._java_obj) == "model-1"
    assert model.uid == estimator.uid and model.getVocabSize() == 5

    predictions = model.transform(df)
    relation = predictions._plan.plan(spark.client)
    transform = relation.ml_relation.transform
    assert transform.obj_ref.id == "model-1"
    assert transform.input == fit.dataset
    assert transform.params.params["vocabSize"] == pb2.Expression.Literal(integer=5)

    selected = predictions.select("counts")._plan.plan(spark.client)
    assert selected.project.input == relation
    # A column of the transformed DataFrame refers to the plan ID of its relation.
    reference = predictions["counts"]._expr.to_plan(None).unresolved_attribute
    assert reference.plan_id == relation.common.plan_id
    assert len(service.execute_requests) == 1


def test_ml_transformers_send_the_relation_of_pyspark(ml_mock):
    # As official PySpark 4.3.0: a transformer of no model is an ml_relation of its name, its
    # uid and its params.
    from pyspark.ml.feature import Tokenizer

    spark, service = ml_mock
    df = spark.range(1).selectExpr("'a b' AS text")
    tokenizer = Tokenizer(inputCol="text", outputCol="words")
    transform = tokenizer.transform(df)._plan.plan(spark.client).ml_relation.transform
    assert transform.transformer.name == "org.apache.spark.ml.feature.Tokenizer"
    assert transform.transformer.uid == tokenizer.uid
    assert transform.transformer.type == pb2.MlOperator.OPERATOR_TYPE_TRANSFORMER
    assert transform.params.params["inputCol"] == pb2.Expression.Literal(string="text")
    assert transform.input == df._plan.plan(spark.client)
    assert service.execute_requests == []


def test_ml_dataframes_of_plans_keep_their_plans(ml_mock):
    # As PySpark Connect's DataFrame(plan, session): the relation of the plan, of which the
    # DataFrame and those of it keep the plan, as pyspark.ml keeps its models with the plans
    # of their transforms.
    from pyspark.sql.connect.dataframe import DataFrame
    from pyspark.sql.connect.plan import LogicalPlan

    spark, _ = ml_mock

    class SQLPlan(LogicalPlan):
        def __init__(self):
            super().__init__(None)

        def plan(self, session):
            plan = self._create_proto_relation()
            plan.sql.query = "SELECT 1 AS a"
            return plan

    plan = SQLPlan()
    alive = weakref.ref(plan)
    df = DataFrame(plan=plan, session=spark)
    relation = df._plan.plan(spark.client)
    assert relation.sql.query == "SELECT 1 AS a"
    assert df["a"]._expr.to_plan(None).unresolved_attribute.plan_id == relation.common.plan_id
    derived = df.filter("a > 0")
    del plan, df
    gc.collect()
    assert alive() is not None
    assert derived._plan.plan(spark.client).filter.input == relation
    del derived
    gc.collect()
    assert alive() is None


def test_ml_logical_plan_ids_as_pyspark():
    # As PySpark Connect's LogicalPlan: a plan ID of each plan, and another of a plan with
    # references, which _plan_id is.
    from pyspark.sql.connect.plan import LogicalPlan

    first, second = LogicalPlan(None), LogicalPlan(None)
    assert second._plan_id == first._plan_id + 1
    assert first._create_proto_relation().common.plan_id == first._plan_id
    assert first.observations == {}
    with_references = LogicalPlan(first, references=[second])
    assert with_references._plan_id != with_references._root_plan_id
    assert with_references._create_proto_relation().common.plan_id == with_references._root_plan_id
    assert first.plan(None) is None and first.command(None) is None

    class Child(LogicalPlan):
        def plan(self, session):
            plan = self._create_proto_relation()
            plan.range.end = 1
            return plan

    child = Child(None)
    assert child._verify(None)
    root = child.plan(None)
    assert child._with_relations(root, None) is root
    parent = LogicalPlan(child, references=[child])
    relations = parent._with_relations(root, None)
    assert relations.common.plan_id == parent._plan_id
    assert relations.with_relations.root == root
    assert list(relations.with_relations.references) == [root]


def test_ml_column_expressions_render_as_pyspark():
    # As PySpark Connect's Column._expr: of which pyspark.ml.stat names the metrics of
    # Summarizer, the expression that Column<'...'> shows.
    from pyspark.ml.stat import Summarizer
    from pyspark.sql import functions as F

    for column in [F.col("a"), F.col("a") + 1, F.lit("x").alias("y"), F.max("b")]:
        assert str(column._expr) == repr(column._expr) == repr(column)[len("Column<'") : -len("'>")]
    assert str((F.col("a") + 1)._expr) == "+(a, 1)"
    assert F.col("a")._expr.to_plan(None).unresolved_attribute.unparsed_identifier == "a"
    assert repr(Summarizer.mean(F.col("features"))) == (
        "Column<'aggregate_metrics(array(mean), features, 1.0)['mean'] AS mean(features)'>"
    )


def test_ml_modules_import_as_pyspark():
    # The modules of pyspark.ml, vendored from Apache Spark v4.3.0, import on this client.
    import importlib

    for name in [
        "pyspark.ml",
        "pyspark.ml.base",
        "pyspark.ml.classification",
        "pyspark.ml.clustering",
        "pyspark.ml.evaluation",
        "pyspark.ml.feature",
        "pyspark.ml.fpm",
        "pyspark.ml.image",
        "pyspark.ml.model_cache",
        "pyspark.ml.pipeline",
        "pyspark.ml.recommendation",
        "pyspark.ml.regression",
        "pyspark.ml.stat",
        "pyspark.ml.tree",
        "pyspark.ml.tuning",
        "pyspark.ml.wrapper",
        "pyspark.ml.connect.proto",
        "pyspark.ml.connect.serialize",
        "pyspark.ml.connect.readwrite",
        "pyspark.ml.connect.summarizer",
        "pyspark.sql.connect.expressions",
        "pyspark.sql.connect.proto.ml_pb2",
    ]:
        assert importlib.import_module(name) is not None
    import pyspark.ml as ml

    assert {"Pipeline", "Estimator", "Transformer", "classification", "feature"} <= set(ml.__all__)


def test_ml_modules_import_without_protobuf():
    # Without the protobuf package, pyspark.ml and the connect modules import, of a progress
    # response of the transport, and a message of the generated modules raises an
    # AttributeError that names protobuf.
    script = textwrap.dedent(
        """
        import importlib.abc, sys

        class Block(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path, target=None):
                if name == "google.protobuf" or name.startswith("google.protobuf."):
                    raise ImportError("no protobuf")

        sys.meta_path.insert(0, Block())
        import pyspark.ml
        from pyspark.ml.classification import LogisticRegression
        import pyspark.sql.connect.expressions, pyspark.sql.connect.plan
        import pyspark.sql.connect.proto as proto
        assert LogisticRegression(maxIter=3).getMaxIter() == 3
        assert hasattr(proto, "ExecutePlanResponse") and not hasattr(proto, "Relation")
        try:
            proto.Relation
        except AttributeError as error:
            print(error)
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stderr
    assert "'Relation'" in result.stdout and "protobuf >= 6.33.5" in result.stdout


def test_session_hooks_rewrite_the_requests_of_executions_as_pyspark(_ml_mock_server):
    # As official PySpark 4.3.0's SparkSession.Hook: a session that the builder creates calls
    # each factory of _registerHook with itself, and the on_execute_plan of each hook with each
    # ExecutePlanRequest before it sends it, which keeps the operation ID of the request.
    _, service = _ml_mock_server
    service.reset()
    created = []

    class TagHook(SparkSession.Hook):
        def __init__(self, session, tag):
            self.session = session
            self.tag = tag
            self.operation_ids = []

        def on_execute_plan(self, request):
            assert isinstance(request, pb2.ExecutePlanRequest)
            self.operation_ids.append(request.operation_id)
            request.tags.append(self.tag)
            request.operation_id = "replaced"
            return request

    def factory(tag):
        def make(session):
            hook = TagHook(session, tag)
            created.append(hook)
            return hook

        return make

    builder = SparkSession.builder.remote(service.remote)
    spark = builder._registerHook(factory("first"))._registerHook(factory("second")).create()
    try:
        assert [hook.session for hook in created] == [spark, spark]
        command = pb2.Command()
        command.ml_command.clean_cache.SetInParent()
        spark.client.execute_command(command)
        request = pb2.ExecutePlanRequest.FromString(service.execute_requests[-1])
        assert list(request.tags)[-2:] == ["first", "second"]
        assert request.operation_id == created[0].operation_ids[0]
        assert created[1].operation_ids == created[0].operation_ids
        assert request.plan.command == command

        # A session got rather than created, the active or the default one, calls no
        # factories.
        got = SparkSession.builder.remote(service.remote)._registerHook(factory("third"))
        assert not got.getOrCreate().is_stopped and len(created) == 2
    finally:
        spark.stop()

    assert SparkSession.Hook().on_execute_plan(request) is request


def test_session_hooks_raise_their_errors_as_pyspark(_ml_mock_server):
    # As official PySpark 4.3.0: the error of a hook is raised by the execution that calls it,
    # as it is.
    _, service = _ml_mock_server
    service.reset()

    class Failing(SparkSession.Hook):
        def on_execute_plan(self, request):
            raise ValueError("the hook failed")

    spark = SparkSession.builder.remote(service.remote)._registerHook(lambda s: Failing()).create()
    try:
        command = pb2.Command()
        command.ml_command.clean_cache.SetInParent()
        with pytest.raises(ValueError, match="the hook failed"):
            spark.client.execute_command(command)
        assert service.execute_requests == []
    finally:
        spark.stop()
