#
# Licensed to the Apache Software Foundation (ASF) under one or more
# contributor license agreements. See the NOTICE file distributed with
# this work for additional information regarding copyright ownership.
# The ASF licenses this file to You under the Apache License, Version 2.0
# (the "License"); you may not use this file except in compliance with
# the License. You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Check public Arrow handoff behavior against a deterministic mock RPC service.

Cases cover Spark-output values, schemas, chunks, and lifetimes through public
APIs. Server-side Python addresses cannot prove Rust-to-Python buffer sharing.
"""

import gc
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import quote

import grpc
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from pyspark.sql import SparkSession
from pyspark.sql import types as T
from pyspark.sql._pandas_input import prepare_int64_table


@pytest.fixture(scope="module")
def protocol(tmp_path_factory):
    """Compile the repository protocol instead of hard-coding protobuf field numbers."""
    protoc = shutil.which(os.environ.get("PROTOC", "protoc"))
    if protoc is None:
        message = "Arrow handoff tests require protoc; install it or set PROTOC"
        if os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)
    proto_root = Path(__file__).resolve().parents[2] / "crates/spark-connect-proto/proto"
    descriptor = tmp_path_factory.mktemp("connect-proto") / "connect.pb"
    command = [protoc, f"-I{proto_root}"]
    if include := os.environ.get("PROTOC_INCLUDE"):
        command.append(f"-I{include}")
    subprocess.run(
        command
        + [
            "--include_imports",
            f"--descriptor_set_out={descriptor}",
            str(proto_root / "spark/connect/base.proto"),
        ],
        check=True,
    )
    pool = descriptor_pool.DescriptorPool()
    for file in descriptor_pb2.FileDescriptorSet.FromString(descriptor.read_bytes()).file:
        pool.Add(file)

    def message(name, **fields):
        return message_factory.GetMessageClass(pool.FindMessageTypeByName(f"spark.connect.{name}"))(
            **fields
        )

    return message


@pytest.mark.parametrize("in_ci", [False, True])
def test_missing_protoc_fails_in_ci_and_skips_locally(monkeypatch, tmp_path_factory, in_ci):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    if in_ci:
        monkeypatch.setenv("CI", "true")
    else:
        monkeypatch.delenv("CI", raising=False)
    error = pytest.fail.Exception if in_ci else pytest.skip.Exception
    with pytest.raises(error, match="require protoc"):
        protocol.__wrapped__(tmp_path_factory)


# The plan compression lookup that a client makes on its first plan. The fake servers answer
# it with no value, so plans are not compressed, and do not count it, so the tests count only
# their own Config requests.
_PLAN_COMPRESSION_THRESHOLD = b"spark.connect.session.planCompression.threshold"


@contextmanager
def _dataframe(batches, *, protocol, payloads=None, calls=None, settings=None):
    """Serve Arrow results using the repository's protobuf message definitions, with the
    configuration `settings` over the defaults below."""
    if payloads is None:
        payloads = []
        for batch in batches:
            sink = pa.BufferOutputStream()
            with pa.ipc.new_stream(sink, batch.schema) as writer:
                writer.write_batch(batch)
            payloads.append((batch.num_rows, sink.getvalue().to_pybytes()))
    server_id = "00000000-0000-0000-0000-000000000001"

    def execute(request, context):
        if calls is not None:
            calls["execute"] = calls.get("execute", 0) + 1
        request = protocol("ExecutePlanRequest").FromString(request)
        header = dict(
            session_id=request.session_id,
            operation_id=request.operation_id,
            server_side_session_id=server_id,
        )
        for index, (rows, payload) in enumerate(payloads):
            yield protocol(
                "ExecutePlanResponse",
                **header,
                response_id=str(index),
                arrow_batch=dict(row_count=rows, data=payload),
            ).SerializeToString()
        yield protocol(
            "ExecutePlanResponse", **header, response_id="complete", result_complete={}
        ).SerializeToString()

    def release_execute(request, context):
        request = protocol("ReleaseExecuteRequest").FromString(request)
        return protocol(
            "ReleaseExecuteResponse",
            session_id=request.session_id,
            operation_id=request.operation_id,
            server_side_session_id=server_id,
        ).SerializeToString()

    def config(request, context):
        if _PLAN_COMPRESSION_THRESHOLD in request:
            return protocol("ConfigResponse",
                            session_id=protocol("ConfigRequest").FromString(request).session_id,
                            server_side_session_id=server_id).SerializeToString()
        # The settings that toPandas reads first, and the defaults that row collection reads.
        if calls is not None:
            calls["config"] = calls.get("config", 0) + 1
        request = protocol("ConfigRequest").FromString(request)
        values = {
            "spark.sql.session.timeZone": "UTC",
            "spark.sql.execution.pandas.structHandlingMode": "legacy",
            "spark.sql.execution.arrow.pyspark.selfDestruct.enabled": "false",
            **(settings or {}),
        }
        operation = request.operation.WhichOneof("op_type")
        if operation == "get_with_default":
            pairs = []
            for pair in request.operation.get_with_default.pairs:
                default = pair.value if pair.HasField("value") else None
                value = values.get(pair.key, default)
                pairs.append(dict(key=pair.key, **({"value": value} if value is not None else {})))
        else:
            if operation != "get":
                context.abort(grpc.StatusCode.UNIMPLEMENTED,
                              f"unexpected Config operation {operation}")
            pairs = [dict(key=key, value=values[key]) for key in request.operation.get.keys]
        return protocol(
            "ConfigResponse",
            session_id=request.session_id,
            server_side_session_id=server_id,
            pairs=pairs,
        ).SerializeToString()

    def release_session(request, context):
        request = protocol("ReleaseSessionRequest").FromString(request)
        return protocol(
            "ReleaseSessionResponse",
            session_id=request.session_id,
            server_side_session_id=server_id,
        ).SerializeToString()

    with ThreadPoolExecutor(max_workers=2) as executor:
        server = grpc.server(executor)
        server.add_generic_rpc_handlers(
            [
                grpc.method_handlers_generic_handler(
                    "spark.connect.SparkConnectService",
                    {
                        "ExecutePlan": grpc.unary_stream_rpc_method_handler(execute),
                        "ReleaseExecute": grpc.unary_unary_rpc_method_handler(release_execute),
                        "ReleaseSession": grpc.unary_unary_rpc_method_handler(release_session),
                        "Config": grpc.unary_unary_rpc_method_handler(config),
                    },
                )
            ]
        )
        port = server.add_insecure_port("127.0.0.1:0")
        assert port != 0
        server.start()
        session = None
        try:
            session = SparkSession.builder.remote(f"sc://127.0.0.1:{port}").getOrCreate()
            yield session.range(1), session
        finally:
            try:
                if session is not None:
                    session.stop()
            finally:
                server.stop(0).wait(timeout=5)


def _mixed_batch():
    nested_type = pa.struct(
        [
            pa.field("items", pa.list_(pa.int64()), metadata={b"nested": b"metadata"}),
            pa.field("labels", pa.map_(pa.string(), pa.int64())),
        ]
    )
    arrays = [
        pa.array([1, None, -3, 4], type=pa.int64()),
        pa.array([True, None, False, True]),
        pa.array(["alpha", None, "", "omega"]),
        pa.array([b"a", None, b"", b"xyz"]),
        pa.array(
            [Decimal("1.23"), None, Decimal("-4.56"), Decimal("0.00")], type=pa.decimal128(12, 2)
        ),
        pa.array(
            [
                datetime(2020, 1, 1, tzinfo=timezone.utc),
                None,
                datetime(2020, 1, 2, tzinfo=timezone.utc),
                datetime(2020, 1, 3, tzinfo=timezone.utc),
            ],
            type=pa.timestamp("us", tz="UTC"),
        ),
        pa.array(
            [
                {"items": [1, None], "labels": [("a", 1)]},
                None,
                {"items": [], "labels": []},
                {"items": None, "labels": [("b", None)]},
            ],
            type=nested_type,
        ),
        pa.nulls(4),
    ]
    fields = [
        pa.field(f"c{index}", array.type, metadata={b"field": str(index).encode()})
        for index, array in enumerate(arrays)
    ]
    schema = pa.schema(fields, metadata={b"schema": b"preserved"})
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def test_collect_reads_binary_default_with_arrow_only_schema(protocol):
    batch = pa.RecordBatch.from_arrays([pa.array([b"value", None], type=pa.binary())], names=["b"])
    calls = {}
    with _dataframe(protocol=protocol, batches=[batch], calls=calls) as (frame, _):
        rows = frame.collect()
    assert type(rows[0].b) is bytes and rows[0].b == b"value"
    assert rows[1].b is None
    assert calls == {"execute": 1, "config": 1}


@pytest.mark.parametrize(
    "action",
    ["collect", "take", "tail", "first", "head", "head_many", "iterator", "prefetch"],
)
def test_native_result_actions_return_tuple_rows_recursively(protocol, action):
    from pyspark import Row

    child_type = pa.struct([pa.field("value", pa.int64())])
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array([1, 2], type=pa.int64()),
            pa.array([{"value": 3}, None], type=child_type),
            pa.array([[{"value": 4}, None], []], type=pa.list_(child_type)),
            pa.array([[("child", {"value": 5})], []], type=pa.map_(pa.string(), child_type)),
        ],
        names=["id", "child", "children", "by_name"],
    )
    with _dataframe(protocol=protocol, batches=[batch]) as (frame, _):
        if action in ("collect", "first", "head"):
            result = getattr(frame, action)()
            rows = result if action == "collect" else [result]
        elif action in ("take", "tail"):
            rows = getattr(frame, action)(2)
        elif action == "head_many":
            rows = frame.head(2)
        else:
            rows = list(frame.toLocalIterator(prefetchPartitions=action == "prefetch"))

    assert rows and all(type(row) is Row and isinstance(row, tuple) for row in rows)
    row = rows[0]
    assert type(row.child) is type(row.children[0]) is type(row.by_name["child"]) is Row
    assert tuple.__getitem__(row.child, 0) == 3
    assert row.children[1] is None
    assert row.asDict(recursive=True) == {
        "id": 1,
        "child": {"value": 3},
        "children": [{"value": 4}, None],
        "by_name": {"child": {"value": 5}},
    }
    if len(rows) == 2:
        assert rows[1] == (2, None, [], {})


def test_native_collect_rows_share_mutable_field_names(protocol):
    batch = pa.RecordBatch.from_arrays([pa.array([1, 2])], names=["value"])
    with _dataframe(protocol=protocol, batches=[batch.slice(0, 1), batch.slice(1, 1)]) as (frame, _):
        rows = frame.collect()

    assert rows[0].__fields__ is rows[1].__fields__
    rows[0].__fields__[0] = "renamed"
    assert rows[1].renamed == 2
    assert rows[1].asDict() == {"renamed": 2}


@pytest.mark.parametrize("prefetch", [False, True])
@pytest.mark.parametrize("one_response", [False, True])
def test_native_iterator_rows_share_field_names_only_within_each_arrow_batch(
    protocol, prefetch, one_response
):
    batch = pa.RecordBatch.from_arrays([pa.array([1, 2, 3, 4])], names=["value"])
    batches = [batch.slice(0, 2), batch.slice(2, 2)]
    payloads = None
    if one_response:
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, batch.schema) as writer:
            for part in batches:
                writer.write_batch(part)
        payloads = [(batch.num_rows, sink.getvalue().to_pybytes())]
    with _dataframe(protocol=protocol, batches=batches, payloads=payloads) as (frame, _):
        rows = list(frame.toLocalIterator(prefetchPartitions=prefetch))

    assert rows[0].__fields__ is rows[1].__fields__
    assert rows[2].__fields__ is rows[3].__fields__
    assert rows[0].__fields__ is not rows[2].__fields__
    rows[0].__fields__[0] = "renamed"
    assert rows[1].renamed == 2
    assert rows[2].value == 3
    assert rows[3].asDict() == {"value": 4}


def test_native_iterator_ends_after_row_materialization_fails(protocol):
    # The first timestamp is year 10000, outside Python datetime's range.
    batch = pa.RecordBatch.from_arrays(
        [pa.array([253402300800000000, 0], type=pa.timestamp("us"))], names=["value"]
    )
    with _dataframe(protocol=protocol, batches=[batch]) as (frame, _):
        rows = frame.toLocalIterator()
        with pytest.raises(OverflowError):
            next(rows)
        assert list(rows) == []


def test_handoff_preserves_values_metadata_and_chunks(protocol):
    batch = _mixed_batch()
    batches = [batch.slice(1, 2), batch.slice(0, 1), batch.slice(3, 1)]
    expected = pa.Table.from_batches(batches)
    calls = {}
    # The table of the execution as it is, of `_to_table`, which toArrow casts to the Arrow schema
    # of the schema of its analysis.
    with _dataframe(protocol=protocol, batches=batches, calls=calls) as (frame, _):
        actual, schema = frame._to_table()
    assert actual.equals(expected, check_metadata=True)
    assert schema is None
    for column in actual.columns:
        assert [len(chunk) for chunk in column.chunks] == [2, 1, 1]
    assert calls == {"execute": 1}


def test_normal_handoff_does_not_reopen_ipc(protocol, monkeypatch):
    def unexpected_ipc(*args, **kwargs):
        pytest.fail("Normal toArrow handoff must use the C Stream importer")

    monkeypatch.setattr(pa.ipc, "open_file", unexpected_ipc)
    batch = _mixed_batch()
    with _dataframe(protocol=protocol, batches=[batch]) as (frame, _):
        assert frame._to_table()[0].equals(pa.Table.from_batches([batch]), check_metadata=True)


def test_handoff_buffers_outlive_dataframe_session_and_table(protocol):
    batch = _mixed_batch()
    expected = batch.to_pydict()
    calls = {}

    def collect():
        with _dataframe(protocol=protocol, batches=[batch], calls=calls) as (frame, session):
            table, _ = frame._to_table()
            session.stop()
            return table

    table = collect()
    gc.collect()
    assert table.to_pydict() == expected
    numeric = table.column(0).chunk(0).slice(1, 3)
    strings = table.column(2).chunk(0).slice(1, 3)
    del table
    gc.collect()
    assert numeric.to_pylist() == [None, -3, 4]
    assert strings.to_pylist() == [None, "", "omega"]
    assert calls == {"execute": 1}


@pytest.mark.parametrize("kind", ["no-batches", "empty-typed", "zero-columns"])
def test_handoff_empty_and_zero_column_results(protocol, kind):
    if kind == "no-batches":
        # As PySpark's `_to_table`, which asserts that the result has a batch: a Spark server
        # sends one for an empty result.
        calls = {}
        with _dataframe(protocol=protocol, batches=[], calls=calls) as (frame, _):
            with pytest.raises(AssertionError):
                frame._to_table()
        assert calls == {"execute": 1}
        return
    if kind == "empty-typed":
        batches = [_mixed_batch().slice(0, 0)]
        expected = pa.Table.from_batches(batches)
    else:
        structs = pa.array([{}] * 5, type=pa.struct([]))
        batch = pa.RecordBatch.from_struct_array(structs)
        batch = batch.replace_schema_metadata({b"zero-column": b"preserved"})
        batches = [batch.slice(0, 3), batch.slice(3, 2)]
        expected = pa.Table.from_batches(batches)
        assert expected.num_rows == 5
    calls = {}
    with _dataframe(protocol=protocol, batches=batches, calls=calls) as (frame, _):
        actual, _ = frame._to_table()
    assert actual.num_rows == expected.num_rows
    assert actual.num_columns == expected.num_columns
    assert actual.equals(expected, check_metadata=True)
    assert calls == {"execute": 1}


def test_to_pandas_uses_the_same_handoff(protocol):
    batch = _mixed_batch()
    indices = [0, 2, 4, 5]
    schema = pa.schema(
        [batch.schema.field(index) for index in indices], metadata=batch.schema.metadata
    )
    arrays = [batch.column(index) for index in indices]
    selected = pa.RecordBatch.from_arrays(arrays, schema=schema)
    batches = [selected.slice(0, 2), selected.slice(2, 2)]
    # As SparkConnectClient.to_pandas converts the same table: of the Spark types of its Arrow
    # schema (the server sends no schema here), in the session time zone.
    from pyspark.sql.pandas.conversion import _convert_arrow_table_to_pandas
    from pyspark.sql.pandas.types import from_arrow_schema

    table = pa.Table.from_batches(batches)
    expected = _convert_arrow_table_to_pandas(
        arrow_table=table,
        schema=from_arrow_schema(table.schema, prefer_timestamp_ntz=True),
        timezone="UTC",
        struct_handling_mode="legacy",
        date_as_object=False,
        self_destruct=False,
    )
    calls = {}
    with _dataframe(protocol=protocol, batches=batches, calls=calls) as (frame, _):
        actual = frame.toPandas()
    pd.testing.assert_frame_equal(actual, expected)
    assert calls == {"config": 1, "execute": 1}


@pytest.mark.parametrize("self_destruct", ["false", "true"])
@pytest.mark.parametrize("columns", [2, 0])
def test_to_pandas_copies_each_column_of_each_batch_with_self_destruct(
    protocol, monkeypatch, self_destruct, columns
):
    # As SparkConnectClient._execute_and_fetch with selfDestruct: each column of each batch, of an
    # empty batch too, is copied into its own allocation before the conversion, which then frees
    # the columns as it converts them; so a result without columns has no rows, as a copied batch
    # without columns has none. Without selfDestruct, nothing is copied.
    if columns:
        batch = pa.RecordBatch.from_arrays(
            [pa.array([1, None, 3]), pa.array(["a", None, "c"])], names=["i", "s"]
        )
    else:
        batch = pa.RecordBatch.from_struct_array(pa.array([{}] * 3, type=pa.struct([])))
    batches = [batch.slice(0, 2), batch.slice(2, 0), batch.slice(2, 1)]
    from pyspark.sql.pandas.conversion import _convert_arrow_table_to_pandas
    from pyspark.sql.pandas.types import from_arrow_schema

    table = pa.Table.from_batches(batches)
    if self_destruct == "true" and not columns:
        table = table.slice(0, 0)
    expected = _convert_arrow_table_to_pandas(
        arrow_table=table,
        schema=from_arrow_schema(table.schema, prefer_timestamp_ntz=True),
        timezone="UTC",
        struct_handling_mode="legacy",
        date_as_object=False,
        self_destruct=False,
    )
    copied = []
    concat_arrays = pa.concat_arrays

    def record(arrays, *args, **kwargs):
        copied.append([len(array) for array in arrays])
        return concat_arrays(arrays, *args, **kwargs)

    monkeypatch.setattr(pa, "concat_arrays", record)
    settings = {"spark.sql.execution.arrow.pyspark.selfDestruct.enabled": self_destruct}
    with _dataframe(protocol=protocol, batches=batches, settings=settings) as (frame, _):
        actual = frame.toPandas()
    pd.testing.assert_frame_equal(actual, expected)
    if self_destruct == "true":
        assert copied == [[2]] * columns + [[0]] * columns + [[1]] * columns
    else:
        assert copied == []


def test_to_table_has_the_metadata_that_pyspark_reads_of_a_spark_server(protocol):
    # As PySpark's `_to_table` of the stream of a Spark server, which writes the metadata of the
    # schema and of each field, an empty one where there is none: PyArrow reads it as {}, of the
    # nested fields too, and the keys of the metadata of a field in the order of the server, which
    # the batches that the client decodes do not keep. The columns and their chunks are of the
    # types of the schema, and the values and chunks are those of the batches.
    wkb = pa.field(
        "wkb", pa.binary(), nullable=False, metadata={b"srid": b"0", b"geometry": b"true"}
    )
    spatial = pa.struct([pa.field("srid", pa.int32(), nullable=False), wkb])
    pairs = pa.map_(pa.int32(), pa.struct([pa.field("x", pa.int32())]))
    items = pa.list_(pa.field("element", pa.struct([pa.field("y", pa.int64())])))
    time_field = pa.field(
        "t",
        pa.time64("ns"),
        metadata={b"SPARK::time::precision": b"6", b"SPARK::metadata::json": b"{}"},
    )
    schema = pa.schema(
        [pa.field("g", spatial), pa.field("m", pairs), pa.field("l", items), time_field,
         pa.field("i", pa.int32())]
    )
    batch = pa.RecordBatch.from_arrays(
        [
            pa.array([{"srid": 0, "wkb": b"\x01"}, None], type=spatial),
            pa.array([[(1, {"x": 1})], None], type=pairs),
            pa.array([[{"y": 2}], []], type=items),
            pa.array([1, None], type=pa.time64("ns")),
            pa.array([3, 4], type=pa.int32()),
        ],
        schema=schema,
    )
    batches = [batch.slice(0, 1), batch.slice(1, 0), batch.slice(1, 1)]
    with _dataframe(protocol=protocol, batches=batches) as (frame, _):
        table, _ = frame._to_table()

    def nested(data_type):
        if pa.types.is_struct(data_type):
            fields = [data_type.field(index) for index in range(data_type.num_fields)]
        elif pa.types.is_map(data_type):
            fields = [data_type.key_field, data_type.item_field]
        elif pa.types.is_list(data_type):
            fields = [data_type.value_field]
        else:
            return []
        return [(field.name, field.metadata and list(field.metadata.items()), nested(field.type))
                for field in fields]

    assert table.schema.metadata == {}
    assert [(field.name, list(field.metadata.items()), nested(field.type))
            for field in table.schema] == [
        ("g", [], [("srid", {}, []), ("wkb", [(b"geometry", b"true"), (b"srid", b"0")], [])]),
        ("m", [], [("key", {}, []), ("value", {}, [("x", {}, [])])]),
        ("l", [], [("element", {}, [("y", {}, [])])]),
        ("t", [(b"SPARK::metadata::json", b"{}"), (b"SPARK::time::precision", b"6")], []),
        ("i", [], []),
    ]
    for index, field in enumerate(table.schema):
        column = table.column(index)
        assert nested(column.type) == nested(field.type)
        assert [nested(chunk.type) for chunk in column.chunks] == [nested(field.type)] * 3
        assert [len(chunk) for chunk in column.chunks] == [1, 0, 1]
    assert table.equals(pa.Table.from_batches(batches), check_metadata=True)


@pytest.mark.parametrize(
    "kind", ["field", "nested-field", "list-struct", "map-value-struct", "timezone"]
)
def test_nul_schema_strings_fall_back_without_executing_again(protocol, kind, monkeypatch):
    open_file = pa.ipc.open_file
    ipc_calls = []

    def record_ipc(*args, **kwargs):
        ipc_calls.append(True)
        return open_file(*args, **kwargs)

    monkeypatch.setattr(pa.ipc, "open_file", record_ipc)
    if kind == "field":
        name = "bad\x00name"
        values = pa.array([1, None, -3, 4])
    elif kind == "nested-field":
        name = "nested"
        data_type = pa.struct([pa.field("bad\x00name", pa.int64())])
        values = pa.array(
            [{"bad\x00name": 1}, None, {"bad\x00name": None}, {"bad\x00name": 4}],
            type=data_type,
        )
    elif kind == "list-struct":
        name = "items"
        data_type = pa.list_(pa.struct([pa.field("bad\x00name", pa.int64())]))
        values = pa.array([[{"bad\x00name": 1}], None, [], [{"bad\x00name": None}]], type=data_type)
    elif kind == "map-value-struct":
        name = "mapping"
        data_type = pa.map_(pa.string(), pa.struct([pa.field("bad\x00name", pa.int64())]))
        values = pa.array(
            [[("a", {"bad\x00name": 1})], None, [], [("b", {"bad\x00name": None})]], type=data_type
        )
    else:
        name = "timestamp"
        values = pa.array([0, None, 2, 3], type=pa.timestamp("us", tz="UTC\x00suffix"))
    schema = pa.schema(
        [pa.field(name, values.type, metadata={b"field": b"preserved"})],
        metadata={b"schema": b"preserved"},
    )
    batch = pa.RecordBatch.from_arrays([values], schema=schema)
    batches = [batch.slice(0, 2), batch.slice(2, 2)]
    expected = pa.Table.from_batches(batches)
    calls = {}
    with _dataframe(protocol=protocol, batches=batches, calls=calls) as (frame, _):
        actual, _ = frame._to_table()
    # Compare raw Arrow data, not Python datetimes: the deliberately unusual
    # timezone string is preserved but is not a Python timezone identifier.
    assert actual.equals(expected, check_metadata=True)
    assert [len(chunk) for chunk in actual.column(0).chunks] == [2, 2]
    assert ipc_calls == [True]
    assert calls == {"execute": 1}


def test_nul_schema_and_field_metadata_are_preserved(protocol):
    field = pa.field("value", pa.int64(), metadata={b"field\x00key": b"field\x00value"})
    schema = pa.schema([field], metadata={b"schema\x00key": b"schema\x00value"})
    batch = pa.RecordBatch.from_arrays([pa.array([1, None, 3])], schema=schema)
    calls = {}
    with _dataframe(protocol=protocol, batches=[batch], calls=calls) as (frame, _):
        actual, _ = frame._to_table()
    assert actual.equals(pa.Table.from_batches([batch]), check_metadata=True)
    assert calls == {"execute": 1}


@pytest.mark.parametrize(
    "difference", ["type", "name", "schema-metadata", "field-metadata", "nullability"]
)
@pytest.mark.parametrize("nul_name", [False, True])
def test_mismatched_batch_schemas_are_rejected_before_fallback(protocol, difference, nul_name):
    name = "bad\x00name" if nul_name else "value"
    first_field = pa.field(name, pa.int64(), metadata={b"field": b"original"})
    first_schema = pa.schema([first_field], metadata={b"schema": b"original"})
    second_values = pa.array(["text", "other"]) if difference == "type" else pa.array([3, 4])
    second_name = name + "-different" if difference == "name" else name
    field_metadata = {b"field": b"different"} if difference == "field-metadata" else None
    schema_metadata = {b"schema": b"different"} if difference == "schema-metadata" else None
    second_field = pa.field(
        second_name,
        second_values.type,
        nullable=difference != "nullability",
        metadata=field_metadata or first_field.metadata,
    )
    second_schema = pa.schema([second_field], metadata=schema_metadata or first_schema.metadata)
    batches = [
        pa.RecordBatch.from_arrays([pa.array([1, 2])], schema=first_schema),
        pa.RecordBatch.from_arrays([second_values], schema=second_schema),
    ]
    calls = {}
    with _dataframe(protocol=protocol, batches=batches, calls=calls) as (frame, _):
        with pytest.raises(ValueError, match="inconsistent schemas"):
            frame._to_table()
    assert calls == {"execute": 1}


def test_malformed_ipc_is_reported(protocol):
    calls = {}
    with _dataframe(
        protocol=protocol, batches=[], payloads=[(1, b"not arrow ipc")], calls=calls
    ) as (frame, _):
        with pytest.raises(RuntimeError, match="Arrow"):
            frame._to_table()
    assert calls == {"execute": 1}


@contextmanager
def _arrow_input_service(
    protocol,
    *,
    upload_error=None,
    config_error=False,
    capture_tables=False,
    user_id=None,
    user_agent=None,
    allow_file_artifacts=False,
    **overrides,
):
    """Resolve cache artifacts at a mock service with a 4 MiB receive limit."""
    import hashlib
    import zlib

    config = {
        "spark.sql.timestampType": "TIMESTAMP_LTZ",
        "spark.sql.session.timeZone": "UTC",
        "spark.sql.execution.arrow.useLargeVarTypes": "false",
        "spark.sql.execution.pandas.convertToArrowArraySafely": "true",
        "spark.sql.session.localRelationCacheThreshold": "1048576",
        "spark.sql.session.localRelationSizeLimit": str(3 * 1024**3),
        "spark.sql.session.localRelationChunkSizeRows": "10000",
        "spark.sql.session.localRelationChunkSizeBytes": str(16 * 1024**2),
        "spark.sql.session.localRelationBatchOfChunksSizeBytes": str(1024**3),
    }
    config.update(overrides)
    # The tests may queue status codes in `status_failures` and `upload_failures`: each
    # ArtifactStatus or AddArtifacts call fails with the next one, while any is left.
    state = {"cache": {}, "uploads": 0, "relations": [], "message_sizes": [],
             "config": config, "config_requests": [], "config_responses": [],
             "input_tables": [], "tables": [],
             "upload_contexts": [], "status_contexts": [],
             "status_failures": [], "upload_failures": [], "upload_attempts": 0,
             "ddl_strings": []}
    server_id = "00000000-0000-0000-0000-000000000001"

    def attribution(request, context):
        grpc_user_agent = dict(context.invocation_metadata()).get("user-agent")
        assert grpc_user_agent, "artifact RPC is missing a non-empty user-agent header"
        # Keep only scalar envelope fields; do not retain another payload copy.
        return {
            "session_id": request.session_id,
            "has_user_context": request.HasField("user_context"),
            "user_id": request.user_context.user_id,
            "user_name": request.user_context.user_name,
            "has_client_type": request.HasField("client_type"),
            "grpc_user_agent": grpc_user_agent,
        }

    def configure(raw, context):
        request = protocol("ConfigRequest").FromString(raw)
        if _PLAN_COMPRESSION_THRESHOLD in raw:
            return protocol("ConfigResponse", session_id=request.session_id).SerializeToString()
        operation = request.operation.WhichOneof("op_type")
        defaults = None
        if operation == "get_with_default":
            defaults = {pair.key: pair.value if pair.HasField("value") else None
                        for pair in request.operation.get_with_default.pairs}
            keys = list(defaults)
        else:
            if operation != "get":
                context.abort(grpc.StatusCode.UNIMPLEMENTED,
                              f"unexpected Config operation {operation}")
            keys = list(request.operation.get.keys)
        state["config_requests"].append((operation, keys))
        if config_error:
            # Not retryable, so that a read that a test does not expect fails it at once.
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "configuration unavailable")
        # Injected errors abort only the next Config call.
        if error := state.pop("config_error", None):
            context.abort(error, "configuration read failed")
        # Deliberately reverse responses; Get omits missing keys to test key mapping.
        # A present null key leaves the optional protobuf value unset.
        if defaults is None:
            pairs = [dict(key=k, value=config[k]) for k in reversed(keys) if k in config]
        else:
            pairs = [dict(key=k, value=config.get(k, defaults[k])) for k in reversed(keys)]
        # Of the server-side session ID of the other responses, as a server sends it, which
        # PySpark's client checks against them.
        response = protocol("ConfigResponse", session_id=request.session_id, pairs=pairs,
                            server_side_session_id=server_id)
        state["config_responses"].append(response)
        return response.SerializeToString()

    def top_level_split(text):
        parts, depth, start = [], 0, 0
        for i, char in enumerate(text):
            depth += {"<": 1, ">": -1}.get(char, 0)
            if char == "," and depth == 0:
                parts.append(text[start:i])
                start = i + 1
        return parts + [text[start:]]

    def ddl_type(text):
        """The DataType of a DDL of these tests, as the server's `ddl_parse` parses it: atomic
        types, arrays and maps of them, and structs of `name type` fields."""
        text, kinds = text.strip(), {
            "long": "long", "bigint": "long", "int": "integer", "integer": "integer",
            "double": "double", "float": "float", "string": "string", "date": "date",
            "boolean": "boolean", "binary": "binary", "timestamp": "timestamp"}
        lower = text.lower()
        if lower.startswith("array<") and lower.endswith(">"):
            return protocol("DataType", array=dict(
                element_type=ddl_type(text[6:-1]), contains_null=True))
        if lower.startswith("map<") and lower.endswith(">"):
            key, value = top_level_split(text[4:-1])
            return protocol("DataType", map=dict(
                key_type=ddl_type(key), value_type=ddl_type(value), value_contains_null=True))
        if lower in kinds:
            return protocol("DataType", **{kinds[lower]: {}})
        fields = [field.strip().split(None, 1) for field in top_level_split(text)]
        return protocol("DataType", struct=dict(fields=[
            dict(name=name, data_type=ddl_type(kind), nullable=True) for name, kind in fields]))

    def analyze(raw, context):
        # As the server, which parses the DDL schemas that the client sends (`_parse_ddl`) and
        # answers whether a plan is local: a local relation, cached or chunked, is local. The
        # schema of a plan is that of the result of `execute`, which toArrow asks for first.
        request = protocol("AnalyzePlanRequest").FromString(raw)
        if request.WhichOneof("analyze") == "schema":
            return protocol("AnalyzePlanResponse", session_id=request.session_id,
                            server_side_session_id=server_id,
                            schema=dict(schema=ddl_type("rows long"))).SerializeToString()
        if request.WhichOneof("analyze") == "is_local":
            local = request.is_local.plan.root.WhichOneof("rel_type") in (
                "local_relation", "cached_local_relation", "chunked_cached_local_relation")
            return protocol("AnalyzePlanResponse", session_id=request.session_id,
                            server_side_session_id=server_id,
                            is_local=dict(is_local=local)).SerializeToString()
        if request.WhichOneof("analyze") != "ddl_parse":
            context.abort(grpc.StatusCode.UNIMPLEMENTED, "Method not found!")
        state["ddl_strings"].append(request.ddl_parse.ddl_string)
        return protocol("AnalyzePlanResponse", session_id=request.session_id,
                        server_side_session_id=server_id,
                        ddl_parse=dict(parsed=ddl_type(request.ddl_parse.ddl_string))
                        ).SerializeToString()

    def statuses(raw, context):
        request = protocol("ArtifactStatusesRequest").FromString(raw)
        state["status_contexts"].append(attribution(request, context))
        if state["status_failures"]:
            context.abort(state["status_failures"].pop(0), "artifact status unavailable")
        return protocol("ArtifactStatusesResponse", session_id=request.session_id,
                        statuses={name: dict(exists=name in state["cache"])
                                  for name in request.names}).SerializeToString()

    def upload(requests, context):
        state["upload_attempts"] += 1
        if state["upload_failures"]:
            context.abort(state["upload_failures"].pop(0), "artifact upload unavailable")
        summaries = []
        name, expected, buffer = None, 0, bytearray()
        session_id = None

        def save(name, data):
            named = ("files/", "forward_to_fs/", "jars/", "classes/", "pyfiles/", "archives/")
            if not (allow_file_artifacts and name.startswith(named)):
                assert name == "cache/" + hashlib.sha256(data).hexdigest()
            assert name not in state["cache"], "cached artifacts should not be uploaded again"
            state["cache"][name] = bytes(data)
            summaries.append(dict(name=name, is_crc_successful=True))

        def checked(chunk):
            assert zlib.crc32(chunk.data) == chunk.crc
            return chunk.data

        for raw in requests:
            state["message_sizes"].append(len(raw))
            request = protocol("AddArtifactsRequest").FromString(raw)
            state["upload_contexts"].append({
                **attribution(request, context), "payload": request.WhichOneof("payload")
            })
            session_id = request.session_id
            if request.HasField("batch"):
                for artifact in request.batch.artifacts:
                    save(artifact.name, checked(artifact.data))
            elif request.HasField("begin_chunk"):
                begin = request.begin_chunk
                name, expected = begin.name, begin.total_bytes
                buffer = bytearray(checked(begin.initial_chunk))
            else:
                buffer.extend(checked(request.chunk))
            if name is not None and len(buffer) == expected:
                save(name, buffer)
                name, buffer = None, bytearray()
        assert name is None
        state["uploads"] += 1
        if upload_error == "crc":
            summaries[0]["is_crc_successful"] = False
        elif upload_error == "missing":
            summaries = []
        return protocol("AddArtifactsResponse", session_id=session_id,
                        artifacts=summaries).SerializeToString()

    def execute(raw, context):
        request = protocol("ExecutePlanRequest").FromString(raw)
        relation = request.plan.root
        while relation.HasField("to_df"):
            state["column_names"] = list(relation.to_df.column_names)
            relation = relation.to_df.input
        state["relations"].append(relation)
        if relation.HasField("chunked_cached_local_relation"):
            cached = relation.chunked_cached_local_relation
            payloads = [state["cache"]["cache/" + h] for h in cached.dataHashes]
            state["spark_schema"] = (
                state["cache"]["cache/" + cached.schemaHash].decode()
                if cached.HasField("schemaHash") else None
            )
        else:
            payloads = [relation.local_relation.data]
            state["spark_schema"] = (
                relation.local_relation.schema
                if relation.local_relation.HasField("schema") else None
            )
        rows, first, last = 0, None, None
        state["arrow_schemas"], state["batch_rows"] = [], []
        input_batches = []
        for payload in payloads:
            if capture_tables:
                state["tables"].append(pa.ipc.open_stream(payload).read_all())
            reader = pa.ipc.open_stream(payload)
            state["arrow_schemas"].append(reader.schema)
            stream_rows = []
            state["batch_rows"].append(stream_rows)
            for batch in reader:
                stream_rows.append(batch.num_rows)
                input_batches.append(batch)
                rows += batch.num_rows
                if batch.num_rows:
                    if first is None:
                        first = batch.column(0)[0].as_py()
                    last = batch.column(0)[-1].as_py()
        state["input_tables"].append(pa.Table.from_batches(
            input_batches, schema=state["arrow_schemas"][0]))
        state["result"] = (rows, first, last)
        result = pa.table({"rows": [rows]})
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, result.schema) as writer:
            writer.write_table(result)
        header = dict(session_id=request.session_id, operation_id=request.operation_id,
                      server_side_session_id=server_id)
        yield protocol("ExecutePlanResponse", **header, response_id="0",
                       arrow_batch=dict(row_count=1, data=sink.getvalue().to_pybytes())
                       ).SerializeToString()
        yield protocol("ExecutePlanResponse", **header, response_id="1",
                       result_complete={}).SerializeToString()

    def release(raw, context):
        request = protocol("ReleaseExecuteRequest").FromString(raw)
        return protocol("ReleaseExecuteResponse", session_id=request.session_id,
                        operation_id=request.operation_id,
                        server_side_session_id=server_id).SerializeToString()

    def stop(raw, context):
        request = protocol("ReleaseSessionRequest").FromString(raw)
        return protocol("ReleaseSessionResponse", session_id=request.session_id,
                        server_side_session_id=server_id).SerializeToString()

    with ThreadPoolExecutor(max_workers=2) as executor:
        server = grpc.server(executor, options=[("grpc.max_receive_message_length", 4 * 1024**2)])
        server.add_generic_rpc_handlers([grpc.method_handlers_generic_handler(
            "spark.connect.SparkConnectService", {
                "Config": grpc.unary_unary_rpc_method_handler(configure),
                "AnalyzePlan": grpc.unary_unary_rpc_method_handler(analyze),
                "ArtifactStatus": grpc.unary_unary_rpc_method_handler(statuses),
                "AddArtifacts": grpc.stream_unary_rpc_method_handler(upload),
                "ExecutePlan": grpc.unary_stream_rpc_method_handler(execute),
                "ReleaseExecute": grpc.unary_unary_rpc_method_handler(release),
                "ReleaseSession": grpc.unary_unary_rpc_method_handler(stop),
            })])
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        remote = f"sc://127.0.0.1:{port}/"
        for key, value in [("user_id", user_id), ("user_agent", user_agent)]:
            if value is not None:
                remote += f";{key}={quote(value, safe='')}"
        session = SparkSession.builder.remote(remote).getOrCreate()
        try:
            yield session, state
        finally:
            session.stop()
            server.stop(0).wait(timeout=5)


def _assert_artifact_contexts(contexts, *, session_id, user_id, user_agent):
    assert contexts
    # As PySpark's ArtifactManager, the requests carry no client type, and so not the user agent:
    # their gRPC header is the client library's own, grpcio's in PySpark and tonic's here.
    grpc_user_agents = {context["grpc_user_agent"] for context in contexts}
    assert len(grpc_user_agents) == 1
    grpc_user_agent = grpc_user_agents.pop()
    assert grpc_user_agent.startswith("tonic/"), grpc_user_agent
    assert user_agent not in grpc_user_agent
    for context in contexts:
        assert context["session_id"] == session_id
        # PySpark's ArtifactManager sends a present context even without a user ID.
        assert context["has_user_context"]
        assert context["user_id"] == user_id
        assert context["user_name"] == ""
        assert not context["has_client_type"]


@pytest.mark.parametrize("user_id,user_agent", [
    (None, None),
    ("artifact-user@example.com", "artifact wire test"),
], ids=["default-user", "configured-user"])
def test_cached_arrow_upload_attribution_on_every_frame(protocol, monkeypatch, user_id, user_agent):
    from pyspark.sql import types as T

    # Exercise ChannelBuilder's default prefix when no URL or environment value is set.
    monkeypatch.delenv("SPARK_CONNECT_USER_AGENT", raising=False)
    # Without a user_id in the URL, the user ID is SPARK_USER, as PySpark's.
    monkeypatch.setenv("SPARK_USER", "env-user")
    table = pa.table({"id": range(10_000)})
    schema = T.StructType([T.StructField("id", T.LongType())])
    with _arrow_input_service(protocol, user_id=user_id, user_agent=user_agent, **{
        "spark.sql.session.localRelationCacheThreshold": "0",
    }) as (spark, state):
        # The explicit schema is batched; the IPC data spans begin/chunk frames.
        spark.createDataFrame(table, schema)
        assert {context["payload"] for context in state["upload_contexts"]} == {
            "batch", "begin_chunk", "chunk"
        }
        assert state["status_contexts"]
        _assert_artifact_contexts(
            state["upload_contexts"] + state["status_contexts"],
            session_id=spark.session_id,
            user_id=user_id or "env-user",
            user_agent=user_agent or "_SPARK_CONNECT_PYTHON",
        )


@pytest.mark.parametrize("user_id,user_agent", [
    (None, None),
    ("file-user@example.com", "file wire test"),
], ids=["default-user", "configured-user"])
@pytest.mark.parametrize("upload", ["file", "named"])
def test_file_upload_attribution_on_every_frame(
    protocol, monkeypatch, tmp_path, user_id, user_agent, upload
):
    monkeypatch.delenv("SPARK_CONNECT_USER_AGENT", raising=False)
    # Without a user_id in the URL, the user ID is SPARK_USER, as PySpark's.
    monkeypatch.setenv("SPARK_USER", "env-user")
    small, large = tmp_path / "small.txt", tmp_path / "large.txt"
    small.write_bytes(b"small artifact")
    chunk_size = 32 * 1024  # Mirrors CHUNK_SIZE in crates/spark-connect-core/src/artifact.rs.
    large_size = chunk_size * 3 + 1  # BeginChunk followed by three Chunk frames.
    large.write_bytes(bytes(index % 256 for index in range(large_size)))
    with _arrow_input_service(
        protocol, user_id=user_id, user_agent=user_agent, allow_file_artifacts=True
    ) as (spark, state):
        if upload == "file":
            spark.addArtifacts(str(small), str(large), file=True)
            artifact_names = ["files/small.txt", "files/large.txt"]
        else:
            destinations = ["/wire/small.txt", "/wire/large.txt"]
            for path, destination in zip([small, large], destinations):
                spark.copyFromLocalToFs(str(path), destination)
            # As PySpark's FORWARD_TO_FS_PREFIX + dest_path: the path is absolute, so no separator.
            artifact_names = ["forward_to_fs" + destination for destination in destinations]
        payloads = {context["payload"] for context in state["upload_contexts"]}
        assert payloads == {"batch", "begin_chunk", "chunk"}, (
            f"Expected all artifact frame kinds, got {payloads}. "
            f"This fixture assumes a {chunk_size}-byte CHUNK_SIZE; "
            "adjust its sizes if CHUNK_SIZE changes."
        )
        assert state["cache"][artifact_names[0]] == small.read_bytes()
        assert state["cache"][artifact_names[1]] == large.read_bytes()
        # One upload per path, as PySpark's add_artifacts makes one.
        assert state["uploads"] == 2
        assert not state["status_contexts"]
        _assert_artifact_contexts(
            state["upload_contexts"],
            session_id=spark.session_id,
            user_id=user_id or "env-user",
            user_agent=user_agent or "_SPARK_CONNECT_PYTHON",
        )


def test_cached_arrow_upload_retries_a_transient_failure_from_the_status_check(protocol):
    # As SparkConnectClient.cache_artifacts, the status check and the upload are one attempt
    # under the retry policy: a transient failure of either is retried from the status check.
    table = pa.table({"id": range(10_000)})
    with _arrow_input_service(protocol, **{
        "spark.sql.session.localRelationCacheThreshold": "0",
    }) as (spark, state):
        state["status_failures"].append(grpc.StatusCode.UNAVAILABLE)
        state["upload_failures"].append(grpc.StatusCode.UNAVAILABLE)
        df = spark.createDataFrame(table)
        assert not state["status_failures"] and not state["upload_failures"]
        assert state["upload_attempts"] == state["uploads"] + 1
        # The failed status check, the one before the failed upload, and the one before the
        # upload; the server then reads the cached data: its rows, first and last values.
        assert len(state["status_contexts"]) == 3
        df.toArrow()
        assert state["result"] == (10_000, 0, 9_999)


def test_cached_arrow_upload_does_not_retry_another_failure(protocol):
    # As PySpark's SparkConnectClient.cache_artifacts, which leaves the failure of its RPC to
    # grpc: createDataFrame raises grpcio's RpcError of the upload, unconverted.
    table = pa.table({"id": range(10_000)})
    with _arrow_input_service(protocol, **{
        "spark.sql.session.localRelationCacheThreshold": "0",
    }) as (spark, state):
        state["upload_failures"].append(grpc.StatusCode.PERMISSION_DENIED)
        with pytest.raises(grpc.RpcError, match="artifact upload unavailable") as raised:
            spark.createDataFrame(table)
        assert raised.value.code() == grpc.StatusCode.PERMISSION_DENIED
        assert state["upload_attempts"] == 1 and state["uploads"] == 0


def test_artifact_upload_retries_a_transient_failure_of_each_path(protocol, tmp_path):
    # As SparkConnectClient.add_artifacts, each path is uploaded in an attempt of its own.
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    first.write_bytes(b"first artifact")
    second.write_bytes(b"second artifact")
    with _arrow_input_service(protocol, allow_file_artifacts=True) as (spark, state):
        state["upload_failures"].append(grpc.StatusCode.UNAVAILABLE)
        spark.addArtifacts(str(first), str(second), file=True)
        assert state["cache"]["files/first.txt"] == b"first artifact"
        assert state["cache"]["files/second.txt"] == b"second artifact"
        assert (state["upload_attempts"], state["uploads"]) == (3, 2)


def test_artifact_upload_sends_each_path_before_it_parses_the_next(protocol, tmp_path):
    # As SparkConnectClient.add_artifacts, which gives its ArtifactManager one path at a time, and
    # so SparkSession.addArtifacts: a path is parsed and uploaded before the next one is parsed,
    # so the error of a later path, of a missing file or of a format that the flags do not take,
    # is raised once the paths before it are uploaded, which the server keeps, and the paths
    # after it are not uploaded.
    from pyspark.errors import PySparkRuntimeError

    for name in ("first.py", "second.py", "third.py", "note.txt"):
        (tmp_path / name).write_text("x = 1\n")
    first, second, third, note = (
        str(tmp_path / name) for name in ("first.py", "second.py", "third.py", "note.txt"))
    missing = str(tmp_path / "missing.py")
    with _arrow_input_service(protocol, allow_file_artifacts=True) as (spark, state):
        with pytest.raises(FileNotFoundError):
            spark.addArtifacts(first, missing, third, pyfile=True)
        assert (sorted(state["cache"]), state["uploads"]) == (["pyfiles/first.py"], 1)
        with pytest.raises(PySparkRuntimeError) as error:
            spark.client.add_artifacts(second, note, pyfile=True, archive=False, file=False)
        assert error.value.getCondition() == "UNSUPPORTED_OPERATION"
        assert error.value.getMessageParameters() == {"operation": f"{note} file format"}
        assert sorted(state["cache"]) == ["pyfiles/first.py", "pyfiles/second.py"]
        with pytest.raises(FileNotFoundError):
            spark.addArtifacts(missing, third, pyfile=True)
        assert state["uploads"] == 2
        spark.addArtifacts(third, note, file=True)
        assert sorted(state["cache"]) == [
            "files/note.txt", "files/third.py", "pyfiles/first.py", "pyfiles/second.py"]
        assert state["uploads"] == 4


def test_artifact_flags_are_any_objects_as_pyspark(protocol, tmp_path):
    # As PySpark's SparkSession.addArtifacts (and addArtifact, the same method), which refuses its
    # flags first with `sum([file, pyfile, archive]) > 1`, as Python adds and compares them, and
    # SparkConnectClient.add_artifacts, which only tests their truth: an integer, a float or
    # another object is a flag of its truth, a single flag of 2 is two, and a value that sum does
    # not add raises its TypeError, also without a path, before anything is sent.
    import inspect
    import re

    from pyspark.errors import PySparkValueError

    names = ["a.txt", "b.txt", "c.txt", "d.txt", "m.py"]
    for name in names:
        (tmp_path / name).write_text("x = 1\n")
    a, b, c, d, module = (str(tmp_path / name) for name in names)
    with _arrow_input_service(protocol, allow_file_artifacts=True) as (spark, state):
        signature = "(*path, pyfile=False, archive=False, file=False)"
        assert str(inspect.signature(spark.addArtifacts)) == signature
        assert str(inspect.signature(spark.addArtifact)) == signature
        spark.addArtifacts(a, file=1)
        spark.addArtifact(b, archive=0, file=1)
        spark.addArtifacts(c, file=0.5)
        spark.client.add_artifacts(d, pyfile=None, archive="", file=[1])
        # The client does not check the flags: a path takes the first one that it matches.
        spark.client.add_artifacts(module, pyfile=1, archive=1, file=0)
        assert sorted(state["cache"]) == [
            "files/a.txt", "files/b.txt", "files/c.txt", "files/d.txt", "pyfiles/m.py"]
        uploads = state["uploads"]
        for flags in [dict(pyfile=1, file=1), dict(pyfile=2), dict(archive=1.5)]:
            with pytest.raises(PySparkValueError) as error:
                spark.addArtifacts(module, **flags)
            assert error.value.getCondition() == "INVALID_MULTIPLE_ARGUMENT_CONDITIONS"
        for flags, operand in [
            (dict(pyfile="yes"), "str"), (dict(pyfile=None), "NoneType"), (dict(file=[1]), "list")
        ]:
            message = f"unsupported operand type(s) for +: 'int' and '{operand}'"
            with pytest.raises(TypeError, match=re.escape(message)):
                spark.addArtifacts(module, **flags)
        with pytest.raises(TypeError, match=re.escape("for +: 'int' and 'str'")):
            spark.addArtifacts(pyfile="yes")
        assert state["uploads"] == uploads


def test_copy_from_local_to_fs_takes_a_path_like_local_path_as_pyspark(
        protocol, tmp_path, monkeypatch):
    # As PySpark's `_parse_forward_to_fs_artifacts`, of `Path(local_path).absolute()`: the local
    # path of SparkSession.copyFromLocalToFs and of SparkConnectClient.copy_from_local_to_fs is a
    # str or an os.PathLike, such as a pathlib.Path, also a relative one, whose file is uploaded
    # under the destination, and another value raises the TypeError of Path. A destination that
    # is not a str stays a TypeError, as the client of PySpark raises one.
    local = tmp_path / "local.txt"
    local.write_bytes(b"local artifact")
    monkeypatch.chdir(tmp_path)
    with _arrow_input_service(protocol, allow_file_artifacts=True) as (spark, state):
        spark.copyFromLocalToFs(local, "/wire/a.txt")
        spark.client.copy_from_local_to_fs(local, "/wire/b.txt")
        spark.copyFromLocalToFs(Path("local.txt"), "/wire/c.txt")
        assert state["cache"] == {
            f"forward_to_fs/wire/{name}.txt": b"local artifact" for name in "abc"}
        uploads = state["uploads"]
        for value in (5, str(local).encode()):
            with pytest.raises(TypeError) as expected:
                Path(value)
            with pytest.raises(TypeError) as error:
                spark.copyFromLocalToFs(value, "/wire/d.txt")
            assert str(error.value) == str(expected.value)
        with pytest.raises(AssertionError, match="local path must be a file path."):
            spark.copyFromLocalToFs(tmp_path / "missing.txt", "/wire/d.txt")
        with pytest.raises(TypeError):
            spark.client.copy_from_local_to_fs(local, Path("/wire/d.txt"))
        assert state["uploads"] == uploads


@pytest.mark.parametrize("cached", [False, True], ids=["inline", "cached"])
@pytest.mark.parametrize("explicit", [False, True], ids=["inferred", "explicit"])
def test_arrow_input_schema_handoff(protocol, cached, explicit):
    import json

    from pyspark.sql import types as T

    table = pa.table({"id": [1, 2]})
    schema = T.StructType([T.StructField("id", T.LongType())]) if explicit else None
    with _arrow_input_service(protocol, **{
        "spark.sql.session.localRelationCacheThreshold": "0" if cached else "1048576",
    }) as (spark, state):
        frame = spark.createDataFrame(table, schema)
        # As in PySpark: inferred column names are applied with toDF, which is not local.
        assert frame.isLocal() == explicit
        assert frame.toArrow().column(0).to_pylist() == [2]
        assert state["result"] == (2, 1, 2)
        assert state["arrow_schemas"] == [table.schema]
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
        relation = state["relations"][0]
        if cached:
            local = relation.chunked_cached_local_relation
            assert local.HasField("schemaHash") == explicit
            assert len(state["cache"]) == (2 if explicit else 1)
            uploads = state["uploads"]
            spark.createDataFrame(table, schema).toArrow()
            assert state["uploads"] == uploads
        else:
            assert state["uploads"] == 0
            assert relation.local_relation.HasField("schema") == explicit
        if explicit:
            assert json.loads(state["spark_schema"]) == json.loads(schema.json())
        else:
            assert state["spark_schema"] is None
            assert state["column_names"] == ["id"]


def test_arrow_input_upload_above_grpc_limit(protocol):
    import numpy as np

    rows = 128 * 1024**2 // 8 + 1
    table = pa.table({"id": np.arange(rows, dtype=np.int64)})
    assert table.nbytes > 128 * 1024**2
    with _arrow_input_service(protocol) as (spark, state):
        df = spark.createDataFrame(table)
        assert not df.isLocal()
        assert df.toArrow().column(0).to_pylist() == [rows]
        assert state["result"] == (rows, 0, rows - 1)
        assert state["relations"][0].HasField("chunked_cached_local_relation")
        assert state["relations"][0].ByteSize() < 1024**2
        assert max(state["message_sizes"]) < 33 * 1024
        assert state["spark_schema"] is None
        assert all(schema == table.schema for schema in state["arrow_schemas"])


@pytest.mark.parametrize("kind", ["inferred", "empty-list", "struct"])
def test_arrow_input_zero_column_table(protocol, kind):
    from pyspark.sql import types as T

    # The input cast drops the rows of a zero-column table, leaving no batches.
    table = pa.table({"a": [1, 2]}).select([])
    schema = {"inferred": None, "empty-list": [], "struct": T.StructType([])}[kind]
    explicit = kind == "struct"
    with _arrow_input_service(protocol) as (spark, state):
        frame = spark.createDataFrame(table, schema)
        # As in PySpark, no toDF is applied without column names, so the frame is local.
        assert frame.isLocal()
        assert frame.toArrow().column(0).to_pylist() == [0]
        assert "column_names" not in state
        # The relation still carries the Arrow schema, as PySpark's IPC stream does.
        assert state["arrow_schemas"] == [pa.schema([])]
        assert state["relations"][0].local_relation.HasField("schema") == explicit


@pytest.mark.parametrize("schema,names", [
    ("x long", "['x']"),
    ("x long, y long, z long", "['x', 'y', 'z']"),
])
def test_arrow_input_ddl_struct_of_another_width_fails_the_cast(protocol, schema, names):
    # As PySpark Connect's, which casts the table to the struct, whose names PyArrow's cast
    # checks against the table's, before any width check.
    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(ValueError) as error:
            spark.createDataFrame(pa.table({"x": [1], "y": [2]}), schema)
        assert type(error.value) is ValueError
        assert str(error.value) == (
            "Target schema's field names are not matching the table's field names: "
            f"['x', 'y'], {names}"
        )
        # As PySpark Connect's, the server parses the DDL schema.
        assert state["ddl_strings"] == [schema]
        assert not state["relations"]
        assert state["uploads"] == 0


def test_arrow_input_ddl_width_errors(protocol):
    from pyspark.errors import PySparkValueError

    # An atomic type types one column: as PySpark Connect's, the table is cast to the schema of
    # its own types, and then has another number of columns.
    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(PySparkValueError) as error:
            spark.createDataFrame(pa.table({"x": [1], "y": [2]}), "long")
        assert error.value.getCondition() == "AXIS_LENGTH_MISMATCH"
        assert error.value.getMessageParameters() == {
            "expected_length": "1", "actual_length": "2",
        }
        assert state["ddl_strings"] == ["long"]
        assert not state["relations"]
        assert state["uploads"] == 0


@pytest.mark.parametrize("schema,expected_type", [
    ("array<long>", "ArrayType"),
    ("map<string,long>", "MapType"),
])
def test_arrow_input_unsupported_ddl_schema(protocol, schema, expected_type):
    from pyspark.errors import PySparkTypeError

    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(PySparkTypeError) as error:
            spark.createDataFrame(pa.table({"x": [1]}), schema)
        assert error.value.getCondition() == "NOT_EXPECTED_TYPE"
        assert error.value.getMessageParameters() == {
            "arg_name": "schema", "expected_type": "list, None or StructType",
            "arg_type": expected_type,
        }
        assert not state["relations"]
        assert state["uploads"] == 0


def test_arrow_input_upload_limits_and_chunk_order(protocol):
    import numpy as np
    from pyspark.errors import AnalysisException
    table = pa.table({"id": np.arange(100, dtype=np.int64)})
    options = {
        "spark.sql.session.localRelationCacheThreshold": "0",
        "spark.sql.session.localRelationChunkSizeRows": "7",
        "spark.sql.session.localRelationChunkSizeBytes": "500",
        "spark.sql.session.localRelationBatchOfChunksSizeBytes": "1000",
    }
    with _arrow_input_service(protocol, **options) as (spark, state):
        df = spark.createDataFrame(table)
        df.toArrow()
        assert state["result"] == (100, 0, 99)
        assert state["uploads"] > 1
        cached = state["relations"][0].chunked_cached_local_relation
        assert len(cached.dataHashes) > 1
        assert all(rows <= 7 for stream in state["batch_rows"] for rows in stream)
        assert any(sum(stream) > 7 for stream in state["batch_rows"])
        actual = [
            value
            for hash in cached.dataHashes
            for batch in pa.ipc.open_stream(state["cache"]["cache/" + hash])
            for value in batch.column(0).to_pylist()
        ]
        assert actual == list(range(100))

    options["spark.sql.session.localRelationSizeLimit"] = "10"
    with _arrow_input_service(protocol, **options) as (spark, state):
        with pytest.raises(AnalysisException, match="LOCAL_RELATION_SIZE_LIMIT_EXCEEDED"):
            spark.createDataFrame(table)
        assert not state["relations"]
        assert state["uploads"] == 0


@pytest.mark.parametrize("setting,value", [
    ("SizeLimit", "0"),
    ("SizeLimit", "-1"),
    ("SizeLimit", "18446744073709551616"),
    ("ChunkSizeRows", "0"),
])
def test_arrow_input_rejects_invalid_upload_sizes(protocol, setting, value):
    key = f"spark.sql.session.localRelation{setting}"
    with _arrow_input_service(protocol, **{
        "spark.sql.session.localRelationCacheThreshold": "0", key: value,
    }) as (spark, state):
        with pytest.raises(Exception, match=f"Invalid configuration: {key}"):
            spark.createDataFrame(pa.table({"id": [1]}))
        assert state["uploads"] == 0


@pytest.mark.parametrize("failure,match", [("crc", "CRC check failed"),
                                          ("missing", "Missing cache artifact")])
@pytest.mark.parametrize("rows", [2, 4000000])
def test_arrow_input_upload_failure(protocol, failure, match, rows):
    import numpy as np

    # The larger input spans multiple upload batches.
    with _arrow_input_service(protocol, upload_error=failure,
                             **{"spark.sql.session.localRelationCacheThreshold": "0",
                                "spark.sql.session.localRelationBatchOfChunksSizeBytes":
                                    str(8 * 1024**2)}) as (spark, state):
        with pytest.raises(Exception, match=match):
            spark.createDataFrame(pa.table({"id": np.arange(rows, dtype=np.int64)}))
        assert not state["relations"]


@pytest.mark.parametrize("batch_bytes", [500, 1024**3])
def test_arrow_input_duplicate_chunks_preserve_multiplicity(protocol, batch_bytes):
    options = {"spark.sql.session.localRelationCacheThreshold": "0",
               "spark.sql.session.localRelationChunkSizeRows": "5",
               "spark.sql.session.localRelationChunkSizeBytes": "1",
               "spark.sql.session.localRelationBatchOfChunksSizeBytes": str(batch_bytes)}
    with _arrow_input_service(protocol, **options) as (spark, state):
        spark.createDataFrame(pa.table({"id": [1] * 20})).toArrow()
        cached = state["relations"][0].chunked_cached_local_relation
        assert len(cached.dataHashes) == 4
        assert len(set(cached.dataHashes)) == 1
        assert not cached.HasField("schemaHash")
        assert len(state["cache"]) == 1
        assert state["result"] == (20, 1, 1)


_NORMALIZATION_KEYS = [
    "spark.sql.timestampType",
    "spark.sql.session.timeZone",
    "spark.sql.execution.arrow.useLargeVarTypes",
]
_UPLOAD_KEYS = [
    "spark.sql.session.localRelationCacheThreshold",
    "spark.sql.session.localRelationSizeLimit",
    "spark.sql.session.localRelationChunkSizeRows",
    "spark.sql.session.localRelationChunkSizeBytes",
    "spark.sql.session.localRelationBatchOfChunksSizeBytes",
]
# The settings that PySpark 4.3.0's createDataFrame reads, in one Config get, in its order: those
# above and those of its other conversions, before it converts any data that is not empty.
_CREATE_DATAFRAME_KEYS = [
    "spark.sql.timestampType",
    "spark.sql.session.timeZone",
    *_UPLOAD_KEYS,
    "spark.sql.execution.pandas.convertToArrowArraySafely",
    "spark.sql.execution.pandas.inferPandasDictAsMap",
    "spark.sql.pyspark.inferNestedDictAsStruct.enabled",
    "spark.sql.pyspark.legacy.inferArrayTypeFromFirstElement.enabled",
    "spark.sql.pyspark.legacy.inferMapTypeFromFirstPair.enabled",
    "spark.sql.execution.arrow.useLargeVarTypes",
]
_BINARY_RESULT_CONFIG = ("get_with_default", ["spark.sql.execution.pyspark.binaryAsBytes"])


@pytest.mark.parametrize("timestamp_type", ["TIMESTAMP_LTZ", "TIMESTAMP_NTZ"])
@pytest.mark.parametrize("zone", ["UTC", "America/Los_Angeles"])
@pytest.mark.parametrize("large", ["false", "true"])
def test_arrow_input_batched_normalization_configs(protocol, timestamp_type, zone, large):
    from zoneinfo import ZoneInfo

    value = datetime(2024, 1, 15, 12, 34, 56, 123456)
    table = pa.table({"t": [value, None], "s": ["hello", None], "b": [b"xyz", None]})
    with _arrow_input_service(protocol, **dict(zip(
        _NORMALIZATION_KEYS, [timestamp_type, zone, large]
    ))) as (spark, state):
        frame = spark.createDataFrame(table)
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
        frame.collect()
        assert state["config_requests"] == [
            ("get", _CREATE_DATAFRAME_KEYS), _BINARY_RESULT_CONFIG]
        expected_time = (value if timestamp_type == "TIMESTAMP_NTZ" else
                         value.replace(tzinfo=ZoneInfo(zone)).astimezone(timezone.utc))
        expected = pa.table({
            "t": pa.array([expected_time, None], type=pa.timestamp(
                "us", tz=None if timestamp_type == "TIMESTAMP_NTZ" else "UTC")),
            "s": pa.array(["hello", None], type=pa.large_string() if large == "true"
                          else pa.string()),
            "b": pa.array([b"xyz", None], type=pa.large_binary() if large == "true"
                          else pa.binary()),
        })
        actual = state["input_tables"][0]
        assert actual.schema == expected.schema
        assert actual.to_pydict() == expected.to_pydict()
        # Inferred input keeps its types in Arrow; names are applied via toDF.
        assert not state["relations"][0].local_relation.HasField("schema")


@pytest.mark.parametrize("large", ["false", "true"])
def test_local_rows_and_pandas_strings_take_the_large_types_of_pyspark(protocol, large):
    # As PySpark's createDataFrame under spark.sql.execution.arrow.useLargeVarTypes, whose
    # LocalDataToArrowConversion and pandas conversion make the strings and binaries of the
    # schema large (conversion.py:994-1005, pandas/types.py:137-139), also nested: local rows,
    # and the object strings of a pandas frame of 10000 rows, which the binding encodes itself.
    string, binary = ((pa.large_string(), pa.large_binary()) if large == "true"
                      else (pa.string(), pa.binary()))
    schema = T.StructType([
        T.StructField("s", T.StringType()),
        T.StructField("b", T.BinaryType()),
        T.StructField("a", T.ArrayType(T.StringType())),
        T.StructField("m", T.MapType(T.StringType(), T.BinaryType())),
        T.StructField("st", T.StructType([T.StructField("x", T.StringType())])),
        T.StructField("l", T.LongType()),
    ])
    rows = [("x", b"y", ["z"], {"k": b"v"}, ("w",), 1), (None,) * 6]
    overrides = {"spark.sql.execution.arrow.useLargeVarTypes": large}
    with _arrow_input_service(protocol, **overrides) as (spark, state):
        spark.createDataFrame(rows, schema).collect()
        table = state["input_tables"][-1]
        types = {field.name: field.type for field in table.schema}
        assert (types["s"], types["b"], types["l"]) == (string, binary, pa.int64())
        assert types["a"].value_type == string
        assert (types["m"].key_type, types["m"].item_type) == (string, binary)
        assert types["st"].field("x").type == string
        assert table.column("s").to_pylist() == ["x", None]
        assert table.column("m").to_pylist() == [[("k", b"v")], None]
        frame = pd.DataFrame({"s": ["abcd"] * 10000})
        one = T.StructType([T.StructField("s", T.StringType())])
        assert spark.createDataFrame(frame, one).collect()[0].rows == 10000
        table = state["input_tables"][-1]
        assert table.schema.field("s").type == string
        assert table.column("s").to_pylist() == ["abcd"] * 10000


@pytest.mark.parametrize("threshold, cached", [("8001", False), ("8004", False), ("8000", True)])
def test_the_cache_threshold_is_compared_with_the_nbytes_of_pyarrow(protocol, threshold, cached):
    # As PySpark's `cache_threshold <= _table.nbytes` (connect/session.py:782): pyarrow's
    # nbytes counts `len` offsets of a string array, not the `len + 1` of its buffer, so 1000
    # strings "abcd" are 8000 bytes, inline below a threshold of 8001 and uploaded from 8000.
    overrides = {"spark.sql.session.localRelationCacheThreshold": threshold}
    one = T.StructType([T.StructField("s", T.StringType())])
    with _arrow_input_service(protocol, **overrides) as (spark, state):
        assert spark.createDataFrame([("abcd",)] * 1000, one).collect()[0].rows == 1000
        relation = state["relations"][-1]
        assert relation.HasField("chunked_cached_local_relation") == cached
        assert relation.HasField("local_relation") == (not cached)


@pytest.mark.parametrize("missing", _NORMALIZATION_KEYS + _UPLOAD_KEYS)
@pytest.mark.parametrize("present_null", [False, True])
def test_arrow_input_missing_config(protocol, missing, present_null):
    with _arrow_input_service(protocol) as (spark, state):
        if present_null:
            state["config"][missing] = None
        else:
            del state["config"][missing]
        # The current Arrow settings snapshot requires every server setting.
        with pytest.raises(Exception, match="SQL_CONF_NOT_FOUND"):
            spark.createDataFrame(pa.table({"x": [1, None], "s": ["hello", None]}))
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
        pairs = {pair.key: pair for pair in state["config_responses"][0].pairs}
        if present_null:
            assert missing in pairs
            assert not pairs[missing].HasField("value")
        else:
            assert missing not in pairs
        assert state["uploads"] == 0
        assert state["input_tables"] == []


def test_arrow_input_normalization_config_failure_then_fresh_read(protocol):
    with _arrow_input_service(protocol) as (spark, state):
        state["config_error"] = grpc.StatusCode.INVALID_ARGUMENT
        table = pa.table({"x": [1]})
        with pytest.raises(Exception, match="configuration read failed"):
            spark.createDataFrame(table)
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
        assert state["uploads"] == 0
        spark.createDataFrame(table).collect()
        assert state["config_requests"] == [
            ("get", _CREATE_DATAFRAME_KEYS),
            ("get", _CREATE_DATAFRAME_KEYS), _BINARY_RESULT_CONFIG]
        assert state["input_tables"][0].to_pydict() == {"x": [1]}


def test_arrow_input_normalization_config_transient_retry(protocol):
    with _arrow_input_service(protocol) as (spark, state):
        state["config_error"] = grpc.StatusCode.UNAVAILABLE
        spark.createDataFrame(pa.table({"x": [1]})).collect()
        assert state["config_requests"] == [
            ("get", _CREATE_DATAFRAME_KEYS),
            ("get", _CREATE_DATAFRAME_KEYS), _BINARY_RESULT_CONFIG]
        assert state["input_tables"][0].to_pydict() == {"x": [1]}


def test_arrow_input_normalization_configs_are_fresh(protocol):
    value = datetime(2024, 1, 15, 12)
    table = pa.table({"t": [value], "s": ["hello"]})
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(table).collect()
        state["config"].update(dict(zip(_NORMALIZATION_KEYS, [
            "TIMESTAMP_NTZ", "America/Los_Angeles", "true"])))
        spark.createDataFrame(table).collect()
        state["config"].update(dict(zip(_NORMALIZATION_KEYS, [
            "TIMESTAMP_LTZ", "America/Los_Angeles", "false"])))
        spark.createDataFrame(table).collect()
        assert state["config_requests"] == [
            ("get", _CREATE_DATAFRAME_KEYS), _BINARY_RESULT_CONFIG] * 3
        tables = state["input_tables"]
        assert [t.column("t")[0].as_py() for t in tables] == [
            value.replace(tzinfo=timezone.utc), value,
            datetime(2024, 1, 15, 20, tzinfo=timezone.utc)]
        assert [t.schema.field("s").type for t in tables] == [
            pa.string(), pa.large_string(), pa.string()]
        assert all(not relation.local_relation.HasField("schema")
                   for relation in state["relations"])


def test_arrow_input_empty_skips_all_configs(protocol):
    from pyspark.sql.types import LongType, StructField, StructType

    with _arrow_input_service(protocol) as (spark, state):
        table = pa.table({"x": pa.array([], type=pa.int64())})
        spark.createDataFrame(table, StructType([StructField("x", LongType())]))
        with pytest.raises(Exception, match="CANNOT_INFER_EMPTY_SCHEMA"):
            spark.createDataFrame(table)
        assert state["config_requests"] == []
        assert state["uploads"] == 0


@pytest.mark.parametrize("data", [[[1]], pd.DataFrame({"id": [1]})])
def test_non_arrow_input_does_not_import_pyarrow(protocol, monkeypatch, data):
    import builtins
    import sys

    imported = []
    original_import = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name == "pyarrow":
            imported.append(name)
            raise ImportError("PyArrow is unavailable")
        return original_import(name, *args, **kwargs)

    with _arrow_input_service(protocol) as (spark, state):
        monkeypatch.setitem(sys.modules, "pyarrow", None)
        monkeypatch.setattr(builtins, "__import__", import_module)
        spark.createDataFrame(data)
        assert imported == []
        # The settings are read, as PySpark's createDataFrame reads them, without pyarrow.
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]


def test_add_artifacts_names_each_artifact_as_pyspark(protocol, tmp_path, monkeypatch):
    import sys

    for name in ["a.jar", "A.class", "m.py", "b.zip", "n.txt"]:
        (tmp_path / name).write_bytes(name.encode())
    monkeypatch.setattr(sys, "path", list(sys.path))
    with _arrow_input_service(protocol, allow_file_artifacts=True) as (spark, state):
        # As PySpark's ArtifactManager names them, by the flags and the file extension. A class
        # file is uploaded only with the file flag: PySpark refuses it otherwise.
        spark.addArtifacts(str(tmp_path / "a.jar"))
        spark.addArtifacts(str(tmp_path / "m.py"), pyfile=True)
        spark.addArtifacts(str(tmp_path / "b.zip"), pyfile=True)
        spark.addArtifacts(str(tmp_path / "b.zip") + "#alias", archive=True)
        spark.addArtifact(str(tmp_path / "n.txt"), file=True)
        spark.addArtifacts(str(tmp_path / "A.class"), file=True)
        assert state["cache"] == {
            "jars/a.jar": b"a.jar",
            "pyfiles/m.py": b"m.py",
            "pyfiles/b.zip": b"b.zip",
            "archives/b.zip#alias": b"b.zip",
            "files/n.txt": b"n.txt",
            "files/A.class": b"A.class",
        }
        # As PySpark's _parse_artifacts, a zip of pyfile is importable by the client.
        assert sys.path[1] == str(tmp_path / "b.zip")


# Local relations of 64 KiB or more read the server's local-relation settings, on every path
# and for either schema form, and follow the cache threshold and the size limit as the Arrow
# importer does, as PySpark does. Smaller relations stay inline without that RPC.

_LOCAL_RELATION_PATHS = ["local_rows", "columns", "pandas_rows", "float64", "string", "int64"]


def _without_pyarrow(monkeypatch):
    """Make createDataFrame convert pandas frames with its row path, as it does where PyArrow,
    which PySpark's conversion and the fast paths of plain frames need, cannot be imported."""
    import importlib

    # The module that the native binding imports, rather than an attribute of the package,
    # which earlier import-hook tests can leave pointing at another module object.
    helper = importlib.import_module("pyspark.sql._pandas_input")
    monkeypatch.setattr(helper, "pyarrow_available", lambda: False)


def _local_relation_input(monkeypatch, path, form):
    """Data of `path` of at least 64 KiB of Arrow data, its row count, and its schema as a
    StructType or as the DDL of the same fields."""
    if path == "local_rows":
        data = [("y" * 100, i) for i in range(1000)]
        fields, ddl = [("v", T.StringType()), ("i", T.LongType())], "v string, i long"
    elif path == "columns":
        # Python ints under a StructType of longs take the columns path.
        data = [(i, i) for i in range(10000)]
        fields, ddl = [("a", T.LongType()), ("b", T.LongType())], "a long, b long"
    elif path == "pandas_rows":
        # Without PyArrow, the rows of a pandas frame are converted.
        _without_pyarrow(monkeypatch)
        data = pd.DataFrame({"v": ["y" * 100] * 1000, "i": np.arange(1000, dtype=np.int64)})
        fields, ddl = [("v", T.StringType()), ("i", T.LongType())], "v string, i long"
    elif path == "float64":
        data = pd.DataFrame({"c": np.ones(100000)})
        fields, ddl = [("c", T.DoubleType())], "c double"
    elif path == "string":
        data = pd.DataFrame({"v": ["x" * 10] * 10000}, dtype=object)
        fields, ddl = [("v", T.StringType())], "v string"
    else:
        data = pd.DataFrame({"i": np.arange(10000, dtype=np.int64)})
        fields, ddl = [("i", T.LongType())], "i long"
    schema = T.StructType([T.StructField(name, kind) for name, kind in fields]) if form == "struct" else ddl
    return data, schema, len(data)


@pytest.mark.parametrize("form", ["struct", "ddl"])
@pytest.mark.parametrize("threshold", ["1", "1048576"])
@pytest.mark.parametrize("path", _LOCAL_RELATION_PATHS)
def test_local_relations_of_64_kib_read_the_settings_at_the_threshold(
    protocol, monkeypatch, path, threshold, form
):
    data, schema, rows = _local_relation_input(monkeypatch, path, form)
    with _arrow_input_service(
        protocol, **{"spark.sql.session.localRelationCacheThreshold": threshold}
    ) as (spark, state):
        df = spark.createDataFrame(data, schema)
        df.toArrow()
        assert any("spark.sql.session.localRelationCacheThreshold" in keys
                   for _, keys in state["config_requests"])
        assert sum(table.num_rows for table in state["input_tables"]) == rows
        # The default threshold of 1 MiB is above these relations: they stay inline.
        uploaded = threshold == "1"
        assert bool(state["uploads"]) == uploaded
        assert state["relations"][0].HasField("chunked_cached_local_relation") == uploaded
        assert state["relations"][0].HasField("local_relation") == (not uploaded)


@pytest.mark.parametrize("form", ["struct", "ddl"])
@pytest.mark.parametrize("path", _LOCAL_RELATION_PATHS)
def test_local_relations_of_64_kib_past_the_size_limit_raise(protocol, monkeypatch, path, form):
    from pyspark.errors import AnalysisException

    data, schema, _ = _local_relation_input(monkeypatch, path, form)
    with _arrow_input_service(protocol, **{
        "spark.sql.session.localRelationCacheThreshold": "1",
        "spark.sql.session.localRelationSizeLimit": "10000",
    }) as (spark, state):
        with pytest.raises(AnalysisException, match="LOCAL_RELATION_SIZE_LIMIT_EXCEEDED") as error:
            spark.createDataFrame(data, schema)
        assert "(10000 bytes)" in str(error.value)
        assert state["uploads"] == 0
        assert not state["relations"]


@pytest.mark.parametrize("path", _LOCAL_RELATION_PATHS)
def test_local_relations_of_64_kib_the_same_for_struct_and_ddl(protocol, monkeypatch, path):
    from pyspark.errors import AnalysisException

    outcomes = {}
    for form in ["struct", "ddl"]:
        data, schema, rows = _local_relation_input(monkeypatch, path, form)
        with _arrow_input_service(protocol, **{
            "spark.sql.session.localRelationCacheThreshold": "1",
        }) as (spark, state):
            spark.createDataFrame(data, schema).toArrow()
            outcomes[form] = (
                sum(table.num_rows for table in state["input_tables"]),
                bool(state["uploads"]),
                len(state["relations"]),
            )
        assert outcomes[form][0] == rows
        with _arrow_input_service(protocol, **{
            "spark.sql.session.localRelationCacheThreshold": "1",
            "spark.sql.session.localRelationSizeLimit": "10000",
        }) as (spark, state):
            with pytest.raises(AnalysisException, match="LOCAL_RELATION_SIZE_LIMIT_EXCEEDED"):
                spark.createDataFrame(data, schema)
            outcomes[form] += ("raised",)
    assert outcomes["struct"] == outcomes["ddl"]
    assert outcomes["struct"][1:] == (True, 1, "raised")


# The inputs below are tens of MiB, beyond what one gRPC message can carry, so they are uploaded
# as cache artifacts at the threshold, whichever path encoded them.

@pytest.mark.parametrize("path", ["local_rows", "pandas_rows", "float64", "strings"])
def test_large_local_relations_follow_the_cache_threshold(protocol, monkeypatch, path):
    # (The mock service keeps gRPC's default 4 MiB message limit, so the inline variant of
    # these inputs, below a higher threshold, can only be checked against a real server.)
    import numpy as np

    if path == "local_rows":
        data = [("y" * 2000, i) for i in range(20_000)]
        schema = T.StructType([T.StructField("v", T.StringType()), T.StructField("i", T.LongType())])
        expected_rows = 20_000
    elif path == "pandas_rows":
        # Without PyArrow, the rows of a pandas frame are converted.
        _without_pyarrow(monkeypatch)
        data = pd.DataFrame(np.ones((100_000, 45), dtype=np.float32),
                            columns=[f"c{i}" for i in range(45)])
        schema = T.StructType([T.StructField(c, T.DoubleType()) for c in data.columns])
        expected_rows = 100_000
    elif path == "float64":
        data = pd.DataFrame(np.ones((100_000, 45)), columns=[f"c{i}" for i in range(45)])
        schema = T.StructType([T.StructField(c, T.DoubleType()) for c in data.columns])
        expected_rows = 100_000
    else:
        data = pd.DataFrame({"v": ["x" * 3500] * 10_000}, dtype=object)
        schema = T.StructType([T.StructField("v", T.StringType())])
        expected_rows = 10_000
    overrides = {"spark.sql.session.localRelationCacheThreshold": "1"}
    with _arrow_input_service(protocol, **overrides) as (spark, state):
        df = spark.createDataFrame(data, schema)
        df.toArrow()
        assert sum(table.num_rows for table in state["input_tables"]) == expected_rows
        assert state["uploads"]
        assert any("spark.sql.session.localRelationCacheThreshold" in keys
                   for _, keys in state["config_requests"])


def test_small_local_relations_read_the_settings_and_stay_inline(protocol):
    # Below the cache threshold, a relation is sent inline, after the one read of the settings
    # that PySpark's createDataFrame makes before it converts any data.
    data = [("y" * 100, i) for i in range(500)]
    assert pa.Table.from_pylist([{"v": v, "i": i} for v, i in data]).nbytes < 64 << 10
    with _arrow_input_service(protocol) as (spark, state):
        schema = T.StructType([T.StructField("v", T.StringType()), T.StructField("i", T.LongType())])
        spark.createDataFrame(data, schema).toArrow()
        assert not state["uploads"]
        assert state["config_requests"][0] == ("get", _CREATE_DATAFRAME_KEYS)
        assert state["input_tables"][0].num_rows == 500


def test_small_local_relations_follow_the_cache_threshold(protocol):
    # As PySpark's, a relation at the cache threshold or above is uploaded at any size.
    data = [("y", i) for i in range(5)]
    with _arrow_input_service(
        protocol, **{"spark.sql.session.localRelationCacheThreshold": "0"}
    ) as (spark, state):
        schema = T.StructType(
            [T.StructField("v", T.StringType()), T.StructField("i", T.LongType())])
        spark.createDataFrame(data, schema).toArrow()
        assert state["uploads"]
        assert state["config_requests"][0] == ("get", _CREATE_DATAFRAME_KEYS)


# Signed-int64 pandas input uses the same Arrow importer and mock service.

@pytest.mark.parametrize("storage", ["object", "default", "python", "pyarrow", "python_nan", "pyarrow_nan"])
@pytest.mark.parametrize("dtype", ["int64", "Int64"])
def test_integer_canonical_columns_storage_stays_columnar(protocol, monkeypatch, storage, dtype):
    from pyspark.sql._pandas_input import _has_canonical_columns

    values = [2**53 + 1, -(2**63), 2**63 - 1, 0] * 250
    if dtype == "Int64":
        values[1] = None
    frame = pd.DataFrame({"original": pd.Series(values, dtype=dtype)})
    if storage == "object":
        frame.columns = pd.Index(["original"], dtype=object)
    elif storage != "default":
        kind = storage.removesuffix("_nan")
        label_dtype = (pd.StringDtype(storage=kind, na_value=np.nan)
                       if storage.endswith("_nan") else pd.StringDtype(storage=kind))
        frame.columns = pd.Index(pd.array(["original"], dtype=label_dtype))
    assert _has_canonical_columns(frame.columns)
    assert prepare_int64_table(frame, None).column(0).to_pylist() == values

    def no_rows(*args, **kwargs):
        raise AssertionError("ordinary columns must retain integer columnar construction")

    monkeypatch.setattr(pd.DataFrame, "astype", no_rows)
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame).toArrow()
        assert state["input_tables"][0].column(0).to_pylist() == values
        assert state["config_requests"]


@pytest.mark.parametrize("dtype", ["int64", "Int64"])
@pytest.mark.parametrize("schema_kind", ["inferred", "names", "explicit"])
@pytest.mark.parametrize("location", [
    "subclass", "class", "owner", "backing_subclass_item", "backing_class_iter",
    "backing_class_getitem", "backing_instance_box",
])
@pytest.mark.parametrize("behavior", ["raise", "mutate"])
def test_integer_columns_iterator_hooks_run_in_pyspark_conversion(
    protocol, monkeypatch, dtype, schema_kind, location, behavior
):
    # The fast path declines labels whose iteration is customized, without iterating them, and
    # the frame converts as PySpark's conversion does: it iterates the labels, for the names
    # without a struct schema and for the columns of `items()`, before it reads any value, so
    # the hook's error is raised, or its change of the values is converted. It never calls the
    # frame's notna or astype.
    import json

    from pandas.core.base import IndexOpsMixin

    values = [2**53 + 1, -(2**63), 2**63 - 1, 0] * 250
    if dtype == "Int64":
        values[1] = None
    frame = pd.DataFrame({"original": pd.Series(values, dtype=dtype)})
    schema = None
    expected_name = "original"
    nullable = True
    metadata = {}
    if schema_kind == "names":
        schema = ["renamed"]
        expected_name = "renamed"
    elif schema_kind == "explicit":
        nullable, metadata = dtype == "Int64", {"source": "native"}
        schema = T.StructType([T.StructField("renamed", T.LongType(), nullable, metadata)])
        expected_name = "renamed"
    original_iter = pd.Index.__iter__
    original_notna, original_astype = frame.notna, frame.astype
    calls = []
    invoked = False

    class ColumnsError(RuntimeError):
        pass

    def notna(*args, **kwargs):
        calls.append("notna")
        return original_notna(*args, **kwargs)

    def astype(*args, **kwargs):
        calls.append("astype")
        return original_astype(*args, **kwargs)

    def before_hook():
        nonlocal invoked
        if invoked:
            return
        invoked = True
        calls.append("columns iterator")
        if behavior == "raise":
            raise ColumnsError("columns hook precedes the conversion of the values")
        # The spies of the frame's normalization methods are installed once the conversion
        # iterates the labels, as the values that it reads next are changed: before, the fast
        # path's guards see the canonical methods.
        monkeypatch.setitem(vars(frame), "notna", notna)
        monkeypatch.setitem(vars(frame), "astype", astype)
        frame.iat[0, 0] = 2**53 + 7
        frame.iat[-1, 0] = None if dtype == "Int64" else -(2**63) + 1

    def iterate(index):
        if index is columns:
            before_hook()
        return original_iter(index)

    if location == "subclass":
        class Columns(pd.Index):
            pass
        monkeypatch.setattr(Columns, "__iter__", iterate)
        labels = pd.Index(["original"], dtype=object)
        frame.columns = Columns._simple_new(labels._values, name=None)
        assert type(frame.columns) is Columns
    elif location == "backing_subclass_item":
        # IndexOpsMixin iteration calls ndarray.item, rather than ndarray.__iter__.
        class Names(np.ndarray):
            def item(self, *args):
                before_hook()
                return super().item(*args)
        backing = np.asarray(["original"], dtype=object).view(Names)
        frame.columns = pd.Index._simple_new(backing, name=None)
        assert type(frame.columns._values) is Names
    elif location.startswith("backing_"):
        frame.columns = pd.Index(pd.array(["original"], dtype=pd.StringDtype(storage="python")))
        backing = frame.columns._values
        if location == "backing_class_iter":
            original = type(backing).__iter__

            def backing_iter(array):
                if array is backing:
                    before_hook()
                return original(array)

            monkeypatch.setattr(type(backing), "__iter__", backing_iter)
        elif location == "backing_class_getitem":
            original = type(backing).__getitem__

            def backing_getitem(array, key):
                if array is backing:
                    before_hook()
                return original(array, key)

            monkeypatch.setattr(type(backing), "__getitem__", backing_getitem)
        else:
            original = backing._box_func

            def box(value):
                before_hook()
                return original(value)

            monkeypatch.setitem(vars(backing), "_box_func", box)
    else:
        monkeypatch.setattr(pd.Index if location == "class" else IndexOpsMixin,
                            "__iter__", iterate)
    columns = frame.columns

    with _arrow_input_service(protocol) as (spark, state):
        if behavior == "raise":
            with pytest.raises(ColumnsError,
                               match="columns hook precedes the conversion of the values"):
                spark.createDataFrame(frame, schema)
            assert calls == ["columns iterator"]
            assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
            assert state["input_tables"] == []
        else:
            spark.createDataFrame(frame, schema).toArrow()
            expected = values.copy()
            expected[0], expected[-1] = 2**53 + 7, (None if dtype == "Int64" else -(2**63) + 1)
            table = state["input_tables"][0]
            assert table.num_rows == 1000
            assert table.column(0).type == pa.int64()
            assert table.column(0).to_pylist() == expected
            assert table.column(0).null_count == sum(value is None for value in expected)
            assert table.schema.field(0).metadata is None
            assert state["column_names"] == [expected_name]
            if schema_kind == "explicit":
                # PySpark's table takes the names and the nullability of the struct, which is
                # the schema of the relation.
                assert table.column_names == [expected_name]
                assert table.schema.field(0).nullable is nullable
                logical = json.loads(state["relations"][0].local_relation.schema)
                assert logical == {"type": "struct", "fields": [{
                    "name": expected_name, "type": "long", "nullable": nullable,
                    "metadata": metadata,
                }]}
            else:
                # Without a struct, PySpark's table is of the columns _0, _1, ..., sent without a
                # schema, which toDF names.
                assert table.column_names == ["_0"]
                assert table.schema.field(0).nullable
                assert state["spark_schema"] is None
            assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
            assert calls == ["columns iterator"]


def test_integer_rebound_index_values_raise_in_pyspark_conversion(protocol, monkeypatch):
    # The fast path declines labels whose storage is rebound without reading it. PySpark's
    # conversion reads it to iterate the labels, for the columns of `items()`, so the hook's
    # error is raised before any value is read.
    frame = pd.DataFrame({"original": np.arange(1000, dtype=np.int64)})
    schema = T.StructType([T.StructField("renamed", T.LongType())])
    columns = frame.columns
    original = pd.Index._values.fget
    calls = []

    def values(index):
        if index is columns:
            calls.append("_values")
            raise RuntimeError("columns backing hook precedes the conversion of the values")
        return original(index)

    monkeypatch.setattr(pd.Index, "_values", property(values))
    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(RuntimeError,
                           match="columns backing hook precedes the conversion of the values"):
            spark.createDataFrame(frame, schema)
        assert calls == ["_values"]
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
        assert state["input_tables"] == []


@pytest.mark.parametrize("schema_kind", ["inferred", "names", "explicit"])
@pytest.mark.parametrize("verify", [True, False])
def test_integer_columns_preserve_values_without_python_rows(protocol, monkeypatch, schema_kind, verify):
    frame = pd.DataFrame({
        "a": pd.Series([2**53 + 1, None, -(2**63)], dtype="Int64"),
        "b": np.array([2**63 - 1, -1, 0], dtype=np.int64),
    })
    schema = None
    names = ["a", "b"]
    if schema_kind == "names":
        schema = names = ["renamed_a", "renamed_b"]
    elif schema_kind == "explicit":
        schema = T.StructType([T.StructField("a", T.LongType()),
                               T.StructField("b", T.LongType(), False, {"source": "test"})])
    import pyspark.sql._arrow_input as helper
    original = helper.prepare_table
    observed = []

    def capture(*args):
        result = original(*args)
        observed.append(result)
        return result

    def no_rows(*args, **kwargs):
        raise AssertionError("supported pandas input must not materialize Python rows")

    monkeypatch.setattr(helper, "prepare_table", capture)
    monkeypatch.setattr(pd.DataFrame, "astype", no_rows)
    with _arrow_input_service(protocol) as (spark, state):
        df = spark.createDataFrame(frame, schema=schema, verifySchema=verify)
        # Every nonempty pandas input applies its names through toDF, including
        # explicit struct schemas, as PySpark's createDataFrame does.
        assert not df.isLocal()
        assert df.toArrow().column(0).to_pylist() == [3]
        assert state["column_names"] == names
        assert state["result"] == (3, 2**53 + 1, -(2**63))
    table, resolved, renamed = observed[0]
    assert table.to_pydict() == {"a": [2**53 + 1, None, -(2**63)],
                                "b": [2**63 - 1, -1, 0]}
    assert (renamed or resolved.names) == names
    if schema_kind == "explicit":
        assert resolved == schema
        assert not table.schema.field(1).nullable
        assert resolved.fields[1].metadata == {"source": "test"}


@pytest.mark.parametrize("dtype", ["int64", "Int64"])
def test_empty_integer_frame_requires_explicit_schema(protocol, dtype):
    frame = pd.DataFrame({"a": pd.Series([], dtype=dtype)})
    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(ValueError):
            spark.createDataFrame(frame)
        df = spark.createDataFrame(frame, T.StructType([T.StructField("a", T.LongType())]))
        assert df.isLocal()


@pytest.mark.parametrize("values", [
    [1.0, float("nan")], [True, False], ["x", None],
    [datetime(2024, 1, 1), None],
    [datetime(2024, 1, 1, tzinfo=timezone.utc), None],
    [date(2024, 1, 1), None], [Decimal("1.20"), None],
    [[1, 2], None], [{"a": 1}, None],
])
def test_other_dtypes_keep_row_fallback(values):
    frame = pd.DataFrame({"a": values})
    assert prepare_int64_table(frame, None) is None
    assert prepare_int64_table(frame, T.StructType([T.StructField("a", T.LongType())])) is None


@pytest.mark.parametrize("schema", [
    "a long", T.LongType(), ["a", "b"], [1],
    T.StructType([T.StructField("a", T.IntegerType())]),
    T.StructType([T.StructField("a", T.DoubleType())]),
    T.StructType([]),
])
def test_schema_conversions_keep_row_fallback(schema):
    assert prepare_int64_table(pd.DataFrame({"a": [1, 2]}), schema) is None


def test_duplicate_names_fallback_and_nullable_masks():
    frame = pd.DataFrame([[2**53 + 1, 2**53 + 3]], columns=["x", "x"], dtype="int64")
    assert prepare_int64_table(frame, None) is None
    assert prepare_int64_table(pd.DataFrame({0: [1]}), None) is None
    frame = pd.DataFrame({"a": pd.Series([None, None], dtype="Int64")})
    assert prepare_int64_table(frame, None).column(0).to_pylist() == [None, None]


@pytest.mark.parametrize("dtype", ["int32", "uint64", "Int32", "UInt64", "int64[pyarrow]"])
def test_other_integer_dtypes_keep_row_fallback(dtype):
    assert prepare_int64_table(pd.DataFrame({"a": pd.Series([1, 2], dtype=dtype)}), None) is None


@pytest.mark.parametrize("rows", [100, 1000, 100000, 1000000])
@pytest.mark.parametrize("explicit", [False, True])
def test_integer_scales_reach_mock_service(protocol, monkeypatch, rows, explicit):
    values = np.arange(rows, dtype=np.int64) + (2**53 + 1)
    frame = pd.DataFrame({name: values for name in ("a", "b", "c")})
    schema = T.StructType([T.StructField(name, T.LongType()) for name in frame.columns])

    def no_rows(*args, **kwargs):
        raise AssertionError("integer input must stay columnar at every scale")

    if rows >= 1000:
        monkeypatch.setattr(pd.DataFrame, "astype", no_rows)
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema if explicit else None).toArrow()
        assert state["result"] == (rows, 2**53 + 1, 2**53 + rows)
        if frame.memory_usage(index=False).sum() > 1048576:
            assert state["relations"][0].HasField("chunked_cached_local_relation")


def test_nullable_inference_does_not_depend_on_first_value(protocol):
    frame = pd.DataFrame({"a": pd.Series([None, 2**53 + 1], dtype="Int64")})
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame).toArrow()
        assert state["result"] == (2, None, 2**53 + 1)


@pytest.mark.parametrize("verify", [True, False])
@pytest.mark.parametrize("config_error", [False, True])
def test_non_nullable_schema_is_preserved(protocol, verify, config_error):
    frame = pd.DataFrame({"a": pd.Series([1, None], dtype="Int64")})
    schema = T.StructType([T.StructField("a", T.LongType(), False)])
    assert prepare_int64_table(frame, schema) is None
    with _arrow_input_service(protocol) as (spark, state):
        if config_error:
            state["config_error"] = grpc.StatusCode.INVALID_ARGUMENT
            with pytest.raises(Exception, match="configuration read failed"):
                spark.createDataFrame(frame, schema, verifySchema=verify)
        else:
            # Official reads the configuration first, then the cast fails with a ValueError.
            with pytest.raises(ValueError) as error:
                spark.createDataFrame(frame, schema, verifySchema=verify)
            assert str(error.value) == "Casting field 'a' with null values to non-nullable"
        assert state["config_requests"][0] == ("get", _CREATE_DATAFRAME_KEYS)
        assert not state["relations"]
        assert not state["uploads"]


def test_small_numpy_frames_avoid_importer_configuration_overhead():
    assert prepare_int64_table(pd.DataFrame({"a": np.arange(999, dtype=np.int64)}), None) is None
    assert prepare_int64_table(pd.DataFrame({"a": np.arange(1000, dtype=np.int64)}), None) is not None
    assert prepare_int64_table(pd.DataFrame({"a": pd.Series([None], dtype="Int64")}), None) is not None


@pytest.mark.parametrize("kind", ["struct", "field", "both", "mixin"])
def test_struct_subclasses_take_pysparks_conversion(protocol, kind):
    # As PySpark's createDataFrame, which converts a frame under any StructType of StructFields
    # with its conversion: of a Python subclass of either that overrides none of the reads of the
    # conversion too, with its errors, such as that of a lossy cast, where the row path raised
    # its own ("Type mismatch in row data").
    from pyspark.errors import PySparkValueError
    from pyspark.sql._pandas_input import accepts_pandas_schema

    class Struct(T.StructType):
        pass

    class Field(T.StructField):
        pass

    class Mixin:
        def extra(self):
            return 1

    class Mixed(Mixin, T.StructType):
        pass

    struct_class = {"struct": Struct, "field": T.StructType, "both": Struct, "mixin": Mixed}[kind]
    field_class = Field if kind in ("field", "both") else T.StructField
    schema = struct_class([field_class("i", T.LongType())])
    assert accepts_pandas_schema(schema)
    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(PySparkValueError) as error:
            spark.createDataFrame(pd.DataFrame({"i": [1.5]}), schema)
        assert str(error.value).startswith(
            "Exception thrown when converting pandas.Series (float64) with name 'i' to Arrow "
            "Array (int64)."
        )
        spark.createDataFrame(pd.DataFrame({"i": [1]}), schema).toArrow()
        assert state["input_tables"][0].to_pydict() == {"i": [1]}


@pytest.mark.parametrize("read", [
    "fields", "names", "__iter__", "__getattribute__", "name", "dataType", "nullable",
    "metadata", "field_getattribute",
])
def test_struct_subclasses_that_override_the_reads_of_the_conversion_keep_the_row_path(read):
    # The conversion reads the fields, names and types of a struct through its classes, and the
    # importer reads the native types: a subclass that overrides one of these reads, whose
    # values the two would read differently, keeps the row path, which reads the native types.
    from pyspark.sql._pandas_input import accepts_pandas_schema

    def override(name):
        if name in ("__iter__", "__getattribute__", "field_getattribute"):
            method = {"__iter__": lambda self: iter([]),
                      "__getattribute__": object.__getattribute__,
                      "field_getattribute": object.__getattribute__}[name]
            return {"__getattribute__" if name == "field_getattribute" else name: method}
        return {name: property(lambda self: None)}

    struct_reads = ("fields", "names", "__iter__", "__getattribute__")
    struct_class = type("Struct", (T.StructType,), override(read) if read in struct_reads else {})
    field_class = (
        T.StructField if read in struct_reads else type("Field", (T.StructField,), override(read))
    )
    schema = struct_class([field_class("i", T.LongType())])
    assert not accepts_pandas_schema(schema)
    assert accepts_pandas_schema(T.StructType([T.StructField("i", T.LongType())]))


@pytest.mark.parametrize("schema_kind", ["inferred", "names", "explicit"])
def test_nul_names_take_the_integer_path_whole(protocol, schema_kind):
    name = "a\x00b"
    frame = pd.DataFrame({name if schema_kind == "inferred" else "original":
                          np.arange(1000, dtype=np.int64)})
    schema = None
    if schema_kind == "names":
        schema = [name]
    elif schema_kind == "explicit":
        schema = T.StructType([T.StructField(name, T.LongType())])
    # A name with a NUL takes the fast path as any other name. The C data interface holds the
    # names as C strings, which end at a NUL, so the importer reads such a table of its IPC
    # stream, which keeps them whole, as PySpark's.
    table = prepare_int64_table(frame, schema)
    assert table.column_names == ["original" if schema_kind == "names" else name]
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert state["result"] == (1000, 0, 999)
        payload = state["relations"][0].local_relation.data
        assert pa.ipc.open_stream(payload).schema.names == table.column_names
        assert state["column_names"] == [name]


@pytest.mark.parametrize(
    "kind", ["float64", "string", "date", "timestamp", "conversion", "nested", "nested_inferred"]
)
def test_nul_names_take_the_arrow_paths_whole(protocol, kind, monkeypatch):
    # As PySpark's createDataFrame, which sends the IPC stream of its table, a name with a NUL,
    # at any depth, reaches the relation whole, from each fast path and from PySpark's
    # conversion, which the importer reads of the IPC stream of the table. The fast path of
    # naive timestamps is taken on a UTC host only (`_timestamp_host_is_utc`).
    import time

    import pyspark.sql._pandas_input as helper

    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.setattr(time, "tzname", ("UTC", "UTC"))

    name = "x\x00y"
    data_type = {
        "float64": T.DoubleType(), "string": T.StringType(), "date": T.DateType(),
        "timestamp": T.TimestampType(), "conversion": T.DecimalType(10, 2),
        "nested": T.StructType([T.StructField("k\x00", T.LongType())]),
    }.get(kind)
    rows = 100000 if kind == "float64" else 10000
    values = {
        "float64": np.full(rows, 1.5), "string": ["text"] * rows,
        "date": [date(2020, 1, 1)] * rows,
        "timestamp": pd.to_datetime(["2020-01-01 00:00:00"] * rows),
        "conversion": [Decimal("1.25")] * rows,
        "nested": [{"k\x00": 1}] * rows, "nested_inferred": [{"k\x00": 1}] * rows,
    }[kind]
    frame = pd.DataFrame({"original": values})
    if kind == "nested_inferred":
        frame.columns = [name]
        schema = None
    else:
        schema = T.StructType([T.StructField(name, data_type)])
    fast_path = {
        "float64": helper.prepare_float64_input, "string": helper.prepare_string_input,
        "date": helper.prepare_date_table, "timestamp": helper.timestamp_columns,
    }.get(kind)
    if fast_path is not None:
        assert fast_path(frame, schema) is not None
    with _arrow_input_service(protocol) as (spark, state):
        result = spark.createDataFrame(frame, schema)
        result.toArrow()
        payload = pa.ipc.open_stream(state["relations"][0].local_relation.data)
        if kind == "nested_inferred":
            # As PySpark Connect's, the Arrow conversion of a frame without a schema names its
            # columns _0, _1, ..., which toDF names.
            assert payload.schema.names == ["_0"]
            assert payload.schema.field(0).type == pa.struct([pa.field("k\x00", pa.int64())])
        else:
            assert payload.schema.names == [name]
        if kind.startswith("nested"):
            assert payload.schema.field(0).type.field(0).name == "k\x00"
        assert state["column_names"] == [name]


@pytest.mark.parametrize("kind", ["labels", "labels_int64", "names", "names_int64", "arrow"])
def test_names_that_utf8_cannot_encode_fail_the_requests_as_pyspark(protocol, kind):
    # As PySpark's createDataFrame, which applies the labels of a pandas DataFrame, a list of
    # names of one, and the names of a pyarrow Table with toDF, whose plan sets them when a
    # request builds it: a name that UTF-8 cannot encode fails the requests of the DataFrame with
    # protobuf's UnicodeEncodeError, and not the call, also of a frame of the integer path. Of
    # local rows, which name the table of PySpark's conversion, it fails the call.
    bad = "a\ud800b"
    message = (
        "'utf-8' codec can't encode character '\\ud800' in position 1: surrogates not allowed"
    )
    values = np.arange(1000 if kind.endswith("int64") else 2, dtype=np.int64)

    def labeled(values):
        # Of an object Index: the str Index of pandas 3, of Arrow, cannot hold such a label.
        return pd.DataFrame({"a": values}).set_axis(pd.Index([bad], dtype=object), axis=1)

    if kind.startswith("labels"):
        data, schema = labeled(values), None
    elif kind.startswith("names"):
        data, schema = pd.DataFrame({"a": values}), [bad]
    else:
        data, schema = pa.table({"a": values}), [bad]
    if kind.endswith("int64"):
        assert prepare_int64_table(data, schema) is None or kind == "names_int64"
    with _arrow_input_service(protocol) as (spark, state):
        df = spark.createDataFrame(data, schema)
        for request in (lambda: df.schema, df.collect, lambda: df.select("*").columns):
            with pytest.raises(UnicodeEncodeError) as error:
                request()
            assert str(error.value) == message
        assert state["relations"] == []
        with pytest.raises(UnicodeEncodeError) as error:
            spark.createDataFrame([(1,)], [bad])
        assert str(error.value) == message
        # The labels name no column of a frame of a list of names or of a struct.
        frame = labeled(values)
        for schema in (["x"], T.StructType([T.StructField("x", T.LongType())])):
            spark.createDataFrame(frame, schema).toArrow()
            assert state["column_names"] == ["x"]


@pytest.mark.parametrize("nullable", [False, True])
@pytest.mark.parametrize("rows", [999, 1000])
def test_integer_explicit_renamed_schema_preserves_all_ipc_cells(protocol, monkeypatch, nullable, rows):
    values = ([2**53 + 1, -(2**63), 2**63 - 1, 0] * 250)[:rows]
    other = [None if nullable and i % 7 == 0 else value
             for i, value in enumerate(reversed(values))]
    frame = pd.DataFrame({"original": np.asarray(values, dtype="int64"),
                          "second": pd.Series(other, dtype="Int64" if nullable else "int64")})
    schema = T.StructType([
        T.StructField("renamed", T.LongType(), False, {"source": "native", "id": 3}),
        T.StructField("nullable", T.LongType(), True, {"tag": [1, 2]}),
    ])

    def no_rows(*args, **kwargs):
        raise AssertionError("trusted integer columns must stay columnar")

    if rows >= 1000 or nullable:
        monkeypatch.setattr(pd.DataFrame, "astype", no_rows)
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        table = state["input_tables"][0]
        assert table.column_names == ["renamed", "nullable"]
        assert [column.to_pylist() for column in table.columns] == [values, other]
        assert table.schema.field(0).type == pa.int64()
        assert table.schema.field(0).nullable is False
        assert table.schema.field(1).nullable is True
        if rows >= 1000 or nullable:
            from pyspark.sql.pandas.types import from_arrow_schema

            # As PySpark's conversion of the frame, of `_deduplicate_field_names`, the Arrow
            # schema has no metadata of the fields, which the schema of the relation has.
            plain = T.StructType([T.StructField(f.name, f.dataType, f.nullable) for f in schema])
            assert from_arrow_schema(table.schema).json() == plain.json(), (
                from_arrow_schema(table.schema).json(), plain.json(), table.schema)
            assert state["config_requests"]
        else:
            assert state["config_requests"][:1] == [("get", _CREATE_DATAFRAME_KEYS)]


def test_integer_dataframe_and_name_subclasses_keep_fallback(monkeypatch):
    class Frame(pd.DataFrame):
        pass

    class Name(str):
        pass

    class Names(list):
        pass

    frame = pd.DataFrame({"a": np.arange(1000, dtype="int64")})
    assert prepare_int64_table(Frame(frame), None) is None
    assert prepare_int64_table(frame, Names(["a"])) is None
    assert prepare_int64_table(frame, [Name("a")]) is None
    # Pandas 3 normalizes constructor labels to plain strings by default.
    # An object Index keeps the subclass this fallback regression exercises.
    named_frame = frame.copy()
    named_frame.columns = pd.Index([Name("a")], dtype=object)
    assert type(named_frame.columns[0]) is Name
    assert prepare_int64_table(named_frame, None) is None
    monkeypatch.setattr(pd, "DataFrame", Frame)
    assert prepare_int64_table(Frame(frame), None) is None


def test_integer_missing_internal_pandas_array_class_keeps_fallback():
    import sys

    script = r'''
import sys
import types
import numpy as np
import pandas as pd
from pyspark.sql._pandas_input import prepare_int64_table
frame = pd.DataFrame({"a": np.arange(1000, dtype="int64")})
nullable = pd.DataFrame({"a": pd.Series([None, 1], dtype="Int64")})
# Older pandas exposes PandasArray rather than NumpyExtensionArray. Missing
# optional internals must disable the optimization without breaking pandas input.
sys.modules["pandas.core.arrays.numpy_"] = types.ModuleType("pandas.core.arrays.numpy_")
assert prepare_int64_table(frame, None) is None
assert prepare_int64_table(nullable, None) is None
'''
    result = subprocess.run([sys.executable, "-c", script],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("class_spoof", [False, True])
def test_integer_custom_extension_dtype_takes_pyspark_conversion(protocol, class_spoof):
    # An extension array of a dtype that equals int64, even of a class that claims to be a NumPy
    # dtype, does not select the fast path of int64 columns: as PySpark's conversion, which
    # hands a column of an array with `__arrow_array__` to Arrow as it is, the values are those
    # that the hook gives, not those of the array.
    from pandas.api.extensions import ExtensionArray, ExtensionDtype, take

    arrow_calls = []

    class Int64EqualDtype(ExtensionDtype):
        name = "int64_equal"
        type = int
        kind = "i"
        na_value = None

        @classmethod
        def construct_array_type(cls):
            return DifferentArrowArray

        def __eq__(self, other):
            return (isinstance(other, Int64EqualDtype)
                    or isinstance(other, np.dtype) and other == np.dtype("int64"))

        def __hash__(self):
            return hash(self.name)

    class DifferentArrowArray(ExtensionArray):
        def __init__(self, values):
            self._values = np.asarray(values, dtype=object)

        @property
        def dtype(self):
            return Int64EqualDtype()

        @property
        def nbytes(self):
            return self._values.nbytes

        def __len__(self):
            return len(self._values)

        def __getitem__(self, key):
            result = self._values[key]
            return type(self)(result) if isinstance(result, np.ndarray) else result

        def isna(self):
            return pd.isna(self._values)

        def take(self, indices, allow_fill=False, fill_value=None):
            return type(self)(take(self._values, indices, allow_fill=allow_fill,
                                   fill_value=fill_value))

        def copy(self):
            return type(self)(self._values.copy())

        @classmethod
        def _from_sequence(cls, scalars, *, dtype=None, copy=False):
            values = np.asarray(scalars, dtype=object)
            return cls(values.copy() if copy else values)

        def to_numpy(self, dtype=None, copy=False, na_value=None):
            if dtype is not None:
                return self._values.astype(dtype, copy=copy)
            return self._values.copy() if copy else self._values

        def __arrow_array__(self, type=None):
            arrow_calls.append("arrow")
            return pa.array([-7] * len(self), type=type)

    values = [2**53 + 1, None, -(2**63), 0] * 250
    frame = pd.DataFrame({"a": pd.Series(DifferentArrowArray(values), copy=False)})
    if class_spoof:
        Int64EqualDtype.__getattribute__ = lambda self, name: (
            type(np.dtype("int64")) if name == "__class__"
            else object.__getattribute__(self, name))
    assert frame.dtypes.iloc[0] == np.dtype("int64")
    assert isinstance(frame.dtypes.iloc[0], np.dtype) is class_spoof
    schema = T.StructType([T.StructField("renamed", T.LongType())])
    assert prepare_int64_table(frame, schema) is None
    assert arrow_calls == []
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
        assert state["input_tables"][0].column(0).to_pylist() == [-7] * len(values)
    assert arrow_calls == ["arrow"]


@pytest.mark.parametrize("custom", ["dtype", "array"])
def test_integer_nullable_dtype_and_array_subclasses_take_pyspark_conversion(protocol, custom):
    # A subclass of pandas' nullable integer array or dtype does not select the fast path of
    # Int64 columns: as PySpark's conversion, which hands the column to Arrow as it is, the
    # values are those of the subclass's `__arrow_array__`.
    from pandas.core.arrays.integer import Int64Dtype, IntegerArray

    arrow_calls = []

    class CustomDtype(Int64Dtype):
        pass

    class CustomArray(IntegerArray):
        @property
        def dtype(self):
            if custom == "dtype":
                return CustomDtype()
            return Int64Dtype()

        def __arrow_array__(self, type=None):
            arrow_calls.append("arrow")
            return pa.array([-7] * len(self), type=type)

    values = [2**53 + 1, 0, -(2**63), 7] * 250
    array = CustomArray(np.asarray(values, dtype="int64"), np.zeros(1000, dtype=bool))
    frame = pd.DataFrame({"a": pd.Series(array, copy=False)})
    assert type(frame["a"].array) is CustomArray
    schema = T.StructType([T.StructField("a", T.LongType())])
    assert prepare_int64_table(frame, schema) is None
    assert arrow_calls == []
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
        assert state["input_tables"][0].column(0).to_pylist() == [-7] * len(values)
    assert arrow_calls == ["arrow"]


# The tail of the scripts of the guard tests below, which run in a process of their own, as they
# patch classes and rebind names: of the frame `frame` and the schema `schema` that the script
# made, of the classes and the names as it left them, it sends the plan at the mock service as
# `_to_table` does, of `createDataFrame(frame, schema)`, and prints the outcome as JSON: the error,
# with the call that raised it, the settings read, the Arrow table of the relation and the
# relation's schema. The tests compare it with PySpark's, which `_guard_raises` and
# `_guard_converts` give.
_GUARD_OUTCOME = r'''
import json as _json
import tempfile as _tempfile
from pathlib import Path as _Path

with _tempfile.TemporaryDirectory(prefix="arrow-handoff-") as _directory:
    class _Factory:
        def mktemp(self, name):
            path = _Path(_directory) / name
            path.mkdir()
            return path
    _messages = capture.protocol.__wrapped__(_Factory())
_outcome = {"error": None, "configs": None, "table": None, "relation_schema": None}
with capture._arrow_input_service(_messages) as (_spark, _state):
    _stage = "createDataFrame"
    try:
        _df = _spark.createDataFrame(frame, schema)
        # toArrow converts the schema to Arrow first, with the classes as they are.
        _stage = "_to_table"
        _df._to_table()
    except Exception as _e:
        _outcome["error"] = [_stage, type(_e).__module__ + "." + type(_e).__qualname__, str(_e)]
    _outcome["configs"] = [[operation, keys] for operation, keys in _state["config_requests"]]
    if _state["input_tables"]:
        _pa = capture.pa
        _table = _state["input_tables"][0]
        _outcome["table"] = {
            "names": _table.schema.names,
            "types": [str(field.type) for field in _table.schema],
            "nullable": [field.nullable for field in _table.schema],
            "metadata": [
                None if field.metadata is None
                else {key.decode(): value.decode() for key, value in field.metadata.items()}
                for field in _table.schema
            ],
            # Of a timestamp, its integer, which does not depend on the class of its time zone.
            "values": [
                repr((column.cast(_pa.int64()) if _pa.types.is_timestamp(column.type)
                      else column).to_pylist())
                for column in _table.columns
            ],
        }
        _schema_text = _state.get("spark_schema")
        _outcome["relation_schema"] = None if _schema_text is None else _json.loads(_schema_text)
print(_json.dumps(_outcome))
'''


def _guard_outcome(script, *args, timeout=60):
    """The outcome of a guard script (`_GUARD_OUTCOME`), run in a process of its own with `args`
    and the path of this module, which it loads as `capture`, last."""
    import json
    import sys

    result = subprocess.run([sys.executable, "-c", script + _GUARD_OUTCOME, *args, __file__],
                            capture_output=True, text=True, timeout=timeout)
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def _relation_of(name, data_type, nullable, metadata):
    """The JSON of the relation's schema of a struct of one field, as `json()` writes it."""
    return {"fields": [{"metadata": metadata, "name": name, "nullable": nullable,
                        "type": data_type}], "type": "struct"}


def _guard_raises(error, message, *, configs=True, stage="createDataFrame"):
    """The outcome of a guard script whose `stage` raises `error` (of its module) of `message`:
    after the settings that createDataFrame reads in one request, as PySpark's reads them before
    it converts the frame, or, of `configs=False`, before it reads any, of no relation."""
    return {
        "error": [stage, error, message],
        "configs": [["get", _CREATE_DATAFRAME_KEYS]] if configs else [],
        "table": None,
        "relation_schema": None,
    }


def _guard_converts(columns, relation_schema, error=None):
    """The outcome of a guard script whose createDataFrame sends a relation of `relation_schema`
    (its JSON) of a table of `columns`, `(name, arrow_type, nullable, values)` or `(name,
    arrow_type, nullable, values, metadata)` of each, after the settings that it reads in one
    request: of `error` (`[stage, type, message]`) of `_to_table`, if any."""
    columns = [column + (None,) * (5 - len(column)) for column in columns]
    return {
        "error": error,
        "configs": [["get", _CREATE_DATAFRAME_KEYS]],
        "table": {
            "names": [column[0] for column in columns],
            "types": [column[1] for column in columns],
            "nullable": [column[2] for column in columns],
            "metadata": [column[4] for column in columns],
            "values": [repr(column[3]) for column in columns],
        },
        "relation_schema": relation_schema,
    }


def _conversion_error(error, dtype, name, arrow_type):
    """PySpark's error of the conversion of a pandas.Series `name` of `dtype` to an Arrow array of
    `arrow_type` (`create_arrow_array_from_pandas`), of a safe cast: its PySparkValueError names the
    setting of the safe casts."""
    message = (f"Exception thrown when converting pandas.Series ({dtype}) with name '{name}' to "
               f"Arrow Array ({arrow_type}).")
    if error == "PySparkValueError":
        message += (" It can be caused by overflows or other unsafe conversions warned by Arrow. "
                    "Arrow safe type check can be disabled by using SQL config "
                    "`spark.sql.execution.pandas.convertToArrowArraySafely`.")
    return ["pyspark.errors.exceptions.base." + error, message]


def _unsupported_type(type_repr):
    """PySpark's error of a type that `to_arrow_type` has no Arrow type of."""
    return ["pyspark.errors.exceptions.base.PySparkTypeError",
            f"[UNSUPPORTED_DATA_TYPE_FOR_ARROW_CONVERSION] {type_repr} is not supported in "
            "conversion to Arrow."]


def _no_setter(class_name):
    """The AttributeError of an assignment of PySpark's `__init__` to a property of the class
    `class_name` that has no setter."""
    return ["builtins.AttributeError", f"property of '{class_name}' object has no setter"]


def _rename_mismatch(name):
    """PyArrow's error of the cast of PySpark's table of the column `name` to an Arrow schema of
    the field `changed`."""
    return ["builtins.ValueError", "Target schema's field names are not matching the table's "
            f"field names: ['{name}'], ['changed']"]


def _assert_guard(outcome, pyspark, known_difference=None):
    """Assert that the outcome of a guard script is PySpark's, `pyspark`, as the reference client
    gives it, or, of a known difference (see "Patched classes and rebound names" of the pandas
    DataFrames in `dev/parity/known-differences.md`), the drop-in's, `known_difference`."""
    expected = pyspark if known_difference is None else known_difference
    assert outcome == expected, {"outcome": outcome, "pyspark": pyspark}


def _integer_guard_process(mode, argument, before_import=False):
    """The outcome of `createDataFrame` of a frame of int64s under `renamed long` of a class patched
    or a name rebound by `mode` and `argument`, in a process of its own (`_guard_outcome`).

    The patches and the rebound names change what PySpark's conversion reads: the session
    converts the frame with that conversion, of the conversions and the types that its module of
    the session binds when it is imported, as PySpark's does. `before_import` imports the helper
    module of the conversions anew, which changes nothing.
    """
    if mode == "captured_alias":
        import pyspark.sql._arrow_input as importer
        if not hasattr(importer, "to_arrow_schema"):
            pytest.skip("this importer resolves converters locally and holds no captured aliases")

    script = r'''
import importlib
import importlib.util
import sys
import numpy as np
import pandas as pd
from pyspark.sql import types as T
from pyspark import _pyspark as native

spec = importlib.util.spec_from_file_location("integer_guard_capture", sys.argv[-1])
capture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capture
spec.loader.exec_module(capture)
values = [2**53 + 1, -(2**63), 2**63 - 1, 0] * 250
frame = pd.DataFrame({"renamed": np.asarray(values, dtype="int64")})
schema = T.StructType([T.StructField("renamed", T.LongType(), False, {"source": "native"})])
mode, argument, before_import = sys.argv[1:4]
if before_import == "True":
    sys.modules.pop("pyspark.sql._pandas_input", None)
if mode == "descriptor":
    class_name, attribute = argument.split(":")
    cls = getattr(native, class_name)
    if attribute == "alias":
        setattr(T, class_name, object)
    elif attribute == "__getattribute__":
        setattr(cls, attribute, lambda self, name: object.__getattribute__(self, name))
    elif attribute == "__class__":
        setattr(cls, "__getattribute__", lambda self, name: (
            object if name == "__class__" else object.__getattribute__(self, name)))
    elif attribute == "__iter__":
        setattr(cls, attribute, lambda self: iter([T.StructField("changed", T.BooleanType())]))
    else:
        value = {"names": ["changed"], "fields": [], "name": "changed",
                 "dataType": T.BooleanType(), "nullable": True,
                 "metadata": {"changed": True}}[attribute]
        setattr(cls, attribute, property(lambda self: value))
elif mode in ("public_alias", "converter_alias", "captured_alias"):
    replacement = native.BooleanType if argument == "LongType" else native.LongType
    if mode == "public_alias":
        sys.modules.pop("pyspark.sql.pandas.types", None)
        assert "pyspark.sql.pandas.types" not in sys.modules
        setattr(T, argument, replacement)
    else:
        converter = importlib.import_module("pyspark.sql.pandas.types")
        if mode == "captured_alias":
            importer = importlib.import_module("pyspark.sql._arrow_input")
        setattr(converter, argument, replacement)
        if mode == "captured_alias":
            sys.modules.pop("pyspark.sql.pandas.types")
            current = importlib.import_module("pyspark.sql.pandas.types")
            assert current is not converter
            assert importer.to_arrow_schema.__globals__ is vars(converter)
elif mode == "atomic_alias":
    scope, replacement_name = argument.split(":")
    replacement = None if replacement_name == "none" else native.LongType
    if scope == "public_before":
        sys.modules.pop("pyspark.sql._arrow_input", None)
        T.AtomicType = replacement
    elif scope == "public_after":
        importlib.import_module("pyspark.sql._arrow_input")
        T.AtomicType = replacement
    else:
        importer = importlib.import_module("pyspark.sql._arrow_input")
        if replacement_name == "missing":
            del importer.AtomicType
        else:
            importer.AtomicType = replacement
elif mode == "importer_alias":
    importer = importlib.import_module("pyspark.sql._arrow_input")
    importer.StructType = None if argument == "none" else native.LongType
else:
    raise AssertionError(mode)
'''
    return _guard_outcome(script, mode, argument, str(before_import))


_INTEGER_VALUES = [2**53 + 1, -(2**63), 2**63 - 1, 0] * 250
_INTEGER_RELATION = {"fields": [{"metadata": {"source": "native"}, "name": "renamed",
                                 "nullable": False, "type": "long"}], "type": "struct"}


def _integer_converts(values=_INTEGER_VALUES, arrow_type="int64", name="renamed", nullable=False,
                      metadata=None, error=None, relation=_INTEGER_RELATION):
    return _guard_converts([(name, arrow_type, nullable, values, metadata)], relation, error)


# The outcomes of PySpark's createDataFrame of the patches of `_integer_guard_process`, as the
# reference client gives them: of each patch of a property of no setter, the AttributeError of
# PySpark's `__init__` of a field or a struct, which `_deduplicate_field_names` makes, and of a
# patched iteration, the error of the cast of the table to the Arrow schema of what it iterates.
# A `StructType` rebound before the session is the type of every response of the results of
# PySpark's client, whose module binds it with the session, so its `_to_table` finds no table.
_INTEGER_DESCRIPTORS = {
    ("StructType", "fields"): _guard_raises(*_no_setter("StructType")),
    ("StructType", "names"): _guard_raises(*_no_setter("StructType")),
    ("StructType", "__iter__"): _guard_raises(*_rename_mismatch("renamed")),
    ("StructField", "name"): _guard_raises(*_no_setter("StructField")),
    ("StructField", "dataType"): _guard_raises(*_no_setter("StructField")),
    ("StructField", "nullable"): _guard_raises(*_no_setter("StructField")),
    ("StructField", "metadata"): _guard_raises(*_no_setter("StructField")),
    ("StructType", "alias"): _integer_converts(
        error=["_to_table", "builtins.AssertionError", ""]),
}
# The known differences: the native `__init__` does not assign through a property patched over
# its attributes (SPARK-62231, SPARK-62251), so the conversion goes on and reads the patch where
# PySpark's reads the attribute, and so does the relation's schema, PySpark's `json()` of it; and
# the drop-in's client is native.
_INTEGER_DESCRIPTORS_OF_THE_DROP_IN = {
    ("StructType", "fields"): _guard_converts([], {"fields": [], "type": "struct"}),
    ("StructType", "names"): _integer_converts(name="changed"),
    ("StructField", "name"): _guard_raises(*_rename_mismatch("renamed")),
    ("StructField", "dataType"): _integer_converts(
        [value != 0 for value in _INTEGER_VALUES], "bool",
        relation=_relation_of("renamed", "boolean", False, {"source": "native"})),
    ("StructField", "nullable"): _integer_converts(
        nullable=True, relation=_relation_of("renamed", "long", True, {"source": "native"})),
    ("StructField", "metadata"): _integer_converts(
        metadata={"SPARK::metadata::json": '{"changed": true}'},
        relation=_relation_of("renamed", "long", False, {"changed": True})),
    ("StructType", "alias"): _integer_converts(),
}


@pytest.mark.parametrize("class_name,attribute", [
    ("StructType", "fields"), ("StructType", "names"), ("StructType", "__iter__"),
    ("StructField", "name"), ("StructField", "dataType"), ("StructField", "nullable"),
    ("StructField", "metadata"), ("StructType", "__getattribute__"),
    ("StructField", "__getattribute__"), ("LongType", "__getattribute__"),
    ("StructType", "__class__"), ("StructField", "__class__"), ("LongType", "__class__"),
    ("StructType", "alias"), ("StructField", "alias"), ("LongType", "alias"),
])
@pytest.mark.parametrize("before_import", [False, True])
def test_integer_custom_schema_descriptors_take_pyspark_conversion(class_name, attribute,
                                                                   before_import):
    outcome = _integer_guard_process("descriptor", f"{class_name}:{attribute}", before_import)
    _assert_guard(outcome, _INTEGER_DESCRIPTORS.get((class_name, attribute), _integer_converts()),
                  _INTEGER_DESCRIPTORS_OF_THE_DROP_IN.get((class_name, attribute)))


@pytest.mark.parametrize("mode", ["public_atomic_none", "session_struct_subclass",
                                  "session_struct_deleted"])
def test_create_dataframe_checks_its_schema_against_the_names_of_the_session_module(mode):
    # As PySpark's createDataFrame, which checks its schema, of a DDL string as the server parses
    # it, against the `AtomicType` and the `StructType` of its module of the session, which binds
    # them with the first session, of any data and before it reads any setting: of a public
    # `AtomicType` of None, rebound before the session, `isinstance` raises; of a `StructType` of
    # that module rebound to a subclass, a struct of the class itself is not expected; and of one
    # removed from it, the name is not defined. In a process of its own, as it rebinds the names.
    import json
    import sys

    script = r'''
import importlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
from pyspark.sql import types as T

spec = importlib.util.spec_from_file_location("schema_check_capture", sys.argv[-1])
capture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capture
spec.loader.exec_module(capture)
mode = sys.argv[1]


class Sub(T.StructType):
    pass


if mode == "public_atomic_none":
    T.AtomicType = None
with tempfile.TemporaryDirectory(prefix="arrow-handoff-") as directory:
    class Factory:
        def mktemp(self, name):
            path = Path(directory) / name
            path.mkdir()
            return path
    messages = capture.protocol.__wrapped__(Factory())
outcomes = {}
with capture._arrow_input_service(messages) as (spark, state):
    session_module = importlib.import_module("pyspark.sql.connect.session")
    if mode == "session_struct_subclass":
        session_module.StructType = Sub
    elif mode == "session_struct_deleted":
        del session_module.StructType
    plain = T.StructType([T.StructField("a", T.LongType())])
    cases = {
        "rows_struct": lambda: spark.createDataFrame([(1,)], plain),
        "rows_ddl": lambda: spark.createDataFrame([(1,)], "a long"),
        "arrow_none": lambda: spark.createDataFrame(pa.table({"a": [1]})),
        "arrow_names": lambda: spark.createDataFrame(pa.table({"a": [1]}), ["a"]),
        "empty_struct": lambda: spark.createDataFrame([], plain),
        "pandas_struct": lambda: spark.createDataFrame(pd.DataFrame({"a": [1]}), plain),
        "arrow_struct": lambda: spark.createDataFrame(pa.table({"a": [1]}), plain),
        "ndarray_bad_dimensions": lambda: spark.createDataFrame(np.zeros((1, 1, 1)), plain),
    }
    for name, make in cases.items():
        configs, ddl_strings = len(state["config_requests"]), len(state["ddl_strings"])
        try:
            make()
            error = None
        except Exception as e:
            error = [type(e).__module__ + "." + type(e).__qualname__, str(e)]
        outcomes[name] = [error, len(state["config_requests"]) - configs,
                          len(state["ddl_strings"]) - ddl_strings]
print(json.dumps(outcomes))
'''
    result = subprocess.run([sys.executable, "-c", script, mode, __file__],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    outcomes = json.loads(result.stdout.strip().splitlines()[-1])
    error = {
        "public_atomic_none": ["builtins.TypeError",
                               "isinstance() arg 2 must be a type, a tuple of types, or a union"],
        "session_struct_subclass": [
            "pyspark.errors.exceptions.base.PySparkTypeError",
            "[NOT_EXPECTED_TYPE] Argument `schema` should be list, None or StructType, got "
            "StructType."],
        "session_struct_deleted": ["builtins.NameError", "name 'StructType' is not defined"],
    }[mode]
    # The DDL string is parsed first, of one request, as PySpark parses it; no setting is read
    # before the error, nor of data whose conversion reads none (of no rows). Names, and no
    # schema, are what a `StructType` subclass takes too.
    expected = {name: [error, 0, 1 if name == "rows_ddl" else 0] for name in outcomes}
    if mode == "session_struct_subclass":
        expected["arrow_none"] = expected["arrow_names"] = [None, 1, 0]
    assert outcomes == expected

# PySpark's outcomes of a name rebound to `LongType` (to `BooleanType` of `LongType`) in
# `pyspark.sql.types` with its converter module removed, so imported anew of it, or in the
# converter module: `to_arrow_type` takes the long field of the first class that it matches. A
# name rebound in a converter module that the module of the session does not hold, as the one of
# `captured_alias`, which is imported anew, changes nothing.
_INTEGER_REBOUND = {
    "BooleanType": _integer_converts([value != 0 for value in _INTEGER_VALUES], "bool"),
    "ByteType": _guard_raises(*_conversion_error("PySparkValueError", "int64", "renamed", "int8")),
    "ShortType": _guard_raises(*_conversion_error("PySparkValueError", "int64", "renamed",
                                                  "int16")),
    "IntegerType": _guard_raises(*_conversion_error("PySparkValueError", "int64", "renamed",
                                                    "int32")),
    "LongType": _guard_raises(*_unsupported_type("LongType()")),
}


@pytest.mark.parametrize("class_name", ["BooleanType", "ByteType", "ShortType", "IntegerType",
                                        "LongType"])
@pytest.mark.parametrize("mode", ["public_alias", "converter_alias", "captured_alias"])
def test_integer_rebound_converter_aliases_take_pyspark_conversion(class_name, mode):
    _assert_guard(_integer_guard_process(mode, class_name),
                  _integer_converts() if mode == "captured_alias" else _INTEGER_REBOUND[class_name])


@pytest.mark.parametrize("replacement", ["none", "long"])
def test_integer_rebound_importer_schema_alias_takes_pyspark_conversion(replacement):
    # The names of the drop-in's own importer are not PySpark's, which has none.
    _assert_guard(_integer_guard_process("importer_alias", replacement), _integer_converts())


# As PySpark's createDataFrame, which checks its schema against the `AtomicType` and the
# `StructType` of its module of the session, which binds them with the session
# (`_check_schema` of `pyspark.sql.connect.session`): a public `AtomicType` of None, rebound
# before, makes its `isinstance` raise, before any setting is read.
_ATOMIC_ALIAS_OF_PYSPARK = _guard_raises(
    "builtins.TypeError", "isinstance() arg 2 must be a type, a tuple of types, or a union",
    configs=False)


@pytest.mark.parametrize("scope", ["public_before", "public_after", "importer"])
@pytest.mark.parametrize("replacement", ["none", "long"])
def test_integer_rebound_atomic_alias_takes_pyspark_conversion(scope, replacement):
    outcome = _integer_guard_process("atomic_alias", f"{scope}:{replacement}")
    if scope != "importer" and replacement == "none":
        _assert_guard(outcome, _ATOMIC_ALIAS_OF_PYSPARK)
    else:
        _assert_guard(outcome, _integer_converts())


def test_integer_deleted_importer_atomic_alias_takes_pyspark_conversion():
    _assert_guard(_integer_guard_process("atomic_alias", "importer:missing"), _integer_converts())


def test_integer_warm_importer_without_global_converter_aliases(protocol, monkeypatch):
    import pyspark.sql._arrow_input as importer

    original = importer.prepare_table

    def local_imports(table, schema, *configs):
        # Newer importers import converters within prepare_table. Exercise that
        # module shape even on the pinned PR118 dependency used by this branch.
        from pyspark.sql.pandas.types import from_arrow_schema, to_arrow_schema
        table, resolved, names = original(table, schema, *configs)
        assert from_arrow_schema(table.schema).fields
        assert to_arrow_schema(resolved).names == table.column_names
        return table, resolved, names

    frame = pd.DataFrame({"a": np.arange(1000, dtype="int64")})
    schema = T.StructType([T.StructField("renamed", T.LongType())])
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(pa.table({"warm": [1, 2]})).toArrow()
        # Keep original's module globals available in the dependency importer;
        # the full-stack validation separately uses the actual local-import helper.
        import types
        substitute = types.ModuleType(importer.__name__)
        substitute.StructType = T.StructType
        substitute.AtomicType = T.AtomicType
        substitute.prepare_table = local_imports
        monkeypatch.setitem(__import__("sys").modules, importer.__name__, substitute)

        def no_rows(*args, **kwargs):
            raise AssertionError("warm integer input must stay columnar")

        monkeypatch.setattr(pd.DataFrame, "astype", no_rows)
        for _ in range(3):
            spark.createDataFrame(frame, schema).toArrow()
        for table in state["input_tables"][1:]:
            assert table.column_names == ["renamed"]
            assert table.column(0).to_pylist() == list(range(1000))







@pytest.mark.parametrize("dtype", ["int64", "Int64"])
@pytest.mark.parametrize("schema_kind", ["inferred", "names", "explicit"])
def test_integer_rebound_index_data_reads_no_more_than_pyspark_conversion(
    protocol, monkeypatch, dtype, schema_kind
):
    import json

    values = [2**53 + 1, -(2**63), 2**63 - 1, 0] * 250
    if dtype == "Int64":
        values[1] = None
    frame = pd.DataFrame({"original": pd.Series(values, dtype=dtype)})
    schema, expected_name, nullable, metadata = None, "original", True, {}
    if schema_kind == "names":
        schema, expected_name = ["renamed"], "renamed"
    elif schema_kind == "explicit":
        expected_name, nullable, metadata = "renamed", dtype == "Int64", {"source": "native"}
        schema = T.StructType([T.StructField(expected_name, T.LongType(), nullable, metadata)])
    columns, backing = frame.columns, vars(frame.columns)["_data"]
    calls = []
    allowed_reads = None

    class ColumnsDataError(RuntimeError):
        pass

    def data(index):
        if index is columns:
            calls.append("_data")
            if allowed_reads is not None and calls.count("_data") > allowed_reads:
                raise ColumnsDataError("eligibility added a columns storage read")
            return backing
        return vars(index)["_data"]

    def set_data(index, value):
        # Preserve ordinary storage initialization for indexes pandas creates
        # while converting values; only reads of the input columns are hooked.
        vars(index)["_data"] = value

    monkeypatch.setattr(pd.Index, "_data", property(data, set_data), raising=False)
    import pyspark.sql._pandas_input as helper

    with _arrow_input_service(protocol) as (spark, state):
        # The reads of PySpark's conversion of the frame, of the labels that it iterates, with
        # the fast path declined.
        with monkeypatch.context() as baseline:
            baseline.setattr(helper, "prepare_int64_table", lambda *_: None)
            spark.createDataFrame(frame, schema).toArrow()
        allowed_reads = calls.count("_data")
        assert allowed_reads > 0
        calls.clear()
        spark.createDataFrame(frame, schema).toArrow()
        # The rebound storage declines the fast path without reading the labels more than
        # PySpark's conversion does. pandas caches some of these reads, so a later conversion
        # of the same frame can read them fewer times than the first.
        assert 0 < calls.count("_data") <= allowed_reads
        # One read of the settings for each of the two createDataFrame calls, as PySpark's.
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)] * 2
        assert len(state["input_tables"]) == 2
        table = state["input_tables"][1]
        assert table.equals(state["input_tables"][0], check_metadata=True)
        assert table.column(0).to_pylist() == values
        assert table.column(0).null_count == sum(value is None for value in values)
        assert state["column_names"] == [expected_name]
        if schema_kind == "explicit":
            assert table.schema.equals(
                pa.schema([pa.field(expected_name, pa.int64(), nullable=nullable)]),
                check_metadata=True,
            )
            assert json.loads(state["relations"][0].local_relation.schema) == {
                "type": "struct", "fields": [
                    {"name": expected_name, "type": "long", "nullable": nullable,
                     "metadata": metadata}
                ],
            }
        else:
            # Of the types that Arrow infers, sent without a schema, which toDF names.
            assert table.schema.equals(pa.schema([pa.field("_0", pa.int64())]),
                                       check_metadata=True)
            assert state["spark_schema"] is None


@pytest.mark.parametrize("zone", ["UTC", "America/Los_Angeles", "Asia/Kathmandu"])
@pytest.mark.parametrize("unit", ["ns", "us"])
def test_pandas_timestamp_ipc_preserves_every_microsecond(protocol, monkeypatch, zone, unit):
    import json

    from pyspark.sql import types as T

    monkeypatch.setenv("TZ", "America/New_York")
    # Include both sides of DST transitions, both fall-back folds, pre-epoch,
    # ns-range boundaries, and NaT. Inspect the full emitted IPC, not a plan flag.
    micros = [-9223372036854775, -1, 0, 1, 1583661599999999, 1583661600000000,
              1604219400000000, 1604223000000000, 9223372036854775, None]
    values = pd.to_datetime(pd.Series(micros, dtype="Int64"), unit="us", utc=True).dt.tz_convert("America/Los_Angeles")
    frame = pd.DataFrame({"a": pd.Series(values).dt.as_unit(unit),
                          "b": pd.Series(values).dt.as_unit(unit)})
    schema = T.StructType([T.StructField("renamed_a", T.TimestampType(), True, {"source": "test"}),
                           T.StructField("b", T.TimestampType())])

    with _arrow_input_service(protocol, capture_tables=True, **{"spark.sql.session.timeZone": zone}) as (spark, state):
        result = spark.createDataFrame(frame, schema)
        # Aware columns carry their zone: only the importer's settings are read.
        assert len(state["config_requests"]) == 1
        assert not result.isLocal()
        result.toArrow()
        assert state["column_names"] == schema.names
        table = pa.concat_tables(state["tables"])
        for column in table.columns:
            assert column.type == pa.timestamp("us", tz="UTC")
            assert column.cast(pa.int64()).to_pylist() == micros
        assert table.column_names == ["renamed_a", "b"]
        relation = state["relations"][0].local_relation
        actual = json.loads(relation.schema)
        assert [field["type"] for field in actual["fields"]] == ["timestamp", "timestamp"]
        assert actual["fields"][0]["metadata"] == {"source": "test"}


def test_pandas_naive_timestamp_utc_ipc(protocol, monkeypatch):
    import time

    from pyspark.sql import types as T
    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.setattr(time, "tzname", ("UTC", "UTC"))
    frame = pd.DataFrame({"a": pd.to_datetime(["1969-12-31 23:59:59.999999", None,
                                               "2020-01-01 00:00:00.123456"])})
    schema = T.StructType([T.StructField("a", T.TimestampType())])
    with _arrow_input_service(protocol, capture_tables=True) as (spark, state):
        result = spark.createDataFrame(frame, schema)
        # The one read of the settings gives the naive column its session time zone, and the
        # importer its settings, as PySpark's createDataFrame reads them before it converts.
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
        assert not result.isLocal()
        result.toArrow()
        assert state["column_names"] == schema.names
        assert state["tables"][0].column(0).cast(pa.int64()).to_pylist() == [
            -1, None, 1577836800123456]


@pytest.mark.parametrize("dtype", ["datetime64[ns]", "datetime64[us]", "datetime64[ms]",
                                    "datetime64[s]", "object"])
def test_pandas_timestamp_eligibility(monkeypatch, dtype):
    import time

    from pyspark.sql import types as T
    from pyspark.sql._pandas_input import timestamp_columns

    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.setattr(time, "tzname", ("UTC", "UTC"))
    frame = pd.DataFrame({"a": pd.Series([datetime(2020, 1, 1), None], dtype=dtype)})
    schema = T.StructType([T.StructField("a", T.TimestampType())])
    assert (timestamp_columns(frame, schema) is not None) == (dtype in (
        "datetime64[ns]", "datetime64[us]"))
    for unsupported in (None, ["a"], "a timestamp",
                        T.StructType([T.StructField("a", T.TimestampNTZType())]),
                        T.StructType([T.StructField("a", T.LongType())])):
        assert timestamp_columns(frame, unsupported) is None


def test_pandas_timestamp_precision_and_timezone_fallback(monkeypatch):
    import time

    from pyspark.sql import types as T
    from pyspark.sql._pandas_input import prepare_timestamp_table, timestamp_columns

    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.setattr(time, "tzname", ("UTC", "UTC"))
    schema = T.StructType([T.StructField("a", T.TimestampType())])
    for nanos in (-1, 1, 1001):
        frame = pd.DataFrame({"a": pd.to_datetime([nanos, None], unit="ns")})
        assert timestamp_columns(frame, schema) is None
    frame = pd.DataFrame({"a": pd.to_datetime([0, None], unit="ns")})
    columns = timestamp_columns(frame, schema)
    assert columns is not None
    assert prepare_timestamp_table(columns, schema, "America/Los_Angeles") is None
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    assert prepare_timestamp_table(columns, schema, "UTC") is None
    monkeypatch.delenv("TZ")
    assert prepare_timestamp_table(columns, schema, "UTC") is None


@pytest.mark.parametrize("unit,mixed", [("ns", False), ("us", True)])
@pytest.mark.parametrize("host_tz,host_names", [
    (None, ("UTC", "UTC")),
    ("America/Los_Angeles", ("PST", "PDT")),
    ("UTC", ("PST", "PDT")),
], ids=["unset-tz", "non-utc-tz", "mismatched-host-zone"])
def test_timestamp_non_utc_host_declines_and_converts_as_pyspark(
    protocol, monkeypatch, unit, mixed, host_tz, host_names
):
    # A host zone other than a literal UTC declines the timestamp fast path before its
    # session time zone RPC; the frame then converts as PySpark's Arrow conversion does,
    # which reads the configs first and takes naive values in the session time zone (UTC).
    import time

    if host_tz is None:
        monkeypatch.delenv("TZ", raising=False)
    else:
        monkeypatch.setenv("TZ", host_tz)
    monkeypatch.setattr(time, "tzname", host_names)
    micros = [-2000000, 0, 123456, None]
    columns = {}
    if mixed:
        columns["aware"] = pd.to_datetime(micros, unit="us", utc=True).as_unit(unit)
    columns["naive"] = pd.to_datetime(micros, unit="us").as_unit(unit)
    frame = pd.DataFrame(columns)
    schema = T.StructType([
        T.StructField(f"renamed_{name}", T.TimestampType(), True, {"source": name})
        for name in columns
    ])
    from pyspark.sql._pandas_input import timestamp_columns

    assert timestamp_columns(frame, schema) is None
    expected = [micros for _ in columns]
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert state["config_requests"]
        assert state["uploads"] == 0
        assert state["column_names"] == schema.names
        assert len(state["input_tables"]) == 1
        table = state["input_tables"][0]
        assert table.num_rows == len(micros)
        for column, values in zip(table.columns, expected):
            assert column.type == pa.timestamp("us", tz="UTC")
            assert column.cast(pa.int64()).to_pylist() == values
            assert column.is_null().to_pylist() == [value is None for value in values]


@pytest.mark.parametrize("has_null", [False, True])
def test_pandas_timestamp_non_nullable_schema(protocol, has_null):
    from pyspark.sql import types as T
    values = [pd.Timestamp("2020-01-01", tz="UTC"),
              pd.NaT if has_null else pd.Timestamp("2020-01-02", tz="UTC")]
    frame = pd.DataFrame({"a": values})
    schema = T.StructType([T.StructField("a", T.TimestampType(), False)])
    with _arrow_input_service(protocol, capture_tables=True) as (spark, state):
        if has_null:
            # Official: the cast of the Arrow table, a ValueError rather than a row error.
            with pytest.raises(ValueError) as error:
                spark.createDataFrame(frame, schema)
            assert str(error.value) == "Casting field 'a' with null values to non-nullable"
            assert not state["relations"]
        else:
            spark.createDataFrame(frame, schema).toArrow()
            assert not state["tables"][0].schema.field(0).nullable


def test_pandas_timestamp_mixed_and_arrow_dtypes_fallback():
    from pyspark.sql import types as T
    from pyspark.sql._pandas_input import timestamp_columns
    schema = T.StructType([T.StructField("a", T.TimestampType())])
    frame = pd.DataFrame({"a": pd.Series([datetime(2020, 1, 1)],
                                         dtype=pd.ArrowDtype(pa.timestamp("us")))})
    assert timestamp_columns(frame, schema) is None
    frame = pd.DataFrame({"a": pd.to_datetime([0], unit="us"), "b": [1]})
    schema.add("b", T.LongType())
    assert timestamp_columns(frame, schema) is None


@pytest.mark.parametrize("config_error", [False, True])
@pytest.mark.parametrize("verify", [False, True])
def test_timestamp_nonnullable_nat_reads_config_before_cast_error(protocol, config_error, verify):
    from pyspark.sql import types as T
    from pyspark.sql._pandas_input import timestamp_columns

    frame = pd.DataFrame({"a": pd.to_datetime([0, None], utc=True)})
    schema = T.StructType([T.StructField("renamed", T.TimestampType(), False)])
    assert timestamp_columns(frame, schema) is None
    with _arrow_input_service(protocol, capture_tables=True) as (spark, state):
        if config_error:
            state["config_error"] = grpc.StatusCode.INVALID_ARGUMENT
            with pytest.raises(Exception, match="configuration read failed"):
                spark.createDataFrame(frame, schema, verifySchema=verify)
        else:
            # Official: ValueError from the cast of the Arrow table to the schema.
            with pytest.raises(ValueError) as error:
                spark.createDataFrame(frame, schema, verifySchema=verify)
            assert str(error.value) == (
                "Casting field 'renamed' with null values to non-nullable")
        assert state["config_requests"][0] == ("get", _CREATE_DATAFRAME_KEYS)
        assert state["relations"] == []
        assert state["uploads"] == 0


@pytest.mark.parametrize("unit,aware", [("ns", True), ("us", False)])
@pytest.mark.parametrize("location,method", [
    ("instance", "notna"), ("instance", "astype"), ("instance", "where"),
    ("class", "notna"), ("class", "astype"), ("class", "where"),
])
@pytest.mark.parametrize("behavior", ["mutate", "raise"])
def test_timestamp_custom_normalization_hooks_are_not_called_as_in_pyspark(
    protocol, monkeypatch, unit, aware, location, method, behavior
):
    # A frame whose notna, astype or where is customized, of its class or of the instance,
    # declines the timestamp fast path, and converts as PySpark's conversion does, which never
    # calls them: the values are the frame's, whatever the hooks would raise or change.
    import json
    import time
    from types import MethodType

    import pyspark.sql._pandas_input as helper

    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.setattr(time, "tzname", ("UTC", "UTC"))
    values = pd.to_datetime([-2000000, -1000000, 0, 1000000, None], unit="us", utc=aware)
    frame = pd.DataFrame({"original": values.as_unit(unit)})
    schema = T.StructType([
        T.StructField("renamed", T.TimestampType(), True, {"source": "normalization"})
    ])
    original = getattr(pd.DataFrame, method)
    calls, tables = [], []
    # The microseconds of the values in the session time zone, UTC, whatever the host's zone:
    # `Timestamp.value` reads a naive value as a UTC wall clock, as the session does.
    expected = [None if pd.isna(value) else value.value // 1000 for value in frame["original"]]
    message = f"custom timestamp {method} sentinel"

    def customized(self, *args, **kwargs):
        calls.append(method)
        if behavior == "raise":
            raise RuntimeError(message)
        result = original(self, *args, **kwargs)
        result.iat[0, 0] = (False if method == "notna"
                            else datetime(1970, 1, 1, 0, 0, 7, tzinfo=timezone.utc))
        return result

    if location == "class":
        monkeypatch.setattr(pd.DataFrame, method, customized)
    else:
        monkeypatch.setitem(vars(frame), method, MethodType(customized, frame))
    assert helper.timestamp_columns(frame, schema) is None

    # PySpark's conversion with the fast path declined, then the ordinary dispatch, which
    # declines it too: neither runs the hooks, nor reads the settings more than once, as
    # PySpark's createDataFrame reads them before it converts any data.
    for bypass in (True, False):
        with monkeypatch.context() as dispatch:
            if bypass:
                dispatch.setattr(helper, "timestamp_columns", lambda *_: None)
            with _arrow_input_service(protocol) as (spark, state):
                spark.createDataFrame(frame, schema).toArrow()
                assert len(state["input_tables"]) == 1
                table = state["input_tables"][0]
                assert state["column_names"] == schema.names
                assert table.column(0).cast(pa.int64()).to_pylist() == expected
                assert table.column(0).is_null().to_pylist() == [
                    value is None for value in expected]
                assert json.loads(state["relations"][0].local_relation.schema) == (
                    json.loads(schema.json()))
                tables.append(table)
                assert calls == []
                assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
                assert state["uploads"] == 0
    assert tables[0].equals(tables[1], check_metadata=True)


@pytest.mark.parametrize("custom", ["dtype", "array"])
def test_timestamp_custom_datetime_array_converts_as_pyspark(protocol, custom):
    # A subclass of pandas' datetime array or dtype declines the timestamp fast path. PySpark's
    # conversion of a timestamp field converts the column with pandas first
    # (`_check_series_convert_timestamps_internal`), whose result is of pandas' own array, so
    # the values are those of the column, and the subclass's `__arrow_array__` is not called.
    from pandas.core.arrays.datetimes import DatetimeArray
    from pandas.core.dtypes.dtypes import DatetimeTZDtype
    from pyspark.sql import types as T
    from pyspark.sql._pandas_input import timestamp_columns

    arrow_calls = []

    class CustomDtype(DatetimeTZDtype):
        pass

    class CustomArray(DatetimeArray):
        @property
        def dtype(self):
            return CustomDtype(unit="ns", tz="UTC") if custom == "dtype" else super().dtype

        def __arrow_array__(self, type=None):
            arrow_calls.append("arrow")
            return pa.array([-7000000] * len(self), type=pa.int64()).cast(type)

    ordinary = pd.to_datetime([0, 1000000, None, -1000000], unit="ns", utc=True).array
    array = CustomArray._simple_new(ordinary._ndarray, dtype=ordinary.dtype)
    frame = pd.DataFrame({"a": pd.Series(array, copy=False)})
    assert type(frame["a"].array) is CustomArray
    schema = T.StructType([T.StructField("a", T.TimestampType(), True, {"source": "custom"})])
    assert timestamp_columns(frame, schema) is None
    with _arrow_input_service(protocol, capture_tables=True) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
        table = pa.concat_tables(state["tables"])
        assert table.column(0).cast(pa.int64()).to_pylist() == [0, 1000, None, -1000]
        logical = __import__("json").loads(state["relations"][0].local_relation.schema)
        assert logical["fields"][0]["metadata"] == {"source": "custom"}
    assert arrow_calls == []


def test_timestamp_rebound_dataframe_and_duplicate_names_keep_fallback(monkeypatch):
    from pyspark.sql import types as T
    from pyspark.sql._pandas_input import timestamp_columns

    class CustomFrame(pd.DataFrame):
        pass

    frame = CustomFrame({"a": pd.to_datetime([0], utc=True)})
    schema = T.StructType([T.StructField("a", T.TimestampType())])
    monkeypatch.setattr(pd, "DataFrame", CustomFrame)
    assert timestamp_columns(frame, schema) is None
    from pandas.core.frame import DataFrame
    frame = DataFrame({"a": pd.to_datetime([0], utc=True), "b": pd.to_datetime([1], utc=True)})
    monkeypatch.setattr(pd, "DataFrame", DataFrame)
    schema.add("a", T.TimestampType())
    assert timestamp_columns(frame, schema) is None


def _timestamp_guard_process(mode, argument, before_import=False):
    """The outcome of `createDataFrame` of a frame of aware datetimes under `renamed timestamp` of
    a class patched or a name rebound by `mode` and `argument`, in a process of its own, which the
    session converts with PySpark's conversion (see `_integer_guard_process`)."""
    if mode == "captured_alias":
        import pyspark.sql._arrow_input as importer
        if not hasattr(importer, "to_arrow_schema"):
            pytest.skip("this importer resolves converters locally")
    script = r'''
import importlib
import importlib.util
import sys
import pandas as pd
from pyspark.sql import types as T
from pyspark import _pyspark as native
spec = importlib.util.spec_from_file_location("timestamp_guard_capture", sys.argv[-1])
capture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capture
spec.loader.exec_module(capture)
frame = pd.DataFrame({"a": pd.to_datetime([0, 1000000, None, -1000000], unit="ns", utc=True)})
schema = T.StructType([T.StructField("renamed", T.TimestampType(), True, {"source": "native"})])
mode, argument, before_import = sys.argv[1:4]
if before_import == "True":
    sys.modules.pop("pyspark.sql._pandas_input", None)
if mode == "descriptor":
    class_name, attribute = argument.split(":")
    cls = getattr(native, class_name)
    if attribute == "alias":
        setattr(T, class_name, object)
    elif attribute == "__getattribute__":
        setattr(cls, attribute, lambda self, name: object.__getattribute__(self, name))
    elif attribute == "__class__":
        setattr(cls, "__getattribute__", lambda self, name: (
            object if name == "__class__" else object.__getattribute__(self, name)))
    elif attribute == "__iter__":
        setattr(cls, attribute, lambda self: iter([T.StructField("changed", T.BooleanType())]))
    else:
        value = {"names": ["changed"], "fields": [], "name": "changed",
                 "dataType": T.BooleanType(), "nullable": False,
                 "metadata": {"changed": True}}[attribute]
        setattr(cls, attribute, property(lambda self: value))
elif mode in ("public_alias", "converter_alias", "captured_alias"):
    replacement = native.BooleanType if argument == "TimestampType" else native.TimestampType
    if mode == "public_alias":
        sys.modules.pop("pyspark.sql.pandas.types", None)
        setattr(T, argument, replacement)
    else:
        converter = importlib.import_module("pyspark.sql.pandas.types")
        if mode == "captured_alias":
            importer = importlib.import_module("pyspark.sql._arrow_input")
        setattr(converter, argument, replacement)
        if mode == "captured_alias":
            sys.modules.pop("pyspark.sql.pandas.types")
            assert importlib.import_module("pyspark.sql.pandas.types") is not converter
elif mode == "atomic_alias":
    scope, replacement_name = argument.split(":")
    replacement = None if replacement_name == "none" else native.LongType
    if scope == "public_before":
        sys.modules.pop("pyspark.sql._arrow_input", None)
        T.AtomicType = replacement
    elif scope == "public_after":
        importlib.import_module("pyspark.sql._arrow_input")
        T.AtomicType = replacement
    else:
        importer = importlib.import_module("pyspark.sql._arrow_input")
        if replacement_name == "missing":
            del importer.AtomicType
        else:
            importer.AtomicType = replacement
elif mode == "importer_alias":
    import pyspark.sql._arrow_input as importer
    importer.StructType = None
else:
    raise AssertionError(mode)
'''
    return _guard_outcome(script, mode, argument, str(before_import))


_TIMESTAMP_RELATION = {"fields": [{"metadata": {"source": "native"}, "name": "renamed",
                                   "nullable": True, "type": "timestamp"}], "type": "struct"}


def _timestamp_converts(values=(0, 1000, None, -1000), arrow_type="timestamp[us, tz=UTC]",
                        name="renamed", metadata=None, error=None, relation=_TIMESTAMP_RELATION):
    """Of the microseconds `values` of the timestamps, or the values of another type."""
    return _guard_converts([(name, arrow_type, True, list(values), metadata)], relation, error)


# PySpark's outcomes (see `_INTEGER_DESCRIPTORS`), and the drop-in's where they differ.
_TIMESTAMP_DESCRIPTORS = {
    ("StructType", "fields"): _guard_raises(*_no_setter("StructType")),
    ("StructType", "names"): _guard_raises(*_no_setter("StructType")),
    ("StructType", "__iter__"): _guard_raises(*_rename_mismatch("renamed")),
    ("StructField", "name"): _guard_raises(*_no_setter("StructField")),
    ("StructField", "dataType"): _guard_raises(*_no_setter("StructField")),
    ("StructField", "nullable"): _guard_raises(*_no_setter("StructField")),
    ("StructField", "metadata"): _guard_raises(*_no_setter("StructField")),
    ("StructType", "alias"): _timestamp_converts(
        error=["_to_table", "builtins.AssertionError", ""]),
}
_TIMESTAMP_DESCRIPTORS_OF_THE_DROP_IN = {
    ("StructType", "fields"): _guard_converts([], {"fields": [], "type": "struct"}),
    ("StructType", "names"): _timestamp_converts(name="changed"),
    ("StructField", "name"): _guard_raises(*_rename_mismatch("renamed")),
    ("StructField", "dataType"): _guard_raises(
        "pyarrow.lib.ArrowNotImplementedError",
        "Unsupported cast from timestamp[ns] to bool using function cast_boolean"),
    ("StructField", "nullable"): _guard_raises(
        "builtins.ValueError", "Casting field 'renamed' with null values to non-nullable"),
    ("StructField", "metadata"): _timestamp_converts(
        metadata={"SPARK::metadata::json": '{"changed": true}'},
        relation=_relation_of("renamed", "timestamp", True, {"changed": True})),
    ("StructType", "alias"): _timestamp_converts(),
}


@pytest.mark.parametrize("class_name,attribute", [
    ("StructType", "fields"), ("StructType", "names"), ("StructType", "__iter__"),
    ("StructField", "name"), ("StructField", "dataType"), ("StructField", "nullable"),
    ("StructField", "metadata"), ("StructType", "__getattribute__"),
    ("StructField", "__getattribute__"), ("TimestampType", "__getattribute__"),
    ("StructType", "__class__"), ("StructField", "__class__"), ("TimestampType", "__class__"),
    ("StructType", "alias"), ("StructField", "alias"), ("TimestampType", "alias"),
])
@pytest.mark.parametrize("before_import", [False, True])
def test_timestamp_custom_schema_descriptors_take_pyspark_conversion(class_name, attribute,
                                                                     before_import):
    outcome = _timestamp_guard_process("descriptor", f"{class_name}:{attribute}", before_import)
    _assert_guard(outcome,
                  _TIMESTAMP_DESCRIPTORS.get((class_name, attribute), _timestamp_converts()),
                  _TIMESTAMP_DESCRIPTORS_OF_THE_DROP_IN.get((class_name, attribute)))


def _timestamp_cast_error(arrow_type, function):
    return _guard_raises("pyarrow.lib.ArrowNotImplementedError",
                         f"Unsupported cast from timestamp[ns] to {arrow_type} using function "
                         f"{function}")


# PySpark's outcomes of a name rebound to `TimestampType` (to `BooleanType` of `TimestampType`):
# `to_arrow_type` takes the timestamp field of the first class that it matches, of which PyArrow
# casts the nanoseconds of the column, after the conversion of the time zone of a timestamp, or
# fails to; a `BinaryType` takes the bytes of the integers, and a `DateType`, which it matches
# before `TimestampType`, the dates of the timestamps.
_TIMESTAMP_REBOUND = {
    "BooleanType": _timestamp_cast_error("bool", "cast_boolean"),
    "ByteType": _timestamp_cast_error("int8", "cast_int8"),
    "ShortType": _timestamp_cast_error("int16", "cast_int16"),
    "IntegerType": _timestamp_cast_error("int32", "cast_int32"),
    "LongType": _timestamp_converts([0, 1000000, None, -1000000], "int64"),
    "FloatType": _timestamp_cast_error("float", "cast_float"),
    "DoubleType": _timestamp_cast_error("double", "cast_double"),
    "DecimalType": _guard_raises("builtins.AttributeError",
                                 "'TimestampType' object has no attribute 'precision'"),
    "StringType": _guard_raises(*_conversion_error("PySparkTypeError", "datetime64[ns, UTC]",
                                                   "a", "string")),
    "BinaryType": _timestamp_converts(
        [b"", b"@B\x0f", None, b"\xc0\xbd\xf0\xff\xff\xff\xff\xff"], "binary"),
    "DateType": _timestamp_converts(
        [date(1970, 1, 1), date(1970, 1, 1), None, date(1969, 12, 31)], "date32[day]"),
    "TimestampType": _guard_raises(*_unsupported_type("TimestampType()")),
}


@pytest.mark.parametrize("class_name", ["BooleanType", "ByteType", "ShortType", "IntegerType",
                                        "LongType", "FloatType", "DoubleType", "DecimalType",
                                        "StringType", "BinaryType", "DateType", "TimestampType"])
@pytest.mark.parametrize("mode", ["public_alias", "converter_alias", "captured_alias"])
def test_timestamp_rebound_converter_aliases_take_pyspark_conversion(class_name, mode):
    _assert_guard(_timestamp_guard_process(mode, class_name),
                  _timestamp_converts() if mode == "captured_alias"
                  else _TIMESTAMP_REBOUND[class_name])


def test_timestamp_rebound_importer_schema_alias_takes_pyspark_conversion():
    _assert_guard(_timestamp_guard_process("importer_alias", "none"), _timestamp_converts())


@pytest.mark.parametrize("scope", ["public_before", "public_after", "importer"])
@pytest.mark.parametrize("replacement", ["none", "long"])
def test_timestamp_rebound_atomic_alias_takes_pyspark_conversion(scope, replacement):
    # See `test_integer_rebound_atomic_alias_takes_pyspark_conversion`.
    outcome = _timestamp_guard_process("atomic_alias", f"{scope}:{replacement}")
    if scope != "importer" and replacement == "none":
        _assert_guard(outcome, _ATOMIC_ALIAS_OF_PYSPARK)
    else:
        _assert_guard(outcome, _timestamp_converts())


def test_timestamp_deleted_importer_atomic_alias_takes_pyspark_conversion():
    _assert_guard(_timestamp_guard_process("atomic_alias", "importer:missing"),
                  _timestamp_converts())


def test_timestamp_warm_local_converter_imports_keep_columnar_route(protocol, monkeypatch):
    import sys
    import types

    import pyspark.sql._arrow_input as importer
    from pyspark.sql import types as T
    from pyspark.sql._pandas_input import timestamp_columns

    original = importer.prepare_table
    substitute = types.ModuleType(importer.__name__)
    substitute.StructType = T.StructType
    substitute.AtomicType = T.AtomicType

    def prepare_table(table, schema, *configs):
        from pyspark.sql.pandas.types import to_arrow_schema
        assert to_arrow_schema(schema, timezone="UTC").names == table.column_names
        return original(table, schema, *configs)

    substitute.prepare_table = prepare_table
    frame = pd.DataFrame({"a": pd.to_datetime([0, 1000000, None], unit="ns", utc=True)})
    schema = T.StructType([T.StructField("renamed", T.TimestampType())])
    with _arrow_input_service(protocol, capture_tables=True) as (spark, state):
        spark.createDataFrame(pa.table({"warm": [1]})).toArrow()
        monkeypatch.setitem(sys.modules, importer.__name__, substitute)
        assert timestamp_columns(frame, schema) is not None
        spark.createDataFrame(frame, schema).toArrow()
        assert len(state["config_requests"]) == 2
        table = state["tables"][-1]
        assert table.column_names == ["renamed"]
        assert table.column(0).cast(pa.int64()).to_pylist() == [0, 1000, None]


def _timestamp_columns_fixture(unit):
    values = [-2000000, -1000000, 0, 1000000, None] * 2000
    frame = pd.DataFrame({
        "original": pd.to_datetime(pd.Series(values, dtype="Int64"), unit="us", utc=True).dt.as_unit(unit)
    })
    schema = T.StructType([
        T.StructField("renamed", T.TimestampType(), True, {"source": "columns", "id": 3})
    ])
    return values, frame, schema


def _assert_timestamp_columns_input(state, values, schema):
    import json

    # As PySpark's createDataFrame, the settings are read before any conversion, and the 80 KB
    # of Arrow data stay inline below the default threshold.
    assert any("spark.sql.session.localRelationCacheThreshold" in keys
               for _, keys in state["config_requests"])
    assert not state["uploads"]
    assert len(state["input_tables"]) == 1
    assert state["column_names"] == schema.names
    table = state["input_tables"][0]
    assert table.num_rows == 10000
    assert table.schema.equals(
        pa.schema([pa.field("renamed", pa.timestamp("us", tz="UTC"), nullable=True)]),
        check_metadata=True,
    )
    assert table.column(0).cast(pa.int64()).to_pylist() == values
    assert table.column(0).is_null().to_pylist() == [value is None for value in values]
    assert json.loads(state["relations"][0].local_relation.schema) == json.loads(schema.json())


@pytest.mark.parametrize("unit", ["ns", "us"])
@pytest.mark.parametrize("location", [
    "subclass", "class", "owner", "backing_subclass_item", "backing_class_iter",
    "backing_class_getitem", "backing_instance_box",
])
@pytest.mark.parametrize("behavior", ["raise", "mutate"])
def test_timestamp_columns_iterator_hooks_run_in_pyspark_conversion(
    protocol, monkeypatch, unit, location, behavior
):
    # As for int64 columns: the fast path declines the customized labels without iterating
    # them, and PySpark's conversion iterates them, for the columns of `items()`, before it
    # reads any value. It never calls the frame's notna or astype.
    from pandas.core.base import IndexOpsMixin

    values, frame, schema = _timestamp_columns_fixture(unit)
    original_iter = pd.Index.__iter__
    original_notna, original_astype = frame.notna, frame.astype
    calls = []
    invoked = False

    class ColumnsError(RuntimeError):
        pass

    def notna(*args, **kwargs):
        calls.append("notna")
        return original_notna(*args, **kwargs)

    def astype(*args, **kwargs):
        calls.append("astype")
        return original_astype(*args, **kwargs)

    def before_hook():
        nonlocal invoked
        if invoked:
            return
        invoked = True
        calls.append("columns iterator")
        if behavior == "raise":
            raise ColumnsError("timestamp columns hook precedes the conversion of the values")
        # The spies of the frame's normalization methods are installed once the conversion
        # iterates the labels, as the values that it reads next are changed: before, the fast
        # path's guards see the canonical methods.
        monkeypatch.setitem(vars(frame), "notna", notna)
        monkeypatch.setitem(vars(frame), "astype", astype)
        frame.iat[0, 0] = pd.Timestamp(7000000, unit="us", tz="UTC")
        frame.iat[-1, 0] = pd.NaT

    def iterate(index):
        if index is columns:
            before_hook()
        return original_iter(index)

    if location == "subclass":
        class Columns(pd.Index):
            pass
        monkeypatch.setattr(Columns, "__iter__", iterate)
        labels = pd.Index(["original"], dtype=object)
        frame.columns = Columns._simple_new(labels._values, name=None)
        assert type(frame.columns) is Columns
    elif location == "backing_subclass_item":
        # IndexOpsMixin iteration calls ndarray.item, rather than ndarray.__iter__.
        class Names(np.ndarray):
            def item(self, *args):
                before_hook()
                return super().item(*args)
        backing = np.asarray(["original"], dtype=object).view(Names)
        frame.columns = pd.Index._simple_new(backing, name=None)
        assert type(frame.columns._values) is Names
    elif location.startswith("backing_"):
        frame.columns = pd.Index(pd.array(["original"], dtype=pd.StringDtype(storage="python")))
        backing = frame.columns._values
        if location == "backing_class_iter":
            original = type(backing).__iter__

            def backing_iter(array):
                if array is backing:
                    before_hook()
                return original(array)

            monkeypatch.setattr(type(backing), "__iter__", backing_iter)
        elif location == "backing_class_getitem":
            original = type(backing).__getitem__

            def backing_getitem(array, key):
                if array is backing:
                    before_hook()
                return original(array, key)

            monkeypatch.setattr(type(backing), "__getitem__", backing_getitem)
        else:
            original = backing._box_func

            def box(value):
                before_hook()
                return original(value)

            monkeypatch.setitem(vars(backing), "_box_func", box)
    else:
        monkeypatch.setattr(pd.Index if location == "class" else IndexOpsMixin,
                            "__iter__", iterate)
    columns = frame.columns

    with _arrow_input_service(protocol) as (spark, state):
        if behavior == "raise":
            with pytest.raises(ColumnsError, match=(
                "timestamp columns hook precedes the conversion of the values"
            )):
                spark.createDataFrame(frame, schema)
            assert calls == ["columns iterator"]
            assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
            assert not state["uploads"]
            assert not state["relations"]
            assert not state["input_tables"]
        else:
            spark.createDataFrame(frame, schema).toArrow()
            expected = values.copy()
            expected[0], expected[-1] = 7000000, None
            _assert_timestamp_columns_input(state, expected, schema)
            assert calls == ["columns iterator"]


@pytest.mark.parametrize("unit", ["ns", "us"])
def test_timestamp_rebound_index_values_raise_in_pyspark_conversion(protocol, monkeypatch, unit):
    # As for int64 columns, PySpark's conversion raises the error of the labels' storage.
    _, frame, schema = _timestamp_columns_fixture(unit)
    columns, original = frame.columns, pd.Index._values.fget
    calls = []

    def values(index):
        if index is columns:
            calls.append("_values")
            raise RuntimeError("columns backing hook precedes the conversion of timestamps")
        return original(index)

    monkeypatch.setattr(pd.Index, "_values", property(values))
    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(RuntimeError,
                           match="columns backing hook precedes the conversion of timestamps"):
            spark.createDataFrame(frame, schema)
        assert calls == ["_values"]
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
        assert not state["uploads"]
        assert not state["relations"]


@pytest.mark.parametrize("unit", ["ns", "us"])
def test_timestamp_rebound_index_data_reads_no_more_than_pyspark_conversion(
    protocol, monkeypatch, unit
):
    values, frame, schema = _timestamp_columns_fixture(unit)
    columns, backing = frame.columns, vars(frame.columns)["_data"]
    calls = []
    allowed_reads = None

    class ColumnsDataError(RuntimeError):
        pass

    def data(index):
        if index is columns:
            calls.append("_data")
            if allowed_reads is not None and calls.count("_data") > allowed_reads:
                raise ColumnsDataError("eligibility added a columns storage read")
            return backing
        return vars(index)["_data"]

    def set_data(index, value):
        vars(index)["_data"] = value

    monkeypatch.setattr(pd.Index, "_data", property(data, set_data), raising=False)
    import pyspark.sql._pandas_input as helper

    # The reads of PySpark's conversion of the frame, with both fast paths declined.
    with monkeypatch.context() as baseline:
        baseline.setattr(helper, "prepare_int64_table", lambda *_: None)
        baseline.setattr(helper, "timestamp_columns", lambda *_: None)
        with _arrow_input_service(protocol) as (spark, state):
            spark.createDataFrame(frame, schema).toArrow()
            _assert_timestamp_columns_input(state, values, schema)
            control_table = state["input_tables"][0]
    allowed_reads = calls.count("_data")
    assert allowed_reads > 0
    calls.clear()
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        # The rebound storage declines the fast path without reading the labels more than
        # PySpark's conversion does. pandas caches some of these reads, so a later conversion
        # of the same frame can read them fewer times than the first.
        assert 0 < calls.count("_data") <= allowed_reads
        _assert_timestamp_columns_input(state, values, schema)
        assert state["input_tables"][0].equals(control_table, check_metadata=True)


@pytest.mark.parametrize("unit", ["ns", "us"])
@pytest.mark.parametrize("storage", ["object", "default", "python", "pyarrow", "python_nan", "pyarrow_nan"])
def test_timestamp_canonical_columns_storage_stays_columnar(protocol, unit, storage):
    import json

    from pyspark.sql.pandas.types import to_arrow_schema

    values, frame, schema = _timestamp_columns_fixture(unit)
    if storage == "object":
        frame.columns = pd.Index(["original"], dtype=object)
    elif storage != "default":
        kind = storage.removesuffix("_nan")
        dtype = (pd.StringDtype(storage=kind, na_value=np.nan)
                 if storage.endswith("_nan") else pd.StringDtype(storage=kind))
        frame.columns = pd.Index(pd.array(["original"], dtype=dtype))

    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert state["config_requests"]
        assert len(state["input_tables"]) == 1
        table = state["input_tables"][0]
        assert table.num_rows == 10000
        assert table.column(0).cast(pa.int64()).to_pylist() == values
        assert table.column(0).is_null().to_pylist() == [value is None for value in values]
        # As PySpark's conversion of the frame, of `_deduplicate_field_names`, the Arrow schema
        # has no metadata of the fields, which the schema of the relation has.
        plain = T.StructType([T.StructField(f.name, f.dataType, f.nullable) for f in schema])
        assert table.schema.equals(to_arrow_schema(plain, timezone="UTC"), check_metadata=True)
        assert json.loads(state["relations"][0].local_relation.schema) == json.loads(schema.json())


def _assert_float64_ipc(table, columns, schema, *, missing=None):
    assert table.num_rows == len(columns[0])
    assert table.schema.names == schema.names
    assert table.num_columns == len(columns)
    for index, (column, expected, field) in enumerate(zip(table.columns, columns, schema.fields)):
        mask = np.isnan(expected) if missing is None else missing[index]
        assert column.type == pa.float64()
        # Check every IPC validity bit and the exact bits of every non-null value,
        # including negative zero, subnormals, and both infinities.
        assert column.is_null().to_pylist() == mask.tolist()
        actual = column.drop_null().combine_chunks().to_numpy(zero_copy_only=False)
        np.testing.assert_array_equal(actual.view(np.uint64), expected[~mask].view(np.uint64))
        actual_field = table.schema.field(index)
        assert actual_field.nullable == field.nullable
        # The Arrow conversion keeps the field metadata; the fast path's table has none.
        from pyspark.sql.pandas.types import from_arrow_metadata

        if actual_field.metadata is not None:
            assert from_arrow_metadata(actual_field.metadata) == field.metadata



@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("verify", [False, True])
def test_float64_columns_preserve_every_ipc_cell(protocol, monkeypatch, cached, verify):
    finite = np.array([
        0.0, -0.0, 1.5, -2.25, np.finfo(np.float64).max, -np.finfo(np.float64).max,
        np.finfo(np.float64).tiny, -np.finfo(np.float64).tiny,
        np.nextafter(0.0, 1.0), -np.nextafter(0.0, 1.0),
    ], dtype=np.float64)
    first = np.tile(finite, 10000)
    second = np.tile(np.array([np.nan, np.inf, -np.inf, -0.0, 1.5], dtype=np.float64), 20000)
    columns = [first, second, np.full(100000, np.nan, dtype=np.float64)]
    frame = pd.DataFrame(dict(zip(["original", "b", "c"], columns)))
    schema = T.StructType([
        T.StructField("renamed.`double", T.DoubleType(), False, {"source": "float64", "id": 3}),
        T.StructField("nullable", T.DoubleType(), True, {"tag": [1, 2], "label": "é"}),
        T.StructField("all_null", T.DoubleType()),
    ])

    def no_rows(*args, **kwargs):
        raise AssertionError("float64 fast path must not materialize Python rows")

    for method in ("to_numpy", "iterrows", "itertuples"):
        monkeypatch.setattr(pd.DataFrame, method, no_rows)
    monkeypatch.setattr(pd.DataFrame, "values", property(no_rows))
    monkeypatch.setattr(pd.Series, "tolist", no_rows)
    overrides = {"spark.sql.session.localRelationCacheThreshold": "1" if cached else "999999999"}
    with _arrow_input_service(protocol, **overrides) as (spark, state):
        result = spark.createDataFrame(frame, schema, verifySchema=verify)
        assert not result.isLocal()
        result.toArrow()
        assert state["column_names"] == schema.names
        _assert_float64_ipc(state["input_tables"][0], columns, schema)
        # The relation is 2.4 MB of Arrow data: the settings are read, and it is uploaded at the
        # threshold of 1 byte, or sent inline below the default threshold.
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
        assert bool(state["uploads"]) == cached



def test_float64_columnar_ipc_matches_existing_row_conversion(protocol, monkeypatch):
    import json

    import pyspark.sql._pandas_input as helper

    values = np.tile(np.array([np.nan, np.inf, -np.inf, -0.0, 0.0, 1.25]), 20000)
    frame = pd.DataFrame({"original": values})
    schema = T.StructType([T.StructField("renamed", T.DoubleType(), True, {"source": "parity"})])
    with _arrow_input_service(protocol) as (spark, state):
        result = spark.createDataFrame(frame, schema)
        assert not result.isLocal()
        result.toArrow()
        assert state["column_names"] == schema.names
        monkeypatch.setattr(helper, "prepare_float64_table", lambda *args: None)
        result = spark.createDataFrame(frame, schema)
        assert not result.isLocal()
        result.toArrow()
        assert state["column_names"] == schema.names
        assert len(state["input_tables"]) == 2
        for table in state["input_tables"]:
            _assert_float64_ipc(table, [values], schema)
        # The Arrow conversion of the second call carries the field metadata in its table
        # and reads the configuration first, as official; the values and types are the same.
        assert state["input_tables"][0].equals(state["input_tables"][1], check_metadata=False)
        assert state["config_requests"][:1] == [("get", _CREATE_DATAFRAME_KEYS)]
        assert state["uploads"] == 0
        assert [json.loads(relation.local_relation.schema) for relation in state["relations"]] == [
            json.loads(schema.json()), json.loads(schema.json())
        ]



@pytest.mark.parametrize("nullable", [False, True])
def test_float64_pandas_infinity_missing_option_keeps_row_semantics(protocol, nullable):

    try:
        pd.get_option("mode.use_inf_as_na")
    except (KeyError, AttributeError):
        pytest.skip("pandas does not expose mode.use_inf_as_na")
    values = np.tile(np.array([np.inf, -np.inf, np.nan, -0.0, 1.25]), 20000)
    frame = pd.DataFrame({"x": values})
    schema = T.StructType([T.StructField("x", T.DoubleType(), nullable)])
    with pd.option_context("mode.use_inf_as_na", True):
        with _arrow_input_service(protocol) as (spark, state):
            if not nullable:
                # Official: the configuration is read, then the cast fails with a ValueError.
                with pytest.raises(ValueError) as error:
                    spark.createDataFrame(frame, schema)
                assert str(error.value) == "Casting field 'x' with null values to non-nullable"
                assert state["config_requests"][:1] == [
                    ("get", _CREATE_DATAFRAME_KEYS)]
                assert state["uploads"] == 0
                return
            spark.createDataFrame(frame, schema).toArrow()
            _assert_float64_ipc(state["input_tables"][0], [values], schema,
                                missing=[~np.isfinite(values)])



@pytest.mark.parametrize("reason", [
    "small", "empty", "zero_columns", "inferred", "names", "ddl", "float_schema",
    "long_schema", "mixed_schema", "width", "duplicate_input", "duplicate_schema",
    "input_number", "input_string_subclass", "frame_subclass",
    "schema_proxy", "extension", "object", "float32", "float16", "non_native",
    "integer", "mixed", "nonnullable_nan", "late_unsupported",
])
def test_float64_column_guard_preserves_row_fallback(reason):
    from pyspark.sql._pandas_input import prepare_float64_table

    class Frame(pd.DataFrame):
        pass

    class Name(str):
        pass

    class SchemaProxy:
        @property
        def __class__(self):
            return T.StructType

    values = np.arange(100000, dtype=np.float64)
    frame = pd.DataFrame({"x": values})
    schema = T.StructType([T.StructField("x", T.DoubleType())])
    if reason == "small":
        frame = frame.iloc[:99999]
    elif reason == "empty":
        frame = frame.iloc[:0]
    elif reason == "zero_columns":
        frame = frame.iloc[:, :0]
        schema = T.StructType([])
    elif reason in ("inferred", "names", "ddl", "float_schema", "long_schema"):
        schema = {
            "inferred": None, "names": ["x"], "ddl": "x DOUBLE",
            "float_schema": T.StructType([T.StructField("x", T.FloatType())]),
            "long_schema": T.StructType([T.StructField("x", T.LongType())]),
        }[reason]
    elif reason == "mixed_schema":
        frame["y"] = values
        schema.add("y", T.FloatType())
    elif reason == "width":
        schema.add("extra", T.DoubleType())
    elif reason in ("duplicate_input", "duplicate_schema"):
        frame["y"] = values
        schema.add("y", T.DoubleType())
        if reason == "duplicate_input":
            frame.columns = ["x", "x"]
        else:
            schema = T.StructType([T.StructField("x", T.DoubleType())] * 2)
    elif reason == "input_string_subclass":
        frame.columns = pd.Index([Name("x")], dtype=object)
        assert type(frame.columns[0]) is Name
    elif reason == "input_number":
        frame.columns = [1]
    elif reason == "frame_subclass":
        frame = Frame(frame)
    elif reason == "schema_proxy":
        schema = SchemaProxy()
    elif reason == "nonnullable_nan":
        frame.iat[-1, 0] = np.nan
        schema = T.StructType([T.StructField("x", T.DoubleType(), False)])
    elif reason in ("mixed", "late_unsupported"):
        frame["y"] = "text" if reason == "mixed" else values.astype(object)
        schema.add("y", T.StringType() if reason == "mixed" else T.DoubleType())
    else:
        dtype = {"extension": "Float64", "object": object, "float32": "float32",
                 "float16": "float16", "non_native": ">f8", "integer": "int64"}[reason]
        frame = pd.DataFrame({"x": pd.Series(values, dtype=dtype)})
    assert prepare_float64_table(frame, schema) is None



@pytest.mark.parametrize("reason", [
    "small", "inferred", "names", "ddl", "float32", "extension", "object", "frame_subclass",
    "duplicate_input", "duplicate_schema",
])
def test_float64_public_fallback_preserves_every_ipc_cell(protocol, reason):

    class Frame(pd.DataFrame):
        pass

    values = np.tile(np.array([1.5, -0.0, np.inf, -np.inf, np.nan]), 20000)
    if reason == "small":
        values = values[:99999]
    dtype = {"float32": "float32", "extension": "Float64", "object": object}.get(reason)
    frame_type = Frame if reason == "frame_subclass" else pd.DataFrame
    frame = frame_type({"x": pd.Series(values, dtype=dtype)})
    columns = [values]
    expected_schema = T.StructType([T.StructField("x", T.DoubleType())])
    schema = expected_schema
    if reason in ("inferred", "names", "ddl"):
        schema = {"inferred": None, "names": ["x"], "ddl": "x DOUBLE"}[reason]
    elif reason in ("duplicate_input", "duplicate_schema"):
        frame["y"] = values
        columns.append(values)
        expected_schema = T.StructType([
            T.StructField("x", T.DoubleType()),
            T.StructField("x" if reason == "duplicate_schema" else "y", T.DoubleType()),
        ])
        schema = expected_schema
        if reason == "duplicate_input":
            frame.columns = ["x", "x"]
        else:
            # As PySpark's conversion of the frame, the table is of the names of
            # `_deduplicate_field_names` of the schema, of the top level too.
            expected_schema = T.StructType([
                T.StructField("x_0", T.DoubleType()), T.StructField("x_1", T.DoubleType())])
    if reason in ("inferred", "names"):
        # As PySpark Connect's, the Arrow conversion of a frame without a schema names its
        # columns _0, _1, ..., which toDF names.
        expected_schema = T.StructType([T.StructField("_0", T.DoubleType())])
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        # The 800 KB of Arrow data are at least 64 KiB, so the settings are read first, as
        # the Arrow importer reads them, whichever path converts the frame.
        assert state["config_requests"][:1] == [("get", _CREATE_DATAFRAME_KEYS)]
        _assert_float64_ipc(state["input_tables"][0], columns, expected_schema)



@pytest.mark.parametrize("verify", [False, True])
def test_float64_nonnullable_nan_reads_config_before_cast_error(protocol, verify):
    frame = pd.DataFrame({"x": np.full(100000, np.nan, dtype=np.float64)})
    schema = T.StructType([T.StructField("x", T.DoubleType(), False)])
    with _arrow_input_service(protocol) as (spark, state):
        state["config_error"] = grpc.StatusCode.INVALID_ARGUMENT
        # Official reads the configuration first, so its error precedes the cast.
        with pytest.raises(Exception, match="configuration read failed"):
            spark.createDataFrame(frame, schema, verifySchema=verify)
        assert state["config_requests"][0] == ("get", _CREATE_DATAFRAME_KEYS)
        assert state["relations"] == []
        assert state["uploads"] == 0
        # Then the cast fails as official does, with a ValueError.
        with pytest.raises(ValueError) as error:
            spark.createDataFrame(frame, schema, verifySchema=verify)
        assert str(error.value) == "Casting field 'x' with null values to non-nullable"
        assert state["relations"] == []



_FLOAT64_VALUES = [1.5, -0.0, float("inf"), float("-inf"), None] * 20000


def _float64_relation(metadata):
    return {"fields": [{"metadata": metadata, "name": "x", "nullable": True, "type": "double"}],
            "type": "struct"}


def _float64_converts(values=_FLOAT64_VALUES, arrow_type="double", name="x", metadata=None,
                      relation_metadata=None, error=None):
    """Of NaN as null, as PySpark's conversion takes it of a float column (`isnull`)."""
    return _guard_converts([(name, arrow_type, True, values, metadata)],
                           _float64_relation(relation_metadata or {}), error)


# PySpark's outcomes (see `_INTEGER_DESCRIPTORS`), and the drop-in's where they differ.
_FLOAT64_DESCRIPTORS = {
    ("StructType", "fields"): _guard_raises(*_no_setter("StructType")),
    ("StructType", "names"): _guard_raises(*_no_setter("StructType")),
    ("StructType", "__iter__"): _guard_raises(*_rename_mismatch("x")),
    ("StructField", "name"): _guard_raises(*_no_setter("StructField")),
    ("StructField", "dataType"): _guard_raises(*_no_setter("StructField")),
    ("StructField", "nullable"): _guard_raises(*_no_setter("StructField")),
    ("StructField", "metadata"): _guard_raises(*_no_setter("StructField")),
    ("StructType", "alias"): _float64_converts(
        relation_metadata={"source": "native"},
        error=["_to_table", "builtins.AssertionError", ""]),
}
_FLOAT64_DESCRIPTORS_OF_THE_DROP_IN = {
    ("StructType", "fields"): _guard_converts([], {"fields": [], "type": "struct"}),
    ("StructType", "names"): _float64_converts(name="changed",
                                               relation_metadata={"source": "native"}),
    ("StructField", "name"): _guard_raises(*_rename_mismatch("x")),
    ("StructField", "dataType"): _float64_converts(relation_metadata={"source": "native"}),
    ("StructField", "nullable"): _guard_raises(
        "builtins.ValueError", "Casting field 'x' with null values to non-nullable"),
    ("StructField", "metadata"): _float64_converts(
        metadata={"SPARK::metadata::json": '{"changed": true}'},
        relation_metadata={"changed": True}),
    ("StructType", "alias"): _float64_converts(relation_metadata={"source": "native"}),
}


@pytest.mark.parametrize("class_name,attribute", [
    ("StructType", "names"), ("StructType", "fields"), ("StructType", "__iter__"),
    ("StructField", "name"), ("StructField", "dataType"), ("StructField", "nullable"),
    ("StructField", "metadata"),
    ("StructType", "__getattribute__"), ("StructField", "__getattribute__"),
    ("DoubleType", "__getattribute__"), ("StructType", "__class__"),
    ("StructField", "__class__"), ("DoubleType", "__class__"),
    ("StructType", "alias"), ("StructField", "alias"), ("DoubleType", "alias"),
])
@pytest.mark.parametrize("before_import", [False, True])
def test_float64_custom_schema_descriptors_take_pyspark_conversion(class_name, attribute,
                                                                   before_import):
    # A patch of a class or a rebound name declines the fast path of double columns: the frame
    # converts with PySpark's conversion, which reads the patch (see `_integer_guard_process`).
    script = r'''
import importlib.util
import sys
import numpy as np
import pandas as pd
from pyspark.sql import types as T

spec = importlib.util.spec_from_file_location("float64_schema_capture", sys.argv[-1])
capture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capture
spec.loader.exec_module(capture)
values = np.tile(np.array([1.5, -0.0, np.inf, -np.inf, np.nan]), 20000)
frame = pd.DataFrame({"x": values})
schema = T.StructType([T.StructField("x", T.DoubleType(), True, {"source": "native"})])
class_name, attribute, before_import = sys.argv[1:4]
if before_import == "True":
    sys.modules.pop("pyspark.sql._pandas_input", None)
assert ("pyspark.sql._pandas_input" in sys.modules) == (before_import == "False")
cls = getattr(T, class_name)
if attribute == "alias":
    setattr(T, class_name, object)
else:
    if attribute == "__getattribute__":
        replacement = lambda self, name: object.__getattribute__(self, name)
    elif attribute == "__iter__":
        replacement = lambda self: iter([T.StructField("changed", T.DoubleType())])
    elif attribute == "__class__":
        attribute = "__getattribute__"
        replacement = lambda self, name: object if name == "__class__" else (
            object.__getattribute__(self, name))
    else:
        value = {"names": ["changed"], "fields": [], "name": "changed",
                 "dataType": T.DoubleType(), "nullable": False,
                 "metadata": {"changed": True}}[attribute]
        replacement = property(lambda self: value)
    setattr(cls, attribute, replacement)
'''
    outcome = _guard_outcome(script, class_name, attribute, str(before_import))
    _assert_guard(outcome,
                  _FLOAT64_DESCRIPTORS.get((class_name, attribute),
                                           _float64_converts(relation_metadata={"source": "native"})),
                  _FLOAT64_DESCRIPTORS_OF_THE_DROP_IN.get((class_name, attribute)))


# PySpark's outcomes of a name rebound to `DoubleType` (to `object` of `DoubleType`, `StructType`
# and `StructField`): `to_arrow_type` takes the double field of the first class that it matches,
# whose cast of the column PyArrow does, as a safe cast; of a `StructType` of `object`,
# `_deduplicate_field_names` takes the double type for a struct and reads its `names`, and of a
# `StructField` of `object`, it calls `object(...)` of the field.
_FLOAT64_REBOUND = {
    "BooleanType": _float64_converts([True, False, True, True, None] * 20000, "bool"),
    "ByteType": _guard_raises(*_conversion_error("PySparkValueError", "float64", "x", "int8")),
    "ShortType": _guard_raises(*_conversion_error("PySparkValueError", "float64", "x", "int16")),
    "IntegerType": _guard_raises(*_conversion_error("PySparkValueError", "float64", "x",
                                                    "int32")),
    "LongType": _guard_raises(*_conversion_error("PySparkValueError", "float64", "x", "int64")),
    "FloatType": _float64_converts(arrow_type="float"),
    "DoubleType": _float64_converts(),
    "StructType": _guard_raises("builtins.AttributeError",
                                "'DoubleType' object has no attribute 'names'"),
    "StructField": _guard_raises("builtins.TypeError", "object() takes no arguments"),
}


@pytest.mark.parametrize("class_name", [
    "BooleanType", "ByteType", "ShortType", "IntegerType", "LongType", "FloatType",
    "DoubleType", "StructType", "StructField",
])
@pytest.mark.parametrize("converter_loaded", [False, True])
def test_float64_rebound_converter_alias_takes_pyspark_conversion(class_name, converter_loaded):
    script = r'''
import importlib.util
import sys
import numpy as np
import pandas as pd
from pyspark.sql import types as T

spec = importlib.util.spec_from_file_location("float64_alias_capture", sys.argv[-1])
capture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capture
spec.loader.exec_module(capture)
values = np.tile(np.array([1.5, -0.0, np.inf, -np.inf, np.nan]), 20000)
frame = pd.DataFrame({"x": values})
schema = T.StructType([T.StructField("x", T.DoubleType())])
class_name, converter_loaded = sys.argv[1:3]
replacement = object if class_name in ("DoubleType", "StructType", "StructField") else T.DoubleType
if converter_loaded == "True":
    import pyspark.sql.pandas.types as converter
    setattr(converter, class_name, replacement)
else:
    sys.modules.pop("pyspark.sql.pandas.types", None)
    setattr(T, class_name, replacement)
    assert "pyspark.sql.pandas.types" not in sys.modules
'''
    _assert_guard(_guard_outcome(script, class_name, str(converter_loaded)),
                  _FLOAT64_REBOUND[class_name])



def test_float64_rebound_pandas_alias_keeps_subclass_fallback(monkeypatch):
    from pyspark.sql._pandas_input import prepare_float64_table

    class Frame(pd.DataFrame):
        pass

    frame = Frame({"x": np.arange(100000, dtype=np.float64)})
    schema = T.StructType([T.StructField("x", T.DoubleType())])
    monkeypatch.setattr(pd, "DataFrame", Frame)
    assert prepare_float64_table(frame, schema) is None



@pytest.mark.parametrize("class_spoof", [False, True])
def test_float64_custom_extension_dtype_cannot_impersonate_numpy_float64(class_spoof):
    from pandas.api.extensions import ExtensionArray, ExtensionDtype, take
    from pyspark.sql._pandas_input import prepare_float64_table

    arrow_calls = []

    class FloatEqualDtype(ExtensionDtype):
        name = "float64_equal"
        type = np.float64
        kind = "f"
        na_value = np.nan

        @classmethod
        def construct_array_type(cls):
            return DifferentArrowArray

        def __eq__(self, other):
            return (isinstance(other, FloatEqualDtype)
                    or isinstance(other, np.dtype) and other == np.dtype("float64"))

        def __hash__(self):
            return hash(self.name)

    class DifferentArrowArray(ExtensionArray):
        def __init__(self, values):
            self._values = np.asarray(values, dtype=np.float64)

        @property
        def dtype(self):
            return FloatEqualDtype()

        @property
        def nbytes(self):
            return self._values.nbytes

        def __len__(self):
            return len(self._values)

        def __getitem__(self, key):
            result = self._values[key]
            return type(self)(result) if isinstance(result, np.ndarray) else result

        def isna(self):
            return np.isnan(self._values)

        def take(self, indices, allow_fill=False, fill_value=None):
            return type(self)(take(self._values, indices, allow_fill=allow_fill,
                                   fill_value=fill_value))

        def copy(self):
            return type(self)(self._values.copy())

        @classmethod
        def _from_sequence(cls, scalars, *, dtype=None, copy=False):
            values = np.asarray(scalars, dtype=np.float64)
            return cls(values.copy() if copy else values)

        def to_numpy(self, dtype=None, copy=False, na_value=None):
            if dtype is not None:
                return self._values.astype(dtype, copy=copy)
            return self._values.copy() if copy else self._values

        def __arrow_array__(self, type=None):
            arrow_calls.append("arrow")
            return pa.array([999.0] * len(self), type=type)

    frame = pd.DataFrame({"x": pd.Series(DifferentArrowArray(np.ones(100000)), copy=False)})
    schema = T.StructType([T.StructField("x", T.DoubleType())])
    if class_spoof:
        def spoof_class(self, name):
            if name == "__class__":
                return type(np.dtype("float64"))
            return object.__getattribute__(self, name)

        FloatEqualDtype.__getattribute__ = spoof_class
    dtype = frame.dtypes.iloc[0]
    assert type(frame) is pd.DataFrame
    assert type(dtype) is FloatEqualDtype
    assert isinstance(dtype, np.dtype) is class_spoof
    assert dtype == np.dtype("float64")
    assert prepare_float64_table(frame, schema) is None
    assert arrow_calls == []



@pytest.mark.parametrize("method", ["astype", "notna"])
def test_float64_instance_conversion_hooks_are_not_called_as_in_pyspark(protocol, method):
    # A frame whose own notna or astype is customized declines the float64 fast path, and
    # converts as PySpark's conversion does, which never calls them: the values are the
    # frame's, whatever the hooks would change.
    from pyspark.sql._pandas_input import prepare_float64_table

    values = np.arange(100000, dtype=np.float64)
    frame = pd.DataFrame({"x": values})
    schema = T.StructType([T.StructField("x", T.DoubleType())])
    original = getattr(frame, method)
    calls = []

    if method == "astype":
        def customized(*args, **kwargs):
            calls.append("astype")
            converted = original(*args, **kwargs)
            converted.iloc[:, 0] = converted.iloc[:, 0] + 7.0
            return converted
    else:
        def customized():
            calls.append("notna")
            mask = original()
            mask.iloc[::3, 0] = False
            return mask

    setattr(frame, method, customized)
    assert prepare_float64_table(frame, schema) is None
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert calls == []
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
        _assert_float64_ipc(state["input_tables"][0], [values], schema)



def _forget_pandas_input(monkeypatch):
    """Make the client import pyspark.sql._pandas_input again, as on its first use. The import
    also binds the new module to the package's attribute, which removing the entry of sys.modules
    does not restore: both are restored at the end of the test, so that a later test patches the
    module that the client imports."""
    import sys

    import pyspark.sql

    monkeypatch.delitem(sys.modules, "pyspark.sql._pandas_input", raising=False)
    monkeypatch.delattr(pyspark.sql, "_pandas_input", raising=False)


@pytest.mark.parametrize("method", ["astype", "notna", "where"])
@pytest.mark.parametrize("before_import", [False, True])
def test_float64_class_conversion_hooks_are_not_called_as_in_pyspark(
    protocol, monkeypatch, method, before_import
):
    # As for the instance: customized methods of the class, also before the helpers are
    # imported, decline the fast path, and PySpark's conversion never calls them.
    import importlib

    helper_name = "pyspark.sql._pandas_input"
    if before_import:
        _forget_pandas_input(monkeypatch)
    else:
        importlib.import_module(helper_name)
    values = np.arange(100000, dtype=np.float64)
    frame = pd.DataFrame({"x": values})
    schema = T.StructType([T.StructField("x", T.DoubleType())])
    original = getattr(pd.DataFrame, method)
    calls = []

    if method == "notna":
        def customized(self):
            calls.append("notna")
            mask = original(self)
            mask.iloc[::3, 0] = False
            return mask
    else:
        def customized(self, *args, **kwargs):
            calls.append(method)
            converted = original(self, *args, **kwargs)
            converted.iloc[:, 0] = converted.iloc[:, 0] + 7.0
            return converted

    monkeypatch.setattr(pd.DataFrame, method, customized)
    if before_import:
        importlib.import_module(helper_name)
    assert importlib.import_module(helper_name).prepare_float64_table(frame, schema) is None
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert calls == []
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
        _assert_float64_ipc(state["input_tables"][0], [values], schema)



def test_float64_literal_numpy_dtype_custom_backing_takes_pyspark_conversion(protocol):
    # An extension array that claims the NumPy dtype float64 does not select the fast path: as
    # PySpark's conversion, which hands a column of an array with `__arrow_array__` to Arrow as
    # it is, the values are those that the hook gives.
    from pyspark.sql._pandas_input import prepare_float64_table

    calls = []

    class DifferentArrowArray(pd.arrays.FloatingArray):
        def __arrow_array__(self, type=None):
            calls.append("arrow")
            return pa.array(np.full(len(self), 999.0), type=type)

    values = np.tile(np.array([1.5, -0.0, np.inf, -np.inf, np.nan]), 20000)
    array = DifferentArrowArray(values.copy(), np.isnan(values))
    frame = pd.DataFrame({"x": pd.Series(array, copy=False)})
    # Construct extension storage before advertising a native dtype; otherwise
    # pandas legitimately normalizes the constructor input into a NumPy block.
    DifferentArrowArray.dtype = property(lambda self: np.dtype("float64"))
    schema = T.StructType([T.StructField("x", T.DoubleType())])
    assert type(frame) is pd.DataFrame
    assert type(frame["x"]) is pd.Series
    assert type(frame["x"].array) is DifferentArrowArray
    assert type(frame.dtypes.iloc[0]) is type(np.dtype("float64"))
    assert frame.dtypes.iloc[0] == np.dtype("float64")
    assert prepare_float64_table(frame, schema) is None
    assert calls == []
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
        _assert_float64_ipc(state["input_tables"][0], [np.full(len(values), 999.0)], schema)
        assert calls == ["arrow"]



@pytest.mark.parametrize("raises", [False, True])
def test_float64_ndarray_subclass_converts_as_pyspark(protocol, raises):
    # The fast path declines a NumPy subclass behind pandas' concrete wrapper; the frame then
    # converts as PySpark's Arrow conversion does, which reads the raw storage and does not
    # call the subclass's astype, whatever it would do.
    from pyspark.sql._pandas_input import prepare_float64_table

    calls = []

    class Values(np.ndarray):
        def astype(self, *args, **kwargs):
            calls.append("astype")
            if raises:
                raise ValueError("custom ndarray astype")
            return np.ndarray.astype(self, *args, **kwargs) + 7.0

    values = np.arange(100000, dtype=np.float64)
    # Avoid pandas consolidating the subclass storage into a plain ndarray.
    frame = pd.DataFrame({"x": values.view(Values)}, copy=False)
    schema = T.StructType([T.StructField("x", T.DoubleType())])
    assert type(frame["x"].array._ndarray) is Values
    assert prepare_float64_table(frame, schema) is None
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        _assert_float64_ipc(state["input_tables"][0], [values], schema)
        assert calls == []
        assert state["uploads"] == 0


def test_float64_missing_schema_alias_takes_pyspark_conversion(protocol, monkeypatch):
    # A public alias of DoubleType that is missing declines the fast path, without the
    # module's `__getattr__`, and the frame converts as PySpark's conversion does, with the
    # classes that its modules hold.
    values = np.arange(100000, dtype=np.float64)
    frame = pd.DataFrame({"x": values})
    schema = T.StructType([T.StructField("x", T.DoubleType())])
    calls = []

    def lazy_attribute(name):
        if name == "DoubleType":
            calls.append(name)
            raise RuntimeError("schema alias hook must not run in the conversion")
        raise AttributeError(name)

    monkeypatch.delitem(vars(T), "DoubleType")
    monkeypatch.setattr(T, "__getattr__", lazy_attribute, raising=False)
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        _assert_float64_ipc(state["input_tables"][0], [values], schema)
        assert calls == []
        assert state["uploads"] == 0



def test_float64_missing_concrete_numpy_backing_class_keeps_fallback(monkeypatch):
    import pandas.core.arrays.numpy_ as numpy_arrays
    from pyspark.sql._pandas_input import prepare_float64_table

    frame = pd.DataFrame({"x": np.arange(100000, dtype=np.float64)})
    schema = T.StructType([T.StructField("x", T.DoubleType())])
    monkeypatch.delattr(numpy_arrays, "NumpyExtensionArray", raising=False)
    monkeypatch.delattr(numpy_arrays, "PandasArray", raising=False)
    assert prepare_float64_table(frame, schema) is None



@pytest.mark.parametrize("rows", [10000, 99999, 100000])
def test_float64_public_threshold_reads_config_below_100k(protocol, rows):
    import json

    values = np.resize(np.array([np.nan, -0.0, 0.0, np.inf, -np.inf, 1.5]), rows)
    frame = pd.DataFrame({"original": values})
    schema = T.StructType([
        T.StructField("renamed", T.DoubleType(), True, {"boundary": {"rows": rows}}),
    ])
    with _arrow_input_service(
        protocol, **{"spark.sql.session.localRelationCacheThreshold": "999999999"}
    ) as (spark, state):
        if rows >= 100000:
            # At this size the fast path reads the settings, as the Arrow importer does for a
            # relation of at least 64 KiB, and sends the relation inline below the threshold.
            spark.createDataFrame(frame, schema).toArrow()
            _assert_float64_ipc(state["input_tables"][0], [values], schema)
            assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
            assert state["uploads"] == 0
            return
        state["config_error"] = grpc.StatusCode.INVALID_ARGUMENT
        # Official reads the configuration before converting, so its error comes first.
        with pytest.raises(Exception, match="configuration read failed"):
            spark.createDataFrame(frame, schema)
        spark.createDataFrame(frame, schema).toArrow()
        _assert_float64_ipc(state["input_tables"][0], [values], schema)
        assert json.loads(state["relations"][0].local_relation.schema) == json.loads(schema.json())
        assert state["uploads"] == 0
        assert state["config_requests"][0] == ("get", _CREATE_DATAFRAME_KEYS)



def _float64_columns_guard_input(rows=100000):
    finite = np.array([
        0.0, -0.0, 1.5, -2.25, np.finfo(np.float64).max, -np.finfo(np.float64).max,
        np.finfo(np.float64).tiny, -np.finfo(np.float64).tiny,
        np.nextafter(0.0, 1.0), -np.nextafter(0.0, 1.0),
    ], dtype=np.float64)
    values = [
        np.resize(finite, rows),
        np.resize(np.array([np.nan, np.inf, -np.inf, -0.0, 1.5]), rows),
        np.full(rows, np.nan, dtype=np.float64),
    ]
    frame = pd.DataFrame(dict(zip(["original", "b", "c"], values)))
    schema = T.StructType([
        T.StructField("renamed.`double", T.DoubleType(), False, {"source": "float64", "id": 3}),
        T.StructField("nullable", T.DoubleType(), True, {"tag": [1, 2]}),
        T.StructField("all_null", T.DoubleType(), True, {"source": "all-null"}),
    ])
    return frame, schema, values


def _assert_float64_columns_guard_payload(state, schema, values):
    import json

    assert len(state["input_tables"]) == 1
    assert len(state["relations"]) == 1
    assert state["column_names"] == schema.names
    _assert_float64_ipc(state["input_tables"][0], values, schema)
    assert json.loads(state["relations"][0].local_relation.schema) == json.loads(schema.json())
    # The relation is at least 64 KiB, so the settings are read, and it is sent inline.
    assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
    assert state["uploads"] == 0



@pytest.mark.parametrize("rows", [10000, 100000])
@pytest.mark.parametrize("location", [
    "subclass", "class", "owner", "backing_subclass_item", "backing_class_iter",
    "backing_class_getitem", "backing_instance_box", "arrow_class_iter",
])
@pytest.mark.parametrize("behavior", ["raise", "mutate"])
def test_float64_columns_iterator_hooks_run_in_pyspark_conversion(
    protocol, monkeypatch, rows, location, behavior
):
    # As for int64 columns: the fast path declines the customized labels without iterating
    # them, and PySpark's conversion iterates them, for the columns of `items()`, before it
    # reads any value. It never calls the frame's notna or astype.
    from pandas.core.base import IndexOpsMixin

    values = np.arange(rows, dtype=np.float64)
    frame = pd.DataFrame({"original": values})
    schema = T.StructType([T.StructField("renamed", T.DoubleType())])
    original_iter = pd.Index.__iter__
    original_notna, original_astype = frame.notna, frame.astype
    calls = []
    invoked = False

    class ColumnsError(RuntimeError):
        pass

    def notna(*args, **kwargs):
        calls.append("notna")
        return original_notna(*args, **kwargs)

    def astype(*args, **kwargs):
        calls.append("astype")
        return original_astype(*args, **kwargs)

    def before_hook():
        nonlocal invoked
        if invoked:
            return
        invoked = True
        calls.append("columns iterator")
        if behavior == "raise":
            raise ColumnsError("columns hook precedes the conversion of the values")
        # The spies of the frame's normalization methods are installed once the conversion
        # iterates the labels, as the values that it reads next are changed: before, the fast
        # path's guards see the canonical methods.
        monkeypatch.setitem(vars(frame), "notna", notna)
        monkeypatch.setitem(vars(frame), "astype", astype)
        frame.iat[0, 0], frame.iat[-1, 0] = -0.0, np.nan

    def iterate(index):
        if index is columns:
            before_hook()
        return original_iter(index)

    if location == "subclass":
        class Columns(pd.Index):
            pass
        monkeypatch.setattr(Columns, "__iter__", iterate)
        labels = pd.Index(["original"], dtype=object)
        frame.columns = Columns._simple_new(labels._values, name=None)
        assert type(frame.columns) is Columns
    elif location == "backing_subclass_item":
        # IndexOpsMixin iteration uses ndarray.item, not ndarray.__iter__.
        class Names(np.ndarray):
            def item(self, *args):
                before_hook()
                return super().item(*args)
        backing = np.asarray(["original"], dtype=object).view(Names)
        frame.columns = pd.Index._simple_new(backing, name=None)
        assert type(frame.columns._values) is Names
    elif location.startswith(("backing_", "arrow_")):
        storage = "pyarrow" if location.startswith("arrow_") else "python"
        frame.columns = pd.Index(pd.array(["original"], dtype=pd.StringDtype(storage=storage)))
        backing = frame.columns._values
        if location.endswith("class_iter"):
            original = type(backing).__iter__

            def backing_iter(array):
                if array is backing:
                    before_hook()
                return original(array)

            monkeypatch.setattr(type(backing), "__iter__", backing_iter)
        elif location.endswith("class_getitem"):
            original = type(backing).__getitem__

            def backing_getitem(array, key):
                if array is backing:
                    before_hook()
                return original(array, key)

            monkeypatch.setattr(type(backing), "__getitem__", backing_getitem)
        else:
            original = backing._box_func

            def box(value):
                before_hook()
                return original(value)

            monkeypatch.setitem(vars(backing), "_box_func", box)
    else:
        monkeypatch.setattr(pd.Index if location == "class" else IndexOpsMixin,
                            "__iter__", iterate)
    columns = frame.columns

    with _arrow_input_service(protocol) as (spark, state):
        if behavior == "raise":
            with pytest.raises(ColumnsError,
                               match="columns hook precedes the conversion of the values"):
                spark.createDataFrame(frame, schema)
            assert calls == ["columns iterator"]
            assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
            assert state["input_tables"] == []
            assert state["uploads"] == 0
        else:
            spark.createDataFrame(frame, schema).toArrow()
            expected = values.copy()
            expected[0], expected[-1] = -0.0, np.nan
            _assert_float64_columns_guard_payload(state, schema, [expected])
            assert calls == ["columns iterator"]



@pytest.mark.parametrize("rows", [10000, 100000])
def test_float64_rebound_index_values_raise_in_pyspark_conversion(protocol, monkeypatch, rows):
    # As for int64 columns, PySpark's conversion raises the error of the labels' storage.
    frame = pd.DataFrame({"original": np.arange(rows, dtype=np.float64)})
    schema = T.StructType([T.StructField("renamed", T.DoubleType())])
    columns = frame.columns
    original = pd.Index._values.fget
    calls = []

    def values(index):
        if index is columns:
            calls.append("_values")
            raise RuntimeError("columns backing hook precedes the conversion of the values")
        return original(index)

    monkeypatch.setattr(pd.Index, "_values", property(values))
    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(RuntimeError,
                           match="columns backing hook precedes the conversion of the values"):
            spark.createDataFrame(frame, schema)
        assert calls == ["_values"]
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
        assert state["input_tables"] == []
        assert state["uploads"] == 0



@pytest.mark.parametrize("rows", [10000, 100000])
def test_float64_rebound_index_data_reads_no_more_than_pyspark_conversion(
    protocol, monkeypatch, rows
):
    import json

    frame, schema, values = _float64_columns_guard_input(rows)
    columns, backing = frame.columns, vars(frame.columns)["_data"]
    calls = []
    allowed_reads = None

    class ColumnsDataError(RuntimeError):
        pass

    def data(index):
        if index is columns:
            calls.append("_data")
            if allowed_reads is not None and calls.count("_data") > allowed_reads:
                raise ColumnsDataError("eligibility added a columns storage read")
            return backing
        return vars(index)["_data"]

    def set_data(index, value):
        vars(index)["_data"] = value

    monkeypatch.setattr(pd.Index, "_data", property(data, set_data), raising=False)
    import pyspark.sql._pandas_input as helper

    # The 2.4 MB of columns stay inline, above the default threshold of 1 MiB.
    with _arrow_input_service(protocol, **{
        "spark.sql.session.localRelationCacheThreshold": str(1 << 30),
    }) as (spark, state):
        # The reads of PySpark's conversion of the frame, with the fast path declined.
        with monkeypatch.context() as baseline:
            baseline.setattr(helper, "prepare_float64_input", lambda *_: None)
            spark.createDataFrame(frame, schema).toArrow()
        allowed_reads = calls.count("_data")
        assert allowed_reads > 0
        calls.clear()
        spark.createDataFrame(frame, schema).toArrow()
        # The rebound storage declines the fast path without reading the labels more than
        # PySpark's conversion does. pandas caches some of these reads, so a later conversion
        # of the same frame can read them fewer times than the first.
        assert 0 < calls.count("_data") <= allowed_reads
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)] * 2
        assert state["uploads"] == 0
        assert len(state["input_tables"]) == 2
        assert len(state["relations"]) == 2
        assert state["input_tables"][1].equals(state["input_tables"][0], check_metadata=True)
        for table, relation in zip(state["input_tables"], state["relations"]):
            _assert_float64_ipc(table, values, schema)
            assert json.loads(relation.local_relation.schema) == json.loads(schema.json())


@pytest.mark.parametrize("storage", [
    "object", "default", "python", "pyarrow", "python_nan", "pyarrow_nan",
])
def test_float64_canonical_columns_storage_stays_columnar(protocol, monkeypatch, storage):
    frame, schema, values = _float64_columns_guard_input()
    labels = ["original", "b", "c"]
    if storage == "object":
        frame.columns = pd.Index(labels, dtype=object)
    elif storage != "default":
        kind = storage.removesuffix("_nan")
        label_dtype = (pd.StringDtype(storage=kind, na_value=np.nan)
                       if storage.endswith("_nan") else pd.StringDtype(storage=kind))
        frame.columns = pd.Index(pd.array(labels, dtype=label_dtype))

    def no_rows(*args, **kwargs):
        raise AssertionError("ordinary float64 columns must retain columnar construction")

    monkeypatch.setattr(pd.DataFrame, "values", property(no_rows))
    # The 2.4 MB of columns stay inline, above the default threshold of 1 MiB.
    with _arrow_input_service(protocol, **{
        "spark.sql.session.localRelationCacheThreshold": str(1 << 30),
    }) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        _assert_float64_columns_guard_payload(state, schema, values)


@pytest.mark.parametrize("location", [
    "subclass", "class", "owner", "backing_subclass_item", "backing_class_iter",
    "backing_class_getitem", "backing_instance_box", "arrow_class_iter",
])
@pytest.mark.parametrize("behavior", ["raise", "mutate"])
def test_string_columns_iterator_hooks_run_in_pyspark_conversion(
    protocol, monkeypatch, location, behavior
):
    # As for int64 columns: the fast path declines the customized labels without iterating
    # them, and PySpark's conversion iterates them, for the columns of `items()`, before it
    # reads any value. It never calls the frame's notna or astype.
    from pandas.core.base import IndexOpsMixin

    values = ["plain"] * 10000
    frame = pd.DataFrame({"original": values}, dtype=object)
    schema = T.StructType([T.StructField("renamed", T.StringType())])
    original_iter = pd.Index.__iter__
    original_notna, original_astype = frame.notna, frame.astype
    calls = []
    invoked = False

    class ColumnsError(RuntimeError):
        pass

    def notna(*args, **kwargs):
        calls.append("notna")
        return original_notna(*args, **kwargs)

    def astype(*args, **kwargs):
        calls.append("astype")
        return original_astype(*args, **kwargs)

    def before_hook():
        nonlocal invoked
        if invoked:
            return
        invoked = True
        calls.append("columns iterator")
        if behavior == "raise":
            raise ColumnsError("columns hook precedes the conversion of the values")
        # The spies of the frame's normalization methods are installed once the conversion
        # iterates the labels, as the values that it reads next are changed: before, the fast
        # path's guards see the canonical methods.
        monkeypatch.setitem(vars(frame), "notna", notna)
        monkeypatch.setitem(vars(frame), "astype", astype)
        frame.iat[0, 0], frame.iat[-1, 0] = "changed", None

    def iterate(index):
        if index is columns:
            before_hook()
        return original_iter(index)

    if location == "subclass":
        class Columns(pd.Index):
            pass
        monkeypatch.setattr(Columns, "__iter__", iterate)
        labels = pd.Index(["original"], dtype=object)
        frame.columns = Columns._simple_new(labels._values, name=None)
        assert type(frame.columns) is Columns
    elif location == "backing_subclass_item":
        # IndexOpsMixin iteration uses ndarray.item, not ndarray.__iter__.
        class Names(np.ndarray):
            def item(self, *args):
                before_hook()
                return super().item(*args)
        backing = np.asarray(["original"], dtype=object).view(Names)
        frame.columns = pd.Index._simple_new(backing, name=None)
        assert type(frame.columns._values) is Names
    elif location.startswith(("backing_", "arrow_")):
        storage = "pyarrow" if location.startswith("arrow_") else "python"
        frame.columns = pd.Index(pd.array(["original"], dtype=pd.StringDtype(storage=storage)))
        backing = frame.columns._values
        if location.endswith("class_iter"):
            original = type(backing).__iter__

            def backing_iter(array):
                if array is backing:
                    before_hook()
                return original(array)

            monkeypatch.setattr(type(backing), "__iter__", backing_iter)
        elif location.endswith("class_getitem"):
            original = type(backing).__getitem__

            def backing_getitem(array, key):
                if array is backing:
                    before_hook()
                return original(array, key)

            monkeypatch.setattr(type(backing), "__getitem__", backing_getitem)
        else:
            original = backing._box_func

            def box(value):
                before_hook()
                return original(value)

            monkeypatch.setitem(vars(backing), "_box_func", box)
    else:
        monkeypatch.setattr(pd.Index if location == "class" else IndexOpsMixin,
                            "__iter__", iterate)
    columns = frame.columns

    with _arrow_input_service(protocol) as (spark, state):
        if behavior == "raise":
            with pytest.raises(ColumnsError,
                               match="columns hook precedes the conversion of the values"):
                spark.createDataFrame(frame, schema)
            assert calls == ["columns iterator"]
            assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
            assert state["input_tables"] == []
            assert state["uploads"] == 0
        else:
            spark.createDataFrame(frame, schema).toArrow()
            expected = values.copy()
            expected[0], expected[-1] = "changed", None
            import json

            table = state["input_tables"][0]
            assert table.schema.names == ["renamed"]
            assert table.column(0).type == pa.string()
            assert table.column(0).to_pylist() == expected
            assert table.column(0).is_null().to_pylist() == [False] * 9999 + [True]
            assert table.schema.equals(pa.schema([pa.field("renamed", pa.string())]),
                                       check_metadata=True)
            assert json.loads(state["relations"][0].local_relation.schema) == json.loads(schema.json())
            assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
            assert state["uploads"] == 0
            assert calls == ["columns iterator"]



def test_string_rebound_index_values_raise_in_pyspark_conversion(protocol, monkeypatch):
    # As for int64 columns, PySpark's conversion raises the error of the labels' storage.
    frame = pd.DataFrame({"original": ["plain"] * 10000}, dtype=object)
    schema = T.StructType([T.StructField("renamed", T.StringType())])
    columns = frame.columns
    original = pd.Index._values.fget
    calls = []

    def values(index):
        if index is columns:
            calls.append("_values")
            raise RuntimeError("columns backing hook precedes the conversion of the values")
        return original(index)

    monkeypatch.setattr(pd.Index, "_values", property(values))
    assert calls == []
    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(RuntimeError,
                           match="columns backing hook precedes the conversion of the values"):
            spark.createDataFrame(frame, schema)
        assert calls == ["_values"]
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
        assert state["input_tables"] == []



@pytest.mark.parametrize("nullable", [False, True])
def test_string_rebound_index_data_reads_no_more_than_pyspark_conversion(
    protocol, monkeypatch, nullable
):
    import json

    values = ["plain", "é", "漢字", ""] * 2500
    if nullable:
        values[1] = None
    frame = pd.DataFrame({"original": values}, dtype=object)
    schema = T.StructType([
        T.StructField("renamed", T.StringType(), nullable, {"source": "native"})
    ])
    columns, backing = frame.columns, vars(frame.columns)["_data"]
    calls = []
    allowed_reads = None

    class ColumnsDataError(RuntimeError):
        pass

    def data(index):
        if index is columns:
            calls.append("_data")
            if allowed_reads is not None and calls.count("_data") > allowed_reads:
                raise ColumnsDataError("eligibility added a columns storage read")
            return backing
        return vars(index)["_data"]

    def set_data(index, value):
        # Keep ordinary storage initialization for indexes created by pandas.
        vars(index)["_data"] = value

    monkeypatch.setattr(pd.Index, "_data", property(data, set_data), raising=False)
    import pyspark.sql._pandas_input as helper

    with _arrow_input_service(protocol) as (spark, state):
        # The reads of PySpark's conversion of the frame, with the fast path declined.
        with monkeypatch.context() as baseline:
            baseline.setattr(helper, "prepare_string_input", lambda *_: None)
            spark.createDataFrame(frame, schema).toArrow()
        allowed_reads = calls.count("_data")
        assert allowed_reads > 0
        calls.clear()
        spark.createDataFrame(frame, schema).toArrow()
        # The rebound storage declines the fast path without reading the labels more than
        # PySpark's conversion does. pandas caches some of these reads, so a later conversion
        # of the same frame can read them fewer times than the first.
        assert 0 < calls.count("_data") <= allowed_reads
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)] * 2
        assert state["uploads"] == 0
        assert len(state["input_tables"]) == 2
        assert len(state["relations"]) == 2
        table = state["input_tables"][1]
        assert table.equals(state["input_tables"][0], check_metadata=True)
        assert table.column(0).to_pylist() == values
        assert table.column(0).is_null().to_pylist() == [value is None for value in values]
        assert table.schema.equals(
            pa.schema([pa.field("renamed", pa.string(), nullable=nullable)]),
            check_metadata=True,
        )
        assert json.loads(state["relations"][1].local_relation.schema) == {
            "type": "struct", "fields": [
                {"name": "renamed", "type": "string", "nullable": nullable,
                 "metadata": {"source": "native"}}
            ],
        }


@pytest.mark.parametrize("kind", [
    "object", "default", "python", "pyarrow", "python_numpy", "pyarrow_numpy",
])
def test_string_canonical_column_storage_stays_eligible(kind):
    from pyspark.sql._pandas_input import _has_canonical_columns, prepare_string_input

    frame = pd.DataFrame({"original": ["plain"] * 10000}, dtype=object)
    schema = T.StructType([T.StructField("renamed", T.StringType())])
    if kind == "object":
        frame.columns = pd.Index(np.asarray(["original"], dtype=object), dtype=object)
    elif kind == "default":
        frame.columns = pd.Index(["original"])
    elif kind.endswith("_numpy"):
        from pandas.core.arrays import string_ as strings
        from pandas.core.arrays import string_arrow as arrow_strings

        module = strings if kind.startswith("python") else arrow_strings
        name = ("StringArrayNumpySemantics" if kind.startswith("python")
                else "ArrowStringArrayNumpySemantics")
        cls = getattr(module, name, None)
        if cls is None:
            pytest.skip("older infer-string storage class is absent")
        frame.columns = pd.Index._simple_new(cls._from_sequence(["original"]), name=None)
    else:
        frame.columns = pd.Index(pd.array(["original"], dtype=pd.StringDtype(storage=kind)))
    assert _has_canonical_columns(frame.columns)
    assert type(prepare_string_input(frame, schema)) is bytes



@pytest.mark.parametrize("missing", ["module", "owner", "base"])
def test_string_absent_private_columns_storage_keeps_row_fallback(monkeypatch, missing):
    import builtins

    from pandas.core.arrays import numpy_ as numpy_module
    from pandas.core.arrays import string_ as string_module
    from pyspark.sql._pandas_input import _has_canonical_columns

    columns = pd.Index(pd.array(["original"], dtype=pd.StringDtype(storage="python")))
    if missing == "module":
        original = builtins.__import__

        def without_arrow(name, *args, **kwargs):
            if name == "pandas.core.arrays.arrow":
                raise ImportError("private storage module is absent")
            return original(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", without_arrow)
    elif missing == "owner":
        monkeypatch.delattr(numpy_module, "NumpyExtensionArray")
    else:
        monkeypatch.delattr(string_module, "BaseStringArray")
    assert not _has_canonical_columns(columns)



@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize("large", ["false", "true"])
def test_string_columns_preserve_every_ipc_cell(protocol, monkeypatch, cached, verify, large):
    import json

    from pyspark.sql.pandas.types import from_arrow_metadata

    strings = ["", "plain", "é", "e\u0301", "\x00embedded\x00", "漢字", "🙂",
               "\U0010ffff", "\uffff", "\u202eright-to-left"] * 1000
    nullable = [None if i % 7 == 0 else value for i, value in enumerate(reversed(strings))]
    expected = [strings, nullable, [None] * len(strings)]
    frame = pd.DataFrame(dict(zip(["original", "b", "c"], expected)), dtype=object)
    schema = T.StructType([
        T.StructField("renamed.`text", T.StringType(), False, {"source": "string", "id": 3}),
        T.StructField("nullable", T.StringType(), True, {"tag": [1, 2]}),
        T.StructField("all_null", T.StringType()),
    ])

    def no_rows(*args, **kwargs):
        raise AssertionError("string fast path must not materialize Python rows")

    for method in ("to_numpy", "iterrows", "itertuples"):
        monkeypatch.setattr(pd.DataFrame, method, no_rows)
    monkeypatch.setattr(pd.DataFrame, "values", property(no_rows))
    monkeypatch.setattr(pd.Series, "tolist", no_rows)
    overrides = {
        "spark.sql.session.localRelationCacheThreshold": "1" if cached else "999999999",
        "spark.sql.execution.arrow.useLargeVarTypes": large,
        "spark.sql.session.localRelationChunkSizeRows": "997",
    }
    with _arrow_input_service(protocol, **overrides) as (spark, state):
        result = spark.createDataFrame(frame, schema, verifySchema=verify)
        assert not result.isLocal()
        result.toArrow()
        assert state["column_names"] == schema.names
        table = state["input_tables"][0]
        assert table.num_rows == len(strings)
        assert table.schema.names == schema.names
        for column, values, field in zip(table.columns, expected, schema.fields):
            # As PySpark's conversion of a schema's strings, large under useLargeVarTypes.
            assert column.type == (pa.large_string() if large == "true" else pa.string())
            assert column.to_pylist() == values
            actual_field = table.schema.field(field.name)
            assert actual_field.nullable == field.nullable
            assert from_arrow_metadata(actual_field.metadata) is None
            spark_field = json.loads(state["spark_schema"])["fields"][
                schema.names.index(field.name)]
            assert spark_field["metadata"] == field.metadata
        # The relation is at least 64 KiB, so the settings are read. It is uploaded at the
        # threshold of 1 byte, and sent inline below the default threshold.
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
        assert bool(state["uploads"]) == cached
        assert state["relations"][0].HasField("chunked_cached_local_relation") == cached



@pytest.mark.parametrize("reason", [
    "small", "empty", "inferred", "names", "ddl", "cast", "atomic", "mixed_schema",
    "width", "duplicate_input", "duplicate_schema",
    "input_number", "datetime", "date", "nan", "pd_na", "nat", "mixed", "bytes",
    "nonnullable_null", "frame_subclass", "string_subclass", "string_dtype",
    "arrow_string_dtype", "late_unsupported", "surrogate", "late_surrogate", "collation",
])
def test_string_column_guard_handles_supported_and_unsupported_inputs(reason):
    from pyspark.sql._pandas_input import prepare_string_input

    class Frame(pd.DataFrame):
        pass

    class Text(str):
        def __str__(self):
            raise AssertionError("guard must not call conversion hooks")

    class Unsupported:
        def __str__(self):
            raise AssertionError("guard must not stringify unsupported objects")

    values = ["plain"] * 10000
    frame = pd.DataFrame({"text": values}, dtype=object)
    schema = T.StructType([T.StructField("text", T.StringType())])
    if reason == "small":
        frame = frame.iloc[:9999]
    elif reason == "empty":
        frame = frame.iloc[:0]
    elif reason in ("inferred", "names", "ddl", "cast", "atomic"):
        schema = {"inferred": None, "names": ["text"], "ddl": "text STRING",
                  "cast": T.StructType([T.StructField("text", T.LongType())]),
                  "atomic": T.StringType()}[reason]
    elif reason == "mixed_schema":
        frame["long"] = np.arange(10000)
        schema.add("long", T.LongType())
    elif reason == "width":
        schema.add("extra", T.StringType())
    elif reason in ("duplicate_input", "duplicate_schema"):
        frame["other"] = values
        schema.add("other", T.StringType())
        if reason == "duplicate_input":
            frame.columns = ["text", "text"]
        else:
            schema = T.StructType([T.StructField("text", T.StringType())] * 2)
    elif reason == "input_number":
        frame.columns = [1]
    elif reason == "frame_subclass":
        frame = Frame(frame)
    elif reason in ("string_dtype", "arrow_string_dtype"):
        frame = pd.DataFrame({"text": pd.Series(
            values, dtype=pd.StringDtype(storage="pyarrow" if reason == "arrow_string_dtype"
                                        else "python"))})
    elif reason == "collation":
        schema = T.StructType([T.StructField("text", T.StringType("UTF8_LCASE"))])
    else:
        value = {
            "datetime": datetime(2020, 1, 1), "date": date(2020, 1, 1),
            "nan": float("nan"), "pd_na": pd.NA, "nat": pd.NaT, "mixed": 42,
            "bytes": b"plain", "nonnullable_null": None, "string_subclass": Text("plain"),
            "late_unsupported": Unsupported(), "surrogate": "\ud800", "late_surrogate": "\udfff",
        }[reason]
        if reason == "nonnullable_null":
            schema = T.StructType([T.StructField("text", T.StringType(), False)])
        if reason in ("late_unsupported", "late_surrogate"):
            # pandas 3 infers StringDtype for a newly assigned list column.
            # Keep these unsupported-object fixtures on concrete object storage.
            frame["other"] = pd.Series(values, dtype=object)
            schema.add("other", T.StringType())
            backing = frame["other"].array._ndarray
            assert type(backing) is np.ndarray and backing.dtype == np.dtype("object")
        frame.iat[-1, -1] = value
        if reason in ("late_unsupported", "late_surrogate"):
            assert frame.iat[-1, -1] is value
    result = prepare_string_input(frame, schema)
    if reason == "arrow_string_dtype":
        # Arrow string storage has a path of its own, which returns an Arrow table.
        assert type(result) is pa.Table
    else:
        assert result is None


# Columns kept in pandas' Arrow string storage reach the importer as Arrow, without Python rows.

def _arrow_string_column(texts, storage):
    if storage == "numpy_semantics":
        try:
            dtype = pd.StringDtype("pyarrow", na_value=float("nan"))
        except TypeError:  # pandas 2.2
            dtype = pd.StringDtype("pyarrow_numpy")
        return pd.array(texts, dtype=dtype)
    if storage == "chunked":
        return pd.arrays.ArrowStringArray(pa.chunked_array(
            [pa.array(texts[:7], pa.large_string()), pa.array(texts[7:], pa.large_string())]))
    # The sliced case uses this storage, then slices the constructed frame below.
    return pd.array(texts, dtype="string[pyarrow]")


def _force_pyspark_pandas_conversion(monkeypatch):
    """Decline the fast paths of plain frames, so that every frame converts with PySpark's
    conversion of each column to Arrow, the table of which a fast path makes too."""
    import importlib

    # Earlier import-hook tests can leave the package attribute pointing at a
    # different module object from the one imported by the native binding.
    helper = importlib.import_module("pyspark.sql._pandas_input")
    for name in ("prepare_int64_table", "timestamp_columns", "prepare_float64_input",
                 "prepare_string_input", "prepare_date_table"):
        monkeypatch.setattr(helper, name, lambda *_: None)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("storage", ["pyarrow", "numpy_semantics", "chunked", "sliced"])
def test_arrow_string_columns_preserve_every_ipc_cell(protocol, monkeypatch, cached, storage):
    from pyspark.sql.pandas.types import from_arrow_metadata

    texts = ["", "a", "\x00", "café", "中文", "\U0001f600", "é", None] * 2000
    other = [None if i % 3 == 0 else str(i) for i in range(len(texts))]
    frame = pd.DataFrame({
        "original": _arrow_string_column(texts, storage),
        "b": _arrow_string_column(other, storage),
        "c": _arrow_string_column([None] * len(texts), storage),
    })
    expected = [texts, other, [None] * len(texts)]
    if storage == "sliced":
        frame = frame.iloc[3:-5]
        expected = [values[3:-5] for values in expected]
    schema = T.StructType([
        T.StructField("renamed.`text", T.StringType(), True, {"source": "string"}),
        T.StructField("other", T.StringType()),
        T.StructField("all_null", T.StringType()),
    ])

    def no_rows(*args, **kwargs):
        raise AssertionError("Arrow string storage must not materialize Python rows")

    for method in ("to_numpy", "iterrows", "itertuples"):
        monkeypatch.setattr(pd.DataFrame, method, no_rows)
    monkeypatch.setattr(pd.DataFrame, "values", property(no_rows))
    monkeypatch.setattr(pd.Series, "tolist", no_rows)
    overrides = {"spark.sql.session.localRelationCacheThreshold": "1" if cached else "999999999"}
    with _arrow_input_service(protocol, **overrides) as (spark, state):
        df = spark.createDataFrame(frame, schema)
        df.toArrow()
        assert state["column_names"] == schema.names
        table = state["input_tables"][0]
        assert table.schema.names == schema.names
        for column, values, field in zip(table.columns, expected, schema.fields):
            assert column.type == pa.string()
            assert column.to_pylist() == values
            actual_field = table.schema.field(field.name)
            assert actual_field.nullable == field.nullable
            # As PySpark's conversion of the frame, of no metadata of the field in Arrow.
            assert from_arrow_metadata(actual_field.metadata) is None
        assert bool(state["uploads"]) == cached


@pytest.mark.parametrize("reason", [
    "python_storage", "mixed", "nonnullable_null", "subclass", "small", "atomic_schema",
])
def test_arrow_string_columns_decline_the_fast_path(reason):
    from pyspark.sql._pandas_input import prepare_string_input

    texts = ["a", None, "b"] * 4000
    columns = {"a": _arrow_string_column(texts, "pyarrow"), "b": _arrow_string_column(texts, "pyarrow")}
    fields = [T.StructField("a", T.StringType()), T.StructField("b", T.StringType())]
    if reason == "python_storage":
        columns["b"] = pd.array(texts, dtype="string[python]")
    elif reason == "mixed":
        columns["b"] = pd.Series(texts, dtype=object)
    elif reason == "nonnullable_null":
        fields[1] = T.StructField("b", T.StringType(), False)
    elif reason == "subclass":
        class Custom(pd.arrays.ArrowStringArray):
            pass

        columns["b"] = Custom(pa.chunked_array([pa.array(texts)]))
    elif reason == "small":
        columns = {name: values[:9999] for name, values in columns.items()}
    elif reason == "atomic_schema":
        fields[1] = T.StructField("b", T.IntegerType())
    frame = pd.DataFrame(columns)
    assert prepare_string_input(frame, T.StructType(fields)) is None


@pytest.mark.parametrize("storage", ["pyarrow", "numpy_semantics"])
def test_arrow_string_columns_match_pyspark_conversion(protocol, monkeypatch, storage):
    import json

    from pyspark.sql.pandas.types import from_arrow_metadata

    texts = ["x" * (i % 5) if i % 4 else None for i in range(12000)]
    frame = pd.DataFrame({"v": _arrow_string_column(texts, storage)})
    schema = T.StructType([T.StructField("v", T.StringType(), True, {"source": "control"})])
    tables = []
    logical_schemas = []
    for control in (False, True):
        with monkeypatch.context() as dispatch:
            if control:
                _force_pyspark_pandas_conversion(dispatch)
            with _arrow_input_service(protocol) as (spark, state):
                spark.createDataFrame(frame, schema).toArrow()
                tables.append(state["input_tables"][0])
                logical_schemas.append(json.loads(state["spark_schema"]))
                # As PySpark's createDataFrame, the settings are read before any conversion.
                assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
    # As PySpark's conversion, the fast path's IPC carries no metadata of the field, which the
    # logical schema of the relation has.
    assert tables[0].equals(tables[1], check_metadata=True)
    assert from_arrow_metadata(tables[0].schema.field(0).metadata) is None
    assert logical_schemas == [json.loads(schema.json())] * 2


@pytest.mark.parametrize("storage", ["pyarrow", "numpy_semantics"])
@pytest.mark.parametrize("layout", ["small", "mixed"])
def test_arrow_string_generic_explicit_schema_stays_columnar(
    protocol, monkeypatch, storage, layout
):
    from pyspark.sql._pandas_input import accepts_pandas_schema, prepare_string_input

    texts = ["", None, "中文", "a\x00b"] * (3 if layout == "small" else 3000)
    frame = pd.DataFrame({"v": _arrow_string_column(texts, storage)})
    fields = [T.StructField("renamed", T.StringType())]
    expected = {"renamed": texts}
    if layout == "mixed":
        frame["number"] = np.arange(len(texts), dtype=np.int64)
        fields.append(T.StructField("number", T.LongType()))
        expected["number"] = list(range(len(texts)))
    schema = T.StructType(fields)
    # These ordinary inputs miss the specialized string route and take PySpark's conversion of
    # an explicit schema.
    assert prepare_string_input(frame, schema) is None
    assert accepts_pandas_schema(schema)

    def no_rows(*args, **kwargs):
        raise AssertionError("generic Arrow strings must not materialize pandas rows")

    monkeypatch.setattr(pd.DataFrame, "values", property(no_rows))
    monkeypatch.setattr(pd.Series, "tolist", no_rows)
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert state["config_requests"]
        assert state["input_tables"][0].to_pydict() == expected
        assert state["input_tables"][0].schema.field(0).type == pa.string()


@pytest.mark.parametrize("scope", ["public", "converter", "captured"])
@pytest.mark.parametrize("class_name", ["BooleanType", "StringType", "StructType"])
def test_arrow_string_rebound_converter_alias_takes_pyspark_conversion(scope, class_name):
    # The rebound name declines the fast path of Arrow strings: the frame converts with PySpark's
    # conversion, of the names as the module of the session and the converter module hold them
    # (see `_integer_guard_process`): of a name rebound to `StringType`, its Arrow type is that of
    # the first class that `to_arrow_type` matches, and of a `StructType`, the converter of the
    # series reads the string type as a struct. A public `StructType` is the one that PySpark's
    # createDataFrame checks its schema against (see `test_date_rebound_converter_aliases_take_
    # pyspark_conversion`).
    script = r'''
import importlib
import importlib.util
import sys
import pandas as pd
from pyspark import _pyspark as native
from pyspark.sql import types as T

_pandas_input = importlib.import_module("pyspark.sql._pandas_input")
spec = importlib.util.spec_from_file_location("arrow_string_alias_capture", sys.argv[-1])
capture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capture
spec.loader.exec_module(capture)
texts = ["original", None] * 5000
frame = pd.DataFrame({"v": pd.array(texts, dtype="string[pyarrow]")})
schema = T.StructType([T.StructField("v", T.StringType(), True, {"source": "native"})])
scope, name = sys.argv[1:3]
replacement = native.BooleanType if name == "StringType" else native.StringType
if scope == "public":
    sys.modules.pop("pyspark.sql.pandas.types", None)
    setattr(T, name, replacement)
else:
    converter = importlib.import_module("pyspark.sql.pandas.types")
    if scope == "captured":
        importer = importlib.import_module("pyspark.sql._arrow_input")
    setattr(converter, name, replacement)
    if scope == "captured":
        sys.modules.pop("pyspark.sql.pandas.types")
        assert importlib.import_module("pyspark.sql.pandas.types") is not converter
        assert importer.to_arrow_schema.__globals__ is vars(converter)
assert _pandas_input.prepare_string_input(frame, schema) is None
'''
    rebound = {
        "BooleanType": _guard_raises(*_conversion_error("PySparkValueError", "string", "v",
                                                        "bool")),
        "StringType": _guard_raises(*_unsupported_type("StringType()")),
        "StructType": _guard_raises("builtins.AttributeError",
                                    "'StringType' object has no attribute 'names'"),
    }
    outcome = _guard_outcome(script, scope, class_name)
    if scope == "captured":
        _assert_guard(outcome, _guard_converts(
            [("v", "string", True, ["original", None] * 5000)],
            {"fields": [{"metadata": {"source": "native"}, "name": "v", "nullable": True,
                         "type": "string"}], "type": "struct"}))
    elif scope == "public" and class_name == "StructType":
        _assert_guard(outcome, _NOT_EXPECTED_STRUCT)
    else:
        _assert_guard(outcome, rebound[class_name])


@pytest.mark.parametrize("storage", ["pyarrow", "numpy_semantics"])
@pytest.mark.parametrize("schema_kind", ["explicit", "inferred", "names"])
@pytest.mark.parametrize("instance_hook", [False, True])
def test_arrow_string_custom_arrow_hook_values_as_in_pyspark(
    protocol, monkeypatch, instance_hook, schema_kind, storage
):
    # A customized `__arrow_array__` of an Arrow string array, of its class or of the instance,
    # declines the fast path, without calling it. PySpark's conversion hands each column of such
    # an array to Arrow, which calls the hook: the values are those that it gives, of each
    # column of the customized class, or of the customized instance only.
    import importlib

    _pandas_input = importlib.import_module("pyspark.sql._pandas_input")

    texts = ["original", None] * 5000
    # The first column authenticates this class before the second instance's
    # override is checked; every instance must still be inspected.
    frame = pd.DataFrame({name: _arrow_string_column(texts, storage)
                          for name in ("first", "v")})
    schema = T.StructType([T.StructField(name, T.StringType()) for name in ("first", "v")])
    input_schema = {"explicit": schema, "inferred": None, "names": ["first", "v"]}[schema_kind]
    array = frame.iloc[:, 1].array
    calls = []
    changed = ["changed"] * len(texts)

    def changed_arrow(*args, **kwargs):
        calls.append("arrow")
        return pa.array(changed, type=pa.string())

    monkeypatch.setattr(array if instance_hook else type(array), "__arrow_array__", changed_arrow)
    assert _pandas_input.prepare_string_input(frame, schema) is None
    assert calls == []
    expected = {"first": texts if instance_hook else changed, "v": changed}
    tables = []
    for control in (False, True):
        calls.clear()
        with monkeypatch.context() as dispatch:
            if control:
                _force_pyspark_pandas_conversion(dispatch)
            with _arrow_input_service(protocol) as (spark, state):
                spark.createDataFrame(frame, input_schema).toArrow()
                table = state["input_tables"][0]
                tables.append(table)
                assert table.rename_columns(state["column_names"]).to_pydict() == expected
                assert calls == ["arrow"] * (1 if instance_hook else 2)
                # As PySpark's createDataFrame, the settings are read before any conversion.
                assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
                assert state["uploads"] == 0
    assert tables[0].equals(tables[1], check_metadata=True)


@pytest.mark.parametrize("schema_kind", ["explicit", "inferred", "names"])
@pytest.mark.parametrize("customization", [
    "mixin_hook", "mixin_alias", "partial_alias", "mixin_storage", "mixin_dtype",
])
def test_arrow_string_numpy_semantics_accessor_converts_as_pyspark(
    protocol, monkeypatch, customization, schema_kind
):
    # pandas 2's NumPy-semantics Arrow string array dispatches its instance lookups through
    # aliases of its module, whose customization declines the fast path without calling them.
    # PySpark's conversion hands the column to Arrow through its `__arrow_array__`, which none
    # of these customizations reaches: the values are the column's, of every pandas.
    import importlib
    import json

    from pandas.core.arrays import _arrow_string_mixins as string_mixins
    from pandas.core.arrays import string_arrow
    from pyspark.sql.pandas.types import from_arrow_schema

    helper = importlib.import_module("pyspark.sql._pandas_input")
    texts = ["original", None] * 5000
    frames = [pd.DataFrame({"v": _arrow_string_column(texts, "numpy_semantics")})
              for _ in range(2)]
    for frame in frames:
        frame.columns = pd.Index(["v"], dtype=object)
    schema = T.StructType([T.StructField("v", T.StringType())])
    input_schema = {"explicit": schema, "inferred": None, "names": ["v"]}[schema_kind]
    array_class = type(frames[0].iloc[:, 0].array)
    dynamic_accessor = (helper._raw_class_attribute(array_class, "__getattribute__")[1]
                        is not object.__getattribute__)
    # These warm-ups populate Block.dtype so mixin_dtype reaches class authentication
    # instead of raising during an earlier frame.dtypes read.
    for frame in frames:
        assert helper.prepare_string_input(frame, schema) is not None
    calls = []

    def changed_arrow(*args, **kwargs):
        calls.append("arrow")
        return pa.array(["changed"] * len(texts), type=pa.string())

    def changed_partial(*args, **kwargs):
        calls.append("partial")
        raise AssertionError("eligibility must not invoke the rebound partial alias")

    def changed_attribute(*args, **kwargs):
        calls.append(customization)
        raise AssertionError("the mixin state callback must not be called")

    if customization == "mixin_hook":
        monkeypatch.setattr(string_mixins.ArrowStringArrayMixin, "__arrow_array__",
                            changed_arrow, raising=False)
    elif customization == "mixin_alias":
        class ChangedMixin:
            __arrow_array__ = changed_arrow

        monkeypatch.setattr(string_arrow, "ArrowStringArrayMixin", ChangedMixin, raising=False)
    elif customization in ("mixin_storage", "mixin_dtype"):
        name = {"mixin_storage": "_storage", "mixin_dtype": "_dtype"}[customization]
        monkeypatch.setattr(string_mixins.ArrowStringArrayMixin, name,
                            changed_attribute, raising=False)
    else:
        monkeypatch.setattr(string_arrow, "partial", changed_partial, raising=False)
    for frame in frames:
        # pandas 2.2 dispatches instance lookups through these mutable aliases.
        # Later layouts use object access and remain unaffected by these changes.
        assert (helper.prepare_string_input(frame, schema) is None) == dynamic_accessor
        assert calls == []
    logical_tables = []
    for control, frame in zip((True, False), frames):
        with monkeypatch.context() as dispatch:
            if control:
                _force_pyspark_pandas_conversion(dispatch)
            with _arrow_input_service(protocol) as (spark, state):
                # The mock returns only a row count, not the uploaded data or resolved schema.
                reply = spark.createDataFrame(frame, input_schema).toArrow()
                assert reply.to_pydict() == {"rows": [len(texts)]}
                assert len(state["input_tables"]) == 1
                uploaded = state["input_tables"][0]
                assert uploaded.num_rows == len(texts)
                assert uploaded.num_columns == 1
                assert uploaded.column(0).to_pylist() == texts
                assert uploaded.column(0).is_null().to_pylist() == [False, True] * 5000
                assert state["column_names"] == ["v"]
                if schema_kind == "explicit":
                    expected_wire = pa.schema([pa.field("v", pa.string())])
                    assert json.loads(state["spark_schema"]) == json.loads(schema.json())
                else:
                    # PySpark's conversion without a struct sends _n fields without a schema;
                    # Spark infers large_string, pandas' storage, as StringType, and toDF
                    # applies the column names above.
                    expected_wire = pa.schema([pa.field("_0", pa.large_string())])
                    assert state["spark_schema"] is None
                assert uploaded.schema.equals(expected_wire, check_metadata=True)
                # As PySpark's createDataFrame, the settings are read before any conversion.
                assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
                assert state["uploads"] == 0
                # Compare logical uploaded data only after checking the exact wire
                # schema and rename plan. This is not a real server schema assertion.
                renamed = uploaded.rename_columns(state["column_names"])
                assert from_arrow_schema(renamed.schema) == schema
                logical_tables.append(renamed.cast(pa.schema([pa.field("v", pa.string())])))
    assert calls == []
    assert len(logical_tables) == 2
    assert logical_tables[0].equals(logical_tables[1], check_metadata=True)


def test_arrow_string_guards_recheck_hooks_between_columns(monkeypatch):
    import importlib

    _pandas_input = importlib.import_module("pyspark.sql._pandas_input")
    texts = ["original", None] * 5000
    frame = pd.DataFrame({name: pd.array(texts, dtype="string[pyarrow]")
                          for name in ("first", "second")})
    schema = T.StructType([T.StructField(name, T.StringType())
                           for name in ("first", "second")])
    array_class = type(frame.iloc[:, 0].array)
    original_iloc = pd.DataFrame.iloc
    lookups = []
    calls = []

    def customized(*args, **kwargs):
        calls.append("__getitem__")
        raise AssertionError("eligibility must not run the customized row hook")

    class RebindingIndexer:
        def __init__(self, indexer):
            self.indexer = indexer

        def __getitem__(self, key):
            column = self.indexer[key]
            lookups.append(key[1])
            if key[1] == 1:
                monkeypatch.setattr(array_class, "__getitem__", customized)
            return column

    def iloc(self):
        indexer = original_iloc.fget(self)
        return RebindingIndexer(indexer) if self is frame else indexer

    # The fast path's guard allows this accessor. It returns ordinary Series while
    # rebinding a hook after the first column, invalidating any per-class cache.
    monkeypatch.setattr(pd.DataFrame, "iloc", property(iloc))
    assert _pandas_input.prepare_string_input(frame, schema) is None
    assert lookups == [0, 1]
    assert calls == []


def test_arrow_string_subclass_guard_does_not_hash_custom_metaclass():
    from pyspark.sql._pandas_input import _arrow_string_class_is_canonical

    calls = []

    class CustomMeta(type(pd.arrays.ArrowStringArray)):
        def __hash__(cls):
            calls.append("class hash")
            raise AssertionError("declining a subclass must not run its hash hook")

    class Custom(pd.arrays.ArrowStringArray, metaclass=CustomMeta):
        pass

    assert not _arrow_string_class_is_canonical(Custom)
    assert calls == []


def test_arrow_string_subclass_guard_does_not_read_custom_mro():
    from pyspark.sql._pandas_input import prepare_string_input

    calls = []
    armed = False

    class CustomMeta(type(pd.arrays.ArrowStringArray)):
        @property
        def __mro__(cls):
            if armed:
                calls.append("class mro")
                raise AssertionError("declining a subclass must not read its custom MRO")
            return type.__dict__["__mro__"].__get__(cls, type(cls))

    class Custom(pd.arrays.ArrowStringArray, metaclass=CustomMeta):
        pass

    # Finish ordinary pandas construction before arming the metaclass callback. The column is
    # long enough for the fast path to reach the classes of its arrays.
    frame = pd.DataFrame({"v": Custom(pa.chunked_array([pa.array(["a", None] * 5000)]))})
    assert type(frame.iloc[:, 0].array) is Custom
    schema = T.StructType([T.StructField("v", T.StringType())])
    armed = True
    assert prepare_string_input(frame, schema) is None
    assert calls == []


def _arrow_string_allocator_process(case, schema_kind="explicit", storage="pyarrow"):
    """Keep CPython allocator-slot changes out of the main pytest interpreter."""
    import signal
    import sys

    script = r'''
import importlib.util
import sys
import tempfile
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("arrow_string_allocator_capture", sys.argv[4])
capture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capture
spec.loader.exec_module(capture)
case, schema_kind, storage = sys.argv[1:4]
with pytest.MonkeyPatch.context() as control:
    if case == "row":
        with tempfile.TemporaryDirectory(prefix="arrow-string-allocator-") as directory:
            class Factory:
                def mktemp(self, name):
                    path = Path(directory) / name
                    path.mkdir()
                    return path
            protocol = capture.protocol.__wrapped__(Factory())
            capture._check_arrow_string_custom_row_hook(
                protocol, control, "__new__", schema_kind, storage)
    else:
        assert case == "descriptor"
        capture._check_arrow_string_class_descriptor(control, "__new__")
'''
    owned_group = os.name == "posix"
    process = subprocess.Popen(
        [sys.executable, "-c", script, case, schema_kind, storage, str(Path(__file__).resolve())],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=owned_group,
    )

    def terminate_child():
        try:
            if owned_group:
                # protoc inherits this session, so a timeout also cleans up that child.
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass

    try:
        stdout, stderr = process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        terminate_child()
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        pytest.fail("Arrow-string allocator child timed out\n" + stdout + stderr)
    except BaseException:
        terminate_child()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        raise
    assert process.returncode == 0, stdout + stderr
    # Guard against accidentally reintroducing in-process allocator mutation.
    healthy = pd.array(["original", None], dtype="string[pyarrow]")
    assert healthy.__arrow_array__().to_pylist() == ["original", None]


@pytest.mark.parametrize("storage", ["pyarrow", "numpy_semantics"])
@pytest.mark.parametrize("schema_kind", ["explicit", "inferred", "names"])
@pytest.mark.parametrize("method", [
    "astype", "isna", "to_numpy", "__getitem__", "dtype", "na_value", "_hasna", "__init__",
    "_from_pyarrow_array", "__new__",
])
def test_arrow_string_custom_row_hooks_run_as_in_pyspark_conversion(
    protocol, monkeypatch, method, schema_kind, storage
):
    if method == "__new__":
        # Other hooks use this shared fixture signature; the child needs its own protocol.
        _arrow_string_allocator_process("row", schema_kind, storage)
    else:
        _check_arrow_string_custom_row_hook(protocol, monkeypatch, method, schema_kind, storage)


def _check_arrow_string_custom_row_hook(
    protocol, monkeypatch, method, schema_kind, storage
):
    """A customized hook of the class of an Arrow string array declines the fast path without
    running it. PySpark's conversion hands the column to Arrow through its `__arrow_array__`:
    only the hooks that pandas runs itself, to iterate the labels, of Arrow strings too but of
    length one, or to read the dtype of the column, run, which depends on the version of pandas.
    The outcome, the hook's error or the column's values, is that of PySpark's conversion with
    the fast path declined."""
    import importlib

    _pandas_input = importlib.import_module("pyspark.sql._pandas_input")

    texts = ["original", None] * 5000
    frames = []
    for _ in range(2):
        frame = pd.DataFrame({"v": _arrow_string_column(texts, storage)})
        # Object labels isolate the additional construction/_hasna data hooks.
        # The other Arrow labels exercise earlier checks when they share the data
        # class (pyarrow storage); a separate NumPy-semantics class is checked later.
        frame.columns = pd.Index(["v"], dtype=object) if method in (
            "_hasna", "__init__", "_from_pyarrow_array", "__new__",
        ) else pd.Index(
            pd.array(["v"], dtype="string[pyarrow]"))
        frames.append(frame)
    schema = T.StructType([T.StructField("v", T.StringType())])
    input_schema = {"explicit": schema, "inferred": None, "names": ["v"]}[schema_kind]
    array_class = type(frames[0].iloc[:, 0].array)
    calls = []
    # Warm all guards before rebinding: class authentication must not be cached
    # across calls, even when this exact class was previously accepted. This also
    # populates pandas' Block.dtype cache, so a customized data dtype reaches the
    # class check; the separate cold-dtype test covers its earlier error.
    for frame in frames:
        assert _pandas_input.prepare_string_input(frame, schema) is not None

    class HookError(Exception):
        pass

    def customized(*args, **kwargs):
        if method == "na_value":
            # This property receives a StringDtype, including during label iteration.
            assert type(args[0]) is pd.StringDtype
            calls.append(method)
        elif method == "__new__":
            # pandas may pass only the class; no initialized receiver is available.
            assert args[0] is array_class
            calls.append(method)
        elif method in ("__init__", "_from_pyarrow_array"):
            # __init__ receives an uninitialized object; inspect the subset instead.
            assert type(args[0]) is array_class
            assert len(args) == 2
            values = args[1]
            assert type(values) is pa.ChunkedArray
            assert values.null_count == 0
            assert values.to_pylist() == ["original"] * 5000
            assert set(kwargs) <= {"dtype"}
            if kwargs:
                assert type(kwargs["dtype"]) is pd.StringDtype
            calls.append(method)
        else:
            # The receiver, the labels of length one or the column of 10,000 values, which
            # the fast path's guards read neither of.
            calls.append((method, len(args[0])))
        raise HookError(f"{method} sentinel")

    class FactoryHook:
        def __get__(self, receiver, owner):
            assert receiver is not None, "eligibility must not bind the factory descriptor"
            assert owner is array_class
            return customized.__get__(receiver, owner)

    if method == "na_value":
        monkeypatch.setattr(pd.StringDtype, method, property(customized))
    elif method == "_from_pyarrow_array":
        # An absent helper on older pandas is also a conservative decline.
        monkeypatch.setattr(array_class, method, FactoryHook(), raising=False)
    elif method == "__new__":
        monkeypatch.setattr(array_class, method, staticmethod(customized))
    elif method in ("dtype", "_hasna"):
        monkeypatch.setattr(array_class, method, property(customized))
    else:
        monkeypatch.setattr(array_class, method, customized)
    for frame in frames:
        assert _pandas_input.prepare_string_input(frame, schema) is None
        assert calls == []

    outcomes = []
    tables = []
    for control, frame in zip((True, False), frames):
        calls.clear()
        with monkeypatch.context() as dispatch:
            if control:
                _force_pyspark_pandas_conversion(dispatch)
            with _arrow_input_service(protocol) as (spark, state):
                try:
                    result = spark.createDataFrame(frame, input_schema)
                    result.toArrow()
                except HookError as error:
                    assert type(error) is HookError
                    assert str(error) == f"{method} sentinel"
                    assert calls
                    outcome = (type(error), str(error))
                    assert state["input_tables"] == []
                else:
                    assert len(state["input_tables"]) == 1
                    table = state["input_tables"][0]
                    assert table.column(0).to_pylist() == texts
                    tables.append(table)
                    outcome = None
                outcomes.append((outcome, tuple(calls)))
                # As PySpark's createDataFrame, the settings are read before any conversion.
                assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
                assert state["uploads"] == 0
    assert outcomes[0] == outcomes[1]
    # Of the column, pandas reads only the dtype; the other hooks of an array that run are of
    # the labels.
    assert all(call[1] == 1 for call in calls if type(call) is tuple and call[0] != "dtype")
    if tables:
        assert len(tables) == 2
        assert tables[0].equals(tables[1], check_metadata=True)


@pytest.mark.parametrize("method", [
    "__arrow_array__", "__array__", "__getitem__", "isna", "to_numpy", "__len__", "astype",
    "__init__", "_from_pyarrow_array", "__new__", "__getattribute__", "dtype", "_hasna", "na_value",
])
def test_arrow_string_class_descriptors_decline_without_binding(monkeypatch, method):
    if method == "__new__":
        _arrow_string_allocator_process("descriptor")
    else:
        _check_arrow_string_class_descriptor(monkeypatch, method)


def _check_arrow_string_class_descriptor(monkeypatch, method):
    import importlib

    _pandas_input = importlib.import_module("pyspark.sql._pandas_input")
    array = pd.array(["original", None], dtype="string[pyarrow]")
    array_class = type(array)
    assert _pandas_input._arrow_string_class_is_canonical(array_class)
    assert _pandas_input._has_canonical_arrow_string_array(array)
    calls = []

    class HookError(Exception):
        pass

    class CustomizedDescriptor:
        def __get__(self, receiver, owner):
            calls.append((receiver, owner))
            raise HookError(f"{method} descriptor sentinel")

    owner = pd.StringDtype if method == "na_value" else array_class
    monkeypatch.setattr(owner, method, CustomizedDescriptor(), raising=False)
    # These concrete-class checks inspect raw dictionaries. Earlier frame.dtypes
    # reads are covered separately, as PySpark's conversion makes them too.
    assert not _pandas_input._arrow_string_class_is_canonical(array_class)
    assert not _pandas_input._has_canonical_arrow_string_array(array)
    assert calls == []


@pytest.mark.parametrize("method", [
    "__getattribute__", "tolist", "__iter__", "fillna", "to_numpy", "isna", "_hasna",
    "dtype", "na_value",
])
def test_arrow_string_label_descriptors_decline_without_binding(monkeypatch, method):
    import importlib

    _pandas_input = importlib.import_module("pyspark.sql._pandas_input")
    # Concrete object data isolates the Arrow-string class used by the labels.
    frame = pd.DataFrame({"v": pd.Series(["original", None] * 5000, dtype=object)})
    frame.columns = pd.Index(pd.array(["v"], dtype="string[pyarrow]"))
    array_class = type(frame.columns._values)
    schema = T.StructType([T.StructField("v", T.StringType())])
    assert _pandas_input.prepare_string_input(frame, schema) is not None
    calls = []

    class HookError(Exception):
        pass

    class CustomizedDescriptor:
        def __get__(self, receiver, owner):
            calls.append((receiver, owner))
            raise HookError(f"label {method} descriptor sentinel")

    owner = pd.StringDtype if method == "na_value" else array_class
    monkeypatch.setattr(owner, method, CustomizedDescriptor())
    assert _pandas_input.prepare_string_input(frame, schema) is None
    assert calls == []


@pytest.mark.parametrize("storage_kind", [
    "plain_array", "nonstring", "invalid_utf8", "bad_storage", "spoofed_array",
    "short_storage", "long_storage",
])
def test_arrow_string_storage_normalization_and_fallback(monkeypatch, storage_kind):
    import importlib
    import struct

    _pandas_input = importlib.import_module("pyspark.sql._pandas_input")
    texts = ["é", None]
    frame = pd.DataFrame({"v": pd.array(texts, dtype="string[pyarrow]")})
    schema = T.StructType([T.StructField("v", T.StringType())])
    # Capture pandas metadata before intentionally replacing its Arrow storage.
    dtypes = frame.dtypes
    array = frame.iloc[:, 0].array
    if storage_kind == "plain_array":
        storage = pa.array(texts, type=pa.string())
        assert isinstance(storage, pa.Array)
    elif storage_kind == "nonstring":
        storage = pa.chunked_array([pa.array([1, None], type=pa.int64())])
    elif storage_kind in ("short_storage", "long_storage"):
        count = 1 if storage_kind == "short_storage" else 3
        storage = pa.chunked_array([pa.array(["é"] * count, type=pa.string())])
        assert len(storage) != len(frame)
    elif storage_kind == "invalid_utf8":
        invalid = pa.Array.from_buffers(pa.string(), 2, [
            None, pa.py_buffer(struct.pack("<iii", 0, 5, 6)), pa.py_buffer(b"valid\xff"),
        ])
        storage = pa.chunked_array([invalid])
        with pytest.raises(pa.ArrowInvalid):
            storage.validate(full=True)
    elif storage_kind == "bad_storage":
        class UnsupportedStorage:
            # pandas' column boxing reads length before the storage-type guard.
            def __len__(self):
                return len(texts)

        storage = UnsupportedStorage()
        assert not isinstance(storage, (pa.Array, pa.ChunkedArray))
    else:
        class SpoofedArray:
            @property
            def __class__(self):
                return pa.Array

            def __len__(self):
                return len(texts)

        storage = SpoofedArray()
        assert isinstance(storage, pa.Array)
        assert not issubclass(type(storage), pa.Array)
    monkeypatch.setattr(array, "_pa_array", storage)
    assert object.__getattribute__(frame.iloc[:, 0].array, "_pa_array") is storage
    result = _pandas_input._arrow_string_table(frame, schema, dtypes)
    if storage_kind == "plain_array":
        assert type(result) is pa.Table
        result.validate(full=True)
        assert result.schema.names == ["v"]
        assert result.column(0).type == pa.string()
        assert result.column(0).to_pylist() == texts
    else:
        assert result is None


@pytest.mark.parametrize("error_kind", ["arrow", "overflow"])
def test_arrow_string_simulated_cast_failures_decline(monkeypatch, error_kind):
    """Exercise exception handling without allocating a greater-than-2-GiB column."""
    import importlib
    import sys
    from types import ModuleType

    _pandas_input = importlib.import_module("pyspark.sql._pandas_input")
    frame = pd.DataFrame({"v": pd.array(["original", None], dtype="string[pyarrow]")})
    schema = T.StructType([T.StructField("v", T.StringType())])
    dtypes = frame.dtypes
    array = frame.iloc[:, 0].array
    values = array.__arrow_array__()
    error_class = pa.ArrowInvalid if error_kind == "arrow" else OverflowError
    calls = []

    class SimulatedCastFailure:
        type = values.type

        def __len__(self):
            return len(values)

        def validate(self, *, full):
            calls.append(("validate", full))
            values.validate(full=full)

        def cast(self, target):
            assert target == pa.string()
            calls.append(("cast", target))
            raise error_class("simulated Arrow-string cast failure")

    # PyArrow extension types are immutable. This local module proxy substitutes
    # the exact recognized storage class only for this simulated cast branch;
    # no real ChunkedArray subclass or capacity behavior is being tested.
    proxy = ModuleType("pyarrow")
    vars(proxy).update(vars(pa))
    proxy.ChunkedArray = SimulatedCastFailure
    with monkeypatch.context() as simulated:
        simulated.setattr(array, "_pa_array", SimulatedCastFailure())
        simulated.setitem(sys.modules, "pyarrow", proxy)
        assert _pandas_input._arrow_string_table(frame, schema, dtypes) is None
    assert calls == [("validate", True), ("cast", pa.string())]


def test_arrow_string_object_labels_dtype_error_matches_pyspark_conversion(protocol, monkeypatch):
    # A customized dtype of the class of an Arrow string column, not cached by an earlier read,
    # raises in the fast path's read of the dtypes, as in PySpark's conversion, which reads the
    # dtype of the column: the same error, with the fast path declined or not.
    import importlib

    _pandas_input = importlib.import_module("pyspark.sql._pandas_input")
    frames = []
    for _ in range(3):
        frame = pd.DataFrame({"v": pd.array(["original", None] * 5000,
                                          dtype="string[pyarrow]")})
        frame.columns = pd.Index(["v"], dtype=object)
        frames.append(frame)
    schema = T.StructType([T.StructField("v", T.StringType())])
    array_class = type(frames[0].iloc[:, 0].array)
    calls = []

    class HookError(Exception):
        pass

    def customized(self):
        assert type(self) is array_class
        assert len(self) == 10000
        calls.append("dtype")
        raise HookError("dtype sentinel")

    monkeypatch.setattr(array_class, "dtype", property(customized))
    errors = []
    with pytest.raises(HookError, match="^dtype sentinel$") as error:
        _pandas_input.prepare_string_input(frames[0], schema)
    errors.append((type(error.value), str(error.value)))
    assert calls == ["dtype"]
    for control, frame in zip((False, True), frames[1:]):
        calls.clear()
        with monkeypatch.context() as dispatch:
            if control:
                _force_pyspark_pandas_conversion(dispatch)
            with _arrow_input_service(protocol) as (spark, state):
                with pytest.raises(HookError, match="^dtype sentinel$") as error:
                    spark.createDataFrame(frame, schema)
                errors.append((type(error.value), str(error.value)))
                # After the one read of the settings, as PySpark's conversion.
                assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
                assert state["uploads"] == 0
                assert state["input_tables"] == []
        assert calls == ["dtype"]
    assert errors[0] == errors[1] == errors[2]


def test_arrow_string_custom_label_astype_runs_as_in_pyspark_conversion(protocol, monkeypatch):
    # A customized astype of the Arrow string labels declines the fast path without calling it.
    # PySpark's conversion calls it where pandas casts the labels to iterate them, as pandas 2
    # does, and not where it does not, as pandas 3: the outcome is that of the conversion with
    # the fast path declined.
    import importlib

    _pandas_input = importlib.import_module("pyspark.sql._pandas_input")

    texts = ["original", None] * 5000
    frames = []
    for _ in range(2):
        frame = pd.DataFrame({"v": pd.array(texts, dtype="string[pyarrow]")})
        frame.columns = pd.Index(pd.array(["v"], dtype="string[pyarrow]"))
        frames.append(frame)
    schema = T.StructType([T.StructField("v", T.StringType())])
    calls = []

    class HookError(Exception):
        pass

    def customized(*args, **kwargs):
        calls.append("label astype")
        raise HookError("label astype sentinel")

    for frame in frames:
        monkeypatch.setattr(frame.columns._values, "astype", customized)
        assert _pandas_input.prepare_string_input(frame, schema) is None
    assert calls == []
    outcomes = []
    tables = []
    for control, frame in zip((True, False), frames):
        calls.clear()
        with monkeypatch.context() as dispatch:
            if control:
                _force_pyspark_pandas_conversion(dispatch)
            with _arrow_input_service(protocol) as (spark, state):
                try:
                    spark.createDataFrame(frame, schema).toArrow()
                except HookError as error:
                    assert str(error) == "label astype sentinel"
                    assert calls == ["label astype"]
                    assert state["input_tables"] == []
                    outcomes.append(str(error))
                else:
                    assert calls == []
                    tables.append(state["input_tables"][0])
                    assert tables[-1].column(0).to_pylist() == texts
                    outcomes.append(None)
                # As PySpark's createDataFrame, the settings are read before any conversion.
                assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
                assert state["uploads"] == 0
    assert outcomes[0] == outcomes[1]
    if tables:
        assert tables[0].equals(tables[1], check_metadata=True)


def test_arrow_string_null_labels_decline_without_getitem_callback(protocol, monkeypatch):
    import importlib

    from pandas.core.arrays.arrow.array import ArrowExtensionArray

    _pandas_input = importlib.import_module("pyspark.sql._pandas_input")
    # Index uniqueness can cast null labels and invoke indexed masking. The fast path's guard
    # must decline without an early callback; PySpark's conversion, which iterates the labels,
    # then has the outcome of the conversion with the fast path declined.
    expected = {"a": ["first"] * 10000, "b": ["second"] * 10000}
    frames = []
    for _ in range(2):
        frame = pd.DataFrame({name: pd.array(values, dtype="string[pyarrow]")
                              for name, values in expected.items()})
        frame.columns = pd.Index(pd.array(["a", None], dtype="string[pyarrow]"))
        frames.append(frame)
    schema = T.StructType([T.StructField(name, T.StringType()) for name in expected])
    calls = []

    class HookError(Exception):
        pass

    def customized(*args, **kwargs):
        calls.append("label getitem")
        raise HookError("label getitem sentinel")

    monkeypatch.setattr(ArrowExtensionArray, "__getitem__", customized)
    for frame in frames:
        assert _pandas_input.prepare_string_input(frame, schema) is None
        assert calls == []

    outcomes = []
    tables = []
    # Observe the control first, using a fresh equal frame for each route.
    for control, frame in zip((True, False), frames):
        calls.clear()
        with monkeypatch.context() as dispatch:
            if control:
                _force_pyspark_pandas_conversion(dispatch)
            with _arrow_input_service(protocol) as (spark, state):
                try:
                    spark.createDataFrame(frame, schema).toArrow()
                except HookError as error:
                    assert type(error) is HookError
                    assert str(error) == "label getitem sentinel"
                    assert calls
                    outcome = (type(error), str(error))
                    assert state["input_tables"] == []
                else:
                    assert len(state["input_tables"]) == 1
                    table = state["input_tables"][0]
                    assert table.to_pydict() == expected
                    tables.append(table)
                    outcome = None
                outcomes.append((outcome, tuple(calls)))
                # Whether the hook runs depends on the pandas version. As PySpark's
                # createDataFrame, the settings are read before any conversion.
                assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
                assert state["uploads"] == 0
    assert outcomes[0] == outcomes[1]
    if tables:
        assert len(tables) == 2
        assert tables[0].equals(tables[1], check_metadata=True)


@pytest.mark.parametrize("label_storage", ["object", "pyarrow"])
@pytest.mark.parametrize("method", ["astype", "__array__", "__getitem__", "__init__"])
@pytest.mark.parametrize("storage", ["pyarrow", "numpy_semantics"])
def test_arrow_string_super_row_hooks_run_as_in_pyspark_conversion(
    protocol, monkeypatch, label_storage, method, storage
):
    # A customized hook of the ArrowExtensionArray base of the Arrow string arrays declines the
    # fast path without running it. PySpark's conversion hands the column to Arrow through its
    # `__arrow_array__`, so it runs only what pandas runs of these hooks to iterate Arrow string
    # labels, of a length of one, which depends on the version of pandas: the outcome is that of
    # the conversion with the fast path declined.
    import importlib

    from pandas.core.arrays.arrow.array import ArrowExtensionArray

    _pandas_input = importlib.import_module("pyspark.sql._pandas_input")
    frames = []
    for _ in range(2):
        frame = pd.DataFrame({"v": _arrow_string_column(["original", None] * 5000, storage)})
        frame.columns = pd.Index(["v"], dtype=object) if label_storage == "object" else pd.Index(
            pd.array(["v"], dtype="string[pyarrow]"))
        frames.append(frame)
    schema = T.StructType([T.StructField("v", T.StringType())])
    calls = []
    for frame in frames:
        assert _pandas_input.prepare_string_input(frame, schema) is not None

    class HookError(Exception):
        pass

    def customized(self, *args, **kwargs):
        if method == "__init__":
            # The receiver has no storage yet. Null-subarray construction passes
            # the 5,000 non-null data values; labels have only one value.
            values = args[0]
            assert values.to_pylist() == ["original"] * 5000
            calls.append(method)
        else:
            # The receiver, the labels of length one or the column of 10,000 values, which
            # the fast path's guards read neither of.
            calls.append((method, len(self)))
        raise HookError(f"super {method} sentinel")

    monkeypatch.setattr(ArrowExtensionArray, method, customized)
    for frame in frames:
        assert _pandas_input.prepare_string_input(frame, schema) is None
    assert calls == []
    outcomes = []
    tables = []
    # The control determines whether this pandas layout calls the hook.
    # pandas 3.1's null-subarray factory allocates through __new__, not __init__.
    for control, frame in zip((True, False), frames):
        calls.clear()
        with monkeypatch.context() as dispatch:
            if control:
                _force_pyspark_pandas_conversion(dispatch)
            with _arrow_input_service(protocol) as (spark, state):
                try:
                    spark.createDataFrame(frame, schema).toArrow()
                except HookError as error:
                    assert type(error) is HookError
                    assert str(error) == f"super {method} sentinel"
                    assert len(calls) == 1
                    assert state["input_tables"] == []
                    outcome = (type(error), str(error))
                else:
                    assert calls == []
                    assert len(state["input_tables"]) == 1
                    table = state["input_tables"][0]
                    table.validate(full=True)
                    assert table.to_pydict() == {"v": ["original", None] * 5000}
                    assert table.schema.names == ["v"]
                    assert table.schema.field(0).type == pa.string()
                    assert table.schema.field(0).nullable
                    tables.append(table)
                    outcome = None
                outcomes.append((outcome, tuple(calls)))
                # Whether the hook runs depends on the pandas version. As PySpark's
                # createDataFrame, the settings are read before any conversion.
                assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
                assert state["uploads"] == 0
    assert outcomes[0] == outcomes[1]
    # The hooks that run are of the labels, not of the column.
    assert all(call[1] == 1 for call in calls if type(call) is tuple)
    if tables:
        assert len(tables) == 2
        assert tables[0].equals(tables[1], check_metadata=True)


@pytest.mark.parametrize("reason", [
    "string_subclass", "frame_subclass", "nan", "pd_na", "nat", "string_dtype",
    "nonnullable_null", "inferred", "names", "ddl", "small", "collation", "cast", "mixed",
])
def test_string_public_fallback_converts_as_pyspark(protocol, monkeypatch, reason):
    # A frame of object strings that the fast path declines, for each of its reasons, converts
    # as PySpark's conversion does: it hands the column to Arrow, which reads the values of a
    # str subclass as they are, without its __str__, and it never calls the frame's astype.
    # Arrow's errors are PySpark's: of a null in a non-nullable field, of a cast of strings to
    # numbers, and of a value that is not a string.
    from pyspark.errors import PySparkTypeError, PySparkValueError

    calls = []

    class Text(str):
        def __str__(self):
            calls.append("str")
            return "changed"

    class Frame(pd.DataFrame):
        pass

    values = ["plain"] * (9999 if reason == "small" else 10000)
    if reason in ("nan", "pd_na", "nat", "nonnullable_null", "mixed"):
        values[-1] = {"nan": float("nan"), "pd_na": pd.NA, "nat": pd.NaT,
                      "nonnullable_null": None, "mixed": 42}[reason]
    elif reason == "string_subclass":
        values[-1] = Text("plain")
    frame_type = Frame if reason == "frame_subclass" else pd.DataFrame
    frame = frame_type({"text": values}, dtype="string" if reason == "string_dtype" else object)
    schema = T.StructType([T.StructField("text", T.StringType(), reason != "nonnullable_null")])
    if reason == "inferred":
        schema = None
    elif reason == "names":
        schema = ["renamed"]
    elif reason == "ddl":
        schema = "text STRING"
    elif reason == "collation":
        schema = T.StructType([T.StructField("text", T.StringType("UTF8_LCASE"))])
    elif reason == "cast":
        schema = T.StructType([T.StructField("text", T.LongType())])
    astype = pd.DataFrame.astype

    def record_rows(self, *args, **kwargs):
        calls.append("astype")
        return astype(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "astype", record_rows)
    errors = {
        "nonnullable_null": (ValueError, "Casting field 'text' with null values to non-nullable"),
        "cast": (PySparkValueError, (
            "Exception thrown when converting pandas.Series (object) with name 'text' to Arrow "
            "Array (int64). It can be caused by overflows or other unsafe conversions warned by "
            "Arrow. Arrow safe type check can be disabled by using SQL config "
            "`spark.sql.execution.pandas.convertToArrowArraySafely`.")),
        "mixed": (PySparkTypeError, (
            "Exception thrown when converting pandas.Series (object) with name 'text' to Arrow "
            "Array (string).")),
    }
    with _arrow_input_service(protocol) as (spark, state):
        if reason in errors:
            error_type, message = errors[reason]
            with pytest.raises(error_type) as error:
                spark.createDataFrame(frame, schema)
            assert type(error.value) is error_type
            assert str(error.value) == message
            assert calls == []
            assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
            assert state["relations"] == []
            return
        spark.createDataFrame(frame, schema).toArrow()
        assert calls == []
        table = state["input_tables"][0]
        assert state["column_names"] == ["renamed" if reason == "names" else "text"]
        expected = ["plain"] * len(values)
        if reason in ("nan", "pd_na", "nat"):
            expected[-1] = None
        assert table.column(0).to_pylist() == expected
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]


@pytest.mark.parametrize("surrogate", ["\ud800", "\udfff"])
def test_string_surrogate_error_follows_the_settings_read(protocol, monkeypatch, surrogate):
    from pyspark.errors import PySparkValueError

    values = ["plain"] * 10000
    values[-1] = surrogate
    frame = pd.DataFrame({"text": values}, dtype=object)
    schema = T.StructType([T.StructField("text", T.StringType())])
    astype = pd.DataFrame.astype
    calls = []

    def record_rows(self, *args, **kwargs):
        calls.append("astype")
        return astype(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "astype", record_rows)
    with _arrow_input_service(protocol) as (spark, state):
        # The fast path cannot encode the string and declines the frame, which converts as
        # PySpark's conversion does, after the settings that its createDataFrame reads before
        # it converts any data: Arrow's UnicodeEncodeError, a ValueError, which PySpark raises
        # as its PySparkValueError.
        with pytest.raises(PySparkValueError) as error:
            spark.createDataFrame(frame, schema)
        assert str(error.value) == (
            "Exception thrown when converting pandas.Series (object) with name 'text' to Arrow "
            "Array (string). It can be caused by overflows or other unsafe conversions warned by "
            "Arrow. Arrow safe type check can be disabled by using SQL config "
            "`spark.sql.execution.pandas.convertToArrowArraySafely`.")
        assert type(error.value.__cause__) is UnicodeEncodeError
        assert "surrogates not allowed" in str(error.value.__cause__)
        assert calls == []
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
        assert state["relations"] == []
        assert state["uploads"] == 0


_STRING_RELATION = {"fields": [{"metadata": {"source": "native"}, "name": "text", "nullable": True,
                                "type": "string"}], "type": "struct"}


def _string_converts(name="text", nullable=True, metadata=None, relation=_STRING_RELATION,
                     error=None):
    return _guard_converts([(name, "string", nullable, ["plain"] * 10000, metadata)], relation,
                           error)


# PySpark's outcomes (see `_INTEGER_DESCRIPTORS`), and the drop-in's where they differ: as
# PySpark's `json()` of the schema, which reads a patched `isUTF8BinaryCollation` too, the relation's
# schema holds the collation of the string.
_STRING_DESCRIPTORS = {
    ("StructType", "fields"): _guard_raises(*_no_setter("StructType")),
    ("StructType", "names"): _guard_raises(*_no_setter("StructType")),
    ("StructType", "__iter__"): _guard_raises(*_rename_mismatch("text")),
    ("StructField", "name"): _guard_raises(*_no_setter("StructField")),
    ("StructField", "dataType"): _guard_raises(*_no_setter("StructField")),
    ("StructField", "nullable"): _guard_raises(*_no_setter("StructField")),
    ("StructField", "metadata"): _guard_raises(*_no_setter("StructField")),
    ("StructType", "alias"): _string_converts(
        error=["_to_table", "builtins.AssertionError", ""]),
    ("StringType", "isUTF8BinaryCollation"): _string_converts(relation={
        "fields": [{"metadata": {"__COLLATIONS": {"text": "spark.UTF8_BINARY"},
                                 "source": "native"},
                    "name": "text", "nullable": True, "type": "string collate UTF8_BINARY"}],
        "type": "struct"}),
}
_STRING_DESCRIPTORS_OF_THE_DROP_IN = {
    ("StructType", "fields"): _guard_converts([], {"fields": [], "type": "struct"}),
    ("StructType", "names"): _string_converts(name="changed"),
    ("StructField", "name"): _guard_raises(*_rename_mismatch("text")),
    ("StructField", "dataType"): _string_converts(),
    ("StructField", "nullable"): _string_converts(
        nullable=False, relation=_relation_of("text", "string", False, {"source": "native"})),
    ("StructField", "metadata"): _string_converts(
        metadata={"SPARK::metadata::json": '{"changed": true}'},
        relation=_relation_of("text", "string", True, {"changed": True})),
    ("StructType", "alias"): _string_converts(),
}


@pytest.mark.parametrize("class_name,attribute", [
    ("StructType", "names"), ("StructType", "fields"), ("StructType", "__iter__"),
    ("StructField", "name"), ("StructField", "dataType"), ("StructField", "nullable"),
    ("StructField", "metadata"), ("StringType", "isUTF8BinaryCollation"),
    ("StructType", "__getattribute__"), ("StructField", "__getattribute__"),
    ("StringType", "__getattribute__"), ("StructType", "__class__"),
    ("StructField", "__class__"), ("StringType", "__class__"),
    ("StructType", "alias"), ("StructField", "alias"), ("StringType", "alias"),
])
@pytest.mark.parametrize("before_import", [False, True])
def test_string_custom_schema_descriptors_take_pyspark_conversion(class_name, attribute,
                                                                  before_import):
    # A patch of a class or a rebound name declines the fast path of object strings: the frame
    # converts with PySpark's conversion, which reads the patch (see `_integer_guard_process`).
    script = r'''
import sys
import pandas as pd
from pyspark.sql import types as T

import importlib.util
spec = importlib.util.spec_from_file_location("string_schema_capture", sys.argv[-1])
capture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capture
spec.loader.exec_module(capture)
frame = pd.DataFrame({"text": ["plain"] * 10000}, dtype=object)
schema = T.StructType([T.StructField("text", T.StringType(), True, {"source": "native"})])
class_name, attribute, before_import = sys.argv[1:4]
if before_import == "True":
    # The capture module imports prepare_int64_table; force the public call to
    # import a new helper after the class or public alias has been customized.
    sys.modules.pop("pyspark.sql._pandas_input", None)
assert ("pyspark.sql._pandas_input" in sys.modules) == (before_import == "False")
cls = getattr(T, class_name)
if attribute == "alias":
    setattr(T, class_name, object)
else:
    if attribute == "__getattribute__":
        replacement = lambda self, name: object.__getattribute__(self, name)
    elif attribute == "__iter__":
        replacement = lambda self: iter([T.StructField("changed", T.StringType())])
    elif attribute == "__class__":
        attribute = "__getattribute__"
        replacement = lambda self, name: object if name == "__class__" else (
            object.__getattribute__(self, name))
    elif attribute == "isUTF8BinaryCollation":
        replacement = lambda self: False
    else:
        value = {"names": ["changed"], "fields": [], "name": "changed",
                 "dataType": T.StringType(), "nullable": False, "metadata": {"changed": True}}[attribute]
        replacement = property(lambda self: value)
    setattr(cls, attribute, replacement)
'''
    outcome = _guard_outcome(script, class_name, attribute, str(before_import))
    _assert_guard(outcome, _STRING_DESCRIPTORS.get((class_name, attribute), _string_converts()),
                  _STRING_DESCRIPTORS_OF_THE_DROP_IN.get((class_name, attribute)))



def test_string_rebound_pandas_alias_keeps_subclass_fallback(monkeypatch):
    from pyspark.sql._pandas_input import prepare_string_input

    class Frame(pd.DataFrame):
        pass

    frame = Frame({"text": ["plain"] * 10000}, dtype=object)
    schema = T.StructType([T.StructField("text", T.StringType())])
    monkeypatch.setattr(pd, "DataFrame", Frame)
    assert prepare_string_input(frame, schema) is None



@pytest.mark.parametrize("dtype_mode", ["equal", "class_spoof", "exact_numpy"])
def test_string_custom_extension_dtype_takes_pyspark_conversion(protocol, dtype_mode):
    # An extension array of a dtype that equals, or claims to be, NumPy's object dtype does not
    # select the fast path of object strings: as PySpark's conversion, which hands a column of
    # an array with `__arrow_array__` to Arrow as it is, the values are those that the hook
    # gives, not those of a scan of the array.
    from pandas.api.extensions import ExtensionArray, ExtensionDtype, take
    from pyspark.sql._pandas_input import prepare_string_input

    arrow_calls = []

    class ObjectEqualDtype(ExtensionDtype):
        name = "object_equal_string"
        type = str
        kind = "O"
        na_value = None

        @classmethod
        def construct_array_type(cls):
            return DifferentArrowArray

        def __eq__(self, other):
            return (isinstance(other, ObjectEqualDtype)
                    or isinstance(other, np.dtype) and other.kind == "O")

        def __hash__(self):
            return hash(self.name)

    class DifferentArrowArray(ExtensionArray):
        def __init__(self, values):
            self._values = np.asarray(values, dtype=object)

        @property
        def dtype(self):
            return ObjectEqualDtype()

        @property
        def nbytes(self):
            return self._values.nbytes

        def __len__(self):
            return len(self._values)

        def __getitem__(self, key):
            result = self._values[key]
            return type(self)(result) if isinstance(result, np.ndarray) else result

        def isna(self):
            return pd.isna(self._values)

        def take(self, indices, allow_fill=False, fill_value=None):
            return type(self)(take(self._values, indices, allow_fill=allow_fill,
                                   fill_value=fill_value))

        def copy(self):
            return type(self)(self._values.copy())

        @classmethod
        def _from_sequence(cls, scalars, *, dtype=None, copy=False):
            values = np.asarray(scalars, dtype=object)
            return cls(values.copy() if copy else values)

        def to_numpy(self, dtype=None, copy=False, na_value=None):
            if dtype is not None:
                return self._values.astype(dtype, copy=copy)
            return self._values.copy() if copy else self._values

        def __arrow_array__(self, type=None):
            arrow_calls.append("arrow")
            # A scan of to_numpy() cannot establish the values this hook emits.
            return pa.array(["different"] * len(self), type=type)

    array = DifferentArrowArray(["plain"] * 10000)
    frame = pd.DataFrame({"text": pd.Series(array, copy=False)})
    schema = T.StructType([T.StructField("text", T.StringType())])
    if dtype_mode == "exact_numpy":
        # An ExtensionArray may change its advertised dtype after construction.
        # The dtype and NumPy scan both pass while its Arrow hook changes cells.
        DifferentArrowArray.dtype = property(lambda self: np.dtype("object"))
        DifferentArrowArray.__array__ = lambda self, dtype=None, copy=None: self._values
    elif dtype_mode == "class_spoof":
        def spoof_class(self, name):
            if name == "__class__":
                return type(np.dtype("object"))
            return object.__getattribute__(self, name)

        ObjectEqualDtype.__getattribute__ = spoof_class
    dtype = frame.dtypes.iloc[0]
    assert type(frame) is pd.DataFrame
    assert type(frame.iloc[:, 0].array) is DifferentArrowArray
    assert type(dtype) is (type(np.dtype("object")) if dtype_mode == "exact_numpy"
                           else ObjectEqualDtype)
    assert isinstance(dtype, np.dtype) is (dtype_mode != "equal")
    assert dtype == np.dtype("object")
    assert prepare_string_input(frame, schema) is None
    assert arrow_calls == []
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert state["input_tables"][0].column(0).to_pylist() == ["different"] * 10000
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
    assert arrow_calls == ["arrow"]



@pytest.mark.parametrize("method", ["astype", "notna"])
def test_string_instance_conversion_hooks_are_not_called_as_in_pyspark(protocol, method):
    # A frame whose own notna or astype is customized declines the fast path of object strings,
    # and converts as PySpark's conversion does, which never calls them: the values are the
    # frame's, whatever the hooks would change.
    from pyspark.sql._pandas_input import prepare_string_input

    frame = pd.DataFrame({"text": ["plain"] * 10000}, dtype=object)
    schema = T.StructType([T.StructField("text", T.StringType())])
    original = getattr(frame, method)
    calls = []

    def customized(*args, **kwargs):
        calls.append(method)
        result = original(*args, **kwargs)
        result.iat[0, 0] = "changed" if method == "astype" else False
        return result

    setattr(frame, method, customized)
    assert prepare_string_input(frame, schema) is None
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert calls == []
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
        assert state["uploads"] == 0
        assert state["input_tables"][0].column(0).to_pylist() == ["plain"] * 10000



@pytest.mark.parametrize("method", ["astype", "notna", "where"])
@pytest.mark.parametrize("before_import", [False, True])
def test_string_class_conversion_hooks_are_not_called_as_in_pyspark(
    protocol, monkeypatch, method, before_import
):
    # As for the instance: customized methods of the class, also before the helpers are
    # imported, decline the fast path, and PySpark's conversion never calls them.
    import importlib

    name = "pyspark.sql._pandas_input"
    if before_import:
        _forget_pandas_input(monkeypatch)
    else:
        importlib.import_module(name)
    frame = pd.DataFrame({"text": ["plain"] * 10000}, dtype=object)
    schema = T.StructType([T.StructField("text", T.StringType())])
    original = getattr(pd.DataFrame, method)
    calls = []

    def customized(self, *args, **kwargs):
        calls.append(method)
        result = original(self, *args, **kwargs)
        result.iat[0, 0] = False if method == "notna" else "changed"
        return result

    monkeypatch.setattr(pd.DataFrame, method, customized)
    assert importlib.import_module(name).prepare_string_input(frame, schema) is None
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert calls == []
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
        assert state["uploads"] == 0
        assert state["input_tables"][0].column(0).to_pylist() == ["plain"] * 10000



def test_string_instance_where_override_is_not_called_as_in_pyspark(protocol):
    # A frame of its own where declines the fast path, and converts as PySpark's conversion
    # does, which never calls it.
    from pyspark.sql._pandas_input import prepare_string_input

    frame = pd.DataFrame({"text": ["plain"] * 10000}, dtype=object)
    schema = T.StructType([T.StructField("text", T.StringType())])

    def customized(*args, **kwargs):
        raise AssertionError("PySpark's conversion never calls the frame's where")

    frame.where = customized
    assert prepare_string_input(frame, schema) is None
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
        assert state["input_tables"][0].column(0).to_pylist() == ["plain"] * 10000



def test_string_native_backing_does_not_add_series_conversion_hooks(protocol, monkeypatch):
    frame = pd.DataFrame({"text": ["plain"] * 10000}, dtype=object)
    schema = T.StructType([T.StructField("text", T.StringType())])

    def customized(*args, **kwargs):
        raise AssertionError("native string input must not call Series.to_numpy")

    monkeypatch.setattr(pd.Series, "to_numpy", customized)
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
        assert state["input_tables"][0].column(0).to_pylist() == ["plain"] * 10000



def test_string_large_inline_input_keeps_row_transport_and_schema(protocol):
    values = ["漢字-🙂-" + str(i) for i in range(100000)]
    frame = pd.DataFrame({"text": values}, dtype=object)
    schema = T.StructType([T.StructField("renamed", T.StringType(), False,
                                        {"source": "inline-string", "size": len(values)})])
    with _arrow_input_service(protocol, **{
        # Above the default threshold, so that the relation stays inline.
        "spark.sql.session.localRelationCacheThreshold": str(1 << 30),
        "spark.sql.session.localRelationSizeLimit": "1",
        "spark.sql.execution.arrow.useLargeVarTypes": "true",
    }) as (spark, state):
        result = spark.createDataFrame(frame, schema)
        assert not result.isLocal()
        result.toArrow()
        assert state["column_names"] == schema.names
        # The relation is 2 MB of Arrow data, so the settings are read; it is sent inline.
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
        assert state["uploads"] == 0
        table = state["input_tables"][0]
        assert table.schema.names == ["renamed"]
        # Of PySpark's large strings, under useLargeVarTypes.
        assert table.column(0).type == pa.large_string()
        assert table.schema.field(0).nullable is False
        assert table.schema.field(0).metadata is None
        assert table.column(0).to_pylist() == values
        assert __import__("json").loads(state["relations"][0].local_relation.schema) == (
            __import__("json").loads(schema.json()))


# Explicit-schema pandas input converts as PySpark's Arrow conversion does: the session time
# zone, the safe-cast setting, the nullability and the errors. Each expectation below was
# checked against official PySpark 4.3.0 with the same frame, schema and session settings.


def _pandas_table(protocol, frame, schema, **overrides):
    """The table that an explicit-schema pandas frame sends, and whether it is local."""
    with _arrow_input_service(protocol, **overrides) as (spark, state):
        df = spark.createDataFrame(frame, schema)
        assert not df.isLocal()
        df.toArrow()
        return state["input_tables"][0]


def _in_host_zone(name, call):
    """Run `call` with the process time zone `name`, as the client's local time zone."""
    import time

    previous = os.environ.get("TZ")
    os.environ["TZ"] = name
    time.tzset()
    try:
        return call()
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()


@pytest.mark.parametrize("values", [[1, None], [1.0, np.nan]], ids=["int", "float"])
def test_pandas_explicit_int_nulls_are_null(protocol, values):
    table = _pandas_table(protocol, pd.DataFrame({"f": values}), "f int")
    # Official: [Row(f=1), Row(f=None)].
    assert table.schema.field("f").type == pa.int32()
    assert table.column("f").to_pylist() == [1, None]


def _tinyint_schema():
    return T.StructType([T.StructField("b", T.ByteType())])


def test_pandas_explicit_tinyint_overflow_raises_as_pyspark(protocol):
    from pyspark.errors import PySparkValueError

    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(PySparkValueError) as error:
            spark.createDataFrame(pd.DataFrame({"b": [257]}), _tinyint_schema())
    assert str(error.value) == (
        "Exception thrown when converting pandas.Series (int64) with name 'b' to Arrow Array "
        "(int8). It can be caused by overflows or other unsafe conversions warned by Arrow. "
        "Arrow safe type check can be disabled by using SQL config "
        "`spark.sql.execution.pandas.convertToArrowArraySafely`."
    )
    assert state["input_tables"] == []


def test_pandas_explicit_tinyint_overflow_wraps_when_unsafe(protocol):
    # Official, with convertToArrowArraySafely=false: [Row(b=1)].
    table = _pandas_table(
        protocol, pd.DataFrame({"b": [257]}), _tinyint_schema(),
        **{"spark.sql.execution.pandas.convertToArrowArraySafely": "false"},
    )
    assert table.column("b").to_pylist() == [1]


def test_pandas_explicit_nonnullable_null_raises_as_pyspark(protocol):
    schema = T.StructType([T.StructField("a", T.LongType(), False)])
    frame = pd.DataFrame({"a": pd.array([1, None], dtype="Int64")})
    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(ValueError) as error:
            spark.createDataFrame(frame, schema)
    # Official: the ValueError of the cast of the table to the schema.
    assert type(error.value) is ValueError
    assert str(error.value) == "Casting field 'a' with null values to non-nullable"
    assert state["input_tables"] == []


class _PandasSubclass(pd.DataFrame):
    """A frame of a class of its own, which converts as PySpark's conversion converts every
    pandas DataFrame, whatever its class."""


@pytest.mark.parametrize("frame_class", [pd.DataFrame, _PandasSubclass], ids=["frame", "subclass"])
@pytest.mark.parametrize("data,schema,rows,names", [
    ({"a": [1], "b": [2]}, "a long", [{"a": 1}], ["a"]),
    ({"a": [1], "b": [2], "c": [3]}, "x long, y long", [{"x": 1, "y": 2}], ["x", "y"]),
    (None, "a long", [{"a": 5}, {"a": 6}], ["a"]),
], ids=["more-columns", "more-columns-of-two-fields", "no-columns"])
def test_pandas_explicit_struct_of_another_width_zips_the_columns_as_pyspark(
        protocol, frame_class, data, schema, rows, names):
    # As PySpark Connect's conversion of a frame of any class, which zips its columns with the
    # fields: the columns past the fields are dropped, and a frame of no columns is the table of
    # its index (one that is not a RangeIndex), which the rename takes.
    frame = frame_class(index=[5, 6]) if data is None else frame_class(data)
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert state["input_tables"][0].to_pylist() == rows
        assert state["column_names"] == names


@pytest.mark.parametrize("frame_class", [pd.DataFrame, _PandasSubclass], ids=["frame", "subclass"])
def test_pandas_explicit_struct_of_more_fields_raises_the_rename_of_pyspark(protocol, frame_class):
    # As PySpark Connect's conversion, of a frame of fewer columns than the fields, whose table it
    # renames to the names of the fields, of PyArrow's ArrowInvalid; and of a struct of no fields,
    # whose table of no columns has no rows.
    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(pa.ArrowInvalid) as error:
            spark.createDataFrame(frame_class({"a": [1]}), "x long, y long")
        assert str(error.value) == (
            "tried to rename a table of 1 columns but only 2 names were provided")
        spark.createDataFrame(frame_class({"a": [1, 2]}), T.StructType([])).toArrow()
        assert [(table.num_columns, table.num_rows) for table in state["input_tables"]] == [
            (0, 0)]


@pytest.mark.parametrize("frame_class", [pd.DataFrame, _PandasSubclass], ids=["frame", "subclass"])
def test_pandas_frame_of_no_columns_under_a_struct_of_no_fields_as_pyspark(protocol, frame_class):
    from pyspark.errors import PySparkValueError

    # As PySpark Connect's, which takes a frame of no columns as `pa.RecordBatch.from_pandas`
    # does, of any class: a column of each level of its index unless it is a RangeIndex, one
    # more than the fields of a struct of none. Official: AXIS_LENGTH_MISMATCH for the index
    # [0, 1], and two rows of no columns for a RangeIndex, which the live suite collects.
    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(PySparkValueError) as error:
            spark.createDataFrame(frame_class(index=[0, 1]), T.StructType([]))
        assert error.value.getCondition() == "AXIS_LENGTH_MISMATCH"
        assert error.value.getMessageParameters() == {
            "expected_length": "0", "actual_length": "1"}
        assert state["relations"] == []
        spark.createDataFrame(frame_class(index=range(2)), T.StructType([]))


@pytest.mark.parametrize("frame_class", [pd.DataFrame, _PandasSubclass], ids=["frame", "subclass"])
@pytest.mark.parametrize("schema", [None, ["p", "q"], ["p"]], ids=["inferred", "names", "fewer"])
def test_pandas_inferred_types_are_those_of_arrow_as_pyspark(protocol, schema, frame_class):
    # As PySpark Connect's, of a frame of any class: without a schema, each column converts
    # with PySpark's Arrow conversion, of the type that Arrow infers (a struct of a column of
    # dicts), and the table goes without a schema, of columns _0, _1, ..., which toDF names,
    # by the labels, or the names and _n past them.
    frame = frame_class({"a": np.array([1, 2], dtype=np.int8), "b": [{"k": 1}, {"k": 2}]})
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        table = state["input_tables"][0]
        assert table.schema.names == ["_0", "_1"]
        assert table.schema.types == [pa.int8(), pa.struct([("k", pa.int64())])]
        assert table.column("_1").to_pylist() == [{"k": 1}, {"k": 2}]
        assert state["spark_schema"] is None
        assert state["column_names"] == {None: ["a", "b"], 2: ["p", "q"], 1: ["p", "_2"]}[
            None if schema is None else len(schema)]


def test_pandas_inferred_dicts_are_maps_of_infer_pandas_dict_as_map(protocol):
    # As PySpark Connect's: of spark.sql.execution.pandas.inferPandasDictAsMap, a column of dicts
    # is a map of the type of the first key, and the table takes the names of the columns.
    frame = pd.DataFrame({"a": [{"k": 1}, {"j": 2}], "b": ["x", "y"]})
    table = _pandas_table(protocol, frame, None, **{
        "spark.sql.execution.pandas.inferPandasDictAsMap": "true"})
    assert table.schema.names == ["a", "b"]
    assert table.schema.field("a").type == pa.map_(pa.string(), pa.int64())
    assert table.column("a").to_pylist() == [[("k", 1)], [("j", 2)]]


@pytest.mark.parametrize("frame_class", [pd.DataFrame, _PandasSubclass], ids=["frame", "subclass"])
@pytest.mark.parametrize("host", ["America/Los_Angeles", "UTC"])
def test_pandas_inferred_naive_timestamp_uses_session_zone(protocol, host, frame_class):
    # As PySpark Connect's, of a frame of any class: a naive time is a time of the session
    # zone, whatever the client's.
    frame = frame_class({"t": [pd.Timestamp("2020-07-01 12:00")]})
    table = _in_host_zone(host, lambda: _pandas_table(
        protocol, frame, None, **{"spark.sql.session.timeZone": "America/Los_Angeles"}))
    assert table.column("_0").to_pylist() == [datetime(2020, 7, 1, 19, tzinfo=timezone.utc)]


def test_pandas_inferred_conversion_errors_are_those_of_pyspark(protocol):
    from pyspark.errors import PySparkValueError

    with _arrow_input_service(protocol) as (spark, state):
        # As PySpark Connect's: Arrow's error of a column of values of several types.
        with pytest.raises(PySparkValueError, match="Exception thrown when converting "
                           "pandas.Series \\(object\\) with name 'a' to Arrow Array"):
            spark.createDataFrame(pd.DataFrame({"a": [1, "x"]}))
        with pytest.raises(PySparkValueError) as error:
            spark.createDataFrame(pd.DataFrame({"a": [1]}), ["p", "q"])
        assert error.value.getCondition() == "AXIS_LENGTH_MISMATCH"
        assert error.value.getMessageParameters() == {"expected_length": "2", "actual_length": "1"}


@pytest.mark.parametrize("host", ["America/Los_Angeles", "UTC"])
def test_pandas_naive_timestamp_uses_session_zone(protocol, host):
    frame = pd.DataFrame({"t": [pd.Timestamp("2020-07-01 12:00")]})
    # Official with session UTC, whatever the client zone: 12:00 UTC.
    table = _in_host_zone(
        host,
        lambda: _pandas_table(
            protocol, frame, "t timestamp",
            **{"spark.sql.session.timeZone": "UTC"},
        ),
    )
    assert table.column("t").to_pylist() == [datetime(2020, 7, 1, 12, tzinfo=timezone.utc)]


@pytest.mark.parametrize("frame_class", [pd.DataFrame, _PandasSubclass], ids=["frame", "subclass"])
def test_pandas_naive_timestamp_in_session_zone_is_converted(protocol, frame_class):
    pytz = pytest.importorskip("pytz")
    frame = frame_class({"t": [pd.Timestamp("2020-07-01 12:00")]})
    # Official with session America/Los_Angeles and a client zone of UTC, of a frame of any
    # class: 19:00 UTC.
    table = _in_host_zone(
        "UTC",
        lambda: _pandas_table(
            protocol, frame, "t timestamp",
            **{"spark.sql.session.timeZone": "America/Los_Angeles"},
        ),
    )
    assert table.column("t").to_pylist() == [datetime(2020, 7, 1, 19, tzinfo=timezone.utc)]

    gap = frame_class({"t": [pd.Timestamp("2020-03-08 02:30")]})
    with _arrow_input_service(protocol, **{
        "spark.sql.session.timeZone": "America/Los_Angeles",
    }) as (spark, _):
        # Official: pytz NonExistentTimeError for a naive time inside the DST gap.
        with pytest.raises(pytz.exceptions.NonExistentTimeError):
            spark.createDataFrame(gap, "t timestamp")


def test_pandas_nanosecond_timestamp_raises_as_pyspark(protocol):
    from pyspark.errors import PySparkValueError

    frame = pd.DataFrame({"t": [pd.Timestamp("2020-01-01 00:00:00.000000001")]})
    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(PySparkValueError) as error:
            spark.createDataFrame(frame, "t timestamp")
    assert str(error.value) == (
        "Exception thrown when converting pandas.Series (datetime64[ns, UTC]) with name 't' "
        "to Arrow Array (timestamp[us, tz=UTC]). It can be caused by overflows or other unsafe "
        "conversions warned by Arrow. Arrow safe type check can be disabled by using SQL config "
        "`spark.sql.execution.pandas.convertToArrowArraySafely`."
    )


def test_pandas_lone_surrogate_string_raises_as_pyspark(protocol):
    from pyspark.errors import PySparkValueError

    frame = pd.DataFrame({"s": ["a\ud800b"]}, dtype=object)
    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(PySparkValueError) as error:
            spark.createDataFrame(frame, "s string")
    assert str(error.value) == (
        "Exception thrown when converting pandas.Series (object) with name 's' to Arrow Array "
        "(string). It can be caused by overflows or other unsafe conversions warned by Arrow. "
        "Arrow safe type check can be disabled by using SQL config "
        "`spark.sql.execution.pandas.convertToArrowArraySafely`."
    )


def test_pandas_datetime_objects_in_date_column(protocol):
    frame = pd.DataFrame({"d": [datetime(2020, 1, 2, 3, 4)]}, dtype=object)
    table = _pandas_table(protocol, frame, "d date")
    # Official: [Row(d=datetime.date(2020, 1, 2))].
    assert table.column("d").type == pa.date32()
    assert table.column("d").to_pylist() == [date(2020, 1, 2)]


def test_pandas_nested_nan_becomes_null(protocol):
    array_frame = pd.DataFrame({"s": [[float("nan"), 1.0]]}, dtype=object)
    array_table = _pandas_table(protocol, array_frame, "s array<double>")
    # Official: [Row(s=[None, 1.0])].
    assert array_table.column("s").to_pylist() == [[None, 1.0]]

    map_frame = pd.DataFrame({"s": [{"k": float("nan")}]}, dtype=object)
    map_table = _pandas_table(protocol, map_frame, "s map<string,double>")
    # Official: [Row(s={'k': None})].
    assert map_table.column("s").to_pylist() == [[("k", None)]]


@pytest.mark.parametrize("schema", [None, ["x"]], ids=["inferred", "names"])
def test_pandas_inferred_int_beyond_int64_raises_as_pyspark(protocol, schema):
    frame = pd.DataFrame({"x": [2**64]}, dtype=object)
    with _arrow_input_service(protocol) as (spark, state):
        # Official: OverflowError, not a double.
        with pytest.raises(OverflowError) as error:
            spark.createDataFrame(frame, schema)
    assert str(error.value) == "Python int too large to convert to C long"


# Exact built-in date columns reuse Arrow without materializing pandas rows.

@pytest.mark.parametrize("accessor", ["values", "to_numpy"])
def test_date_series_accessors_do_not_replace_validated_cells(monkeypatch, accessor):
    from pyspark.sql._pandas_input import prepare_date_table

    dates = [date.min, date.max, date(1969, 12, 31), date(2000, 2, 29)] * 2500
    nullable = [None if i % 7 == 0 else value for i, value in enumerate(dates)]
    expected = {"renamed": dates, "nullable": nullable}
    frame = pd.DataFrame({"original": dates, "nullable": nullable}, dtype=object)
    schema = T.StructType([
        T.StructField("renamed", T.DateType(), False),
        T.StructField("nullable", T.DateType()),
    ])
    row_values = frame.astype(object).where(frame.notna(), None).values.tolist()
    calls = []

    def replacement(series, *args, **kwargs):
        calls.append(accessor)
        return np.full(len(series), date(2021, 2, 3), dtype=object)

    monkeypatch.setattr(pd.Series, accessor,
                        property(replacement) if accessor == "values" else replacement)
    table = prepare_date_table(frame, schema)
    assert table is not None
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    actual = pa.ipc.open_stream(sink.getvalue()).read_all()
    assert actual.to_pydict() == expected
    assert frame.astype(object).where(frame.notna(), None).values.tolist() == row_values
    assert calls == []


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("verify", [False, True])
def test_date_columns_preserve_every_ipc_cell(protocol, monkeypatch, cached, verify):
    from pyspark.sql.pandas.types import from_arrow_metadata

    dates = [date.min, date.max, date(4, 2, 29), date(1600, 2, 29),
             date(1900, 2, 28), date(1900, 3, 1), date(1969, 12, 31),
             date(1970, 1, 1), date(2000, 2, 29), date(2100, 3, 1)] * 1000
    nullable = [None if i % 7 == 0 else value for i, value in enumerate(reversed(dates))]
    expected = [dates, nullable, [None] * len(dates)]
    frame = pd.DataFrame(dict(zip(["original", "b", "c"], expected)), dtype=object)
    schema = T.StructType([
        T.StructField("renamed.`day", T.DateType(), False, {"source": "date", "id": 3}),
        T.StructField("nullable", T.DateType(), True, {"tag": [1, 2]}),
        T.StructField("all_null", T.DateType()),
    ])

    def no_rows(*args, **kwargs):
        raise AssertionError("date fast path must not materialize Python rows")

    for method in ("to_numpy", "iterrows", "itertuples"):
        monkeypatch.setattr(pd.DataFrame, method, no_rows)
    monkeypatch.setattr(pd.DataFrame, "values", property(no_rows))
    monkeypatch.setattr(pd.Series, "tolist", no_rows)
    overrides = {"spark.sql.session.localRelationCacheThreshold": "1" if cached else "999999999"}
    with _arrow_input_service(protocol, **overrides) as (spark, state):
        df = spark.createDataFrame(frame, schema, verifySchema=verify)
        assert not df.isLocal()
        df.toArrow()
        assert state["column_names"] == schema.names
        table = state["input_tables"][0]
        assert table.num_rows == len(dates)
        assert table.schema.names == schema.names
        epoch = date(1970, 1, 1)
        for column, values, field in zip(table.columns, expected, schema.fields):
            assert column.type == pa.date32()
            # Independent expectations cover every date and validity bit in IPC.
            assert column.cast(pa.int32()).to_pylist() == [
                None if value is None else (value - epoch).days for value in values
            ]
            assert column.to_pylist() == values
            actual_field = table.schema.field(field.name)
            assert actual_field.nullable == field.nullable
            # As PySpark's conversion of the frame, of no metadata of the field in Arrow.
            assert from_arrow_metadata(actual_field.metadata) is None
        assert bool(state["uploads"]) == cached


@pytest.mark.parametrize("reason", [
    "small", "empty", "inferred", "names", "ddl", "cast", "mixed_schema",
    "width", "duplicate_input", "duplicate_schema",
    "input_number", "datetime", "timestamp", "nan", "pd_na", "nat", "mixed",
    "nonnullable_null", "frame_subclass", "date_subclass", "late_unsupported",
])
def test_date_column_guard_preserves_row_fallback(reason):
    from pyspark.sql._pandas_input import prepare_date_table

    class Frame(pd.DataFrame):
        pass

    class Day(date):
        def toordinal(self):
            raise AssertionError("guard must not call conversion hooks")

    values = [date(2020, 1, 1)] * 10000
    frame = pd.DataFrame({"day": values}, dtype=object)
    schema = T.StructType([T.StructField("day", T.DateType())])
    if reason == "small":
        frame = frame.iloc[:9999]
    elif reason == "empty":
        frame = frame.iloc[:0]
    elif reason in ("inferred", "names", "ddl", "cast"):
        schema = {"inferred": None, "names": ["day"], "ddl": "day DATE",
                  "cast": T.StructType([T.StructField("day", T.StringType())])}[reason]
    elif reason == "mixed_schema":
        frame["long"] = np.arange(10000)
        schema.add("long", T.LongType())
    elif reason == "width":
        schema.add("extra", T.DateType())
    elif reason in ("duplicate_input", "duplicate_schema"):
        frame["other"] = values
        schema.add("other", T.DateType())
        if reason == "duplicate_input":
            frame.columns = ["day", "day"]
        else:
            schema = T.StructType([T.StructField("day", T.DateType())] * 2)
    elif reason == "input_number":
        frame.columns = [1]
    elif reason == "frame_subclass":
        frame = Frame(frame)
    else:
        value = {
            "datetime": datetime(2020, 1, 1), "timestamp": pd.Timestamp("2020-01-01"),
            "nan": float("nan"), "pd_na": pd.NA, "nat": pd.NaT, "mixed": "2020-01-01",
            "nonnullable_null": None, "date_subclass": Day(2020, 1, 1),
            "late_unsupported": Day(2020, 1, 1),
        }[reason]
        if reason == "nonnullable_null":
            schema = T.StructType([T.StructField("day", T.DateType(), False)])
        if reason == "late_unsupported":
            frame["other"] = values
            schema.add("other", T.DateType())
        frame.iat[-1, -1] = value
    assert prepare_date_table(frame, schema) is None


@pytest.mark.parametrize("reason", ["date_subclass", "frame_subclass", "nan", "pd_na", "nat",
                                    "nonnullable_null", "inferred", "ddl", "small"])
def test_date_public_fallback_converts_as_pyspark(protocol, monkeypatch, reason):
    # A frame of object dates that the fast path declines, for each of its reasons, converts as
    # PySpark's conversion does: it hands the column to Arrow, which reads the fields of a date
    # subclass, without its toordinal, and it never calls the frame's astype. A null in a
    # non-nullable field fails the cast of the table, as PySpark's.
    calls = []

    class Day(date):
        def toordinal(self):
            calls.append("toordinal")
            return super().toordinal() + 7

    class Frame(pd.DataFrame):
        pass

    values = [date(2020, 1, 1)] * (9999 if reason == "small" else 10000)
    if reason in ("nan", "pd_na", "nat", "nonnullable_null"):
        values[-1] = {"nan": float("nan"), "pd_na": pd.NA, "nat": pd.NaT,
                      "nonnullable_null": None}[reason]
    elif reason == "date_subclass":
        values[-1] = Day(2020, 1, 1)
    frame_type = Frame if reason == "frame_subclass" else pd.DataFrame
    frame = frame_type({"day": values}, dtype=object)
    schema = T.StructType([T.StructField("day", T.DateType(), reason != "nonnullable_null")])
    if reason == "inferred":
        schema = None
    elif reason == "ddl":
        schema = "day DATE"
    astype = pd.DataFrame.astype

    def record_rows(self, *args, **kwargs):
        calls.append("astype")
        return astype(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "astype", record_rows)
    with _arrow_input_service(protocol) as (spark, state):
        if reason == "nonnullable_null":
            with pytest.raises(ValueError) as error:
                spark.createDataFrame(frame, schema)
            assert type(error.value) is ValueError
            assert str(error.value) == "Casting field 'day' with null values to non-nullable"
            assert calls == []
            assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
            assert state["relations"] == []
            return
        spark.createDataFrame(frame, schema).toArrow()
        assert calls == []
        actual = state["input_tables"][0].column(0).to_pylist()
        expected = [date(2020, 1, 1)] * len(values)
        if reason in ("nan", "pd_na", "nat"):
            expected[-1] = None
        assert actual == expected


_DATE_RELATION = {"fields": [{"metadata": {"source": "native"}, "name": "day", "nullable": True,
                              "type": "date"}], "type": "struct"}


def _date_converts(name="day", nullable=True, metadata=None, relation=_DATE_RELATION):
    from datetime import date

    return _guard_converts([(name, "date32[day]", nullable, [date(2020, 1, 1)] * 10000,
                             metadata)], relation)


# PySpark's outcomes (see `_INTEGER_DESCRIPTORS`), and the drop-in's where they differ.
_DATE_DESCRIPTORS = {
    ("StructType", "fields"): _guard_raises(*_no_setter("StructType")),
    ("StructType", "names"): _guard_raises(*_no_setter("StructType")),
    ("StructType", "__iter__"): _guard_raises(*_rename_mismatch("day")),
    ("StructField", "name"): _guard_raises(*_no_setter("StructField")),
    ("StructField", "dataType"): _guard_raises(*_no_setter("StructField")),
    ("StructField", "nullable"): _guard_raises(*_no_setter("StructField")),
    ("StructField", "metadata"): _guard_raises(*_no_setter("StructField")),
}
_DATE_DESCRIPTORS_OF_THE_DROP_IN = {
    ("StructType", "fields"): _guard_converts([], {"fields": [], "type": "struct"}),
    ("StructType", "names"): _date_converts(name="changed"),
    ("StructField", "name"): _guard_raises(*_rename_mismatch("day")),
    ("StructField", "dataType"): _date_converts(),
    ("StructField", "nullable"): _date_converts(
        nullable=False, relation=_relation_of("day", "date", False, {"source": "native"})),
    ("StructField", "metadata"): _date_converts(
        metadata={"SPARK::metadata::json": '{"changed": true}'},
        relation=_relation_of("day", "date", True, {"changed": True})),
}


@pytest.mark.parametrize("class_name,attribute", [
    ("StructType", "names"), ("StructType", "fields"), ("StructType", "__iter__"),
    ("StructField", "name"), ("StructField", "dataType"), ("StructField", "nullable"),
    ("StructField", "metadata"),
    ("StructType", "__getattribute__"), ("StructField", "__getattribute__"),
    ("DateType", "__getattribute__"), ("StructType", "__class__"),
    ("StructField", "__class__"), ("DateType", "__class__"),
])
@pytest.mark.parametrize("before_import", [False, True])
def test_date_custom_schema_descriptors_take_pyspark_conversion(class_name, attribute,
                                                                before_import):
    # A patch of a class declines the fast path of object dates: the frame converts with
    # PySpark's conversion, which reads the patch (see `_integer_guard_process`).
    script = r'''
import sys
from datetime import date
import pandas as pd
from pyspark.sql import types as T

frame = pd.DataFrame({"day": [date(2020, 1, 1)] * 10000}, dtype=object)
schema = T.StructType([T.StructField("day", T.DateType(), True, {"source": "native"})])
class_name, attribute, before_import = sys.argv[1:4]
if before_import == "False":
    import pyspark.sql._pandas_input
cls = getattr(T, class_name)
if attribute == "__getattribute__":
    replacement = lambda self, name: object.__getattribute__(self, name)
elif attribute == "__iter__":
    replacement = lambda self: iter([T.StructField("changed", T.DateType())])
elif attribute == "__class__":
    attribute = "__getattribute__"
    replacement = lambda self, name: object if name == "__class__" else (
        object.__getattribute__(self, name))
else:
    value = {"names": ["changed"], "fields": [], "name": "changed",
             "dataType": T.DateType(), "nullable": False, "metadata": {"changed": True}}[attribute]
    replacement = property(lambda self: value)
setattr(cls, attribute, replacement)
import importlib.util
spec = importlib.util.spec_from_file_location("date_schema_capture", sys.argv[-1])
capture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capture
spec.loader.exec_module(capture)
'''
    outcome = _guard_outcome(script, class_name, attribute, str(before_import))
    _assert_guard(outcome, _DATE_DESCRIPTORS.get((class_name, attribute), _date_converts()),
                  _DATE_DESCRIPTORS_OF_THE_DROP_IN.get((class_name, attribute)))


@pytest.mark.parametrize("dtype_mode", ["equal", "class_spoof", "exact_numpy"])
def test_date_custom_extension_dtype_takes_pyspark_conversion(protocol, dtype_mode):
    # As for object strings: an extension array of a dtype that equals, or claims to be, NumPy's
    # object dtype does not select the fast path of object dates, and PySpark's conversion
    # takes the values that its `__arrow_array__` gives.
    from pandas.api.extensions import ExtensionArray, ExtensionDtype, take
    from pyspark.sql._pandas_input import prepare_date_table

    arrow_calls = []

    class ObjectEqualDtype(ExtensionDtype):
        name = "object_equal_date"
        type = date
        kind = "O"
        na_value = None

        @classmethod
        def construct_array_type(cls):
            return DifferentArrowArray

        def __eq__(self, other):
            return (isinstance(other, ObjectEqualDtype)
                    or isinstance(other, np.dtype) and other.kind == "O")

        def __hash__(self):
            return hash(self.name)

    class DifferentArrowArray(ExtensionArray):
        def __init__(self, values):
            self._values = np.asarray(values, dtype=object)

        @property
        def dtype(self):
            return ObjectEqualDtype()

        @property
        def nbytes(self):
            return self._values.nbytes

        def __len__(self):
            return len(self._values)

        def __getitem__(self, key):
            result = self._values[key]
            return type(self)(result) if isinstance(result, np.ndarray) else result

        def isna(self):
            return pd.isna(self._values)

        def take(self, indices, allow_fill=False, fill_value=None):
            return type(self)(take(self._values, indices, allow_fill=allow_fill,
                                   fill_value=fill_value))

        def copy(self):
            return type(self)(self._values.copy())

        @classmethod
        def _from_sequence(cls, scalars, *, dtype=None, copy=False):
            values = np.asarray(scalars, dtype=object)
            return cls(values.copy() if copy else values)

        def to_numpy(self, dtype=None, copy=False, na_value=None):
            if dtype is not None:
                return self._values.astype(dtype, copy=copy)
            return self._values.copy() if copy else self._values

        def __arrow_array__(self, type=None):
            arrow_calls.append("arrow")
            # A scan of to_numpy() cannot establish the values this hook emits.
            return pa.array([date(2020, 1, 2)] * len(self), type=type)

    array = DifferentArrowArray([date(2020, 1, 1)] * 10000)
    frame = pd.DataFrame({"day": pd.Series(array, copy=False)})
    schema = T.StructType([T.StructField("day", T.DateType())])
    if dtype_mode == "exact_numpy":
        # An ExtensionArray may change its advertised dtype after construction.
        # The dtype and NumPy scan both pass while its Arrow hook changes cells.
        DifferentArrowArray.dtype = property(lambda self: np.dtype("object"))
        DifferentArrowArray.__array__ = lambda self, dtype=None, copy=None: self._values
    elif dtype_mode == "class_spoof":
        def spoof_class(self, name):
            if name == "__class__":
                return type(np.dtype("object"))
            return object.__getattribute__(self, name)

        ObjectEqualDtype.__getattribute__ = spoof_class
    dtype = frame.dtypes.iloc[0]
    assert type(frame) is pd.DataFrame
    assert type(frame.iloc[:, 0].array) is DifferentArrowArray
    assert type(dtype) is (type(np.dtype("object")) if dtype_mode == "exact_numpy"
                           else ObjectEqualDtype)
    assert isinstance(dtype, np.dtype) is (dtype_mode != "equal")
    assert dtype == np.dtype("object")
    assert prepare_date_table(frame, schema) is None
    assert arrow_calls == []
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert state["input_tables"][0].column(0).to_pylist() == [date(2020, 1, 2)] * 10000
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
    assert arrow_calls == ["arrow"]


def test_date_rebound_dataframe_alias_keeps_subclass_off_the_fast_path(monkeypatch):
    from pyspark.sql._pandas_input import prepare_date_table

    class Frame(pd.DataFrame):
        pass

    frame = Frame({"day": [date(2020, 1, 1)] * 10000}, dtype=object)
    schema = T.StructType([T.StructField("day", T.DateType())])
    monkeypatch.setattr(pd, "DataFrame", Frame)
    assert prepare_date_table(frame, schema) is None


def _date_alias_guard_process(mode, argument):
    """The outcome of `createDataFrame` of a frame of object dates under `renamed date` of a name
    rebound by `mode` and `argument`, in a process of its own, which the session converts with
    PySpark's conversion (see `_integer_guard_process`)."""
    if mode == "captured_alias":
        import pyspark.sql._arrow_input as importer
        if not hasattr(importer, "to_arrow_schema"):
            pytest.skip("this importer resolves converters locally and holds no captured aliases")

    script = r'''
import importlib
import importlib.util
import sys
from datetime import date
import pandas as pd
from pyspark.sql import types as T
from pyspark import _pyspark as native

spec = importlib.util.spec_from_file_location("date_alias_capture", sys.argv[-1])
capture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capture
spec.loader.exec_module(capture)
values = [date.min, date.max, date(1969, 12, 31), None] * 2500
frame = pd.DataFrame({"original": values}, dtype=object)
schema = T.StructType([T.StructField("renamed", T.DateType(), True, {"source": "native"})])
mode, argument = sys.argv[1:3]
if mode in ("public_alias", "converter_alias", "captured_alias"):
    replacement = native.BooleanType if argument == "DateType" else native.DateType
    if mode == "public_alias":
        sys.modules.pop("pyspark.sql.pandas.types", None)
        setattr(T, argument, replacement)
    else:
        converter = importlib.import_module("pyspark.sql.pandas.types")
        if mode == "captured_alias":
            importer = importlib.import_module("pyspark.sql._arrow_input")
        setattr(converter, argument, replacement)
        if mode == "captured_alias":
            sys.modules.pop("pyspark.sql.pandas.types")
            current = importlib.import_module("pyspark.sql.pandas.types")
            assert current is not converter
            assert importer.to_arrow_schema.__globals__ is vars(converter)
elif mode == "atomic_alias":
    scope, replacement_name = argument.split(":")
    replacement = None if replacement_name == "none" else native.DateType
    if scope == "public_before":
        sys.modules.pop("pyspark.sql._arrow_input", None)
        T.AtomicType = replacement
    elif scope == "public_after":
        importlib.import_module("pyspark.sql._arrow_input")
        T.AtomicType = replacement
    else:
        importer = importlib.import_module("pyspark.sql._arrow_input")
        if replacement_name == "missing":
            del importer.AtomicType
        else:
            importer.AtomicType = replacement
elif mode == "importer_alias":
    importer = importlib.import_module("pyspark.sql._arrow_input")
    importer.StructType = None if argument == "none" else native.DateType
elif mode == "importer_function":
    importer = importlib.import_module("pyspark.sql._arrow_input")
    setattr(importer, argument, None)
else:
    raise AssertionError(mode)
'''
    return _guard_outcome(script, mode, argument)


def _date_alias_converts():
    from datetime import date

    return _guard_converts(
        [("renamed", "date32[day]", True, [date.min, date.max, date(1969, 12, 31), None] * 2500)],
        {"fields": [{"metadata": {"source": "native"}, "name": "renamed", "nullable": True,
                     "type": "date"}], "type": "struct"})


def _date_conversion_error(error, arrow_type):
    return _guard_raises(*_conversion_error(error, "object", "original", arrow_type))


# PySpark's outcomes of a name rebound to `DateType` (to `BooleanType` of `DateType`):
# `to_arrow_type` takes the date field of the first class that it matches, whose conversion of
# the column of dates PyArrow fails, or the converter of the series reads it as a decimal or a
# struct; a `StructField` of `DateType` is called with the arguments of a field. A public
# `StructType` rebound before the session is the one that PySpark's createDataFrame checks its
# schema against (see `_ATOMIC_ALIAS_OF_PYSPARK`), which raises `NOT_EXPECTED_TYPE` of a struct
# before any setting is read.
_DATE_REBOUND = {
    "BooleanType": _date_conversion_error("PySparkValueError", "bool"),
    "ByteType": _date_conversion_error("PySparkValueError", "int8"),
    "ShortType": _date_conversion_error("PySparkValueError", "int16"),
    "IntegerType": _date_conversion_error("PySparkValueError", "int32"),
    "LongType": _date_conversion_error("PySparkValueError", "int64"),
    "FloatType": _date_conversion_error("PySparkValueError", "float"),
    "DoubleType": _date_conversion_error("PySparkValueError", "double"),
    "DecimalType": _guard_raises("builtins.AttributeError",
                                 "'DateType' object has no attribute 'precision'"),
    "StringType": _date_conversion_error("PySparkTypeError", "string"),
    "BinaryType": _date_conversion_error("PySparkTypeError", "binary"),
    "DateType": _guard_raises(*_unsupported_type("DateType()")),
    "StructField": _guard_raises(
        "builtins.TypeError",
        "DataTypeSingleton.__call__() got an unexpected keyword argument 'nullable'"),
    "StructType": _guard_raises("builtins.AttributeError",
                                "'DateType' object has no attribute 'names'"),
}
_NOT_EXPECTED_STRUCT = _guard_raises(
    "pyspark.errors.exceptions.base.PySparkTypeError",
    "[NOT_EXPECTED_TYPE] Argument `schema` should be list, None or StructType, got StructType.",
    configs=False)


@pytest.mark.parametrize("class_name", [
    "BooleanType", "ByteType", "ShortType", "IntegerType", "LongType", "FloatType",
    "DoubleType", "DecimalType", "StringType", "BinaryType", "DateType", "StructField", "StructType",
])
@pytest.mark.parametrize("mode", ["public_alias", "converter_alias", "captured_alias"])
def test_date_rebound_converter_aliases_take_pyspark_conversion(class_name, mode):
    outcome = _date_alias_guard_process(mode, class_name)
    if mode == "captured_alias":
        _assert_guard(outcome, _date_alias_converts())
    elif mode == "public_alias" and class_name == "StructType":
        _assert_guard(outcome, _NOT_EXPECTED_STRUCT)
    else:
        _assert_guard(outcome, _DATE_REBOUND[class_name])


@pytest.mark.parametrize("replacement", ["none", "date"])
def test_date_rebound_importer_schema_alias_takes_pyspark_conversion(replacement):
    _assert_guard(_date_alias_guard_process("importer_alias", replacement), _date_alias_converts())


@pytest.mark.parametrize("scope", ["public_before", "public_after", "importer"])
@pytest.mark.parametrize("replacement", ["none", "date"])
def test_date_rebound_atomic_alias_takes_pyspark_conversion(scope, replacement):
    # See `test_integer_rebound_atomic_alias_takes_pyspark_conversion`.
    outcome = _date_alias_guard_process("atomic_alias", f"{scope}:{replacement}")
    if scope != "importer" and replacement == "none":
        _assert_guard(outcome, _ATOMIC_ALIAS_OF_PYSPARK)
    else:
        _assert_guard(outcome, _date_alias_converts())


def test_date_deleted_importer_atomic_alias_takes_pyspark_conversion():
    _assert_guard(_date_alias_guard_process("atomic_alias", "importer:missing"),
                  _date_alias_converts())


@pytest.mark.parametrize("name", ["from_arrow_schema", "to_arrow_schema"])
def test_date_present_invalid_importer_function_takes_pyspark_conversion(name):
    _assert_guard(_date_alias_guard_process("importer_function", name), _date_alias_converts())


def test_date_warm_importer_without_global_converter_aliases(protocol, monkeypatch):
    import sys
    import types

    import pyspark.sql._arrow_input as importer

    original = importer.prepare_table

    def local_imports(table, schema, *configs):
        from pyspark.sql.pandas.types import from_arrow_schema, to_arrow_schema

        table, resolved, names = original(table, schema, *configs)
        assert from_arrow_schema(table.schema).fields
        assert to_arrow_schema(resolved).names == table.column_names
        return table, resolved, names

    values = [date.min, date.max, date(1970, 1, 1), None] * 2500
    frame = pd.DataFrame({"day": values}, dtype=object)
    schema = T.StructType([T.StructField("renamed", T.DateType())])
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(pa.table({"warm": [1, 2]})).toArrow()
        # Exercise the local-import shape on this branch's pinned dependency.
        substitute = types.ModuleType(importer.__name__)
        substitute.StructType = T.StructType
        substitute.AtomicType = T.AtomicType
        substitute.prepare_table = local_imports
        monkeypatch.setitem(sys.modules, importer.__name__, substitute)

        def no_rows(*args, **kwargs):
            raise AssertionError("warm date input must stay columnar")

        monkeypatch.setattr(pd.DataFrame, "values", property(no_rows))
        for _ in range(3):
            spark.createDataFrame(frame, schema).toArrow()
        for table in state["input_tables"][1:]:
            assert table.column_names == ["renamed"]
            assert table.column(0).to_pylist() == values


@pytest.mark.parametrize("location", ["class", "instance"])
@pytest.mark.parametrize("method", ["astype", "notna", "where"])
def test_date_custom_normalization_hooks_are_not_called_as_in_pyspark(
    protocol, monkeypatch, location, method
):
    # A frame whose notna, astype or where is customized, of its class or of the instance,
    # declines the fast path of object dates, and converts as PySpark's conversion does, which
    # never calls them: the values are the frame's, whatever the hooks would change.
    from types import MethodType

    from pyspark.sql._pandas_input import prepare_date_table

    values = [date(2020, 1, 1)] * 10000
    frame = pd.DataFrame({"original": values}, dtype=object)
    schema = T.StructType([T.StructField("renamed", T.DateType())])
    original = getattr(pd.DataFrame, method)
    calls = []

    def replacement(self, *args, **kwargs):
        calls.append(method)
        result = original(self, *args, **kwargs)
        if method == "notna":
            result.iat[-1, 0] = False
        else:
            result.iat[0, 0] = date(2020, 1, 2 if method == "astype" else 3)
        return result

    if location == "class":
        monkeypatch.setattr(pd.DataFrame, method, replacement)
    else:
        # Avoid pandas' attribute-to-column routing when installing an instance hook.
        monkeypatch.setitem(vars(frame), method, MethodType(replacement, frame))
    assert prepare_date_table(frame, schema) is None
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
        table = state["input_tables"][0]
        assert table.column_names == ["renamed"]
        assert table.column(0).to_pylist() == values
    assert calls == []


def _date_columns_guard_input(subclass=False):
    """Three date columns, of a DataFrame subclass with `subclass`, which the fast path declines
    and PySpark's conversion converts as any other frame."""
    class Frame(pd.DataFrame):
        pass

    days = [date.min, date.max, date(4, 2, 29), date(1600, 2, 29),
            date(1900, 2, 28), date(1900, 3, 1), date(1969, 12, 31),
            date(1970, 1, 1), date(2000, 2, 29), date(2100, 3, 1)] * 1000
    nullable = [None if i % 7 == 0 else value for i, value in enumerate(reversed(days))]
    expected = [days, nullable, [None] * len(days)]
    frame_type = Frame if subclass else pd.DataFrame
    frame = frame_type(dict(zip(["original", "b", "c"], expected)), dtype=object)
    schema = T.StructType([
        T.StructField("renamed.`day", T.DateType(), False, {"source": "date", "id": 3}),
        T.StructField("nullable", T.DateType(), True, {"tag": [1, 2]}),
        T.StructField("all_null", T.DateType(), True, {"source": "all-null"}),
    ])
    return frame, schema, expected


def _assert_date_columns_guard_payload(state, schema, expected):
    import json

    assert len(state["input_tables"]) == 1
    assert state["column_names"] == schema.names
    table = state["input_tables"][0]
    assert table.num_rows == 10000
    assert table.column_names == ["renamed.`day", "nullable", "all_null"]
    assert not state["uploads"]
    epoch = date(1970, 1, 1)
    for column, values, field in zip(table.columns, expected, schema.fields):
        assert column.type == pa.date32()
        assert column.to_pylist() == values
        assert column.cast(pa.int32()).to_pylist() == [
            None if value is None else (value - epoch).days for value in values
        ]
        assert column.null_count == sum(value is None for value in values)
        physical = table.schema.field(field.name)
        assert physical.nullable is field.nullable
        # The fast path's IPC, as that of PySpark's conversion, carries no metadata of the
        # fields, which the logical schema has.
        assert physical.metadata is None
    assert json.loads(state["relations"][0].local_relation.schema) == {
        "type": "struct", "fields": [
            {"name": "renamed.`day", "type": "date", "nullable": False,
             "metadata": {"source": "date", "id": 3}},
            {"name": "nullable", "type": "date", "nullable": True,
             "metadata": {"tag": [1, 2]}},
            {"name": "all_null", "type": "date", "nullable": True,
             "metadata": {"source": "all-null"}},
        ],
    }
    # As PySpark's createDataFrame, both paths read the settings before any conversion, and the
    # 120 KB of Arrow data are sent inline below the default threshold.
    assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]


@pytest.mark.parametrize("subclass", [False, True])
@pytest.mark.parametrize("location", [
    "subclass", "class", "owner", "backing_subclass_item", "backing_class_iter",
    "backing_class_getitem", "backing_instance_box",
])
@pytest.mark.parametrize("behavior", ["raise", "mutate"])
def test_date_columns_iterator_hooks_run_in_pyspark_conversion(
    protocol, monkeypatch, subclass, location, behavior
):
    # As for int64 columns, of a plain frame or of a subclass: the fast path declines the
    # customized labels without iterating them, and PySpark's conversion iterates them, for
    # the columns of `items()`, before it reads any value. It never calls the frame's notna or
    # astype.
    from pandas.core.base import IndexOpsMixin

    frame, schema, values = _date_columns_guard_input(subclass)
    original_iter = pd.Index.__iter__
    original_notna, original_astype = frame.notna, frame.astype
    calls = []
    invoked = False

    class ColumnsError(RuntimeError):
        pass

    def notna(*args, **kwargs):
        calls.append("notna")
        return original_notna(*args, **kwargs)

    def astype(*args, **kwargs):
        calls.append("astype")
        return original_astype(*args, **kwargs)

    def before_hook():
        nonlocal invoked
        if invoked:
            return
        invoked = True
        calls.append("columns iterator")
        if behavior == "raise":
            raise ColumnsError("date columns hook precedes the conversion of the values")
        # The spies of the frame's normalization methods are installed once the conversion
        # iterates the labels, as the values that it reads next are changed: before, the fast
        # path's guards see the canonical methods.
        monkeypatch.setitem(vars(frame), "notna", notna)
        monkeypatch.setitem(vars(frame), "astype", astype)
        frame.iat[0, 0] = date(2004, 2, 29)
        frame.iat[-1, 1] = None
        frame.iat[17, 2] = date.min

    def iterate(index):
        if index is columns:
            before_hook()
        return original_iter(index)

    if location == "subclass":
        class Columns(pd.Index):
            pass
        monkeypatch.setattr(Columns, "__iter__", iterate)
        labels = pd.Index(["original", "b", "c"], dtype=object)
        frame.columns = Columns._simple_new(labels._values, name=None)
        assert type(frame.columns) is Columns
    elif location == "backing_subclass_item":
        # IndexOpsMixin iteration calls ndarray.item, rather than ndarray.__iter__.
        class Names(np.ndarray):
            def item(self, *args):
                before_hook()
                return super().item(*args)
        backing = np.asarray(["original", "b", "c"], dtype=object).view(Names)
        frame.columns = pd.Index._simple_new(backing, name=None)
        assert type(frame.columns._values) is Names
    elif location.startswith("backing_"):
        frame.columns = pd.Index(pd.array(["original", "b", "c"], dtype=pd.StringDtype(storage="python")))
        backing = frame.columns._values
        if location == "backing_class_iter":
            original = type(backing).__iter__

            def backing_iter(array):
                if array is backing:
                    before_hook()
                return original(array)

            monkeypatch.setattr(type(backing), "__iter__", backing_iter)
        elif location == "backing_class_getitem":
            original = type(backing).__getitem__

            def backing_getitem(array, key):
                if array is backing:
                    before_hook()
                return original(array, key)

            monkeypatch.setattr(type(backing), "__getitem__", backing_getitem)
        else:
            original = backing._box_func

            def box(value):
                before_hook()
                return original(value)

            monkeypatch.setitem(vars(backing), "_box_func", box)
    else:
        monkeypatch.setattr(pd.Index if location == "class" else IndexOpsMixin,
                            "__iter__", iterate)
    columns = frame.columns

    with _arrow_input_service(protocol) as (spark, state):
        if behavior == "raise":
            with pytest.raises(ColumnsError,
                               match="date columns hook precedes the conversion of the values"):
                spark.createDataFrame(frame, schema)
            assert calls == ["columns iterator"]
            assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
            assert not state["uploads"]
            assert not state["relations"]
            assert not state["input_tables"]
        else:
            spark.createDataFrame(frame, schema).toArrow()
            expected = [column.copy() for column in values]
            expected[0][0], expected[1][-1], expected[2][17] = date(2004, 2, 29), None, date.min
            _assert_date_columns_guard_payload(state, schema, expected)
            assert calls == ["columns iterator"]


@pytest.mark.parametrize("subclass", [False, True])
def test_date_rebound_index_values_raise_in_pyspark_conversion(protocol, monkeypatch, subclass):
    # As for int64 columns, PySpark's conversion raises the error of the labels' storage.
    frame, schema, _ = _date_columns_guard_input(subclass)
    columns = frame.columns
    original = pd.Index._values.fget
    calls = []

    def values(index):
        if index is columns:
            calls.append("_values")
            raise RuntimeError("date columns backing hook precedes the conversion of the values")
        return original(index)

    monkeypatch.setattr(pd.Index, "_values", property(values))
    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(RuntimeError,
                           match="date columns backing hook precedes the conversion of the values"):
            spark.createDataFrame(frame, schema)
        assert calls == ["_values"]
        assert state["config_requests"] == [("get", _CREATE_DATAFRAME_KEYS)]
        assert state["input_tables"] == []
        assert state["uploads"] == 0


@pytest.mark.parametrize("subclass", [False, True])
def test_date_rebound_index_data_reads_no_more_than_pyspark_conversion(
    protocol, monkeypatch, subclass
):
    frame, schema, values = _date_columns_guard_input(subclass)
    columns, backing = frame.columns, vars(frame.columns)["_data"]
    calls = []
    allowed_reads = None

    class ColumnsDataError(RuntimeError):
        pass

    def data(index):
        if index is columns:
            calls.append("_data")
            if allowed_reads is not None and calls.count("_data") > allowed_reads:
                raise ColumnsDataError("eligibility added a columns storage read")
            return backing
        return vars(index)["_data"]

    def set_data(index, value):
        vars(index)["_data"] = value

    monkeypatch.setattr(pd.Index, "_data", property(data, set_data), raising=False)
    import pyspark.sql._pandas_input as helper

    # The reads of PySpark's conversion of the frame, with both fast paths declined.
    with monkeypatch.context() as baseline:
        baseline.setattr(helper, "prepare_int64_table", lambda *_: None)
        baseline.setattr(helper, "prepare_date_table", lambda *_: None)
        with _arrow_input_service(protocol) as (spark, state):
            spark.createDataFrame(frame, schema).toArrow()
            _assert_date_columns_guard_payload(state, schema, values)
            control_table = state["input_tables"][0]
    allowed_reads = calls.count("_data")
    assert allowed_reads > 0
    calls.clear()
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        # The rebound storage declines the fast path without reading the labels more than
        # PySpark's conversion does. pandas caches some of these reads, so a later conversion
        # of the same frame can read them fewer times than the first.
        assert 0 < calls.count("_data") <= allowed_reads
        _assert_date_columns_guard_payload(state, schema, values)
        assert state["input_tables"][0].equals(control_table, check_metadata=True)


@pytest.mark.parametrize("storage", ["object", "default", "python", "pyarrow", "python_nan", "pyarrow_nan"])
def test_date_canonical_columns_storage_stays_columnar(protocol, monkeypatch, storage):
    frame, schema, values = _date_columns_guard_input()
    labels = ["original", "b", "c"]
    if storage == "object":
        frame.columns = pd.Index(labels, dtype=object)
    elif storage != "default":
        kind = storage.removesuffix("_nan")
        label_dtype = (pd.StringDtype(storage=kind, na_value=np.nan)
                       if storage.endswith("_nan") else pd.StringDtype(storage=kind))
        frame.columns = pd.Index(pd.array(labels, dtype=label_dtype))

    def no_rows(*args, **kwargs):
        raise AssertionError("ordinary date columns must retain columnar construction")

    monkeypatch.setattr(pd.DataFrame, "values", property(no_rows))
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        _assert_date_columns_guard_payload(state, schema, values)
