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

Public cases cover Spark-output values, schemas, chunks, and lifetimes. Isolated
checks observe buffers at Rust's C Stream export boundary and verify schema
preflight. Server-side Python addresses cannot prove handoff buffer sharing.
"""

import ctypes
import gc
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal

import grpc
import pandas as pd
import pyarrow as pa
import pytest
from pyspark.sql import SparkSession


def _varint(value):
    encoded = bytearray()
    while value > 127:
        encoded.append((value & 127) | 128)
        value >>= 7
    encoded.append(value)
    return bytes(encoded)


def _bytes_field(number, value):
    return _varint((number << 3) | 2) + _varint(len(value)) + value


def _request_fields(data):
    """Read only the protobuf envelope, not the plan or other nested messages."""
    position = 0

    def varint():
        nonlocal position
        value = shift = 0
        while True:
            byte = data[position]
            position += 1
            value |= (byte & 127) << shift
            if byte < 128:
                return value
            shift += 7

    fields = {}
    while position < len(data):
        tag = varint()
        wire_type = tag & 7
        if wire_type == 2:
            length = varint()
            fields[tag >> 3] = data[position : position + length]
            position += length
        elif wire_type == 0:
            varint()
        elif wire_type in (1, 5):
            position += 8 if wire_type == 1 else 4
        else:
            raise AssertionError(f"Unexpected protobuf wire type: {wire_type}")
    return fields


@contextmanager
def _dataframe(batches, *, payloads=None, calls=None):
    """Serve just ExecutePlan and release RPCs; unexpected RPCs fail closed.

    Field numbers are from spark/connect/base.proto. Using the wire envelope
    avoids depending on official pyspark's generated protobuf package, which
    cannot be imported alongside the Rust-backed drop-in under the same name.
    """
    if payloads is None:
        payloads = []
        for batch in batches:
            sink = pa.BufferOutputStream()
            with pa.ipc.new_stream(sink, batch.schema) as writer:
                writer.write_batch(batch)
            payloads.append((batch.num_rows, sink.getvalue().to_pybytes()))
    server_id = b"00000000-0000-0000-0000-000000000001"

    def execute(request, context):
        if calls is not None:
            calls["execute"] = calls.get("execute", 0) + 1
        fields = _request_fields(request)
        header = (
            _bytes_field(1, fields[1]) + _bytes_field(12, fields[6]) + _bytes_field(15, server_id)
        )
        for index, (rows, payload) in enumerate(payloads):
            arrow_batch = _varint(8) + _varint(rows) + _bytes_field(2, payload)
            yield (header + _bytes_field(13, str(index).encode()) + _bytes_field(2, arrow_batch))
        yield header + _bytes_field(13, b"complete") + _bytes_field(14, b"")

    def release_execute(request, context):
        fields = _request_fields(request)
        return _bytes_field(1, fields[1]) + _bytes_field(2, fields[3]) + _bytes_field(3, server_id)

    def release_session(request, context):
        return _bytes_field(1, _request_fields(request)[1]) + _bytes_field(2, server_id)

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


class _ArrowArray(ctypes.Structure):
    pass


_ArrowArray._fields_ = [
    ("length", ctypes.c_int64),
    ("null_count", ctypes.c_int64),
    ("offset", ctypes.c_int64),
    ("n_buffers", ctypes.c_int64),
    ("n_children", ctypes.c_int64),
    ("buffers", ctypes.POINTER(ctypes.c_void_p)),
    ("children", ctypes.POINTER(ctypes.POINTER(_ArrowArray))),
    ("dictionary", ctypes.POINTER(_ArrowArray)),
    ("release", ctypes.c_void_p),
    ("private_data", ctypes.c_void_p),
]


class _ArrowArrayStream(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_void_p)
        for name in ("get_schema", "get_next", "get_last_error", "release", "private_data")
    ]


_GetNext = ctypes.CFUNCTYPE(
    ctypes.c_int, ctypes.POINTER(_ArrowArrayStream), ctypes.POINTER(_ArrowArray)
)


def _snapshot(array):
    """Copy only addresses and shape while Rust's exported array is live."""
    return {
        "length": array.length,
        "offset": array.offset,
        "buffers": [array.buffers[i] for i in range(array.n_buffers)],
        "children": [_snapshot(array.children[i].contents) for i in range(array.n_children)],
    }


def _assert_buffers(array, exported):
    assert len(array) == exported["length"]
    assert array.offset == exported["offset"]
    own_buffers = array.buffers()[: len(exported["buffers"])]
    for buffer, address in zip(own_buffers, exported["buffers"]):
        if buffer is None:
            assert address is None
        elif buffer.size:
            assert buffer.address == address
    if pa.types.is_struct(array.type):
        assert array.type.num_fields == len(exported["children"])
        for index, child in enumerate(exported["children"]):
            _assert_buffers(array.field(index), child)
    elif pa.types.is_list(array.type) or pa.types.is_map(array.type):
        assert len(exported["children"]) == 1
        _assert_buffers(array.values, exported["children"][0])
    else:
        assert not exported["children"]


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


def _check_stream_buffers():
    # These proxies must be installed before arrow-pyarrow caches its classes.
    # Each check has a fresh process, so the caches cannot affect another test.
    batch = _mixed_batch()
    schema = pa.schema(
        [
            field.with_metadata({**field.metadata, b"nul\x00key": b"nul\x00value"})
            for field in batch.schema
        ],
        metadata={**batch.schema.metadata, b"nul\x00key": b"nul\x00value"},
    )
    batch = pa.RecordBatch.from_arrays(batch.columns, schema=schema)
    batches = [batch.slice(1, 2), batch.slice(0, 1), batch.slice(3, 1)]
    expected = pa.Table.from_batches(batches)
    observed, errors, callbacks, imports, next_calls = [], [], [], [], []
    real_reader = pa.RecordBatchReader

    class ReaderProxy:
        @staticmethod
        def _import_from_c(address):
            imports.append(address)
            stream = ctypes.cast(address, ctypes.POINTER(_ArrowArrayStream))
            original_next = _GetNext(stream.contents.get_next)

            @_GetNext
            def get_next(current, output):
                next_calls.append(None)
                result = original_next(current, output)
                if result == 0 and output.contents.release:
                    try:
                        observed.append(_snapshot(output.contents))
                    except Exception as error:
                        # Exceptions cannot cross a ctypes callback boundary.
                        errors.append(error)
                return result

            # Arrow keeps raw function pointers, not the Python callback owners.
            # Keep both alive until the native reader has consumed the stream.
            callbacks.extend((original_next, get_next))
            stream.contents.get_next = ctypes.cast(get_next, ctypes.c_void_p).value
            return real_reader._import_from_c(address)

    calls = {}

    def collect():
        with _dataframe(batches, calls=calls) as (frame, session):
            table = frame.toArrow()
            session.stop()
            return table

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pa, "RecordBatchReader", ReaderProxy)
        actual = collect()
        gc.collect()
        assert actual.equals(expected, check_metadata=True)
        assert calls == {"execute": 1}
        assert len(imports) == 1
        assert len(next_calls) == len(batches) + 1
        assert len(observed) == len(batches)
        assert not errors
        for batch_index, exported in enumerate(observed):
            assert exported["length"] == batches[batch_index].num_rows
            assert len(exported["children"]) == actual.num_columns
            for column, child in zip(actual.columns, exported["children"]):
                assert column.num_chunks == len(batches)
                _assert_buffers(column.chunk(batch_index), child)
        numeric = actual.column(0).chunk(0).slice(1, 1)
        strings = actual.column(2).chunk(2).slice(0, 1)
        addresses = [numeric.buffers()[1].address, strings.buffers()[2].address]
        del column, actual
        gc.collect()
        assert numeric.to_pylist() == [-3]
        assert strings.to_pylist() == ["omega"]
        assert [numeric.buffers()[1].address, strings.buffers()[2].address] == addresses


def _check_schema_preflight():
    batch = _mixed_batch()
    seen = []

    class SchemaProxy:
        @staticmethod
        def _import_from_c(address):
            seen.append("schema")
            raise RuntimeError("test schema importer rejected the schema")

    class ReaderProxy:
        @staticmethod
        def _import_from_c(address):
            seen.append("stream")
            raise AssertionError("A rejected schema must not reach the stream importer")

    calls = {}
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pa, "Schema", SchemaProxy)
        patch.setattr(pa, "RecordBatchReader", ReaderProxy)
        with _dataframe([batch], calls=calls) as (frame, _):
            with pytest.raises(RuntimeError, match="test schema importer rejected"):
                frame.toArrow()
    assert seen == ["schema"]
    assert calls == {"execute": 1}


@pytest.mark.parametrize("check", ["_check_stream_buffers", "_check_schema_preflight"])
def test_isolated_handoff_boundaries(check):
    # arrow-pyarrow uses process-global PyOnceLock caches. A subprocess keeps
    # test proxies out of other tests, regardless of collection/execution order.
    code = (
        "import runpy, sys\n"
        f"sys.path[:] = {sys.path!r}\n"
        f"namespace = runpy.run_path({__file__!r})\n"
        f"namespace[{check!r}]()\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, (
        f"{check} exited with {result.returncode}\n{result.stdout}\n{result.stderr}"
    )


def test_handoff_preserves_values_metadata_and_chunks():
    batch = _mixed_batch()
    batches = [batch.slice(1, 2), batch.slice(0, 1), batch.slice(3, 1)]
    expected = pa.Table.from_batches(batches)
    calls = {}
    with _dataframe(batches, calls=calls) as (frame, _):
        actual = frame.toArrow()
    assert actual.equals(expected, check_metadata=True)
    for column in actual.columns:
        assert [len(chunk) for chunk in column.chunks] == [2, 1, 1]
    assert calls == {"execute": 1}


def test_handoff_buffers_outlive_dataframe_session_and_table():
    batch = _mixed_batch()
    expected = batch.to_pydict()
    calls = {}

    def collect():
        with _dataframe([batch], calls=calls) as (frame, session):
            table = frame.toArrow()
            session.stop()
            return table

    table = collect()
    gc.collect()
    assert table.to_pydict() == expected
    numeric = table.column(0).chunk(0).slice(1, 3)
    strings = table.column(2).chunk(0).slice(1, 3)
    addresses = [buffer.address for buffer in (numeric.buffers()[1], strings.buffers()[2])]
    del table
    gc.collect()
    assert numeric.to_pylist() == [None, -3, 4]
    assert strings.to_pylist() == [None, "", "omega"]
    assert numeric.buffers()[1].address == addresses[0]
    assert strings.buffers()[2].address == addresses[1]
    assert calls == {"execute": 1}


@pytest.mark.parametrize("kind", ["no-batches", "empty-typed", "zero-columns"])
def test_handoff_empty_and_zero_column_results(kind):
    if kind == "no-batches":
        batches = []
        expected = pa.table({})
    elif kind == "empty-typed":
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
    with _dataframe(batches, calls=calls) as (frame, _):
        actual = frame.toArrow()
    assert actual.num_rows == expected.num_rows
    assert actual.num_columns == expected.num_columns
    assert actual.equals(expected, check_metadata=True)
    assert calls == {"execute": 1}


def test_to_pandas_uses_the_same_handoff():
    batch = _mixed_batch()
    indices = [0, 2, 4, 5]
    schema = pa.schema(
        [batch.schema.field(index) for index in indices], metadata=batch.schema.metadata
    )
    arrays = [batch.column(index) for index in indices]
    selected = pa.RecordBatch.from_arrays(arrays, schema=schema)
    batches = [selected.slice(0, 2), selected.slice(2, 2)]
    expected = pa.Table.from_batches(batches).to_pandas()
    calls = {}
    with _dataframe(batches, calls=calls) as (frame, _):
        actual = frame.toPandas()
    pd.testing.assert_frame_equal(actual, expected)
    assert calls == {"execute": 1}


@pytest.mark.parametrize("kind", ["field", "nested-field", "timezone"])
def test_nul_schema_strings_fall_back_without_executing_again(kind):
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
    with _dataframe(batches, calls=calls) as (frame, _):
        actual = frame.toArrow()
    # Compare raw Arrow data, not Python datetimes: the deliberately unusual
    # timezone string is preserved but is not a Python timezone identifier.
    assert actual.equals(expected, check_metadata=True)
    assert [len(chunk) for chunk in actual.column(0).chunks] == [2, 2]
    assert calls == {"execute": 1}


def test_nul_schema_and_field_metadata_are_preserved():
    field = pa.field("value", pa.int64(), metadata={b"field\x00key": b"field\x00value"})
    schema = pa.schema([field], metadata={b"schema\x00key": b"schema\x00value"})
    batch = pa.RecordBatch.from_arrays([pa.array([1, None, 3])], schema=schema)
    calls = {}
    with _dataframe([batch], calls=calls) as (frame, _):
        actual = frame.toArrow()
    assert actual.equals(pa.Table.from_batches([batch]), check_metadata=True)
    assert calls == {"execute": 1}


@pytest.mark.parametrize("difference", ["type", "name", "schema-metadata", "field-metadata"])
@pytest.mark.parametrize("nul_name", [False, True])
def test_mismatched_batch_schemas_are_rejected_before_fallback(difference, nul_name):
    name = "bad\x00name" if nul_name else "value"
    first_field = pa.field(name, pa.int64(), metadata={b"field": b"original"})
    first_schema = pa.schema([first_field], metadata={b"schema": b"original"})
    second_values = pa.array(["text", "other"]) if difference == "type" else pa.array([3, 4])
    second_name = name + "-different" if difference == "name" else name
    field_metadata = {b"field": b"different"} if difference == "field-metadata" else None
    schema_metadata = {b"schema": b"different"} if difference == "schema-metadata" else None
    second_field = pa.field(
        second_name, second_values.type, metadata=field_metadata or first_field.metadata
    )
    second_schema = pa.schema(
        [second_field], metadata=schema_metadata or first_schema.metadata
    )
    batches = [
        pa.RecordBatch.from_arrays([pa.array([1, 2])], schema=first_schema),
        pa.RecordBatch.from_arrays([second_values], schema=second_schema),
    ]
    calls = {}
    with _dataframe(batches, calls=calls) as (frame, _):
        with pytest.raises(ValueError, match="inconsistent schemas"):
            frame.toArrow()
    assert calls == {"execute": 1}


def test_malformed_ipc_is_reported():
    calls = {}
    with _dataframe([], payloads=[(1, b"not arrow ipc")], calls=calls) as (frame, _):
        with pytest.raises(RuntimeError, match="Arrow"):
            frame.toArrow()
    assert calls == {"execute": 1}
