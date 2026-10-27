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
"""Exercise the actual Rust drop-in's Arrow handoff, without a Spark server.

The localhost service returns deterministic Arrow IPC results. Buffer addresses
are captured after Rust decodes IPC, at its C Data export boundary: addresses
in the server's Python arrays would not prove anything about this handoff.
"""

import ctypes
import gc
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


_Release = ctypes.CFUNCTYPE(None, ctypes.POINTER(_ArrowArray))

# ctypes owns the executable callback trampolines, but Arrow only retains raw
# function pointers. Keep these test callbacks alive even if a failed assertion
# leaves an imported array in a pytest traceback. No Arrow buffers are retained.
_RELEASE_CALLBACK_KEEPALIVE = []


@pytest.fixture(autouse=True)
def _forbid_c_stream_import(monkeypatch):
    class NoStreamReader:
        @staticmethod
        def _import_from_c(*args):
            raise AssertionError("The C Data handoff must not import a C Stream")

        _import_from_c_capsule = _import_from_c

    monkeypatch.setattr(pa, "RecordBatchReader", NoStreamReader)


def _snapshot(array):
    """Copy only addresses/shape while the producer's exported array is live."""
    return {
        "length": array.length,
        "offset": array.offset,
        "buffers": [array.buffers[i] for i in range(array.n_buffers)],
        "children": [_snapshot(array.children[i].contents) for i in range(array.n_children)],
        "dictionary": _snapshot(array.dictionary.contents) if array.dictionary else None,
    }


class _ArraySpy:
    def __init__(self, monkeypatch, *, reject_at=None, consume_before_reject=False):
        self.batches = []
        self.schemas = []
        self.release_counts = []
        self.import_count = 0
        self.callbacks = []
        real_batch = pa.RecordBatch
        spy = self

        class ProxyMeta(type):
            def __getattr__(cls, name):
                return getattr(real_batch, name)

            def __instancecheck__(cls, instance):
                return isinstance(instance, real_batch)

        class RecordBatchProxy(metaclass=ProxyMeta):
            @staticmethod
            def _import_from_c(address, schema):
                array = ctypes.cast(address, ctypes.POINTER(_ArrowArray))
                batch_index = len(spy.batches)
                spy.batches.append(_snapshot(array.contents))
                spy.schemas.append(schema)
                spy.release_counts.append(0)
                original_release = _Release(array.contents.release)

                @_Release
                def release(current):
                    spy.release_counts[batch_index] += 1
                    original_release(current)

                spy.callbacks.extend((original_release, release))
                _RELEASE_CALLBACK_KEEPALIVE.extend((original_release, release))
                array.contents.release = ctypes.cast(release, ctypes.c_void_p).value
                reject = batch_index + 1 == reject_at
                if reject and not consume_before_reject:
                    raise RuntimeError("test importer rejected the array")
                imported = real_batch._import_from_c(address, schema)
                spy.import_count += 1
                if reject:
                    # PyArrow owns the array now. The exception traceback may
                    # retain this local until gc.collect(), just like user code.
                    raise RuntimeError("test importer rejected the consumed array")
                return imported

        monkeypatch.setattr(pa, "RecordBatch", RecordBatchProxy)


def _assert_buffers(array, exported):
    assert len(array) == exported["length"]
    assert array.offset == exported["offset"]
    own_buffers = array.buffers()[: len(exported["buffers"])]
    for buffer, address in zip(own_buffers, exported["buffers"]):
        if buffer is None:
            assert address is None
        elif buffer.size:
            assert buffer.address == address
    if pa.types.is_dictionary(array.type):
        _assert_buffers(array.dictionary, exported["dictionary"])
    elif pa.types.is_struct(array.type):
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
        pa.DictionaryArray.from_arrays(
            pa.array([0, None, 1, 0], type=pa.int8()), pa.array(["red", "blue"])
        ),
        pa.nulls(4),
    ]
    fields = [
        pa.field(f"c{index}", array.type, metadata={b"field": str(index).encode()})
        for index, array in enumerate(arrays)
    ]
    schema = pa.schema(fields, metadata={b"schema": b"preserved"})
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def test_handoff_preserves_values_metadata_chunks_and_buffer_addresses(monkeypatch):
    batch = _mixed_batch()
    # Sliced fixtures exercise nontrivial IPC input. The measured zero-copy
    # boundary starts after Rust decodes it, not before IPC serialization.
    batches = [batch.slice(1, 2), batch.slice(0, 1), batch.slice(3, 1)]
    expected = pa.Table.from_batches(batches)
    spy = _ArraySpy(monkeypatch)
    with _dataframe(batches) as (frame, _):
        actual = frame.toArrow()
    assert actual.equals(expected, check_metadata=True)
    assert len(spy.batches) == len(batches)
    assert spy.import_count == len(batches)
    assert all(schema is spy.schemas[0] for schema in spy.schemas)
    assert spy.release_counts == [0] * len(batches)
    for batch_index, exported in enumerate(spy.batches):
        assert exported["length"] == batches[batch_index].num_rows
        assert len(exported["children"]) == actual.num_columns
        for column_index, child in enumerate(exported["children"]):
            column = actual.column(column_index)
            assert column.num_chunks == len(batches)
            _assert_buffers(column.chunk(batch_index), child)
    del column, actual
    gc.collect()
    assert spy.release_counts == [1] * len(batches)


def test_handoff_buffers_outlive_dataframe_session_and_table(monkeypatch):
    batch = _mixed_batch()
    expected = batch.to_pydict()
    spy = _ArraySpy(monkeypatch)

    def collect():
        with _dataframe([batch]) as (frame, session):
            table = frame.toArrow()
            session.stop()
            return table

    table = collect()
    gc.collect()
    assert spy.release_counts == [0]
    assert table.to_pydict() == expected
    chunk = table.column(0).chunk(0).slice(1, 3)
    address = chunk.buffers()[1].address
    del table
    gc.collect()
    assert chunk.to_pylist() == [None, -3, 4]
    assert chunk.buffers()[1].address == address
    assert spy.release_counts == [0]
    del chunk
    gc.collect()
    assert spy.release_counts == [1]


@pytest.mark.parametrize("kind", ["no-batches", "empty-typed", "zero-columns"])
def test_handoff_empty_and_zero_column_results(kind):
    if kind == "no-batches":
        batches = []
        expected = pa.table({})
    elif kind == "empty-typed":
        batches = [_mixed_batch().slice(0, 0)]
        expected = pa.Table.from_batches(batches)
    else:
        structs = pa.array([{}, {}, {}], type=pa.struct([]))
        batch = pa.RecordBatch.from_struct_array(structs)
        batches = [batch.replace_schema_metadata({b"zero-column": b"preserved"})]
        expected = pa.Table.from_batches(batches)
    with _dataframe(batches) as (frame, _):
        actual = frame.toArrow()
    assert actual.num_rows == expected.num_rows
    assert actual.num_columns == expected.num_columns
    assert actual.equals(expected, check_metadata=True)


def test_to_pandas_uses_the_same_handoff(monkeypatch):
    batch = _mixed_batch()
    indices = [0, 2, 4, 5, 7]
    schema = pa.schema(
        [batch.schema.field(index) for index in indices], metadata=batch.schema.metadata
    )
    arrays = [batch.column(index) for index in indices]
    batches = [pa.RecordBatch.from_arrays(arrays, schema=schema)]
    expected = pa.Table.from_batches(batches).to_pandas()
    spy = _ArraySpy(monkeypatch)
    with _dataframe(batches) as (frame, _):
        actual = frame.toPandas()
    pd.testing.assert_frame_equal(actual, expected)
    assert len(spy.batches) == 1
    del actual
    gc.collect()
    assert spy.release_counts == [1]


def test_import_rejection_releases_unconsumed_array(monkeypatch):
    batch = _mixed_batch()
    spy = _ArraySpy(monkeypatch, reject_at=1)
    with _dataframe([batch]) as (frame, _):
        with pytest.raises(RuntimeError, match="test importer rejected"):
            frame.toArrow()
    gc.collect()
    assert spy.import_count == 0
    assert spy.release_counts == [1]


@pytest.mark.parametrize("consume_before_reject", [False, True])
def test_second_import_failure_releases_arrays_once(monkeypatch, consume_before_reject):
    batch = _mixed_batch()
    spy = _ArraySpy(monkeypatch, reject_at=2, consume_before_reject=consume_before_reject)
    with _dataframe([batch, batch, batch]) as (frame, _):
        with pytest.raises(RuntimeError, match="test importer rejected"):
            frame.toArrow()
    gc.collect()
    assert len(spy.batches) == 2
    assert spy.import_count == (2 if consume_before_reject else 1)
    # The first batch has been imported and the second may or may not have
    # transferred ownership. Neither can leak or be released twice on failure.
    assert spy.release_counts == [1, 1]


def test_schema_import_rejection_prevents_array_import(monkeypatch):
    batch = _mixed_batch()
    spy = _ArraySpy(monkeypatch)

    class SchemaProxy:
        @staticmethod
        def _import_from_c(address):
            raise RuntimeError("test schema importer rejected the schema")

    monkeypatch.setattr(pa, "Schema", SchemaProxy)
    with _dataframe([batch]) as (frame, _):
        with pytest.raises(RuntimeError, match="test schema importer rejected"):
            frame.toArrow()
    assert not spy.callbacks


@pytest.mark.parametrize("kind", ["field", "nested-field", "timezone", "dictionary-timezone"])
def test_nul_schema_strings_fall_back_without_executing_again(monkeypatch, kind):
    if kind == "field":
        batch = pa.record_batch([pa.array([1, None])], names=["bad\x00name"])
    elif kind == "nested-field":
        data_type = pa.struct([pa.field("bad\x00name", pa.int64())])
        values = pa.array([{"bad\x00name": 1}, None], type=data_type)
        batch = pa.record_batch([values], names=["nested"])
    else:
        values = pa.array([0, 1], type=pa.timestamp("us", tz="UTC\x00suffix"))
        if kind == "dictionary-timezone":
            indices = pa.array([0, None, 1], type=pa.int8())
            values = pa.DictionaryArray.from_arrays(indices, values)
        batch = pa.record_batch([values], names=["timestamp"])
    expected = pa.Table.from_batches([batch])
    spy = _ArraySpy(monkeypatch)
    calls = {}
    with _dataframe([batch], calls=calls) as (frame, _):
        actual = frame.toArrow()
    # Compare raw Arrow data, not Python datetimes: the deliberately unusual
    # timezone string is preserved but is not a Python timezone identifier.
    assert actual.equals(expected, check_metadata=True)
    assert calls == {"execute": 1}
    assert not spy.callbacks


def test_nul_metadata_still_uses_zero_copy_handoff(monkeypatch):
    field = pa.field("value", pa.int64(), metadata={b"field\x00key": b"field\x00value"})
    schema = pa.schema([field], metadata={b"schema\x00key": b"schema\x00value"})
    batch = pa.RecordBatch.from_arrays([pa.array([1, None, 3])], schema=schema)
    spy = _ArraySpy(monkeypatch)
    with _dataframe([batch]) as (frame, _):
        actual = frame.toArrow()
    assert actual.equals(pa.Table.from_batches([batch]), check_metadata=True)
    assert len(spy.batches) == 1
    _assert_buffers(actual.column(0).chunk(0), spy.batches[0]["children"][0])
    del actual
    gc.collect()
    assert spy.release_counts == [1]


def test_mismatched_batch_schemas_are_rejected_before_import(monkeypatch):
    batches = [
        pa.record_batch([pa.array([1])], names=["value"]),
        pa.record_batch([pa.array(["text"])], names=["value"]),
    ]
    spy = _ArraySpy(monkeypatch)
    with _dataframe(batches) as (frame, _):
        with pytest.raises(ValueError, match="inconsistent schemas"):
            frame.toArrow()
    assert not spy.callbacks


def test_malformed_ipc_is_reported_before_import(monkeypatch):
    spy = _ArraySpy(monkeypatch)
    with _dataframe([], payloads=[(1, b"not arrow ipc")]) as (frame, _):
        with pytest.raises(RuntimeError, match="Arrow"):
            frame.toArrow()
    assert not spy.callbacks
