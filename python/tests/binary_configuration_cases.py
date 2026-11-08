# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Binary result configuration through native actions and the mock transport."""

from concurrent.futures import ThreadPoolExecutor
import subprocess
import sys
import time

import pytest

from _mock_connect_server import data_type_proto, message_field, parse_message, varint_field
from pyspark.errors import PySparkException
from pyspark.sql import types as T


_BINARY_CONF = "spark.sql.execution.pyspark.binaryAsBytes"
_EAGER_ACTIONS = ["collect", "first", "head", "head_n", "take", "tail"]
_ACTIONS = _EAGER_ACTIONS + ["iterator", "prefetch"]


_ITERATOR_RELEASE_CHILD = r"""
import sys
import pytest
from pyspark.sql import SparkSession, types as T

spark = SparkSession.builder.remote(sys.argv[1]).getOrCreate()
reason = sys.argv[3]
original = T._udt_field_converters

def fail(schema):
    raise LookupError("converter initialization failed")

if reason == "converter_failure":
    T._udt_field_converters = fail
iterator = spark.range(2).toLocalIterator(sys.argv[2] == "True")
try:
    if reason == "converter_failure":
        with pytest.raises(LookupError, match="converter initialization failed"):
            next(iterator)
        T._udt_field_converters = original
    else:
        assert next(iterator).b == b"x"
        iterator.close()
    with pytest.raises(StopIteration):
        next(iterator)
    # Keep the process and its runtime alive until the parent verifies ReleaseAll.
    assert sys.stdin.readline() == "finish\n"
finally:
    T._udt_field_converters = original
    iterator.close()
    spark.stop()
"""


def _results(service, batch, schema=None):
    import pyarrow as pa

    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    service.execute_payload = message_field(
        2, varint_field(1, batch.num_rows) + message_field(2, sink.getvalue().to_pybytes())
    )
    if schema is not None:
        service.execute_payload += message_field(7, data_type_proto(schema.jsonValue()))


def _action(frame, method):
    if method in ("first", "head"):
        row = getattr(frame, method)()
        return [] if row is None else [row]
    if method in ("head_n", "take", "tail"):
        return getattr(frame, "head" if method == "head_n" else method)(1)
    if method == "collect":
        return frame.collect()
    return list(frame.toLocalIterator(method == "prefetch"))


def _wait_for(predicate, timeout=5):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("mock RPC did not arrive")
        time.sleep(0.01)


class TestBinaryConfiguration:
    @pytest.mark.parametrize("method", _ACTIONS)
    @pytest.mark.parametrize("include_schema", [False, True])
    @pytest.mark.parametrize("values", [[], [None], [b"value"]])
    def test_result_setting_applies_without_a_spark_schema(
        self, mock_connect, method, include_schema, values
    ):
        pa = pytest.importorskip("pyarrow")
        spark, service = mock_connect
        service.confs = {_BINARY_CONF: "false"}
        batch = pa.RecordBatch.from_arrays([pa.array(values, type=pa.binary())], names=["b"])
        schema = T.StructType([T.StructField("b", T.BinaryType())])
        _results(service, batch, schema if include_schema else None)
        rows = _action(spark.range(1), method)
        assert [row.b for row in rows] == values
        if values and values[0] is not None:
            assert type(rows[0].b) is bytearray
        assert service.read_confs == [_BINARY_CONF]
        assert service.calls == 0, "binary conversion must not analyze the schema"

    @pytest.mark.parametrize("method", ["collect", "iterator", "prefetch"])
    @pytest.mark.parametrize("include_schema", [False, True])
    def test_schema_free_nested_binary_and_null_first_rows(
        self, mock_connect, method, include_schema
    ):
        pa = pytest.importorskip("pyarrow")
        spark, service = mock_connect
        service.confs = {_BINARY_CONF: "false"}
        batch = pa.RecordBatch.from_arrays([
            pa.array([None, b"x", b""], type=pa.binary()),
            pa.array([None, [b"a", None], []], type=pa.list_(pa.binary())),
            pa.array([None, {"b": b"s"}, {"b": None}], type=pa.struct([("b", pa.binary())])),
            pa.array([None, [("k", b"m")], []], type=pa.map_(pa.string(), pa.binary())),
        ], names=["b", "a", "s", "m"])
        schema = T.StructType([
            T.StructField("b", T.BinaryType()),
            T.StructField("a", T.ArrayType(T.BinaryType())),
            T.StructField("s", T.StructType([T.StructField("b", T.BinaryType())])),
            T.StructField("m", T.MapType(T.StringType(), T.BinaryType())),
        ])
        _results(service, batch, schema if include_schema else None)
        rows = _action(spark.range(3), method)
        assert tuple(rows[0]) == (None, None, None, None)
        assert all(type(value) is bytearray for value in (
            rows[1].b, rows[1].a[0], rows[1].s.b, rows[1].m["k"], rows[2].b,
        ))
        assert rows[1].a == [b"a", None] and rows[2].b == b""
        assert rows[2].a == [] and rows[2].s.b is None and rows[2].m == {}
        assert service.read_confs == [_BINARY_CONF] and service.calls == 0

    @pytest.mark.parametrize("setting,expected", [
        ("unset", bytes), ("TrUe", bytes), ("FaLsE", bytearray),
        (" true ", bytearray), (None, bytearray),
    ])
    def test_result_setting_keeps_reference_default_and_boolean_handling(
        self, mock_connect, setting, expected
    ):
        pa = pytest.importorskip("pyarrow")
        spark, service = mock_connect
        if setting != "unset":
            service.confs = {_BINARY_CONF: setting}
        _results(service, pa.RecordBatch.from_arrays([pa.array([b"x"])], names=["b"]))
        assert type(spark.range(1).collect()[0].b) is expected
        operation = parse_message(parse_message(service.config_requests[0])[3][0])
        pair = parse_message(parse_message(operation[3][0])[1][0])
        assert pair == {1: [_BINARY_CONF.encode()], 2: [b"true"]}

    @pytest.mark.parametrize("prefetch", [False, True])
    def test_iterator_snapshots_on_first_next_before_execution(self, mock_connect, prefetch):
        pa = pytest.importorskip("pyarrow")
        spark, service = mock_connect
        _results(service, pa.RecordBatch.from_arrays([pa.array([b"x", b"y"])], names=["b"]))
        service.confs = {_BINARY_CONF: "true"}
        service.block_execute = True
        iterator = spark.range(2).toLocalIterator(prefetch)
        assert service.rpc_events == []
        service.confs = {_BINARY_CONF: "false"}
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(next, iterator)
            try:
                _wait_for(lambda: service.execute_calls == 1)
                assert service.rpc_events == ["config", "execute"]
                service.confs = {_BINARY_CONF: "true"}
            finally:
                service.release_response()
            assert type(pending.result(timeout=10).b) is bytearray
        assert type(next(iterator).b) is bytearray
        with pytest.raises(StopIteration):
            next(iterator)
        assert service.read_confs == [_BINARY_CONF]

    def test_eager_result_reads_setting_after_execution(self, mock_connect):
        pa = pytest.importorskip("pyarrow")
        spark, service = mock_connect
        _results(service, pa.RecordBatch.from_arrays([pa.array([b"x"])], names=["b"]))
        service.confs = {_BINARY_CONF: "false"}
        service.block_execute = True
        frame = spark.range(1)
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(frame.collect)
            try:
                _wait_for(lambda: service.execute_calls == 1)
                assert service.rpc_events == ["execute"]
                service.confs = {_BINARY_CONF: "true"}
            finally:
                service.release_response()
            assert type(pending.result(timeout=10)[0].b) is bytes
        assert service.rpc_events == ["execute", "config"]

    @pytest.mark.parametrize("method", _EAGER_ACTIONS)
    def test_eager_configuration_failure_follows_execution(self, mock_connect, method):
        pa = pytest.importorskip("pyarrow")
        spark, service = mock_connect
        _results(service, pa.RecordBatch.from_arrays([pa.array([b"x"])], names=["b"]))
        service.config_failure = "binary setting unavailable"
        with pytest.raises(PySparkException, match="binary setting unavailable"):
            _action(spark.range(1), method)
        assert service.rpc_events == ["execute", "config"]
        assert service.execute_calls == 1 and len(service.config_requests) == 1

    def test_eager_execution_error_precedes_configuration(self, mock_connect):
        spark, service = mock_connect
        service.execute_failure = "execution failed"
        service.config_failure = "binary setting unavailable"
        with pytest.raises(PySparkException, match="execution failed"):
            spark.range(1).collect()
        assert service.rpc_events == ["execute"] and service.config_requests == []

    @pytest.mark.parametrize("prefetch", [False, True])
    def test_iterator_config_failure_is_terminal_without_execution(self, mock_connect, prefetch):
        spark, service = mock_connect
        service.config_failure = "binary setting unavailable"
        iterator = spark.range(2).toLocalIterator(prefetch)
        assert service.rpc_events == []
        with pytest.raises(PySparkException, match="binary setting unavailable"):
            next(iterator)
        service.config_failure = None
        with pytest.raises(StopIteration):
            next(iterator)
        iterator.close()
        assert service.rpc_events == ["config"]
        assert service.execute_calls == 0

    @pytest.mark.parametrize("prefetch", [False, True])
    @pytest.mark.parametrize("reason", ["explicit", "converter_failure"])
    def test_iterator_close_releases_execution_before_completion(
        self, mock_connect, prefetch, reason
    ):
        pa = pytest.importorskip("pyarrow")
        _, service = mock_connect
        if reason == "converter_failure":
            spatial = pa.struct([("wkb", pa.binary()), ("srid", pa.int32())])
            batch = pa.RecordBatch.from_arrays([
                pa.array([{"wkb": b"x", "srid": 4326}] * 2, type=spatial),
            ], names=["g"])
            schema = T.StructType([T.StructField("g", T.GeometryType(4326))])
        else:
            batch = pa.RecordBatch.from_arrays([pa.array([b"x", b"y"])], names=["b"])
            schema = None
        _results(service, batch, schema)
        service.block_result_complete = True
        # Isolate the native close/failure path: retaining the GIL or waiting forever in
        # close must fail this bounded test without hanging the pytest interpreter.
        child = subprocess.Popen(
            [sys.executable, "-c", _ITERATOR_RELEASE_CHILD, service.remote, str(prefetch), reason],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        try:
            _wait_for(lambda: bool(service.execute_requests) or child.poll() is not None, timeout=20)
            assert service.execute_requests, "Iterator did not start an execution"
            operation_id = parse_message(service.execute_requests[0])[6][0]

            def released():
                # ReleaseExecuteRequest.release_all=5 and operation_id=3.
                return any(
                    5 in fields and fields.get(3) == [operation_id]
                    for fields in map(parse_message, service.release_execute_requests)
                )

            _wait_for(lambda: released() or child.poll() is not None)
            assert released(), "Closing the iterator did not release its execution"
            # ResultComplete cannot have caused this release while r2 is still held.
            assert operation_id not in service.result_complete_operations
            stdout, stderr = child.communicate("finish\n", timeout=10)
            assert child.returncode == 0, stdout + stderr
        finally:
            service.release_response()
            if child.poll() is None:
                child.kill()
            child.communicate(timeout=10)
        assert service.rpc_events == ["config", "execute"]

    @pytest.mark.parametrize("prefetch", [False, True])
    def test_iterator_closed_before_first_next_makes_no_requests(self, mock_connect, prefetch):
        spark, service = mock_connect
        iterator = spark.range(1).toLocalIterator(prefetch)
        iterator.close()
        assert list(iterator) == []
        assert service.rpc_events == []
        assert service.calls == 0
        assert service.release_execute_requests == []

    @pytest.mark.parametrize("iterator", [False, True])
    def test_configuration_wait_releases_gil(self, mock_connect, iterator):
        pa = pytest.importorskip("pyarrow")
        spark, service = mock_connect
        _results(service, pa.RecordBatch.from_arrays([pa.array([b"x"])], names=["b"]))
        service.block_config = True

        def release_configuration():
            _wait_for(lambda: bool(service.config_requests))
            service.release_response()

        frame = spark.range(1)
        with ThreadPoolExecutor(max_workers=1) as pool:
            released = pool.submit(release_configuration)
            rows = _action(frame, "iterator" if iterator else "collect")
            released.result(timeout=10)
        assert type(rows[0].b) is bytes

    @pytest.mark.parametrize("method", ["collect", "iterator", "prefetch"])
    def test_direct_binary_map_keys_keep_reference_unhashable_error(self, mock_connect, method):
        pa = pytest.importorskip("pyarrow")
        spark, service = mock_connect
        service.confs = {_BINARY_CONF: "false"}
        batch = pa.RecordBatch.from_arrays([
            pa.array([[(b"key", b"value")]], type=pa.map_(pa.binary(), pa.binary())),
        ], names=["m"])
        schema = T.StructType([T.StructField("m", T.MapType(T.BinaryType(), T.BinaryType()))])
        _results(service, batch, schema)
        with pytest.raises(TypeError, match="unhashable type: 'bytearray'"):
            _action(spark.range(1), method)[0].m
