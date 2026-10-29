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

"""Ordinary DataFrame repr through the real client transport, without Spark.

The fixture serves only AnalyzePlan. Tiny wire encoders keep the tests independent
of the reference PySpark client and its generated protobuf modules.
"""
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

import pytest

grpc = pytest.importorskip("grpc")

from pyspark.sql import SparkSession


def _varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def _message(field, value):
    if isinstance(value, str):
        value = value.encode()
    return _varint(field * 8 + 2) + _varint(len(value)) + value


def _field(name, data_type):
    return (_message(1, name) + _message(2, data_type) + b"\x18\x01"
            + _message(4, '{"note":"original"}'))


def _serve(connection):
    # A separate process can serve schema RPCs while repr holds the client GIL.
    class Service:
        calls = 0
        data_type = _struct(("id", _message(7, b"")))
        failure = None
        execute_calls = 0

        def execute(self, request, context):
            self.execute_calls += 1
            context.abort(grpc.StatusCode.UNIMPLEMENTED, "repr must not execute rows")

        def analyze(self, request, context):
            self.calls += 1
            if self.failure == "rpc":
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "schema unavailable")
            if self.failure == "no-result":
                return b""
            if self.failure == "missing-schema":
                return _message(2, b"")
            if self.failure == "invalid-type":
                return _message(2, _message(1, b""))
            # base.proto: AnalyzePlanResponse.schema=2, Schema.schema=1.
            return _message(2, _message(1, self.data_type))

    service = Service()
    with ThreadPoolExecutor(max_workers=4) as executor:
        server = grpc.server(executor)
        handler = grpc.unary_unary_rpc_method_handler(service.analyze)
        server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(
            "spark.connect.SparkConnectService", {"AnalyzePlan": handler,
                "ExecutePlan": grpc.unary_stream_rpc_method_handler(service.execute)}),))
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


class _RemoteService:
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
def analyzed():
    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=_serve, args=(child,))
    process.start()
    try:
        assert parent.poll(20), "Mock server did not start"
        port = parent.recv()
        spark = SparkSession.builder.remote(f"sc://127.0.0.1:{port}").getOrCreate()
        yield spark, _RemoteService(parent)
    finally:
        if process.is_alive():
            parent.send(("stop",))
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
        child.close()


def _struct(*fields):
    return _message(21, b"".join(_message(1, _field(name, dtype)) for name, dtype in fields))


@pytest.mark.parametrize("schema, expected", [
    (_struct(), "DataFrame[]"),
    (_struct(("id", _message(7, b""))), "DataFrame[id: bigint]"),
    (_struct(("a.b", _message(6, b"")), ("with space", _message(11, b"")),
             ("a.b", _message(9, b"")), ("雪`[]", _message(3, b""))),
     "DataFrame[a.b: int, with space: string, a.b: double, 雪`[]: boolean]"),
    (_struct(("nested", _struct(("inner", _message(7, b"")),
                               ("text", _message(11, b""))))),
     "DataFrame[nested: struct<inner:bigint,text:string>]"),
    (_struct(("items", _message(20, _message(1, _message(6, b"")) + b"\x10\x01")),
             ("lookup", _message(22, _message(1, _message(11, b""))
                                 + _message(2, _message(7, b"")) + b"\x18\x01")),
             ("amount", _message(10, b"\x08\x02\x10\x0a"))),
     "DataFrame[items: array<int>, lookup: map<string,bigint>, amount: decimal(10,2)]"),
    (_struct(("missing", _message(1, b"")), ("raw", _message(2, b"")),
             ("day", _message(14, b"")), ("ts", _message(15, b"")),
             ("local", _message(16, b"")),
             ("label", _message(11, _message(2, "UNICODE")))),
     "DataFrame[missing: void, raw: binary, day: date, ts: timestamp, "
     "local: timestamp_ntz, label: string collate UNICODE]"),
])
def test_repr_formats_schema_without_executing_rows(analyzed, schema, expected):
    spark, service = analyzed
    service.data_type = schema
    df = spark.range(10)
    assert service.calls == 0
    assert repr(df) == expected
    assert service.calls == 1
    assert service.execute_calls == 0


@pytest.mark.parametrize("failure, message", [
    ("rpc", "schema unavailable"),
    ("no-result", "no schema in response"),
    ("missing-schema", "Schema is missing"),
    ("invalid-type", "DataType kind not set"),
])
def test_repr_propagates_schema_errors_and_can_retry(analyzed, failure, message):
    spark, service = analyzed
    df = spark.range(10)
    service.failure = failure
    with pytest.raises(Exception, match=message):
        repr(df)
    assert service.calls == 1
    service.failure = None
    assert repr(df) == "DataFrame[id: bigint]"
    assert service.calls == 2
    assert service.execute_calls == 0


def test_repr_rejects_non_struct_schema(analyzed):
    spark, service = analyzed
    service.data_type = _message(7, b"")
    with pytest.raises(Exception, match="Schema is not a struct type"):
        repr(spark.range(10))
    assert service.execute_calls == 0


def test_repr_handles_wide_schema_and_later_analysis(analyzed):
    spark, service = analyzed
    service.data_type = _struct(*[(f"c{i}", _message(7, b"")) for i in range(256)])
    df = spark.range(10)
    assert repr(df) == "DataFrame[" + ", ".join(f"c{i}: bigint" for i in range(256)) + "]"
    # Ordinary repr delegates to metadata resolution; this change adds no cache.
    service.data_type = _struct(("renamed", _message(11, b"")))
    assert repr(df) == "DataFrame[renamed: string]"
    assert service.calls == 2
    assert service.execute_calls == 0
