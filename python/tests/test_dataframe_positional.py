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

"""Positional indexing through the real client transport, without a Spark server.

The fixture serves only AnalyzePlan. Tiny wire encoders keep the tests independent
of the reference PySpark client and its generated protobuf modules.
"""
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

import pytest

grpc = pytest.importorskip("grpc")

from pyspark.sql import SparkSession, Column


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


def _schema(name="id", nested=False):
    # types.proto: DataType.long=7, .struct=21; Struct.fields=1;
    # StructField: name=1, data_type=2, nullable=3, metadata=4.
    data_type = _message(21, _message(1, _field("inner", _message(7, b"")))) if nested else _message(7, b"")
    return _message(21, _message(1, _field(name, data_type)))


def _field(name, data_type):
    return (_message(1, name) + _message(2, data_type) + b"\x18\x01"
            + _message(4, '{"note":"original"}'))


def _serve(connection):
    # A separate process also exercises columns/dtypes, whose first access may
    # hold the client's GIL while waiting for the RPC.
    class Service:
        calls = 0
        data_type = _schema()
        failure = None

        def analyze(self, request, context):
            self.calls += 1
            # Both clients serialize the UUID session identifier as field 1.
            # Echo it so reference Python can verify response integrity.
            assert request[0] == 10 and request[1] < 128
            session = request[:2 + request[1]]
            if self.failure == "rpc":
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "schema unavailable")
            if self.failure == "no-result":
                return b""
            if self.failure == "missing-schema":
                return _message(2, b"")
            if self.failure == "invalid-type":
                return _message(2, _message(1, b""))
            # base.proto: AnalyzePlanResponse.schema=2, Schema.schema=1.
            return session + _message(2, _message(1, self.data_type))

    service = Service()
    with ThreadPoolExecutor(max_workers=4) as executor:
        server = grpc.server(executor)
        handler = grpc.unary_unary_rpc_method_handler(service.analyze)
        release = grpc.unary_unary_rpc_method_handler(
            lambda request, context: request[:2 + request[1]])
        config = grpc.unary_unary_rpc_method_handler(
            lambda request, context: request[:2 + request[1]] + _message(
                2, _message(1, "spark.python.sql.dataFrameDebugging.enabled")
                + _message(2, "false")))
        server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(
            "spark.connect.SparkConnectService", {
                "AnalyzePlan": handler, "ReleaseSession": release, "Config": config}),))
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
    spark = None
    try:
        assert parent.poll(20), "Mock server did not start"
        port = parent.recv()
        spark = SparkSession.builder.remote(f"sc://127.0.0.1:{port}").create()
        yield spark, _RemoteService(parent)
    finally:
        if spark is not None:
            spark.stop()
        if process.is_alive():
            parent.send(("stop",))
        process.join(10)
        if process.is_alive():
            process.terminate()
            process.join(10)
        parent.close()
        child.close()


def _columns(*names):
    return _message(21, b"".join(_message(1, _field(name, _message(7, b""))) for name in names))


@pytest.mark.parametrize("index, name", [
    (0, "first"), (1, "second"), (2, "last"),
    (-1, "last"), (-2, "second"), (-3, "first"),
    (False, "first"), (True, "second"),
])
def test_positional_column(analyzed, index, name):
    spark, service = analyzed
    service.data_type = _columns("first", "second", "last")
    df = spark.range(10)
    column = df[index]
    assert isinstance(column, Column)
    assert repr(column) == repr(df[name])
    assert service.calls == 1


@pytest.mark.parametrize("names, index", [
    (("a", "b"), 2), (("a", "b"), -3),
    (("a",), 10**100), (("a",), -(10**100)),
    ((), 0), ((), -1), ((), False), (("a",), True),
])
def test_positional_bounds(analyzed, names, index):
    spark, service = analyzed
    service.data_type = _columns(*names)
    with pytest.raises(IndexError):
        spark.range(10)[index]
    assert service.calls == 1


def test_integer_subclass_and_string_lookup(analyzed):
    class Position(int):
        pass
    class Name(str):
        pass
    spark, service = analyzed
    service.data_type = _columns("a.b", "with space", "a.b")
    df = spark.range(10)
    assert repr(df[Position(-2)]) == repr(df[Name("with space")])
    assert repr(df[0]) == repr(df["a.b"])
    before = service.calls
    for name in ["missing", "*", "`a.b`", ""]:
        assert isinstance(df[name], Column)
    assert service.calls == before


def test_invalid_index_types_do_not_analyze(analyzed):
    class Position:
        def __index__(self):
            return 0
    spark, service = analyzed
    df = spark.range(10)
    for value in [0.0, None, b"id", slice(0, 1), Position()]:
        with pytest.raises(TypeError):
            df[value]
    assert service.calls == 0


def test_positional_propagates_schema_failure(analyzed):
    spark, service = analyzed
    service.failure = "rpc"
    df = spark.range(10)
    with pytest.raises(Exception, match="schema unavailable"):
        df[0]
    assert service.calls == 1
    assert isinstance(df["id"], Column)
    assert service.calls == 1
