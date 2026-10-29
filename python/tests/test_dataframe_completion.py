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

"""DataFrame completion through the real client transport, without a Spark server.

The fixture serves only AnalyzePlan. Tiny wire encoders keep the tests independent
of the reference PySpark client and its generated protobuf modules.
"""
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

import pytest

grpc = pytest.importorskip("grpc")

from pyspark.sql import Column, SparkSession


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


def _schema(names=("id",)):
    # DataType.struct=21, Struct.fields=1, DataType.long=7.
    return _message(21, b"".join(_message(1, _field(name, _message(7, b"")))
                                  for name in names))


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
            "spark.connect.SparkConnectService", {"AnalyzePlan": handler}),))
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


@pytest.mark.parametrize("names", [
    ["z", "a"],
    ["with space", "a.b", "with`tick", "", "123", "class", "雪", "_private"],
    ["id", "id", "select", "columns", "__dir__", "_ipython_key_completions_"],
    [],
])
def test_completion_matches_python_connect(analyzed, names):
    spark, service = analyzed
    service.data_type = _schema(names)
    df = spark.range(10)
    assert df._ipython_key_completions_() == names
    assert service.calls == 1
    expected = sorted(set(dir(type(df))) | set(names))
    assert df.__dir__() == expected
    assert dir(df) == expected
    assert service.calls == 3  # No schema caching in this independent change.
    assert callable(df.select)
    assert callable(df.__dir__)
    assert callable(df._ipython_key_completions_)
    assert df.columns == names
    for name in names:
        assert isinstance(df[name], Column)
    assert df.schema.fieldNames() == names


def test_completion_lists_are_independent(analyzed):
    spark, service = analyzed
    df = spark.range(10)
    df._ipython_key_completions_().append("extra")
    df.__dir__().clear()
    assert df._ipython_key_completions_() == ["id"]
    assert "select" in dir(df)
    service.data_type = _schema(["changed"])
    assert df._ipython_key_completions_() == ["changed"]


@pytest.mark.parametrize("method", ["__dir__", "_ipython_key_completions_"])
@pytest.mark.parametrize("failure", ["rpc", "no-result", "missing-schema", "invalid-type"])
def test_completion_propagates_schema_errors_and_can_retry(analyzed, method, failure):
    spark, service = analyzed
    df = spark.range(10)
    service.failure = failure
    message = {
        "rpc": "schema unavailable",
        "no-result": "no schema in response",
        "missing-schema": "Schema is missing",
        "invalid-type": "DataType kind not set",
    }[failure]
    for count in (1, 2):
        with pytest.raises(Exception, match=message):
            getattr(df, method)()
        assert service.calls == count
    service.failure = None
    assert "id" in getattr(df, method)()
    assert service.calls == 3
