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

"""Schema caching through the real client transport, without a Spark server.

The fixture serves only AnalyzePlan. Tiny wire encoders keep the tests independent
of the reference PySpark client and its generated protobuf modules.
"""
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

import pytest

grpc = pytest.importorskip("grpc")

from pyspark.sql import SparkSession, types as T


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


@pytest.mark.parametrize("first", ["schema", "columns", "dtypes"])
def test_one_analysis_shared_by_metadata_accessors(analyzed, first):
    spark, service = analyzed
    df = spark.range(10)
    assert service.calls == 0
    getattr(df, first)
    assert service.calls == 1
    for _ in range(3):
        schema = df.schema
        assert isinstance(schema, T.StructType)
        expected = T.StructType([T.StructField("id", T.LongType(), True, {"note": "original"})])
        assert schema.jsonValue() == expected.jsonValue()
        assert df.columns == ["id"]
        assert df.dtypes == [("id", "bigint")]
    assert service.calls == 1


def test_returned_schema_and_lists_are_independent(analyzed):
    spark, service = analyzed
    service.data_type = _schema(nested=True)
    df = spark.range(10)
    original = df.schema
    modified = df.schema
    assert original is not modified
    modified.fields[0].metadata["note"] = "changed"
    nested = modified.fields[0].dataType
    nested.add("extra_nested", T.StringType())
    assert len(nested.fields) == 2
    modified.add("extra", T.StringType())
    assert len(modified.fields) == 2
    assert len(original.fields) == 1
    df.columns.append("extra")
    df.dtypes.clear()
    assert df.schema.jsonValue() == original.jsonValue()
    assert df.columns == ["id"]
    assert df.dtypes == [("id", "struct<inner:bigint>")]
    assert service.calls == 1


def test_derived_and_new_plans_resolve_independently(analyzed):
    spark, service = analyzed
    df = spark.range(10)
    assert df.columns == ["id"]
    derived = df.selectExpr("id AS renamed")
    service.data_type = _schema("renamed")
    assert derived.columns == ["renamed"]
    assert service.calls == 2
    assert df.columns == ["id"]
    # Rebuilding the same source must observe updated server metadata, while the
    # original immutable DataFrame retains its already-resolved schema.
    fresh = spark.range(10)
    assert fresh.columns == ["renamed"]
    other_session = spark.newSession()
    assert other_session.range(10).columns == ["renamed"]
    assert service.calls == 4
    assert df.columns == ["id"]


@pytest.mark.parametrize("failure", ["rpc", "no-result", "missing-schema", "invalid-type"])
def test_failures_are_not_cached(analyzed, failure):
    spark, service = analyzed
    df = spark.range(10)
    service.failure = failure
    for count in (1, 2):
        message = {
            "rpc": "schema unavailable",
            "no-result": "no schema in response",
            "missing-schema": "Schema is missing",
            "invalid-type": "DataType kind not set",
        }[failure]
        with pytest.raises(Exception, match=message):
            _ = df.schema
        assert service.calls == count
    service.failure = None
    assert df.columns == ["id"]
    assert df.dtypes == [("id", "bigint")]
    assert service.calls == 3


def test_concurrent_schema_access_issues_one_analysis(analyzed):
    spark, service = analyzed
    df = spark.range(10)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: df.schema, range(8)))
    assert all(schema.jsonValue() == results[0].jsonValue() for schema in results)
    assert len({id(schema) for schema in results}) == 8
    assert service.calls == 1
