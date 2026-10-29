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

"""Session version cache tests using a separate-process localhost gRPC fixture."""
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


def _serve(connection):
    class Service:
        calls = 0
        releases = 0
        failure = None
        version = "4.3.0"

        def analyze(self, request, context):
            self.calls += 1
            if self.failure == "rpc":
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "version unavailable")
            if self.failure == "no-result":
                return b""
            # AnalyzePlanResponse.spark_version=8, SparkVersion.version=1.
            return _message(8, _message(1, self.version))

        def release(self, request, context):
            self.releases += 1
            return b""

    service = Service()
    with ThreadPoolExecutor(max_workers=4) as executor:
        server = grpc.server(executor)
        server.add_generic_rpc_handlers((grpc.method_handlers_generic_handler(
            "spark.connect.SparkConnectService", {
                "AnalyzePlan": grpc.unary_unary_rpc_method_handler(service.analyze),
                "ReleaseSession": grpc.unary_unary_rpc_method_handler(service.release),
            }),))
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
def version_session():
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


@pytest.mark.parametrize("version", ["4.3.0", ""])
def test_success_is_cached(version_session, version):
    spark, service = version_session
    service.version = version
    assert service.calls == 0
    for _ in range(10):
        assert spark.version == version
    assert service.calls == 1
    service.version = "changed"
    assert spark.version == version
    assert service.calls == 1


@pytest.mark.parametrize("failure", ["rpc", "no-result"])
def test_failed_lookup_can_retry(version_session, failure):
    spark, service = version_session
    service.failure = failure
    with pytest.raises(Exception):
        _ = spark.version
    assert service.calls == 1
    service.failure = None
    assert spark.version == "4.3.0"
    assert spark.version == "4.3.0"
    assert service.calls == 2


def test_new_sessions_have_independent_caches(version_session):
    spark, service = version_session
    assert spark.version == "4.3.0"
    service.version = "other"
    for other in [spark.newSession(), spark.cloneSession()]:
        assert other.session_id != spark.session_id
        assert other.version == "other"
        assert other.version == "other"
    assert service.calls == 3
    assert spark.version == "4.3.0"


def test_success_remains_cached_after_stop(version_session):
    spark, service = version_session
    assert spark.version == "4.3.0"
    spark.stop()
    assert spark.is_stopped
    assert service.releases == 1
    service.failure = "rpc"
    assert spark.version == "4.3.0"
    assert service.calls == 1


def test_concurrent_first_lookup(version_session):
    spark, service = version_session
    with ThreadPoolExecutor(max_workers=8) as executor:
        assert list(executor.map(lambda _: spark.version, range(24))) == ["4.3.0"] * 24
    assert service.calls == 1


def test_active_session_handle_shares_cache(version_session):
    spark, service = version_session
    active = SparkSession.getActiveSession()
    assert active.session_id == spark.session_id
    assert active.version == "4.3.0"
    service.version = "changed"
    assert spark.version == "4.3.0"
    assert service.calls == 1


def test_stop_before_first_lookup_keeps_rpc_behavior(version_session):
    spark, service = version_session
    spark.stop()
    assert service.calls == 0
    service.failure = "rpc"
    with pytest.raises(Exception):
        _ = spark.version
    assert service.calls == 1
