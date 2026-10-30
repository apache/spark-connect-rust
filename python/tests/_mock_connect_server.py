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
"""Small, separately hosted Spark Connect service for transport-level tests."""

import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Event


# The drop-in package does not ship generated service protobuf classes, so these
# encoders provide just the messages needed by this test server.
def varint(value):
    result = bytearray()
    while value > 127:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def message_field(field, value):
    if isinstance(value, str):
        value = value.encode()
    return varint(field * 8 + 2) + varint(len(value)) + value


# Field numbers from crates/spark-connect-proto/proto/spark/connect/:
_ANALYZE_SCHEMA = 4  # base.proto: AnalyzePlanRequest.schema
_SCHEMA_PLAN = 1  # base.proto: AnalyzePlanRequest.Schema.plan
_PLAN_ROOT = 1  # base.proto: Plan.root
_RELATION_LOCAL_RELATION = 11  # relations.proto: Relation.local_relation
_LOCAL_RELATION_DATA = 1  # relations.proto: LocalRelation.data


def _message_fields(data):
    """Read singular protobuf fields without generated service classes."""
    def read_varint(position):
        value = 0
        for shift in range(0, 64, 7):
            if position >= len(data):
                raise ValueError("truncated protobuf varint")
            byte = data[position]
            position += 1
            if shift == 63 and byte > 1:
                raise ValueError("protobuf varint exceeds 64 bits")
            value |= (byte & 127) << shift
            if byte < 128:
                return value, position
        raise ValueError("protobuf varint exceeds 64 bits")

    fields = {}
    position = 0
    while position < len(data):
        tag, position = read_varint(position)
        field, wire_type = tag >> 3, tag & 7
        if not field:
            raise ValueError("protobuf field number must be positive")
        if wire_type == 0:
            value, position = read_varint(position)
        else:
            if wire_type == 1:
                size = 8
            elif wire_type == 2:
                size, position = read_varint(position)
            elif wire_type == 5:
                size = 4
            else:
                raise ValueError(f"unsupported protobuf wire type {wire_type}")
            end = position + size
            if end > len(data):
                raise ValueError("truncated protobuf field")
            value, position = data[position:end], end
        fields[field] = value
    return fields


def _local_relation_data(request):
    for field in (
        _ANALYZE_SCHEMA, _SCHEMA_PLAN, _PLAN_ROOT,
        _RELATION_LOCAL_RELATION, _LOCAL_RELATION_DATA,
    ):
        request = _message_fields(request).get(field)
        if request is None:
            return None
    return request


def struct_schema(name="id"):
    # types.proto: DataType.long=7, .struct=21; Struct.fields=1;
    # StructField: name=1, data_type=2, nullable=3, metadata=4.
    field = (
        message_field(1, name)
        + message_field(2, message_field(7, b""))
        + b"\x18\x01"
        + message_field(4, '{"note":"original"}')
    )
    return message_field(21, message_field(1, field))


def serve(connection):
    import grpc

    # A separate interpreter lets tests detect clients retaining the GIL while
    # they wait for a response.
    class Service:
        calls = 0
        local_relation = None
        data_type = struct_schema()
        failure = None
        block_response = False
        response_gate = Event()

        def analyze(self, request, context):
            self.calls += 1
            payload = _local_relation_data(request)
            if payload is not None:
                import pyarrow as pa

                table = pa.ipc.open_stream(payload).read_all()
                self.local_relation = (table.to_pylist(), str(table.schema))
            if self.block_response and not self.response_gate.wait(5):
                context.abort(
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    "Python thread could not run while metadata lookup waited",
                )
            if self.failure == "rpc":
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "schema unavailable")
            if self.failure == "no-result":
                return b""
            if self.failure == "missing-schema":
                return message_field(2, b"")
            if self.failure == "invalid-type":
                return message_field(2, message_field(1, b""))
            # base.proto: AnalyzePlanResponse.schema=2, Schema.schema=1.
            return message_field(2, message_field(1, self.data_type))

    service = Service()
    with ThreadPoolExecutor(max_workers=4) as executor:
        server = grpc.server(executor)
        handlers = {
            "AnalyzePlan": grpc.unary_unary_rpc_method_handler(service.analyze),
            "ReleaseSession": grpc.unary_unary_rpc_method_handler(lambda request, context: b""),
        }
        server.add_generic_rpc_handlers(
            (grpc.method_handlers_generic_handler("spark.connect.SparkConnectService", handlers),)
        )
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
                elif command[0] == "release":
                    service.response_gate.set()
                    connection.send(None)
        finally:
            server.stop(0).wait()


class RemoteService:
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

    def release_response(self):
        self.connection.send(("release",))
        assert self.connection.poll(10), "Mock server did not respond"
        self.connection.recv()


@contextmanager
def mock_connect_session():
    from pyspark.sql import SparkSession

    context = multiprocessing.get_context("spawn")
    parent, child = context.Pipe()
    process = context.Process(target=serve, args=(child,))
    process.start()
    spark = None
    try:
        assert parent.poll(20), "Mock server did not start"
        port = parent.recv()
        spark = SparkSession.builder.remote(f"sc://127.0.0.1:{port}").getOrCreate()
        yield spark, RemoteService(parent)
    finally:
        try:
            if spark is not None:
                spark.stop()
        finally:
            if process.is_alive():
                parent.send(("stop",))
            process.join(10)
            if process.is_alive():
                process.terminate()
                process.join(10)
            parent.close()
            child.close()
