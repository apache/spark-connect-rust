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


def varint_field(field, value):
    return varint(field * 8) + varint(value & (2**64 - 1))


def parse_message(data):
    """Decode protobuf wire bytes into {field: [value, ...]}: varints as ints and
    length-delimited fields as bytes (decode nested messages with another call)."""
    fields, i = {}, 0

    def read_varint():
        nonlocal i
        shift = result = 0
        while True:
            byte = data[i]
            i += 1
            result |= (byte & 127) << shift
            shift += 7
            if byte < 128:
                return result

    while i < len(data):
        key = read_varint()
        field, wire = key >> 3, key & 7
        if wire == 0:
            value = read_varint()
        elif wire == 2:
            size = read_varint()
            value, i = data[i : i + size], i + size
        elif wire == 1:
            value, i = data[i : i + 8], i + 8
        elif wire == 5:
            value, i = data[i : i + 4], i + 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        fields.setdefault(field, []).append(value)
    return fields


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
        data_type = struct_schema()
        failure = None
        block_response = False
        response_gate = Event()

        def analyze(self, request, context):
            self.calls += 1
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

        # ExecutePlan records each request and streams one response carrying
        # `execute_payload` (raw ExecutePlanResponse fields), then ResultComplete; with
        # `execute_failure` set it aborts instead. Responses echo the request's
        # operation id, as the server does.
        execute_requests = []
        execute_payload = b""
        execute_failure = None

        def execute(self, request, context):
            self.execute_requests.append(request)
            if self.execute_failure:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, self.execute_failure)
            # ExecutePlanRequest.operation_id=6; ExecutePlanResponse.operation_id=12,
            # response_id=13, result_complete=14.
            operation_id = parse_message(request).get(6, [b""])[0]
            header = message_field(12, operation_id)
            yield header + message_field(13, "r1") + self.execute_payload
            yield header + message_field(13, "r2") + message_field(14, b"")

    service = Service()
    with ThreadPoolExecutor(max_workers=4) as executor:
        server = grpc.server(executor)
        handlers = {
            "AnalyzePlan": grpc.unary_unary_rpc_method_handler(service.analyze),
            "ReleaseSession": grpc.unary_unary_rpc_method_handler(lambda request, context: b""),
            "ExecutePlan": grpc.unary_stream_rpc_method_handler(service.execute),
            "ReattachExecute": grpc.unary_stream_rpc_method_handler(
                lambda request, context: iter([message_field(14, b"")])
            ),
            "ReleaseExecute": grpc.unary_unary_rpc_method_handler(lambda request, context: b""),
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
