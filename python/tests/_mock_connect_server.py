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
        execute_calls = 0
        data_type = struct_schema()
        failure = None
        block_response = False
        response_gate = Event()
        # With `block_execute` set, ExecutePlan waits for the gate too, as AnalyzePlan does.
        block_execute = False

        # Each AnalyzePlanRequest, recorded.
        analyze_requests = []

        def reset(self):
            # Back to the state of a new server: the tests of a module share one.
            self.calls = 0
            self.execute_calls = 0
            self.data_type = struct_schema()
            self.failure = None
            self.block_response = False
            self.response_gate.clear()
            self.block_execute = False
            self.analyze_requests = []
            self.execute_requests = []
            self.execute_payload = b""
            self.execute_failure = None
            self.execute_error = None
            self.error_details = b""
            self.fetched_error_ids = []
            self.confs = {}
            self.read_confs = []

        def analyze(self, request, context):
            self.calls += 1
            self.analyze_requests.append(request)
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
        # `execute_failure` set it aborts instead, with INVALID_ARGUMENT and that message, or
        # with the status code name and message of a tuple. Responses echo the request's
        # operation id, as the server does.
        execute_requests = []
        execute_payload = b""
        execute_failure = None
        # With `execute_error` set to (message, reason, metadata), ExecutePlan fails as the
        # server does for an error that it raised: INTERNAL, with a google.rpc.Status of an
        # ErrorInfo of `reason` and `metadata` in grpc-status-details-bin.
        execute_error = None
        # FetchErrorDetails returns `error_details` (raw FetchErrorDetailsResponse fields) and
        # records the error id asked for. Config Get returns the values in `confs`, and fails
        # for a key without one, as the server does for an unset config; it records the keys.
        error_details = b""
        fetched_error_ids = []
        confs = {}
        read_confs = []

        def execute(self, request, context):
            self.execute_calls += 1
            self.execute_requests.append(request)
            if self.block_execute and not self.response_gate.wait(5):
                context.abort(
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    "Python thread could not run while the statement ran",
                )
            if self.execute_failure:
                code, message = (
                    self.execute_failure
                    if isinstance(self.execute_failure, tuple)
                    else ("INVALID_ARGUMENT", self.execute_failure)
                )
                context.abort(getattr(grpc.StatusCode, code), message)
            if self.execute_error:
                message, reason, metadata = self.execute_error
                # google.rpc: Status.code=1, message=2, details=3; Any.type_url=1, value=2;
                # ErrorInfo.reason=1, domain=2, metadata=3 (entries of key=1, value=2).
                info = message_field(1, reason) + message_field(2, "org.apache.spark")
                for key, value in metadata.items():
                    info += message_field(3, message_field(1, key) + message_field(2, value))
                status = (
                    varint_field(1, grpc.StatusCode.INTERNAL.value[0])
                    + message_field(2, message)
                    + message_field(
                        3,
                        message_field(1, "type.googleapis.com/google.rpc.ErrorInfo")
                        + message_field(2, info),
                    )
                )
                context.set_trailing_metadata((("grpc-status-details-bin", status),))
                context.abort(grpc.StatusCode.INTERNAL, message)
            # ExecutePlanRequest.operation_id=6; ExecutePlanResponse.operation_id=12,
            # response_id=13, result_complete=14.
            operation_id = parse_message(request).get(6, [b""])[0]
            header = message_field(12, operation_id)
            yield header + message_field(13, "r1") + self.execute_payload
            yield header + message_field(13, "r2") + message_field(14, b"")

        def fetch_error_details(self, request, context):
            # FetchErrorDetailsRequest.error_id=3.
            self.fetched_error_ids.append(parse_message(request)[3][0].decode())
            return self.error_details

        def config(self, request, context):
            # ConfigRequest.session_id=1, operation=3, Operation.get=2, Get.keys=1;
            # ConfigResponse.session_id=1, pairs=2, KeyValue.key=1, value=2.
            request = parse_message(request)
            operation = parse_message(request[3][0])
            if 2 not in operation:
                context.abort(grpc.StatusCode.UNIMPLEMENTED, "only Get is served")
            keys = [key.decode() for key in parse_message(operation[2][0]).get(1, [])]
            self.read_confs.extend(keys)
            if any(key not in self.confs for key in keys):
                context.abort(grpc.StatusCode.INTERNAL, "[SQL_CONF_NOT_FOUND]")
            # The response names the session, as the server's does.
            return message_field(1, request[1][0]) + b"".join(
                message_field(2, message_field(1, key) + message_field(2, self.confs[key]))
                for key in keys
            )

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
            "FetchErrorDetails": grpc.unary_unary_rpc_method_handler(service.fetch_error_details),
            "Config": grpc.unary_unary_rpc_method_handler(service.config),
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
                elif command[0] == "reset":
                    service.reset()
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

    def reset(self):
        self.connection.send(("reset",))
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
