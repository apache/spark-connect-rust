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

import json
import multiprocessing
import re
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


# The JSON names of the atomic types of DDL names, and the DataType kinds (types.proto) of the
# JSON names.
_ATOMIC_JSON = {
    "void": "void", "null": "void", "binary": "binary", "boolean": "boolean",
    "bool": "boolean", "byte": "byte", "tinyint": "byte", "short": "short",
    "smallint": "short", "int": "integer", "integer": "integer", "long": "long",
    "bigint": "long", "float": "float", "real": "float", "double": "double",
    "string": "string", "date": "date", "timestamp": "timestamp", "timestamp_ltz": "timestamp",
    "timestamp_ntz": "timestamp_ntz", "variant": "variant",
}
_PROTO_KINDS = {
    "void": 1, "binary": 2, "boolean": 3, "byte": 4, "short": 5, "integer": 6, "long": 7,
    "float": 8, "double": 9, "string": 11, "date": 14, "timestamp": 15, "timestamp_ntz": 16,
    "variant": 25,
}
_DAY_TIME = ["day", "hour", "minute", "second"]
_YEAR_MONTH = ["year", "month"]
_DDL_FIELD = re.compile(
    r"\s*(?P<name>`[^`]+`|\w+)(?:\s*:\s*|\s+)(?P<type>.+?)(?P<not_null>\s+not\s+null)?"
    r"(?:\s+comment\s+'(?P<comment>[^']*)')?\s*", re.IGNORECASE | re.DOTALL)


def _top_level_split(text):
    parts, depth, start = [], 0, 0
    for i, char in enumerate(text):
        depth += {"<": 1, ">": -1, "(": 1, ")": -1}.get(char, 0)
        if char == "," and depth == 0:
            parts.append(text[start:i])
            start = i + 1
    return parts + [text[start:]]


def ddl_json(text):
    """The JSON value of the type of a DDL string, as Spark's DDL parser parses those of the
    tests: atomic types, decimal(p, s), time(p), intervals, arrays, maps and structs, and
    schemas of `name type` and `name: type` fields, of NOT NULL and COMMENT. None for any
    other DDL."""
    text = text.strip()
    lower = text.lower()
    if lower in _ATOMIC_JSON:
        return _ATOMIC_JSON[lower]
    match = re.fullmatch(
        r"(?:decimal|dec|numeric)\s*(?:\(\s*(\d+)\s*(?:,\s*(\d+)\s*)?\))?", lower)
    if match:
        return f"decimal({int(match[1] or 10)},{int(match[2] or 0)})"
    match = re.fullmatch(r"time\s*(?:\(\s*(\d+)\s*\))?", lower)
    if match:
        return f"time({int(match[1] or 6)})"
    match = re.fullmatch(r"(geometry|geography)\s*\(\s*(\d+)\s*\)", lower)
    if match:
        crs = "OGC:CRS84" if match[2] == "4326" else f"SRID:{match[2]}"
        return f"geometry({crs})" if match[1] == "geometry" else f"geography({crs}, SPHERICAL)"
    match = re.fullmatch(r"interval\s+(\w+)(?:\s+to\s+(\w+))?", lower)
    if match:
        start, end = match[1], match[2] or match[1]
        for units in (_DAY_TIME, _YEAR_MONTH):
            if start in units and end in units and units.index(start) <= units.index(end):
                return f"interval {start}" + ("" if start == end else f" to {end}")
        return None
    if lower.startswith("array<") and lower.endswith(">"):
        element = ddl_json(text[6:-1])
        return element and {"type": "array", "elementType": element, "containsNull": True}
    if lower.startswith("map<") and lower.endswith(">"):
        types = [ddl_json(part) for part in _top_level_split(text[4:-1])]
        if len(types) != 2 or None in types:
            return None
        return {"type": "map", "keyType": types[0], "valueType": types[1],
                "valueContainsNull": True}
    if lower.startswith("struct<") and lower.endswith(">"):
        text = text[7:-1]
        if not text.strip():
            return {"type": "struct", "fields": []}
    fields = []
    for part in _top_level_split(text):
        match = _DDL_FIELD.fullmatch(part)
        data_type = match and ddl_json(match["type"])
        if not data_type:
            return None
        metadata = {} if match["comment"] is None else {"comment": match["comment"]}
        fields.append({"name": match["name"].strip("`"), "type": data_type,
                       "nullable": not match["not_null"], "metadata": metadata})
    return {"type": "struct", "fields": fields}


def data_type_proto(value):
    """The DataType (types.proto) of the JSON value of a type."""
    if isinstance(value, dict):
        # Array: element_type=1, contains_null=2; Map: key_type=1, value_type=2,
        # value_contains_null=3; Struct.fields=1, StructField: name=1, data_type=2,
        # nullable=3, metadata=4.
        if value["type"] == "array":
            return message_field(20, message_field(1, data_type_proto(value["elementType"]))
                                 + varint_field(2, int(value["containsNull"])))
        if value["type"] == "map":
            return message_field(22, message_field(1, data_type_proto(value["keyType"]))
                                 + message_field(2, data_type_proto(value["valueType"]))
                                 + varint_field(3, int(value["valueContainsNull"])))
        fields = b""
        for field in value["fields"]:
            encoded = message_field(1, field["name"]) + message_field(
                2, data_type_proto(field["type"]))
            if field["nullable"]:
                encoded += varint_field(3, 1)
            if field["metadata"]:
                encoded += message_field(4, json.dumps(field["metadata"]))
            fields += message_field(1, encoded)
        return message_field(21, fields)
    if value in _PROTO_KINDS:
        return message_field(_PROTO_KINDS[value], b"")
    # Decimal: scale=1, precision=2; Time: precision=1; DayTimeInterval=19 and
    # YearMonthInterval=18: start_field=1, end_field=2.
    match = re.fullmatch(r"decimal\((\d+),(\d+)\)", value)
    if match:
        return message_field(10, varint_field(1, int(match[2])) + varint_field(2, int(match[1])))
    match = re.fullmatch(r"time\((\d+)\)", value)
    if match:
        return message_field(28, varint_field(1, int(match[1])))
    # Geometry=26 and Geography=27: srid=1.
    match = re.fullmatch(r"(geometry|geography)\((?:OGC:CRS84|SRID:(\d+))(?:, \w+)?\)", value)
    if match:
        return message_field(26 if match[1] == "geometry" else 27,
                             varint_field(1, int(match[2] or 4326)))
    match = re.fullmatch(r"interval (\w+)(?: to (\w+))?", value)
    units = _DAY_TIME if match[1] in _DAY_TIME else _YEAR_MONTH
    start, end = units.index(match[1]), units.index(match[2] or match[1])
    return message_field(19 if units is _DAY_TIME else 18,
                         varint_field(1, start) + varint_field(2, end))


def ddl_parse_response(request):
    """The AnalyzePlanResponse of an AnalyzePlanRequest of a DDLParse, as the server parses the
    DDL strings of the tests, or None for another request: AnalyzePlanRequest.ddl_parse=11 ->
    DDLParse.ddl_string=1; AnalyzePlanResponse.ddl_parse=9 -> DDLParse.parsed=1."""
    message = parse_message(request)
    if 11 not in message:
        return None
    parsed = ddl_data_type(parse_message(message[11][0])[1][0].decode())
    return message_field(9, message_field(1, parsed or b""))


def ddl_data_type(text):
    """The DataType of a DDL string, as the server's DDLParse parses those of the tests (see
    `ddl_json`); None for any other DDL."""
    value = ddl_json(text)
    return None if value is None else data_type_proto(value)


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
        # AnalyzePlan returns `explain_string` for an Explain request, else `data_type`.
        explain_string = "== Physical Plan ==\n*(1) Range (0, 1, step=1, splits=1)\n"
        failure = None
        block_response = False
        response_gate = Event()
        # With `block_execute` set, ExecutePlan waits for the gate too, as AnalyzePlan does.
        block_execute = False
        # Hold r2 after the first Arrow batch so cleanup tests cannot confuse an
        # execution released by close with one released by normal completion.
        block_result_complete = False
        result_complete_operations = []

        # Each AnalyzePlanRequest, recorded.
        analyze_requests = []

        def reset(self):
            # Back to the state of a new server: the tests of a module share one.
            self.calls = 0
            self.execute_calls = 0
            self.data_type = struct_schema()
            self.failure = None
            self.block_response = False
            # An old ExecutePlan generator keeps its own gate across test resets.
            self.response_gate = Event()
            self.block_execute = False
            self.block_result_complete = False
            self.result_complete_operations = []
            self.analyze_requests = []
            self.execute_requests = []
            self.release_execute_requests = []
            self.execute_payload = b""
            self.execute_failure = None
            self.execute_error = None
            self.error_details = b""
            self.fetched_error_ids = []
            # The warnings of each Set and Unset, as ConfigResponse.warnings=3, and the keys
            # of each Unset.
            self.config_warnings = []
            self.unset_confs = []
            self.confs = {}
            self.read_confs = []
            self.config_requests = []
            self.config_failure = None
            self.block_config = False
            self.rpc_events = []
            self.set_confs = []
            self.clone_requests = []
            # The DDL of the JSON of each JsonToDDL.
            self.ddl_string = ""

        def analyze(self, request, context):
            self.calls += 1
            self.analyze_requests.append(request)
            # base.proto: AnalyzePlanRequest.is_streaming=8; AnalyzePlanResponse.is_streaming=6,
            # IsStreaming.is_streaming=1, of `streaming`, which a test sets.
            if 8 in parse_message(request):
                streaming = getattr(self, "streaming", False)
                return message_field(6, varint_field(1, 1) if streaming else b"")
            if self.block_response and not self.response_gate.wait(5):
                context.abort(
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    "Python thread could not run while metadata lookup waited",
                )
            if self.failure == "rpc":
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "schema unavailable")
            # base.proto: AnalyzePlanRequest.ddl_parse=11 -> DDLParse.ddl_string=1;
            # AnalyzePlanResponse.ddl_parse=9 -> DDLParse.parsed=1: the type of the DDL, as the
            # server parses it, or `data_type` for one that `ddl_data_type` does not parse.
            if 11 in parse_message(request):
                ddl = parse_message(parse_message(request)[11][0])[1][0].decode()
                parsed = ddl_data_type(ddl) or self.data_type
                return message_field(9, message_field(1, parsed))
            # base.proto: AnalyzePlanRequest.json_to_ddl=18 -> JsonToDDL.json_string=1;
            # AnalyzePlanResponse.json_to_ddl=16 -> JsonToDDL.ddl_string=1: `ddl_string`, which
            # a test sets.
            if 18 in parse_message(request):
                return message_field(16, message_field(1, self.ddl_string.encode()))
            if self.failure == "no-result":
                return b""
            if self.failure == "missing-schema":
                return message_field(2, b"")
            if self.failure == "invalid-type":
                return message_field(2, message_field(1, b""))
            # base.proto: AnalyzePlanRequest.explain=5; AnalyzePlanResponse.explain=3,
            # Explain.explain_string=1.
            if 5 in parse_message(request):
                return message_field(3, message_field(1, self.explain_string))
            # base.proto: AnalyzePlanResponse.schema=2, Schema.schema=1.
            return message_field(2, message_field(1, self.data_type))

        # ExecutePlan records each request and streams one response carrying
        # `execute_payload` (raw ExecutePlanResponse fields), then ResultComplete; with
        # `execute_failure` set it aborts instead, with INVALID_ARGUMENT and that message, or
        # with the status code name and message of a tuple. Responses echo the request's
        # operation id, as the server does.
        execute_requests = []
        release_execute_requests = []
        execute_payload = b""
        execute_failure = None
        # With `execute_error` set to (message, reason, metadata), ExecutePlan fails as the
        # server does for an error that it raised: INTERNAL, with a google.rpc.Status of an
        # ErrorInfo of `reason` and `metadata` in grpc-status-details-bin.
        execute_error = None
        # FetchErrorDetails returns `error_details` (raw FetchErrorDetailsResponse fields) and
        # records the error id asked for. Config Get returns the values in `confs`, or else
        # those of `default_confs`, the defaults of SQL configs of the server, and fails for a
        # key without one, as the server does for an unset config; it records the keys, as
        # does GetWithDefault, which returns the values in `confs` or else the defaults that
        # it is given. Config Set records the pairs and the silent flag of each request.
        error_details = b""
        fetched_error_ids = []
        confs = {}
        default_confs = {
            "spark.sql.execution.pythonUDF.arrow.enabled": "true",
            "spark.sql.execution.pythonUDTF.arrow.enabled": "true",
            # Those that the Arrow conversion of pandas input reads, as a Spark 4.3.0 server.
            "spark.sql.session.timeZone": "UTC",
            "spark.sql.timestampType": "TIMESTAMP_LTZ",
            "spark.sql.execution.arrow.useLargeVarTypes": "false",
            "spark.sql.execution.pandas.convertToArrowArraySafely": "true",
            "spark.sql.execution.pandas.inferPandasDictAsMap": "false",
            "spark.sql.session.localRelationCacheThreshold": "1048576",
            "spark.sql.session.localRelationSizeLimit": "3221225472",
            "spark.sql.session.localRelationChunkSizeRows": "10000",
            "spark.sql.session.localRelationChunkSizeBytes": "16777216",
            "spark.sql.session.localRelationBatchOfChunksSizeBytes": "1073741824",
        }
        read_confs = []
        config_requests = []
        config_failure = None
        block_config = False
        rpc_events = []
        set_confs = []

        def execute(self, request, context):
            # Capture these before yielding: an earlier call can resume after reset.
            response_gate = self.response_gate
            block_result_complete = self.block_result_complete
            result_complete_operations = self.result_complete_operations
            self.execute_calls += 1
            self.execute_requests.append(request)
            self.rpc_events.append("execute")
            if self.block_execute and not response_gate.wait(5):
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
            if block_result_complete and not response_gate.wait(30):
                context.abort(
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    "ResultComplete was not released by the test",
                )
            result_complete_operations.append(operation_id)
            yield header + message_field(13, "r2") + message_field(14, b"")

        def warnings_fields(self):
            # ConfigResponse.warnings=3, of `config_warnings`.
            return b"".join(message_field(3, warning) for warning in self.config_warnings)

        def release_execute(self, request, context):
            self.release_execute_requests.append(request)
            return b""

        def fetch_error_details(self, request, context):
            # FetchErrorDetailsRequest.error_id=3.
            self.fetched_error_ids.append(parse_message(request)[3][0].decode())
            return self.error_details

        # CloneSession records the requests and answers with the new session id, the one
        # requested or else `cloned_session_id`.
        clone_requests = []
        cloned_session_id = "00000000-0000-4000-8000-000000000002"

        def clone_session(self, request, context):
            self.clone_requests.append(request)
            # CloneSessionRequest.session_id=1, new_session_id=4; CloneSessionResponse
            # .session_id=1, new_session_id=3.
            fields = parse_message(request)
            new_id = fields.get(4, [self.cloned_session_id.encode()])[0]
            return message_field(1, fields[1][0]) + message_field(3, new_id)

        def config(self, request, context):
            # ConfigRequest.session_id=1, operation=3, Operation.get=2, Get.keys=1;
            # ConfigResponse.session_id=1, pairs=2, KeyValue.key=1, value=2.
            self.config_requests.append(request)
            self.rpc_events.append("config")
            if self.block_config and not self.response_gate.wait(5):
                context.abort(
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    "Python thread could not run while configuration lookup waited",
                )
            if self.config_failure:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, self.config_failure)
            request = parse_message(request)
            operation = parse_message(request[3][0])
            if 1 in operation:
                # Operation.set=1: Set.pairs=1 (KeyValue.key=1, value=2), silent=2.
                set_request = parse_message(operation[1][0])
                pairs = []
                for pair in set_request.get(1, []):
                    pair = parse_message(pair)
                    # A KeyValue of no value=2, as PySpark sends None, is recorded as None.
                    value = pair.get(2, [None])[0]
                    value = None if value is None else value.decode()
                    pairs.append((pair[1][0].decode(), value))
                self.set_confs.append((pairs, set_request.get(2, [None])[0]))
                return message_field(1, request[1][0]) + self.warnings_fields()
            if 6 in operation:
                # Operation.unset=6: Unset.keys=1.
                keys = parse_message(operation[6][0]).get(1, [])
                self.unset_confs.append([key.decode() for key in keys])
                return message_field(1, request[1][0]) + self.warnings_fields()
            if 3 in operation:
                # Operation.get_with_default=3: GetWithDefault.pairs=1 (KeyValue.key=1,
                # value=2), the default of each key that `confs` does not have.
                pairs = []
                for pair in parse_message(operation[3][0]).get(1, []):
                    pair = parse_message(pair)
                    key = pair[1][0].decode()
                    self.read_confs.append(key)
                    value = self.confs.get(key, pair.get(2, [b""])[0].decode())
                    encoded = message_field(1, key)
                    if value is not None:
                        encoded += message_field(2, value)
                    pairs.append(message_field(2, encoded))
                return message_field(1, request[1][0]) + b"".join(pairs)
            if 2 not in operation:
                context.abort(
                    grpc.StatusCode.UNIMPLEMENTED,
                    "only Get, GetWithDefault, Set and Unset are served",
                )
            keys = [key.decode() for key in parse_message(operation[2][0]).get(1, [])]
            self.read_confs.extend(keys)
            confs = {**self.default_confs, **self.confs}
            if any(key not in confs for key in keys):
                context.abort(grpc.StatusCode.INTERNAL, "[SQL_CONF_NOT_FOUND]")
            # The response names the session, as the server's does.
            return message_field(1, request[1][0]) + b"".join(
                message_field(2, message_field(1, key) + message_field(2, confs[key]))
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
            "ReleaseExecute": grpc.unary_unary_rpc_method_handler(service.release_execute),
            "FetchErrorDetails": grpc.unary_unary_rpc_method_handler(service.fetch_error_details),
            "Config": grpc.unary_unary_rpc_method_handler(service.config),
            "CloneSession": grpc.unary_unary_rpc_method_handler(service.clone_session),
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
        remote = f"sc://127.0.0.1:{parent.recv()}"
        spark = SparkSession.builder.remote(remote).getOrCreate()
        service = RemoteService(parent)
        # The URL of the server, for more sessions of it.
        object.__setattr__(service, "remote", remote)
        yield spark, service
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
