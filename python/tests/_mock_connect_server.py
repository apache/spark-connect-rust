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
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from queue import Empty, Queue
from threading import Condition, Event

# The drop-in package does not ship generated service protobuf classes, so these
# encoders provide just the messages needed by this test server.
# The key that a client reads on its first plan, to compress the plans of the session.
PLAN_COMPRESSION_LOOKUP = b"spark.connect.session.planCompression.threshold"


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


def retry_info_status(code, delay_ms):
    """The grpc-status-details-bin of a failure of `code` with a RetryInfo of `delay_ms`."""
    import grpc

    # google.rpc: Status.code=1, message=2, details=3; Any.type_url=1, value=2;
    # RetryInfo.retry_delay=1; Duration.seconds=1, nanos=2.
    delay = varint_field(1, delay_ms // 1000) + varint_field(2, (delay_ms % 1000) * 1_000_000)
    return (
        varint_field(1, getattr(grpc.StatusCode, code).value[0])
        + message_field(2, "retry later")
        + message_field(
            3,
            message_field(1, "type.googleapis.com/google.rpc.RetryInfo")
            + message_field(2, message_field(1, delay)),
        )
    )


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
        # With `block_add_artifacts` set, AddArtifacts holds its answer until the gate is released.
        block_add_artifacts = False
        add_artifacts_gate = Event()

        # Each AnalyzePlanRequest, recorded.
        analyze_requests = []

        def __init__(self):
            self.listener_condition = Condition()
            self.listener_epoch = 0
            # The sessions whose requests are answered but no longer recorded (see
            # RemoteService.forget_session). A reset keeps them.
            self.forgotten_sessions = set()

        def is_recorded(self, request, kind):
            # Whether the server records the request, of the kind that it counts it by. A request
            # of a forgotten session is answered, and only counted in `forgotten_requests`.
            # Each request message has session_id=1, as base.proto declares.
            session_id = parse_message(request).get(1, [b""])[0].decode()
            with self.listener_condition:
                if session_id not in self.forgotten_sessions:
                    return True
                self.forgotten_requests[kind] += 1
                return False

        def reset(self):
            # Back to the state of a new server: the tests of a module share one.
            self.calls = 0
            self.execute_calls = 0
            self.data_type = struct_schema()
            self.failure = None
            self.block_response = False
            self.block_execute = False
            self.block_result_complete = False
            self.result_complete_operations = []
            self.analyze_requests = []
            self.block_add_artifacts = False
            # The isLocal answer of a test, or None for the answer of the plan, and a failure.
            self.local = None
            self.local_failure = False
            self.execute_requests = []
            # The requests of forgotten sessions that were answered, by kind: "execute", "analyze",
            # "config", "release_session", "release_execute", "reattach", "interrupt", "clone",
            # "fetch_error_details".
            self.forgotten_requests = Counter()
            self.release_execute_requests = []
            self.execute_payload = b""
            self.execute_payloads = None
            self.execute_payload_queue = None
            self.query_payload = None
            self.execute_failure = None
            self.execute_error = None
            self.stream_failure = None
            self.reattach_requests = []
            self.retry_info_failures = []
            # The InterruptRequests of each Interrupt, and the operations that they interrupt.
            self.interrupt_requests = []
            self.interrupted_operations = set()
            self.error_details = b""
            self.fetched_error_ids = []
            # The warnings of each Set and Unset, as ConfigResponse.warnings=3, and the keys
            # of each Unset.
            self.config_warnings = []
            self.unset_confs = []
            self.confs = {}
            self.read_confs = []
            self.config_requests = []
            self.plan_compression_lookups = 0
            self.config_failure = None
            self.block_config = False
            self.plan_compression_lookup_delay = 0
            self.interrupt_failure = None
            self.rpc_events = []
            self.set_confs = []
            self.clone_requests = []
            # The DDL of the JSON of each JsonToDDL.
            self.ddl_string = ""
            with self.listener_condition:
                # Late completion of an old RPC must not count towards the next test.
                self.listener_epoch += 1
                # Release old waiters without opening the next test's response gate.
                self.response_gate.set()
                self.response_gate = Event()
                self.execute_started = 0
                self.execute_gate_timeouts = 0
                self.release_session_unblocks_execute = False
                for queue in getattr(self, "listener_queues", {}).values():
                    queue.put(None)
                self.listener_queues = {}
                self.listener_subscriptions = 0
                self.listener_removals = 0
                self.listener_streams_finished = 0
                self.listener_condition.notify_all()
            for name in (
                "listener_register_gate",
                "listener_remove_gate",
                "release_session_gate",
                "add_artifacts_gate",
            ):
                gate = getattr(self, name, None)
                if gate is not None:
                    gate.set()
            self.listener_bus_enabled = False
            self.listener_register_failure = False
            self.listener_register_blocked = False
            self.listener_register_gate = Event()
            self.listener_remove_failure = False
            self.listener_remove_blocked = False
            self.listener_remove_gate = Event()
            self.release_session_failure = False
            self.release_session_calls = 0
            # The listener events that a release sends, as a server that stops the running
            # queries of the session, and a gate that holds the release's response.
            self.release_session_events = []
            self.release_session_blocked = False
            self.release_session_gate = Event()
            self.add_artifacts_gate = Event()

        def analyze(self, request, context):
            # AnalyzePlanRequest.session_id=1, echoed as AnalyzePlanResponse.session_id=1, as the
            # server does: the client checks it.
            session_id = parse_message(request).get(1, [b""])[0]
            return message_field(1, session_id) + self._analyze(request, context)

        def _analyze(self, request, context):
            if self.is_recorded(request, "analyze"):
                self.calls += 1
                self.analyze_requests.append(request)
            with self.listener_condition:
                response_gate = self.response_gate
                blocked = self.block_response
            # base.proto: AnalyzePlanRequest.is_streaming=8; AnalyzePlanResponse.is_streaming=6,
            # IsStreaming.is_streaming=1, of `streaming`, which a test sets.
            if 8 in parse_message(request):
                streaming = getattr(self, "streaming", False)
                return message_field(6, varint_field(1, 1) if streaming else b"")
            # base.proto: AnalyzePlanRequest.is_local=7 -> IsLocal.plan=1 -> Plan.root=1;
            # AnalyzePlanResponse.is_local=5 -> IsLocal.is_local=1. A local relation is
            # Relation.local_relation=11, cached_local_relation=36 or
            # chunked_cached_local_relation=45.
            # `local`, which a test may set, overrides the answer derived from the plan.
            if 7 in parse_message(request):
                if getattr(self, "local_failure", False):
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, "isLocal unavailable")
                local = getattr(self, "local", None)
                if local is None:
                    is_local = parse_message(parse_message(request)[7][0])
                    plan = parse_message(is_local[1][0])
                    relation = parse_message(plan[1][0])
                    local = any(field in relation for field in (11, 36, 45))
                return message_field(5, varint_field(1, 1) if local else b"")
            if blocked and not response_gate.wait(5):
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
        # With `execute_payloads` set (a list of raw ExecutePlanResponse fields), the execution
        # streams one response of each, r1, r2, ..., instead of `execute_payload`.
        execute_payloads = None
        # With `execute_payload_queue` set (a list of raw ExecutePlanResponse fields), each
        # execution streams the first of the list, which it removes, instead of `execute_payload`,
        # while the list is non-empty: the responses of the requests that a fit makes in turn.
        execute_payload_queue = None
        # With `query_payload` set (raw ExecutePlanResponse fields), the execution of a DataFrame's
        # query, a plan with no command, such as the label query of a LogisticRegression fit,
        # streams it instead of the payloads of a command.
        query_payload = None
        execute_failure = None
        # With `stream_failure` set to a status code name, or a (name, delay in ms) of a failure
        # with that RetryInfo delay, ExecutePlan streams its responses and then fails with it,
        # before ResultComplete, as a dropped stream does. Each ReattachExecute is recorded in
        # `reattach_requests`.
        stream_failure = None
        reattach_requests = []
        # With `execute_error` set to (message, reason, metadata), ExecutePlan fails as the
        # server does for an error that it raised: INTERNAL, with a google.rpc.Status of an
        # ErrorInfo of `reason` and `metadata` in grpc-status-details-bin.
        execute_error = None
        # Each (code name, delay in ms) of this list fails an ExecutePlan, once, as the server
        # does for a failure with a google.rpc.RetryInfo that asks for the delay.
        retry_info_failures = ()
        # FetchErrorDetails returns `error_details` (raw FetchErrorDetailsResponse fields)
        # and records the error id asked for. Config Get returns the values in `confs`, or
        # else those of `default_confs`, the defaults of SQL configs of the server, and
        # fails for a key without one, as the server does for an unset config. A key
        # stored as None is returned without the optional value. Get records the keys, as
        # does GetWithDefault, which returns the values in `confs` or else the defaults
        # that it is given. Config Set records the pairs and the silent flag of each
        # request.
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
        plan_compression_lookups = 0
        config_failure = None
        block_config = False
        # The seconds that the plan compression lookup of the first plan of a session takes.
        plan_compression_lookup_delay = 0
        # The (code, message) of each Interrupt, which the server fails with, or None.
        interrupt_failure = None
        rpc_events = []
        set_confs = []

        def add_artifacts(self, requests, context):
            # AddArtifactsRequest.session_id=1, echoed as AddArtifactsResponse.session_id=2 with
            # no artifact summary. With `block_add_artifacts` set, the answer waits for the gate,
            # as a server that accepts an upload and does not answer it.
            session_id = b""
            for request in requests:
                session_id = parse_message(request).get(1, [b""])[0]
                break
            if self.block_add_artifacts and not self.add_artifacts_gate.wait(15):
                context.abort(
                    grpc.StatusCode.INVALID_ARGUMENT, "artifact upload gate was not released"
                )
            return message_field(2, session_id)

        def release_session(self, request, context):
            with self.listener_condition:
                if self.is_recorded(request, "release_session"):
                    self.release_session_calls += 1
                # Opt in only for the terminal callback-RPC regression. Ordinary tests
                # retain independent response and session-release behavior.
                if self.release_session_unblocks_execute:
                    self.response_gate.set()
                # ReleaseSessionRequest.session_id = 1.
                session_id = parse_message(request).get(1, [b""])[0].decode()
                queue = self.listener_queues.get(session_id)
                events = list(self.release_session_events)
                blocked = self.release_session_blocked
                gate = self.release_session_gate
            if queue is not None:
                for event in events:
                    queue.put(event)
            if blocked and not gate.wait(15):
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "release gate was not released")
            if self.release_session_failure:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "session release failed")
            return b""

        def finish_listener_stream(self, epoch, session_id, queue, recorded):
            with self.listener_condition:
                if epoch != self.listener_epoch:
                    return
                if self.listener_queues.get(session_id) is queue:
                    self.listener_queues.pop(session_id)
                # The stream of a forgotten session is answered, but not counted.
                if recorded:
                    self.listener_streams_finished += 1
                self.listener_condition.notify_all()

        def wait_listener_calls(self, counter, count, timeout):
            with self.listener_condition:
                return self.listener_condition.wait_for(
                    lambda: getattr(self, counter) >= count, timeout=timeout
                )

        def wait_listener_streams_finished(self, count, timeout):
            return self.wait_listener_calls("listener_streams_finished", count, timeout)

        def wait_execute_started(self, count, timeout):
            with self.listener_condition:
                epoch = self.listener_epoch
                reached = self.listener_condition.wait_for(
                    lambda: epoch != self.listener_epoch or self.execute_started >= count,
                    timeout=timeout,
                )
                return reached and epoch == self.listener_epoch and self.execute_started >= count

        def execute(self, request, context):
            fields = parse_message(request)
            with self.listener_condition:
                # Capture these before yielding: an earlier call can resume after reset.
                block_result_complete = self.block_result_complete
                result_complete_operations = self.result_complete_operations
                # Counter waiters on another thread must see both records when ready.
                recorded = self.is_recorded(request, "execute")
                if recorded:
                    self.execute_requests.append(request)
                    self.rpc_events.append("execute")
                    self.execute_calls += 1
                execute_epoch = self.listener_epoch
                response_gate = self.response_gate
                blocked = self.block_execute
            plan = parse_message(fields.get(3, [b""])[0])
            command = parse_message(plan.get(2, [b""])[0])
            listener = parse_message(command.get(11, [b""])[0])
            if self.listener_bus_enabled and listener:
                session_id = fields[1][0].decode()
                header = message_field(12, fields.get(6, [b""])[0])
                if 1 in listener:
                    queue = Queue()
                    with self.listener_condition:
                        epoch = self.listener_epoch
                        # A registration of a forgotten session is answered, but not counted.
                        if recorded:
                            self.listener_subscriptions += 1
                        registration_failed = self.listener_register_failure
                        if not registration_failed:
                            self.listener_queues[session_id] = queue
                        self.listener_condition.notify_all()
                    if registration_failed:
                        context.abort(
                            grpc.StatusCode.INVALID_ARGUMENT, "listener registration failed"
                        )
                    counted = False
                    try:
                        if self.listener_register_blocked:
                            gate = self.listener_register_gate
                            # Force an established-stream/pre-acknowledgement window with
                            # early headers, which real servers seldom expose. An RPC still
                            # being established follows its transport deadline/retry policy.
                            context.send_initial_metadata(())
                            context.add_callback(gate.set)
                            if not gate.wait(15):
                                context.abort(
                                    grpc.StatusCode.INVALID_ARGUMENT,
                                    "listener registration gate was not released",
                                )
                            if not context.is_active():
                                return
                        yield (header + message_field(13, "registered")
                               + message_field(16, varint_field(2, 1)))
                        sequence = 0
                        while context.is_active():
                            try:
                                event = queue.get(timeout=0.1)
                            except Empty:
                                continue
                            if event is None:
                                break
                            event_type, event_json = event
                            if event_type is None:
                                context.abort(grpc.StatusCode.INVALID_ARGUMENT, event_json)
                            sequence += 1
                            payload = message_field(1, event_json) + varint_field(2, event_type)
                            yield (header + message_field(13, f"event-{sequence}")
                                   + message_field(16, message_field(1, payload)))
                        # The client returns at ResultComplete, before server-side EOF/finally.
                        self.finish_listener_stream(epoch, session_id, queue, recorded)
                        counted = True
                        yield header + message_field(13, "finished") + message_field(14, b"")
                    finally:
                        if not counted:
                            self.finish_listener_stream(epoch, session_id, queue, recorded)
                    return
                with self.listener_condition:
                    # A removal of a forgotten session is answered, but not counted.
                    if recorded:
                        self.listener_removals += 1
                    # Keep an old, gated removal from closing a queue created after reset.
                    queue = self.listener_queues.get(session_id)
                    gate = self.listener_remove_gate
                    blocked = self.listener_remove_blocked
                    removal_failed = self.listener_remove_failure
                    self.listener_condition.notify_all()
                if blocked and not gate.wait(15):
                    context.abort(
                        grpc.StatusCode.INVALID_ARGUMENT, "listener removal gate was not released"
                    )
                if removal_failed:
                    context.abort(grpc.StatusCode.INVALID_ARGUMENT, "listener removal failed")
                if queue is not None:
                    queue.put(None)
                yield header + message_field(13, "removed") + message_field(14, b"")
                return
            if blocked:
                with self.listener_condition:
                    if recorded and execute_epoch == self.listener_epoch:
                        self.execute_started += 1
                        self.listener_condition.notify_all()
                # ExecutePlanRequest.operation_id=6.
                held_operation_id = parse_message(request).get(6, [b""])[0]
                released = self.wait_for_response(response_gate, held_operation_id, 5)
                if released == "interrupted":
                    context.abort(grpc.StatusCode.CANCELLED, "Operation has been canceled")
                if not released:
                    with self.listener_condition:
                        if execute_epoch == self.listener_epoch:
                            self.execute_gate_timeouts += 1
                            self.listener_condition.notify_all()
                    context.abort(
                        grpc.StatusCode.DEADLINE_EXCEEDED,
                        "Python thread could not run while the statement ran",
                    )
            if self.retry_info_failures:
                code, delay_ms = self.retry_info_failures.pop(0)
                status = retry_info_status(code, delay_ms)
                context.set_trailing_metadata((("grpc-status-details-bin", status),))
                context.abort(getattr(grpc.StatusCode, code), "retry later")
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
            # ExecutePlanRequest.session_id=1, echoed as ExecutePlanResponse.session_id=1.
            header = message_field(1, parse_message(request).get(1, [b""])[0]) + message_field(
                12, operation_id
            )
            payloads = (
                [self.execute_payload] if self.execute_payloads is None else self.execute_payloads
            )
            if self.execute_payload_queue:
                payloads = [self.execute_payload_queue.pop(0)]
            # ExecutePlanRequest.plan=3, of which Plan.command=2 is set for a command only.
            plan = parse_message(parse_message(request).get(3, [b""])[0])
            if self.query_payload is not None and 2 not in plan:
                payloads = [self.query_payload]
            for index, payload in enumerate(payloads, 1):
                yield header + message_field(13, f"r{index}") + payload
            if self.stream_failure:
                code, delay_ms = (
                    self.stream_failure
                    if isinstance(self.stream_failure, tuple)
                    else (self.stream_failure, None)
                )
                if delay_ms is not None:
                    status = retry_info_status(code, delay_ms)
                    context.set_trailing_metadata((("grpc-status-details-bin", status),))
                context.abort(getattr(grpc.StatusCode, code), "stream dropped")
            if block_result_complete:
                released = self.wait_for_response(response_gate, operation_id, 30)
                if released == "interrupted":
                    context.abort(grpc.StatusCode.CANCELLED, "Operation has been canceled")
                if not released:
                    context.abort(
                        grpc.StatusCode.DEADLINE_EXCEEDED,
                        "ResultComplete was not released by the test",
                    )
            result_complete_operations.append(operation_id)
            yield header + message_field(13, f"r{len(payloads) + 1}") + message_field(14, b"")

        def warnings_fields(self):
            # ConfigResponse.warnings=3, of `config_warnings`.
            return b"".join(message_field(3, warning) for warning in self.config_warnings)

        def reattach_execute(self, request, context):
            if self.is_recorded(request, "reattach"):
                self.reattach_requests.append(request)
            # ExecutePlanResponse.result_complete=14.
            yield message_field(14, b"")

        def release_execute(self, request, context):
            if self.is_recorded(request, "release_execute"):
                self.release_execute_requests.append(request)
            return b""

        def wait_for_response(self, gate, operation_id, timeout):
            # Whether the gate is released, or "interrupted" once an Interrupt of the operation
            # arrives, which ends the execution as the server cancels it.
            deadline = time.monotonic() + timeout
            while not gate.wait(0.01):
                if operation_id in self.interrupted_operations:
                    return "interrupted"
                if time.monotonic() > deadline:
                    return False
            return True

        def interrupt(self, request, context):
            # InterruptRequest.session_id=1, interrupt_type=4, operation_id=6; InterruptResponse
            # .session_id=1, interrupted_ids=2.
            fields = parse_message(request)
            if self.is_recorded(request, "interrupt"):
                self.interrupt_requests.append(request)
            if self.interrupt_failure:
                code, message = self.interrupt_failure
                context.abort(getattr(grpc.StatusCode, code), message)
            if fields.get(4, [0])[0] != 3:
                context.abort(grpc.StatusCode.UNIMPLEMENTED, "only Interrupt of an operation")
            operation_id = fields[6][0]
            self.interrupted_operations.add(operation_id)
            return message_field(1, fields[1][0]) + message_field(2, operation_id)

        def fetch_error_details(self, request, context):
            # FetchErrorDetailsRequest.error_id=3.
            if self.is_recorded(request, "fetch_error_details"):
                self.fetched_error_ids.append(parse_message(request)[3][0].decode())
            return self.error_details

        # CloneSession records the requests and answers with the new session id, the one
        # requested or else `cloned_session_id`.
        clone_requests = []
        cloned_session_id = "00000000-0000-4000-8000-000000000002"

        def clone_session(self, request, context):
            if self.is_recorded(request, "clone"):
                self.clone_requests.append(request)
            # CloneSessionRequest.session_id=1, new_session_id=4; CloneSessionResponse
            # .session_id=1, new_session_id=3.
            fields = parse_message(request)
            new_id = fields.get(4, [self.cloned_session_id.encode()])[0]
            return message_field(1, fields[1][0]) + message_field(3, new_id)

        def config(self, request, context):
            # ConfigRequest.session_id=1, operation=3, Operation.get=2, Get.keys=1;
            # ConfigResponse.session_id=1, pairs=2, KeyValue.key=1, value=2.
            # The plan compression lookup of a client, which PySpark makes on its first plan, is
            # not recorded, blocked or failed, so that the tests of the other requests count,
            # block and fail only their own.
            lookup = PLAN_COMPRESSION_LOOKUP in request
            if lookup:
                self.plan_compression_lookups += 1
            # The requests of a forgotten session are answered, but not recorded below.
            recorded = not lookup and self.is_recorded(request, "config")
            if recorded:
                self.config_requests.append(request)
                self.rpc_events.append("config")
            if lookup and self.plan_compression_lookup_delay:
                # The plan compression lookup that PySpark makes on the first plan, held for the
                # seconds of the delay, as a slow server holds it.
                time.sleep(self.plan_compression_lookup_delay)
            if not lookup and self.block_config and not self.response_gate.wait(5):
                context.abort(
                    grpc.StatusCode.DEADLINE_EXCEEDED,
                    "Python thread could not run while configuration lookup waited",
                )
            if not lookup and self.config_failure:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, self.config_failure)
            request = parse_message(request)
            operation = parse_message(request[3][0])
            if 1 in operation:
                # Operation.set=1: Set.pairs=1 (KeyValue.key=1, value=2), silent=2.
                set_request = parse_message(operation[1][0])
                pairs = []
                for pair in set_request.get(1, []):
                    pair = parse_message(pair)
                    # A KeyValue of no value=2, as PySpark sends None, is recorded as None, and
                    # one of no key=1, the empty key of proto3, as "".
                    value = pair.get(2, [None])[0]
                    value = None if value is None else value.decode()
                    pairs.append((pair.get(1, [b""])[0].decode(), value))
                if recorded:
                    self.set_confs.append((pairs, set_request.get(2, [None])[0]))
                return message_field(1, request[1][0]) + self.warnings_fields()
            if 6 in operation:
                # Operation.unset=6: Unset.keys=1.
                keys = parse_message(operation[6][0]).get(1, [])
                if recorded:
                    self.unset_confs.append([key.decode() for key in keys])
                return message_field(1, request[1][0]) + self.warnings_fields()
            if 3 in operation:
                # Operation.get_with_default=3: GetWithDefault.pairs=1 (KeyValue.key=1,
                # value=2), the default of each key that `confs` does not have. A missing
                # default stays unset in the response.
                pairs = []
                for pair in parse_message(operation[3][0]).get(1, []):
                    pair = parse_message(pair)
                    key = pair[1][0].decode()
                    if recorded:
                        self.read_confs.append(key)
                    default = pair.get(2, [None])[0]
                    default = None if default is None else default.decode()
                    value = self.confs.get(key, default)
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
            if recorded:
                self.read_confs.extend(keys)
            confs = {**self.default_confs, **self.confs}
            if any(key not in confs for key in keys):
                context.abort(grpc.StatusCode.INTERNAL, "[SQL_CONF_NOT_FOUND]")
            # The response names the session, as the server's does.
            return message_field(1, request[1][0]) + b"".join(
                message_field(
                    2,
                    message_field(1, key)
                    + (b"" if confs[key] is None else message_field(2, confs[key])),
                )
                for key in keys
            )

    service = Service()
    service.reset()
    with ThreadPoolExecutor(max_workers=4) as executor:
        server = grpc.server(executor)
        handlers = {
            "AnalyzePlan": grpc.unary_unary_rpc_method_handler(service.analyze),
            "ReleaseSession": grpc.unary_unary_rpc_method_handler(service.release_session),
            "ExecutePlan": grpc.unary_stream_rpc_method_handler(service.execute),
            "ReattachExecute": grpc.unary_stream_rpc_method_handler(service.reattach_execute),
            "ReleaseExecute": grpc.unary_unary_rpc_method_handler(service.release_execute),
            "FetchErrorDetails": grpc.unary_unary_rpc_method_handler(service.fetch_error_details),
            "Config": grpc.unary_unary_rpc_method_handler(service.config),
            "CloneSession": grpc.unary_unary_rpc_method_handler(service.clone_session),
            "Interrupt": grpc.unary_unary_rpc_method_handler(service.interrupt),
            "AddArtifacts": grpc.stream_unary_rpc_method_handler(service.add_artifacts),
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
                elif command[0] == "del":
                    delattr(service, command[1])
                    connection.send(None)
                elif command[0] == "release":
                    service.response_gate.set()
                    connection.send(None)
                elif command[0] == "reset":
                    service.reset()
                    connection.send(None)
                elif command[0] == "forget_session":
                    with service.listener_condition:
                        service.forgotten_sessions.add(command[1])
                    connection.send(None)
                elif command[0] == "listener_event":
                    _, session_id, event_type, event_json = command
                    with service.listener_condition:
                        queue = service.listener_queues.get(session_id)
                    if queue is not None:
                        queue.put((event_type, event_json))
                    connection.send(None)
                elif command[0] == "end_listener_stream":
                    _, session_id, failure = command
                    with service.listener_condition:
                        queue = service.listener_queues.get(session_id)
                    if queue is not None:
                        queue.put((None, "listener event stream failed") if failure else None)
                    connection.send(queue is not None)
                elif command[0] == "wait_listener_removal":
                    connection.send(service.wait_listener_calls("listener_removals", *command[1:]))
                elif command[0] == "wait_listener_registration":
                    connection.send(service.wait_listener_calls(
                        "listener_subscriptions", *command[1:]
                    ))
                elif command[0] == "wait_listener_streams_finished":
                    connection.send(service.wait_listener_streams_finished(*command[1:]))
                elif command[0] == "wait_execute_started":
                    connection.send(service.wait_execute_started(*command[1:]))
                elif command[0] == "release_listener_registration":
                    service.listener_register_gate.set()
                    connection.send(None)
                elif command[0] == "release_listener_removal":
                    service.listener_remove_gate.set()
                    connection.send(None)
                elif command[0] == "release_session_release":
                    service.release_session_gate.set()
                    connection.send(None)
                elif command[0] == "release_add_artifacts":
                    service.add_artifacts_gate.set()
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

    def __delattr__(self, name):
        # pytest 9's monkeypatch undoes an attribute that the instance did not hold with
        # delattr, which removes the override of the service's class attribute.
        self.connection.send(("del", name))
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

    def forget_session(self, session_id):
        # From now on, the server answers the requests of the session of every kind but does not
        # record them, for a test that ends with the session: a request that it sent without
        # waiting for it, such as the Delete of a released model or a ReleaseExecute, can reach
        # the server during a later test, which would count it as one of its own. The server
        # counts them in `forgotten_requests`, by kind.
        self.connection.send(("forget_session", session_id))
        assert self.connection.poll(10), "Mock server did not respond"
        self.connection.recv()

    def listener_event(self, session_id, event_type, event_json):
        self.listener_event_json(session_id, event_type, json.dumps(event_json))

    def listener_event_json(self, session_id, event_type, event_json):
        """Send raw event JSON, including malformed payloads for protocol error tests."""
        self.connection.send(("listener_event", session_id, event_type, event_json))
        assert self.connection.poll(10), "Mock server did not respond"
        self.connection.recv()

    def end_listener_stream(self, session_id, failure=False):
        self.connection.send(("end_listener_stream", session_id, failure))
        assert self.connection.poll(10), "Mock server did not respond"
        return self.connection.recv()

    def wait_listener_removal(self, count=1, timeout=5):
        self.connection.send(("wait_listener_removal", count, timeout))
        assert self.connection.poll(timeout + 5), "Mock server did not respond"
        return self.connection.recv()

    def wait_listener_registration(self, count=1, timeout=5):
        self.connection.send(("wait_listener_registration", count, timeout))
        assert self.connection.poll(timeout + 5), "Mock server did not respond"
        return self.connection.recv()

    def wait_listener_streams_finished(self, count, timeout=5):
        self.connection.send(("wait_listener_streams_finished", count, timeout))
        assert self.connection.poll(timeout + 5), "Mock server did not respond"
        return self.connection.recv()

    def wait_execute_started(self, count=1, timeout=5):
        self.connection.send(("wait_execute_started", count, timeout))
        assert self.connection.poll(timeout + 5), "Mock server did not respond"
        return self.connection.recv()

    def release_listener_registration(self):
        self.connection.send(("release_listener_registration",))
        assert self.connection.poll(10), "Mock server did not respond"
        self.connection.recv()

    def release_listener_removal(self):
        self.connection.send(("release_listener_removal",))
        assert self.connection.poll(10), "Mock server did not respond"
        self.connection.recv()

    def release_session_release(self):
        self.connection.send(("release_session_release",))
        assert self.connection.poll(10), "Mock server did not respond"
        self.connection.recv()

    def release_add_artifacts(self):
        self.connection.send(("release_add_artifacts",))
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
