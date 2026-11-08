#
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
#
"""Check public Arrow handoff behavior against a deterministic mock RPC service.

Cases cover Spark-output values, schemas, chunks, and lifetimes through public
APIs. Server-side Python addresses cannot prove Rust-to-Python buffer sharing.
"""

import gc
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import date, datetime, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import quote

import grpc
import numpy as np
import pandas as pd
import pyarrow as pa
import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from pyspark.sql import SparkSession, types as T
from pyspark.sql._pandas_input import prepare_int64_table


@pytest.fixture(scope="module")
def protocol(tmp_path_factory):
    """Compile the repository protocol instead of hard-coding protobuf field numbers."""
    protoc = shutil.which(os.environ.get("PROTOC", "protoc"))
    if protoc is None:
        message = "Arrow handoff tests require protoc; install it or set PROTOC"
        if os.environ.get("CI"):
            pytest.fail(message)
        pytest.skip(message)
    proto_root = Path(__file__).resolve().parents[2] / "crates/spark-connect-proto/proto"
    descriptor = tmp_path_factory.mktemp("connect-proto") / "connect.pb"
    command = [protoc, f"-I{proto_root}"]
    if include := os.environ.get("PROTOC_INCLUDE"):
        command.append(f"-I{include}")
    subprocess.run(
        command
        + [
            "--include_imports",
            f"--descriptor_set_out={descriptor}",
            str(proto_root / "spark/connect/base.proto"),
        ],
        check=True,
    )
    pool = descriptor_pool.DescriptorPool()
    for file in descriptor_pb2.FileDescriptorSet.FromString(descriptor.read_bytes()).file:
        pool.Add(file)

    def message(name, **fields):
        return message_factory.GetMessageClass(pool.FindMessageTypeByName(f"spark.connect.{name}"))(
            **fields
        )

    return message


@pytest.mark.parametrize("in_ci", [False, True])
def test_missing_protoc_fails_in_ci_and_skips_locally(monkeypatch, tmp_path_factory, in_ci):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    if in_ci:
        monkeypatch.setenv("CI", "true")
    else:
        monkeypatch.delenv("CI", raising=False)
    error = pytest.fail.Exception if in_ci else pytest.skip.Exception
    with pytest.raises(error, match="require protoc"):
        protocol.__wrapped__(tmp_path_factory)


@contextmanager
def _dataframe(batches, *, protocol, payloads=None, calls=None):
    """Serve Arrow results using the repository's protobuf message definitions."""
    if payloads is None:
        payloads = []
        for batch in batches:
            sink = pa.BufferOutputStream()
            with pa.ipc.new_stream(sink, batch.schema) as writer:
                writer.write_batch(batch)
            payloads.append((batch.num_rows, sink.getvalue().to_pybytes()))
    server_id = "00000000-0000-0000-0000-000000000001"

    def execute(request, context):
        if calls is not None:
            calls["execute"] = calls.get("execute", 0) + 1
        request = protocol("ExecutePlanRequest").FromString(request)
        header = dict(
            session_id=request.session_id,
            operation_id=request.operation_id,
            server_side_session_id=server_id,
        )
        for index, (rows, payload) in enumerate(payloads):
            yield protocol(
                "ExecutePlanResponse",
                **header,
                response_id=str(index),
                arrow_batch=dict(row_count=rows, data=payload),
            ).SerializeToString()
        yield protocol(
            "ExecutePlanResponse", **header, response_id="complete", result_complete={}
        ).SerializeToString()

    def release_execute(request, context):
        request = protocol("ReleaseExecuteRequest").FromString(request)
        return protocol(
            "ReleaseExecuteResponse",
            session_id=request.session_id,
            operation_id=request.operation_id,
            server_side_session_id=server_id,
        ).SerializeToString()

    def config(request, context):
        # The settings that toPandas reads first, as SparkConnectClient.to_pandas does.
        if calls is not None:
            calls["config"] = calls.get("config", 0) + 1
        request = protocol("ConfigRequest").FromString(request)
        settings = {
            "spark.sql.session.timeZone": "UTC",
            "spark.sql.execution.pandas.structHandlingMode": "legacy",
            "spark.sql.execution.arrow.pyspark.selfDestruct.enabled": "false",
        }
        return protocol(
            "ConfigResponse",
            session_id=request.session_id,
            server_side_session_id=server_id,
            pairs=[dict(key=key, value=settings[key]) for key in request.operation.get.keys],
        ).SerializeToString()

    def release_session(request, context):
        request = protocol("ReleaseSessionRequest").FromString(request)
        return protocol(
            "ReleaseSessionResponse",
            session_id=request.session_id,
            server_side_session_id=server_id,
        ).SerializeToString()

    with ThreadPoolExecutor(max_workers=2) as executor:
        server = grpc.server(executor)
        server.add_generic_rpc_handlers(
            [
                grpc.method_handlers_generic_handler(
                    "spark.connect.SparkConnectService",
                    {
                        "ExecutePlan": grpc.unary_stream_rpc_method_handler(execute),
                        "ReleaseExecute": grpc.unary_unary_rpc_method_handler(release_execute),
                        "ReleaseSession": grpc.unary_unary_rpc_method_handler(release_session),
                        "Config": grpc.unary_unary_rpc_method_handler(config),
                    },
                )
            ]
        )
        port = server.add_insecure_port("127.0.0.1:0")
        assert port != 0
        server.start()
        session = None
        try:
            session = SparkSession.builder.remote(f"sc://127.0.0.1:{port}").getOrCreate()
            yield session.range(1), session
        finally:
            try:
                if session is not None:
                    session.stop()
            finally:
                server.stop(0).wait(timeout=5)


def _mixed_batch():
    nested_type = pa.struct(
        [
            pa.field("items", pa.list_(pa.int64()), metadata={b"nested": b"metadata"}),
            pa.field("labels", pa.map_(pa.string(), pa.int64())),
        ]
    )
    arrays = [
        pa.array([1, None, -3, 4], type=pa.int64()),
        pa.array([True, None, False, True]),
        pa.array(["alpha", None, "", "omega"]),
        pa.array([b"a", None, b"", b"xyz"]),
        pa.array(
            [Decimal("1.23"), None, Decimal("-4.56"), Decimal("0.00")], type=pa.decimal128(12, 2)
        ),
        pa.array(
            [
                datetime(2020, 1, 1, tzinfo=timezone.utc),
                None,
                datetime(2020, 1, 2, tzinfo=timezone.utc),
                datetime(2020, 1, 3, tzinfo=timezone.utc),
            ],
            type=pa.timestamp("us", tz="UTC"),
        ),
        pa.array(
            [
                {"items": [1, None], "labels": [("a", 1)]},
                None,
                {"items": [], "labels": []},
                {"items": None, "labels": [("b", None)]},
            ],
            type=nested_type,
        ),
        pa.nulls(4),
    ]
    fields = [
        pa.field(f"c{index}", array.type, metadata={b"field": str(index).encode()})
        for index, array in enumerate(arrays)
    ]
    schema = pa.schema(fields, metadata={b"schema": b"preserved"})
    return pa.RecordBatch.from_arrays(arrays, schema=schema)


def test_handoff_preserves_values_metadata_and_chunks(protocol):
    batch = _mixed_batch()
    batches = [batch.slice(1, 2), batch.slice(0, 1), batch.slice(3, 1)]
    expected = pa.Table.from_batches(batches)
    calls = {}
    with _dataframe(protocol=protocol, batches=batches, calls=calls) as (frame, _):
        actual = frame.toArrow()
    assert actual.equals(expected, check_metadata=True)
    for column in actual.columns:
        assert [len(chunk) for chunk in column.chunks] == [2, 1, 1]
    assert calls == {"execute": 1}


def test_normal_handoff_does_not_reopen_ipc(protocol, monkeypatch):
    def unexpected_ipc(*args, **kwargs):
        pytest.fail("Normal toArrow handoff must use the C Stream importer")

    monkeypatch.setattr(pa.ipc, "open_file", unexpected_ipc)
    batch = _mixed_batch()
    with _dataframe(protocol=protocol, batches=[batch]) as (frame, _):
        assert frame.toArrow().equals(pa.Table.from_batches([batch]), check_metadata=True)


def test_handoff_buffers_outlive_dataframe_session_and_table(protocol):
    batch = _mixed_batch()
    expected = batch.to_pydict()
    calls = {}

    def collect():
        with _dataframe(protocol=protocol, batches=[batch], calls=calls) as (frame, session):
            table = frame.toArrow()
            session.stop()
            return table

    table = collect()
    gc.collect()
    assert table.to_pydict() == expected
    numeric = table.column(0).chunk(0).slice(1, 3)
    strings = table.column(2).chunk(0).slice(1, 3)
    del table
    gc.collect()
    assert numeric.to_pylist() == [None, -3, 4]
    assert strings.to_pylist() == [None, "", "omega"]
    assert calls == {"execute": 1}


@pytest.mark.parametrize("kind", ["no-batches", "empty-typed", "zero-columns"])
def test_handoff_empty_and_zero_column_results(protocol, kind):
    if kind == "no-batches":
        batches = []
        expected = pa.table({})
    elif kind == "empty-typed":
        batches = [_mixed_batch().slice(0, 0)]
        expected = pa.Table.from_batches(batches)
    else:
        structs = pa.array([{}] * 5, type=pa.struct([]))
        batch = pa.RecordBatch.from_struct_array(structs)
        batch = batch.replace_schema_metadata({b"zero-column": b"preserved"})
        batches = [batch.slice(0, 3), batch.slice(3, 2)]
        expected = pa.Table.from_batches(batches)
        assert expected.num_rows == 5
    calls = {}
    with _dataframe(protocol=protocol, batches=batches, calls=calls) as (frame, _):
        actual = frame.toArrow()
    assert actual.num_rows == expected.num_rows
    assert actual.num_columns == expected.num_columns
    assert actual.equals(expected, check_metadata=True)
    assert calls == {"execute": 1}


def test_to_pandas_uses_the_same_handoff(protocol):
    batch = _mixed_batch()
    indices = [0, 2, 4, 5]
    schema = pa.schema(
        [batch.schema.field(index) for index in indices], metadata=batch.schema.metadata
    )
    arrays = [batch.column(index) for index in indices]
    selected = pa.RecordBatch.from_arrays(arrays, schema=schema)
    batches = [selected.slice(0, 2), selected.slice(2, 2)]
    # As SparkConnectClient.to_pandas converts the same table: of the Spark types of its Arrow
    # schema (the server sends no schema here), in the session time zone.
    from pyspark.sql.pandas.conversion import _convert_arrow_table_to_pandas
    from pyspark.sql.pandas.types import from_arrow_schema

    table = pa.Table.from_batches(batches)
    expected = _convert_arrow_table_to_pandas(
        arrow_table=table,
        schema=from_arrow_schema(table.schema, prefer_timestamp_ntz=True),
        timezone="UTC",
        struct_handling_mode="legacy",
        date_as_object=False,
        self_destruct=False,
    )
    calls = {}
    with _dataframe(protocol=protocol, batches=batches, calls=calls) as (frame, _):
        actual = frame.toPandas()
    pd.testing.assert_frame_equal(actual, expected)
    assert calls == {"config": 1, "execute": 1}


@pytest.mark.parametrize(
    "kind", ["field", "nested-field", "list-struct", "map-value-struct", "timezone"]
)
def test_nul_schema_strings_fall_back_without_executing_again(protocol, kind, monkeypatch):
    open_file = pa.ipc.open_file
    ipc_calls = []

    def record_ipc(*args, **kwargs):
        ipc_calls.append(True)
        return open_file(*args, **kwargs)

    monkeypatch.setattr(pa.ipc, "open_file", record_ipc)
    if kind == "field":
        name = "bad\x00name"
        values = pa.array([1, None, -3, 4])
    elif kind == "nested-field":
        name = "nested"
        data_type = pa.struct([pa.field("bad\x00name", pa.int64())])
        values = pa.array(
            [{"bad\x00name": 1}, None, {"bad\x00name": None}, {"bad\x00name": 4}],
            type=data_type,
        )
    elif kind == "list-struct":
        name = "items"
        data_type = pa.list_(pa.struct([pa.field("bad\x00name", pa.int64())]))
        values = pa.array([[{"bad\x00name": 1}], None, [], [{"bad\x00name": None}]], type=data_type)
    elif kind == "map-value-struct":
        name = "mapping"
        data_type = pa.map_(pa.string(), pa.struct([pa.field("bad\x00name", pa.int64())]))
        values = pa.array(
            [[("a", {"bad\x00name": 1})], None, [], [("b", {"bad\x00name": None})]], type=data_type
        )
    else:
        name = "timestamp"
        values = pa.array([0, None, 2, 3], type=pa.timestamp("us", tz="UTC\x00suffix"))
    schema = pa.schema(
        [pa.field(name, values.type, metadata={b"field": b"preserved"})],
        metadata={b"schema": b"preserved"},
    )
    batch = pa.RecordBatch.from_arrays([values], schema=schema)
    batches = [batch.slice(0, 2), batch.slice(2, 2)]
    expected = pa.Table.from_batches(batches)
    calls = {}
    with _dataframe(protocol=protocol, batches=batches, calls=calls) as (frame, _):
        actual = frame.toArrow()
    # Compare raw Arrow data, not Python datetimes: the deliberately unusual
    # timezone string is preserved but is not a Python timezone identifier.
    assert actual.equals(expected, check_metadata=True)
    assert [len(chunk) for chunk in actual.column(0).chunks] == [2, 2]
    assert ipc_calls == [True]
    assert calls == {"execute": 1}


def test_nul_schema_and_field_metadata_are_preserved(protocol):
    field = pa.field("value", pa.int64(), metadata={b"field\x00key": b"field\x00value"})
    schema = pa.schema([field], metadata={b"schema\x00key": b"schema\x00value"})
    batch = pa.RecordBatch.from_arrays([pa.array([1, None, 3])], schema=schema)
    calls = {}
    with _dataframe(protocol=protocol, batches=[batch], calls=calls) as (frame, _):
        actual = frame.toArrow()
    assert actual.equals(pa.Table.from_batches([batch]), check_metadata=True)
    assert calls == {"execute": 1}


@pytest.mark.parametrize(
    "difference", ["type", "name", "schema-metadata", "field-metadata", "nullability"]
)
@pytest.mark.parametrize("nul_name", [False, True])
def test_mismatched_batch_schemas_are_rejected_before_fallback(protocol, difference, nul_name):
    name = "bad\x00name" if nul_name else "value"
    first_field = pa.field(name, pa.int64(), metadata={b"field": b"original"})
    first_schema = pa.schema([first_field], metadata={b"schema": b"original"})
    second_values = pa.array(["text", "other"]) if difference == "type" else pa.array([3, 4])
    second_name = name + "-different" if difference == "name" else name
    field_metadata = {b"field": b"different"} if difference == "field-metadata" else None
    schema_metadata = {b"schema": b"different"} if difference == "schema-metadata" else None
    second_field = pa.field(
        second_name,
        second_values.type,
        nullable=difference != "nullability",
        metadata=field_metadata or first_field.metadata,
    )
    second_schema = pa.schema([second_field], metadata=schema_metadata or first_schema.metadata)
    batches = [
        pa.RecordBatch.from_arrays([pa.array([1, 2])], schema=first_schema),
        pa.RecordBatch.from_arrays([second_values], schema=second_schema),
    ]
    calls = {}
    with _dataframe(protocol=protocol, batches=batches, calls=calls) as (frame, _):
        with pytest.raises(ValueError, match="inconsistent schemas"):
            frame.toArrow()
    assert calls == {"execute": 1}


def test_malformed_ipc_is_reported(protocol):
    calls = {}
    with _dataframe(
        protocol=protocol, batches=[], payloads=[(1, b"not arrow ipc")], calls=calls
    ) as (frame, _):
        with pytest.raises(RuntimeError, match="Arrow"):
            frame.toArrow()
    assert calls == {"execute": 1}


@contextmanager
def _arrow_input_service(
    protocol,
    *,
    upload_error=None,
    config_error=False,
    capture_tables=False,
    user_id=None,
    user_agent=None,
    allow_file_artifacts=False,
    **overrides,
):
    """Resolve cache artifacts at a mock service with a 4 MiB receive limit."""
    import hashlib
    import zlib

    config = {
        "spark.sql.timestampType": "TIMESTAMP_LTZ",
        "spark.sql.session.timeZone": "UTC",
        "spark.sql.execution.arrow.useLargeVarTypes": "false",
        "spark.sql.execution.pandas.convertToArrowArraySafely": "true",
        "spark.sql.session.localRelationCacheThreshold": "1048576",
        "spark.sql.session.localRelationSizeLimit": str(3 * 1024**3),
        "spark.sql.session.localRelationChunkSizeRows": "10000",
        "spark.sql.session.localRelationChunkSizeBytes": str(16 * 1024**2),
        "spark.sql.session.localRelationBatchOfChunksSizeBytes": str(1024**3),
    }
    config.update(overrides)
    # The tests may queue status codes in `status_failures` and `upload_failures`: each
    # ArtifactStatus or AddArtifacts call fails with the next one, while any is left.
    state = {"cache": {}, "uploads": 0, "relations": [], "message_sizes": [],
             "config": config, "config_requests": [], "config_responses": [],
             "input_tables": [], "tables": [],
             "upload_contexts": [], "status_contexts": [],
             "status_failures": [], "upload_failures": [], "upload_attempts": 0,
             "ddl_strings": []}
    server_id = "00000000-0000-0000-0000-000000000001"

    def attribution(request, context):
        grpc_user_agent = dict(context.invocation_metadata()).get("user-agent")
        assert grpc_user_agent, "artifact RPC is missing a non-empty user-agent header"
        # Keep only scalar envelope fields; do not retain another payload copy.
        return {
            "session_id": request.session_id,
            "has_user_context": request.HasField("user_context"),
            "user_id": request.user_context.user_id,
            "user_name": request.user_context.user_name,
            "has_client_type": request.HasField("client_type"),
            "client_type": request.client_type,
            "grpc_user_agent": grpc_user_agent,
        }

    def configure(raw, context):
        request = protocol("ConfigRequest").FromString(raw)
        operation = request.operation.WhichOneof("op_type")
        keys = list(getattr(request.operation, operation).keys)
        state["config_requests"].append((operation, keys))
        if config_error:
            context.abort(grpc.StatusCode.UNAVAILABLE, "configuration unavailable")
        # Injected errors abort only the next Config call.
        if error := state.pop("config_error", None):
            context.abort(error, "configuration read failed")
        # Deliberately reverse responses and omit missing keys to test key mapping.
        # A present null key leaves the optional protobuf value unset.
        response = protocol("ConfigResponse", session_id=request.session_id,
                            pairs=[dict(key=k, value=config[k]) for k in reversed(keys)
                                   if k in config])
        state["config_responses"].append(response)
        return response.SerializeToString()

    def top_level_split(text):
        parts, depth, start = [], 0, 0
        for i, char in enumerate(text):
            depth += {"<": 1, ">": -1}.get(char, 0)
            if char == "," and depth == 0:
                parts.append(text[start:i])
                start = i + 1
        return parts + [text[start:]]

    def ddl_type(text):
        """The DataType of a DDL of these tests, as the server's `ddl_parse` parses it: atomic
        types, arrays and maps of them, and structs of `name type` fields."""
        text, kinds = text.strip(), {
            "long": "long", "bigint": "long", "int": "integer", "integer": "integer",
            "double": "double", "float": "float", "string": "string", "date": "date",
            "boolean": "boolean", "binary": "binary", "timestamp": "timestamp"}
        lower = text.lower()
        if lower.startswith("array<") and lower.endswith(">"):
            return protocol("DataType", array=dict(
                element_type=ddl_type(text[6:-1]), contains_null=True))
        if lower.startswith("map<") and lower.endswith(">"):
            key, value = top_level_split(text[4:-1])
            return protocol("DataType", map=dict(
                key_type=ddl_type(key), value_type=ddl_type(value), value_contains_null=True))
        if lower in kinds:
            return protocol("DataType", **{kinds[lower]: {}})
        fields = [field.strip().split(None, 1) for field in top_level_split(text)]
        return protocol("DataType", struct=dict(fields=[
            dict(name=name, data_type=ddl_type(kind), nullable=True) for name, kind in fields]))

    def analyze(raw, context):
        # As the server, which parses the DDL schemas that the client sends (`_parse_ddl`);
        # the tests ask for no other analysis.
        request = protocol("AnalyzePlanRequest").FromString(raw)
        if request.WhichOneof("analyze") != "ddl_parse":
            context.abort(grpc.StatusCode.UNIMPLEMENTED, "Method not found!")
        state["ddl_strings"].append(request.ddl_parse.ddl_string)
        return protocol("AnalyzePlanResponse", session_id=request.session_id,
                        server_side_session_id=server_id,
                        ddl_parse=dict(parsed=ddl_type(request.ddl_parse.ddl_string))
                        ).SerializeToString()

    def statuses(raw, context):
        request = protocol("ArtifactStatusesRequest").FromString(raw)
        state["status_contexts"].append(attribution(request, context))
        if state["status_failures"]:
            context.abort(state["status_failures"].pop(0), "artifact status unavailable")
        return protocol("ArtifactStatusesResponse", session_id=request.session_id,
                        statuses={name: dict(exists=name in state["cache"])
                                  for name in request.names}).SerializeToString()

    def upload(requests, context):
        state["upload_attempts"] += 1
        if state["upload_failures"]:
            context.abort(state["upload_failures"].pop(0), "artifact upload unavailable")
        summaries = []
        name, expected, buffer = None, 0, bytearray()
        session_id = None

        def save(name, data):
            named = ("files/", "forward_to_fs/", "jars/", "classes/", "pyfiles/", "archives/")
            if not (allow_file_artifacts and name.startswith(named)):
                assert name == "cache/" + hashlib.sha256(data).hexdigest()
            assert name not in state["cache"], "cached artifacts should not be uploaded again"
            state["cache"][name] = bytes(data)
            summaries.append(dict(name=name, is_crc_successful=True))

        def checked(chunk):
            assert zlib.crc32(chunk.data) == chunk.crc
            return chunk.data

        for raw in requests:
            state["message_sizes"].append(len(raw))
            request = protocol("AddArtifactsRequest").FromString(raw)
            state["upload_contexts"].append({
                **attribution(request, context), "payload": request.WhichOneof("payload")
            })
            session_id = request.session_id
            if request.HasField("batch"):
                for artifact in request.batch.artifacts:
                    save(artifact.name, checked(artifact.data))
            elif request.HasField("begin_chunk"):
                begin = request.begin_chunk
                name, expected = begin.name, begin.total_bytes
                buffer = bytearray(checked(begin.initial_chunk))
            else:
                buffer.extend(checked(request.chunk))
            if name is not None and len(buffer) == expected:
                save(name, buffer)
                name, buffer = None, bytearray()
        assert name is None
        state["uploads"] += 1
        if upload_error == "crc":
            summaries[0]["is_crc_successful"] = False
        elif upload_error == "missing":
            summaries = []
        return protocol("AddArtifactsResponse", session_id=session_id,
                        artifacts=summaries).SerializeToString()

    def execute(raw, context):
        request = protocol("ExecutePlanRequest").FromString(raw)
        relation = request.plan.root
        while relation.HasField("to_df"):
            state["column_names"] = list(relation.to_df.column_names)
            relation = relation.to_df.input
        state["relations"].append(relation)
        if relation.HasField("chunked_cached_local_relation"):
            cached = relation.chunked_cached_local_relation
            payloads = [state["cache"]["cache/" + h] for h in cached.dataHashes]
            state["spark_schema"] = (
                state["cache"]["cache/" + cached.schemaHash].decode()
                if cached.HasField("schemaHash") else None
            )
        else:
            payloads = [relation.local_relation.data]
            state["spark_schema"] = (
                relation.local_relation.schema
                if relation.local_relation.HasField("schema") else None
            )
        rows, first, last = 0, None, None
        state["arrow_schemas"], state["batch_rows"] = [], []
        input_batches = []
        for payload in payloads:
            if capture_tables:
                state["tables"].append(pa.ipc.open_stream(payload).read_all())
            reader = pa.ipc.open_stream(payload)
            state["arrow_schemas"].append(reader.schema)
            stream_rows = []
            state["batch_rows"].append(stream_rows)
            for batch in reader:
                stream_rows.append(batch.num_rows)
                input_batches.append(batch)
                rows += batch.num_rows
                if batch.num_rows:
                    if first is None:
                        first = batch.column(0)[0].as_py()
                    last = batch.column(0)[-1].as_py()
        state["input_tables"].append(pa.Table.from_batches(
            input_batches, schema=state["arrow_schemas"][0]))
        state["result"] = (rows, first, last)
        result = pa.table({"rows": [rows]})
        sink = pa.BufferOutputStream()
        with pa.ipc.new_stream(sink, result.schema) as writer:
            writer.write_table(result)
        header = dict(session_id=request.session_id, operation_id=request.operation_id,
                      server_side_session_id=server_id)
        yield protocol("ExecutePlanResponse", **header, response_id="0",
                       arrow_batch=dict(row_count=1, data=sink.getvalue().to_pybytes())
                       ).SerializeToString()
        yield protocol("ExecutePlanResponse", **header, response_id="1",
                       result_complete={}).SerializeToString()

    def release(raw, context):
        request = protocol("ReleaseExecuteRequest").FromString(raw)
        return protocol("ReleaseExecuteResponse", session_id=request.session_id,
                        operation_id=request.operation_id,
                        server_side_session_id=server_id).SerializeToString()

    def stop(raw, context):
        request = protocol("ReleaseSessionRequest").FromString(raw)
        return protocol("ReleaseSessionResponse", session_id=request.session_id,
                        server_side_session_id=server_id).SerializeToString()

    with ThreadPoolExecutor(max_workers=2) as executor:
        server = grpc.server(executor, options=[("grpc.max_receive_message_length", 4 * 1024**2)])
        server.add_generic_rpc_handlers([grpc.method_handlers_generic_handler(
            "spark.connect.SparkConnectService", {
                "Config": grpc.unary_unary_rpc_method_handler(configure),
                "AnalyzePlan": grpc.unary_unary_rpc_method_handler(analyze),
                "ArtifactStatus": grpc.unary_unary_rpc_method_handler(statuses),
                "AddArtifacts": grpc.stream_unary_rpc_method_handler(upload),
                "ExecutePlan": grpc.unary_stream_rpc_method_handler(execute),
                "ReleaseExecute": grpc.unary_unary_rpc_method_handler(release),
                "ReleaseSession": grpc.unary_unary_rpc_method_handler(stop),
            })])
        port = server.add_insecure_port("127.0.0.1:0")
        server.start()
        remote = f"sc://127.0.0.1:{port}/"
        for key, value in [("user_id", user_id), ("user_agent", user_agent)]:
            if value is not None:
                remote += f";{key}={quote(value, safe='')}"
        session = SparkSession.builder.remote(remote).getOrCreate()
        try:
            yield session, state
        finally:
            session.stop()
            server.stop(0).wait(timeout=5)


def _assert_artifact_contexts(contexts, *, session_id, user_id, user_agent):
    assert contexts
    client_types = {context["client_type"] for context in contexts}
    assert len(client_types) == 1
    client_type = client_types.pop()
    assert client_type
    assert client_type.startswith(user_agent + " spark/")
    for context in contexts:
        grpc_user_agent = context["grpc_user_agent"]
        assert grpc_user_agent == client_type or grpc_user_agent.startswith(client_type + " ")
        assert context["session_id"] == session_id
        # PySpark's ArtifactManager sends a present context even without a user ID.
        assert context["has_user_context"]
        assert context["user_id"] == user_id
        assert context["user_name"] == ""
        assert context["has_client_type"]


@pytest.mark.parametrize("user_id,user_agent", [
    (None, None),
    ("artifact-user@example.com", "artifact wire test"),
], ids=["default-user", "configured-user"])
def test_cached_arrow_upload_attribution_on_every_frame(protocol, monkeypatch, user_id, user_agent):
    from pyspark.sql import types as T

    # Exercise ChannelBuilder's default prefix when no URL or environment value is set.
    monkeypatch.delenv("SPARK_CONNECT_USER_AGENT", raising=False)
    table = pa.table({"id": range(10_000)})
    schema = T.StructType([T.StructField("id", T.LongType())])
    with _arrow_input_service(protocol, user_id=user_id, user_agent=user_agent, **{
        "spark.sql.session.localRelationCacheThreshold": "0",
    }) as (spark, state):
        # The explicit schema is batched; the IPC data spans begin/chunk frames.
        spark.createDataFrame(table, schema)
        assert {context["payload"] for context in state["upload_contexts"]} == {
            "batch", "begin_chunk", "chunk"
        }
        assert state["status_contexts"]
        _assert_artifact_contexts(
            state["upload_contexts"] + state["status_contexts"],
            session_id=spark.session_id,
            user_id=user_id or "",
            user_agent=user_agent or "_SPARK_CONNECT_PYTHON",
        )


@pytest.mark.parametrize("user_id,user_agent", [
    (None, None),
    ("file-user@example.com", "file wire test"),
], ids=["default-user", "configured-user"])
@pytest.mark.parametrize("upload", ["file", "named"])
def test_file_upload_attribution_on_every_frame(
    protocol, monkeypatch, tmp_path, user_id, user_agent, upload
):
    monkeypatch.delenv("SPARK_CONNECT_USER_AGENT", raising=False)
    small, large = tmp_path / "small.txt", tmp_path / "large.txt"
    small.write_bytes(b"small artifact")
    chunk_size = 32 * 1024  # Mirrors CHUNK_SIZE in crates/spark-connect-core/src/artifact.rs.
    large_size = chunk_size * 3 + 1  # BeginChunk followed by three Chunk frames.
    large.write_bytes(bytes(index % 256 for index in range(large_size)))
    with _arrow_input_service(
        protocol, user_id=user_id, user_agent=user_agent, allow_file_artifacts=True
    ) as (spark, state):
        if upload == "file":
            spark.addArtifacts(str(small), str(large), file=True)
            artifact_names = ["files/small.txt", "files/large.txt"]
        else:
            destinations = ["/wire/small.txt", "/wire/large.txt"]
            for path, destination in zip([small, large], destinations):
                spark.copyFromLocalToFs(str(path), destination)
            # As PySpark's FORWARD_TO_FS_PREFIX + dest_path: the path is absolute, so no separator.
            artifact_names = ["forward_to_fs" + destination for destination in destinations]
        payloads = {context["payload"] for context in state["upload_contexts"]}
        assert payloads == {"batch", "begin_chunk", "chunk"}, (
            f"Expected all artifact frame kinds, got {payloads}. "
            f"This fixture assumes a {chunk_size}-byte CHUNK_SIZE; "
            "adjust its sizes if CHUNK_SIZE changes."
        )
        assert state["cache"][artifact_names[0]] == small.read_bytes()
        assert state["cache"][artifact_names[1]] == large.read_bytes()
        # One upload per path, as PySpark's add_artifacts makes one.
        assert state["uploads"] == 2
        assert not state["status_contexts"]
        _assert_artifact_contexts(
            state["upload_contexts"],
            session_id=spark.session_id,
            user_id=user_id or "",
            user_agent=user_agent or "_SPARK_CONNECT_PYTHON",
        )


def test_cached_arrow_upload_retries_a_transient_failure_from_the_status_check(protocol):
    # As SparkConnectClient.cache_artifacts, the status check and the upload are one attempt
    # under the retry policy: a transient failure of either is retried from the status check.
    table = pa.table({"id": range(10_000)})
    with _arrow_input_service(protocol, **{
        "spark.sql.session.localRelationCacheThreshold": "0",
    }) as (spark, state):
        state["status_failures"].append(grpc.StatusCode.UNAVAILABLE)
        state["upload_failures"].append(grpc.StatusCode.UNAVAILABLE)
        df = spark.createDataFrame(table)
        assert not state["status_failures"] and not state["upload_failures"]
        assert state["upload_attempts"] == state["uploads"] + 1
        # The failed status check, the one before the failed upload, and the one before the
        # upload; the server then reads the cached data: its rows, first and last values.
        assert len(state["status_contexts"]) == 3
        df.toArrow()
        assert state["result"] == (10_000, 0, 9_999)


def test_cached_arrow_upload_does_not_retry_another_failure(protocol):
    from pyspark.errors.exceptions.connect import SparkConnectGrpcException

    table = pa.table({"id": range(10_000)})
    with _arrow_input_service(protocol, **{
        "spark.sql.session.localRelationCacheThreshold": "0",
    }) as (spark, state):
        state["upload_failures"].append(grpc.StatusCode.PERMISSION_DENIED)
        with pytest.raises(SparkConnectGrpcException, match="artifact upload unavailable"):
            spark.createDataFrame(table)
        assert state["upload_attempts"] == 1 and state["uploads"] == 0


def test_artifact_upload_retries_a_transient_failure_of_each_path(protocol, tmp_path):
    # As SparkConnectClient.add_artifacts, each path is uploaded in an attempt of its own.
    first, second = tmp_path / "first.txt", tmp_path / "second.txt"
    first.write_bytes(b"first artifact")
    second.write_bytes(b"second artifact")
    with _arrow_input_service(protocol, allow_file_artifacts=True) as (spark, state):
        state["upload_failures"].append(grpc.StatusCode.UNAVAILABLE)
        spark.addArtifacts(str(first), str(second), file=True)
        assert state["cache"]["files/first.txt"] == b"first artifact"
        assert state["cache"]["files/second.txt"] == b"second artifact"
        assert (state["upload_attempts"], state["uploads"]) == (3, 2)


@pytest.mark.parametrize("cached", [False, True], ids=["inline", "cached"])
@pytest.mark.parametrize("explicit", [False, True], ids=["inferred", "explicit"])
def test_arrow_input_schema_handoff(protocol, cached, explicit):
    import json
    from pyspark.sql import types as T

    table = pa.table({"id": [1, 2]})
    schema = T.StructType([T.StructField("id", T.LongType())]) if explicit else None
    with _arrow_input_service(protocol, **{
        "spark.sql.session.localRelationCacheThreshold": "0" if cached else "1048576",
    }) as (spark, state):
        frame = spark.createDataFrame(table, schema)
        # As in PySpark: inferred column names are applied with toDF, which is not local.
        assert frame.isLocal() == explicit
        assert frame.toArrow().column(0).to_pylist() == [2]
        assert state["result"] == (2, 1, 2)
        assert state["arrow_schemas"] == [table.schema]
        assert state["config_requests"] == [("get", _NORMALIZATION_KEYS + _UPLOAD_KEYS)]
        relation = state["relations"][0]
        if cached:
            local = relation.chunked_cached_local_relation
            assert local.HasField("schemaHash") == explicit
            assert len(state["cache"]) == (2 if explicit else 1)
            uploads = state["uploads"]
            spark.createDataFrame(table, schema).toArrow()
            assert state["uploads"] == uploads
        else:
            assert state["uploads"] == 0
            assert relation.local_relation.HasField("schema") == explicit
        if explicit:
            assert json.loads(state["spark_schema"]) == json.loads(schema.json())
        else:
            assert state["spark_schema"] is None
            assert state["column_names"] == ["id"]


def test_arrow_input_upload_above_grpc_limit(protocol):
    import numpy as np

    rows = 128 * 1024**2 // 8 + 1
    table = pa.table({"id": np.arange(rows, dtype=np.int64)})
    assert table.nbytes > 128 * 1024**2
    with _arrow_input_service(protocol) as (spark, state):
        df = spark.createDataFrame(table)
        assert not df.isLocal()
        assert df.toArrow().column(0).to_pylist() == [rows]
        assert state["result"] == (rows, 0, rows - 1)
        assert state["relations"][0].HasField("chunked_cached_local_relation")
        assert state["relations"][0].ByteSize() < 1024**2
        assert max(state["message_sizes"]) < 33 * 1024
        assert state["spark_schema"] is None
        assert all(schema == table.schema for schema in state["arrow_schemas"])


@pytest.mark.parametrize("kind", ["inferred", "empty-list", "struct"])
def test_arrow_input_zero_column_table(protocol, kind):
    from pyspark.sql import types as T

    # The input cast drops the rows of a zero-column table, leaving no batches.
    table = pa.table({"a": [1, 2]}).select([])
    schema = {"inferred": None, "empty-list": [], "struct": T.StructType([])}[kind]
    explicit = kind == "struct"
    with _arrow_input_service(protocol) as (spark, state):
        frame = spark.createDataFrame(table, schema)
        # As in PySpark, no toDF is applied without column names, so the frame is local.
        assert frame.isLocal()
        assert frame.toArrow().column(0).to_pylist() == [0]
        assert "column_names" not in state
        # The relation still carries the Arrow schema, as PySpark's IPC stream does.
        assert state["arrow_schemas"] == [pa.schema([])]
        assert state["relations"][0].local_relation.HasField("schema") == explicit


@pytest.mark.parametrize("schema,expected_width", [
    ("x long", "1"),
    ("x long, y long, z long", "3"),
    ("long", "1"),
])
def test_arrow_input_ddl_width_errors(protocol, schema, expected_width):
    from pyspark.errors import PySparkValueError

    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(PySparkValueError) as error:
            spark.createDataFrame(pa.table({"x": [1], "y": [2]}), schema)
        assert error.value.getCondition() == "AXIS_LENGTH_MISMATCH"
        assert error.value.getMessageParameters() == {
            "expected_length": expected_width, "actual_length": "2",
        }
        # As PySpark Connect's, the server parses the DDL schema.
        assert state["ddl_strings"] == [schema]
        assert not state["relations"]
        assert state["uploads"] == 0


@pytest.mark.parametrize("schema,expected_type", [
    ("array<long>", "ArrayType"),
    ("map<string,long>", "MapType"),
])
def test_arrow_input_unsupported_ddl_schema(protocol, schema, expected_type):
    from pyspark.errors import PySparkTypeError

    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(PySparkTypeError) as error:
            spark.createDataFrame(pa.table({"x": [1]}), schema)
        assert error.value.getCondition() == "NOT_EXPECTED_TYPE"
        assert error.value.getMessageParameters() == {
            "arg_name": "schema", "expected_type": "list, None or StructType",
            "arg_type": expected_type,
        }
        assert not state["relations"]
        assert state["uploads"] == 0


def test_arrow_input_upload_limits_and_chunk_order(protocol):
    import numpy as np
    from pyspark.errors import AnalysisException
    table = pa.table({"id": np.arange(100, dtype=np.int64)})
    options = {
        "spark.sql.session.localRelationCacheThreshold": "0",
        "spark.sql.session.localRelationChunkSizeRows": "7",
        "spark.sql.session.localRelationChunkSizeBytes": "500",
        "spark.sql.session.localRelationBatchOfChunksSizeBytes": "1000",
    }
    with _arrow_input_service(protocol, **options) as (spark, state):
        df = spark.createDataFrame(table)
        df.toArrow()
        assert state["result"] == (100, 0, 99)
        assert state["uploads"] > 1
        cached = state["relations"][0].chunked_cached_local_relation
        assert len(cached.dataHashes) > 1
        assert all(rows <= 7 for stream in state["batch_rows"] for rows in stream)
        assert any(sum(stream) > 7 for stream in state["batch_rows"])
        actual = [
            value
            for hash in cached.dataHashes
            for batch in pa.ipc.open_stream(state["cache"]["cache/" + hash])
            for value in batch.column(0).to_pylist()
        ]
        assert actual == list(range(100))

    options["spark.sql.session.localRelationSizeLimit"] = "10"
    with _arrow_input_service(protocol, **options) as (spark, state):
        with pytest.raises(AnalysisException, match="LOCAL_RELATION_SIZE_LIMIT_EXCEEDED"):
            spark.createDataFrame(table)
        assert not state["relations"]
        assert state["uploads"] == 0


@pytest.mark.parametrize("setting,value", [
    ("SizeLimit", "0"),
    ("SizeLimit", "-1"),
    ("SizeLimit", "18446744073709551616"),
    ("ChunkSizeRows", "0"),
])
def test_arrow_input_rejects_invalid_upload_sizes(protocol, setting, value):
    key = f"spark.sql.session.localRelation{setting}"
    with _arrow_input_service(protocol, **{
        "spark.sql.session.localRelationCacheThreshold": "0", key: value,
    }) as (spark, state):
        with pytest.raises(Exception, match=f"Invalid configuration: {key}"):
            spark.createDataFrame(pa.table({"id": [1]}))
        assert state["uploads"] == 0


@pytest.mark.parametrize("failure,match", [("crc", "CRC check failed"),
                                          ("missing", "Missing cache artifact")])
@pytest.mark.parametrize("rows", [2, 4000000])
def test_arrow_input_upload_failure(protocol, failure, match, rows):
    import numpy as np

    # The larger input spans multiple upload batches.
    with _arrow_input_service(protocol, upload_error=failure,
                             **{"spark.sql.session.localRelationCacheThreshold": "0",
                                "spark.sql.session.localRelationBatchOfChunksSizeBytes":
                                    str(8 * 1024**2)}) as (spark, state):
        with pytest.raises(Exception, match=match):
            spark.createDataFrame(pa.table({"id": np.arange(rows, dtype=np.int64)}))
        assert not state["relations"]


@pytest.mark.parametrize("batch_bytes", [500, 1024**3])
def test_arrow_input_duplicate_chunks_preserve_multiplicity(protocol, batch_bytes):
    options = {"spark.sql.session.localRelationCacheThreshold": "0",
               "spark.sql.session.localRelationChunkSizeRows": "5",
               "spark.sql.session.localRelationChunkSizeBytes": "1",
               "spark.sql.session.localRelationBatchOfChunksSizeBytes": str(batch_bytes)}
    with _arrow_input_service(protocol, **options) as (spark, state):
        spark.createDataFrame(pa.table({"id": [1] * 20})).toArrow()
        cached = state["relations"][0].chunked_cached_local_relation
        assert len(cached.dataHashes) == 4
        assert len(set(cached.dataHashes)) == 1
        assert not cached.HasField("schemaHash")
        assert len(state["cache"]) == 1
        assert state["result"] == (20, 1, 1)


_NORMALIZATION_KEYS = [
    "spark.sql.timestampType",
    "spark.sql.session.timeZone",
    "spark.sql.execution.arrow.useLargeVarTypes",
]
_UPLOAD_KEYS = [
    "spark.sql.session.localRelationCacheThreshold",
    "spark.sql.session.localRelationSizeLimit",
    "spark.sql.session.localRelationChunkSizeRows",
    "spark.sql.session.localRelationChunkSizeBytes",
    "spark.sql.session.localRelationBatchOfChunksSizeBytes",
]


@pytest.mark.parametrize("timestamp_type", ["TIMESTAMP_LTZ", "TIMESTAMP_NTZ"])
@pytest.mark.parametrize("zone", ["UTC", "America/Los_Angeles"])
@pytest.mark.parametrize("large", ["false", "true"])
def test_arrow_input_batched_normalization_configs(protocol, timestamp_type, zone, large):
    from zoneinfo import ZoneInfo

    value = datetime(2024, 1, 15, 12, 34, 56, 123456)
    table = pa.table({"t": [value, None], "s": ["hello", None], "b": [b"xyz", None]})
    with _arrow_input_service(protocol, **dict(zip(
        _NORMALIZATION_KEYS, [timestamp_type, zone, large]
    ))) as (spark, state):
        frame = spark.createDataFrame(table)
        assert state["config_requests"] == [("get", _NORMALIZATION_KEYS + _UPLOAD_KEYS)]
        frame.collect()
        expected_time = (value if timestamp_type == "TIMESTAMP_NTZ" else
                         value.replace(tzinfo=ZoneInfo(zone)).astimezone(timezone.utc))
        expected = pa.table({
            "t": pa.array([expected_time, None], type=pa.timestamp(
                "us", tz=None if timestamp_type == "TIMESTAMP_NTZ" else "UTC")),
            "s": pa.array(["hello", None], type=pa.large_string() if large == "true"
                          else pa.string()),
            "b": pa.array([b"xyz", None], type=pa.large_binary() if large == "true"
                          else pa.binary()),
        })
        actual = state["input_tables"][0]
        assert actual.schema == expected.schema
        assert actual.to_pydict() == expected.to_pydict()
        # Inferred input keeps its types in Arrow; names are applied via toDF.
        assert not state["relations"][0].local_relation.HasField("schema")


@pytest.mark.parametrize("missing", _NORMALIZATION_KEYS + _UPLOAD_KEYS)
@pytest.mark.parametrize("present_null", [False, True])
def test_arrow_input_missing_config(protocol, missing, present_null):
    with _arrow_input_service(protocol) as (spark, state):
        if present_null:
            state["config"][missing] = None
        else:
            del state["config"][missing]
        # The current Arrow settings snapshot requires every server setting.
        with pytest.raises(Exception, match="SQL_CONF_NOT_FOUND"):
            spark.createDataFrame(pa.table({"x": [1, None], "s": ["hello", None]}))
        assert state["config_requests"] == [("get", _NORMALIZATION_KEYS + _UPLOAD_KEYS)]
        pairs = {pair.key: pair for pair in state["config_responses"][0].pairs}
        if present_null:
            assert missing in pairs
            assert not pairs[missing].HasField("value")
        else:
            assert missing not in pairs
        assert state["uploads"] == 0
        assert state["input_tables"] == []


def test_arrow_input_normalization_config_failure_then_fresh_read(protocol):
    with _arrow_input_service(protocol) as (spark, state):
        state["config_error"] = grpc.StatusCode.INVALID_ARGUMENT
        table = pa.table({"x": [1]})
        with pytest.raises(Exception, match="configuration read failed"):
            spark.createDataFrame(table)
        assert state["config_requests"] == [("get", _NORMALIZATION_KEYS + _UPLOAD_KEYS)]
        assert state["uploads"] == 0
        spark.createDataFrame(table).collect()
        assert state["config_requests"] == [
            ("get", _NORMALIZATION_KEYS + _UPLOAD_KEYS),
            ("get", _NORMALIZATION_KEYS + _UPLOAD_KEYS)]
        assert state["input_tables"][0].to_pydict() == {"x": [1]}


def test_arrow_input_normalization_config_transient_retry(protocol):
    with _arrow_input_service(protocol) as (spark, state):
        state["config_error"] = grpc.StatusCode.UNAVAILABLE
        spark.createDataFrame(pa.table({"x": [1]})).collect()
        assert state["config_requests"] == [
            ("get", _NORMALIZATION_KEYS + _UPLOAD_KEYS),
            ("get", _NORMALIZATION_KEYS + _UPLOAD_KEYS)]
        assert state["input_tables"][0].to_pydict() == {"x": [1]}


def test_arrow_input_normalization_configs_are_fresh(protocol):
    value = datetime(2024, 1, 15, 12)
    table = pa.table({"t": [value], "s": ["hello"]})
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(table).collect()
        state["config"].update(dict(zip(_NORMALIZATION_KEYS, [
            "TIMESTAMP_NTZ", "America/Los_Angeles", "true"])))
        spark.createDataFrame(table).collect()
        state["config"].update(dict(zip(_NORMALIZATION_KEYS, [
            "TIMESTAMP_LTZ", "America/Los_Angeles", "false"])))
        spark.createDataFrame(table).collect()
        assert state["config_requests"] == [("get", _NORMALIZATION_KEYS + _UPLOAD_KEYS)] * 3
        tables = state["input_tables"]
        assert [t.column("t")[0].as_py() for t in tables] == [
            value.replace(tzinfo=timezone.utc), value,
            datetime(2024, 1, 15, 20, tzinfo=timezone.utc)]
        assert [t.schema.field("s").type for t in tables] == [
            pa.string(), pa.large_string(), pa.string()]
        assert all(not relation.local_relation.HasField("schema")
                   for relation in state["relations"])


def test_arrow_input_empty_skips_all_configs(protocol):
    from pyspark.sql.types import LongType, StructField, StructType

    with _arrow_input_service(protocol) as (spark, state):
        table = pa.table({"x": pa.array([], type=pa.int64())})
        spark.createDataFrame(table, StructType([StructField("x", LongType())]))
        with pytest.raises(Exception, match="CANNOT_INFER_EMPTY_SCHEMA"):
            spark.createDataFrame(table)
        assert state["config_requests"] == []
        assert state["uploads"] == 0


@pytest.mark.parametrize("data", [[[1]], pd.DataFrame({"id": [1]})])
def test_non_arrow_input_does_not_import_pyarrow(protocol, monkeypatch, data):
    import builtins
    import sys

    imported = []
    original_import = builtins.__import__

    def import_module(name, *args, **kwargs):
        if name == "pyarrow":
            imported.append(name)
            raise ImportError("PyArrow is unavailable")
        return original_import(name, *args, **kwargs)

    with _arrow_input_service(protocol) as (spark, state):
        monkeypatch.setitem(sys.modules, "pyarrow", None)
        monkeypatch.setattr(builtins, "__import__", import_module)
        spark.createDataFrame(data)
        assert imported == []
        assert state["config_requests"] == []


def test_add_artifacts_names_each_artifact_as_pyspark(protocol, tmp_path, monkeypatch):
    import sys

    for name in ["a.jar", "A.class", "m.py", "b.zip", "n.txt"]:
        (tmp_path / name).write_bytes(name.encode())
    monkeypatch.setattr(sys, "path", list(sys.path))
    with _arrow_input_service(protocol, allow_file_artifacts=True) as (spark, state):
        # As PySpark's ArtifactManager names them, by the flags and the file extension. A class
        # file is uploaded only with the file flag: PySpark refuses it otherwise.
        spark.addArtifacts(str(tmp_path / "a.jar"))
        spark.addArtifacts(str(tmp_path / "m.py"), pyfile=True)
        spark.addArtifacts(str(tmp_path / "b.zip"), pyfile=True)
        spark.addArtifacts(str(tmp_path / "b.zip") + "#alias", archive=True)
        spark.addArtifact(str(tmp_path / "n.txt"), file=True)
        spark.addArtifacts(str(tmp_path / "A.class"), file=True)
        assert state["cache"] == {
            "jars/a.jar": b"a.jar",
            "pyfiles/m.py": b"m.py",
            "pyfiles/b.zip": b"b.zip",
            "archives/b.zip#alias": b"b.zip",
            "files/n.txt": b"n.txt",
            "files/A.class": b"A.class",
        }
        # As PySpark's _parse_artifacts, a zip of pyfile is importable by the client.
        assert sys.path[1] == str(tmp_path / "b.zip")


# Local relations of at least 32 MiB read the server's local-relation settings, and are uploaded
# as cache artifacts at its threshold, whichever path encoded them; a single gRPC message
# cannot carry a relation beyond the server's message size limit.

@pytest.mark.parametrize("path", ["local_rows", "pandas_rows", "float64", "strings"])
def test_large_local_relations_follow_the_cache_threshold(protocol, path):
    # (The mock service keeps gRPC's default 4 MiB message limit, so the inline variant of
    # these inputs, below a higher threshold, can only be checked against a real server.)
    import numpy as np

    if path == "local_rows":
        data = [("y" * 2000, i) for i in range(20_000)]
        schema = T.StructType([T.StructField("v", T.StringType()), T.StructField("i", T.LongType())])
        expected_rows = 20_000
    elif path == "pandas_rows":
        # float32 columns into DoubleType keep the row path.
        data = pd.DataFrame(np.ones((100_000, 45), dtype=np.float32),
                            columns=[f"c{i}" for i in range(45)])
        schema = T.StructType([T.StructField(c, T.DoubleType()) for c in data.columns])
        expected_rows = 100_000
    elif path == "float64":
        data = pd.DataFrame(np.ones((100_000, 45)), columns=[f"c{i}" for i in range(45)])
        schema = T.StructType([T.StructField(c, T.DoubleType()) for c in data.columns])
        expected_rows = 100_000
    else:
        data = pd.DataFrame({"v": ["x" * 3500] * 10_000}, dtype=object)
        schema = T.StructType([T.StructField("v", T.StringType())])
        expected_rows = 10_000
    overrides = {"spark.sql.session.localRelationCacheThreshold": "1"}
    with _arrow_input_service(protocol, **overrides) as (spark, state):
        df = spark.createDataFrame(data, schema)
        df.toArrow()
        assert sum(table.num_rows for table in state["input_tables"]) == expected_rows
        assert state["uploads"]
        assert any("spark.sql.session.localRelationCacheThreshold" in keys
                   for _, keys in state["config_requests"])


def test_small_local_relations_stay_inline_without_reading_settings(protocol):
    with _arrow_input_service(protocol) as (spark, state):
        schema = T.StructType([T.StructField("v", T.StringType()), T.StructField("i", T.LongType())])
        spark.createDataFrame([("y" * 100, i) for i in range(100)], schema).toArrow()
        assert not state["uploads"]
        assert not any("spark.sql.session.localRelationCacheThreshold" in keys
                       for _, keys in state["config_requests"])
        assert state["input_tables"][0].num_rows == 100


# Signed-int64 pandas input uses the same Arrow importer and mock service.

@pytest.mark.parametrize("storage", ["object", "default", "python", "pyarrow", "python_nan", "pyarrow_nan"])
@pytest.mark.parametrize("dtype", ["int64", "Int64"])
def test_integer_canonical_columns_storage_stays_columnar(protocol, monkeypatch, storage, dtype):
    from pyspark.sql._pandas_input import _has_canonical_columns

    values = [2**53 + 1, -(2**63), 2**63 - 1, 0] * 250
    if dtype == "Int64":
        values[1] = None
    frame = pd.DataFrame({"original": pd.Series(values, dtype=dtype)})
    if storage == "object":
        frame.columns = pd.Index(["original"], dtype=object)
    elif storage != "default":
        kind = storage.removesuffix("_nan")
        label_dtype = (pd.StringDtype(storage=kind, na_value=np.nan)
                       if storage.endswith("_nan") else pd.StringDtype(storage=kind))
        frame.columns = pd.Index(pd.array(["original"], dtype=label_dtype))
    assert _has_canonical_columns(frame.columns)
    assert prepare_int64_table(frame, None).column(0).to_pylist() == values

    def no_rows(*args, **kwargs):
        raise AssertionError("ordinary columns must retain integer columnar construction")

    monkeypatch.setattr(pd.DataFrame, "astype", no_rows)
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame).toArrow()
        assert state["input_tables"][0].column(0).to_pylist() == values
        assert state["config_requests"]


@pytest.mark.parametrize("dtype", ["int64", "Int64"])
@pytest.mark.parametrize("schema_kind", ["inferred", "names", "explicit"])
@pytest.mark.parametrize("location", [
    "subclass", "class", "owner", "backing_subclass_item", "backing_class_iter",
    "backing_class_getitem", "backing_instance_box",
])
@pytest.mark.parametrize("behavior", ["raise", "mutate"])
def test_integer_columns_iterator_hooks_keep_native_order(
    protocol, monkeypatch, dtype, schema_kind, location, behavior
):
    from pandas.core.base import IndexOpsMixin

    values = [2**53 + 1, -(2**63), 2**63 - 1, 0] * 250
    if dtype == "Int64":
        values[1] = None
    frame = pd.DataFrame({"original": pd.Series(values, dtype=dtype)})
    schema = None
    expected_name = "original"
    nullable = True
    metadata = {}
    if schema_kind == "names":
        schema = ["renamed"]
        expected_name = "renamed"
    elif schema_kind == "explicit":
        nullable, metadata = dtype == "Int64", {"source": "native"}
        schema = T.StructType([T.StructField("renamed", T.LongType(), nullable, metadata)])
        expected_name = "renamed"
    original_iter = pd.Index.__iter__
    original_notna, original_astype = frame.notna, frame.astype
    calls = []
    invoked = False

    class ColumnsError(RuntimeError):
        pass

    def notna(*args, **kwargs):
        calls.append("notna")
        return original_notna(*args, **kwargs)

    def astype(*args, **kwargs):
        calls.append("astype")
        return original_astype(*args, **kwargs)

    def before_hook():
        nonlocal invoked
        if invoked:
            return
        invoked = True
        calls.append("columns iterator")
        if behavior == "raise":
            raise ColumnsError("columns hook precedes pandas normalization")
        # Install spies only when native iteration reaches the columns hook.
        # Before eligibility, normalization methods are the canonical methods.
        monkeypatch.setitem(vars(frame), "notna", notna)
        monkeypatch.setitem(vars(frame), "astype", astype)
        frame.iat[0, 0] = 2**53 + 7
        frame.iat[-1, 0] = None if dtype == "Int64" else -(2**63) + 1

    def iterate(index):
        if index is columns:
            before_hook()
        return original_iter(index)

    if location == "subclass":
        class Columns(pd.Index):
            pass
        monkeypatch.setattr(Columns, "__iter__", iterate)
        labels = pd.Index(["original"], dtype=object)
        frame.columns = Columns._simple_new(labels._values, name=None)
        assert type(frame.columns) is Columns
    elif location == "backing_subclass_item":
        # IndexOpsMixin iteration calls ndarray.item, rather than ndarray.__iter__.
        class Names(np.ndarray):
            def item(self, *args):
                before_hook()
                return super().item(*args)
        backing = np.asarray(["original"], dtype=object).view(Names)
        frame.columns = pd.Index._simple_new(backing, name=None)
        assert type(frame.columns._values) is Names
    elif location.startswith("backing_"):
        frame.columns = pd.Index(pd.array(["original"], dtype=pd.StringDtype(storage="python")))
        backing = frame.columns._values
        if location == "backing_class_iter":
            original = type(backing).__iter__

            def backing_iter(array):
                if array is backing:
                    before_hook()
                return original(array)

            monkeypatch.setattr(type(backing), "__iter__", backing_iter)
        elif location == "backing_class_getitem":
            original = type(backing).__getitem__

            def backing_getitem(array, key):
                if array is backing:
                    before_hook()
                return original(array, key)

            monkeypatch.setattr(type(backing), "__getitem__", backing_getitem)
        else:
            original = backing._box_func

            def box(value):
                before_hook()
                return original(value)

            monkeypatch.setitem(vars(backing), "_box_func", box)
    else:
        monkeypatch.setattr(pd.Index if location == "class" else IndexOpsMixin,
                            "__iter__", iterate)
    columns = frame.columns

    with _arrow_input_service(protocol) as (spark, state):
        if behavior == "raise":
            with pytest.raises(ColumnsError,
                               match="columns hook precedes pandas normalization"):
                spark.createDataFrame(frame, schema)
            assert calls == ["columns iterator"]
            assert state["config_requests"] == []
            assert state["input_tables"] == []
        else:
            spark.createDataFrame(frame, schema).toArrow()
            expected = values.copy()
            expected[0], expected[-1] = 2**53 + 7, (None if dtype == "Int64" else -(2**63) + 1)
            table = state["input_tables"][0]
            assert table.num_rows == 1000
            assert table.column_names == [expected_name]
            assert table.column(0).type == pa.int64()
            assert table.column(0).to_pylist() == expected
            assert table.column(0).null_count == sum(value is None for value in expected)
            assert table.schema.field(0).nullable is nullable
            assert table.schema.field(0).metadata is None
            import json
            logical = json.loads(state["relations"][0].local_relation.schema)
            assert logical == {"type": "struct", "fields": [{
                "name": expected_name, "type": "long", "nullable": nullable,
                "metadata": metadata,
            }]}
            assert state["config_requests"] == []
            assert calls == ["columns iterator", "notna", "astype"]


def test_integer_rebound_index_values_keeps_native_error(protocol, monkeypatch):
    frame = pd.DataFrame({"original": np.arange(1000, dtype=np.int64)})
    schema = T.StructType([T.StructField("renamed", T.LongType())])
    columns = frame.columns
    original = pd.Index._values.fget
    calls = []

    def values(index):
        if index is columns:
            calls.append("_values")
            raise RuntimeError("columns backing hook precedes normalization")
        return original(index)

    monkeypatch.setattr(pd.Index, "_values", property(values))
    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(RuntimeError,
                           match="columns backing hook precedes normalization"):
            spark.createDataFrame(frame, schema)
        assert calls == ["_values"]
        assert state["config_requests"] == []
        assert state["input_tables"] == []


@pytest.mark.parametrize("schema_kind", ["inferred", "names", "explicit"])
@pytest.mark.parametrize("verify", [True, False])
def test_integer_columns_preserve_values_without_python_rows(protocol, monkeypatch, schema_kind, verify):
    frame = pd.DataFrame({
        "a": pd.Series([2**53 + 1, None, -(2**63)], dtype="Int64"),
        "b": np.array([2**63 - 1, -1, 0], dtype=np.int64),
    })
    schema = None
    names = ["a", "b"]
    if schema_kind == "names":
        schema = names = ["renamed_a", "renamed_b"]
    elif schema_kind == "explicit":
        schema = T.StructType([T.StructField("a", T.LongType()),
                               T.StructField("b", T.LongType(), False, {"source": "test"})])
    import pyspark.sql._arrow_input as helper
    original = helper.prepare_table
    observed = []

    def capture(*args):
        result = original(*args)
        observed.append(result)
        return result

    def no_rows(*args, **kwargs):
        raise AssertionError("supported pandas input must not materialize Python rows")

    monkeypatch.setattr(helper, "prepare_table", capture)
    monkeypatch.setattr(pd.DataFrame, "astype", no_rows)
    with _arrow_input_service(protocol) as (spark, state):
        df = spark.createDataFrame(frame, schema=schema, verifySchema=verify)
        # Every nonempty pandas input applies its names through toDF, including
        # explicit struct schemas, just as the current row fallback does.
        assert not df.isLocal()
        assert df.toArrow().column(0).to_pylist() == [3]
        assert state["column_names"] == names
        assert state["result"] == (3, 2**53 + 1, -(2**63))
    table, resolved, renamed = observed[0]
    assert table.to_pydict() == {"a": [2**53 + 1, None, -(2**63)],
                                "b": [2**63 - 1, -1, 0]}
    assert (renamed or resolved.names) == names
    if schema_kind == "explicit":
        assert resolved == schema
        assert not table.schema.field(1).nullable
        assert resolved.fields[1].metadata == {"source": "test"}


@pytest.mark.parametrize("dtype", ["int64", "Int64"])
def test_empty_integer_frame_requires_explicit_schema(protocol, dtype):
    frame = pd.DataFrame({"a": pd.Series([], dtype=dtype)})
    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(ValueError):
            spark.createDataFrame(frame)
        df = spark.createDataFrame(frame, T.StructType([T.StructField("a", T.LongType())]))
        assert df.isLocal()


@pytest.mark.parametrize("values", [
    [1.0, float("nan")], [True, False], ["x", None],
    [datetime(2024, 1, 1), None],
    [datetime(2024, 1, 1, tzinfo=timezone.utc), None],
    [date(2024, 1, 1), None], [Decimal("1.20"), None],
    [[1, 2], None], [{"a": 1}, None],
])
def test_other_dtypes_keep_row_fallback(values):
    frame = pd.DataFrame({"a": values})
    assert prepare_int64_table(frame, None) is None
    assert prepare_int64_table(frame, T.StructType([T.StructField("a", T.LongType())])) is None


@pytest.mark.parametrize("schema", [
    "a long", T.LongType(), ["a", "b"], [1],
    T.StructType([T.StructField("a", T.IntegerType())]),
    T.StructType([T.StructField("a", T.DoubleType())]),
    T.StructType([]),
])
def test_schema_conversions_keep_row_fallback(schema):
    assert prepare_int64_table(pd.DataFrame({"a": [1, 2]}), schema) is None


def test_duplicate_names_fallback_and_nullable_masks():
    frame = pd.DataFrame([[2**53 + 1, 2**53 + 3]], columns=["x", "x"], dtype="int64")
    assert prepare_int64_table(frame, None) is None
    assert prepare_int64_table(pd.DataFrame({0: [1]}), None) is None
    frame = pd.DataFrame({"a": pd.Series([None, None], dtype="Int64")})
    assert prepare_int64_table(frame, None).column(0).to_pylist() == [None, None]


@pytest.mark.parametrize("dtype", ["int32", "uint64", "Int32", "UInt64", "int64[pyarrow]"])
def test_other_integer_dtypes_keep_row_fallback(dtype):
    assert prepare_int64_table(pd.DataFrame({"a": pd.Series([1, 2], dtype=dtype)}), None) is None


@pytest.mark.parametrize("rows", [100, 1000, 100000, 1000000])
@pytest.mark.parametrize("explicit", [False, True])
def test_integer_scales_reach_mock_service(protocol, monkeypatch, rows, explicit):
    values = np.arange(rows, dtype=np.int64) + (2**53 + 1)
    frame = pd.DataFrame({name: values for name in ("a", "b", "c")})
    schema = T.StructType([T.StructField(name, T.LongType()) for name in frame.columns])

    def no_rows(*args, **kwargs):
        raise AssertionError("integer input must stay columnar at every scale")

    if rows >= 1000:
        monkeypatch.setattr(pd.DataFrame, "astype", no_rows)
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema if explicit else None).toArrow()
        assert state["result"] == (rows, 2**53 + 1, 2**53 + rows)
        if frame.memory_usage(index=False).sum() > 1048576:
            assert state["relations"][0].HasField("chunked_cached_local_relation")


def test_nullable_inference_does_not_depend_on_first_value(protocol):
    frame = pd.DataFrame({"a": pd.Series([None, 2**53 + 1], dtype="Int64")})
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame).toArrow()
        assert state["result"] == (2, None, 2**53 + 1)


@pytest.mark.parametrize("verify", [True, False])
@pytest.mark.parametrize("config_error", [False, True])
def test_non_nullable_schema_is_preserved(protocol, verify, config_error):
    frame = pd.DataFrame({"a": pd.Series([1, None], dtype="Int64")})
    schema = T.StructType([T.StructField("a", T.LongType(), False)])
    assert prepare_int64_table(frame, schema) is None
    with _arrow_input_service(protocol) as (spark, state):
        if config_error:
            state["config_error"] = grpc.StatusCode.INVALID_ARGUMENT
            with pytest.raises(Exception, match="configuration read failed"):
                spark.createDataFrame(frame, schema, verifySchema=verify)
        else:
            # Official reads the configuration first, then the cast fails with a ValueError.
            with pytest.raises(ValueError) as error:
                spark.createDataFrame(frame, schema, verifySchema=verify)
            assert str(error.value) == "Casting field 'a' with null values to non-nullable"
        assert state["config_requests"][0] == ("get", _NORMALIZATION_KEYS + _UPLOAD_KEYS)
        assert not state["relations"]
        assert not state["uploads"]


def test_small_numpy_frames_avoid_importer_configuration_overhead():
    assert prepare_int64_table(pd.DataFrame({"a": np.arange(999, dtype=np.int64)}), None) is None
    assert prepare_int64_table(pd.DataFrame({"a": np.arange(1000, dtype=np.int64)}), None) is not None
    assert prepare_int64_table(pd.DataFrame({"a": pd.Series([None], dtype="Int64")}), None) is not None


@pytest.mark.parametrize("schema_kind", ["inferred", "names", "explicit"])
def test_nul_names_keep_lossless_ipc_row_fallback(protocol, schema_kind):
    import pyarrow as pa

    name = "a\x00b"
    frame = pd.DataFrame({name if schema_kind == "inferred" else "original":
                          np.arange(1000, dtype=np.int64)})
    schema = None
    if schema_kind == "names":
        schema = [name]
    elif schema_kind == "explicit":
        schema = T.StructType([T.StructField(name, T.LongType())])
    # C Stream field names are NUL-terminated; IPC preserves the full string.
    assert prepare_int64_table(frame, schema) is None
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert state["result"] == (1000, 0, 999)
        payload = state["relations"][0].local_relation.data
        if schema_kind == "explicit":
            assert pa.ipc.open_stream(payload).schema.names == [name]
        else:
            # As PySpark Connect's, the Arrow conversion of a frame without a schema names its
            # columns _0, _1, ..., which toDF names.
            assert pa.ipc.open_stream(payload).schema.names == ["_0"]
            assert state["column_names"] == [name]


@pytest.mark.parametrize("nullable", [False, True])
@pytest.mark.parametrize("rows", [999, 1000])
def test_integer_explicit_renamed_schema_preserves_all_ipc_cells(protocol, monkeypatch, nullable, rows):
    values = ([2**53 + 1, -(2**63), 2**63 - 1, 0] * 250)[:rows]
    other = [None if nullable and i % 7 == 0 else value
             for i, value in enumerate(reversed(values))]
    frame = pd.DataFrame({"original": np.asarray(values, dtype="int64"),
                          "second": pd.Series(other, dtype="Int64" if nullable else "int64")})
    schema = T.StructType([
        T.StructField("renamed", T.LongType(), False, {"source": "native", "id": 3}),
        T.StructField("nullable", T.LongType(), True, {"tag": [1, 2]}),
    ])

    def no_rows(*args, **kwargs):
        raise AssertionError("trusted integer columns must stay columnar")

    if rows >= 1000 or nullable:
        monkeypatch.setattr(pd.DataFrame, "astype", no_rows)
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        table = state["input_tables"][0]
        assert table.column_names == ["renamed", "nullable"]
        assert [column.to_pylist() for column in table.columns] == [values, other]
        assert table.schema.field(0).type == pa.int64()
        assert table.schema.field(0).nullable is False
        assert table.schema.field(1).nullable is True
        if rows >= 1000 or nullable:
            from pyspark.sql.pandas.types import from_arrow_schema
            assert from_arrow_schema(table.schema).json() == schema.json(), (
                from_arrow_schema(table.schema).json(), schema.json(), table.schema)
            assert state["config_requests"]
        else:
            assert state["config_requests"][:1] == [("get", _NORMALIZATION_KEYS + _UPLOAD_KEYS)]


def test_integer_dataframe_and_name_subclasses_keep_fallback(monkeypatch):
    class Frame(pd.DataFrame):
        pass

    class Name(str):
        pass

    class Names(list):
        pass

    frame = pd.DataFrame({"a": np.arange(1000, dtype="int64")})
    assert prepare_int64_table(Frame(frame), None) is None
    assert prepare_int64_table(frame, Names(["a"])) is None
    assert prepare_int64_table(frame, [Name("a")]) is None
    # Pandas 3 normalizes constructor labels to plain strings by default.
    # An object Index keeps the subclass this fallback regression exercises.
    named_frame = frame.copy()
    named_frame.columns = pd.Index([Name("a")], dtype=object)
    assert type(named_frame.columns[0]) is Name
    assert prepare_int64_table(named_frame, None) is None
    monkeypatch.setattr(pd, "DataFrame", Frame)
    assert prepare_int64_table(Frame(frame), None) is None


def test_integer_missing_internal_pandas_array_class_keeps_fallback():
    import sys

    script = r'''
import sys
import types
import numpy as np
import pandas as pd
from pyspark.sql._pandas_input import prepare_int64_table
frame = pd.DataFrame({"a": np.arange(1000, dtype="int64")})
nullable = pd.DataFrame({"a": pd.Series([None, 1], dtype="Int64")})
# Older pandas exposes PandasArray rather than NumpyExtensionArray. Missing
# optional internals must disable the optimization without breaking pandas input.
sys.modules["pandas.core.arrays.numpy_"] = types.ModuleType("pandas.core.arrays.numpy_")
assert prepare_int64_table(frame, None) is None
assert prepare_int64_table(nullable, None) is None
'''
    result = subprocess.run([sys.executable, "-c", script],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("class_spoof", [False, True])
def test_integer_custom_extension_dtype_cannot_select_arrow(protocol, class_spoof):
    from pandas.api.extensions import ExtensionArray, ExtensionDtype, take

    arrow_calls = []

    class Int64EqualDtype(ExtensionDtype):
        name = "int64_equal"
        type = int
        kind = "i"
        na_value = None

        @classmethod
        def construct_array_type(cls):
            return DifferentArrowArray

        def __eq__(self, other):
            return (isinstance(other, Int64EqualDtype)
                    or isinstance(other, np.dtype) and other == np.dtype("int64"))

        def __hash__(self):
            return hash(self.name)

    class DifferentArrowArray(ExtensionArray):
        def __init__(self, values):
            self._values = np.asarray(values, dtype=object)

        @property
        def dtype(self):
            return Int64EqualDtype()

        @property
        def nbytes(self):
            return self._values.nbytes

        def __len__(self):
            return len(self._values)

        def __getitem__(self, key):
            result = self._values[key]
            return type(self)(result) if isinstance(result, np.ndarray) else result

        def isna(self):
            return pd.isna(self._values)

        def take(self, indices, allow_fill=False, fill_value=None):
            return type(self)(take(self._values, indices, allow_fill=allow_fill,
                                   fill_value=fill_value))

        def copy(self):
            return type(self)(self._values.copy())

        @classmethod
        def _from_sequence(cls, scalars, *, dtype=None, copy=False):
            values = np.asarray(scalars, dtype=object)
            return cls(values.copy() if copy else values)

        def to_numpy(self, dtype=None, copy=False, na_value=None):
            if dtype is not None:
                return self._values.astype(dtype, copy=copy)
            return self._values.copy() if copy else self._values

        def __arrow_array__(self, type=None):
            arrow_calls.append("arrow")
            return pa.array([-7] * len(self), type=type)

    values = [2**53 + 1, None, -(2**63), 0] * 250
    frame = pd.DataFrame({"a": pd.Series(DifferentArrowArray(values), copy=False)})
    if class_spoof:
        Int64EqualDtype.__getattribute__ = lambda self, name: (
            type(np.dtype("int64")) if name == "__class__"
            else object.__getattribute__(self, name))
    assert frame.dtypes.iloc[0] == np.dtype("int64")
    assert isinstance(frame.dtypes.iloc[0], np.dtype) is class_spoof
    schema = T.StructType([T.StructField("renamed", T.LongType())])
    assert prepare_int64_table(frame, schema) is None
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert state["config_requests"] == []
        assert state["input_tables"][0].column(0).to_pylist() == values
    assert arrow_calls == []


@pytest.mark.parametrize("custom", ["dtype", "array"])
def test_integer_nullable_dtype_and_array_subclasses_keep_fallback(protocol, custom):
    from pandas.core.arrays.integer import IntegerArray, Int64Dtype

    arrow_calls = []

    class CustomDtype(Int64Dtype):
        pass

    class CustomArray(IntegerArray):
        @property
        def dtype(self):
            if custom == "dtype":
                return CustomDtype()
            return Int64Dtype()

        def __arrow_array__(self, type=None):
            arrow_calls.append("arrow")
            return pa.array([-7] * len(self), type=type)

    values = [2**53 + 1, 0, -(2**63), 7] * 250
    array = CustomArray(np.asarray(values, dtype="int64"), np.zeros(1000, dtype=bool))
    frame = pd.DataFrame({"a": pd.Series(array, copy=False)})
    assert type(frame["a"].array) is CustomArray
    schema = T.StructType([T.StructField("a", T.LongType())])
    assert prepare_int64_table(frame, schema) is None
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert state["config_requests"] == []
        assert state["input_tables"][0].column(0).to_pylist() == values
    assert arrow_calls == []


def _integer_guard_process(mode, argument, before_import=False):
    """Class mutation and converter imports must be isolated from other tests."""
    import sys

    if mode == "captured_alias":
        import pyspark.sql._arrow_input as importer
        if not hasattr(importer, "to_arrow_schema"):
            pytest.skip("this importer resolves converters locally and holds no captured aliases")

    script = r'''
import importlib
import importlib.util
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from pyspark.sql import types as T
from pyspark import _pyspark as native

spec = importlib.util.spec_from_file_location("integer_guard_capture", sys.argv[4])
capture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capture
spec.loader.exec_module(capture)
values = [2**53 + 1, -(2**63), 2**63 - 1, 0] * 250
frame = pd.DataFrame({"renamed": np.asarray(values, dtype="int64")})
schema = T.StructType([T.StructField("renamed", T.LongType(), False, {"source": "native"})])
mode, argument, before_import = sys.argv[1:4]
if before_import == "True":
    sys.modules.pop("pyspark.sql._pandas_input", None)
if mode == "descriptor":
    class_name, attribute = argument.split(":")
    cls = getattr(native, class_name)
    if attribute == "alias":
        setattr(T, class_name, object)
    elif attribute == "__getattribute__":
        setattr(cls, attribute, lambda self, name: object.__getattribute__(self, name))
    elif attribute == "__class__":
        setattr(cls, "__getattribute__", lambda self, name: (
            object if name == "__class__" else object.__getattribute__(self, name)))
    elif attribute == "__iter__":
        setattr(cls, attribute, lambda self: iter([T.StructField("changed", T.BooleanType())]))
    else:
        value = {"names": ["changed"], "fields": [], "name": "changed",
                 "dataType": T.BooleanType(), "nullable": True,
                 "metadata": {"changed": True}}[attribute]
        setattr(cls, attribute, property(lambda self: value))
elif mode in ("public_alias", "converter_alias", "captured_alias"):
    replacement = native.BooleanType if argument == "LongType" else native.LongType
    if mode == "public_alias":
        sys.modules.pop("pyspark.sql.pandas.types", None)
        assert "pyspark.sql.pandas.types" not in sys.modules
        setattr(T, argument, replacement)
    else:
        converter = importlib.import_module("pyspark.sql.pandas.types")
        if mode == "captured_alias":
            importer = importlib.import_module("pyspark.sql._arrow_input")
        setattr(converter, argument, replacement)
        if mode == "captured_alias":
            sys.modules.pop("pyspark.sql.pandas.types")
            current = importlib.import_module("pyspark.sql.pandas.types")
            assert current is not converter
            assert importer.to_arrow_schema.__globals__ is vars(converter)
elif mode == "atomic_alias":
    scope, replacement_name = argument.split(":")
    replacement = None if replacement_name == "none" else native.LongType
    if scope == "public_before":
        sys.modules.pop("pyspark.sql._arrow_input", None)
        T.AtomicType = replacement
    elif scope == "public_after":
        importlib.import_module("pyspark.sql._arrow_input")
        T.AtomicType = replacement
    else:
        importer = importlib.import_module("pyspark.sql._arrow_input")
        if replacement_name == "missing":
            del importer.AtomicType
        else:
            importer.AtomicType = replacement
elif mode == "importer_alias":
    importer = importlib.import_module("pyspark.sql._arrow_input")
    importer.StructType = None if argument == "none" else native.LongType
else:
    raise AssertionError(mode)
class Factory:
    def mktemp(self, name):
        return Path(tempfile.mkdtemp(prefix=name))
with capture._arrow_input_service(capture.protocol.__wrapped__(Factory())) as (spark, state):
    spark.createDataFrame(frame, schema).toArrow()
    assert state["config_requests"] == [], state["config_requests"]
    table = state["input_tables"][0]
    assert table.schema.names == ["renamed"], table.schema
    assert table.column(0).type == __import__("pyarrow").int64()
    assert table.column(0).to_pylist() == values
'''
    result = subprocess.run([sys.executable, "-c", script, mode, argument,
                             str(before_import), __file__],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("class_name,attribute", [
    ("StructType", "fields"), ("StructType", "names"), ("StructType", "__iter__"),
    ("StructField", "name"), ("StructField", "dataType"), ("StructField", "nullable"),
    ("StructField", "metadata"), ("StructType", "__getattribute__"),
    ("StructField", "__getattribute__"), ("LongType", "__getattribute__"),
    ("StructType", "__class__"), ("StructField", "__class__"), ("LongType", "__class__"),
    ("StructType", "alias"), ("StructField", "alias"), ("LongType", "alias"),
])
@pytest.mark.parametrize("before_import", [False, True])
def test_integer_custom_schema_descriptors_keep_native_row_path(class_name, attribute, before_import):
    _integer_guard_process("descriptor", f"{class_name}:{attribute}", before_import)


@pytest.mark.parametrize("class_name", ["BooleanType", "ByteType", "ShortType", "IntegerType", "LongType"])
@pytest.mark.parametrize("mode", ["public_alias", "converter_alias", "captured_alias"])
def test_integer_rebound_converter_aliases_keep_native_row_path(class_name, mode):
    _integer_guard_process(mode, class_name)


@pytest.mark.parametrize("replacement", ["none", "long"])
def test_integer_rebound_importer_schema_alias_keeps_native_row_path(replacement):
    _integer_guard_process("importer_alias", replacement)


@pytest.mark.parametrize("scope", ["public_before", "public_after", "importer"])
@pytest.mark.parametrize("replacement", ["none", "long"])
def test_integer_rebound_atomic_alias_keeps_native_row_path(scope, replacement):
    _integer_guard_process("atomic_alias", f"{scope}:{replacement}")


def test_integer_deleted_importer_atomic_alias_keeps_native_row_path():
    _integer_guard_process("atomic_alias", "importer:missing")


def test_integer_warm_importer_without_global_converter_aliases(protocol, monkeypatch):
    import pyspark.sql._arrow_input as importer

    original = importer.prepare_table

    def local_imports(table, schema, *configs):
        # Newer importers import converters within prepare_table. Exercise that
        # module shape even on the pinned PR118 dependency used by this branch.
        from pyspark.sql.pandas.types import from_arrow_schema, to_arrow_schema
        table, resolved, names = original(table, schema, *configs)
        assert from_arrow_schema(table.schema).fields
        assert to_arrow_schema(resolved).names == table.column_names
        return table, resolved, names

    frame = pd.DataFrame({"a": np.arange(1000, dtype="int64")})
    schema = T.StructType([T.StructField("renamed", T.LongType())])
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(pa.table({"warm": [1, 2]})).toArrow()
        # Keep original's module globals available in the dependency importer;
        # the full-stack validation separately uses the actual local-import helper.
        import types
        substitute = types.ModuleType(importer.__name__)
        substitute.StructType = T.StructType
        substitute.AtomicType = T.AtomicType
        substitute.prepare_table = local_imports
        monkeypatch.setitem(__import__("sys").modules, importer.__name__, substitute)

        def no_rows(*args, **kwargs):
            raise AssertionError("warm integer input must stay columnar")

        monkeypatch.setattr(pd.DataFrame, "astype", no_rows)
        for _ in range(3):
            spark.createDataFrame(frame, schema).toArrow()
        for table in state["input_tables"][1:]:
            assert table.column_names == ["renamed"]
            assert table.column(0).to_pylist() == list(range(1000))







@pytest.mark.parametrize("dtype", ["int64", "Int64"])
@pytest.mark.parametrize("schema_kind", ["inferred", "names", "explicit"])
def test_integer_rebound_index_data_keeps_native_read_order(
    protocol, monkeypatch, dtype, schema_kind
):
    import json

    values = [2**53 + 1, -(2**63), 2**63 - 1, 0] * 250
    if dtype == "Int64":
        values[1] = None
    frame = pd.DataFrame({"original": pd.Series(values, dtype=dtype)})
    schema, expected_name, nullable, metadata = None, "original", True, {}
    if schema_kind == "names":
        schema, expected_name = ["renamed"], "renamed"
    elif schema_kind == "explicit":
        expected_name, nullable, metadata = "renamed", dtype == "Int64", {"source": "native"}
        schema = T.StructType([T.StructField(expected_name, T.LongType(), nullable, metadata)])
    columns, backing = frame.columns, vars(frame.columns)["_data"]
    original_notna, original_astype = frame.notna, frame.astype
    calls = []
    normalizing = False
    allowed_reads = None

    class ColumnsDataError(RuntimeError):
        pass

    def notna(*args, **kwargs):
        nonlocal normalizing
        normalizing = True
        calls.append("notna")
        return original_notna(*args, **kwargs)

    def astype(*args, **kwargs):
        calls.append("astype")
        return original_astype(*args, **kwargs)

    def data(index):
        if index is columns:
            if not normalizing:
                calls.append("_data")
                if allowed_reads is not None and calls.count("_data") > allowed_reads:
                    raise ColumnsDataError("eligibility added a columns storage read")
                # Install spies only when the native columns storage hook runs.
                # Canonical frame methods remain intact before eligibility.
                monkeypatch.setitem(vars(frame), "notna", notna)
                monkeypatch.setitem(vars(frame), "astype", astype)
            return backing
        return vars(index)["_data"]

    def set_data(index, value):
        # Preserve ordinary storage initialization for indexes pandas creates
        # while normalizing values; only reads of the input columns are hooked.
        vars(index)["_data"] = value

    monkeypatch.setattr(pd.Index, "_data", property(data, set_data), raising=False)
    import pyspark.sql._pandas_input as helper

    with _arrow_input_service(protocol) as (spark, state):
        # Record current master's fallback rather than assuming the old tolist
        # path's single read. Index iteration reads NumPy-backed _values three
        # times, and extension-backed _values twice, before normalization.
        with monkeypatch.context() as baseline:
            baseline.setattr(helper, "prepare_int64_table", lambda *_: None)
            spark.createDataFrame(frame, schema).toArrow()
        baseline_calls = calls.copy()
        allowed_reads = baseline_calls.count("_data")
        assert allowed_reads > 0 and "notna" in baseline_calls
        calls.clear()
        normalizing = False
        # Restore canonical methods before eligibility, just as for the first
        # call. The storage hook installs them when native iteration starts.
        monkeypatch.delitem(vars(frame), "notna")
        monkeypatch.delitem(vars(frame), "astype")
        spark.createDataFrame(frame, schema).toArrow()
        # The rebound storage declines the Arrow path without reading the columns more
        # than the row path does. pandas caches some of these reads, so a later row
        # conversion of the same frame can read them fewer times than the first.
        assert "notna" in calls
        assert calls.count("_data") <= allowed_reads
        assert not state["config_requests"]
        assert len(state["input_tables"]) == 2
        table = state["input_tables"][1]
        assert table.equals(state["input_tables"][0], check_metadata=True)
        assert table.column(0).to_pylist() == values
        assert table.column(0).null_count == sum(value is None for value in values)
        assert table.schema.equals(
            pa.schema([pa.field(expected_name, pa.int64(), nullable=nullable)]),
            check_metadata=True,
        )
        assert json.loads(state["relations"][0].local_relation.schema) == {
            "type": "struct", "fields": [
                {"name": expected_name, "type": "long", "nullable": nullable,
                 "metadata": metadata}
            ],
        }


@pytest.mark.parametrize("zone", ["UTC", "America/Los_Angeles", "Asia/Kathmandu"])
@pytest.mark.parametrize("unit", ["ns", "us"])
def test_pandas_timestamp_ipc_preserves_every_microsecond(protocol, monkeypatch, zone, unit):
    import json
    from pyspark.sql import types as T

    monkeypatch.setenv("TZ", "America/New_York")
    # Include both sides of DST transitions, both fall-back folds, pre-epoch,
    # ns-range boundaries, and NaT. Inspect the full emitted IPC, not a plan flag.
    micros = [-9223372036854775, -1, 0, 1, 1583661599999999, 1583661600000000,
              1604219400000000, 1604223000000000, 9223372036854775, None]
    values = pd.to_datetime(pd.Series(micros, dtype="Int64"), unit="us", utc=True).dt.tz_convert("America/Los_Angeles")
    frame = pd.DataFrame({"a": pd.Series(values).dt.as_unit(unit),
                          "b": pd.Series(values).dt.as_unit(unit)})
    schema = T.StructType([T.StructField("renamed_a", T.TimestampType(), True, {"source": "test"}),
                           T.StructField("b", T.TimestampType())])

    with _arrow_input_service(protocol, capture_tables=True, **{"spark.sql.session.timeZone": zone}) as (spark, state):
        result = spark.createDataFrame(frame, schema)
        # Aware columns carry their zone: only the importer's settings are read.
        assert len(state["config_requests"]) == 1
        assert not result.isLocal()
        result.toArrow()
        assert state["column_names"] == schema.names
        table = pa.concat_tables(state["tables"])
        for column in table.columns:
            assert column.type == pa.timestamp("us", tz="UTC")
            assert column.cast(pa.int64()).to_pylist() == micros
        assert table.column_names == ["renamed_a", "b"]
        relation = state["relations"][0].local_relation
        actual = json.loads(relation.schema)
        assert [field["type"] for field in actual["fields"]] == ["timestamp", "timestamp"]
        assert actual["fields"][0]["metadata"] == {"source": "test"}


def test_pandas_naive_timestamp_utc_ipc(protocol, monkeypatch):
    from pyspark.sql import types as T
    import time
    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.setattr(time, "tzname", ("UTC", "UTC"))
    frame = pd.DataFrame({"a": pd.to_datetime(["1969-12-31 23:59:59.999999", None,
                                               "2020-01-01 00:00:00.123456"])})
    schema = T.StructType([T.StructField("a", T.TimestampType())])
    with _arrow_input_service(protocol, capture_tables=True) as (spark, state):
        result = spark.createDataFrame(frame, schema)
        # A naive column reads the session time zone, then the importer its settings.
        assert state["config_requests"][0] == ("get", ["spark.sql.session.timeZone"])
        assert len(state["config_requests"]) == 2
        assert not result.isLocal()
        result.toArrow()
        assert state["column_names"] == schema.names
        assert state["tables"][0].column(0).cast(pa.int64()).to_pylist() == [
            -1, None, 1577836800123456]


@pytest.mark.parametrize("dtype", ["datetime64[ns]", "datetime64[us]", "datetime64[ms]",
                                    "datetime64[s]", "object"])
def test_pandas_timestamp_eligibility(monkeypatch, dtype):
    import time
    from pyspark.sql import types as T
    from pyspark.sql._pandas_input import timestamp_columns

    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.setattr(time, "tzname", ("UTC", "UTC"))
    frame = pd.DataFrame({"a": pd.Series([datetime(2020, 1, 1), None], dtype=dtype)})
    schema = T.StructType([T.StructField("a", T.TimestampType())])
    assert (timestamp_columns(frame, schema) is not None) == (dtype in (
        "datetime64[ns]", "datetime64[us]"))
    for unsupported in (None, ["a"], "a timestamp",
                        T.StructType([T.StructField("a", T.TimestampNTZType())]),
                        T.StructType([T.StructField("a", T.LongType())])):
        assert timestamp_columns(frame, unsupported) is None


def test_pandas_timestamp_precision_and_timezone_fallback(monkeypatch):
    import time
    from pyspark.sql import types as T
    from pyspark.sql._pandas_input import timestamp_columns, prepare_timestamp_table

    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.setattr(time, "tzname", ("UTC", "UTC"))
    schema = T.StructType([T.StructField("a", T.TimestampType())])
    for nanos in (-1, 1, 1001):
        frame = pd.DataFrame({"a": pd.to_datetime([nanos, None], unit="ns")})
        assert timestamp_columns(frame, schema) is None
    frame = pd.DataFrame({"a": pd.to_datetime([0, None], unit="ns")})
    columns = timestamp_columns(frame, schema)
    assert columns is not None
    assert prepare_timestamp_table(columns, schema, "America/Los_Angeles") is None
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    assert prepare_timestamp_table(columns, schema, "UTC") is None
    monkeypatch.delenv("TZ")
    assert prepare_timestamp_table(columns, schema, "UTC") is None


@pytest.mark.parametrize("unit,mixed", [("ns", False), ("us", True)])
@pytest.mark.parametrize("host_tz,host_names", [
    (None, ("UTC", "UTC")),
    ("America/Los_Angeles", ("PST", "PDT")),
    ("UTC", ("PST", "PDT")),
], ids=["unset-tz", "non-utc-tz", "mismatched-host-zone"])
def test_timestamp_non_utc_host_declines_and_converts_as_pyspark(
    protocol, monkeypatch, unit, mixed, host_tz, host_names
):
    # A host zone other than a literal UTC declines the timestamp fast path before its
    # session time zone RPC; the frame then converts as PySpark's Arrow conversion does,
    # which reads the configs first and takes naive values in the session time zone (UTC).
    import json
    import time

    if host_tz is None:
        monkeypatch.delenv("TZ", raising=False)
    else:
        monkeypatch.setenv("TZ", host_tz)
    monkeypatch.setattr(time, "tzname", host_names)
    micros = [-2000000, 0, 123456, None]
    columns = {}
    if mixed:
        columns["aware"] = pd.to_datetime(micros, unit="us", utc=True).as_unit(unit)
    columns["naive"] = pd.to_datetime(micros, unit="us").as_unit(unit)
    frame = pd.DataFrame(columns)
    schema = T.StructType([
        T.StructField(f"renamed_{name}", T.TimestampType(), True, {"source": name})
        for name in columns
    ])
    from pyspark.sql._pandas_input import timestamp_columns

    assert timestamp_columns(frame, schema) is None
    expected = [micros for _ in columns]
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert state["config_requests"]
        assert state["uploads"] == 0
        assert state["column_names"] == schema.names
        assert len(state["input_tables"]) == 1
        table = state["input_tables"][0]
        assert table.num_rows == len(micros)
        for column, values in zip(table.columns, expected):
            assert column.type == pa.timestamp("us", tz="UTC")
            assert column.cast(pa.int64()).to_pylist() == values
            assert column.is_null().to_pylist() == [value is None for value in values]


@pytest.mark.parametrize("has_null", [False, True])
def test_pandas_timestamp_non_nullable_schema(protocol, has_null):
    from pyspark.sql import types as T
    values = [pd.Timestamp("2020-01-01", tz="UTC"),
              pd.NaT if has_null else pd.Timestamp("2020-01-02", tz="UTC")]
    frame = pd.DataFrame({"a": values})
    schema = T.StructType([T.StructField("a", T.TimestampType(), False)])
    with _arrow_input_service(protocol, capture_tables=True) as (spark, state):
        if has_null:
            # Official: the cast of the Arrow table, a ValueError rather than a row error.
            with pytest.raises(ValueError) as error:
                spark.createDataFrame(frame, schema)
            assert str(error.value) == "Casting field 'a' with null values to non-nullable"
            assert not state["relations"]
        else:
            spark.createDataFrame(frame, schema).toArrow()
            assert not state["tables"][0].schema.field(0).nullable


def test_pandas_timestamp_mixed_and_arrow_dtypes_fallback():
    from pyspark.sql import types as T
    from pyspark.sql._pandas_input import timestamp_columns
    schema = T.StructType([T.StructField("a", T.TimestampType())])
    frame = pd.DataFrame({"a": pd.Series([datetime(2020, 1, 1)],
                                         dtype=pd.ArrowDtype(pa.timestamp("us")))})
    assert timestamp_columns(frame, schema) is None
    frame = pd.DataFrame({"a": pd.to_datetime([0], unit="us"), "b": [1]})
    schema.add("b", T.LongType())
    assert timestamp_columns(frame, schema) is None


@pytest.mark.parametrize("config_error", [False, True])
@pytest.mark.parametrize("verify", [False, True])
def test_timestamp_nonnullable_nat_reads_config_before_cast_error(protocol, config_error, verify):
    from pyspark.sql import types as T
    from pyspark.sql._pandas_input import timestamp_columns

    frame = pd.DataFrame({"a": pd.to_datetime([0, None], utc=True)})
    schema = T.StructType([T.StructField("renamed", T.TimestampType(), False)])
    assert timestamp_columns(frame, schema) is None
    with _arrow_input_service(protocol, capture_tables=True) as (spark, state):
        if config_error:
            state["config_error"] = grpc.StatusCode.INVALID_ARGUMENT
            with pytest.raises(Exception, match="configuration read failed"):
                spark.createDataFrame(frame, schema, verifySchema=verify)
        else:
            # Official: ValueError from the cast of the Arrow table to the schema.
            with pytest.raises(ValueError) as error:
                spark.createDataFrame(frame, schema, verifySchema=verify)
            assert str(error.value) == (
                "Casting field 'renamed' with null values to non-nullable")
        assert state["config_requests"][0] == ("get", _NORMALIZATION_KEYS + _UPLOAD_KEYS)
        assert state["relations"] == []
        assert state["uploads"] == 0


@pytest.mark.parametrize("unit,aware", [("ns", True), ("us", False)])
@pytest.mark.parametrize("location,method", [
    ("instance", "notna"), ("instance", "astype"), ("instance", "where"),
    ("class", "notna"), ("class", "astype"), ("class", "where"),
])
@pytest.mark.parametrize("behavior", ["mutate", "raise"])
def test_timestamp_custom_normalization_hooks_keep_native_row_path(
    protocol, monkeypatch, unit, aware, location, method, behavior
):
    import json
    import time
    from types import MethodType
    import pyspark.sql._pandas_input as helper

    monkeypatch.setenv("TZ", "UTC")
    monkeypatch.setattr(time, "tzname", ("UTC", "UTC"))
    values = pd.to_datetime([-2000000, -1000000, 0, 1000000, None], unit="us", utc=aware)
    frame = pd.DataFrame({"original": values.as_unit(unit)})
    schema = T.StructType([
        T.StructField("renamed", T.TimestampType(), True, {"source": "normalization"})
    ])
    original = getattr(pd.DataFrame, method)
    calls, tables = [], []
    # astype returns a new frame, so the row path never calls an instance where.
    invokes_hook = (location, method) != ("instance", "where")
    expected = [None if pd.isna(value) else round(value.to_pydatetime().timestamp() * 1000000)
                for value in frame["original"]]
    if invokes_hook:
        expected[0] = None if method == "notna" else 7000000
    message = f"custom timestamp {method} precedes configuration"

    def customized(self, *args, **kwargs):
        calls.append(method)
        if behavior == "raise":
            raise RuntimeError(message)
        result = original(self, *args, **kwargs)
        result.iat[0, 0] = (False if method == "notna"
                            else datetime(1970, 1, 1, 0, 0, 7, tzinfo=timezone.utc))
        return result

    if location == "class":
        monkeypatch.setattr(pd.DataFrame, method, customized)
    else:
        monkeypatch.setitem(vars(frame), method, MethodType(customized, frame))

    # Compare with native row conversion, then restore ordinary dispatch. A failing
    # configuration service makes any premature Arrow configuration read observable.
    for bypass in (True, False):
        calls.clear()
        with monkeypatch.context() as dispatch:
            if bypass:
                dispatch.setattr(helper, "timestamp_columns", lambda *_: None)
            with _arrow_input_service(protocol, config_error=True) as (spark, state):
                if behavior == "raise" and invokes_hook:
                    with pytest.raises(RuntimeError, match=message):
                        spark.createDataFrame(frame, schema)
                    assert state["relations"] == []
                    assert state["input_tables"] == []
                else:
                    spark.createDataFrame(frame, schema).toArrow()
                    assert len(state["input_tables"]) == 1
                    table = state["input_tables"][0]
                    assert state["column_names"] == schema.names
                    assert table.column(0).cast(pa.int64()).to_pylist() == expected
                    assert table.column(0).is_null().to_pylist() == [
                        value is None for value in expected]
                    assert json.loads(state["relations"][0].local_relation.schema) == json.loads(schema.json())
                    tables.append(table)
                assert calls == ([method] if invokes_hook else [])
                assert state["config_requests"] == []
                assert state["uploads"] == 0
    if behavior == "mutate" or not invokes_hook:
        assert tables[0].equals(tables[1], check_metadata=True)


@pytest.mark.parametrize("custom", ["dtype", "array"])
def test_timestamp_custom_datetime_array_keeps_row_values(protocol, custom):
    from pandas.core.arrays.datetimes import DatetimeArray
    from pandas.core.dtypes.dtypes import DatetimeTZDtype
    from pyspark.sql import types as T
    from pyspark.sql._pandas_input import timestamp_columns

    arrow_calls = []

    class CustomDtype(DatetimeTZDtype):
        pass

    class CustomArray(DatetimeArray):
        @property
        def dtype(self):
            return CustomDtype(unit="ns", tz="UTC") if custom == "dtype" else super().dtype

        def __arrow_array__(self, type=None):
            arrow_calls.append("arrow")
            return pa.array([-7000000] * len(self), type=pa.int64()).cast(type)

    ordinary = pd.to_datetime([0, 1000000, None, -1000000], unit="ns", utc=True).array
    array = CustomArray._simple_new(ordinary._ndarray, dtype=ordinary.dtype)
    frame = pd.DataFrame({"a": pd.Series(array, copy=False)})
    assert type(frame["a"].array) is CustomArray
    schema = T.StructType([T.StructField("a", T.TimestampType(), True, {"source": "custom"})])
    assert timestamp_columns(frame, schema) is None
    with _arrow_input_service(protocol, capture_tables=True) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert state["config_requests"] == []
        table = pa.concat_tables(state["tables"])
        assert table.column(0).cast(pa.int64()).to_pylist() == [0, 1000, None, -1000]
        logical = __import__("json").loads(state["relations"][0].local_relation.schema)
        assert logical["fields"][0]["metadata"] == {"source": "custom"}
    assert arrow_calls == []


def test_timestamp_rebound_dataframe_and_duplicate_names_keep_fallback(monkeypatch):
    from pyspark.sql import types as T
    from pyspark.sql._pandas_input import timestamp_columns

    class CustomFrame(pd.DataFrame):
        pass

    frame = CustomFrame({"a": pd.to_datetime([0], utc=True)})
    schema = T.StructType([T.StructField("a", T.TimestampType())])
    monkeypatch.setattr(pd, "DataFrame", CustomFrame)
    assert timestamp_columns(frame, schema) is None
    from pandas.core.frame import DataFrame
    frame = DataFrame({"a": pd.to_datetime([0], utc=True), "b": pd.to_datetime([1], utc=True)})
    monkeypatch.setattr(pd, "DataFrame", DataFrame)
    schema.add("a", T.TimestampType())
    assert timestamp_columns(frame, schema) is None


def _timestamp_guard_process(mode, argument, before_import=False):
    import subprocess
    import sys

    if mode == "captured_alias":
        import pyspark.sql._arrow_input as importer
        if not hasattr(importer, "to_arrow_schema"):
            pytest.skip("this importer resolves converters locally")
    script = r'''
import importlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
import pandas as pd
from pyspark.sql import types as T
from pyspark import _pyspark as native
spec = importlib.util.spec_from_file_location("timestamp_guard_capture", sys.argv[4])
capture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capture
spec.loader.exec_module(capture)
frame = pd.DataFrame({"a": pd.to_datetime([0, 1000000, None, -1000000], unit="ns", utc=True)})
schema = T.StructType([T.StructField("renamed", T.TimestampType(), True, {"source": "native"})])
mode, argument, before_import = sys.argv[1:4]
if before_import == "True":
    sys.modules.pop("pyspark.sql._pandas_input", None)
if mode == "descriptor":
    class_name, attribute = argument.split(":")
    cls = getattr(native, class_name)
    if attribute == "alias":
        setattr(T, class_name, object)
    elif attribute == "__getattribute__":
        setattr(cls, attribute, lambda self, name: object.__getattribute__(self, name))
    elif attribute == "__class__":
        setattr(cls, "__getattribute__", lambda self, name: (
            object if name == "__class__" else object.__getattribute__(self, name)))
    elif attribute == "__iter__":
        setattr(cls, attribute, lambda self: iter([T.StructField("changed", T.BooleanType())]))
    else:
        value = {"names": ["changed"], "fields": [], "name": "changed",
                 "dataType": T.BooleanType(), "nullable": False,
                 "metadata": {"changed": True}}[attribute]
        setattr(cls, attribute, property(lambda self: value))
elif mode in ("public_alias", "converter_alias", "captured_alias"):
    replacement = native.BooleanType if argument == "TimestampType" else native.TimestampType
    if mode == "public_alias":
        sys.modules.pop("pyspark.sql.pandas.types", None)
        setattr(T, argument, replacement)
    else:
        converter = importlib.import_module("pyspark.sql.pandas.types")
        if mode == "captured_alias":
            importer = importlib.import_module("pyspark.sql._arrow_input")
        setattr(converter, argument, replacement)
        if mode == "captured_alias":
            sys.modules.pop("pyspark.sql.pandas.types")
            assert importlib.import_module("pyspark.sql.pandas.types") is not converter
elif mode == "atomic_alias":
    scope, replacement_name = argument.split(":")
    replacement = None if replacement_name == "none" else native.LongType
    if scope == "public_before":
        sys.modules.pop("pyspark.sql._arrow_input", None)
        T.AtomicType = replacement
    elif scope == "public_after":
        importlib.import_module("pyspark.sql._arrow_input")
        T.AtomicType = replacement
    else:
        importer = importlib.import_module("pyspark.sql._arrow_input")
        if replacement_name == "missing":
            del importer.AtomicType
        else:
            importer.AtomicType = replacement
elif mode == "importer_alias":
    import pyspark.sql._arrow_input as importer
    importer.StructType = None
else:
    raise AssertionError(mode)
class Factory:
    def mktemp(self, name):
        return Path(tempfile.mkdtemp(prefix=name))
with capture._arrow_input_service(capture.protocol.__wrapped__(Factory()), capture_tables=True) as (spark, state):
    spark.createDataFrame(frame, schema).toArrow()
    assert state["config_requests"] == [], state["config_requests"]
    table = capture.pa.concat_tables(state["tables"])
    assert table.column_names == ["renamed"]
    assert table.column(0).cast(capture.pa.int64()).to_pylist() == [0, 1000, None, -1000]
    logical = json.loads(state["relations"][0].local_relation.schema)
    assert logical["fields"][0]["metadata"] == {"source": "native"}
'''
    result = subprocess.run([sys.executable, "-c", script, mode, argument,
                             str(before_import), __file__],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("class_name,attribute", [
    ("StructType", "fields"), ("StructType", "names"), ("StructType", "__iter__"),
    ("StructField", "name"), ("StructField", "dataType"), ("StructField", "nullable"),
    ("StructField", "metadata"), ("StructType", "__getattribute__"),
    ("StructField", "__getattribute__"), ("TimestampType", "__getattribute__"),
    ("StructType", "__class__"), ("StructField", "__class__"), ("TimestampType", "__class__"),
    ("StructType", "alias"), ("StructField", "alias"), ("TimestampType", "alias"),
])
@pytest.mark.parametrize("before_import", [False, True])
def test_timestamp_custom_schema_descriptors_keep_row_path(class_name, attribute, before_import):
    _timestamp_guard_process("descriptor", f"{class_name}:{attribute}", before_import)


@pytest.mark.parametrize("class_name", ["BooleanType", "ByteType", "ShortType", "IntegerType",
                                        "LongType", "FloatType", "DoubleType", "DecimalType",
                                        "StringType", "BinaryType", "DateType", "TimestampType"])
@pytest.mark.parametrize("mode", ["public_alias", "converter_alias", "captured_alias"])
def test_timestamp_rebound_converter_aliases_keep_row_path(class_name, mode):
    _timestamp_guard_process(mode, class_name)


def test_timestamp_rebound_importer_schema_alias_keeps_row_path():
    _timestamp_guard_process("importer_alias", "none")


@pytest.mark.parametrize("scope", ["public_before", "public_after", "importer"])
@pytest.mark.parametrize("replacement", ["none", "long"])
def test_timestamp_rebound_atomic_alias_keeps_row_path(scope, replacement):
    _timestamp_guard_process("atomic_alias", f"{scope}:{replacement}")


def test_timestamp_deleted_importer_atomic_alias_keeps_row_path():
    _timestamp_guard_process("atomic_alias", "importer:missing")


def test_timestamp_warm_local_converter_imports_keep_columnar_route(protocol, monkeypatch):
    import types
    import sys
    import pyspark.sql._arrow_input as importer
    from pyspark.sql import types as T
    from pyspark.sql._pandas_input import timestamp_columns

    original = importer.prepare_table
    substitute = types.ModuleType(importer.__name__)
    substitute.StructType = T.StructType
    substitute.AtomicType = T.AtomicType

    def prepare_table(table, schema, *configs):
        from pyspark.sql.pandas.types import to_arrow_schema
        assert to_arrow_schema(schema, timezone="UTC").names == table.column_names
        return original(table, schema, *configs)

    substitute.prepare_table = prepare_table
    frame = pd.DataFrame({"a": pd.to_datetime([0, 1000000, None], unit="ns", utc=True)})
    schema = T.StructType([T.StructField("renamed", T.TimestampType())])
    with _arrow_input_service(protocol, capture_tables=True) as (spark, state):
        spark.createDataFrame(pa.table({"warm": [1]})).toArrow()
        monkeypatch.setitem(sys.modules, importer.__name__, substitute)
        assert timestamp_columns(frame, schema) is not None
        spark.createDataFrame(frame, schema).toArrow()
        assert len(state["config_requests"]) == 2
        table = state["tables"][-1]
        assert table.column_names == ["renamed"]
        assert table.column(0).cast(pa.int64()).to_pylist() == [0, 1000, None]


def _timestamp_columns_fixture(unit):
    values = [-2000000, -1000000, 0, 1000000, None] * 2000
    frame = pd.DataFrame({
        "original": pd.to_datetime(pd.Series(values, dtype="Int64"), unit="us", utc=True).dt.as_unit(unit)
    })
    schema = T.StructType([
        T.StructField("renamed", T.TimestampType(), True, {"source": "columns", "id": 3})
    ])
    return values, frame, schema


def _assert_timestamp_columns_row_input(state, values, schema):
    import json

    assert not state["config_requests"]
    assert not state["uploads"]
    assert len(state["input_tables"]) == 1
    assert state["column_names"] == schema.names
    table = state["input_tables"][0]
    assert table.num_rows == 10000
    assert table.schema.equals(
        pa.schema([pa.field("renamed", pa.timestamp("us", tz="UTC"), nullable=True)]),
        check_metadata=True,
    )
    assert table.column(0).cast(pa.int64()).to_pylist() == values
    assert table.column(0).is_null().to_pylist() == [value is None for value in values]
    assert json.loads(state["relations"][0].local_relation.schema) == json.loads(schema.json())


@pytest.mark.parametrize("unit", ["ns", "us"])
@pytest.mark.parametrize("location", [
    "subclass", "class", "owner", "backing_subclass_item", "backing_class_iter",
    "backing_class_getitem", "backing_instance_box",
])
@pytest.mark.parametrize("behavior", ["raise", "mutate"])
def test_timestamp_columns_iterator_hooks_keep_native_order(
    protocol, monkeypatch, unit, location, behavior
):
    from pandas.core.base import IndexOpsMixin

    values, frame, schema = _timestamp_columns_fixture(unit)
    original_iter = pd.Index.__iter__
    original_notna, original_astype = frame.notna, frame.astype
    calls = []
    invoked = False

    class ColumnsError(RuntimeError):
        pass

    def notna(*args, **kwargs):
        calls.append("notna")
        return original_notna(*args, **kwargs)

    def astype(*args, **kwargs):
        calls.append("astype")
        return original_astype(*args, **kwargs)

    def before_hook():
        nonlocal invoked
        if invoked:
            return
        invoked = True
        calls.append("columns iterator")
        if behavior == "raise":
            raise ColumnsError("timestamp columns hook precedes pandas normalization")
        # Install spies only when native iteration reaches the columns hook.
        # Before eligibility, normalization methods are the canonical methods.
        monkeypatch.setitem(vars(frame), "notna", notna)
        monkeypatch.setitem(vars(frame), "astype", astype)
        frame.iat[0, 0] = pd.Timestamp(7000000, unit="us", tz="UTC")
        frame.iat[-1, 0] = pd.NaT

    def iterate(index):
        if index is columns:
            before_hook()
        return original_iter(index)

    if location == "subclass":
        class Columns(pd.Index):
            pass
        monkeypatch.setattr(Columns, "__iter__", iterate)
        labels = pd.Index(["original"], dtype=object)
        frame.columns = Columns._simple_new(labels._values, name=None)
        assert type(frame.columns) is Columns
    elif location == "backing_subclass_item":
        # IndexOpsMixin iteration calls ndarray.item, rather than ndarray.__iter__.
        class Names(np.ndarray):
            def item(self, *args):
                before_hook()
                return super().item(*args)
        backing = np.asarray(["original"], dtype=object).view(Names)
        frame.columns = pd.Index._simple_new(backing, name=None)
        assert type(frame.columns._values) is Names
    elif location.startswith("backing_"):
        frame.columns = pd.Index(pd.array(["original"], dtype=pd.StringDtype(storage="python")))
        backing = frame.columns._values
        if location == "backing_class_iter":
            original = type(backing).__iter__

            def backing_iter(array):
                if array is backing:
                    before_hook()
                return original(array)

            monkeypatch.setattr(type(backing), "__iter__", backing_iter)
        elif location == "backing_class_getitem":
            original = type(backing).__getitem__

            def backing_getitem(array, key):
                if array is backing:
                    before_hook()
                return original(array, key)

            monkeypatch.setattr(type(backing), "__getitem__", backing_getitem)
        else:
            original = backing._box_func

            def box(value):
                before_hook()
                return original(value)

            monkeypatch.setitem(vars(backing), "_box_func", box)
    else:
        monkeypatch.setattr(pd.Index if location == "class" else IndexOpsMixin,
                            "__iter__", iterate)
    columns = frame.columns

    with _arrow_input_service(protocol) as (spark, state):
        if behavior == "raise":
            with pytest.raises(ColumnsError,
                               match="timestamp columns hook precedes pandas normalization"):
                spark.createDataFrame(frame, schema)
            assert calls == ["columns iterator"]
            assert not state["config_requests"]
            assert not state["uploads"]
            assert not state["relations"]
            assert not state["input_tables"]
        else:
            spark.createDataFrame(frame, schema).toArrow()
            expected = values.copy()
            expected[0], expected[-1] = 7000000, None
            _assert_timestamp_columns_row_input(state, expected, schema)
            assert calls == ["columns iterator", "notna", "astype"]


@pytest.mark.parametrize("unit", ["ns", "us"])
def test_timestamp_rebound_index_values_keeps_native_error(protocol, monkeypatch, unit):
    _, frame, schema = _timestamp_columns_fixture(unit)
    columns, original = frame.columns, pd.Index._values.fget
    calls = []

    def values(index):
        if index is columns:
            calls.append("_values")
            raise RuntimeError("columns backing hook precedes timestamp normalization")
        return original(index)

    monkeypatch.setattr(pd.Index, "_values", property(values))
    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(RuntimeError, match="columns backing hook precedes timestamp normalization"):
            spark.createDataFrame(frame, schema)
        assert calls == ["_values"]
        assert not state["config_requests"]
        assert not state["uploads"]
        assert not state["relations"]


@pytest.mark.parametrize("unit", ["ns", "us"])
def test_timestamp_rebound_index_data_keeps_native_read_order(protocol, monkeypatch, unit):
    values, frame, schema = _timestamp_columns_fixture(unit)
    columns, backing = frame.columns, vars(frame.columns)["_data"]
    original_notna, original_astype = frame.notna, frame.astype
    calls, normalizing = [], False
    allowed_reads = None

    class ColumnsDataError(RuntimeError):
        pass

    def notna(*args, **kwargs):
        nonlocal normalizing
        normalizing = True
        calls.append("notna")
        return original_notna(*args, **kwargs)

    def astype(*args, **kwargs):
        calls.append("astype")
        return original_astype(*args, **kwargs)

    def data(index):
        if index is columns:
            if not normalizing:
                calls.append("_data")
                if allowed_reads is not None and calls.count("_data") > allowed_reads:
                    raise ColumnsDataError("eligibility added a columns storage read")
                monkeypatch.setitem(vars(frame), "notna", notna)
                monkeypatch.setitem(vars(frame), "astype", astype)
            return backing
        return vars(index)["_data"]

    def set_data(index, value):
        vars(index)["_data"] = value

    monkeypatch.setattr(pd.Index, "_data", property(data, set_data), raising=False)
    import pyspark.sql._pandas_input as helper

    # Measure current native fallback dispatch without either eligibility probe.
    with monkeypatch.context() as baseline:
        baseline.setattr(helper, "prepare_int64_table", lambda *_: None)
        baseline.setattr(helper, "timestamp_columns", lambda *_: None)
        with _arrow_input_service(protocol) as (spark, state):
            spark.createDataFrame(frame, schema).toArrow()
            _assert_timestamp_columns_row_input(state, values, schema)
            control_table = state["input_tables"][0]
    baseline_calls = calls.copy()
    allowed_reads = baseline_calls.count("_data")
    assert allowed_reads > 0 and "notna" in baseline_calls
    calls.clear()
    normalizing = False
    # Restore ordinary methods before candidate eligibility; the storage hook
    # installs observers only when the native columns iterator starts.
    monkeypatch.delitem(vars(frame), "notna")
    monkeypatch.delitem(vars(frame), "astype")
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        # The rebound storage declines the Arrow path without reading the columns more
        # than the row path does. pandas caches some of these reads, so a later row
        # conversion of the same frame can read them fewer times than the first.
        assert "notna" in calls
        assert calls.count("_data") <= allowed_reads
        _assert_timestamp_columns_row_input(state, values, schema)
        assert state["input_tables"][0].equals(control_table, check_metadata=True)


@pytest.mark.parametrize("unit", ["ns", "us"])
@pytest.mark.parametrize("storage", ["object", "default", "python", "pyarrow", "python_nan", "pyarrow_nan"])
def test_timestamp_canonical_columns_storage_stays_columnar(protocol, unit, storage):
    from pyspark.sql.pandas.types import to_arrow_schema
    import json

    values, frame, schema = _timestamp_columns_fixture(unit)
    if storage == "object":
        frame.columns = pd.Index(["original"], dtype=object)
    elif storage != "default":
        kind = storage.removesuffix("_nan")
        dtype = (pd.StringDtype(storage=kind, na_value=np.nan)
                 if storage.endswith("_nan") else pd.StringDtype(storage=kind))
        frame.columns = pd.Index(pd.array(["original"], dtype=dtype))

    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert state["config_requests"]
        assert len(state["input_tables"]) == 1
        table = state["input_tables"][0]
        assert table.num_rows == 10000
        assert table.column(0).cast(pa.int64()).to_pylist() == values
        assert table.column(0).is_null().to_pylist() == [value is None for value in values]
        assert table.schema.equals(to_arrow_schema(schema, timezone="UTC"), check_metadata=True)
        assert json.loads(state["relations"][0].local_relation.schema) == json.loads(schema.json())


def _assert_float64_ipc(table, columns, schema, *, missing=None):
    assert table.num_rows == len(columns[0])
    assert table.schema.names == schema.names
    assert table.num_columns == len(columns)
    for index, (column, expected, field) in enumerate(zip(table.columns, columns, schema.fields)):
        mask = np.isnan(expected) if missing is None else missing[index]
        assert column.type == pa.float64()
        # Check every IPC validity bit and the exact bits of every non-null value,
        # including negative zero, subnormals, and both infinities.
        assert column.is_null().to_pylist() == mask.tolist()
        actual = column.drop_null().combine_chunks().to_numpy(zero_copy_only=False)
        np.testing.assert_array_equal(actual.view(np.uint64), expected[~mask].view(np.uint64))
        actual_field = table.schema.field(index)
        assert actual_field.nullable == field.nullable
        # The Arrow conversion keeps the field metadata; the fast path's table has none.
        from pyspark.sql.pandas.types import from_arrow_metadata

        if actual_field.metadata is not None:
            assert from_arrow_metadata(actual_field.metadata) == field.metadata



@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("verify", [False, True])
def test_float64_columns_preserve_every_ipc_cell(protocol, monkeypatch, cached, verify):
    finite = np.array([
        0.0, -0.0, 1.5, -2.25, np.finfo(np.float64).max, -np.finfo(np.float64).max,
        np.finfo(np.float64).tiny, -np.finfo(np.float64).tiny,
        np.nextafter(0.0, 1.0), -np.nextafter(0.0, 1.0),
    ], dtype=np.float64)
    first = np.tile(finite, 10000)
    second = np.tile(np.array([np.nan, np.inf, -np.inf, -0.0, 1.5], dtype=np.float64), 20000)
    columns = [first, second, np.full(100000, np.nan, dtype=np.float64)]
    frame = pd.DataFrame(dict(zip(["original", "b", "c"], columns)))
    schema = T.StructType([
        T.StructField("renamed.`double", T.DoubleType(), False, {"source": "float64", "id": 3}),
        T.StructField("nullable", T.DoubleType(), True, {"tag": [1, 2], "label": "é"}),
        T.StructField("all_null", T.DoubleType()),
    ])

    def no_rows(*args, **kwargs):
        raise AssertionError("float64 fast path must not materialize Python rows")

    for method in ("to_numpy", "iterrows", "itertuples"):
        monkeypatch.setattr(pd.DataFrame, method, no_rows)
    monkeypatch.setattr(pd.DataFrame, "values", property(no_rows))
    monkeypatch.setattr(pd.Series, "tolist", no_rows)
    overrides = {"spark.sql.session.localRelationCacheThreshold": "1" if cached else "999999999"}
    with _arrow_input_service(protocol, config_error=True, **overrides) as (spark, state):
        result = spark.createDataFrame(frame, schema, verifySchema=verify)
        assert not result.isLocal()
        result.toArrow()
        assert state["column_names"] == schema.names
        _assert_float64_ipc(state["input_tables"][0], columns, schema)
        assert state["uploads"] == 0
        assert state["config_requests"] == []



def test_float64_columnar_ipc_matches_existing_row_conversion(protocol, monkeypatch):
    import json

    import pyspark.sql._pandas_input as helper

    values = np.tile(np.array([np.nan, np.inf, -np.inf, -0.0, 0.0, 1.25]), 20000)
    frame = pd.DataFrame({"original": values})
    schema = T.StructType([T.StructField("renamed", T.DoubleType(), True, {"source": "parity"})])
    with _arrow_input_service(protocol) as (spark, state):
        result = spark.createDataFrame(frame, schema)
        assert not result.isLocal()
        result.toArrow()
        assert state["column_names"] == schema.names
        monkeypatch.setattr(helper, "prepare_float64_table", lambda *args: None)
        result = spark.createDataFrame(frame, schema)
        assert not result.isLocal()
        result.toArrow()
        assert state["column_names"] == schema.names
        assert len(state["input_tables"]) == 2
        for table in state["input_tables"]:
            _assert_float64_ipc(table, [values], schema)
        # The Arrow conversion of the second call carries the field metadata in its table
        # and reads the configuration first, as official; the values and types are the same.
        assert state["input_tables"][0].equals(state["input_tables"][1], check_metadata=False)
        assert state["config_requests"][:1] == [("get", _NORMALIZATION_KEYS + _UPLOAD_KEYS)]
        assert state["uploads"] == 0
        assert [json.loads(relation.local_relation.schema) for relation in state["relations"]] == [
            json.loads(schema.json()), json.loads(schema.json())
        ]



@pytest.mark.parametrize("nullable", [False, True])
def test_float64_pandas_infinity_missing_option_keeps_row_semantics(protocol, nullable):
    from pyspark.errors import PySparkRuntimeError

    try:
        pd.get_option("mode.use_inf_as_na")
    except (KeyError, AttributeError):
        pytest.skip("pandas does not expose mode.use_inf_as_na")
    values = np.tile(np.array([np.inf, -np.inf, np.nan, -0.0, 1.25]), 20000)
    frame = pd.DataFrame({"x": values})
    schema = T.StructType([T.StructField("x", T.DoubleType(), nullable)])
    with pd.option_context("mode.use_inf_as_na", True):
        with _arrow_input_service(protocol) as (spark, state):
            if not nullable:
                # Official: the configuration is read, then the cast fails with a ValueError.
                with pytest.raises(ValueError) as error:
                    spark.createDataFrame(frame, schema)
                assert str(error.value) == "Casting field 'x' with null values to non-nullable"
                assert state["config_requests"][:1] == [
                    ("get", _NORMALIZATION_KEYS + _UPLOAD_KEYS)]
                assert state["uploads"] == 0
                return
            spark.createDataFrame(frame, schema).toArrow()
            _assert_float64_ipc(state["input_tables"][0], [values], schema,
                                missing=[~np.isfinite(values)])



@pytest.mark.parametrize("reason", [
    "small", "empty", "zero_columns", "inferred", "names", "ddl", "float_schema",
    "long_schema", "mixed_schema", "width", "duplicate_input", "duplicate_schema",
    "input_nul", "schema_nul", "input_number", "input_string_subclass", "frame_subclass",
    "schema_proxy", "extension", "object", "float32", "float16", "non_native",
    "integer", "mixed", "nonnullable_nan", "late_unsupported",
])
def test_float64_column_guard_preserves_row_fallback(reason):
    from pyspark.sql._pandas_input import prepare_float64_table

    class Frame(pd.DataFrame):
        pass

    class Name(str):
        pass

    class SchemaProxy:
        @property
        def __class__(self):
            return T.StructType

    values = np.arange(100000, dtype=np.float64)
    frame = pd.DataFrame({"x": values})
    schema = T.StructType([T.StructField("x", T.DoubleType())])
    if reason == "small":
        frame = frame.iloc[:99999]
    elif reason == "empty":
        frame = frame.iloc[:0]
    elif reason == "zero_columns":
        frame = frame.iloc[:, :0]
        schema = T.StructType([])
    elif reason in ("inferred", "names", "ddl", "float_schema", "long_schema"):
        schema = {
            "inferred": None, "names": ["x"], "ddl": "x DOUBLE",
            "float_schema": T.StructType([T.StructField("x", T.FloatType())]),
            "long_schema": T.StructType([T.StructField("x", T.LongType())]),
        }[reason]
    elif reason == "mixed_schema":
        frame["y"] = values
        schema.add("y", T.FloatType())
    elif reason == "width":
        schema.add("extra", T.DoubleType())
    elif reason in ("duplicate_input", "duplicate_schema"):
        frame["y"] = values
        schema.add("y", T.DoubleType())
        if reason == "duplicate_input":
            frame.columns = ["x", "x"]
        else:
            schema = T.StructType([T.StructField("x", T.DoubleType())] * 2)
    elif reason == "input_string_subclass":
        frame.columns = pd.Index([Name("x")], dtype=object)
        assert type(frame.columns[0]) is Name
    elif reason in ("input_nul", "input_number"):
        frame.columns = [{"input_nul": "x\x00", "input_number": 1}[reason]]
    elif reason == "schema_nul":
        schema = T.StructType([T.StructField("x\x00", T.DoubleType())])
    elif reason == "frame_subclass":
        frame = Frame(frame)
    elif reason == "schema_proxy":
        schema = SchemaProxy()
    elif reason == "nonnullable_nan":
        frame.iat[-1, 0] = np.nan
        schema = T.StructType([T.StructField("x", T.DoubleType(), False)])
    elif reason in ("mixed", "late_unsupported"):
        frame["y"] = "text" if reason == "mixed" else values.astype(object)
        schema.add("y", T.StringType() if reason == "mixed" else T.DoubleType())
    else:
        dtype = {"extension": "Float64", "object": object, "float32": "float32",
                 "float16": "float16", "non_native": ">f8", "integer": "int64"}[reason]
        frame = pd.DataFrame({"x": pd.Series(values, dtype=dtype)})
    assert prepare_float64_table(frame, schema) is None



@pytest.mark.parametrize("reason", [
    "small", "inferred", "names", "ddl", "float32", "extension", "object", "frame_subclass",
    "duplicate_input", "duplicate_schema", "input_nul", "schema_nul",
])
def test_float64_public_fallback_preserves_every_ipc_cell(protocol, reason):

    class Frame(pd.DataFrame):
        pass

    values = np.tile(np.array([1.5, -0.0, np.inf, -np.inf, np.nan]), 20000)
    if reason == "small":
        values = values[:99999]
    dtype = {"float32": "float32", "extension": "Float64", "object": object}.get(reason)
    frame_type = Frame if reason == "frame_subclass" else pd.DataFrame
    frame = frame_type({"x": pd.Series(values, dtype=dtype)})
    columns = [values]
    expected_schema = T.StructType([T.StructField("x", T.DoubleType())])
    schema = expected_schema
    if reason in ("inferred", "names", "ddl"):
        schema = {"inferred": None, "names": ["x"], "ddl": "x DOUBLE"}[reason]
    elif reason in ("duplicate_input", "duplicate_schema"):
        frame["y"] = values
        columns.append(values)
        expected_schema = T.StructType([
            T.StructField("x", T.DoubleType()),
            T.StructField("x" if reason == "duplicate_schema" else "y", T.DoubleType()),
        ])
        schema = expected_schema
        if reason == "duplicate_input":
            frame.columns = ["x", "x"]
    elif reason == "input_nul":
        frame.columns = ["x\x00original"]
    elif reason == "schema_nul":
        schema = expected_schema = T.StructType([T.StructField("x\x00renamed", T.DoubleType())])
    if reason in ("inferred", "names"):
        # As PySpark Connect's, the Arrow conversion of a frame without a schema names its
        # columns _0, _1, ..., which toDF names.
        expected_schema = T.StructType([T.StructField("_0", T.DoubleType())])
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        # Official converts pandas input with Arrow, reading the configuration first.
        native = reason in ("frame_subclass", "schema_nul")
        assert state["config_requests"][:1] == (
            [] if native else [("get", _NORMALIZATION_KEYS + _UPLOAD_KEYS)])
        _assert_float64_ipc(state["input_tables"][0], columns, expected_schema)



@pytest.mark.parametrize("verify", [False, True])
def test_float64_nonnullable_nan_reads_config_before_cast_error(protocol, verify):
    frame = pd.DataFrame({"x": np.full(100000, np.nan, dtype=np.float64)})
    schema = T.StructType([T.StructField("x", T.DoubleType(), False)])
    with _arrow_input_service(protocol) as (spark, state):
        state["config_error"] = grpc.StatusCode.INVALID_ARGUMENT
        # Official reads the configuration first, so its error precedes the cast.
        with pytest.raises(Exception, match="configuration read failed"):
            spark.createDataFrame(frame, schema, verifySchema=verify)
        assert state["config_requests"][0] == ("get", _NORMALIZATION_KEYS + _UPLOAD_KEYS)
        assert state["relations"] == []
        assert state["uploads"] == 0
        # Then the cast fails as official does, with a ValueError.
        with pytest.raises(ValueError) as error:
            spark.createDataFrame(frame, schema, verifySchema=verify)
        assert str(error.value) == "Casting field 'x' with null values to non-nullable"
        assert state["relations"] == []



@pytest.mark.parametrize("class_name,attribute", [
    ("StructType", "names"), ("StructType", "fields"), ("StructType", "__iter__"),
    ("StructField", "name"), ("StructField", "dataType"), ("StructField", "nullable"),
    ("StructField", "metadata"),
    ("StructType", "__getattribute__"), ("StructField", "__getattribute__"),
    ("DoubleType", "__getattribute__"), ("StructType", "__class__"),
    ("StructField", "__class__"), ("DoubleType", "__class__"),
    ("StructType", "alias"), ("StructField", "alias"), ("DoubleType", "alias"),
])
@pytest.mark.parametrize("before_import", [False, True])
def test_float64_custom_schema_descriptors_keep_native_row_path(
    protocol, class_name, attribute, before_import
):
    import sys

    script = r'''
import importlib.util
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from pyspark.sql import types as T

spec = importlib.util.spec_from_file_location("float64_schema_capture", sys.argv[4])
capture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capture
spec.loader.exec_module(capture)
_arrow_input_service, protocol = capture._arrow_input_service, capture.protocol
values = np.tile(np.array([1.5, -0.0, np.inf, -np.inf, np.nan]), 20000)
frame = pd.DataFrame({"x": values})
schema = T.StructType([T.StructField("x", T.DoubleType(), True, {"source": "native"})])
class_name, attribute, before_import = sys.argv[1:4]
if before_import == "True":
    sys.modules.pop("pyspark.sql._pandas_input", None)
assert ("pyspark.sql._pandas_input" in sys.modules) == (before_import == "False")
cls = getattr(T, class_name)
if attribute == "alias":
    setattr(T, class_name, object)
else:
    if attribute == "__getattribute__":
        replacement = lambda self, name: object.__getattribute__(self, name)
    elif attribute == "__iter__":
        replacement = lambda self: iter([T.StructField("changed", T.DoubleType())])
    elif attribute == "__class__":
        attribute = "__getattribute__"
        replacement = lambda self, name: object if name == "__class__" else (
            object.__getattribute__(self, name))
    else:
        value = {"names": ["changed"], "fields": [], "name": "changed",
                 "dataType": T.DoubleType(), "nullable": False,
                 "metadata": {"changed": True}}[attribute]
        replacement = property(lambda self: value)
    setattr(cls, attribute, replacement)
class Factory:
    def mktemp(self, name):
        return Path(tempfile.mkdtemp(prefix=name))
with _arrow_input_service(protocol.__wrapped__(Factory())) as (spark, state):
    spark.createDataFrame(frame, schema).toArrow()
    assert state["config_requests"] == [], state["config_requests"]
    table = state["input_tables"][0]
    assert table.schema.names == ["x"], table.schema
    assert table.schema.field(0).nullable is True
    assert table.schema.field(0).type == capture.pa.float64()
    assert capture.T is T
    mask = np.isnan(values)
    assert table.column(0).is_null().to_pylist() == mask.tolist()
    actual = table.column(0).drop_null().combine_chunks().to_numpy(zero_copy_only=False)
    np.testing.assert_array_equal(actual.view(np.uint64), values[~mask].view(np.uint64))
    from pyspark.sql.pandas.types import from_arrow_metadata
    assert from_arrow_metadata(table.schema.field(0).metadata) is None
    import json
    assert json.loads(state["relations"][0].local_relation.schema)["fields"][0]["metadata"] == {
        "source": "native"
    }
'''
    result = subprocess.run([sys.executable, "-c", script, class_name, attribute,
                             str(before_import), __file__],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr



@pytest.mark.parametrize("class_name", [
    "BooleanType", "ByteType", "ShortType", "IntegerType", "LongType", "FloatType",
    "DoubleType", "StructType", "StructField",
])
@pytest.mark.parametrize("converter_loaded", [False, True])
def test_float64_rebound_converter_alias_keeps_native_row_path(
    protocol, class_name, converter_loaded
):
    import sys

    script = r'''
import importlib.util
import sys
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from pyspark.sql import types as T

spec = importlib.util.spec_from_file_location("float64_alias_capture", sys.argv[3])
capture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capture
spec.loader.exec_module(capture)
values = np.tile(np.array([1.5, -0.0, np.inf, -np.inf, np.nan]), 20000)
frame = pd.DataFrame({"x": values})
schema = T.StructType([T.StructField("x", T.DoubleType())])
class_name, converter_loaded = sys.argv[1:3]
replacement = object if class_name in ("DoubleType", "StructType", "StructField") else T.DoubleType
if converter_loaded == "True":
    import pyspark.sql.pandas.types as converter
    setattr(converter, class_name, replacement)
else:
    sys.modules.pop("pyspark.sql.pandas.types", None)
    setattr(T, class_name, replacement)
    assert "pyspark.sql.pandas.types" not in sys.modules
class Factory:
    def mktemp(self, name):
        return Path(tempfile.mkdtemp(prefix=name))
with capture._arrow_input_service(capture.protocol.__wrapped__(Factory())) as (spark, state):
    spark.createDataFrame(frame, schema).toArrow()
    assert state["config_requests"] == [], state["config_requests"]
    table = state["input_tables"][0]
    assert table.schema.names == ["x"], table.schema
    assert table.schema.field(0).type == capture.pa.float64()
    mask = np.isnan(values)
    assert table.column(0).is_null().to_pylist() == mask.tolist()
    actual = table.column(0).drop_null().combine_chunks().to_numpy(zero_copy_only=False)
    np.testing.assert_array_equal(actual.view(np.uint64), values[~mask].view(np.uint64))
'''
    result = subprocess.run([sys.executable, "-c", script, class_name,
                             str(converter_loaded), __file__],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr



def test_float64_rebound_pandas_alias_keeps_subclass_fallback(monkeypatch):
    from pyspark.sql._pandas_input import prepare_float64_table

    class Frame(pd.DataFrame):
        pass

    frame = Frame({"x": np.arange(100000, dtype=np.float64)})
    schema = T.StructType([T.StructField("x", T.DoubleType())])
    monkeypatch.setattr(pd, "DataFrame", Frame)
    assert prepare_float64_table(frame, schema) is None



@pytest.mark.parametrize("class_spoof", [False, True])
def test_float64_custom_extension_dtype_cannot_impersonate_numpy_float64(class_spoof):
    from pandas.api.extensions import ExtensionArray, ExtensionDtype, take
    from pyspark.sql._pandas_input import prepare_float64_table

    arrow_calls = []

    class FloatEqualDtype(ExtensionDtype):
        name = "float64_equal"
        type = np.float64
        kind = "f"
        na_value = np.nan

        @classmethod
        def construct_array_type(cls):
            return DifferentArrowArray

        def __eq__(self, other):
            return (isinstance(other, FloatEqualDtype)
                    or isinstance(other, np.dtype) and other == np.dtype("float64"))

        def __hash__(self):
            return hash(self.name)

    class DifferentArrowArray(ExtensionArray):
        def __init__(self, values):
            self._values = np.asarray(values, dtype=np.float64)

        @property
        def dtype(self):
            return FloatEqualDtype()

        @property
        def nbytes(self):
            return self._values.nbytes

        def __len__(self):
            return len(self._values)

        def __getitem__(self, key):
            result = self._values[key]
            return type(self)(result) if isinstance(result, np.ndarray) else result

        def isna(self):
            return np.isnan(self._values)

        def take(self, indices, allow_fill=False, fill_value=None):
            return type(self)(take(self._values, indices, allow_fill=allow_fill,
                                   fill_value=fill_value))

        def copy(self):
            return type(self)(self._values.copy())

        @classmethod
        def _from_sequence(cls, scalars, *, dtype=None, copy=False):
            values = np.asarray(scalars, dtype=np.float64)
            return cls(values.copy() if copy else values)

        def to_numpy(self, dtype=None, copy=False, na_value=None):
            if dtype is not None:
                return self._values.astype(dtype, copy=copy)
            return self._values.copy() if copy else self._values

        def __arrow_array__(self, type=None):
            arrow_calls.append("arrow")
            return pa.array([999.0] * len(self), type=type)

    frame = pd.DataFrame({"x": pd.Series(DifferentArrowArray(np.ones(100000)), copy=False)})
    schema = T.StructType([T.StructField("x", T.DoubleType())])
    if class_spoof:
        def spoof_class(self, name):
            if name == "__class__":
                return type(np.dtype("float64"))
            return object.__getattribute__(self, name)

        FloatEqualDtype.__getattribute__ = spoof_class
    dtype = frame.dtypes.iloc[0]
    assert type(frame) is pd.DataFrame
    assert type(dtype) is FloatEqualDtype
    assert isinstance(dtype, np.dtype) is class_spoof
    assert dtype == np.dtype("float64")
    assert prepare_float64_table(frame, schema) is None
    assert arrow_calls == []



@pytest.mark.parametrize("method", ["astype", "notna"])
def test_float64_instance_conversion_hooks_keep_row_fallback(protocol, method):
    values = np.arange(100000, dtype=np.float64)
    frame = pd.DataFrame({"x": values})
    schema = T.StructType([T.StructField("x", T.DoubleType())])
    original = getattr(frame, method)
    calls = []
    missing = np.zeros(len(values), dtype=bool)
    expected = values

    if method == "astype":
        expected = values + 7.0

        def customized(*args, **kwargs):
            calls.append("astype")
            converted = original(*args, **kwargs)
            converted.iloc[:, 0] = converted.iloc[:, 0] + 7.0
            return converted
    else:
        missing[::3] = True

        def customized():
            calls.append("notna")
            mask = original()
            mask.iloc[::3, 0] = False
            return mask

    setattr(frame, method, customized)
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert calls == [method]
        assert state["config_requests"] == []
        _assert_float64_ipc(state["input_tables"][0], [expected], schema, missing=[missing])



@pytest.mark.parametrize("method", ["astype", "notna", "where"])
@pytest.mark.parametrize("before_import", [False, True])
def test_float64_class_conversion_hooks_keep_row_fallback(
    protocol, monkeypatch, method, before_import
):
    import importlib
    import sys

    helper_name = "pyspark.sql._pandas_input"
    if before_import:
        monkeypatch.delitem(sys.modules, helper_name, raising=False)
    else:
        importlib.import_module(helper_name)
    values = np.arange(100000, dtype=np.float64)
    frame = pd.DataFrame({"x": values})
    schema = T.StructType([T.StructField("x", T.DoubleType())])
    original = getattr(pd.DataFrame, method)
    calls = []
    missing = np.zeros(len(values), dtype=bool)
    expected = values

    if method == "astype":
        expected = values + 7.0

        def customized(self, *args, **kwargs):
            calls.append("astype")
            converted = original(self, *args, **kwargs)
            converted.iloc[:, 0] = converted.iloc[:, 0] + 7.0
            return converted
    elif method == "notna":
        missing[::3] = True

        def customized(self):
            calls.append("notna")
            mask = original(self)
            mask.iloc[::3, 0] = False
            return mask
    else:
        expected = values + 11.0

        def customized(self, *args, **kwargs):
            calls.append("where")
            converted = original(self, *args, **kwargs)
            converted.iloc[:, 0] = converted.iloc[:, 0] + 11.0
            return converted

    monkeypatch.setattr(pd.DataFrame, method, customized)
    if before_import:
        importlib.import_module(helper_name)
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert calls == [method]
        assert state["config_requests"] == []
        _assert_float64_ipc(state["input_tables"][0], [expected], schema, missing=[missing])



def test_float64_literal_numpy_dtype_custom_backing_keeps_row_error(protocol):
    from pyspark.sql._pandas_input import prepare_float64_table

    calls = []

    class DifferentArrowArray(pd.arrays.FloatingArray):
        def __arrow_array__(self, type=None):
            calls.append("arrow")
            return pa.array(np.full(len(self), 999.0), type=type)

    values = np.tile(np.array([1.5, -0.0, np.inf, -np.inf, np.nan]), 20000)
    array = DifferentArrowArray(values.copy(), np.isnan(values))
    frame = pd.DataFrame({"x": pd.Series(array, copy=False)})
    # Construct extension storage before advertising a native dtype; otherwise
    # pandas legitimately normalizes the constructor input into a NumPy block.
    DifferentArrowArray.dtype = property(lambda self: np.dtype("float64"))
    schema = T.StructType([T.StructField("x", T.DoubleType())])
    assert type(frame) is pd.DataFrame
    assert type(frame["x"]) is pd.Series
    assert type(frame["x"].array) is DifferentArrowArray
    assert type(frame.dtypes.iloc[0]) is type(np.dtype("float64"))
    assert frame.dtypes.iloc[0] == np.dtype("float64")
    assert prepare_float64_table(frame, schema) is None
    assert calls == []
    with _arrow_input_service(protocol) as (spark, state):
        # This inconsistent ExtensionArray already fails pandas object conversion.
        # Preserve that baseline error before Arrow hooks or configuration RPCs.
        with pytest.raises(AttributeError, match="na_value"):
            spark.createDataFrame(frame, schema)
        assert state["config_requests"] == []
        assert state["input_tables"] == []
        assert calls == []



@pytest.mark.parametrize("raises", [False, True])
def test_float64_ndarray_subclass_converts_as_pyspark(protocol, raises):
    # The fast path declines a NumPy subclass behind pandas' concrete wrapper; the frame then
    # converts as PySpark's Arrow conversion does, which reads the raw storage and does not
    # call the subclass's astype, whatever it would do.
    from pyspark.sql._pandas_input import prepare_float64_table

    calls = []

    class Values(np.ndarray):
        def astype(self, *args, **kwargs):
            calls.append("astype")
            if raises:
                raise ValueError("custom ndarray astype")
            return np.ndarray.astype(self, *args, **kwargs) + 7.0

    values = np.arange(100000, dtype=np.float64)
    # Avoid pandas consolidating the subclass storage into a plain ndarray.
    frame = pd.DataFrame({"x": values.view(Values)}, copy=False)
    schema = T.StructType([T.StructField("x", T.DoubleType())])
    assert type(frame["x"].array._ndarray) is Values
    assert prepare_float64_table(frame, schema) is None
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        _assert_float64_ipc(state["input_tables"][0], [values], schema)
        assert calls == []
        assert state["uploads"] == 0


def test_float64_missing_schema_alias_keeps_native_row_path(protocol, monkeypatch):
    values = np.arange(100000, dtype=np.float64)
    frame = pd.DataFrame({"x": values})
    schema = T.StructType([T.StructField("x", T.DoubleType())])
    calls = []

    def lazy_attribute(name):
        if name == "DoubleType":
            calls.append(name)
            raise RuntimeError("schema alias hook must not run before row conversion")
        raise AttributeError(name)

    monkeypatch.delitem(vars(T), "DoubleType")
    monkeypatch.setattr(T, "__getattr__", lazy_attribute, raising=False)
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        _assert_float64_ipc(state["input_tables"][0], [values], schema)
        assert calls == []
        assert state["uploads"] == 0



def test_float64_missing_concrete_numpy_backing_class_keeps_fallback(monkeypatch):
    import pandas.core.arrays.numpy_ as numpy_arrays
    from pyspark.sql._pandas_input import prepare_float64_table

    frame = pd.DataFrame({"x": np.arange(100000, dtype=np.float64)})
    schema = T.StructType([T.StructField("x", T.DoubleType())])
    monkeypatch.delattr(numpy_arrays, "NumpyExtensionArray", raising=False)
    monkeypatch.delattr(numpy_arrays, "PandasArray", raising=False)
    assert prepare_float64_table(frame, schema) is None



@pytest.mark.parametrize("rows", [10000, 99999, 100000])
def test_float64_public_threshold_reads_config_below_100k(protocol, rows):
    import json

    values = np.resize(np.array([np.nan, -0.0, 0.0, np.inf, -np.inf, 1.5]), rows)
    frame = pd.DataFrame({"original": values})
    schema = T.StructType([
        T.StructField("renamed", T.DoubleType(), True, {"boundary": {"rows": rows}}),
    ])
    with _arrow_input_service(
        protocol, **{"spark.sql.session.localRelationCacheThreshold": "999999999"}
    ) as (spark, state):
        if rows >= 100000:
            # The float64 fast path converts at this size without a configuration read.
            spark.createDataFrame(frame, schema).toArrow()
            _assert_float64_ipc(state["input_tables"][0], [values], schema)
            assert state["config_requests"] == []
            return
        state["config_error"] = grpc.StatusCode.INVALID_ARGUMENT
        # Official reads the configuration before converting, so its error comes first.
        with pytest.raises(Exception, match="configuration read failed"):
            spark.createDataFrame(frame, schema)
        spark.createDataFrame(frame, schema).toArrow()
        _assert_float64_ipc(state["input_tables"][0], [values], schema)
        assert json.loads(state["relations"][0].local_relation.schema) == json.loads(schema.json())
        assert state["uploads"] == 0
        assert state["config_requests"][0] == ("get", _NORMALIZATION_KEYS + _UPLOAD_KEYS)



def _float64_columns_guard_input(rows=100000):
    finite = np.array([
        0.0, -0.0, 1.5, -2.25, np.finfo(np.float64).max, -np.finfo(np.float64).max,
        np.finfo(np.float64).tiny, -np.finfo(np.float64).tiny,
        np.nextafter(0.0, 1.0), -np.nextafter(0.0, 1.0),
    ], dtype=np.float64)
    values = [
        np.resize(finite, rows),
        np.resize(np.array([np.nan, np.inf, -np.inf, -0.0, 1.5]), rows),
        np.full(rows, np.nan, dtype=np.float64),
    ]
    frame = pd.DataFrame(dict(zip(["original", "b", "c"], values)))
    schema = T.StructType([
        T.StructField("renamed.`double", T.DoubleType(), False, {"source": "float64", "id": 3}),
        T.StructField("nullable", T.DoubleType(), True, {"tag": [1, 2]}),
        T.StructField("all_null", T.DoubleType(), True, {"source": "all-null"}),
    ])
    return frame, schema, values


def _assert_float64_columns_guard_payload(state, schema, values):
    import json

    assert len(state["input_tables"]) == 1
    assert len(state["relations"]) == 1
    assert state["column_names"] == schema.names
    _assert_float64_ipc(state["input_tables"][0], values, schema)
    assert json.loads(state["relations"][0].local_relation.schema) == json.loads(schema.json())
    assert state["config_requests"] == []
    assert state["uploads"] == 0



@pytest.mark.parametrize("rows", [10000, 100000])
@pytest.mark.parametrize("location", [
    "subclass", "class", "owner", "backing_subclass_item", "backing_class_iter",
    "backing_class_getitem", "backing_instance_box", "arrow_class_iter",
])
@pytest.mark.parametrize("behavior", ["raise", "mutate"])
def test_float64_columns_iterator_hooks_keep_native_order(
    protocol, monkeypatch, rows, location, behavior
):
    from pandas.core.base import IndexOpsMixin

    values = np.arange(rows, dtype=np.float64)
    frame = pd.DataFrame({"original": values})
    schema = T.StructType([T.StructField("renamed", T.DoubleType())])
    original_iter = pd.Index.__iter__
    original_notna, original_astype = frame.notna, frame.astype
    calls = []
    invoked = False

    class ColumnsError(RuntimeError):
        pass

    def notna(*args, **kwargs):
        calls.append("notna")
        return original_notna(*args, **kwargs)

    def astype(*args, **kwargs):
        calls.append("astype")
        return original_astype(*args, **kwargs)

    def before_hook():
        nonlocal invoked
        if invoked:
            return
        invoked = True
        calls.append("columns iterator")
        if behavior == "raise":
            raise ColumnsError("columns hook precedes pandas normalization")
        # Install spies only when native iteration reaches the columns hook.
        # Before eligibility, normalization methods are the canonical methods.
        monkeypatch.setitem(vars(frame), "notna", notna)
        monkeypatch.setitem(vars(frame), "astype", astype)
        frame.iat[0, 0], frame.iat[-1, 0] = -0.0, np.nan

    def iterate(index):
        if index is columns:
            before_hook()
        return original_iter(index)

    if location == "subclass":
        class Columns(pd.Index):
            pass
        monkeypatch.setattr(Columns, "__iter__", iterate)
        labels = pd.Index(["original"], dtype=object)
        frame.columns = Columns._simple_new(labels._values, name=None)
        assert type(frame.columns) is Columns
    elif location == "backing_subclass_item":
        # IndexOpsMixin iteration uses ndarray.item, not ndarray.__iter__.
        class Names(np.ndarray):
            def item(self, *args):
                before_hook()
                return super().item(*args)
        backing = np.asarray(["original"], dtype=object).view(Names)
        frame.columns = pd.Index._simple_new(backing, name=None)
        assert type(frame.columns._values) is Names
    elif location.startswith(("backing_", "arrow_")):
        storage = "pyarrow" if location.startswith("arrow_") else "python"
        frame.columns = pd.Index(pd.array(["original"], dtype=pd.StringDtype(storage=storage)))
        backing = frame.columns._values
        if location.endswith("class_iter"):
            original = type(backing).__iter__

            def backing_iter(array):
                if array is backing:
                    before_hook()
                return original(array)

            monkeypatch.setattr(type(backing), "__iter__", backing_iter)
        elif location.endswith("class_getitem"):
            original = type(backing).__getitem__

            def backing_getitem(array, key):
                if array is backing:
                    before_hook()
                return original(array, key)

            monkeypatch.setattr(type(backing), "__getitem__", backing_getitem)
        else:
            original = backing._box_func

            def box(value):
                before_hook()
                return original(value)

            monkeypatch.setitem(vars(backing), "_box_func", box)
    else:
        monkeypatch.setattr(pd.Index if location == "class" else IndexOpsMixin,
                            "__iter__", iterate)
    columns = frame.columns

    with _arrow_input_service(protocol) as (spark, state):
        if behavior == "raise":
            with pytest.raises(ColumnsError,
                               match="columns hook precedes pandas normalization"):
                spark.createDataFrame(frame, schema)
            assert calls == ["columns iterator"]
            assert state["config_requests"] == []
            assert state["input_tables"] == []
            assert state["uploads"] == 0
        else:
            spark.createDataFrame(frame, schema).toArrow()
            expected = values.copy()
            expected[0], expected[-1] = -0.0, np.nan
            _assert_float64_columns_guard_payload(state, schema, [expected])
            assert calls == ["columns iterator", "notna", "astype"]



@pytest.mark.parametrize("rows", [10000, 100000])
def test_float64_rebound_index_values_keeps_native_error(protocol, monkeypatch, rows):

    frame = pd.DataFrame({"original": np.arange(rows, dtype=np.float64)})
    schema = T.StructType([T.StructField("renamed", T.DoubleType())])
    columns = frame.columns
    original = pd.Index._values.fget
    calls = []

    def values(index):
        if index is columns:
            calls.append("_values")
            raise RuntimeError("columns backing hook precedes normalization")
        return original(index)

    monkeypatch.setattr(pd.Index, "_values", property(values))
    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(RuntimeError,
                           match="columns backing hook precedes normalization"):
            spark.createDataFrame(frame, schema)
        assert calls == ["_values"]
        assert state["config_requests"] == []
        assert state["input_tables"] == []
        assert state["uploads"] == 0



@pytest.mark.parametrize("rows", [10000, 100000])
def test_float64_rebound_index_data_keeps_native_read_order(protocol, monkeypatch, rows):
    frame, schema, values = _float64_columns_guard_input(rows)
    columns, backing = frame.columns, vars(frame.columns)["_data"]
    original_notna, original_astype = frame.notna, frame.astype
    calls = []
    normalizing = False
    allowed_reads = None

    class ColumnsDataError(RuntimeError):
        pass

    def notna(*args, **kwargs):
        nonlocal normalizing
        normalizing = True
        calls.append("notna")
        return original_notna(*args, **kwargs)

    def astype(*args, **kwargs):
        calls.append("astype")
        return original_astype(*args, **kwargs)

    def data(index):
        if index is columns:
            if not normalizing:
                calls.append("_data")
                if allowed_reads is not None and calls.count("_data") > allowed_reads:
                    raise ColumnsDataError("eligibility added a columns storage read")
                # Preserve the native first columns read and only then attach
                # observers; no private optimization call determines fallback.
                monkeypatch.setitem(vars(frame), "notna", notna)
                monkeypatch.setitem(vars(frame), "astype", astype)
            return backing
        return vars(index)["_data"]

    def set_data(index, value):
        vars(index)["_data"] = value

    monkeypatch.setattr(pd.Index, "_data", property(data, set_data), raising=False)
    import pyspark.sql._pandas_input as helper

    with _arrow_input_service(protocol) as (spark, state):
        # Measure the current native fallback's storage reads, rather than the
        # former tolist path's single-read contract.
        with monkeypatch.context() as baseline:
            baseline.setattr(helper, "prepare_float64_input", lambda *_: None)
            spark.createDataFrame(frame, schema).toArrow()
        baseline_calls = calls.copy()
        allowed_reads = baseline_calls.count("_data")
        assert allowed_reads > 0 and "notna" in baseline_calls
        calls.clear()
        normalizing = False
        # Restore canonical methods before the candidate eligibility check;
        # native label iteration will install the spies through the storage hook.
        monkeypatch.delitem(vars(frame), "notna")
        monkeypatch.delitem(vars(frame), "astype")
        spark.createDataFrame(frame, schema).toArrow()
        # The rebound storage declines the columnar path without reading the columns
        # more than the row path does. pandas caches some of these reads, so a later
        # row conversion of the same frame can read them fewer times than the first.
        assert "notna" in calls
        assert calls.count("_data") <= allowed_reads
        import json

        assert state["config_requests"] == []
        assert state["uploads"] == 0
        assert len(state["input_tables"]) == 2
        assert len(state["relations"]) == 2
        assert state["input_tables"][1].equals(state["input_tables"][0], check_metadata=True)
        for table, relation in zip(state["input_tables"], state["relations"]):
            _assert_float64_ipc(table, values, schema)
            assert json.loads(relation.local_relation.schema) == json.loads(schema.json())



@pytest.mark.parametrize("storage", [
    "object", "default", "python", "pyarrow", "python_nan", "pyarrow_nan",
])
def test_float64_canonical_columns_storage_stays_columnar(protocol, monkeypatch, storage):
    frame, schema, values = _float64_columns_guard_input()
    labels = ["original", "b", "c"]
    if storage == "object":
        frame.columns = pd.Index(labels, dtype=object)
    elif storage != "default":
        kind = storage.removesuffix("_nan")
        label_dtype = (pd.StringDtype(storage=kind, na_value=np.nan)
                       if storage.endswith("_nan") else pd.StringDtype(storage=kind))
        frame.columns = pd.Index(pd.array(labels, dtype=label_dtype))

    def no_rows(*args, **kwargs):
        raise AssertionError("ordinary float64 columns must retain columnar construction")

    monkeypatch.setattr(pd.DataFrame, "values", property(no_rows))
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        _assert_float64_columns_guard_payload(state, schema, values)


@pytest.mark.parametrize("location", [
    "subclass", "class", "owner", "backing_subclass_item", "backing_class_iter",
    "backing_class_getitem", "backing_instance_box", "arrow_class_iter",
])
@pytest.mark.parametrize("behavior", ["raise", "mutate"])
def test_string_columns_iterator_hooks_keep_native_order(
    protocol, monkeypatch, location, behavior
):
    from pandas.core.base import IndexOpsMixin

    values = ["plain"] * 10000
    frame = pd.DataFrame({"original": values}, dtype=object)
    schema = T.StructType([T.StructField("renamed", T.StringType())])
    original_iter = pd.Index.__iter__
    original_notna, original_astype = frame.notna, frame.astype
    calls = []
    invoked = False

    class ColumnsError(RuntimeError):
        pass

    def notna(*args, **kwargs):
        calls.append("notna")
        return original_notna(*args, **kwargs)

    def astype(*args, **kwargs):
        calls.append("astype")
        return original_astype(*args, **kwargs)

    def before_hook():
        nonlocal invoked
        if invoked:
            return
        invoked = True
        calls.append("columns iterator")
        if behavior == "raise":
            raise ColumnsError("columns hook precedes pandas normalization")
        # Install spies only when native iteration reaches the columns hook.
        # Before eligibility, normalization methods are the canonical methods.
        monkeypatch.setitem(vars(frame), "notna", notna)
        monkeypatch.setitem(vars(frame), "astype", astype)
        frame.iat[0, 0], frame.iat[-1, 0] = "changed", None

    def iterate(index):
        if index is columns:
            before_hook()
        return original_iter(index)

    if location == "subclass":
        class Columns(pd.Index):
            pass
        monkeypatch.setattr(Columns, "__iter__", iterate)
        labels = pd.Index(["original"], dtype=object)
        frame.columns = Columns._simple_new(labels._values, name=None)
        assert type(frame.columns) is Columns
    elif location == "backing_subclass_item":
        # IndexOpsMixin iteration uses ndarray.item, not ndarray.__iter__.
        class Names(np.ndarray):
            def item(self, *args):
                before_hook()
                return super().item(*args)
        backing = np.asarray(["original"], dtype=object).view(Names)
        frame.columns = pd.Index._simple_new(backing, name=None)
        assert type(frame.columns._values) is Names
    elif location.startswith(("backing_", "arrow_")):
        storage = "pyarrow" if location.startswith("arrow_") else "python"
        frame.columns = pd.Index(pd.array(["original"], dtype=pd.StringDtype(storage=storage)))
        backing = frame.columns._values
        if location.endswith("class_iter"):
            original = type(backing).__iter__

            def backing_iter(array):
                if array is backing:
                    before_hook()
                return original(array)

            monkeypatch.setattr(type(backing), "__iter__", backing_iter)
        elif location.endswith("class_getitem"):
            original = type(backing).__getitem__

            def backing_getitem(array, key):
                if array is backing:
                    before_hook()
                return original(array, key)

            monkeypatch.setattr(type(backing), "__getitem__", backing_getitem)
        else:
            original = backing._box_func

            def box(value):
                before_hook()
                return original(value)

            monkeypatch.setitem(vars(backing), "_box_func", box)
    else:
        monkeypatch.setattr(pd.Index if location == "class" else IndexOpsMixin,
                            "__iter__", iterate)
    columns = frame.columns

    with _arrow_input_service(protocol) as (spark, state):
        if behavior == "raise":
            with pytest.raises(ColumnsError,
                               match="columns hook precedes pandas normalization"):
                spark.createDataFrame(frame, schema)
            assert calls == ["columns iterator"]
            assert state["config_requests"] == []
            assert state["input_tables"] == []
            assert state["uploads"] == 0
        else:
            spark.createDataFrame(frame, schema).toArrow()
            expected = values.copy()
            expected[0], expected[-1] = "changed", None
            import json

            table = state["input_tables"][0]
            assert table.schema.names == ["renamed"]
            assert table.column(0).type == pa.string()
            assert table.column(0).to_pylist() == expected
            assert table.column(0).is_null().to_pylist() == [False] * 9999 + [True]
            assert table.schema.equals(pa.schema([pa.field("renamed", pa.string())]),
                                       check_metadata=True)
            assert json.loads(state["relations"][0].local_relation.schema) == json.loads(schema.json())
            assert state["config_requests"] == []
            assert state["uploads"] == 0
            assert calls == ["columns iterator", "notna", "astype"]



def test_string_rebound_index_values_keeps_native_error(protocol, monkeypatch):

    frame = pd.DataFrame({"original": ["plain"] * 10000}, dtype=object)
    schema = T.StructType([T.StructField("renamed", T.StringType())])
    columns = frame.columns
    original = pd.Index._values.fget
    calls = []

    def values(index):
        if index is columns:
            calls.append("_values")
            raise RuntimeError("columns backing hook precedes normalization")
        return original(index)

    monkeypatch.setattr(pd.Index, "_values", property(values))
    assert calls == []
    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(RuntimeError,
                           match="columns backing hook precedes normalization"):
            spark.createDataFrame(frame, schema)
        assert calls == ["_values"]
        assert state["config_requests"] == []
        assert state["input_tables"] == []



@pytest.mark.parametrize("nullable", [False, True])
def test_string_rebound_index_data_keeps_native_read_order(protocol, monkeypatch, nullable):
    import json

    values = ["plain", "é", "漢字", ""] * 2500
    if nullable:
        values[1] = None
    frame = pd.DataFrame({"original": values}, dtype=object)
    schema = T.StructType([
        T.StructField("renamed", T.StringType(), nullable, {"source": "native"})
    ])
    columns, backing = frame.columns, vars(frame.columns)["_data"]
    original_notna, original_astype = frame.notna, frame.astype
    calls = []
    normalizing = False
    allowed_reads = None

    class ColumnsDataError(RuntimeError):
        pass

    def notna(*args, **kwargs):
        nonlocal normalizing
        normalizing = True
        calls.append("notna")
        return original_notna(*args, **kwargs)

    def astype(*args, **kwargs):
        calls.append("astype")
        return original_astype(*args, **kwargs)

    def data(index):
        if index is columns:
            if not normalizing:
                calls.append("_data")
                if allowed_reads is not None and calls.count("_data") > allowed_reads:
                    raise ColumnsDataError("eligibility added a columns storage read")
                # Canonical normalization remains intact before eligibility.
                monkeypatch.setitem(vars(frame), "notna", notna)
                monkeypatch.setitem(vars(frame), "astype", astype)
            return backing
        return vars(index)["_data"]

    def set_data(index, value):
        # Keep ordinary storage initialization for indexes created by pandas.
        vars(index)["_data"] = value

    monkeypatch.setattr(pd.Index, "_data", property(data, set_data), raising=False)
    import pyspark.sql._pandas_input as helper

    with _arrow_input_service(protocol) as (spark, state):
        # Measure the current native fallback's storage reads, rather than the
        # former tolist path's single-read contract.
        with monkeypatch.context() as baseline:
            baseline.setattr(helper, "prepare_string_input", lambda *_: None)
            spark.createDataFrame(frame, schema).toArrow()
        baseline_calls = calls.copy()
        allowed_reads = baseline_calls.count("_data")
        assert allowed_reads > 0 and "notna" in baseline_calls
        calls.clear()
        normalizing = False
        # Restore canonical methods before the candidate eligibility check;
        # native label iteration will install the spies through the storage hook.
        monkeypatch.delitem(vars(frame), "notna")
        monkeypatch.delitem(vars(frame), "astype")
        spark.createDataFrame(frame, schema).toArrow()
        # The rebound storage declines the string path without reading the columns
        # more than the row path does. pandas caches some of these reads, so a later
        # row conversion of the same frame can read them fewer times than the first.
        assert "notna" in calls
        assert calls.count("_data") <= allowed_reads
        assert not state["config_requests"]
        assert state["uploads"] == 0
        assert len(state["input_tables"]) == 2
        assert len(state["relations"]) == 2
        table = state["input_tables"][1]
        assert table.equals(state["input_tables"][0], check_metadata=True)
        assert table.column(0).to_pylist() == values
        assert table.column(0).is_null().to_pylist() == [value is None for value in values]
        assert table.schema.equals(
            pa.schema([pa.field("renamed", pa.string(), nullable=nullable)]),
            check_metadata=True,
        )
        assert json.loads(state["relations"][1].local_relation.schema) == {
            "type": "struct", "fields": [
                {"name": "renamed", "type": "string", "nullable": nullable,
                 "metadata": {"source": "native"}}
            ],
        }



@pytest.mark.parametrize("kind", [
    "object", "default", "python", "pyarrow", "python_numpy", "pyarrow_numpy",
])
def test_string_canonical_column_storage_stays_eligible(kind):
    from pyspark.sql._pandas_input import _has_canonical_columns, prepare_string_input

    frame = pd.DataFrame({"original": ["plain"] * 10000}, dtype=object)
    schema = T.StructType([T.StructField("renamed", T.StringType())])
    if kind == "object":
        frame.columns = pd.Index(np.asarray(["original"], dtype=object), dtype=object)
    elif kind == "default":
        frame.columns = pd.Index(["original"])
    elif kind.endswith("_numpy"):
        from pandas.core.arrays import string_ as strings, string_arrow as arrow_strings

        module = strings if kind.startswith("python") else arrow_strings
        name = ("StringArrayNumpySemantics" if kind.startswith("python")
                else "ArrowStringArrayNumpySemantics")
        cls = getattr(module, name, None)
        if cls is None:
            pytest.skip("older infer-string storage class is absent")
        frame.columns = pd.Index._simple_new(cls._from_sequence(["original"]), name=None)
    else:
        frame.columns = pd.Index(pd.array(["original"], dtype=pd.StringDtype(storage=kind)))
    assert _has_canonical_columns(frame.columns)
    assert type(prepare_string_input(frame, schema)) is bytes



@pytest.mark.parametrize("missing", ["module", "owner", "base"])
def test_string_absent_private_columns_storage_keeps_row_fallback(monkeypatch, missing):
    import builtins
    from pandas.core.arrays import numpy_ as numpy_module, string_ as string_module
    from pyspark.sql._pandas_input import _has_canonical_columns

    columns = pd.Index(pd.array(["original"], dtype=pd.StringDtype(storage="python")))
    if missing == "module":
        original = builtins.__import__

        def without_arrow(name, *args, **kwargs):
            if name == "pandas.core.arrays.arrow":
                raise ImportError("private storage module is absent")
            return original(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", without_arrow)
    elif missing == "owner":
        monkeypatch.delattr(numpy_module, "NumpyExtensionArray")
    else:
        monkeypatch.delattr(string_module, "BaseStringArray")
    assert not _has_canonical_columns(columns)



@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize("large", ["false", "true"])
def test_string_columns_preserve_every_ipc_cell(protocol, monkeypatch, cached, verify, large):
    import json

    from pyspark.sql.pandas.types import from_arrow_metadata

    strings = ["", "plain", "é", "e\u0301", "\x00embedded\x00", "漢字", "🙂",
               "\U0010ffff", "\uffff", "\u202eright-to-left"] * 1000
    nullable = [None if i % 7 == 0 else value for i, value in enumerate(reversed(strings))]
    expected = [strings, nullable, [None] * len(strings)]
    frame = pd.DataFrame(dict(zip(["original", "b", "c"], expected)), dtype=object)
    schema = T.StructType([
        T.StructField("renamed.`text", T.StringType(), False, {"source": "string", "id": 3}),
        T.StructField("nullable", T.StringType(), True, {"tag": [1, 2]}),
        T.StructField("all_null", T.StringType()),
    ])

    def no_rows(*args, **kwargs):
        raise AssertionError("string fast path must not materialize Python rows")

    for method in ("to_numpy", "iterrows", "itertuples"):
        monkeypatch.setattr(pd.DataFrame, method, no_rows)
    monkeypatch.setattr(pd.DataFrame, "values", property(no_rows))
    monkeypatch.setattr(pd.Series, "tolist", no_rows)
    overrides = {
        "spark.sql.session.localRelationCacheThreshold": "1" if cached else "999999999",
        "spark.sql.execution.arrow.useLargeVarTypes": large,
        "spark.sql.session.localRelationChunkSizeRows": "997",
    }
    with _arrow_input_service(protocol, **overrides) as (spark, state):
        result = spark.createDataFrame(frame, schema, verifySchema=verify)
        assert not result.isLocal()
        result.toArrow()
        assert state["column_names"] == schema.names
        table = state["input_tables"][0]
        assert table.num_rows == len(strings)
        assert table.schema.names == schema.names
        for column, values, field in zip(table.columns, expected, schema.fields):
            assert column.type == pa.string()
            assert column.to_pylist() == values
            actual_field = table.schema.field(field.name)
            assert actual_field.nullable == field.nullable
            assert from_arrow_metadata(actual_field.metadata) is None
            spark_field = json.loads(state["relations"][0].local_relation.schema)["fields"][
                schema.names.index(field.name)]
            assert spark_field["metadata"] == field.metadata
        assert state["config_requests"] == []
        assert state["uploads"] == 0



@pytest.mark.parametrize("reason", [
    "small", "empty", "inferred", "names", "ddl", "cast", "atomic", "mixed_schema",
    "width", "duplicate_input", "duplicate_schema", "input_nul", "schema_nul",
    "input_number", "datetime", "date", "nan", "pd_na", "nat", "mixed", "bytes",
    "nonnullable_null", "frame_subclass", "string_subclass", "string_dtype",
    "arrow_string_dtype", "late_unsupported", "surrogate", "late_surrogate", "collation",
])
def test_string_column_guard_handles_supported_and_unsupported_inputs(reason):
    from pyspark.sql._pandas_input import prepare_string_input

    class Frame(pd.DataFrame):
        pass

    class Text(str):
        def __str__(self):
            raise AssertionError("guard must not call conversion hooks")

    class Unsupported:
        def __str__(self):
            raise AssertionError("guard must not stringify unsupported objects")

    values = ["plain"] * 10000
    frame = pd.DataFrame({"text": values}, dtype=object)
    schema = T.StructType([T.StructField("text", T.StringType())])
    if reason == "small":
        frame = frame.iloc[:9999]
    elif reason == "empty":
        frame = frame.iloc[:0]
    elif reason in ("inferred", "names", "ddl", "cast", "atomic"):
        schema = {"inferred": None, "names": ["text"], "ddl": "text STRING",
                  "cast": T.StructType([T.StructField("text", T.LongType())]),
                  "atomic": T.StringType()}[reason]
    elif reason == "mixed_schema":
        frame["long"] = np.arange(10000)
        schema.add("long", T.LongType())
    elif reason == "width":
        schema.add("extra", T.StringType())
    elif reason in ("duplicate_input", "duplicate_schema"):
        frame["other"] = values
        schema.add("other", T.StringType())
        if reason == "duplicate_input":
            frame.columns = ["text", "text"]
        else:
            schema = T.StructType([T.StructField("text", T.StringType())] * 2)
    elif reason in ("input_nul", "input_number"):
        frame.columns = ["text\x00" if reason == "input_nul" else 1]
    elif reason == "schema_nul":
        schema = T.StructType([T.StructField("text\x00", T.StringType())])
    elif reason == "frame_subclass":
        frame = Frame(frame)
    elif reason in ("string_dtype", "arrow_string_dtype"):
        frame = pd.DataFrame({"text": pd.Series(
            values, dtype=pd.StringDtype(storage="pyarrow" if reason == "arrow_string_dtype"
                                        else "python"))})
    elif reason == "collation":
        schema = T.StructType([T.StructField("text", T.StringType("UTF8_LCASE"))])
    else:
        value = {
            "datetime": datetime(2020, 1, 1), "date": date(2020, 1, 1),
            "nan": float("nan"), "pd_na": pd.NA, "nat": pd.NaT, "mixed": 42,
            "bytes": b"plain", "nonnullable_null": None, "string_subclass": Text("plain"),
            "late_unsupported": Unsupported(), "surrogate": "\ud800", "late_surrogate": "\udfff",
        }[reason]
        if reason == "nonnullable_null":
            schema = T.StructType([T.StructField("text", T.StringType(), False)])
        if reason in ("late_unsupported", "late_surrogate"):
            # pandas 3 infers StringDtype for a newly assigned list column.
            # Keep these unsupported-object fixtures on concrete object storage.
            frame["other"] = pd.Series(values, dtype=object)
            schema.add("other", T.StringType())
            backing = frame["other"].array._ndarray
            assert type(backing) is np.ndarray and backing.dtype == np.dtype("object")
        frame.iat[-1, -1] = value
        if reason in ("late_unsupported", "late_surrogate"):
            assert frame.iat[-1, -1] is value
    result = prepare_string_input(frame, schema)
    if reason == "arrow_string_dtype":
        # Arrow string storage has a path of its own, which returns an Arrow table.
        assert type(result) is pa.Table
    else:
        assert result is None


# Columns kept in pandas' Arrow string storage reach the importer as Arrow, without Python rows.

def _arrow_string_column(texts, storage):
    if storage == "numpy_semantics":
        try:
            dtype = pd.StringDtype("pyarrow", na_value=float("nan"))
        except TypeError:  # pandas 2.2
            dtype = pd.StringDtype("pyarrow_numpy")
        return pd.array(texts, dtype=dtype)
    if storage == "chunked":
        return pd.arrays.ArrowStringArray(pa.chunked_array(
            [pa.array(texts[:7], pa.large_string()), pa.array(texts[7:], pa.large_string())]))
    # The sliced case uses this storage, then slices the constructed frame below.
    return pd.array(texts, dtype="string[pyarrow]")


def _force_native_pandas_rows(monkeypatch):
    import importlib

    # Earlier import-hook tests can leave the package attribute pointing at a
    # different module object from the one imported by the native binding.
    helper = importlib.import_module("pyspark.sql._pandas_input")
    for name in ("prepare_int64_table", "timestamp_columns", "prepare_float64_input",
                 "prepare_string_input", "prepare_date_table"):
        monkeypatch.setattr(helper, name, lambda *_: None)
    for name in ("accepts_pandas_table", "accepts_pandas_frame"):
        monkeypatch.setattr(helper, name, lambda *_: False)


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("storage", ["pyarrow", "numpy_semantics", "chunked", "sliced"])
def test_arrow_string_columns_preserve_every_ipc_cell(protocol, monkeypatch, cached, storage):
    from pyspark.sql.pandas.types import from_arrow_metadata

    texts = ["", "a", "\x00", "café", "中文", "\U0001f600", "é", None] * 2000
    other = [None if i % 3 == 0 else str(i) for i in range(len(texts))]
    frame = pd.DataFrame({
        "original": _arrow_string_column(texts, storage),
        "b": _arrow_string_column(other, storage),
        "c": _arrow_string_column([None] * len(texts), storage),
    })
    expected = [texts, other, [None] * len(texts)]
    if storage == "sliced":
        frame = frame.iloc[3:-5]
        expected = [values[3:-5] for values in expected]
    schema = T.StructType([
        T.StructField("renamed.`text", T.StringType(), True, {"source": "string"}),
        T.StructField("other", T.StringType()),
        T.StructField("all_null", T.StringType()),
    ])

    def no_rows(*args, **kwargs):
        raise AssertionError("Arrow string storage must not materialize Python rows")

    for method in ("to_numpy", "iterrows", "itertuples"):
        monkeypatch.setattr(pd.DataFrame, method, no_rows)
    monkeypatch.setattr(pd.DataFrame, "values", property(no_rows))
    monkeypatch.setattr(pd.Series, "tolist", no_rows)
    overrides = {"spark.sql.session.localRelationCacheThreshold": "1" if cached else "999999999"}
    with _arrow_input_service(protocol, **overrides) as (spark, state):
        df = spark.createDataFrame(frame, schema)
        df.toArrow()
        assert state["column_names"] == schema.names
        table = state["input_tables"][0]
        assert table.schema.names == schema.names
        for column, values, field in zip(table.columns, expected, schema.fields):
            assert column.type == pa.string()
            assert column.to_pylist() == values
            actual_field = table.schema.field(field.name)
            assert actual_field.nullable == field.nullable
            assert from_arrow_metadata(actual_field.metadata) == (field.metadata or None)
        assert bool(state["uploads"]) == cached


@pytest.mark.parametrize("reason", [
    "python_storage", "mixed", "nonnullable_null", "subclass", "small", "atomic_schema",
])
def test_arrow_string_columns_keep_the_row_path(reason):
    from pyspark.sql._pandas_input import prepare_string_input

    texts = ["a", None, "b"] * 4000
    columns = {"a": _arrow_string_column(texts, "pyarrow"), "b": _arrow_string_column(texts, "pyarrow")}
    fields = [T.StructField("a", T.StringType()), T.StructField("b", T.StringType())]
    if reason == "python_storage":
        columns["b"] = pd.array(texts, dtype="string[python]")
    elif reason == "mixed":
        columns["b"] = pd.Series(texts, dtype=object)
    elif reason == "nonnullable_null":
        fields[1] = T.StructField("b", T.StringType(), False)
    elif reason == "subclass":
        class Custom(pd.arrays.ArrowStringArray):
            pass

        columns["b"] = Custom(pa.chunked_array([pa.array(texts)]))
    elif reason == "small":
        columns = {name: values[:9999] for name, values in columns.items()}
    elif reason == "atomic_schema":
        fields[1] = T.StructField("b", T.IntegerType())
    frame = pd.DataFrame(columns)
    assert prepare_string_input(frame, T.StructType(fields)) is None


@pytest.mark.parametrize("storage", ["pyarrow", "numpy_semantics"])
def test_arrow_string_columns_match_the_row_path(protocol, monkeypatch, storage):
    import json

    from pyspark.sql.pandas.types import from_arrow_metadata

    texts = ["x" * (i % 5) if i % 4 else None for i in range(12000)]
    frame = pd.DataFrame({"v": _arrow_string_column(texts, storage)})
    schema = T.StructType([T.StructField("v", T.StringType(), True, {"source": "row-control"})])
    tables = []
    logical_schemas = []
    for row_control in (False, True):
        with monkeypatch.context() as control:
            if row_control:
                _force_native_pandas_rows(control)
            with _arrow_input_service(protocol) as (spark, state):
                spark.createDataFrame(frame, schema).toArrow()
                tables.append(state["input_tables"][0])
                logical_schemas.append(json.loads(state["spark_schema"]))
                assert bool(state["config_requests"]) == (not row_control)
    assert tables[0].equals(tables[1])
    # Row IPC omits field metadata; Spark overlays the explicit logical schema.
    # The Arrow importer also embeds metadata in IPC, so compare both contracts.
    assert from_arrow_metadata(tables[0].schema.field(0).metadata) == {"source": "row-control"}
    assert tables[1].schema.field(0).metadata is None
    assert logical_schemas == [json.loads(schema.json())] * 2


@pytest.mark.parametrize("storage", ["pyarrow", "numpy_semantics"])
@pytest.mark.parametrize("layout", ["small", "mixed"])
def test_arrow_string_generic_explicit_schema_stays_columnar(
    protocol, monkeypatch, storage, layout
):
    from pyspark.sql._pandas_input import accepts_pandas_table, prepare_string_input

    texts = ["", None, "中文", "a\x00b"] * (3 if layout == "small" else 3000)
    frame = pd.DataFrame({"v": _arrow_string_column(texts, storage)})
    fields = [T.StructField("renamed", T.StringType())]
    expected = {"renamed": texts}
    if layout == "mixed":
        frame["number"] = np.arange(len(texts), dtype=np.int64)
        fields.append(T.StructField("number", T.LongType()))
        expected["number"] = list(range(len(texts)))
    schema = T.StructType(fields)
    # These ordinary inputs miss the specialized string route but remain eligible
    # for upstream's generic explicit-schema conversion.
    assert prepare_string_input(frame, schema) is None
    assert accepts_pandas_table(frame, schema)

    def no_rows(*args, **kwargs):
        raise AssertionError("generic Arrow strings must not materialize pandas rows")

    monkeypatch.setattr(pd.DataFrame, "values", property(no_rows))
    monkeypatch.setattr(pd.Series, "tolist", no_rows)
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert state["config_requests"]
        assert state["input_tables"][0].to_pydict() == expected
        assert state["input_tables"][0].schema.field(0).type == pa.string()


@pytest.mark.parametrize("scope", ["public", "converter", "captured"])
@pytest.mark.parametrize("class_name", ["BooleanType", "StringType", "StructType"])
def test_arrow_string_rebound_converter_alias_keeps_native_rows(scope, class_name):
    import sys

    script = r'''
import importlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
import pandas as pd
import pytest
from pyspark import _pyspark as native
from pyspark.sql import types as T

_pandas_input = importlib.import_module("pyspark.sql._pandas_input")
spec = importlib.util.spec_from_file_location("arrow_string_alias_capture", sys.argv[3])
capture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capture
spec.loader.exec_module(capture)
texts = ["original", None] * 5000
frame = pd.DataFrame({"v": pd.array(texts, dtype="string[pyarrow]")})
schema = T.StructType([T.StructField("v", T.StringType(), True, {"source": "native"})])
scope, name = sys.argv[1:3]
replacement = native.BooleanType if name == "StringType" else native.StringType
if scope == "public":
    sys.modules.pop("pyspark.sql.pandas.types", None)
    setattr(T, name, replacement)
else:
    converter = importlib.import_module("pyspark.sql.pandas.types")
    if scope == "captured":
        importer = importlib.import_module("pyspark.sql._arrow_input")
    setattr(converter, name, replacement)
    if scope == "captured":
        sys.modules.pop("pyspark.sql.pandas.types")
        assert importlib.import_module("pyspark.sql.pandas.types") is not converter
        assert importer.to_arrow_schema.__globals__ is vars(converter)
assert _pandas_input.prepare_string_input(frame, schema) is None
assert not _pandas_input.accepts_pandas_table(frame, schema)
class Factory:
    def mktemp(self, name):
        return Path(tempfile.mkdtemp(prefix=name))
protocol = capture.protocol.__wrapped__(Factory())
tables = []
for row_control in (False, True):
    with pytest.MonkeyPatch.context() as control:
        if row_control:
            capture._force_native_pandas_rows(control)
        with capture._arrow_input_service(protocol) as (spark, state):
            spark.createDataFrame(frame, schema).toArrow()
            assert state["config_requests"] == []
            assert state["uploads"] == 0
            assert json.loads(state["spark_schema"]) == json.loads(schema.json())
            tables.append(state["input_tables"][0])
assert tables[0].column(0).to_pylist() == texts
assert tables[0].equals(tables[1], check_metadata=True)
'''
    result = subprocess.run([sys.executable, "-c", script, scope, class_name, __file__],
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("storage", ["pyarrow", "numpy_semantics"])
@pytest.mark.parametrize("schema_kind", ["explicit", "inferred", "names"])
@pytest.mark.parametrize("instance_hook", [False, True])
def test_arrow_string_custom_arrow_hook_keeps_native_row_values(
    protocol, monkeypatch, instance_hook, schema_kind, storage
):
    import importlib

    _pandas_input = importlib.import_module("pyspark.sql._pandas_input")

    texts = ["original", None] * 5000
    # The first column authenticates this class before the second instance's
    # override is checked; every instance must still be inspected.
    frame = pd.DataFrame({name: _arrow_string_column(texts, storage)
                          for name in ("first", "v")})
    schema = T.StructType([T.StructField(name, T.StringType()) for name in ("first", "v")])
    input_schema = {"explicit": schema, "inferred": None, "names": ["first", "v"]}[schema_kind]
    array = frame.iloc[:, 1].array
    calls = []

    def changed_arrow(*args, **kwargs):
        calls.append("arrow")
        return pa.array(["changed"] * len(texts), type=pa.string())

    monkeypatch.setattr(array if instance_hook else type(array), "__arrow_array__", changed_arrow)
    assert _pandas_input.prepare_string_input(frame, schema) is None
    assert not _pandas_input.accepts_pandas_table(frame, schema)
    assert not _pandas_input.accepts_pandas_frame(frame)
    assert calls == []
    tables = []
    for row_control in (False, True):
        with monkeypatch.context() as control:
            if row_control:
                _force_native_pandas_rows(control)
            with _arrow_input_service(protocol, config_error=True) as (spark, state):
                spark.createDataFrame(frame, input_schema).toArrow()
                tables.append(state["input_tables"][0])
                assert state["config_requests"] == []
                assert state["uploads"] == 0
    assert calls == []
    assert tables[0].to_pydict() == {"first": texts, "v": texts}
    assert tables[0].equals(tables[1], check_metadata=True)


@pytest.mark.parametrize("schema_kind", ["explicit", "inferred", "names"])
@pytest.mark.parametrize("customization", [
    "mixin_hook", "mixin_alias", "partial_alias", "mixin_storage", "mixin_dtype",
])
def test_arrow_string_numpy_semantics_accessor_keeps_native_outcomes(
    protocol, monkeypatch, customization, schema_kind
):
    import importlib
    import json
    import traceback
    from pandas.core.arrays import _arrow_string_mixins as string_mixins, string_arrow
    from pyspark.sql.pandas.types import from_arrow_schema

    helper = importlib.import_module("pyspark.sql._pandas_input")
    texts = ["original", None] * 5000
    frames = [pd.DataFrame({"v": _arrow_string_column(texts, "numpy_semantics")})
              for _ in range(2)]
    for frame in frames:
        frame.columns = pd.Index(["v"], dtype=object)
    schema = T.StructType([T.StructField("v", T.StringType())])
    input_schema = {"explicit": schema, "inferred": None, "names": ["v"]}[schema_kind]
    array_class = type(frames[0].iloc[:, 0].array)
    dynamic_accessor = (helper._raw_class_attribute(array_class, "__getattribute__")[1]
                        is not object.__getattribute__)
    # These warm-ups populate Block.dtype so mixin_dtype reaches class authentication
    # instead of raising during the earlier native-compatible frame.dtypes read.
    for frame in frames:
        assert helper.prepare_string_input(frame, schema) is not None
        assert helper.accepts_pandas_table(frame, schema)
        assert helper.accepts_pandas_frame(frame)
    calls = []

    def changed_arrow(*args, **kwargs):
        calls.append("arrow")
        return pa.array(["changed"] * len(texts), type=pa.string())

    def changed_partial(*args, **kwargs):
        calls.append("partial")
        raise AssertionError("eligibility must not invoke the rebound partial alias")

    def changed_attribute(*args, **kwargs):
        calls.append(customization)
        raise AssertionError("the mixin state callback must not be called")

    if customization == "mixin_hook":
        monkeypatch.setattr(string_mixins.ArrowStringArrayMixin, "__arrow_array__",
                            changed_arrow, raising=False)
    elif customization == "mixin_alias":
        class ChangedMixin:
            __arrow_array__ = changed_arrow

        monkeypatch.setattr(string_arrow, "ArrowStringArrayMixin", ChangedMixin, raising=False)
    elif customization in ("mixin_storage", "mixin_dtype"):
        name = {"mixin_storage": "_storage", "mixin_dtype": "_dtype"}[customization]
        monkeypatch.setattr(string_mixins.ArrowStringArrayMixin, name,
                            changed_attribute, raising=False)
    else:
        monkeypatch.setattr(string_arrow, "partial", changed_partial, raising=False)
    for frame in frames:
        # pandas 2.2 dispatches instance lookups through these mutable aliases.
        # Later layouts use object access and remain unaffected by these changes.
        assert (helper.prepare_string_input(frame, schema) is None) == dynamic_accessor
        assert helper.accepts_pandas_table(frame, schema) == (not dynamic_accessor)
        assert helper.accepts_pandas_frame(frame) == (not dynamic_accessor)
        assert calls == []
    logical_tables = []
    outcomes = []
    for row_control, frame in zip((True, False), frames):
        with monkeypatch.context() as control:
            if row_control:
                _force_native_pandas_rows(control)
            with _arrow_input_service(protocol, config_error=dynamic_accessor) as (spark, state):
                # The mock returns only a row count, not the uploaded data or resolved schema.
                try:
                    reply = spark.createDataFrame(frame, input_schema).toArrow()
                except Exception as error:
                    assert dynamic_accessor and customization in ("mixin_storage", "mixin_dtype")
                    assert not isinstance(error, AssertionError)
                    assert any(
                        trace_frame.f_globals.get("__name__", "").startswith("pandas.core.arrays.")
                        for trace_frame, _ in traceback.walk_tb(error.__traceback__)
                    )
                    assert state["input_tables"] == []
                    assert state["config_requests"] == []
                    assert state["uploads"] == 0
                    outcomes.append((type(error), str(error)))
                    continue
                outcomes.append(None)
                assert reply.to_pydict() == {"rows": [len(texts)]}
                assert len(state["input_tables"]) == 1
                uploaded = state["input_tables"][0]
                assert uploaded.num_rows == len(texts)
                assert uploaded.num_columns == 1
                assert uploaded.column(0).to_pylist() == texts
                assert uploaded.column(0).is_null().to_pylist() == [False, True] * 5000
                assert state["column_names"] == ["v"]
                if row_control or dynamic_accessor or schema_kind == "explicit":
                    expected_wire = pa.schema([pa.field("v", pa.string())])
                    assert json.loads(state["spark_schema"]) == json.loads(schema.json())
                else:
                    # Inherited inferred Arrow conversion sends _n fields without a
                    # declared schema; Spark infers large_string as StringType and toDF
                    # applies the actual captured column names above.
                    expected_wire = pa.schema([pa.field("_0", pa.large_string())])
                    assert state["spark_schema"] is None
                assert uploaded.schema.equals(expected_wire, check_metadata=True)
                if row_control or dynamic_accessor:
                    assert state["config_requests"] == []
                    assert state["uploads"] == 0
                else:
                    assert state["config_requests"]
                # Compare logical uploaded data only after checking the exact wire
                # schema and rename plan. This is not a real server schema assertion.
                renamed = uploaded.rename_columns(state["column_names"])
                assert from_arrow_schema(renamed.schema) == schema
                logical_tables.append(renamed.cast(pa.schema([pa.field("v", pa.string())])))
    assert calls == []
    assert outcomes[0] == outcomes[1]
    if dynamic_accessor and customization in ("mixin_storage", "mixin_dtype"):
        assert outcomes[0] is not None
    if logical_tables:
        assert len(logical_tables) == 2
        assert logical_tables[0].equals(logical_tables[1], check_metadata=True)


@pytest.mark.parametrize("route", ["specialized", "generic", "inferred"])
def test_arrow_string_guards_recheck_hooks_between_columns(monkeypatch, route):
    import importlib

    _pandas_input = importlib.import_module("pyspark.sql._pandas_input")
    texts = ["original", None] * 5000
    frame = pd.DataFrame({name: pd.array(texts, dtype="string[pyarrow]")
                          for name in ("first", "second")})
    schema = T.StructType([T.StructField(name, T.StringType())
                           for name in ("first", "second")])
    array_class = type(frame.iloc[:, 0].array)
    original_iloc = pd.DataFrame.iloc
    lookups = []
    calls = []

    def customized(*args, **kwargs):
        calls.append("__getitem__")
        raise AssertionError("eligibility must not run the customized row hook")

    class RebindingIndexer:
        def __init__(self, indexer):
            self.indexer = indexer

        def __getitem__(self, key):
            column = self.indexer[key]
            lookups.append(key[1])
            if key[1] == 1:
                monkeypatch.setattr(array_class, "__getitem__", customized)
            return column

    def iloc(self):
        indexer = original_iloc.fget(self)
        return RebindingIndexer(indexer) if self is frame else indexer

    # Both guards allow this accessor today. It returns ordinary Series while
    # rebinding a hook after the first column, invalidating any per-class cache.
    monkeypatch.setattr(pd.DataFrame, "iloc", property(iloc))
    if route == "specialized":
        assert _pandas_input.prepare_string_input(frame, schema) is None
    elif route == "generic":
        assert not _pandas_input.accepts_pandas_table(frame, schema)
    else:
        assert not _pandas_input.accepts_pandas_frame(frame)
    assert lookups == [0, 1]
    assert calls == []


def test_arrow_string_subclass_guard_does_not_hash_custom_metaclass():
    from pyspark.sql._pandas_input import _arrow_string_class_is_canonical

    calls = []

    class CustomMeta(type(pd.arrays.ArrowStringArray)):
        def __hash__(cls):
            calls.append("class hash")
            raise AssertionError("declining a subclass must not run its hash hook")

    class Custom(pd.arrays.ArrowStringArray, metaclass=CustomMeta):
        pass

    assert not _arrow_string_class_is_canonical(Custom)
    assert calls == []


def test_arrow_string_generic_subclass_guard_does_not_read_custom_mro():
    from pyspark.sql._pandas_input import accepts_pandas_table

    calls = []
    armed = False

    class CustomMeta(type(pd.arrays.ArrowStringArray)):
        @property
        def __mro__(cls):
            if armed:
                calls.append("class mro")
                raise AssertionError("declining a subclass must not read its custom MRO")
            return type.__dict__["__mro__"].__get__(cls, type(cls))

    class Custom(pd.arrays.ArrowStringArray, metaclass=CustomMeta):
        pass

    # Finish ordinary pandas construction before arming the metaclass callback.
    frame = pd.DataFrame({"v": Custom(pa.chunked_array([pa.array(["a", None])]))})
    assert type(frame.iloc[:, 0].array) is Custom
    schema = T.StructType([T.StructField("v", T.StringType())])
    armed = True
    assert not accepts_pandas_table(frame, schema)
    assert calls == []


def _arrow_string_allocator_process(case, schema_kind="explicit", storage="pyarrow"):
    """Keep CPython allocator-slot changes out of the main pytest interpreter."""
    import signal
    import sys

    script = r'''
import importlib.util
import sys
import tempfile
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location("arrow_string_allocator_capture", sys.argv[4])
capture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capture
spec.loader.exec_module(capture)
case, schema_kind, storage = sys.argv[1:4]
with pytest.MonkeyPatch.context() as control:
    if case == "row":
        with tempfile.TemporaryDirectory(prefix="arrow-string-allocator-") as directory:
            class Factory:
                def mktemp(self, name):
                    path = Path(directory) / name
                    path.mkdir()
                    return path
            protocol = capture.protocol.__wrapped__(Factory())
            capture._check_arrow_string_custom_row_hook(
                protocol, control, "__new__", schema_kind, storage)
    else:
        assert case == "descriptor"
        capture._check_arrow_string_class_descriptor(control, "__new__")
'''
    owned_group = os.name == "posix"
    process = subprocess.Popen(
        [sys.executable, "-c", script, case, schema_kind, storage, str(Path(__file__).resolve())],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=owned_group,
    )

    def terminate_child():
        try:
            if owned_group:
                # protoc inherits this session, so a timeout also cleans up that child.
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass

    try:
        stdout, stderr = process.communicate(timeout=60)
    except subprocess.TimeoutExpired:
        terminate_child()
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            stdout, stderr = "", ""
        pytest.fail("Arrow-string allocator child timed out\n" + stdout + stderr)
    except BaseException:
        terminate_child()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        raise
    assert process.returncode == 0, stdout + stderr
    # Guard against accidentally reintroducing in-process allocator mutation.
    healthy = pd.array(["original", None], dtype="string[pyarrow]")
    assert healthy.__arrow_array__().to_pylist() == ["original", None]


@pytest.mark.parametrize("storage", ["pyarrow", "numpy_semantics"])
@pytest.mark.parametrize("schema_kind", ["explicit", "inferred", "names"])
@pytest.mark.parametrize("method", [
    "astype", "isna", "to_numpy", "__getitem__", "dtype", "na_value", "_hasna", "__init__",
    "_from_pyarrow_array", "__new__",
])
def test_arrow_string_custom_row_hooks_keep_native_errors(
    protocol, monkeypatch, method, schema_kind, storage
):
    if method == "__new__":
        # Other hooks use this shared fixture signature; the child needs its own protocol.
        _arrow_string_allocator_process("row", schema_kind, storage)
    else:
        _check_arrow_string_custom_row_hook(protocol, monkeypatch, method, schema_kind, storage)


def _check_arrow_string_custom_row_hook(
    protocol, monkeypatch, method, schema_kind, storage
):
    import importlib

    _pandas_input = importlib.import_module("pyspark.sql._pandas_input")

    texts = ["original", None] * 5000
    frames = []
    for _ in range(2):
        frame = pd.DataFrame({"v": _arrow_string_column(texts, storage)})
        # Object labels isolate the additional construction/_hasna data hooks.
        # The other Arrow labels exercise earlier checks when they share the data
        # class (pyarrow storage); a separate NumPy-semantics class is checked later.
        frame.columns = pd.Index(["v"], dtype=object) if method in (
            "_hasna", "__init__", "_from_pyarrow_array", "__new__",
        ) else pd.Index(
            pd.array(["v"], dtype="string[pyarrow]"))
        frames.append(frame)
    schema = T.StructType([T.StructField("v", T.StringType())])
    input_schema = {"explicit": schema, "inferred": None, "names": ["v"]}[schema_kind]
    array_class = type(frames[0].iloc[:, 0].array)
    calls = []
    # Warm all guards before rebinding: class authentication must not be cached
    # across calls, even when this exact class was previously accepted. This also
    # populates pandas' Block.dtype cache, so a customized data dtype reaches the
    # class check; the separate cold-dtype test covers its earlier native error.
    for frame in frames:
        assert _pandas_input.prepare_string_input(frame, schema) is not None
        assert _pandas_input.accepts_pandas_table(frame, schema)
        assert _pandas_input.accepts_pandas_frame(frame)

    class HookError(Exception):
        pass

    def customized(*args, **kwargs):
        if method == "na_value":
            # This property receives a StringDtype, including during native label iteration.
            assert type(args[0]) is pd.StringDtype
        elif method == "__new__":
            # pandas may pass only the class; no initialized receiver is available.
            assert args[0] is array_class
        elif method in ("__init__", "_from_pyarrow_array"):
            # __init__ receives an uninitialized object; inspect the subset instead.
            assert type(args[0]) is array_class
            assert len(args) == 2
            values = args[1]
            assert type(values) is pa.ChunkedArray
            assert values.null_count == 0
            assert values.to_pylist() == ["original"] * 5000
            assert set(kwargs) <= {"dtype"}
            if kwargs:
                assert type(kwargs["dtype"]) is pd.StringDtype
        else:
            assert len(args[0]) == 10000  # Labels have length one; guards must not cast them.
        calls.append(method)
        raise HookError(f"{method} sentinel")

    class FactoryHook:
        def __get__(self, receiver, owner):
            assert receiver is not None, "eligibility must not bind the factory descriptor"
            assert owner is array_class
            return customized.__get__(receiver, owner)

    if method == "na_value":
        monkeypatch.setattr(pd.StringDtype, method, property(customized))
    elif method == "_from_pyarrow_array":
        # An absent helper on older pandas is also a conservative decline.
        monkeypatch.setattr(array_class, method, FactoryHook(), raising=False)
    elif method == "__new__":
        monkeypatch.setattr(array_class, method, staticmethod(customized))
    elif method in ("dtype", "_hasna"):
        monkeypatch.setattr(array_class, method, property(customized))
    else:
        monkeypatch.setattr(array_class, method, customized)
    for frame in frames:
        assert _pandas_input.prepare_string_input(frame, schema) is None
        assert not _pandas_input.accepts_pandas_table(frame, schema)
        assert not _pandas_input.accepts_pandas_frame(frame)
        assert calls == []

    outcomes = []
    tables = []
    for row_control, frame in zip((True, False), frames):
        calls.clear()
        with monkeypatch.context() as control:
            if row_control:
                _force_native_pandas_rows(control)
            with _arrow_input_service(protocol, config_error=True) as (spark, state):
                try:
                    result = spark.createDataFrame(frame, input_schema)
                    result.toArrow()
                except HookError as error:
                    assert type(error) is HookError
                    assert str(error) == f"{method} sentinel"
                    if method in ("__init__", "_from_pyarrow_array", "__new__"):
                        assert calls
                    else:
                        assert calls == [method]
                    outcome = (type(error), str(error))
                    assert state["input_tables"] == []
                else:
                    # pandas versions use different subset-construction helpers.
                    assert method in ("__init__", "_from_pyarrow_array", "__new__")
                    assert len(state["input_tables"]) == 1
                    table = state["input_tables"][0]
                    assert table.column(0).to_pylist() == texts
                    tables.append(table)
                    outcome = None
                outcomes.append((outcome, tuple(calls)))
                assert state["config_requests"] == []
                assert state["uploads"] == 0
    assert outcomes[0] == outcomes[1]
    if tables:
        assert len(tables) == 2
        assert tables[0].equals(tables[1], check_metadata=True)


@pytest.mark.parametrize("method", [
    "__arrow_array__", "__array__", "__getitem__", "isna", "to_numpy", "__len__", "astype",
    "__init__", "_from_pyarrow_array", "__new__", "__getattribute__", "dtype", "_hasna", "na_value",
])
def test_arrow_string_class_descriptors_decline_without_binding(monkeypatch, method):
    if method == "__new__":
        _arrow_string_allocator_process("descriptor")
    else:
        _check_arrow_string_class_descriptor(monkeypatch, method)


def _check_arrow_string_class_descriptor(monkeypatch, method):
    import importlib

    _pandas_input = importlib.import_module("pyspark.sql._pandas_input")
    array = pd.array(["original", None], dtype="string[pyarrow]")
    array_class = type(array)
    assert _pandas_input._arrow_string_class_is_canonical(array_class)
    assert _pandas_input._has_canonical_arrow_string_array(array)
    calls = []

    class HookError(Exception):
        pass

    class CustomizedDescriptor:
        def __get__(self, receiver, owner):
            calls.append((receiver, owner))
            raise HookError(f"{method} descriptor sentinel")

    owner = pd.StringDtype if method == "na_value" else array_class
    monkeypatch.setattr(owner, method, CustomizedDescriptor(), raising=False)
    # These concrete-class checks inspect raw dictionaries. Earlier frame.dtypes
    # reads are covered separately because they intentionally keep native ordering.
    assert not _pandas_input._arrow_string_class_is_canonical(array_class)
    assert not _pandas_input._has_canonical_arrow_string_array(array)
    assert calls == []


@pytest.mark.parametrize("method", [
    "__getattribute__", "tolist", "__iter__", "fillna", "to_numpy", "isna", "_hasna",
    "dtype", "na_value",
])
def test_arrow_string_label_descriptors_decline_without_binding(monkeypatch, method):
    import importlib

    _pandas_input = importlib.import_module("pyspark.sql._pandas_input")
    # Concrete object data isolates the Arrow-string class used by the labels.
    frame = pd.DataFrame({"v": pd.Series(["original", None] * 5000, dtype=object)})
    frame.columns = pd.Index(pd.array(["v"], dtype="string[pyarrow]"))
    array_class = type(frame.columns._values)
    schema = T.StructType([T.StructField("v", T.StringType())])
    assert _pandas_input.prepare_string_input(frame, schema) is not None
    assert _pandas_input.accepts_pandas_table(frame, schema)
    assert _pandas_input.accepts_pandas_frame(frame)
    calls = []

    class HookError(Exception):
        pass

    class CustomizedDescriptor:
        def __get__(self, receiver, owner):
            calls.append((receiver, owner))
            raise HookError(f"label {method} descriptor sentinel")

    owner = pd.StringDtype if method == "na_value" else array_class
    monkeypatch.setattr(owner, method, CustomizedDescriptor())
    assert _pandas_input.prepare_string_input(frame, schema) is None
    assert not _pandas_input.accepts_pandas_table(frame, schema)
    assert not _pandas_input.accepts_pandas_frame(frame)
    assert calls == []


@pytest.mark.parametrize("storage_kind", [
    "plain_array", "nonstring", "invalid_utf8", "bad_storage", "spoofed_array",
])
def test_arrow_string_storage_normalization_and_fallback(monkeypatch, storage_kind):
    import importlib
    import struct

    _pandas_input = importlib.import_module("pyspark.sql._pandas_input")
    texts = ["é", None]
    frame = pd.DataFrame({"v": pd.array(texts, dtype="string[pyarrow]")})
    schema = T.StructType([T.StructField("v", T.StringType())])
    # Capture pandas metadata before intentionally replacing its Arrow storage.
    dtypes = frame.dtypes
    array = frame.iloc[:, 0].array
    if storage_kind == "plain_array":
        storage = pa.array(texts, type=pa.string())
        assert isinstance(storage, pa.Array)
    elif storage_kind == "nonstring":
        storage = pa.chunked_array([pa.array([1, None], type=pa.int64())])
    elif storage_kind == "invalid_utf8":
        invalid = pa.Array.from_buffers(pa.string(), 2, [
            None, pa.py_buffer(struct.pack("<iii", 0, 5, 6)), pa.py_buffer(b"valid\xff"),
        ])
        storage = pa.chunked_array([invalid])
        with pytest.raises(pa.ArrowInvalid):
            storage.validate(full=True)
    elif storage_kind == "bad_storage":
        class UnsupportedStorage:
            # pandas' column boxing reads length before the storage-type guard.
            def __len__(self):
                return len(texts)

        storage = UnsupportedStorage()
        assert not isinstance(storage, (pa.Array, pa.ChunkedArray))
    else:
        class SpoofedArray:
            @property
            def __class__(self):
                return pa.Array

            def __len__(self):
                return len(texts)

        storage = SpoofedArray()
        assert isinstance(storage, pa.Array)
        assert not issubclass(type(storage), pa.Array)
    monkeypatch.setattr(array, "_pa_array", storage)
    assert object.__getattribute__(frame.iloc[:, 0].array, "_pa_array") is storage
    result = _pandas_input._arrow_string_table(frame, schema, dtypes)
    if storage_kind == "plain_array":
        assert type(result) is pa.Table
        result.validate(full=True)
        assert result.schema.names == ["v"]
        assert result.column(0).type == pa.string()
        assert result.column(0).to_pylist() == texts
    else:
        assert result is None


@pytest.mark.parametrize("error_kind", ["arrow", "overflow"])
def test_arrow_string_simulated_cast_failures_decline(monkeypatch, error_kind):
    """Exercise exception handling without allocating a greater-than-2-GiB column."""
    import importlib
    import sys
    from types import ModuleType

    _pandas_input = importlib.import_module("pyspark.sql._pandas_input")
    frame = pd.DataFrame({"v": pd.array(["original", None], dtype="string[pyarrow]")})
    schema = T.StructType([T.StructField("v", T.StringType())])
    dtypes = frame.dtypes
    array = frame.iloc[:, 0].array
    values = array.__arrow_array__()
    error_class = pa.ArrowInvalid if error_kind == "arrow" else OverflowError
    calls = []

    class SimulatedCastFailure:
        type = values.type

        def __len__(self):
            return len(values)

        def validate(self, *, full):
            calls.append(("validate", full))
            values.validate(full=full)

        def cast(self, target):
            assert target == pa.string()
            calls.append(("cast", target))
            raise error_class("simulated Arrow-string cast failure")

    # PyArrow extension types are immutable. This local module proxy substitutes
    # the exact recognized storage class only for this simulated cast branch;
    # no real ChunkedArray subclass or capacity behavior is being tested.
    proxy = ModuleType("pyarrow")
    vars(proxy).update(vars(pa))
    proxy.ChunkedArray = SimulatedCastFailure
    with monkeypatch.context() as simulated:
        simulated.setattr(array, "_pa_array", SimulatedCastFailure())
        simulated.setitem(sys.modules, "pyarrow", proxy)
        assert _pandas_input._arrow_string_table(frame, schema, dtypes) is None
    assert calls == [("validate", True), ("cast", pa.string())]


def test_arrow_string_object_labels_dtype_error_matches_native_rows(protocol, monkeypatch):
    import importlib

    _pandas_input = importlib.import_module("pyspark.sql._pandas_input")
    frames = []
    for _ in range(3):
        frame = pd.DataFrame({"v": pd.array(["original", None] * 5000,
                                          dtype="string[pyarrow]")})
        frame.columns = pd.Index(["v"], dtype=object)
        frames.append(frame)
    schema = T.StructType([T.StructField("v", T.StringType())])
    array_class = type(frames[0].iloc[:, 0].array)
    calls = []

    class HookError(Exception):
        pass

    def customized(self):
        assert type(self) is array_class
        assert len(self) == 10000
        calls.append("dtype")
        raise HookError("dtype sentinel")

    monkeypatch.setattr(array_class, "dtype", property(customized))
    # The generic route can authenticate the data class before reading its dtype.
    assert not _pandas_input.accepts_pandas_table(frames[0], schema)
    assert calls == []
    errors = []
    with pytest.raises(HookError, match="^dtype sentinel$") as error:
        _pandas_input.prepare_string_input(frames[0], schema)
    errors.append((type(error.value), str(error.value)))
    assert calls == ["dtype"]
    for row_control, frame in zip((False, True), frames[1:]):
        calls.clear()
        with monkeypatch.context() as control:
            if row_control:
                _force_native_pandas_rows(control)
            with _arrow_input_service(protocol, config_error=True) as (spark, state):
                with pytest.raises(HookError, match="^dtype sentinel$") as error:
                    spark.createDataFrame(frame, schema)
                errors.append((type(error.value), str(error.value)))
                assert state["config_requests"] == []
                assert state["uploads"] == 0
                assert state["input_tables"] == []
        assert calls == ["dtype"]
    assert errors[0] == errors[1] == errors[2]


def test_arrow_string_custom_label_astype_keeps_native_row_values(protocol, monkeypatch):
    import importlib

    _pandas_input = importlib.import_module("pyspark.sql._pandas_input")

    texts = ["original", None] * 5000
    frame = pd.DataFrame({"v": pd.array(texts, dtype="string[pyarrow]")})
    frame.columns = pd.Index(pd.array(["v"], dtype="string[pyarrow]"))
    schema = T.StructType([T.StructField("v", T.StringType())])
    calls = []

    def customized(*args, **kwargs):
        calls.append("label astype")
        raise AssertionError("native row conversion does not cast column labels")

    monkeypatch.setattr(frame.columns._values, "astype", customized)
    assert _pandas_input.prepare_string_input(frame, schema) is None
    assert not _pandas_input.accepts_pandas_table(frame, schema)
    assert calls == []
    tables = []
    for row_control in (False, True):
        with monkeypatch.context() as control:
            if row_control:
                _force_native_pandas_rows(control)
            with _arrow_input_service(protocol, config_error=True) as (spark, state):
                spark.createDataFrame(frame, schema).toArrow()
                tables.append(state["input_tables"][0])
                assert state["config_requests"] == []
                assert state["uploads"] == 0
    assert calls == []
    assert tables[0].column(0).to_pylist() == texts
    assert tables[0].equals(tables[1], check_metadata=True)


def test_arrow_string_null_labels_decline_without_getitem_callback(protocol, monkeypatch):
    import importlib

    from pandas.core.arrays.arrow.array import ArrowExtensionArray

    _pandas_input = importlib.import_module("pyspark.sql._pandas_input")
    # Index uniqueness can cast null labels and invoke indexed masking. Native
    # normalization may also slice labels while aligning DataFrame.where; guards
    # must decline without an early callback and preserve that native outcome.
    expected = {"a": ["first"] * 10000, "b": ["second"] * 10000}
    frames = []
    for _ in range(2):
        frame = pd.DataFrame({name: pd.array(values, dtype="string[pyarrow]")
                              for name, values in expected.items()})
        frame.columns = pd.Index(pd.array(["a", None], dtype="string[pyarrow]"))
        frames.append(frame)
    schema = T.StructType([T.StructField(name, T.StringType()) for name in expected])
    calls = []

    class HookError(Exception):
        pass

    def customized(*args, **kwargs):
        calls.append("label getitem")
        raise HookError("label getitem sentinel")

    monkeypatch.setattr(ArrowExtensionArray, "__getitem__", customized)
    for frame in frames:
        assert _pandas_input.prepare_string_input(frame, schema) is None
        assert not _pandas_input.accepts_pandas_table(frame, schema)
        assert calls == []

    outcomes = []
    tables = []
    # Observe native behavior first, using a fresh equal frame for each route.
    for row_control, frame in zip((True, False), frames):
        calls.clear()
        with monkeypatch.context() as control:
            if row_control:
                _force_native_pandas_rows(control)
            with _arrow_input_service(protocol, config_error=True) as (spark, state):
                try:
                    spark.createDataFrame(frame, schema).toArrow()
                except HookError as error:
                    assert type(error) is HookError
                    assert str(error) == "label getitem sentinel"
                    assert calls
                    outcome = (type(error), str(error))
                    assert state["input_tables"] == []
                else:
                    assert len(state["input_tables"]) == 1
                    table = state["input_tables"][0]
                    assert table.to_pydict() == expected
                    tables.append(table)
                    outcome = None
                outcomes.append((outcome, tuple(calls)))
                assert state["config_requests"] == []
                assert state["uploads"] == 0
    assert outcomes[0] == outcomes[1]
    if tables:
        assert len(tables) == 2
        assert tables[0].equals(tables[1], check_metadata=True)


@pytest.mark.parametrize("label_storage", ["object", "pyarrow"])
@pytest.mark.parametrize("method", ["astype", "__array__", "__getitem__", "__init__"])
@pytest.mark.parametrize("storage", ["pyarrow", "numpy_semantics"])
def test_arrow_string_super_row_hooks_keep_native_errors(
    protocol, monkeypatch, label_storage, method, storage
):
    import importlib

    from pandas.core.arrays.arrow.array import ArrowExtensionArray

    _pandas_input = importlib.import_module("pyspark.sql._pandas_input")
    frame = pd.DataFrame({"v": _arrow_string_column(["original", None] * 5000, storage)})
    frame.columns = pd.Index(["v"], dtype=object) if label_storage == "object" else pd.Index(
        pd.array(["v"], dtype="string[pyarrow]"))
    schema = T.StructType([T.StructField("v", T.StringType())])
    calls = []
    assert _pandas_input.prepare_string_input(frame, schema) is not None
    assert _pandas_input.accepts_pandas_table(frame, schema)
    assert _pandas_input.accepts_pandas_frame(frame)

    class HookError(Exception):
        pass

    def customized(self, *args, **kwargs):
        if method == "__init__":
            # The receiver has no storage yet. Null-subarray construction passes
            # the 5,000 non-null data values; labels have only one value.
            values = args[0]
            assert values.to_pylist() == ["original"] * 5000
        else:
            assert len(self) == 10000  # Preflight must not cast the length-one labels.
        calls.append(method)
        raise HookError(f"super {method} sentinel")

    monkeypatch.setattr(ArrowExtensionArray, method, customized)
    assert _pandas_input.prepare_string_input(frame, schema) is None
    assert not _pandas_input.accepts_pandas_table(frame, schema)
    assert not _pandas_input.accepts_pandas_frame(frame)
    assert calls == []
    outcomes = []
    tables = []
    # The native control determines whether this pandas layout calls __init__.
    # pandas 3.1's null-subarray factory instead allocates through __new__.
    for row_control in (True, False):
        calls.clear()
        with monkeypatch.context() as control:
            if row_control:
                _force_native_pandas_rows(control)
            with _arrow_input_service(protocol, config_error=True) as (spark, state):
                try:
                    spark.createDataFrame(frame, schema).toArrow()
                except HookError as error:
                    assert type(error) is HookError
                    assert str(error) == f"super {method} sentinel"
                    assert calls == [method]
                    assert state["input_tables"] == []
                    outcome = (type(error), str(error))
                else:
                    assert method == "__init__"
                    assert calls == []
                    assert len(state["input_tables"]) == 1
                    table = state["input_tables"][0]
                    table.validate(full=True)
                    assert table.to_pydict() == {"v": ["original", None] * 5000}
                    assert table.schema.names == ["v"]
                    assert table.schema.field(0).type == pa.string()
                    assert table.schema.field(0).nullable
                    tables.append(table)
                    outcome = None
                outcomes.append((outcome, tuple(calls)))
                assert state["config_requests"] == []
                assert state["uploads"] == 0
    assert outcomes[0] == outcomes[1]
    if tables:
        assert len(tables) == 2
        assert tables[0].equals(tables[1], check_metadata=True)


@pytest.mark.parametrize("reason", [
    "string_subclass", "frame_subclass", "nan", "pd_na", "nat", "string_dtype",
    "nonnullable_null", "inferred", "names", "ddl", "small", "collation", "cast", "mixed",
])
def test_string_public_fallback_keeps_row_semantics(protocol, monkeypatch, reason):
    calls = []

    class Text(str):
        def __str__(self):
            calls.append("str")
            return "changed"

    class Frame(pd.DataFrame):
        pass

    values = ["plain"] * (9999 if reason == "small" else 10000)
    if reason in ("nan", "pd_na", "nat", "nonnullable_null", "mixed"):
        values[-1] = {"nan": float("nan"), "pd_na": pd.NA, "nat": pd.NaT,
                      "nonnullable_null": None, "mixed": 42}[reason]
    elif reason == "string_subclass":
        values[-1] = Text("plain")
    frame_type = Frame if reason == "frame_subclass" else pd.DataFrame
    frame = frame_type({"text": values}, dtype="string" if reason == "string_dtype" else object)
    schema = T.StructType([T.StructField("text", T.StringType(), reason != "nonnullable_null")])
    if reason == "inferred":
        schema = None
    elif reason == "names":
        schema = ["renamed"]
    elif reason == "ddl":
        schema = "text STRING"
    elif reason == "collation":
        schema = T.StructType([T.StructField("text", T.StringType("UTF8_LCASE"))])
    elif reason == "cast":
        schema = T.StructType([T.StructField("text", T.LongType())])
    astype = pd.DataFrame.astype

    def record_rows(self, *args, **kwargs):
        calls.append("astype")
        return astype(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "astype", record_rows)
    with _arrow_input_service(protocol) as (spark, state):
        if reason in ("nonnullable_null", "cast", "mixed"):
            from pyspark.errors import PySparkRuntimeError

            message = ("Column 'text' is declared as non-nullable but contains null values"
                       if reason == "nonnullable_null" else "Type mismatch in row data")
            with pytest.raises(PySparkRuntimeError, match=message):
                spark.createDataFrame(frame, schema)
            assert calls == ["astype"]
            assert state["config_requests"] == []
            assert state["relations"] == []
            return
        spark.createDataFrame(frame, schema).toArrow()
        assert calls == ["astype"]  # Native extraction keeps str.__str__ overrides uncalled.
        table = state["input_tables"][0]
        assert table.schema.names == ["renamed" if reason == "names" else "text"]
        expected = ["plain"] * len(values)
        if reason in ("nan", "pd_na", "nat"):
            expected[-1] = None
        assert table.column(0).to_pylist() == expected
        assert state["config_requests"] == []


@pytest.mark.parametrize("surrogate", ["\ud800", "\udfff"])
def test_string_surrogate_fallback_precedes_importer_configuration(protocol, monkeypatch, surrogate):
    values = ["plain"] * 10000
    values[-1] = surrogate
    frame = pd.DataFrame({"text": values}, dtype=object)
    schema = T.StructType([T.StructField("text", T.StringType())])
    astype = pd.DataFrame.astype
    calls = []

    def record_rows(self, *args, **kwargs):
        calls.append("astype")
        return astype(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "astype", record_rows)
    with _arrow_input_service(protocol) as (spark, state):
        state["config_error"] = grpc.StatusCode.INVALID_ARGUMENT
        with pytest.raises(TypeError, match="unsupported value type for createDataFrame: str"):
            spark.createDataFrame(frame, schema)
        assert calls == ["astype"]
        assert state["config_requests"] == []
        assert state["relations"] == []
        assert state["uploads"] == 0


@pytest.mark.parametrize("class_name,attribute", [
    ("StructType", "names"), ("StructType", "fields"), ("StructType", "__iter__"),
    ("StructField", "name"), ("StructField", "dataType"), ("StructField", "nullable"),
    ("StructField", "metadata"), ("StringType", "isUTF8BinaryCollation"),
    ("StructType", "__getattribute__"), ("StructField", "__getattribute__"),
    ("StringType", "__getattribute__"), ("StructType", "__class__"),
    ("StructField", "__class__"), ("StringType", "__class__"),
    ("StructType", "alias"), ("StructField", "alias"), ("StringType", "alias"),
])
@pytest.mark.parametrize("before_import", [False, True])
def test_string_custom_schema_descriptors_keep_native_row_path(
    protocol, class_name, attribute, before_import
):
    import sys

    script = r'''
import sys
import tempfile
from pathlib import Path
import pandas as pd
from pyspark.sql import types as T

import importlib.util
spec = importlib.util.spec_from_file_location("string_schema_capture", sys.argv[4])
capture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capture
spec.loader.exec_module(capture)
_arrow_input_service, protocol = capture._arrow_input_service, capture.protocol
frame = pd.DataFrame({"text": ["plain"] * 10000}, dtype=object)
schema = T.StructType([T.StructField("text", T.StringType(), True, {"source": "native"})])
class_name, attribute, before_import = sys.argv[1:4]
if before_import == "True":
    # The capture module imports prepare_int64_table; force the public call to
    # import a new helper after the class or public alias has been customized.
    sys.modules.pop("pyspark.sql._pandas_input", None)
assert ("pyspark.sql._pandas_input" in sys.modules) == (before_import == "False")
cls = getattr(T, class_name)
if attribute == "alias":
    setattr(T, class_name, object)
else:
    if attribute == "__getattribute__":
        replacement = lambda self, name: object.__getattribute__(self, name)
    elif attribute == "__iter__":
        replacement = lambda self: iter([T.StructField("changed", T.StringType())])
    elif attribute == "__class__":
        attribute = "__getattribute__"
        replacement = lambda self, name: object if name == "__class__" else (
            object.__getattribute__(self, name))
    elif attribute == "isUTF8BinaryCollation":
        replacement = lambda self: False
    else:
        value = {"names": ["changed"], "fields": [], "name": "changed",
                 "dataType": T.StringType(), "nullable": False, "metadata": {"changed": True}}[attribute]
        replacement = property(lambda self: value)
    setattr(cls, attribute, replacement)
class Factory:
    def mktemp(self, name):
        return Path(tempfile.mkdtemp(prefix=name))
with _arrow_input_service(protocol.__wrapped__(Factory())) as (spark, state):
    spark.createDataFrame(frame, schema).toArrow()
    if attribute == "isUTF8BinaryCollation":
        # Official converts this explicit schema with Arrow, after reading the configuration;
        # it never reads this attribute, so the result is the same as the row path's.
        assert state["config_requests"][0][0] == "get", state["config_requests"]
    else:
        assert state["config_requests"] == [], state["config_requests"]
    table = state["input_tables"][0]
    assert table.schema.names == ["text"], table.schema
    assert table.column(0).to_pylist() == ["plain"] * 10000
'''
    result = subprocess.run([sys.executable, "-c", script, class_name, attribute,
                             str(before_import), __file__],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr



def test_string_rebound_pandas_alias_keeps_subclass_fallback(monkeypatch):
    from pyspark.sql._pandas_input import prepare_string_input

    class Frame(pd.DataFrame):
        pass

    frame = Frame({"text": ["plain"] * 10000}, dtype=object)
    schema = T.StructType([T.StructField("text", T.StringType())])
    monkeypatch.setattr(pd, "DataFrame", Frame)
    assert prepare_string_input(frame, schema) is None



@pytest.mark.parametrize("dtype_mode", ["equal", "class_spoof", "exact_numpy"])
def test_string_custom_extension_dtype_cannot_impersonate_numpy_object(protocol, dtype_mode):
    from pandas.api.extensions import ExtensionArray, ExtensionDtype, take
    from pyspark.sql._pandas_input import prepare_string_input

    arrow_calls = []

    class ObjectEqualDtype(ExtensionDtype):
        name = "object_equal_string"
        type = str
        kind = "O"
        na_value = None

        @classmethod
        def construct_array_type(cls):
            return DifferentArrowArray

        def __eq__(self, other):
            return (isinstance(other, ObjectEqualDtype)
                    or isinstance(other, np.dtype) and other.kind == "O")

        def __hash__(self):
            return hash(self.name)

    class DifferentArrowArray(ExtensionArray):
        def __init__(self, values):
            self._values = np.asarray(values, dtype=object)

        @property
        def dtype(self):
            return ObjectEqualDtype()

        @property
        def nbytes(self):
            return self._values.nbytes

        def __len__(self):
            return len(self._values)

        def __getitem__(self, key):
            result = self._values[key]
            return type(self)(result) if isinstance(result, np.ndarray) else result

        def isna(self):
            return pd.isna(self._values)

        def take(self, indices, allow_fill=False, fill_value=None):
            return type(self)(take(self._values, indices, allow_fill=allow_fill,
                                   fill_value=fill_value))

        def copy(self):
            return type(self)(self._values.copy())

        @classmethod
        def _from_sequence(cls, scalars, *, dtype=None, copy=False):
            values = np.asarray(scalars, dtype=object)
            return cls(values.copy() if copy else values)

        def to_numpy(self, dtype=None, copy=False, na_value=None):
            if dtype is not None:
                return self._values.astype(dtype, copy=copy)
            return self._values.copy() if copy else self._values

        def __arrow_array__(self, type=None):
            arrow_calls.append("arrow")
            # A scan of to_numpy() cannot establish the values this hook emits.
            return pa.array(["different"] * len(self), type=type)

    array = DifferentArrowArray(["plain"] * 10000)
    frame = pd.DataFrame({"text": pd.Series(array, copy=False)})
    schema = T.StructType([T.StructField("text", T.StringType())])
    if dtype_mode == "exact_numpy":
        # An ExtensionArray may change its advertised dtype after construction.
        # The dtype and NumPy scan both pass while its Arrow hook changes cells.
        DifferentArrowArray.dtype = property(lambda self: np.dtype("object"))
        DifferentArrowArray.__array__ = lambda self, dtype=None, copy=None: self._values
    elif dtype_mode == "class_spoof":
        def spoof_class(self, name):
            if name == "__class__":
                return type(np.dtype("object"))
            return object.__getattribute__(self, name)

        ObjectEqualDtype.__getattribute__ = spoof_class
    dtype = frame.dtypes.iloc[0]
    assert type(frame) is pd.DataFrame
    assert type(frame.iloc[:, 0].array) is DifferentArrowArray
    assert type(dtype) is (type(np.dtype("object")) if dtype_mode == "exact_numpy"
                           else ObjectEqualDtype)
    assert isinstance(dtype, np.dtype) is (dtype_mode != "equal")
    assert dtype == np.dtype("object")
    assert prepare_string_input(frame, schema) is None
    with _arrow_input_service(protocol) as (spark, state):
        if dtype_mode == "exact_numpy":
            # This deliberately inconsistent EA already fails pandas' row-path
            # astype(object). Preserve that error before importer configuration.
            with pytest.raises(TypeError, match="cannot set astype"):
                spark.createDataFrame(frame, schema)
            assert state["input_tables"] == []
        else:
            spark.createDataFrame(frame, schema).toArrow()
            assert state["input_tables"][0].column(0).to_pylist() == ["plain"] * 10000
        assert state["config_requests"] == []
    assert arrow_calls == []



@pytest.mark.parametrize("method", ["astype", "notna"])
def test_string_instance_conversion_hooks_keep_row_fallback(protocol, method):
    frame = pd.DataFrame({"text": ["plain"] * 10000}, dtype=object)
    schema = T.StructType([T.StructField("text", T.StringType())])
    original = getattr(frame, method)
    calls = []
    expected = ["plain"] * 10000
    if method == "astype":
        expected[0] = "changed"

        def customized(*args, **kwargs):
            calls.append(method)
            result = original(*args, **kwargs)
            result.iat[0, 0] = "changed"
            return result
    else:
        expected[0] = None

        def customized(*args, **kwargs):
            calls.append(method)
            result = original(*args, **kwargs)
            result.iat[0, 0] = False
            return result
    setattr(frame, method, customized)
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert calls == [method]
        assert state["config_requests"] == []
        assert state["uploads"] == 0
        assert state["input_tables"][0].column(0).to_pylist() == expected



@pytest.mark.parametrize("method", ["astype", "notna", "where"])
@pytest.mark.parametrize("before_import", [False, True])
def test_string_class_conversion_hooks_keep_row_fallback(
    protocol, monkeypatch, method, before_import
):
    import importlib
    import sys

    name = "pyspark.sql._pandas_input"
    if before_import:
        monkeypatch.delitem(sys.modules, name, raising=False)
    else:
        importlib.import_module(name)
    frame = pd.DataFrame({"text": ["plain"] * 10000}, dtype=object)
    schema = T.StructType([T.StructField("text", T.StringType())])
    original = getattr(pd.DataFrame, method)
    calls = []
    expected = ["plain"] * 10000
    expected[0] = None if method == "notna" else "changed"

    def customized(self, *args, **kwargs):
        calls.append(method)
        result = original(self, *args, **kwargs)
        result.iat[0, 0] = False if method == "notna" else "changed"
        return result

    monkeypatch.setattr(pd.DataFrame, method, customized)
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert calls == [method]
        assert state["config_requests"] == []
        assert state["uploads"] == 0
        assert state["input_tables"][0].column(0).to_pylist() == expected



def test_string_instance_where_override_keeps_existing_row_behavior(protocol):
    from pyspark.sql._pandas_input import prepare_string_input

    frame = pd.DataFrame({"text": ["plain"] * 10000}, dtype=object)
    schema = T.StructType([T.StructField("text", T.StringType())])

    def customized(*args, **kwargs):
        raise AssertionError("row normalization calls where on the astype result")

    frame.where = customized
    assert prepare_string_input(frame, schema) is None
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert state["config_requests"] == []
        assert state["input_tables"][0].column(0).to_pylist() == ["plain"] * 10000



def test_string_native_backing_does_not_add_series_conversion_hooks(protocol, monkeypatch):
    frame = pd.DataFrame({"text": ["plain"] * 10000}, dtype=object)
    schema = T.StructType([T.StructField("text", T.StringType())])

    def customized(*args, **kwargs):
        raise AssertionError("native string input must not call Series.to_numpy")

    monkeypatch.setattr(pd.Series, "to_numpy", customized)
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert state["config_requests"] == []
        assert state["input_tables"][0].column(0).to_pylist() == ["plain"] * 10000



def test_string_large_inline_input_keeps_row_transport_and_schema(protocol):
    values = ["漢字-🙂-" + str(i) for i in range(100000)]
    frame = pd.DataFrame({"text": values}, dtype=object)
    schema = T.StructType([T.StructField("renamed", T.StringType(), False,
                                        {"source": "inline-string", "size": len(values)})])
    with _arrow_input_service(protocol, **{
        "spark.sql.session.localRelationCacheThreshold": "1",
        "spark.sql.session.localRelationSizeLimit": "1",
        "spark.sql.execution.arrow.useLargeVarTypes": "true",
    }) as (spark, state):
        result = spark.createDataFrame(frame, schema)
        assert not result.isLocal()
        result.toArrow()
        assert state["column_names"] == schema.names
        assert state["config_requests"] == []
        assert state["uploads"] == 0
        table = state["input_tables"][0]
        assert table.schema.names == ["renamed"]
        assert table.column(0).type == pa.string()
        assert table.schema.field(0).nullable is False
        assert table.schema.field(0).metadata is None
        assert table.column(0).to_pylist() == values
        assert __import__("json").loads(state["relations"][0].local_relation.schema) == (
            __import__("json").loads(schema.json()))


# Explicit-schema pandas input converts as PySpark's Arrow conversion does: the session time
# zone, the safe-cast setting, the nullability and the errors. Each expectation below was
# checked against official PySpark 4.3.0 with the same frame, schema and session settings.


def _pandas_table(protocol, frame, schema, **overrides):
    """The table that an explicit-schema pandas frame sends, and whether it is local."""
    with _arrow_input_service(protocol, **overrides) as (spark, state):
        df = spark.createDataFrame(frame, schema)
        assert not df.isLocal()
        df.toArrow()
        return state["input_tables"][0]


def _in_host_zone(name, call):
    """Run `call` with the process time zone `name`, as the client's local time zone."""
    import time

    previous = os.environ.get("TZ")
    os.environ["TZ"] = name
    time.tzset()
    try:
        return call()
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()


@pytest.mark.parametrize("values", [[1, None], [1.0, np.nan]], ids=["int", "float"])
def test_pandas_explicit_int_nulls_are_null(protocol, values):
    table = _pandas_table(protocol, pd.DataFrame({"f": values}), "f int")
    # Official: [Row(f=1), Row(f=None)].
    assert table.schema.field("f").type == pa.int32()
    assert table.column("f").to_pylist() == [1, None]


def _tinyint_schema():
    return T.StructType([T.StructField("b", T.ByteType())])


def test_pandas_explicit_tinyint_overflow_raises_as_pyspark(protocol):
    from pyspark.errors import PySparkValueError

    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(PySparkValueError) as error:
            spark.createDataFrame(pd.DataFrame({"b": [257]}), _tinyint_schema())
    assert str(error.value) == (
        "Exception thrown when converting pandas.Series (int64) with name 'b' to Arrow Array "
        "(int8). It can be caused by overflows or other unsafe conversions warned by Arrow. "
        "Arrow safe type check can be disabled by using SQL config "
        "`spark.sql.execution.pandas.convertToArrowArraySafely`."
    )
    assert state["input_tables"] == []


def test_pandas_explicit_tinyint_overflow_wraps_when_unsafe(protocol):
    # Official, with convertToArrowArraySafely=false: [Row(b=1)].
    table = _pandas_table(
        protocol, pd.DataFrame({"b": [257]}), _tinyint_schema(),
        **{"spark.sql.execution.pandas.convertToArrowArraySafely": "false"},
    )
    assert table.column("b").to_pylist() == [1]


def test_pandas_explicit_nonnullable_null_raises_as_pyspark(protocol):
    schema = T.StructType([T.StructField("a", T.LongType(), False)])
    frame = pd.DataFrame({"a": pd.array([1, None], dtype="Int64")})
    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(ValueError) as error:
            spark.createDataFrame(frame, schema)
    # Official: ValueError, not a PySparkRuntimeError from the row path.
    assert type(error.value) is ValueError
    assert str(error.value) == "Casting field 'a' with null values to non-nullable"
    assert state["input_tables"] == []


@pytest.mark.parametrize("schema", [None, ["p", "q"], ["p"]], ids=["inferred", "names", "fewer"])
def test_pandas_inferred_types_are_those_of_arrow_as_pyspark(protocol, schema):
    # As PySpark Connect's: without a schema, each column converts with PySpark's Arrow
    # conversion, of the type that Arrow infers, and the table goes without a schema, of
    # columns _0, _1, ..., which toDF names, by the labels, or the names and _n past them.
    frame = pd.DataFrame({"a": np.array([1, 2], dtype=np.int8), "b": [{"k": 1}, {"k": 2}]})
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        table = state["input_tables"][0]
        assert table.schema.names == ["_0", "_1"]
        assert table.schema.types == [pa.int8(), pa.struct([("k", pa.int64())])]
        assert table.column("_1").to_pylist() == [{"k": 1}, {"k": 2}]
        assert state["spark_schema"] is None
        assert state["column_names"] == {None: ["a", "b"], 2: ["p", "q"], 1: ["p", "_2"]}[
            None if schema is None else len(schema)]


def test_pandas_inferred_dicts_are_maps_of_infer_pandas_dict_as_map(protocol):
    # As PySpark Connect's: of spark.sql.execution.pandas.inferPandasDictAsMap, a column of dicts
    # is a map of the type of the first key, and the table takes the names of the columns.
    frame = pd.DataFrame({"a": [{"k": 1}, {"j": 2}], "b": ["x", "y"]})
    table = _pandas_table(protocol, frame, None, **{
        "spark.sql.execution.pandas.inferPandasDictAsMap": "true"})
    assert table.schema.names == ["a", "b"]
    assert table.schema.field("a").type == pa.map_(pa.string(), pa.int64())
    assert table.column("a").to_pylist() == [[("k", 1)], [("j", 2)]]


@pytest.mark.parametrize("host", ["America/Los_Angeles", "UTC"])
def test_pandas_inferred_naive_timestamp_uses_session_zone(protocol, host):
    # As PySpark Connect's: a naive time is a time of the session zone, whatever the client's.
    frame = pd.DataFrame({"t": [pd.Timestamp("2020-07-01 12:00")]})
    table = _in_host_zone(host, lambda: _pandas_table(
        protocol, frame, None, **{"spark.sql.session.timeZone": "America/Los_Angeles"}))
    assert table.column("_0").to_pylist() == [datetime(2020, 7, 1, 19, tzinfo=timezone.utc)]


def test_pandas_inferred_conversion_errors_are_those_of_pyspark(protocol):
    from pyspark.errors import PySparkValueError

    with _arrow_input_service(protocol) as (spark, state):
        # As PySpark Connect's: Arrow's error of a column of values of several types.
        with pytest.raises(PySparkValueError, match="Exception thrown when converting "
                           "pandas.Series \\(object\\) with name 'a' to Arrow Array"):
            spark.createDataFrame(pd.DataFrame({"a": [1, "x"]}))
        with pytest.raises(PySparkValueError) as error:
            spark.createDataFrame(pd.DataFrame({"a": [1]}), ["p", "q"])
        assert error.value.getCondition() == "AXIS_LENGTH_MISMATCH"
        assert error.value.getMessageParameters() == {"expected_length": "2", "actual_length": "1"}


@pytest.mark.parametrize("host", ["America/Los_Angeles", "UTC"])
def test_pandas_naive_timestamp_uses_session_zone(protocol, host):
    frame = pd.DataFrame({"t": [pd.Timestamp("2020-07-01 12:00")]})
    # Official with session UTC, whatever the client zone: 12:00 UTC.
    table = _in_host_zone(
        host,
        lambda: _pandas_table(
            protocol, frame, "t timestamp",
            **{"spark.sql.session.timeZone": "UTC"},
        ),
    )
    assert table.column("t").to_pylist() == [datetime(2020, 7, 1, 12, tzinfo=timezone.utc)]


def test_pandas_naive_timestamp_in_session_zone_is_converted(protocol):
    pytz = pytest.importorskip("pytz")
    frame = pd.DataFrame({"t": [pd.Timestamp("2020-07-01 12:00")]})
    # Official with session America/Los_Angeles and a client zone of UTC: 19:00 UTC.
    table = _in_host_zone(
        "UTC",
        lambda: _pandas_table(
            protocol, frame, "t timestamp",
            **{"spark.sql.session.timeZone": "America/Los_Angeles"},
        ),
    )
    assert table.column("t").to_pylist() == [datetime(2020, 7, 1, 19, tzinfo=timezone.utc)]

    gap = pd.DataFrame({"t": [pd.Timestamp("2020-03-08 02:30")]})
    with _arrow_input_service(protocol, **{
        "spark.sql.session.timeZone": "America/Los_Angeles",
    }) as (spark, _):
        # Official: pytz NonExistentTimeError for a naive time inside the DST gap.
        with pytest.raises(pytz.exceptions.NonExistentTimeError):
            spark.createDataFrame(gap, "t timestamp")


def test_pandas_nanosecond_timestamp_raises_as_pyspark(protocol):
    from pyspark.errors import PySparkValueError

    frame = pd.DataFrame({"t": [pd.Timestamp("2020-01-01 00:00:00.000000001")]})
    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(PySparkValueError) as error:
            spark.createDataFrame(frame, "t timestamp")
    assert str(error.value) == (
        "Exception thrown when converting pandas.Series (datetime64[ns, UTC]) with name 't' "
        "to Arrow Array (timestamp[us, tz=UTC]). It can be caused by overflows or other unsafe "
        "conversions warned by Arrow. Arrow safe type check can be disabled by using SQL config "
        "`spark.sql.execution.pandas.convertToArrowArraySafely`."
    )


def test_pandas_lone_surrogate_string_raises_as_pyspark(protocol):
    from pyspark.errors import PySparkValueError

    frame = pd.DataFrame({"s": ["a\ud800b"]}, dtype=object)
    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(PySparkValueError) as error:
            spark.createDataFrame(frame, "s string")
    assert str(error.value) == (
        "Exception thrown when converting pandas.Series (object) with name 's' to Arrow Array "
        "(string). It can be caused by overflows or other unsafe conversions warned by Arrow. "
        "Arrow safe type check can be disabled by using SQL config "
        "`spark.sql.execution.pandas.convertToArrowArraySafely`."
    )


def test_pandas_datetime_objects_in_date_column(protocol):
    frame = pd.DataFrame({"d": [datetime(2020, 1, 2, 3, 4)]}, dtype=object)
    table = _pandas_table(protocol, frame, "d date")
    # Official: [Row(d=datetime.date(2020, 1, 2))].
    assert table.column("d").type == pa.date32()
    assert table.column("d").to_pylist() == [date(2020, 1, 2)]


def test_pandas_nested_nan_becomes_null(protocol):
    array_frame = pd.DataFrame({"s": [[float("nan"), 1.0]]}, dtype=object)
    array_table = _pandas_table(protocol, array_frame, "s array<double>")
    # Official: [Row(s=[None, 1.0])].
    assert array_table.column("s").to_pylist() == [[None, 1.0]]

    map_frame = pd.DataFrame({"s": [{"k": float("nan")}]}, dtype=object)
    map_table = _pandas_table(protocol, map_frame, "s map<string,double>")
    # Official: [Row(s={'k': None})].
    assert map_table.column("s").to_pylist() == [[("k", None)]]


@pytest.mark.parametrize("schema", [None, ["x"]], ids=["inferred", "names"])
def test_pandas_inferred_int_beyond_int64_raises_as_pyspark(protocol, schema):
    frame = pd.DataFrame({"x": [2**64]}, dtype=object)
    with _arrow_input_service(protocol) as (spark, state):
        # Official: OverflowError, not a double.
        with pytest.raises(OverflowError) as error:
            spark.createDataFrame(frame, schema)
    assert str(error.value) == "Python int too large to convert to C long"


# Exact built-in date columns reuse Arrow without materializing pandas rows.

@pytest.mark.parametrize("accessor", ["values", "to_numpy"])
def test_date_series_accessors_do_not_replace_validated_cells(monkeypatch, accessor):
    from pyspark.sql._pandas_input import prepare_date_table

    dates = [date.min, date.max, date(1969, 12, 31), date(2000, 2, 29)] * 2500
    nullable = [None if i % 7 == 0 else value for i, value in enumerate(dates)]
    expected = {"renamed": dates, "nullable": nullable}
    frame = pd.DataFrame({"original": dates, "nullable": nullable}, dtype=object)
    schema = T.StructType([
        T.StructField("renamed", T.DateType(), False),
        T.StructField("nullable", T.DateType()),
    ])
    row_values = frame.astype(object).where(frame.notna(), None).values.tolist()
    calls = []

    def replacement(series, *args, **kwargs):
        calls.append(accessor)
        return np.full(len(series), date(2021, 2, 3), dtype=object)

    monkeypatch.setattr(pd.Series, accessor,
                        property(replacement) if accessor == "values" else replacement)
    table = prepare_date_table(frame, schema)
    assert table is not None
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    actual = pa.ipc.open_stream(sink.getvalue()).read_all()
    assert actual.to_pydict() == expected
    assert frame.astype(object).where(frame.notna(), None).values.tolist() == row_values
    assert calls == []


@pytest.mark.parametrize("cached", [False, True])
@pytest.mark.parametrize("verify", [False, True])
def test_date_columns_preserve_every_ipc_cell(protocol, monkeypatch, cached, verify):
    from pyspark.sql.pandas.types import from_arrow_metadata

    dates = [date.min, date.max, date(4, 2, 29), date(1600, 2, 29),
             date(1900, 2, 28), date(1900, 3, 1), date(1969, 12, 31),
             date(1970, 1, 1), date(2000, 2, 29), date(2100, 3, 1)] * 1000
    nullable = [None if i % 7 == 0 else value for i, value in enumerate(reversed(dates))]
    expected = [dates, nullable, [None] * len(dates)]
    frame = pd.DataFrame(dict(zip(["original", "b", "c"], expected)), dtype=object)
    schema = T.StructType([
        T.StructField("renamed.`day", T.DateType(), False, {"source": "date", "id": 3}),
        T.StructField("nullable", T.DateType(), True, {"tag": [1, 2]}),
        T.StructField("all_null", T.DateType()),
    ])

    def no_rows(*args, **kwargs):
        raise AssertionError("date fast path must not materialize Python rows")

    for method in ("to_numpy", "iterrows", "itertuples"):
        monkeypatch.setattr(pd.DataFrame, method, no_rows)
    monkeypatch.setattr(pd.DataFrame, "values", property(no_rows))
    monkeypatch.setattr(pd.Series, "tolist", no_rows)
    overrides = {"spark.sql.session.localRelationCacheThreshold": "1" if cached else "999999999"}
    with _arrow_input_service(protocol, **overrides) as (spark, state):
        df = spark.createDataFrame(frame, schema, verifySchema=verify)
        assert not df.isLocal()
        df.toArrow()
        assert state["column_names"] == schema.names
        table = state["input_tables"][0]
        assert table.num_rows == len(dates)
        assert table.schema.names == schema.names
        epoch = date(1970, 1, 1)
        for column, values, field in zip(table.columns, expected, schema.fields):
            assert column.type == pa.date32()
            # Independent expectations cover every date and validity bit in IPC.
            assert column.cast(pa.int32()).to_pylist() == [
                None if value is None else (value - epoch).days for value in values
            ]
            assert column.to_pylist() == values
            actual_field = table.schema.field(field.name)
            assert actual_field.nullable == field.nullable
            assert from_arrow_metadata(actual_field.metadata) == (field.metadata or None)
        assert bool(state["uploads"]) == cached


@pytest.mark.parametrize("reason", [
    "small", "empty", "inferred", "names", "ddl", "cast", "mixed_schema",
    "width", "duplicate_input", "duplicate_schema", "input_nul", "schema_nul",
    "input_number", "datetime", "timestamp", "nan", "pd_na", "nat", "mixed",
    "nonnullable_null", "frame_subclass", "date_subclass", "late_unsupported",
])
def test_date_column_guard_preserves_row_fallback(reason):
    from pyspark.sql._pandas_input import prepare_date_table

    class Frame(pd.DataFrame):
        pass

    class Day(date):
        def toordinal(self):
            raise AssertionError("guard must not call conversion hooks")

    values = [date(2020, 1, 1)] * 10000
    frame = pd.DataFrame({"day": values}, dtype=object)
    schema = T.StructType([T.StructField("day", T.DateType())])
    if reason == "small":
        frame = frame.iloc[:9999]
    elif reason == "empty":
        frame = frame.iloc[:0]
    elif reason in ("inferred", "names", "ddl", "cast"):
        schema = {"inferred": None, "names": ["day"], "ddl": "day DATE",
                  "cast": T.StructType([T.StructField("day", T.StringType())])}[reason]
    elif reason == "mixed_schema":
        frame["long"] = np.arange(10000)
        schema.add("long", T.LongType())
    elif reason == "width":
        schema.add("extra", T.DateType())
    elif reason in ("duplicate_input", "duplicate_schema"):
        frame["other"] = values
        schema.add("other", T.DateType())
        if reason == "duplicate_input":
            frame.columns = ["day", "day"]
        else:
            schema = T.StructType([T.StructField("day", T.DateType())] * 2)
    elif reason in ("input_nul", "input_number"):
        frame.columns = ["day\x00" if reason == "input_nul" else 1]
    elif reason == "schema_nul":
        schema = T.StructType([T.StructField("day\x00", T.DateType())])
    elif reason == "frame_subclass":
        frame = Frame(frame)
    else:
        value = {
            "datetime": datetime(2020, 1, 1), "timestamp": pd.Timestamp("2020-01-01"),
            "nan": float("nan"), "pd_na": pd.NA, "nat": pd.NaT, "mixed": "2020-01-01",
            "nonnullable_null": None, "date_subclass": Day(2020, 1, 1),
            "late_unsupported": Day(2020, 1, 1),
        }[reason]
        if reason == "nonnullable_null":
            schema = T.StructType([T.StructField("day", T.DateType(), False)])
        if reason == "late_unsupported":
            frame["other"] = values
            schema.add("other", T.DateType())
        frame.iat[-1, -1] = value
    assert prepare_date_table(frame, schema) is None


@pytest.mark.parametrize("reason", ["date_subclass", "frame_subclass", "nan", "pd_na", "nat",
                                    "nonnullable_null", "inferred", "ddl", "small"])
def test_date_public_fallback_keeps_hooks(protocol, monkeypatch, reason):
    calls = []

    class Day(date):
        def toordinal(self):
            calls.append("toordinal")
            return super().toordinal() + 7

    class Frame(pd.DataFrame):
        pass

    values = [date(2020, 1, 1)] * (9999 if reason == "small" else 10000)
    if reason in ("nan", "pd_na", "nat", "nonnullable_null"):
        values[-1] = {"nan": float("nan"), "pd_na": pd.NA, "nat": pd.NaT,
                      "nonnullable_null": None}[reason]
    elif reason == "date_subclass":
        values[-1] = Day(2020, 1, 1)
    frame_type = Frame if reason == "frame_subclass" else pd.DataFrame
    frame = frame_type({"day": values}, dtype=object)
    schema = T.StructType([T.StructField("day", T.DateType(), reason != "nonnullable_null")])
    if reason == "inferred":
        schema = None
    elif reason == "ddl":
        schema = "day DATE"
    astype = pd.DataFrame.astype

    def record_rows(self, *args, **kwargs):
        calls.append("astype")
        return astype(self, *args, **kwargs)

    monkeypatch.setattr(pd.DataFrame, "astype", record_rows)
    with _arrow_input_service(protocol) as (spark, state):
        if reason == "nonnullable_null":
            from pyspark.errors import PySparkRuntimeError

            with pytest.raises(PySparkRuntimeError, match=(
                "Column 'day' is declared as non-nullable but contains null values"
            )):
                spark.createDataFrame(frame, schema)
            assert calls == ["astype"]
            assert state["config_requests"] == []
            return
        spark.createDataFrame(frame, schema).toArrow()
        assert "astype" in calls
        actual = state["input_tables"][0].column(0).to_pylist()
        expected = [date(2020, 1, 1)] * len(values)
        if reason in ("nan", "pd_na", "nat", "nonnullable_null"):
            expected[-1] = None
        elif reason == "date_subclass":
            assert calls.count("toordinal") == 1
            expected[-1] = date(2020, 1, 8)
        assert actual == expected


@pytest.mark.parametrize("class_name,attribute", [
    ("StructType", "names"), ("StructType", "fields"), ("StructType", "__iter__"),
    ("StructField", "name"), ("StructField", "dataType"), ("StructField", "nullable"),
    ("StructField", "metadata"),
    ("StructType", "__getattribute__"), ("StructField", "__getattribute__"),
    ("DateType", "__getattribute__"), ("StructType", "__class__"),
    ("StructField", "__class__"), ("DateType", "__class__"),
])
@pytest.mark.parametrize("before_import", [False, True])
def test_date_custom_schema_descriptors_keep_native_row_path(
    protocol, class_name, attribute, before_import
):
    import sys

    script = r'''
import sys
import tempfile
from datetime import date
from pathlib import Path
import pandas as pd
from pyspark.sql import types as T

frame = pd.DataFrame({"day": [date(2020, 1, 1)] * 10000}, dtype=object)
schema = T.StructType([T.StructField("day", T.DateType(), True, {"source": "native"})])
class_name, attribute, before_import = sys.argv[1:4]
if before_import == "False":
    import pyspark.sql._pandas_input
cls = getattr(T, class_name)
if attribute == "__getattribute__":
    replacement = lambda self, name: object.__getattribute__(self, name)
elif attribute == "__iter__":
    replacement = lambda self: iter([T.StructField("changed", T.DateType())])
elif attribute == "__class__":
    attribute = "__getattribute__"
    replacement = lambda self, name: object if name == "__class__" else (
        object.__getattribute__(self, name))
else:
    value = {"names": ["changed"], "fields": [], "name": "changed",
             "dataType": T.DateType(), "nullable": False, "metadata": {"changed": True}}[attribute]
    replacement = property(lambda self: value)
setattr(cls, attribute, replacement)
import importlib.util
spec = importlib.util.spec_from_file_location("date_schema_capture", sys.argv[4])
capture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capture
spec.loader.exec_module(capture)
_arrow_input_service, protocol = capture._arrow_input_service, capture.protocol
class Factory:
    def mktemp(self, name):
        return Path(tempfile.mkdtemp(prefix=name))
with _arrow_input_service(protocol.__wrapped__(Factory())) as (spark, state):
    spark.createDataFrame(frame, schema).toArrow()
    assert state["config_requests"] == [], state["config_requests"]
    table = state["input_tables"][0]
    assert table.schema.names == ["day"], table.schema
    assert table.column(0).to_pylist() == [date(2020, 1, 1)] * 10000
'''
    result = subprocess.run([sys.executable, "-c", script, class_name, attribute,
                             str(before_import), __file__],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("dtype_mode", ["equal", "class_spoof", "exact_numpy"])
def test_date_custom_extension_dtype_cannot_impersonate_numpy_object(protocol, dtype_mode):
    from pandas.api.extensions import ExtensionArray, ExtensionDtype, take
    from pyspark.sql._pandas_input import prepare_date_table

    arrow_calls = []

    class ObjectEqualDtype(ExtensionDtype):
        name = "object_equal_date"
        type = date
        kind = "O"
        na_value = None

        @classmethod
        def construct_array_type(cls):
            return DifferentArrowArray

        def __eq__(self, other):
            return (isinstance(other, ObjectEqualDtype)
                    or isinstance(other, np.dtype) and other.kind == "O")

        def __hash__(self):
            return hash(self.name)

    class DifferentArrowArray(ExtensionArray):
        def __init__(self, values):
            self._values = np.asarray(values, dtype=object)

        @property
        def dtype(self):
            return ObjectEqualDtype()

        @property
        def nbytes(self):
            return self._values.nbytes

        def __len__(self):
            return len(self._values)

        def __getitem__(self, key):
            result = self._values[key]
            return type(self)(result) if isinstance(result, np.ndarray) else result

        def isna(self):
            return pd.isna(self._values)

        def take(self, indices, allow_fill=False, fill_value=None):
            return type(self)(take(self._values, indices, allow_fill=allow_fill,
                                   fill_value=fill_value))

        def copy(self):
            return type(self)(self._values.copy())

        @classmethod
        def _from_sequence(cls, scalars, *, dtype=None, copy=False):
            values = np.asarray(scalars, dtype=object)
            return cls(values.copy() if copy else values)

        def to_numpy(self, dtype=None, copy=False, na_value=None):
            if dtype is not None:
                return self._values.astype(dtype, copy=copy)
            return self._values.copy() if copy else self._values

        def __arrow_array__(self, type=None):
            arrow_calls.append("arrow")
            # A scan of to_numpy() cannot establish the values this hook emits.
            return pa.array([date(2020, 1, 2)] * len(self), type=type)

    array = DifferentArrowArray([date(2020, 1, 1)] * 10000)
    frame = pd.DataFrame({"day": pd.Series(array, copy=False)})
    schema = T.StructType([T.StructField("day", T.DateType())])
    if dtype_mode == "exact_numpy":
        # An ExtensionArray may change its advertised dtype after construction.
        # The dtype and NumPy scan both pass while its Arrow hook changes cells.
        DifferentArrowArray.dtype = property(lambda self: np.dtype("object"))
        DifferentArrowArray.__array__ = lambda self, dtype=None, copy=None: self._values
    elif dtype_mode == "class_spoof":
        def spoof_class(self, name):
            if name == "__class__":
                return type(np.dtype("object"))
            return object.__getattribute__(self, name)

        ObjectEqualDtype.__getattribute__ = spoof_class
    dtype = frame.dtypes.iloc[0]
    assert type(frame) is pd.DataFrame
    assert type(frame.iloc[:, 0].array) is DifferentArrowArray
    assert type(dtype) is (type(np.dtype("object")) if dtype_mode == "exact_numpy"
                           else ObjectEqualDtype)
    assert isinstance(dtype, np.dtype) is (dtype_mode != "equal")
    assert dtype == np.dtype("object")
    assert prepare_date_table(frame, schema) is None
    with _arrow_input_service(protocol) as (spark, state):
        if dtype_mode == "exact_numpy":
            # This deliberately inconsistent EA already fails pandas' row-path
            # astype(object). Preserve that error before importer configuration.
            with pytest.raises(TypeError, match="cannot set astype"):
                spark.createDataFrame(frame, schema)
            assert state["input_tables"] == []
        else:
            spark.createDataFrame(frame, schema).toArrow()
            assert state["input_tables"][0].column(0).to_pylist() == [date(2020, 1, 1)] * 10000
        assert state["config_requests"] == []
    assert arrow_calls == []


def test_date_rebound_dataframe_alias_keeps_subclass_on_row_path(monkeypatch):
    from pyspark.sql._pandas_input import prepare_date_table

    class Frame(pd.DataFrame):
        pass

    frame = Frame({"day": [date(2020, 1, 1)] * 10000}, dtype=object)
    schema = T.StructType([T.StructField("day", T.DateType())])
    monkeypatch.setattr(pd, "DataFrame", Frame)
    assert prepare_date_table(frame, schema) is None


def _date_alias_guard_process(mode, argument):
    """Keep mutations of converter aliases isolated from other tests."""
    import sys

    if mode == "captured_alias":
        import pyspark.sql._arrow_input as importer
        if not hasattr(importer, "to_arrow_schema"):
            pytest.skip("this importer resolves converters locally and holds no captured aliases")

    script = r'''
import importlib
import importlib.util
import json
import sys
import tempfile
from datetime import date
from pathlib import Path
import pandas as pd
import pyarrow as pa
from pyspark.sql import types as T
from pyspark import _pyspark as native

spec = importlib.util.spec_from_file_location("date_alias_capture", sys.argv[3])
capture = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = capture
spec.loader.exec_module(capture)
values = [date.min, date.max, date(1969, 12, 31), None] * 2500
frame = pd.DataFrame({"original": values}, dtype=object)
schema = T.StructType([T.StructField("renamed", T.DateType(), True, {"source": "native"})])
mode, argument = sys.argv[1:3]
if mode in ("public_alias", "converter_alias", "captured_alias"):
    replacement = native.BooleanType if argument == "DateType" else native.DateType
    if mode == "public_alias":
        sys.modules.pop("pyspark.sql.pandas.types", None)
        setattr(T, argument, replacement)
    else:
        converter = importlib.import_module("pyspark.sql.pandas.types")
        if mode == "captured_alias":
            importer = importlib.import_module("pyspark.sql._arrow_input")
        setattr(converter, argument, replacement)
        if mode == "captured_alias":
            sys.modules.pop("pyspark.sql.pandas.types")
            current = importlib.import_module("pyspark.sql.pandas.types")
            assert current is not converter
            assert importer.to_arrow_schema.__globals__ is vars(converter)
elif mode == "atomic_alias":
    scope, replacement_name = argument.split(":")
    replacement = None if replacement_name == "none" else native.DateType
    if scope == "public_before":
        sys.modules.pop("pyspark.sql._arrow_input", None)
        T.AtomicType = replacement
    elif scope == "public_after":
        importlib.import_module("pyspark.sql._arrow_input")
        T.AtomicType = replacement
    else:
        importer = importlib.import_module("pyspark.sql._arrow_input")
        if replacement_name == "missing":
            del importer.AtomicType
        else:
            importer.AtomicType = replacement
elif mode == "importer_alias":
    importer = importlib.import_module("pyspark.sql._arrow_input")
    importer.StructType = None if argument == "none" else native.DateType
elif mode == "importer_function":
    importer = importlib.import_module("pyspark.sql._arrow_input")
    setattr(importer, argument, None)
else:
    raise AssertionError(mode)
class Factory:
    def mktemp(self, name):
        return Path(tempfile.mkdtemp(prefix=name))
with capture._arrow_input_service(capture.protocol.__wrapped__(Factory())) as (spark, state):
    spark.createDataFrame(frame, schema).toArrow()
    assert state["config_requests"] == [], state["config_requests"]
    table = state["input_tables"][0]
    assert table.schema.names == ["renamed"], table.schema
    assert table.column(0).type == pa.date32()
    assert table.column(0).to_pylist() == values
    assert table.schema.field(0).nullable
    assert table.schema.field(0).metadata is None
    relation_schema = json.loads(state["relations"][0].local_relation.schema)
    assert relation_schema["fields"][0]["metadata"] == {"source": "native"}
'''
    result = subprocess.run([sys.executable, "-c", script, mode, argument, __file__],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("class_name", [
    "BooleanType", "ByteType", "ShortType", "IntegerType", "LongType", "FloatType",
    "DoubleType", "DecimalType", "StringType", "BinaryType", "DateType", "StructField", "StructType",
])
@pytest.mark.parametrize("mode", ["public_alias", "converter_alias", "captured_alias"])
def test_date_rebound_converter_aliases_keep_native_row_path(class_name, mode):
    _date_alias_guard_process(mode, class_name)


@pytest.mark.parametrize("replacement", ["none", "date"])
def test_date_rebound_importer_schema_alias_keeps_native_row_path(replacement):
    _date_alias_guard_process("importer_alias", replacement)


@pytest.mark.parametrize("scope", ["public_before", "public_after", "importer"])
@pytest.mark.parametrize("replacement", ["none", "date"])
def test_date_rebound_atomic_alias_keeps_native_row_path(scope, replacement):
    _date_alias_guard_process("atomic_alias", f"{scope}:{replacement}")


def test_date_deleted_importer_atomic_alias_keeps_native_row_path():
    _date_alias_guard_process("atomic_alias", "importer:missing")


@pytest.mark.parametrize("name", ["from_arrow_schema", "to_arrow_schema"])
def test_date_present_invalid_importer_function_keeps_native_row_path(name):
    _date_alias_guard_process("importer_function", name)


def test_date_warm_importer_without_global_converter_aliases(protocol, monkeypatch):
    import pyspark.sql._arrow_input as importer
    import sys
    import types

    original = importer.prepare_table

    def local_imports(table, schema, *configs):
        from pyspark.sql.pandas.types import from_arrow_schema, to_arrow_schema

        table, resolved, names = original(table, schema, *configs)
        assert from_arrow_schema(table.schema).fields
        assert to_arrow_schema(resolved).names == table.column_names
        return table, resolved, names

    values = [date.min, date.max, date(1970, 1, 1), None] * 2500
    frame = pd.DataFrame({"day": values}, dtype=object)
    schema = T.StructType([T.StructField("renamed", T.DateType())])
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(pa.table({"warm": [1, 2]})).toArrow()
        # Exercise the local-import shape on this branch's pinned dependency.
        substitute = types.ModuleType(importer.__name__)
        substitute.StructType = T.StructType
        substitute.AtomicType = T.AtomicType
        substitute.prepare_table = local_imports
        monkeypatch.setitem(sys.modules, importer.__name__, substitute)

        def no_rows(*args, **kwargs):
            raise AssertionError("warm date input must stay columnar")

        monkeypatch.setattr(pd.DataFrame, "values", property(no_rows))
        for _ in range(3):
            spark.createDataFrame(frame, schema).toArrow()
        for table in state["input_tables"][1:]:
            assert table.column_names == ["renamed"]
            assert table.column(0).to_pylist() == values


@pytest.mark.parametrize("location", ["class", "instance"])
@pytest.mark.parametrize("method", ["astype", "notna", "where"])
def test_date_custom_normalization_hooks_keep_native_row_path(
    protocol, monkeypatch, location, method
):
    from types import MethodType
    from pyspark.sql._pandas_input import prepare_date_table

    values = [date(2020, 1, 1)] * 10000
    frame = pd.DataFrame({"original": values}, dtype=object)
    schema = T.StructType([T.StructField("renamed", T.DateType())])
    original = getattr(pd.DataFrame, method)
    calls = []

    def replacement(self, *args, **kwargs):
        calls.append(method)
        result = original(self, *args, **kwargs)
        if method == "notna":
            result.iat[-1, 0] = False
        else:
            result.iat[0, 0] = date(2020, 1, 2 if method == "astype" else 3)
        return result

    if location == "class":
        monkeypatch.setattr(pd.DataFrame, method, replacement)
    else:
        # Avoid pandas' attribute-to-column routing when installing an instance hook.
        monkeypatch.setitem(vars(frame), method, MethodType(replacement, frame))
    assert prepare_date_table(frame, schema) is None
    expected = values.copy()
    if method == "notna":
        expected[-1] = None
    elif method == "astype":
        expected[0] = date(2020, 1, 2)
    elif location == "class":
        expected[0] = date(2020, 1, 3)
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert state["config_requests"] == []
        table = state["input_tables"][0]
        assert table.column_names == ["renamed"]
        assert table.column(0).to_pylist() == expected
    # astype returns a new frame, so an original instance's where hook is not
    # used by the native normalization sequence. Preserve that ordering too.
    assert calls == ([] if method == "where" and location == "instance" else [method])


def _date_columns_guard_input(native_rows=False):
    class NativeRows(pd.DataFrame):
        pass

    days = [date.min, date.max, date(4, 2, 29), date(1600, 2, 29),
            date(1900, 2, 28), date(1900, 3, 1), date(1969, 12, 31),
            date(1970, 1, 1), date(2000, 2, 29), date(2100, 3, 1)] * 1000
    nullable = [None if i % 7 == 0 else value for i, value in enumerate(reversed(days))]
    expected = [days, nullable, [None] * len(days)]
    frame_type = NativeRows if native_rows else pd.DataFrame
    frame = frame_type(dict(zip(["original", "b", "c"], expected)), dtype=object)
    schema = T.StructType([
        T.StructField("renamed.`day", T.DateType(), False, {"source": "date", "id": 3}),
        T.StructField("nullable", T.DateType(), True, {"tag": [1, 2]}),
        T.StructField("all_null", T.DateType(), True, {"source": "all-null"}),
    ])
    return frame, schema, expected


def _assert_date_columns_guard_payload(state, schema, expected, *, row_path):
    import json
    from pyspark.sql.pandas.types import from_arrow_metadata

    assert len(state["input_tables"]) == 1
    assert state["column_names"] == schema.names
    table = state["input_tables"][0]
    assert table.num_rows == 10000
    assert table.column_names == ["renamed.`day", "nullable", "all_null"]
    assert not state["uploads"]
    epoch = date(1970, 1, 1)
    for column, values, field in zip(table.columns, expected, schema.fields):
        assert column.type == pa.date32()
        assert column.to_pylist() == values
        assert column.cast(pa.int32()).to_pylist() == [
            None if value is None else (value - epoch).days for value in values
        ]
        assert column.null_count == sum(value is None for value in values)
        physical = table.schema.field(field.name)
        assert physical.nullable is field.nullable
        if row_path:
            assert physical.metadata is None
        else:
            assert from_arrow_metadata(physical.metadata) == field.metadata
    assert json.loads(state["relations"][0].local_relation.schema) == {
        "type": "struct", "fields": [
            {"name": "renamed.`day", "type": "date", "nullable": False,
             "metadata": {"source": "date", "id": 3}},
            {"name": "nullable", "type": "date", "nullable": True,
             "metadata": {"tag": [1, 2]}},
            {"name": "all_null", "type": "date", "nullable": True,
             "metadata": {"source": "all-null"}},
        ],
    }
    assert (state["config_requests"] == []) is row_path


@pytest.mark.parametrize("native_rows", [False, True])
@pytest.mark.parametrize("location", [
    "subclass", "class", "owner", "backing_subclass_item", "backing_class_iter",
    "backing_class_getitem", "backing_instance_box",
])
@pytest.mark.parametrize("behavior", ["raise", "mutate"])
def test_date_columns_iterator_hooks_keep_native_order(
    protocol, monkeypatch, native_rows, location, behavior
):
    from pandas.core.base import IndexOpsMixin

    frame, schema, values = _date_columns_guard_input(native_rows)
    original_iter = pd.Index.__iter__
    original_notna, original_astype = frame.notna, frame.astype
    calls = []
    invoked = False

    class ColumnsError(RuntimeError):
        pass

    def notna(*args, **kwargs):
        calls.append("notna")
        return original_notna(*args, **kwargs)

    def astype(*args, **kwargs):
        calls.append("astype")
        return original_astype(*args, **kwargs)

    def before_hook():
        nonlocal invoked
        if invoked:
            return
        invoked = True
        calls.append("columns iterator")
        if behavior == "raise":
            raise ColumnsError("date columns hook precedes pandas normalization")
        # Install spies only when native iteration reaches the columns hook.
        # Before eligibility, normalization methods are the canonical methods.
        monkeypatch.setitem(vars(frame), "notna", notna)
        monkeypatch.setitem(vars(frame), "astype", astype)
        frame.iat[0, 0] = date(2004, 2, 29)
        frame.iat[-1, 1] = None
        frame.iat[17, 2] = date.min

    def iterate(index):
        if index is columns:
            before_hook()
        return original_iter(index)

    if location == "subclass":
        class Columns(pd.Index):
            pass
        monkeypatch.setattr(Columns, "__iter__", iterate)
        labels = pd.Index(["original", "b", "c"], dtype=object)
        frame.columns = Columns._simple_new(labels._values, name=None)
        assert type(frame.columns) is Columns
    elif location == "backing_subclass_item":
        # IndexOpsMixin iteration calls ndarray.item, rather than ndarray.__iter__.
        class Names(np.ndarray):
            def item(self, *args):
                before_hook()
                return super().item(*args)
        backing = np.asarray(["original", "b", "c"], dtype=object).view(Names)
        frame.columns = pd.Index._simple_new(backing, name=None)
        assert type(frame.columns._values) is Names
    elif location.startswith("backing_"):
        frame.columns = pd.Index(pd.array(["original", "b", "c"], dtype=pd.StringDtype(storage="python")))
        backing = frame.columns._values
        if location == "backing_class_iter":
            original = type(backing).__iter__

            def backing_iter(array):
                if array is backing:
                    before_hook()
                return original(array)

            monkeypatch.setattr(type(backing), "__iter__", backing_iter)
        elif location == "backing_class_getitem":
            original = type(backing).__getitem__

            def backing_getitem(array, key):
                if array is backing:
                    before_hook()
                return original(array, key)

            monkeypatch.setattr(type(backing), "__getitem__", backing_getitem)
        else:
            original = backing._box_func

            def box(value):
                before_hook()
                return original(value)

            monkeypatch.setitem(vars(backing), "_box_func", box)
    else:
        monkeypatch.setattr(pd.Index if location == "class" else IndexOpsMixin,
                            "__iter__", iterate)
    columns = frame.columns

    with _arrow_input_service(protocol) as (spark, state):
        if behavior == "raise":
            with pytest.raises(ColumnsError,
                               match="date columns hook precedes pandas normalization"):
                spark.createDataFrame(frame, schema)
            assert calls == ["columns iterator"]
            assert not state["config_requests"]
            assert not state["uploads"]
            assert not state["relations"]
            assert not state["input_tables"]
        else:
            spark.createDataFrame(frame, schema).toArrow()
            expected = [column.copy() for column in values]
            expected[0][0], expected[1][-1], expected[2][17] = date(2004, 2, 29), None, date.min
            _assert_date_columns_guard_payload(state, schema, expected, row_path=True)
            assert calls == ["columns iterator", "notna", "astype"]


@pytest.mark.parametrize("native_rows", [False, True])
def test_date_rebound_index_values_keeps_native_error(protocol, monkeypatch, native_rows):
    frame, schema, _ = _date_columns_guard_input(native_rows)
    columns = frame.columns
    original = pd.Index._values.fget
    calls = []

    def values(index):
        if index is columns:
            calls.append("_values")
            raise RuntimeError("date columns backing hook precedes normalization")
        return original(index)

    monkeypatch.setattr(pd.Index, "_values", property(values))
    with _arrow_input_service(protocol) as (spark, state):
        with pytest.raises(RuntimeError,
                           match="date columns backing hook precedes normalization"):
            spark.createDataFrame(frame, schema)
        assert calls == ["_values"]
        assert state["config_requests"] == []
        assert state["input_tables"] == []
        assert state["uploads"] == 0


@pytest.mark.parametrize("native_rows", [False, True])
def test_date_rebound_index_data_keeps_native_read_order(protocol, monkeypatch, native_rows):
    frame, schema, values = _date_columns_guard_input(native_rows)
    columns, backing = frame.columns, vars(frame.columns)["_data"]
    original_notna, original_astype = frame.notna, frame.astype
    calls = []
    normalizing = False
    allowed_reads = None

    class ColumnsDataError(RuntimeError):
        pass

    def notna(*args, **kwargs):
        nonlocal normalizing
        normalizing = True
        calls.append("notna")
        return original_notna(*args, **kwargs)

    def astype(*args, **kwargs):
        calls.append("astype")
        return original_astype(*args, **kwargs)

    def data(index):
        if index is columns:
            if not normalizing:
                calls.append("_data")
                if allowed_reads is not None and calls.count("_data") > allowed_reads:
                    raise ColumnsDataError("eligibility added a columns storage read")
                # Preserve the native first columns read and only then attach
                # observers; no private optimization call determines fallback.
                monkeypatch.setitem(vars(frame), "notna", notna)
                monkeypatch.setitem(vars(frame), "astype", astype)
            return backing
        return vars(index)["_data"]

    def set_data(index, value):
        vars(index)["_data"] = value

    monkeypatch.setattr(pd.Index, "_data", property(data, set_data), raising=False)
    import pyspark.sql._pandas_input as helper

    # Measure current native fallback dispatch without either eligibility probe.
    with monkeypatch.context() as baseline:
        baseline.setattr(helper, "prepare_int64_table", lambda *_: None)
        baseline.setattr(helper, "prepare_date_table", lambda *_: None)
        with _arrow_input_service(protocol) as (spark, state):
            spark.createDataFrame(frame, schema).toArrow()
            _assert_date_columns_guard_payload(state, schema, values, row_path=True)
            control_table = state["input_tables"][0]
    baseline_calls = calls.copy()
    allowed_reads = baseline_calls.count("_data")
    assert allowed_reads > 0 and "notna" in baseline_calls
    calls.clear()
    normalizing = False
    # Restore ordinary methods before candidate eligibility; the storage hook
    # installs observers only when the native columns iterator starts.
    monkeypatch.delitem(vars(frame), "notna")
    monkeypatch.delitem(vars(frame), "astype")
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        # The rebound storage declines the Arrow path without reading the columns more
        # than the row path does. pandas caches some of these reads, so a later row
        # conversion of the same frame can read them fewer times than the first.
        assert "notna" in calls
        assert calls.count("_data") <= allowed_reads
        _assert_date_columns_guard_payload(state, schema, values, row_path=True)
        assert state["input_tables"][0].equals(control_table, check_metadata=True)


@pytest.mark.parametrize("storage", ["object", "default", "python", "pyarrow", "python_nan", "pyarrow_nan"])
def test_date_canonical_columns_storage_stays_columnar(protocol, monkeypatch, storage):
    frame, schema, values = _date_columns_guard_input()
    labels = ["original", "b", "c"]
    if storage == "object":
        frame.columns = pd.Index(labels, dtype=object)
    elif storage != "default":
        kind = storage.removesuffix("_nan")
        label_dtype = (pd.StringDtype(storage=kind, na_value=np.nan)
                       if storage.endswith("_nan") else pd.StringDtype(storage=kind))
        frame.columns = pd.Index(pd.array(labels, dtype=label_dtype))

    def no_rows(*args, **kwargs):
        raise AssertionError("ordinary date columns must retain columnar construction")

    monkeypatch.setattr(pd.DataFrame, "values", property(no_rows))
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        _assert_date_columns_guard_payload(state, schema, values, row_path=False)
