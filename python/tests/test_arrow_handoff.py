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
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from urllib.parse import quote

import grpc
import pandas as pd
import pyarrow as pa
import pytest
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory
from pyspark.sql import SparkSession


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
    expected = pa.Table.from_batches(batches).to_pandas()
    calls = {}
    with _dataframe(protocol=protocol, batches=batches, calls=calls) as (frame, _):
        actual = frame.toPandas()
    pd.testing.assert_frame_equal(actual, expected)
    assert calls == {"execute": 1}


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
        "spark.sql.session.localRelationCacheThreshold": "1048576",
        "spark.sql.session.localRelationSizeLimit": str(3 * 1024**3),
        "spark.sql.session.localRelationChunkSizeRows": "10000",
        "spark.sql.session.localRelationChunkSizeBytes": str(16 * 1024**2),
        "spark.sql.session.localRelationBatchOfChunksSizeBytes": str(1024**3),
    }
    config.update(overrides)
    state = {"cache": {}, "uploads": 0, "relations": [], "message_sizes": [],
             "config_requests": [], "upload_contexts": [], "status_contexts": []}
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
        keys = getattr(request.operation, operation).keys
        state["config_requests"].append((operation, list(keys)))
        return protocol("ConfigResponse", session_id=request.session_id,
                        pairs=[dict(key=k, value=config[k]) for k in keys
                               if config.get(k) is not None]
                        ).SerializeToString()

    def statuses(raw, context):
        request = protocol("ArtifactStatusesRequest").FromString(raw)
        state["status_contexts"].append(attribution(request, context))
        return protocol("ArtifactStatusesResponse", session_id=request.session_id,
                        statuses={name: dict(exists=name in state["cache"])
                                  for name in request.names}).SerializeToString()

    def upload(requests, context):
        summaries = []
        name, expected, buffer = None, 0, bytearray()
        session_id = None

        def save(name, data):
            if not (allow_file_artifacts and name.startswith(("files/", "forward_to_fs/"))):
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
        for payload in payloads:
            reader = pa.ipc.open_stream(payload)
            state["arrow_schemas"].append(reader.schema)
            stream_rows = []
            state["batch_rows"].append(stream_rows)
            for batch in reader:
                stream_rows.append(batch.num_rows)
                rows += batch.num_rows
                if batch.num_rows:
                    if first is None:
                        first = batch.column(0)[0].as_py()
                    last = batch.column(0)[-1].as_py()
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
            spark.addArtifacts(str(small), str(large))
            artifact_names = ["files/small.txt", "files/large.txt"]
        else:
            destinations = ["/wire/small.txt", "/wire/large.txt"]
            for path, destination in zip([small, large], destinations):
                spark.copyFromLocalToFs(str(path), destination)
            artifact_names = ["forward_to_fs/" + destination for destination in destinations]
        payloads = {context["payload"] for context in state["upload_contexts"]}
        assert payloads == {"batch", "begin_chunk", "chunk"}, (
            f"Expected all artifact frame kinds, got {payloads}. "
            "This fixture assumes a 32 KiB CHUNK_SIZE; adjust its sizes if CHUNK_SIZE changes."
        )
        assert state["cache"][artifact_names[0]] == small.read_bytes()
        assert state["cache"][artifact_names[1]] == large.read_bytes()
        assert state["uploads"] == (1 if upload == "file" else 2)
        assert not state["status_contexts"]
        _assert_artifact_contexts(
            state["upload_contexts"],
            session_id=spark.session_id,
            user_id=user_id or "",
            user_agent=user_agent or "_SPARK_CONNECT_PYTHON",
        )


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
        assert len(state["config_requests"]) == 1
        assert state["config_requests"][0][0] == "get"
        assert len(state["config_requests"][0][1]) == 8
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
