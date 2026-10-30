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
    addresses = [buffer.address for buffer in (numeric.buffers()[1], strings.buffers()[2])]
    del table
    gc.collect()
    assert numeric.to_pylist() == [None, -3, 4]
    assert strings.to_pylist() == [None, "", "omega"]
    assert numeric.buffers()[1].address == addresses[0]
    assert strings.buffers()[2].address == addresses[1]
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
def _arrow_input_service(protocol, *, upload_error=None, **overrides):
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
             "config": config, "config_requests": [], "input_tables": []}
    server_id = "00000000-0000-0000-0000-000000000001"

    def configure(raw, context):
        request = protocol("ConfigRequest").FromString(raw)
        keys = list(request.operation.get.keys)
        state["config_requests"].append(keys)
        if error := state.pop("config_error", None):
            context.abort(error, "configuration read failed")
        # Deliberately reverse responses and omit missing keys to test key mapping.
        return protocol("ConfigResponse", session_id=request.session_id,
                        pairs=[dict(key=k, value=config[k]) for k in reversed(keys)
                               if k in config]
                        ).SerializeToString()

    def statuses(raw, context):
        request = protocol("ArtifactStatusesRequest").FromString(raw)
        return protocol("ArtifactStatusesResponse", session_id=request.session_id,
                        statuses={name: dict(exists=name in state["cache"])
                                  for name in request.names}).SerializeToString()

    def upload(requests, context):
        summaries = []
        name, expected, buffer = None, 0, bytearray()
        session_id = None

        def save(name, data):
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
        state["relations"].append(relation)
        if relation.HasField("chunked_cached_local_relation"):
            cached = relation.chunked_cached_local_relation
            payloads = [state["cache"]["cache/" + h] for h in cached.dataHashes]
            state["spark_schema"] = state["cache"]["cache/" + cached.schemaHash].decode()
        else:
            payloads = [relation.local_relation.data]
        rows, first, last = 0, None, None
        tables = [pa.ipc.open_stream(payload).read_all() for payload in payloads]
        state["input_tables"].append(pa.concat_tables(tables))
        for table in tables:
            for batch in table.to_batches():
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
        session = SparkSession.builder.remote(f"sc://127.0.0.1:{port}").getOrCreate()
        try:
            yield session, state
        finally:
            session.stop()
            server.stop(0).wait(timeout=5)


@pytest.mark.parametrize("rows", [2, 17000000])
def test_arrow_input_upload_above_grpc_limit(protocol, rows):
    import numpy as np
    import json

    table = pa.table({"id": np.arange(rows, dtype=np.int64)})
    with _arrow_input_service(protocol) as (spark, state):
        df = spark.createDataFrame(table)
        assert df.isLocal()
        assert df.toArrow().column(0).to_pylist() == [rows]
        assert state["result"] == (rows, 0, rows - 1)
        if rows == 2:
            assert state["uploads"] == 0
            assert state["relations"][0].HasField("local_relation")
        else:
            assert table.nbytes > 128 * 1024**2
            assert state["relations"][0].HasField("chunked_cached_local_relation")
            assert state["relations"][0].ByteSize() < 1024**2
            assert max(state["message_sizes"]) < 33 * 1024
            assert json.loads(state["spark_schema"])["fields"][0]["name"] == "id"
            uploads = state["uploads"]
            spark.createDataFrame(table).toArrow()
            assert state["uploads"] == uploads


def test_arrow_input_upload_limits_and_chunk_order(protocol):
    import numpy as np
    from pyspark.errors import AnalysisException
    table = pa.table({"id": np.arange(100, dtype=np.int64)})
    options = {
        "spark.sql.session.localRelationCacheThreshold": "0",
        "spark.sql.session.localRelationChunkSizeRows": "7",
        "spark.sql.session.localRelationChunkSizeBytes": "1000",
        "spark.sql.session.localRelationBatchOfChunksSizeBytes": "1000",
    }
    with _arrow_input_service(protocol, **options) as (spark, state):
        df = spark.createDataFrame(table)
        df.toArrow()
        assert state["result"] == (100, 0, 99)
        assert state["uploads"] > 1
        cached = state["relations"][0].chunked_cached_local_relation
        assert len(cached.dataHashes) > 1
        for hash in cached.dataHashes:
            assert pa.ipc.open_stream(state["cache"]["cache/" + hash]).read_all().num_rows <= 7

    options["spark.sql.session.localRelationSizeLimit"] = "10"
    with _arrow_input_service(protocol, **options) as (spark, state):
        with pytest.raises(AnalysisException, match="LOCAL_RELATION_SIZE_LIMIT_EXCEEDED"):
            spark.createDataFrame(table)
        assert not state["relations"]
        assert state["uploads"] == 0


@pytest.mark.parametrize("failure,match", [("crc", "CRC check failed"),
                                          ("missing", "Missing cache artifact")])
def test_arrow_input_upload_failure(protocol, failure, match):
    with _arrow_input_service(protocol, upload_error=failure,
                             **{"spark.sql.session.localRelationCacheThreshold": "0"}) as (spark, state):
        with pytest.raises(Exception, match=match):
            spark.createDataFrame(pa.table({"id": [1, 2]}))
        assert not state["relations"]


def test_arrow_input_duplicate_chunks_preserve_multiplicity(protocol):
    options = {"spark.sql.session.localRelationCacheThreshold": "0",
               "spark.sql.session.localRelationChunkSizeRows": "5"}
    with _arrow_input_service(protocol, **options) as (spark, state):
        spark.createDataFrame(pa.table({"id": [1] * 20})).toArrow()
        cached = state["relations"][0].chunked_cached_local_relation
        assert len(cached.dataHashes) == 4
        assert len(set(cached.dataHashes)) == 1
        assert len(state["cache"]) == 2  # One schema and one unique data chunk.
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
    import json
    from zoneinfo import ZoneInfo

    value = datetime(2024, 1, 15, 12, 34, 56, 123456)
    table = pa.table({"t": [value, None], "s": ["hello", None], "b": [b"xyz", None]})
    with _arrow_input_service(protocol, **dict(zip(
        _NORMALIZATION_KEYS, [timestamp_type, zone, large]
    ))) as (spark, state):
        frame = spark.createDataFrame(table)
        assert state["config_requests"] == [_NORMALIZATION_KEYS, _UPLOAD_KEYS]
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
        fields = json.loads(state["relations"][0].local_relation.schema)["fields"]
        assert [f["type"] for f in fields] == [
            "timestamp_ntz" if timestamp_type == "TIMESTAMP_NTZ" else "timestamp",
            "string", "binary",
        ]


@pytest.mark.parametrize("missing", _NORMALIZATION_KEYS)
@pytest.mark.parametrize("present_null", [False, True])
def test_arrow_input_missing_normalization_config(protocol, missing, present_null):
    with _arrow_input_service(protocol) as (spark, state):
        if present_null:
            state["config"][missing] = None
        else:
            del state["config"][missing]
        spark.createDataFrame(pa.table({"x": [1, None], "s": ["hello", None]})).collect()
        assert state["config_requests"] == [_NORMALIZATION_KEYS, _UPLOAD_KEYS]
        expected = pa.table({"x": pa.array([1, None], type=pa.int64()),
                             "s": ["hello", None]})
        assert state["input_tables"][0].equals(expected)


def test_arrow_input_normalization_config_failure_then_fresh_read(protocol):
    with _arrow_input_service(protocol) as (spark, state):
        state["config_error"] = grpc.StatusCode.INVALID_ARGUMENT
        table = pa.table({"x": [1]})
        with pytest.raises(Exception, match="configuration read failed"):
            spark.createDataFrame(table)
        assert state["config_requests"] == [_NORMALIZATION_KEYS]
        assert state["uploads"] == 0
        spark.createDataFrame(table).collect()
        assert state["config_requests"] == [
            _NORMALIZATION_KEYS, _NORMALIZATION_KEYS, _UPLOAD_KEYS]
        assert state["input_tables"][0].to_pydict() == {"x": [1]}


def test_arrow_input_normalization_config_transient_retry(protocol):
    with _arrow_input_service(protocol) as (spark, state):
        state["config_error"] = grpc.StatusCode.UNAVAILABLE
        spark.createDataFrame(pa.table({"x": [1]})).collect()
        assert state["config_requests"] == [
            _NORMALIZATION_KEYS, _NORMALIZATION_KEYS, _UPLOAD_KEYS]
        assert state["input_tables"][0].to_pydict() == {"x": [1]}


def test_arrow_input_normalization_configs_are_fresh(protocol):
    import json

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
        assert state["config_requests"] == [_NORMALIZATION_KEYS, _UPLOAD_KEYS] * 3
        tables = state["input_tables"]
        assert [t.column("t")[0].as_py() for t in tables] == [
            value.replace(tzinfo=timezone.utc), value,
            datetime(2024, 1, 15, 20, tzinfo=timezone.utc)]
        assert [t.schema.field("s").type for t in tables] == [
            pa.string(), pa.large_string(), pa.string()]
        assert [json.loads(r.local_relation.schema)["fields"][0]["type"]
                for r in state["relations"]] == ["timestamp", "timestamp_ntz", "timestamp"]


def test_arrow_input_empty_skips_all_configs(protocol):
    from pyspark.sql.types import LongType, StructField, StructType

    with _arrow_input_service(protocol) as (spark, state):
        table = pa.table({"x": pa.array([], type=pa.int64())})
        spark.createDataFrame(table, StructType([StructField("x", LongType())]))
        with pytest.raises(Exception, match="CANNOT_INFER_EMPTY_SCHEMA"):
            spark.createDataFrame(table)
        assert state["config_requests"] == []
        assert state["uploads"] == 0


@pytest.mark.parametrize("rows", [10000, 30000, 60000, 1000000])
def test_canonical_arrow_input_matches_normalized_ipc_and_uploads(protocol, monkeypatch, rows):
    import json
    import sys
    import numpy as np
    from pyspark.sql import pandas as pandas_package
    from pyspark.sql.types import LongType, StructField, StructType

    values = np.arange(rows, dtype=np.int64)
    values[0], values[-1] = -(2**63), 2**63 - 1
    # Different column chunk boundaries also exercise native record batch alignment.
    table = pa.table({
        "x": pa.chunked_array([values[:7], values[7:]]),
        "y": pa.chunked_array([pa.array([None], type=pa.int64()),
                               pa.array(values[1:])]),
        "z": pa.chunked_array([values[:31], values[31:]]),
    })
    schema = StructType([StructField(name, LongType()) for name in table.column_names])
    monkeypatch.setattr(pandas_package, "types", getattr(pandas_package, "types", None),
                        raising=False)
    monkeypatch.delitem(sys.modules, "pyspark.sql.pandas.types", raising=False)
    with _arrow_input_service(protocol) as (spark, state):
        spark.createDataFrame(table, schema).collect()
        assert "pyspark.sql.pandas.types" not in sys.modules
        first_uploads = state["uploads"]
        first_cache = dict(state["cache"])
        # A loaded converter deliberately selects the established normalization path.
        from importlib import import_module

        converters = import_module("pyspark.sql.pandas.types")

        assert converters is sys.modules["pyspark.sql.pandas.types"]
        state["config"].update(dict(zip(_NORMALIZATION_KEYS, [
            "TIMESTAMP_NTZ", "America/Los_Angeles", "true"])))
        spark.createDataFrame(table, schema).collect()
        assert state["config_requests"] == [_NORMALIZATION_KEYS, _UPLOAD_KEYS] * 2
        assert state["uploads"] == first_uploads
        assert state["cache"] == first_cache
        for actual in state["input_tables"]:
            assert actual.equals(table, check_metadata=True)
            assert actual.to_pydict() == table.to_pydict()
        first, second = state["relations"]
        assert first.WhichOneof("rel_type") == second.WhichOneof("rel_type")
        if table.nbytes < 1048576:
            assert first.HasField("local_relation")
            assert first.local_relation.data == second.local_relation.data
            assert first.local_relation.schema == second.local_relation.schema
            assert first_uploads == 0
            fields = json.loads(first.local_relation.schema)["fields"]
        else:
            assert first.HasField("chunked_cached_local_relation")
            assert first.chunked_cached_local_relation == second.chunked_cached_local_relation
            assert first_uploads == 1
            fields = json.loads(state["spark_schema"])["fields"]
        assert fields == [dict(name=name, type="long", nullable=True, metadata={})
                          for name in table.column_names]


@pytest.mark.parametrize("failure,match", [
    ("crc", "CRC check failed"), ("missing", "Missing cache artifact"),
    ("limit", "LOCAL_RELATION_SIZE_LIMIT_EXCEEDED"),
    ("config", "configuration read failed"),
])
def test_canonical_arrow_input_preserves_failures(protocol, monkeypatch, failure, match):
    import sys
    from pyspark.sql.types import LongType, StructField, StructType

    monkeypatch.delitem(sys.modules, "pyspark.sql.pandas.types", raising=False)
    options = {"spark.sql.session.localRelationCacheThreshold": "0"}
    if failure == "limit":
        options["spark.sql.session.localRelationSizeLimit"] = "10"
    with _arrow_input_service(protocol, upload_error=failure, **options) as (spark, state):
        if failure == "config":
            state["config_error"] = grpc.StatusCode.INVALID_ARGUMENT
        with pytest.raises(Exception, match=match):
            spark.createDataFrame(pa.table({"x": [1, None, 2]}),
                                  StructType([StructField("x", LongType())]))
        assert "pyspark.sql.pandas.types" not in sys.modules
        assert not state["relations"]
        assert state["uploads"] == (1 if failure in ("crc", "missing") else 0)
        assert state["config_requests"] == (
            [_NORMALIZATION_KEYS] if failure == "config" else [_NORMALIZATION_KEYS, _UPLOAD_KEYS]
        )


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


def test_normal_handoff_does_not_reopen_ipc(protocol, monkeypatch):
    def unexpected_ipc(*args, **kwargs):
        pytest.fail("Normal toArrow handoff must use the C Stream importer")

    monkeypatch.setattr(pa.ipc, "open_file", unexpected_ipc)
    batch = _mixed_batch()
    with _dataframe(protocol=protocol, batches=[batch]) as (frame, _):
        assert frame.toArrow().equals(pa.Table.from_batches([batch]), check_metadata=True)
