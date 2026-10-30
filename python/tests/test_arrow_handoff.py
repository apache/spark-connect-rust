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
def _arrow_input_service(protocol, *, upload_error=None, config_error=False, **overrides):
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
             "input_tables": [], "config_requests": []}
    server_id = "00000000-0000-0000-0000-000000000001"

    def configure(raw, context):
        request = protocol("ConfigRequest").FromString(raw)
        state["config_requests"].append(list(request.operation.get.keys))
        if config_error:
            context.abort(grpc.StatusCode.UNAVAILABLE, "configuration unavailable")
        return protocol("ConfigResponse", session_id=request.session_id,
                        pairs=[dict(key=k, value=config[k]) for k in request.operation.get.keys]
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
        # Column-name schemas wrap the local input with a rename relation.
        if relation.HasField("to_df"):
            relation = relation.to_df.input
        if relation.HasField("chunked_cached_local_relation"):
            cached = relation.chunked_cached_local_relation
            payloads = [state["cache"]["cache/" + h] for h in cached.dataHashes]
            state["spark_schema"] = state["cache"]["cache/" + cached.schemaHash].decode()
        else:
            payloads = [relation.local_relation.data]
        rows, first, last = 0, None, None
        batches = []
        for payload in payloads:
            for batch in pa.ipc.open_stream(payload):
                batches.append(batch)
                rows += batch.num_rows
                if batch.num_rows:
                    if first is None:
                        first = batch.column(0)[0].as_py()
                    last = batch.column(0)[-1].as_py()
        state["result"] = (rows, first, last)
        state["input_tables"].append(pa.Table.from_batches(batches))
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


# Signed-int64 pandas input uses the same Arrow importer and mock service.

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
        if schema_kind != "names":
            assert df.isLocal()
        assert df.toArrow().column(0).to_pylist() == [3]
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
    with _arrow_input_service(protocol, config_error=config_error) as (spark, state):
        with pytest.raises(RuntimeError, match="null"):
            spark.createDataFrame(frame, schema, verifySchema=verify)
        assert not state["relations"]
        assert not state["uploads"]
        assert state["config_requests"] == []


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
        assert pa.ipc.open_stream(payload).schema.names == [name]


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
            assert state["config_requests"] == []


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
