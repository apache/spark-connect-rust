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
        pytest.skip("Arrow handoff tests require protoc; install it or set PROTOC")
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


@pytest.mark.parametrize("kind", ["field", "nested-field", "timezone"])
def test_nul_schema_strings_fall_back_without_executing_again(protocol, kind):
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


@pytest.mark.parametrize("difference", ["type", "name", "schema-metadata", "field-metadata"])
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
        second_name, second_values.type, metadata=field_metadata or first_field.metadata
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
def _arrow_input_service(protocol, *, upload_error=None, capture_tables=False, config_error=False, **overrides):
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
    state = {"cache": {}, "uploads": 0, "relations": [], "message_sizes": [], "tables": [], "config_requests": []}
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
        for payload in payloads:
            if capture_tables:
                state["tables"].append(pa.ipc.open_stream(payload).read_all())
            for batch in pa.ipc.open_stream(payload):
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

    def no_rows(*args, **kwargs):
        raise AssertionError("supported timestamps must not materialize Python rows")

    monkeypatch.setattr(pd.DataFrame, "astype", no_rows)
    with _arrow_input_service(protocol, capture_tables=True, **{"spark.sql.session.timeZone": zone}) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
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
    monkeypatch.setattr(pd.DataFrame, "astype", lambda *a, **k: pytest.fail("row conversion"))
    with _arrow_input_service(protocol, capture_tables=True) as (spark, state):
        spark.createDataFrame(frame, schema).toArrow()
        assert state["tables"][0].column(0).cast(pa.int64()).to_pylist() == [
            -1, None, 1577836800123456]


@pytest.mark.parametrize("dtype", ["datetime64[ns]", "datetime64[us]", "datetime64[ms]",
                                    "datetime64[s]", "object"])
def test_pandas_timestamp_eligibility(dtype):
    from pyspark.sql import types as T
    from pyspark.sql._pandas_input import timestamp_columns
    frame = pd.DataFrame({"a": pd.Series([datetime(2020, 1, 1), None], dtype=dtype)})
    schema = T.StructType([T.StructField("a", T.TimestampType())])
    assert (timestamp_columns(frame, schema) is not None) == (dtype in (
        "datetime64[ns]", "datetime64[us]"))
    for unsupported in (None, ["a"], "a timestamp",
                        T.StructType([T.StructField("a", T.TimestampNTZType())]),
                        T.StructType([T.StructField("a", T.LongType())])):
        assert timestamp_columns(frame, unsupported) is None


def test_pandas_timestamp_precision_and_timezone_fallback(monkeypatch):
    from pyspark.sql import types as T
    from pyspark.sql._pandas_input import timestamp_columns, prepare_timestamp_table
    schema = T.StructType([T.StructField("a", T.TimestampType())])
    for nanos in (-1, 1, 1001):
        frame = pd.DataFrame({"a": pd.to_datetime([nanos, None], unit="ns")})
        assert timestamp_columns(frame, schema) is None
    frame = pd.DataFrame({"a": pd.to_datetime([0, None], unit="ns")})
    columns = timestamp_columns(frame, schema)
    monkeypatch.setenv("TZ", "UTC")
    assert prepare_timestamp_table(columns, schema, "America/Los_Angeles") is None
    monkeypatch.setenv("TZ", "America/Los_Angeles")
    assert prepare_timestamp_table(columns, schema, "UTC") is None
    monkeypatch.delenv("TZ")
    assert prepare_timestamp_table(columns, schema, "UTC") is None


@pytest.mark.parametrize("has_null", [False, True])
def test_pandas_timestamp_non_nullable_schema(protocol, has_null):
    from pyspark.sql import types as T
    values = [pd.Timestamp("2020-01-01", tz="UTC"),
              pd.NaT if has_null else pd.Timestamp("2020-01-02", tz="UTC")]
    frame = pd.DataFrame({"a": values})
    schema = T.StructType([T.StructField("a", T.TimestampType(), False)])
    with _arrow_input_service(protocol, capture_tables=True) as (spark, state):
        if has_null:
            with pytest.raises(RuntimeError, match="null"):
                spark.createDataFrame(frame, schema)
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
def test_timestamp_nonnullable_nat_keeps_local_error_before_config(protocol, config_error, verify):
    from pyspark.sql import types as T
    from pyspark.sql._pandas_input import timestamp_columns

    frame = pd.DataFrame({"a": pd.to_datetime([0, None], utc=True)})
    schema = T.StructType([T.StructField("renamed", T.TimestampType(), False)])
    assert timestamp_columns(frame, schema) is None
    with _arrow_input_service(protocol, capture_tables=True, config_error=config_error) as (spark, state):
        with pytest.raises(RuntimeError, match="null"):
            spark.createDataFrame(frame, schema, verifySchema=verify)
        assert state["config_requests"] == []
        assert state["relations"] == []
        assert state["uploads"] == 0


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


def test_timestamp_warm_local_converter_imports_keep_columnar_route(protocol, monkeypatch):
    import types
    import sys
    import pyspark.sql._arrow_input as importer
    from pyspark.sql import types as T
    from pyspark.sql._pandas_input import timestamp_columns

    original = importer.prepare_table
    substitute = types.ModuleType(importer.__name__)
    substitute.StructType = T.StructType

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
        monkeypatch.setattr(pd.DataFrame, "astype", lambda *a, **k: pytest.fail("row conversion"))
        spark.createDataFrame(frame, schema).toArrow()
        table = state["tables"][-1]
        assert table.column_names == ["renamed"]
        assert table.column(0).cast(pa.int64()).to_pylist() == [0, 1000, None]
