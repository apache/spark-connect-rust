import uuid

import pyarrow as pa
import pytest
from _mock_connect_server import data_type_proto, message_field, varint_field
from _row_conversion_probe import RowConversionProbe
from pyspark.sql import Row
from pyspark.sql import types as T


def conversion_result(service, layout, count, field_name):
    child = pa.struct([(field_name, pa.int64())])
    values = [{field_name: index} for index in range(count)]
    child_schema = T.StructType([T.StructField(field_name, T.LongType())])
    if layout == "rows":
        table = pa.table({field_name: pa.array(range(count), pa.int64())})
        schema = child_schema
    elif layout == "array":
        table = pa.table({"items": pa.array([values], pa.list_(child))})
        schema = T.StructType([T.StructField("items", T.ArrayType(child_schema))])
    else:
        table = pa.table({
            "items": pa.array([list(enumerate(values))], pa.map_(pa.int64(), child)),
        })
        schema = T.StructType([
            T.StructField("items", T.MapType(T.LongType(), child_schema)),
        ])
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    service.execute_payload = message_field(7, data_type_proto(schema.jsonValue())) + message_field(
        2, varint_field(1, table.num_rows) + message_field(2, sink.getvalue().to_pybytes()))
    return table.num_rows


class TestRowConversionInterrupts:
    @pytest.mark.parametrize("layout", ["rows", "array", "map"])
    @pytest.mark.parametrize("cancel", [False, True])
    def test_python_thread_runs_before_native_materialization_finishes(
        self, mock_connect, layout, cancel,
    ):
        spark, service = mock_connect
        count = 131072
        field_name = f"conversion_{uuid.uuid4().hex}"
        row_count = conversion_result(service, layout, count, field_name)
        frame = spark.range(row_count)
        probe = RowConversionProbe(Row, field_name, count, cancel=cancel)
        interrupted = False
        rows = []
        with probe:
            try:
                rows = probe.collect(frame.collect)
            except KeyboardInterrupt:
                interrupted = True
            if not cancel:
                assert probe.snapshot()["rows"] == count, probe.observed
        assert not probe.failures, probe.failures
        snapshot = probe.observed.get("progress_snapshot", {})
        assert 0 < snapshot.get("rows", 0) < count, probe.observed
        assert snapshot["captured_at"] < probe.observed["collect_ended_at"], probe.observed
        assert interrupted is cancel
        if cancel:
            assert 0 < probe.observed["signal_snapshot"]["rows"] < count, probe.observed
        if not cancel:
            assert len(rows) == row_count
            if layout == "rows":
                assert rows[0][0] == 0 and rows[-1][0] == count - 1
                assert rows[0].__fields__ is rows[-1].__fields__
            else:
                items = rows[0][0]
                assert items[0][0] == 0 and items[count - 1][0] == count - 1
                assert items[0].__fields__ is items[count - 1].__fields__
        else:
            conversion_result(service, "rows", 2, field_name)
            assert [tuple(row) for row in spark.range(2).collect()] == [(0,), (1,)]
