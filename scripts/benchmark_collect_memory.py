#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
"""Measure collection with the installed client against a local protocol fixture.

Activate the Rust-backed client and run each build in a fresh process, with the
same Python and dependency versions:
    python scripts/benchmark_collect_memory.py --kind string
    python scripts/benchmark_collect_memory.py --kind binary

Requires pyarrow and grpcio, but no Spark server. Peak RSS includes client-side
fixture setup and excludes the separately hosted server. Supported on Unix.
"""

import argparse
import gc
import json
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python" / "tests"))


def main():
    import pyarrow as pa
    from _mock_connect_server import (
        data_type_proto,
        message_field,
        mock_connect_session,
        varint_field,
    )
    from pyspark import _pyspark
    from pyspark.sql import types as T

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["string", "binary"], default="string")
    parser.add_argument("--rows", type=int, default=131072)
    parser.add_argument("--batch-rows", type=int, default=4096)
    parser.add_argument("--width", type=int, default=1024)
    args = parser.parse_args()
    if min(args.rows, args.batch_rows, args.width) <= 0 or args.rows % args.batch_rows:
        parser.error("sizes must be positive and rows must be divisible by batch-rows")

    values = [str(i).zfill(args.width) for i in range(args.batch_rows)]
    if args.kind == "binary":
        values = [value.encode() for value in values]
    data_type = T.BinaryType() if args.kind == "binary" else T.StringType()
    table = pa.table({"left": values, "right": values})
    schema = T.StructType([T.StructField(name, data_type) for name in table.column_names])
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    payload = message_field(7, data_type_proto(schema.jsonValue())) + message_field(
        2, varint_field(1, args.batch_rows) + message_field(2, sink.getvalue().to_pybytes())
    )
    with mock_connect_session() as (spark, service):
        service.execute_payloads = [payload] * (args.rows // args.batch_rows)
        frame = spark.range(args.rows)
        assert isinstance(frame, _pyspark.DataFrame), "Activate the Rust-backed client"
        gc.collect()
        start, cpu = time.perf_counter(), time.process_time()
        rows = frame.collect()
        wall, cpu = time.perf_counter() - start, time.process_time() - cpu
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        assert len(rows) == args.rows
        assert rows[0] == (values[0], values[0])
        assert rows[-1] == (values[-1], values[-1])
        print(
            json.dumps(
                {
                    **vars(args),
                    "wall_s": wall,
                    "cpu_s": cpu,
                    "peak_rss_bytes": peak * (1 if sys.platform == "darwin" else 1024),
                }
            )
        )


if __name__ == "__main__":
    main()
