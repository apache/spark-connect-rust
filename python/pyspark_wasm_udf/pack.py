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
"""Pack a WASM scalar UDF into a Spark ``PythonUDF`` command.

Invoked by the Rust client as ``python -m pyspark_wasm_udf.pack``. Reads a JSON
spec from stdin and writes the raw cloudpickled ``command`` bytes to stdout.

Input JSON (stdin)::

    {
      "wasm_b64":   "<base64 of the .wasm module>",
      "entrypoint": "run",
      "arg_types":  ["i64", "i64"],
      "ret_type":   "i64",
      "output_type": "long"      # a Spark atomic type token, or "string"
    }

The command mirrors ``pyspark.sql.connect.expressions.PythonUDF``, whose command
is ``cloudpickle.dumps((func, output_type))``. We use the standalone PyPI
``cloudpickle`` package (byte-for-byte the same as the copy bundled in
``pyspark.cloudpickle``). The ``func`` here is a
:class:`~pyspark_wasm_udf.WasmScalarUDF`, serialized **by value** via
``cloudpickle.register_pickle_by_value`` so executors need not have this package
installed. ``pyspark.sql.types`` is still required to build the output-type
object the command carries.
"""

import base64
import json
import sys

import cloudpickle

import pyspark_wasm_udf
from pyspark_wasm_udf import WasmScalarUDF

from pyspark.sql.types import (
    BinaryType,
    BooleanType,
    ByteType,
    DataType,
    DateType,
    DoubleType,
    FloatType,
    IntegerType,
    LongType,
    NullType,
    ShortType,
    StringType,
    TimestampNTZType,
    TimestampType,
)

# Atomic Spark output types the prototype supports, keyed by the token the Rust
# client sends (see `spark_connect::wasm_udf::output_type_token`).
_OUTPUT_TYPES = {
    "null": NullType,
    "boolean": BooleanType,
    "byte": ByteType,
    "short": ShortType,
    "integer": IntegerType,
    "long": LongType,
    "float": FloatType,
    "double": DoubleType,
    "binary": BinaryType,
    "date": DateType,
    "timestamp": TimestampType,
    "timestamp_ntz": TimestampNTZType,
    "string": StringType,
}


def build_command(spec: dict) -> bytes:
    wasm = base64.b64decode(spec["wasm_b64"])
    runner = WasmScalarUDF(
        wasm,
        spec["entrypoint"],
        spec["arg_types"],
        spec["ret_type"],
    )

    token = spec["output_type"]
    if token not in _OUTPUT_TYPES:
        raise ValueError(f"unsupported output_type token: {token!r}")
    output_type: DataType = _OUTPUT_TYPES[token]()

    # Force by-value serialization of the runner class so the executors do not
    # need `pyspark_wasm_udf` installed.
    cloudpickle.register_pickle_by_value(pyspark_wasm_udf)
    return cloudpickle.dumps((runner, output_type))


def main() -> None:
    spec = json.loads(sys.stdin.read())
    sys.stdout.buffer.write(build_command(spec))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
