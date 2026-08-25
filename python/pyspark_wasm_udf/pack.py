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
"""Pack a WASM UDF into a Spark ``PythonUDF`` command.

Invoked by the Rust client as ``python -m pyspark_wasm_udf.pack``. Reads a JSON
spec from stdin and writes the raw cloudpickled ``command`` bytes to stdout.

Input JSON (stdin)::

    {
      "wasm_b64":    "<base64 of the .wasm module>",
      "entrypoint":  "add_one",
      "arg_types":   ["i64", "array:string", ...],   # ABI descriptors
      "ret_type":    "i64",
      "output_type": "long"                           # Spark type JSON value
    }

``output_type`` is Spark's canonical type-JSON value (a string like ``"long"``
for atomic types, or an object like ``{"type": "array", ...}``); it is parsed
generically with :func:`pyspark.sql.types._parse_datatype_json_value`, so any
Spark type is supported without a token table.

The command mirrors ``pyspark.sql.connect.expressions.PythonUDF``:
``cloudpickle.dumps((func, output_type))``. We use the bundled
``pyspark.cloudpickle`` (the same cloudpickle the reference client ships) and
register :mod:`pyspark_wasm_udf` for pickling **by value**, so executors need not
have this package installed.
"""

import base64
import json
import sys

from pyspark import cloudpickle

import pyspark_wasm_udf
from pyspark_wasm_udf import WasmScalarUDF

from pyspark.sql.types import _parse_datatype_json_value


def build_command(spec: dict) -> bytes:
    wasm = base64.b64decode(spec["wasm_b64"])
    runner = WasmScalarUDF(
        wasm,
        spec["entrypoint"],
        spec["arg_types"],
        spec["ret_type"],
    )
    # `output_type` is an already-parsed JSON value (str or dict), so use
    # `_parse_datatype_json_value` (pure Python; no JVM), not the string form.
    output_type = _parse_datatype_json_value(spec["output_type"])

    cloudpickle.register_pickle_by_value(pyspark_wasm_udf)
    return cloudpickle.dumps((runner, output_type))


def main() -> None:
    spec = json.loads(sys.stdin.read())
    sys.stdout.buffer.write(build_command(spec))
    sys.stdout.buffer.flush()


if __name__ == "__main__":
    main()
