# SPDX-License-Identifier: Apache-2.0
"""Capture the literals of official PySpark's ML (de)serialization for tests/golden/ml_values.json.

Run with official pyspark-client 4.3.0 (no server needed):
    python scripts/capture_ml_values_golden.py > tests/golden/ml_values.json
Each case is the base64 of `serialize_param(value)` (or of a `(value, DataType)` pair as `serialize`
sends it), and `deserialize_param` of it, rendered as a typed tree.
"""

import base64
import json

from pyspark.ml.connect.serialize import deserialize_param, serialize, serialize_param
from pyspark.ml.linalg import DenseMatrix, DenseVector, SparseMatrix, SparseVector
from pyspark.sql.types import ArrayType, DoubleType, StringType


def render(value):
    if value is None:
        return {"null": None}
    if isinstance(value, bool):
        return {"bool": value}
    if isinstance(value, int):
        return {"int": value}
    if isinstance(value, float):
        return {"double": value}
    if isinstance(value, str):
        return {"string": value}
    if isinstance(value, DenseVector):
        return {"dense_vector": value.values.tolist()}
    if isinstance(value, SparseVector):
        return {"sparse_vector": [value.size, value.indices.tolist(), value.values.tolist()]}
    if isinstance(value, DenseMatrix):
        return {
            "dense_matrix": [
                value.numRows,
                value.numCols,
                value.values.tolist(),
                value.isTransposed,
            ]
        }
    if isinstance(value, SparseMatrix):
        return {
            "sparse_matrix": [
                value.numRows,
                value.numCols,
                value.colPtrs.tolist(),
                value.rowIndices.tolist(),
                value.values.tolist(),
                value.isTransposed,
            ]
        }
    if isinstance(value, list):
        return {"array": [render(v) for v in value]}
    raise TypeError(type(value))


CASES = {
    "double": 1.5,
    "int": 7,
    "long": 2**40,
    "string": "s",
    "bool": True,
    "null": None,
    "double_array": [1.0, 2.5],
    "string_array": ["a", "b"],
    "int_array": [1, 2],
    "nested_double_array": [[1.0], [2.0, 3.0]],
    "dense_vector": DenseVector([1.0, 2.0]),
    "sparse_vector": SparseVector(3, [1], [2.0]),
    "dense_matrix": DenseMatrix(1, 2, [0.0, -1.0]),
    "dense_matrix_transposed": DenseMatrix(2, 1, [3.0, 4.0], True),
    "sparse_matrix": SparseMatrix(2, 2, [0, 1, 2], [0, 1], [1.0, 2.0]),
}
TYPED = {
    "typed_empty_string_array": ([], ArrayType(StringType())),
    "typed_double_array": ([1.0, 2.0], ArrayType(DoubleType())),
}

out = {}
for name, value in CASES.items():
    literal = serialize_param(value, None)
    out[name] = {
        "literal_b64": base64.b64encode(literal.SerializeToString(deterministic=True)).decode(),
        "value": render(deserialize_param(literal)),
    }
for name, pair in TYPED.items():
    (arg,) = serialize(None, pair)
    literal = arg.param
    out[name] = {
        "literal_b64": base64.b64encode(literal.SerializeToString(deterministic=True)).decode(),
        "value": render(deserialize_param(literal)),
    }
print(json.dumps(out, indent=1, sort_keys=True))
