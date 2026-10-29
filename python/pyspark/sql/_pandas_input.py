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


"""Conservative columnar pandas input; unsupported conversions use the row path."""

from pyspark.sql.types import LongType, StructType


def prepare_int64_table(frame, schema):
    """Return an Arrow table only when each column is already a signed 64-bit integer.

    In particular, never coerce a mixed pandas frame to a common NumPy dtype:
    doing so can round integers above 2**53. Other dtypes and schema conversions
    deliberately retain the existing row path until their semantics are covered.
    ``verifySchema`` is ignored for pandas Arrow input, as in Spark Connect.
    """
    import numpy as np
    import pandas as pd

    width = len(frame.columns)
    if not width or not frame.columns.is_unique or not all(
        isinstance(name, str) for name in frame.columns
    ):
        return None
    if not all(dtype == np.dtype("int64") or isinstance(dtype, pd.Int64Dtype)
               for dtype in frame.dtypes):
        return None
    if isinstance(schema, StructType):
        if len(schema.fields) != width or not all(
            isinstance(field.dataType, LongType) for field in schema.fields
        ):
            return None
    elif schema is not None:
        if not isinstance(schema, (list, tuple)) or len(schema) != width or not all(
            isinstance(name, str) for name in schema
        ):
            return None

    import pyarrow as pa

    # Arrow consumes each Series' integer buffer and nullable mask independently.
    return pa.Table.from_arrays(
        [pa.array(frame.iloc[:, i], type=pa.int64(), from_pandas=True) for i in range(width)],
        names=list(frame.columns),
    )
