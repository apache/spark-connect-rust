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


"""Schema normalization for the native Arrow Table input path."""

from pyspark.errors import PySparkTypeError, PySparkValueError
from pyspark.sql.pandas.types import (
    _check_arrow_table_timestamps_localize,
    from_arrow_schema,
    to_arrow_schema,
)
from pyspark.sql.types import AtomicType, StructType


def prepare_table(table, schema, timestamp_type, timezone, large_types):
    """Match Connect's Arrow casts without converting any column to Python rows."""
    if schema is not None and not isinstance(schema, (AtomicType, StructType, list, tuple)):
        raise PySparkTypeError(
            errorClass="NOT_EXPECTED_TYPE",
            messageParameters={
                "arg_name": "schema",
                "expected_type": "list, None or StructType",
                "arg_type": type(schema).__name__,
            },
        )
    names = list(schema) if isinstance(schema, (list, tuple)) else None
    explicit = schema is not None and names is None
    if len(table) == 0:
        if not explicit:
            raise PySparkValueError(errorClass="CANNOT_INFER_EMPTY_SCHEMA", messageParameters={})
        return table, schema, None

    expected_columns = None
    if isinstance(schema, StructType):
        expected_columns = len(schema.fields)
    elif isinstance(schema, AtomicType):
        expected_columns = 1
    if expected_columns is not None and expected_columns != table.num_columns:
        raise PySparkValueError(
            errorClass="AXIS_LENGTH_MISMATCH",
            messageParameters={
                "expected_length": str(expected_columns),
                "actual_length": str(table.num_columns),
            },
        )

    if names is not None:
        names.extend(f"_{i + 1}" for i in range(len(names), table.num_columns))
    inferred = (
        schema
        if isinstance(schema, StructType)
        else from_arrow_schema(table.schema, prefer_timestamp_ntz=timestamp_type == "TIMESTAMP_NTZ")
    )
    normalized = (
        _check_arrow_table_timestamps_localize(table, inferred, True, timezone)
        .cast(
            to_arrow_schema(
                inferred,
                error_on_duplicated_field_names_in_struct=True,
                timezone="UTC",
                prefers_large_types=large_types == "true",
            )
        )
        .rename_columns(inferred.names)
    )
    if names is not None and len(names) != normalized.num_columns:
        raise PySparkValueError(
            errorClass="AXIS_LENGTH_MISMATCH",
            messageParameters={
                "expected_length": str(len(names)),
                "actual_length": str(normalized.num_columns),
            },
        )
    return normalized, schema if explicit else inferred, names
