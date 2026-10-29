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

import sys
from types import GetSetDescriptorType, WrapperDescriptorType

from pyspark._pyspark import (
    BooleanType,
    ByteType,
    IntegerType,
    LongType,
    ShortType,
    StructField,
    StructType as NativeStructType,
)
from pyspark.sql import types as sql_types
from pyspark.sql.types import StructType


def _canonical_long_table(table, schema):
    # A loaded converter may have customized hooks. Keep its normal behavior,
    # and avoid importing it solely to normalize an already canonical table.
    if ("pyspark.sql.pandas.types" in sys.modules
            or sys.modules.get("pyspark.sql.types") is not sql_types
            or type(schema) is not NativeStructType):
        return False
    for name, cls in (("BooleanType", BooleanType), ("ByteType", ByteType),
                      ("ShortType", ShortType), ("IntegerType", IntegerType),
                      ("LongType", LongType), ("StructField", StructField),
                      ("StructType", NativeStructType)):
        if getattr(sql_types, name, None) is not cls:
            return False
    if LongType.__getattribute__ is not object.__getattribute__:
        return False
    # Native classes reject subclasses, but their class attributes are mutable.
    # Reject customized access/iteration, including changes made before import.
    for cls, names, descriptor_type in (
        (NativeStructType, ("fields", "names"), GetSetDescriptorType),
        (NativeStructType, ("__iter__",), WrapperDescriptorType),
        (StructField, ("name", "dataType", "nullable", "metadata"), GetSetDescriptorType),
    ):
        if cls.__getattribute__ is not object.__getattribute__:
            return False
        for name in names:
            descriptor = vars(cls).get(name)
            if (type(descriptor) is not descriptor_type
                    or descriptor.__objclass__ is not cls or descriptor.__name__ != name):
                return False

    import pyarrow as pa

    if type(table) is not pa.Table:
        return False
    arrow_schema = table.schema
    fields = schema.fields
    if (not fields or len(fields) != len(arrow_schema)
            or arrow_schema.metadata is not None):
        return False
    names = set()
    for field, arrow_field in zip(fields, arrow_schema):
        if (type(field) is not StructField or type(field.dataType) is not LongType
                or field.nullable is not True or field.metadata
                or field.name != arrow_field.name or field.name in names
                or arrow_field.type != pa.int64() or arrow_field.nullable is not True
                or arrow_field.metadata is not None):
            return False
        names.add(field.name)
    return True


def prepare_table(table, schema, timestamp_type, timezone, large_types):
    """Match Connect's Arrow casts without converting any column to Python rows."""
    names = list(schema) if isinstance(schema, (list, tuple)) else None
    explicit = schema is not None and names is None
    if len(table) == 0:
        if not explicit:
            from pyspark.errors import PySparkValueError

            raise PySparkValueError(errorClass="CANNOT_INFER_EMPTY_SCHEMA", messageParameters={})
        return table, schema, None

    if explicit and _canonical_long_table(table, schema):
        return table, schema, None

    from pyspark.sql.pandas.types import (
        _check_arrow_table_timestamps_localize,
        from_arrow_schema,
        to_arrow_schema,
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
        from pyspark.errors import PySparkValueError

        raise PySparkValueError(
            errorClass="AXIS_LENGTH_MISMATCH",
            messageParameters={
                "expected_length": str(len(names)),
                "actual_length": str(normalized.num_columns),
            },
        )
    return normalized, schema if explicit else inferred, names
