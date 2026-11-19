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
from pyspark.sql.types import AtomicType, StructField, StructType


def prepare_table(table, schema, timestamp_type, timezone, large_types, pandas=False):
    """Match Connect's Arrow casts without converting any column to Python rows. A table of the
    columns of a pandas DataFrame (`pandas`) is cast as PySpark's conversion of the frame casts
    it, to the Arrow types of `_deduplicate_field_names` of the schema, whose fields keep no
    metadata, but the fields of the SQL type of a `UserDefinedType`, of the names of the schema."""
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

    if names is not None:
        names.extend(f"_{i + 1}" for i in range(len(names), table.num_columns))
    inferred = (
        schema
        if isinstance(schema, StructType)
        else from_arrow_schema(table.schema, prefer_timestamp_ntz=timestamp_type == "TIMESTAMP_NTZ")
    )
    # A struct of other names than the table's, also of another number of fields, fails the
    # cast with PyArrow's ValueError, or the assert of the localization of nested or timestamp
    # columns before it, as PySpark Connect's.
    if pandas:
        import pyarrow as pa

        from pyspark.sql.pandas.types import _deduplicate_field_names

        deduplicated = to_arrow_schema(
            _deduplicate_field_names(inferred),
            timezone="UTC",
            prefers_large_types=large_types == "true",
        )
        arrow_schema = pa.schema(
            [
                pa.field(name, field.type, field.nullable, field.metadata)
                for name, field in zip(inferred.names, deduplicated)
            ]
        )
    else:
        arrow_schema = to_arrow_schema(
            inferred,
            error_on_duplicated_field_names_in_struct=True,
            timezone="UTC",
            prefers_large_types=large_types == "true",
        )
    normalized = (
        _check_arrow_table_timestamps_localize(table, inferred, True, timezone)
        .cast(arrow_schema)
        .rename_columns(inferred.names)
    )
    # Then the number of columns of the schema, one of an atomic type, or of the names.
    if isinstance(schema, StructType):
        expected_columns = len(schema.fields)
    elif isinstance(schema, AtomicType):
        expected_columns = 1
    else:
        expected_columns = None if names is None else len(names)
    if expected_columns is not None and expected_columns != normalized.num_columns:
        raise PySparkValueError(
            errorClass="AXIS_LENGTH_MISMATCH",
            messageParameters={
                "expected_length": str(expected_columns),
                "actual_length": str(normalized.num_columns),
            },
        )
    # Match Connect: inferred Arrow schemas travel in IPC, without an extra
    # Spark schema that would trigger server-side casts on every query.
    if not explicit and names is None:
        names = table.column_names
    return normalized, schema if explicit else None, names


def ndarray_table(data, names):
    """The Arrow table of a numpy array of 1 or 2 dimensions, of a column of each column of the
    array, named `names`, as PySpark Connect's createDataFrame builds it."""
    import pyarrow as pa

    if data.ndim == 1:
        return pa.Table.from_arrays([pa.array(data)], names)
    return pa.Table.from_arrays([pa.array(data[::, i]) for i in range(0, data.shape[1])], names)



def local_rows(data):
    """The rows of the local data `data`, as PySpark Connect's createDataFrame takes them: dicts of
    sorted keys, and a value that is no row, such as 1, as the row of the value."""
    from pyspark.sql.types import Row

    rows = list(data)
    if isinstance(rows[0], dict):
        rows = [dict(sorted(d.items())) if d is not None else None for d in rows]
    elif not isinstance(rows[0], (Row, tuple, list, dict)) and not hasattr(rows[0], "__dict__"):
        rows = [[d] for d in rows]
    return rows


def convert_local_rows(rows, schema, large_types):
    """The Arrow table of the rows `rows` (of `local_rows`) under the struct `schema`, as PySpark
    Connect's createDataFrame converts them (`LocalDataToArrowConversion`), of its errors. The
    drop-in converts the rows of a schema that it lowers itself: this converts those of a schema
    that it does not lower, such as one of a `UserDefinedType` whose `module()` or `sqlType()` is
    not implemented, whose converters call its `sqlType()` and `serialize`."""
    from pyspark.sql.conversion import LocalDataToArrowConversion

    return LocalDataToArrowConversion.convert(rows, schema, large_types)


def local_arrow_schema(schema, large_types):
    """The Arrow schema of local rows under the struct `schema`, as PySpark Connect's conversion
    makes it after it converts their values (`LocalDataToArrowConversion.convert`), of its errors:
    of the fields of deduplicated names, which keep no metadata, but the fields of the SQL type of a
    `UserDefinedType`, whose metadata `json.dumps` writes."""
    from pyspark.sql.pandas.types import _deduplicate_field_names

    return to_arrow_schema(
        StructType(
            [
                StructField(field.name, _deduplicate_field_names(field.dataType), field.nullable)
                for field in schema.fields
            ]
        ),
        timezone="UTC",
        prefers_large_types=large_types,
    )


def infer_local_schema(rows, names, dict_as_struct, array_from_first_element, map_from_first_pair,
                       prefer_timestamp_ntz):
    """The schema of the rows `rows` (of `local_rows`) and the names of its fields, as PySpark
    Connect's `_inferSchemaFromList` infers them, whose `_infer_schema` adds a name to the names of
    a row of more values."""
    import functools

    from pyspark.sql.types import _infer_schema, _merge_type

    names = None if names is None else list(names)
    schema = functools.reduce(
        _merge_type,
        (
            _infer_schema(
                row,
                names,
                infer_dict_as_struct=dict_as_struct,
                infer_array_from_first_element=array_from_first_element,
                infer_map_from_first_pair=map_from_first_pair,
                prefer_timestamp_ntz=prefer_timestamp_ntz,
            )
            for row in rows
        ),
    )
    return schema, names
