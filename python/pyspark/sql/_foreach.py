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
"""``DataFrame.foreach`` and ``DataFrame.foreachPartition``, as PySpark Connect's.

The functions that call ``f`` are pickled with cloudpickle and run by the server's Python
worker, so they are Python, as PySpark's ``pyspark.sql.connect.dataframe`` builds them.
"""
from typing import TYPE_CHECKING, Any, Callable, Iterator

from pyspark.sql import functions as F
from pyspark.sql.conversion import ArrowTableToRowsConversion
from pyspark.sql.types import Row, StructType, _create_row

if TYPE_CHECKING:
    import pyarrow as pa

    from pyspark.sql.dataframe import DataFrame


def foreach(df: "DataFrame", f: Callable[[Row], None]) -> None:
    def foreach_func(row: Any) -> None:
        f(row)

    df.select(F.struct(*df.schema.fieldNames()).alias("row")).select(
        F.udf(foreach_func, StructType())("row")  # type: ignore[arg-type]
    ).collect()


def foreachPartition(df: "DataFrame", f: Callable[[Iterator[Row]], None]) -> None:
    schema = df.schema
    conf_value = df.sparkSession.conf.get("spark.sql.execution.pyspark.binaryAsBytes", "true")
    binary_as_bytes = conf_value is not None and conf_value.lower() == "true"
    field_converters = [
        ArrowTableToRowsConversion._create_converter(
            f.dataType, none_on_identity=False, binary_as_bytes=binary_as_bytes
        )
        for f in schema.fields
    ]

    def foreach_partition_func(itr: Iterator["pa.RecordBatch"]) -> Iterator["pa.RecordBatch"]:
        def flatten() -> Iterator[Row]:
            for table in itr:
                columnar_data = [
                    ArrowTableToRowsConversion._to_pylist(column) for column in table.columns
                ]
                for i in range(0, table.num_rows):
                    values = [
                        field_converters[j](columnar_data[j][i])  # type: ignore[misc]
                        for j in range(table.num_columns)
                    ]
                    yield _create_row(fields=schema.fieldNames(), values=values)

        f(flatten())
        return iter([])

    df.mapInArrow(foreach_partition_func, schema=StructType()).collect()
