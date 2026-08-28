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
"""Register dataflow-graph elements in a Spark Connect server.

Mirrors upstream ``pyspark.pipelines.spark_connect_graph_element_registry`` but builds
and sends the ``PipelineCommand`` protos through the Rust core
(``pyspark._pyspark.pipeline_*``) rather than assembling ``pb2`` messages and calling
the Python gRPC client.
"""
from pathlib import Path
from typing import Any, cast

from pyspark.errors import PySparkTypeError
from pyspark.sql import SparkSession
from pyspark.pipelines.output import (
    Output,
    MaterializedView,
    Sink,
    StreamingTable,
    Table,
    TemporaryView,
)
from pyspark.pipelines.flow import AutoCdcFlow, Flow
from pyspark.pipelines.graph_element_registry import GraphElementRegistry
from pyspark.pipelines.source_code_location import SourceCodeLocation
from pyspark.sql.types import StructType
from pyspark._pyspark import (
    pipeline_define_output as _define_output,
    pipeline_define_flow as _define_flow,
    pipeline_define_auto_cdc_flow as _define_auto_cdc_flow,
    pipeline_define_sql_graph_elements as _define_sql_graph_elements,
)

# spark.connect.OutputType enum values.
_OUTPUT_TYPE_MATERIALIZED_VIEW = 1
_OUTPUT_TYPE_TABLE = 2
_OUTPUT_TYPE_TEMPORARY_VIEW = 3
_OUTPUT_TYPE_SINK = 4

# spark.connect.PipelineCommand.DefineFlow.SCDType enum values.
_SCD_TYPE_1 = 1


def _scl(loc: SourceCodeLocation):
    return loc.filename, loc.line_number


class SparkConnectGraphElementRegistry(GraphElementRegistry):
    """Registers outputs and flows in a dataflow graph held in a Spark Connect server."""

    def __init__(self, spark: SparkSession, dataflow_graph_id: str) -> None:
        self._spark = spark
        self._dataflow_graph_id = dataflow_graph_id

    def register_output(self, output: Output) -> None:
        scl_file, scl_line = _scl(output.source_code_location)
        if isinstance(output, Table):
            if isinstance(output.schema, str):
                schema_string, schema_data_type = output.schema, None
            elif isinstance(output.schema, StructType):
                schema_string, schema_data_type = None, output.schema
            else:
                schema_string, schema_data_type = None, None

            if isinstance(output, MaterializedView):
                output_type = _OUTPUT_TYPE_MATERIALIZED_VIEW
            elif isinstance(output, StreamingTable):
                output_type = _OUTPUT_TYPE_TABLE
            else:
                raise PySparkTypeError(
                    errorClass="UNSUPPORTED_PIPELINES_DATASET_TYPE",
                    messageParameters={"output_type": type(output).__name__},
                )

            _define_output(
                self._spark,
                self._dataflow_graph_id,
                output.name,
                output_type,
                output.comment,
                dict(output.table_properties),
                list(output.partition_cols) if output.partition_cols else None,
                list(output.cluster_by) if output.cluster_by else None,
                output.format,
                schema_string,
                schema_data_type,
                None,
                None,
                scl_file,
                scl_line,
            )
        elif isinstance(output, TemporaryView):
            _define_output(
                self._spark,
                self._dataflow_graph_id,
                output.name,
                _OUTPUT_TYPE_TEMPORARY_VIEW,
                output.comment,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
                scl_file,
                scl_line,
            )
        elif isinstance(output, Sink):
            _define_output(
                self._spark,
                self._dataflow_graph_id,
                output.name,
                _OUTPUT_TYPE_SINK,
                output.comment,
                None,
                None,
                None,
                None,
                None,
                None,
                dict(output.options),
                output.format,
                scl_file,
                scl_line,
            )
        else:
            raise PySparkTypeError(
                errorClass="UNSUPPORTED_PIPELINES_DATASET_TYPE",
                messageParameters={"output_type": type(output).__name__},
            )

    def register_flow(self, flow: Flow) -> None:
        # flow.func() builds the query DataFrame lazily (no server round-trip), so unlike
        # the reference client there is no client-side analysis to scope with a
        # PipelineAnalysisContext; the DefineFlow proto carries the graph id + flow name,
        # which is what the server uses to analyze the relation.
        df = flow.func()
        scl_file, scl_line = _scl(flow.source_code_location)
        _define_flow(
            self._spark,
            self._dataflow_graph_id,
            flow.name,
            flow.target,
            df,
            dict(flow.spark_conf) if flow.spark_conf else None,
            scl_file,
            scl_line,
        )

    def register_auto_cdc_flow(self, flow: AutoCdcFlow) -> None:
        scl_file, scl_line = _scl(flow.source_code_location)
        stored_as_scd_type = _SCD_TYPE_1 if flow.stored_as_scd_type is not None else 0
        _define_auto_cdc_flow(
            self._spark,
            self._dataflow_graph_id,
            cast(str, flow.name),
            flow.target,
            flow.source,
            list(flow.keys),
            flow.sequence_by,
            flow.apply_as_deletes,
            list(flow.column_list) if flow.column_list is not None else None,
            list(flow.except_column_list) if flow.except_column_list is not None else None,
            stored_as_scd_type,
            scl_file,
            scl_line,
        )

    def register_sql(self, sql_text: str, file_path: Path) -> None:
        _define_sql_graph_elements(
            self._spark,
            self._dataflow_graph_id,
            sql_text,
            str(file_path),
        )
