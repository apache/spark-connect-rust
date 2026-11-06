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

"""Compatibility shim for pyspark.sql.connect.client: re-exports SparkConnectClient and related classes."""

try:
    from pyspark._pyspark import SparkConnectClientStub as SparkConnectClient
    __all__ = ["SparkConnectClient"]
except ImportError:
    # If Rust backend not available, guard the export
    __all__ = []


from typing import Any, Dict, List

from pyspark.sql.metrics import ObservedMetrics


class _UnsupportedLiteral:
    """An observed-metric literal of a kind ``LiteralExpression._to_value`` rejects. The
    Rust transport decodes eagerly, so it leaves this marker and ``pairs`` raises the
    reference's ``UNSUPPORTED_LITERAL`` when read, as the reference does."""

    def __init__(self, literal: str):
        self.literal = literal

    def __repr__(self) -> str:
        return f"_UnsupportedLiteral({self.literal})"


class PlanObservedMetrics(ObservedMetrics):
    """Observed metrics of one ``observe(...)`` node, as carried by
    ``DataFrame.executionInfo.flows``. Mirrors ``pyspark.sql.connect.client.core
    .PlanObservedMetrics``; the Rust transport has already decoded the literal values."""

    def __init__(self, name: str, metrics: List[Any], keys: List[str]):
        self._name = name
        self._metrics = metrics
        self._keys = keys if keys else [f"observed_metric_{i}" for i in range(len(metrics))]

    def __repr__(self) -> str:
        return f"Plan observed({self._name}={self._metrics})"

    @property
    def name(self) -> str:
        return self._name

    @property
    def metrics(self) -> List[Any]:
        return self._metrics

    @property
    def pairs(self) -> Dict[str, Any]:
        for metric in self._metrics:
            if isinstance(metric, _UnsupportedLiteral):
                from pyspark.errors import PySparkTypeError

                raise PySparkTypeError(
                    errorClass="UNSUPPORTED_LITERAL",
                    messageParameters={"literal": metric.literal},
                )
        return {self._keys[i]: self._metrics[i] for i in range(len(self._metrics))}

    @property
    def keys(self) -> List[str]:
        return self._keys

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self._name, "keys": self._keys, "pairs": self.pairs}


def _to_pandas(
    table: Any,
    schema: Any,
    metrics: List[Any],
    observed_metrics: List[Any],
    *,
    timezone: Any,
    struct_handling_mode: Any,
    self_destruct: Any,
) -> Any:
    """The pandas DataFrame of the Arrow ``table`` of a result, of ``schema`` if the server sent
    it, as ``SparkConnectClient.to_pandas`` converts it: each column by its Spark type, in the
    session time zone, with the metrics of the execution in ``attrs``."""
    from pyspark.sql.pandas.conversion import _convert_arrow_table_to_pandas
    from pyspark.sql.pandas.types import from_arrow_schema

    schema = schema or from_arrow_schema(table.schema, prefer_timestamp_ntz=True)
    pdf = _convert_arrow_table_to_pandas(
        arrow_table=table,
        schema=schema,
        timezone=timezone,
        struct_handling_mode=struct_handling_mode,
        date_as_object=False,
        self_destruct=self_destruct == "true",
    )
    if len(metrics) > 0:
        pdf.attrs["metrics"] = metrics
    if len(observed_metrics) > 0:
        pdf.attrs["observed_metrics"] = observed_metrics
    return pdf
