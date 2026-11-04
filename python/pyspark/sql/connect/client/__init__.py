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
