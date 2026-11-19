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

"""The module of PySpark's ``pyspark.sql.connect.client.core``, which official PySpark code and
tests import, and of which its classes are, as PySpark's: so a pickle names them here. The results
of the analyses, the configurations and the observed metrics of the client are defined here, as
PySpark's are; the classes of ``__all__`` are those of the package, which sets them here once it
has defined them. The gRPC client itself is the Rust transport: the drop-in does not provide the
rest of ``core`` (the gRPC stub of the client)."""

__all__ = ["ChannelBuilder", "DefaultChannelBuilder", "RpcDeadlines", "SparkConnectClient"]

from typing import TYPE_CHECKING, Any, Optional

# As PySpark's core, which imports the messages of Spark Connect, so that protobuf's default pool
# has their files once this module is imported: a pickle of PySpark's PlanObservedMetrics, which
# imports this module before it makes the Expression.Literal messages of its metrics of their full
# names (protobuf's `_InternalConstructMessage`), loads in a new process, as with PySpark.
import pyspark.sql.connect.proto  # noqa: F401
from pyspark.errors.exceptions.connect import SparkConnectException

# As PySpark's core, which has the metrics classes of the execution info of its results.
from pyspark.sql.metrics import (  # noqa: F401
    ExecutionInfo,
    MetricValue,
    ObservedMetrics,
    PlanMetrics,
)
from pyspark.sql.types import DataType

if TYPE_CHECKING:
    import pyspark.sql.connect.proto as pb2

    # The classes of `__all__`, which the package sets here once it has defined them.
    from pyspark.sql.connect.client import (
        ChannelBuilder,
        DefaultChannelBuilder,
        RpcDeadlines,
        SparkConnectClient,
    )
    from pyspark.storagelevel import StorageLevel

# The names of pyspark.sql.connect.client.retries that PySpark's core imports, and so has too.
_RETRIES_NAMES = (
    "RetryPolicy",
    "Retrying",
    "DefaultPolicy",
    "DEFAULT_MAX_RETRY_EXCEPTION_ELAPSED_TIME",
)


def __getattr__(name: str) -> Any:
    # Of the retries module when they are first read: it imports grpc, which this client does
    # not need otherwise.
    if name in _RETRIES_NAMES:
        from pyspark.sql.connect.client import retries

        return getattr(retries, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class PlanObservedMetrics(ObservedMetrics):
    """Observed metrics of one ``observe(...)`` node, as carried by
    ``DataFrame.executionInfo.flows``, as PySpark Connect's: its ``metrics`` are the
    ``Expression.Literal`` messages (of ``pyspark.sql.connect.proto``) that the server sent,
    which ``pairs`` converts with ``LiteralExpression._to_value``, and so raises PySpark's
    ``UNSUPPORTED_LITERAL`` of a literal that it does not convert."""

    def __init__(self, name: str, metrics: list["pb2.Expression.Literal"], keys: list[str]):
        self._name = name
        self._metrics = metrics
        self._keys = keys if keys else [f"observed_metric_{i}" for i in range(len(self.metrics))]

    def __repr__(self) -> str:
        return f"Plan observed({self._name}={self._metrics})"

    @property
    def name(self) -> str:
        return self._name

    @property
    def metrics(self) -> list["pb2.Expression.Literal"]:
        return self._metrics

    @property
    def pairs(self) -> dict[str, Any]:
        from pyspark.sql.connect.expressions import LiteralExpression

        result = {}
        for x in range(len(self._metrics)):
            result[self.keys[x]] = LiteralExpression._to_value(self.metrics[x])
        return result

    @property
    def keys(self) -> list[str]:
        return self._keys

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation of this observed metrics,
        with the keys ``name``, ``keys`` and ``pairs``, as PySpark Connect's."""
        return {
            "name": self._name,
            "keys": self._keys,
            "pairs": self.pairs,
        }


class AnalyzeResult:
    """The result of an analysis of SparkConnectClient's ``_analyze``, as PySpark Connect's: the
    one field of the result of the analysis, and None for the others."""

    def __init__(
        self,
        schema: Optional[DataType],
        explain_string: Optional[str],
        tree_string: Optional[str],
        is_local: Optional[bool],
        is_streaming: Optional[bool],
        input_files: Optional[list[str]],
        spark_version: Optional[str],
        parsed: Optional[DataType],
        is_same_semantics: Optional[bool],
        semantic_hash: Optional[int],
        storage_level: Optional["StorageLevel"],
        ddl_string: Optional[str],
    ):
        self.schema = schema
        self.explain_string = explain_string
        self.tree_string = tree_string
        self.is_local = is_local
        self.is_streaming = is_streaming
        self.input_files = input_files
        self.spark_version = spark_version
        self.parsed = parsed
        self.is_same_semantics = is_same_semantics
        self.semantic_hash = semantic_hash
        self.storage_level = storage_level
        self.ddl_string = ddl_string

    @classmethod
    def fromProto(cls, pb: Any) -> "AnalyzeResult":
        from pyspark.sql.connect import types
        from pyspark.sql.connect.conversion import proto_to_storage_level

        schema: Optional[DataType] = None
        explain_string: Optional[str] = None
        tree_string: Optional[str] = None
        is_local: Optional[bool] = None
        is_streaming: Optional[bool] = None
        input_files: Optional[list[str]] = None
        spark_version: Optional[str] = None
        parsed: Optional[DataType] = None
        is_same_semantics: Optional[bool] = None
        semantic_hash: Optional[int] = None
        storage_level: Optional[StorageLevel] = None
        ddl_string: Optional[str] = None

        if pb.HasField("schema"):
            schema = types.proto_schema_to_pyspark_data_type(pb.schema.schema)
        elif pb.HasField("explain"):
            explain_string = pb.explain.explain_string
        elif pb.HasField("tree_string"):
            tree_string = pb.tree_string.tree_string
        elif pb.HasField("is_local"):
            is_local = pb.is_local.is_local
        elif pb.HasField("is_streaming"):
            is_streaming = pb.is_streaming.is_streaming
        elif pb.HasField("input_files"):
            input_files = pb.input_files.files
        elif pb.HasField("spark_version"):
            spark_version = pb.spark_version.version
        elif pb.HasField("ddl_parse"):
            parsed = types.proto_schema_to_pyspark_data_type(pb.ddl_parse.parsed)
        elif pb.HasField("same_semantics"):
            is_same_semantics = pb.same_semantics.result
        elif pb.HasField("semantic_hash"):
            semantic_hash = pb.semantic_hash.result
        elif pb.HasField("persist"):
            pass
        elif pb.HasField("unpersist"):
            pass
        elif pb.HasField("get_storage_level"):
            storage_level = proto_to_storage_level(pb.get_storage_level.storage_level)
        elif pb.HasField("json_to_ddl"):
            ddl_string = pb.json_to_ddl.ddl_string
        else:
            raise SparkConnectException("No analyze result found!")

        return AnalyzeResult(
            schema,
            explain_string,
            tree_string,
            is_local,
            is_streaming,
            input_files,
            spark_version,
            parsed,
            is_same_semantics,
            semantic_hash,
            storage_level,
            ddl_string,
        )


class ConfigResult:
    """The result of SparkConnectClient's ``config``, as PySpark Connect's: the pairs of a key and
    its value, None for a key without one, and the warnings of the server."""

    def __init__(self, pairs: list[tuple[str, Optional[str]]], warnings: list[str]):
        self.pairs = pairs
        self.warnings = warnings

    @classmethod
    def fromProto(cls, pb: "pb2.ConfigResponse") -> "ConfigResult":
        return ConfigResult(
            pairs=[(pair.key, pair.value if pair.HasField("value") else None) for pair in pb.pairs],
            warnings=list(pb.warnings),
        )
