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

"""Compatibility shim for pyspark.sql.connect.client: ``SparkConnectClient``, the class of
``SparkSession.client``, and related classes.

The client runs over the Rust transport of its session, which sends the messages that it is
given, such as the plans of ``to_table`` and ``schema``, as they are: the client builds the
requests of those as PySpark Connect's does, and reads their responses as it does. The retry
policies, the reattachable executions, the result chunking and the RPC deadlines of the client are
those of the transport."""

__all__ = ["ChannelBuilder", "DefaultChannelBuilder", "RpcDeadlines", "SparkConnectClient"]


import copy
import functools
import logging
import math
import os
import platform
import sys
import threading
import urllib.parse
import uuid
import warnings
import weakref
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Union

from pyspark import _pyspark
from pyspark.errors import (
    PySparkNotImplementedError,
    PySparkValueError,
)
from pyspark.errors.exceptions.connect import SparkConnectException
from pyspark.sql.connect.logging import logger
# As PySpark's package, which exports the log level of Spark Connect's logger.
from pyspark.sql.connect.logging import getLogLevel  # noqa: F401
from pyspark.sql.metrics import ExecutionInfo, ObservedMetrics, PlanMetrics
from pyspark.sql.types import DataType, StructType
from pyspark.util import PythonEvalType
from pyspark.version import __version__

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa

    import pyspark.sql.connect.proto as pb2
    from pyspark.sql.connect.client.retries import RetryPolicy
    from pyspark.sql.connect.observation import Observation
    from pyspark.sql.datasource import DataSource
    from pyspark.storagelevel import StorageLevel


@dataclass(frozen=True)
class RpcDeadlines:
    """Per-RPC timeout configuration for :class:`SparkConnectClient`, as PySpark Connect's.

    Each field controls the timeout (in seconds as a float) for one gRPC call type. Set a field to
    ``None`` to disable the per-RPC timeout for that call. Use :meth:`RpcDeadlines.disabled` to
    create an instance with all timeouts disabled.

    Note on ``reattachable_execute_plan`` and ``reattach_execute``: these timeouts apply to each
    individual gRPC stream segment, not to the overall query execution lifetime. When a deadline
    fires, the server-side operation continues running; the client opens a new ReattachExecute
    stream to resume receiving results. Non-reattachable ExecutePlan has no deadline because a
    timeout there would kill the execution with no recovery path.

    The Rust transport applies the deadline of each of its RPCs, ``get_status`` that of the
    GetStatus RPC of ``SparkConnectClient._get_operation_statuses``.
    """

    # As PySpark Connect's, of pyspark.sql.connect.client.core, whose name this class has too.
    __module__ = "pyspark.sql.connect.client.core"

    reattachable_execute_plan: Optional[float] = 10 * 60  # 10 min
    reattach_execute: Optional[float] = 10 * 60  # 10 min
    analyze_plan: Optional[float] = 60 * 60  # 1 hour
    add_artifacts: Optional[float] = 60 * 60  # 1 hour
    config: Optional[float] = 10 * 60  # 10 min
    interrupt: Optional[float] = 10 * 60  # 10 min
    release_session: Optional[float] = 10 * 60  # 10 min
    artifact_status: Optional[float] = 10 * 60  # 10 min
    clone_session: Optional[float] = 10 * 60  # 10 min
    get_status: Optional[float] = 10 * 60  # 10 min
    fetch_error_details: Optional[float] = 10 * 60  # 10 min

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if value is not None and value <= 0:
                raise PySparkValueError(
                    message=(
                        f"RpcDeadlines.{field.name} must be a positive number or None, "
                        f"got {value!r}"
                    ),
                )

    @classmethod
    def disabled(cls) -> "RpcDeadlines":
        """Create an :class:`RpcDeadlines` with all per-RPC timeouts disabled.

        Use this when you want to rely solely on server-side or network-layer timeouts.
        """
        return cls(
            reattachable_execute_plan=None,
            reattach_execute=None,
            analyze_plan=None,
            add_artifacts=None,
            config=None,
            interrupt=None,
            release_session=None,
            artifact_status=None,
            clone_session=None,
            get_status=None,
            fetch_error_details=None,
        )


class ChannelBuilder:
    """The parameters of a Spark Connect connection, as PySpark Connect's ``ChannelBuilder``:
    what ``SparkSession.builder.channelBuilder`` and ``SparkSession(connection=...)`` take. The
    Rust transport opens the channel of the session from the ``host``, the ``port`` and the
    ``_params`` of the builder, as a ``sc://`` connection string, so the endpoint and the
    parameters apply. A custom gRPC channel (``toChannel``), its interceptors and its channel
    options do not: the transport has no gRPC channel to give them to."""

    # As PySpark Connect's, of pyspark.sql.connect.client.core, whose name this class has too.
    __module__ = "pyspark.sql.connect.client.core"

    PARAM_USE_SSL = "use_ssl"
    PARAM_TOKEN = "token"
    PARAM_USER_ID = "user_id"
    PARAM_SESSION_ID = "session_id"
    PARAM_USER_AGENT = "user_agent"
    PARAM_GRPC_KEEPALIVE_ENABLED = "grpc_keepalive_enabled"
    PARAM_GRPC_KEEPALIVE_TIME_MS = "grpc_keepalive_time_ms"
    PARAM_GRPC_KEEPALIVE_TIMEOUT_MS = "grpc_keepalive_timeout_ms"
    PARAM_GRPC_KEEPALIVE_WITHOUT_CALLS = "grpc_keepalive_without_calls"

    # The keepalive PINGs that detect a silently-dead connection, as PySpark's (SPARK-58094).
    GRPC_DEFAULT_KEEPALIVE_ENABLED = True
    GRPC_DEFAULT_KEEPALIVE_TIME_MS = 60 * 1000
    GRPC_DEFAULT_KEEPALIVE_TIMEOUT_MS = 20 * 1000
    GRPC_DEFAULT_KEEPALIVE_WITHOUT_CALLS = True

    GRPC_MAX_MESSAGE_LENGTH_DEFAULT = 128 * 1024 * 1024
    GRPC_DEFAULT_OPTIONS = [
        ("grpc.max_send_message_length", GRPC_MAX_MESSAGE_LENGTH_DEFAULT),
        ("grpc.max_receive_message_length", GRPC_MAX_MESSAGE_LENGTH_DEFAULT),
    ]

    def __init__(
        self,
        channelOptions: Optional[list[tuple[str, Any]]] = None,
        params: Optional[dict[str, str]] = None,
    ):
        if channelOptions:
            raise PySparkNotImplementedError(
                errorClass="NOT_IMPLEMENTED", messageParameters={"feature": "channelOptions"}
            )
        self._params: dict[str, str] = params or dict()

    def get(self, key: str) -> Any:
        """The value of the parameter ``key``; raises ``KeyError`` if it is not set."""
        return self._params[key]

    def getDefault(self, key: str, default: Any) -> Any:
        """The value of the parameter ``key``, or ``default`` if it is not set."""
        return self._params.get(key, default)

    def set(self, key: str, value: Any) -> None:
        """Set the parameter ``key`` to ``value``."""
        self._params[key] = value

    @property
    def host(self) -> str:
        """The hostname where this client intends to connect."""
        raise PySparkNotImplementedError(
            errorClass="NOT_IMPLEMENTED", messageParameters={"feature": "host"}
        )

    @property
    def userId(self) -> Optional[str]:
        """The user_id, extracted from the connection string or configured by other means."""
        return self._params.get(ChannelBuilder.PARAM_USER_ID, None)

    @property
    def token(self) -> Optional[str]:
        """The authentication token: of the parameters, else of the environment variable
        ``SPARK_CONNECT_AUTHENTICATE_TOKEN``."""
        return self._params.get(
            ChannelBuilder.PARAM_TOKEN, os.environ.get("SPARK_CONNECT_AUTHENTICATE_TOKEN")
        )

    @property
    def keepalive_enabled(self) -> bool:
        """Whether the client sends gRPC/HTTP2 keepalive PINGs to detect a silently-dead
        connection. Enabled by default."""
        return (
            self.getDefault(
                ChannelBuilder.PARAM_GRPC_KEEPALIVE_ENABLED,
                str(ChannelBuilder.GRPC_DEFAULT_KEEPALIVE_ENABLED),
            ).lower()
            == "true"
        )

    @property
    def keepalive_time_ms(self) -> int:
        """Idle time (in milliseconds) before sending a gRPC/HTTP2 keepalive PING."""
        return int(
            self.getDefault(
                ChannelBuilder.PARAM_GRPC_KEEPALIVE_TIME_MS,
                ChannelBuilder.GRPC_DEFAULT_KEEPALIVE_TIME_MS,
            )
        )

    @property
    def keepalive_timeout_ms(self) -> int:
        """Time (in milliseconds) to wait for a keepalive PING ack before failing."""
        return int(
            self.getDefault(
                ChannelBuilder.PARAM_GRPC_KEEPALIVE_TIMEOUT_MS,
                ChannelBuilder.GRPC_DEFAULT_KEEPALIVE_TIMEOUT_MS,
            )
        )

    @property
    def keepalive_without_calls(self) -> bool:
        """Whether to keep sending keepalive PINGs when there are no in-flight RPCs."""
        return (
            self.getDefault(
                ChannelBuilder.PARAM_GRPC_KEEPALIVE_WITHOUT_CALLS,
                str(ChannelBuilder.GRPC_DEFAULT_KEEPALIVE_WITHOUT_CALLS),
            ).lower()
            == "true"
        )

    def _effective_channel_options(self) -> list[tuple[str, Any]]:
        """The gRPC options of the channel: the defaults, and the keepalive options unless the
        keepalive is disabled. The Rust channel applies the same keepalive, from the same
        parameters, and the same message lengths."""
        options = list(ChannelBuilder.GRPC_DEFAULT_OPTIONS)
        if not self.keepalive_enabled:
            return options
        options += [
            ("grpc.keepalive_time_ms", self.keepalive_time_ms),
            ("grpc.keepalive_timeout_ms", self.keepalive_timeout_ms),
            ("grpc.keepalive_permit_without_calls", 1 if self.keepalive_without_calls else 0),
        ]
        return options

    @property
    def session_id(self) -> Optional[str]:
        """The session_id of the parameters, or None if it is not set."""
        session_id = self._params.get(ChannelBuilder.PARAM_SESSION_ID, None)
        if session_id is not None:
            try:
                uuid.UUID(session_id, version=4)
            except ValueError as ve:
                # As PySpark's: no `from`, so the __cause__ of the error stays None.
                raise PySparkValueError(  # noqa: B904
                    errorClass="INVALID_SESSION_UUID_ID",
                    messageParameters={"arg_name": "session_id", "origin": str(ve)},
                )
        return session_id

    @property
    def userAgent(self) -> str:
        """The user agent: the ``user_agent`` parameter, else the environment variable
        ``SPARK_CONNECT_USER_AGENT``, else ``_SPARK_CONNECT_PYTHON``, with the versions of
        Spark, the operating system and Python. Its URL-escaped form is at most 2048 characters."""
        user_agent = self._params.get(
            ChannelBuilder.PARAM_USER_AGENT,
            os.getenv("SPARK_CONNECT_USER_AGENT", "_SPARK_CONNECT_PYTHON"),
        )
        ua_len = len(urllib.parse.quote(user_agent))
        if ua_len > 2048:
            raise SparkConnectException(
                f"'user_agent' parameter should not exceed 2048 characters after URL "
                f"escaping, found {ua_len} characters."
            )
        return " ".join(
            [
                user_agent,
                f"spark/{__version__}",
                f"os/{platform.uname().system.lower()}",
                f"python/{platform.python_version()}",
            ]
        )

    def metadata(self) -> Iterable[tuple[str, str]]:
        """The parameters that are not used by the channel, as gRPC metadata of each request."""
        reserved = [
            ChannelBuilder.PARAM_TOKEN,
            ChannelBuilder.PARAM_USE_SSL,
            ChannelBuilder.PARAM_USER_ID,
            ChannelBuilder.PARAM_USER_AGENT,
            ChannelBuilder.PARAM_SESSION_ID,
            ChannelBuilder.PARAM_GRPC_KEEPALIVE_ENABLED,
            ChannelBuilder.PARAM_GRPC_KEEPALIVE_TIME_MS,
            ChannelBuilder.PARAM_GRPC_KEEPALIVE_TIMEOUT_MS,
            ChannelBuilder.PARAM_GRPC_KEEPALIVE_WITHOUT_CALLS,
        ]
        return [(k, self._params[k]) for k in self._params if k not in reserved]


class DefaultChannelBuilder(ChannelBuilder):
    """The channel builder of a ``sc://`` connection string, as PySpark Connect's
    ``DefaultChannelBuilder``.

    Examples
    --------
    >>> cb = DefaultChannelBuilder("sc://localhost")
    >>> cb.endpoint
    'localhost:15002'
    >>> cb = DefaultChannelBuilder("sc://localhost/;use_ssl=true;token=aaa")
    >>> cb.secure
    True
    """

    # As PySpark Connect's, of pyspark.sql.connect.client.core, whose name this class has too.
    __module__ = "pyspark.sql.connect.client.core"

    @staticmethod
    def default_port() -> int:
        # The port of a Spark Connect server, which the drop-in does not discover from a local
        # JVM as PySpark does in the testing mode.
        return 15002

    def __init__(self, url: str, channelOptions: Optional[list[tuple[str, Any]]] = None) -> None:
        """Construct a channel builder of the connection string ``url``, such as
        ``sc://hostname:port/;key=value``."""
        super().__init__(channelOptions=channelOptions)

        # Explicitly check the scheme of the URL.
        if url[:5] != "sc://":
            raise PySparkValueError(
                errorClass="INVALID_CONNECT_URL",
                messageParameters={
                    "detail": "The URL must start with 'sc://'. Please update the URL to "
                    "follow the correct format, e.g., 'sc://hostname:port'.",
                },
            )
        # Rewrite the URL to use http as the scheme so that we can leverage Python's parser.
        tmp_url = "http" + url[2:]
        self.url = urllib.parse.urlparse(tmp_url)
        if len(self.url.path) > 0 and self.url.path != "/":
            raise PySparkValueError(
                errorClass="INVALID_CONNECT_URL",
                messageParameters={
                    "detail": f"The path component '{self.url.path}' must be empty. Please update "
                    f"the URL to follow the correct format, e.g., 'sc://hostname:port'.",
                },
            )
        self._extract_attributes()

    def _extract_attributes(self) -> None:
        if len(self.url.params) > 0:
            parts = self.url.params.split(";")
            for p in parts:
                kv = p.split("=")
                if len(kv) != 2:
                    raise PySparkValueError(
                        errorClass="INVALID_CONNECT_URL",
                        messageParameters={
                            "detail": f"Parameter '{p}' should be provided as a "
                            f"key-value pair separated by an equal sign (=). Please update "
                            f"the parameter to follow the correct format, e.g., 'key=value'.",
                        },
                    )
                self.set(kv[0], urllib.parse.unquote(kv[1]))

        if not self.url.hostname:
            raise PySparkValueError(
                errorClass="INVALID_CONNECT_URL",
                messageParameters={
                    "detail": f"Hostname is missing in the URL: '{self.url.geturl()}'. "
                    "Please update the URL to follow the correct format, "
                    "e.g., 'sc://hostname:port'.",
                },
            )
        self._host = f"[{self.url.hostname}]" if ":" in self.url.hostname else self.url.hostname
        self._port = self.url.port if self.url.port is not None else self.default_port()

    @property
    def secure(self) -> bool:
        """Whether the connection is secured: by TLS, or by a token."""
        return self.use_ssl or self.token is not None

    @property
    def use_ssl(self) -> bool:
        return self.getDefault(ChannelBuilder.PARAM_USE_SSL, "").lower() == "true"

    @property
    def host(self) -> str:
        """The hostname where this client intends to connect."""
        return self._host

    @property
    def port(self) -> int:
        """The port where this client intends to connect."""
        return self._port

    @property
    def endpoint(self) -> str:
        """The ``host:port`` of the connection."""
        return f"{self._host}:{self._port}"


class _UnsupportedLiteral:
    """The value that the native conversion of a literal (``literal_to_py``) gives for a literal of
    a kind that ``LiteralExpression._to_value`` rejects, of its text. The metrics of an
    Observation are checked before they are converted, so that the action raises PySpark's
    ``UNSUPPORTED_LITERAL`` instead; those of ``PlanObservedMetrics`` are the literals themselves,
    which its ``pairs`` converts as PySpark's."""

    def __init__(self, literal: str):
        self.literal = literal

    def __repr__(self) -> str:
        return f"_UnsupportedLiteral({self.literal})"


class PlanObservedMetrics(ObservedMetrics):
    """Observed metrics of one ``observe(...)`` node, as carried by
    ``DataFrame.executionInfo.flows``, as PySpark Connect's: its ``metrics`` are the
    ``Expression.Literal`` messages (of ``pyspark.sql.connect.proto``) that the server sent,
    which ``pairs`` converts with ``LiteralExpression._to_value``, and so raises PySpark's
    ``UNSUPPORTED_LITERAL`` of a literal that it does not convert."""

    # As PySpark Connect's, of pyspark.sql.connect.client.core, whose name this class has too.
    __module__ = "pyspark.sql.connect.client.core"

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

    # As PySpark Connect's, of pyspark.sql.connect.client.core, whose name this class has too.
    __module__ = "pyspark.sql.connect.client.core"

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

    # As PySpark Connect's, of pyspark.sql.connect.client.core, whose name this class has too.
    __module__ = "pyspark.sql.connect.client.core"

    def __init__(self, pairs: list[tuple[str, Optional[str]]], warnings: list[str]):
        self.pairs = pairs
        self.warnings = warnings

    @classmethod
    def fromProto(cls, pb: "pb2.ConfigResponse") -> "ConfigResult":
        return ConfigResult(
            pairs=[(pair.key, pair.value if pair.HasField("value") else None) for pair in pb.pairs],
            warnings=list(pb.warnings),
        )


def _self_destruct_table(table: Any) -> Any:
    """``table``, of the record batches of a result, each a chunk of its columns, with each column
    of each batch copied into its own allocation, as ``SparkConnectClient._execute_and_fetch``
    copies the batches of ``to_pandas`` with ``self_destruct``, which frees memory column-wise
    where Arrow record batches are oriented row-wise. So it raises what PyArrow raises of a column
    that it cannot copy, as the ``KeyError`` of a year-month interval column, and a result without
    columns has no rows, as a copied batch without columns has none."""
    import pyarrow as pa

    columns = table.columns
    results = []
    for chunk in range(columns[0].num_chunks if columns else 0):
        results.append(
            pa.RecordBatch.from_arrays(
                # This call actually reallocates the array
                [pa.concat_arrays([column.chunk(chunk)]) for column in columns],
                schema=table.schema,
            )
        )
    return pa.Table.from_batches(results, schema=table.schema)


def _table_as_received(table: Any) -> Any:
    """``table``, of the record batches of a result that the Rust client decodes, with the metadata
    that PySpark's client reads of the Arrow IPC stream of a Spark server, as its ``_to_table``
    gives it. The server writes the metadata of the schema and of each field, nested ones too, an
    empty one where there is none, which PyArrow reads as ``{}``, and the keys of the metadata of
    a field in byte order (``geometry`` or ``geography`` before ``srid``,
    ``SPARK::metadata::json`` before a precision); the decoded batches keep neither, as no
    metadata is None there and its keys are in no order. The cast shares the buffers of the
    columns, whose chunks it keeps."""
    import pyarrow as pa

    def metadata_of(metadata: Any) -> dict:
        return dict(sorted((metadata or {}).items()))

    def field_of(field: Any) -> Any:
        return pa.field(
            field.name, type_of(field.type), field.nullable, metadata=metadata_of(field.metadata)
        )

    def type_of(data_type: Any) -> Any:
        if pa.types.is_struct(data_type):
            return pa.struct([field_of(data_type.field(i)) for i in range(data_type.num_fields)])
        if pa.types.is_map(data_type):
            return pa.map_(
                field_of(data_type.key_field),
                field_of(data_type.item_field),
                data_type.keys_sorted,
            )
        if pa.types.is_large_list(data_type):
            return pa.large_list(field_of(data_type.value_field))
        if pa.types.is_list(data_type):
            return pa.list_(field_of(data_type.value_field))
        return data_type

    metadata = metadata_of(table.schema.metadata)
    if table.num_columns == 0:
        # A cast, or a replacement of the schema's metadata, would lose the rows of a table
        # without columns, which its batches keep.
        schema = table.schema.with_metadata(metadata)
        return pa.Table.from_batches(table.to_batches(), schema=schema)
    return table.cast(pa.schema([field_of(field) for field in table.schema], metadata=metadata))


def _to_pandas(
    table: Any,
    schema: Any,
    metrics: list[Any],
    observed_metrics: list[Any],
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


class SparkConnectClient:
    """The client of a Spark Connect session, as PySpark Connect's ``SparkConnectClient``: the
    client of ``SparkSession.client``, and one that ``SparkConnectClient(connection, ...)`` makes
    of a session of its own. It sends the messages that it is given, of
    ``pyspark.sql.connect.proto``, over the Rust transport of its session, as the modules of
    PySpark that build their own messages, such as ``pyspark.ml``, send them: commands, plans and
    their analyses, configurations, and the artifacts, the tags, the interrupts and the progress
    handlers of the session."""

    # As PySpark Connect's, of pyspark.sql.connect.client.core, whose name this class has too.
    __module__ = "pyspark.sql.connect.client.core"

    # The session of the client is kept apart from its attributes, which are those of its session:
    # the clients of a session share them (see `_client_state`), as PySpark's session has one
    # client, of one `thread_local`, one list of global user context extensions, its lock, and the
    # attributes that a caller sets.
    __slots__ = ("_session", "__dict__", "__weakref__")

    # The thread of an ML cache RPC in flight, as SparkConnectClient's re-entrancy guard of a
    # RemoteModelRef finalizer that deletes a model.
    _ml_cache_rpc_thread: Optional[int] = None

    def __init__(
        self,
        connection: Union[str, ChannelBuilder],
        user_id: Optional[str] = None,
        channel_options: Optional[list[tuple[str, Any]]] = None,
        retry_policy: Optional[dict[str, Any]] = None,
        use_reattachable_execute: bool = True,
        session_hooks: Optional[list[Any]] = None,
        allow_arrow_batch_chunking: bool = True,
        preferred_arrow_chunk_size: Optional[int] = None,
        rpc_deadlines: Optional[RpcDeadlines] = None,
        max_retry_exception_elapsed_time: Optional[float] = None,
    ):
        """A client of a new session of the connection string or the channel builder
        ``connection``, as PySpark Connect's ``SparkConnectClient``: of the user ``user_id``
        unless the connection names one, else of ``SPARK_USER`` or ``USER``; of the retries of a
        ``DefaultPolicy`` of the keyword arguments ``retry_policy``, and of a
        ``RetryException`` for up to ``max_retry_exception_elapsed_time`` seconds; of reattachable
        executions unless ``use_reattachable_execute`` is False; whose executions call the
        ``on_execute_plan`` of ``session_hooks``; of the result chunking of
        ``allow_arrow_batch_chunking`` and ``preferred_arrow_chunk_size``; and of the deadlines
        of ``rpc_deadlines``. The session is neither the default session nor the active one.

        The Rust transport opens the channel of the connection, so a ``channel_options`` of a
        connection string raises ``NOT_IMPLEMENTED``, as its ``DefaultChannelBuilder`` does; as
        PySpark's, the client of a channel builder does not read it."""
        # As PySpark's, the channel builder of a connection string, which checks it, and then the
        # retry policy and the session ID that the client reads of them.
        builder = (
            connection
            if isinstance(connection, ChannelBuilder)
            else DefaultChannelBuilder(connection, channel_options)
        )
        policies = None
        if retry_policy is not None:
            from pyspark.sql.connect.client.retries import DefaultPolicy

            policies = [DefaultPolicy(**retry_policy)]
        _ = builder.session_id
        hook_factories = [_hook_factory(hook) for hook in session_hooks or []]
        deadlines = (
            None
            if rpc_deadlines is None
            else {field.name: getattr(rpc_deadlines, field.name) for field in fields(RpcDeadlines)}
        )
        session = _pyspark.SparkSession._of_client_options(
            connection,
            user_id,
            hook_factories,
            allow_arrow_batch_chunking,
            preferred_arrow_chunk_size,
            deadlines,
        )
        self._init_of_session(session, builder)
        if not use_reattachable_execute:
            self.disable_reattachable_execute()
        if policies is not None or max_retry_exception_elapsed_time is not None:
            self._set_retry_policies(policies, max_retry_exception_elapsed_time)
        # As PySpark's client, which registers its exit hook once it is made: the registration
        # keeps it until the exit, and no session object owns it, whose __del__ would close it.
        session._keep_until_exit()

    @classmethod
    def _of_session(
        cls, session: Any, builder: Optional[ChannelBuilder] = None
    ) -> "SparkConnectClient":
        """The client of ``session``, a session of the drop-in, as ``SparkSession.client`` and
        the clients of ``newSession`` and ``clone`` are, of the channel builder ``builder`` of the
        connection, if there is one."""
        client = cls.__new__(cls)
        client._init_of_session(session, builder)
        return client

    def _init_of_session(self, session: Any, builder: Optional[ChannelBuilder]) -> None:
        self._session = session
        # The attributes of the client are those of its session, which its other clients have
        # too: its channel builder, its thread_local, its global user context extensions and their
        # lock, and the attributes that a caller set.
        object.__setattr__(self, "__dict__", _client_state(session, builder))

    def __setattr__(self, name: str, value: Any) -> None:
        # As PySpark's client, whose requests read its `global_user_context_extensions`, a list
        # that its methods and a caller may change or rebind: the session takes the extensions of
        # the list that the attribute is set to.
        if name == "global_user_context_extensions" and hasattr(self, "_session"):
            value = _GlobalUserContextExtensions(self._session, value)
            object.__setattr__(self, name, value)
            value._sync()
            return
        object.__setattr__(self, name, value)

    def __copy__(self) -> "SparkConnectClient":
        # As copy.copy of PySpark's client, a plain object: a client of the same session and of a
        # copy of its attributes, whose rebinding of a list is its own. The requests of the
        # session carry the extensions of the list of the session's own client.
        copied = object.__new__(type(self))
        object.__setattr__(copied, "_session", self._session)
        copied.__dict__.update(self.__dict__)
        return copied

    def __reduce__(self) -> Any:
        # PySpark's client keeps its thread-local state in a threading.local, which pickle,
        # cloudpickle and copy.deepcopy refuse: so they fail at one, with the error of each
        # protocol ("cannot pickle '_thread._local' object", or '_local' at 0 and 1).
        import threading

        return (type(self), (threading.local(),))

    @property
    def host(self) -> str:
        """The hostname where this client intends to connect, as SparkConnectClient's ``host``:
        of its channel builder."""
        if self._builder is not None:
            return self._builder.host
        return self._session._endpoint_host()

    @property
    def token(self) -> Optional[str]:
        """The authentication bearer token during connection, as SparkConnectClient's ``token``:
        that of its channel builder, the ``token`` parameter of the connection string, else the
        environment variable ``SPARK_CONNECT_AUTHENTICATE_TOKEN``, or None."""
        if self._builder is not None:
            return self._builder.token
        token = self._session._token_parameter()
        if token is not None:
            return token
        return os.environ.get("SPARK_CONNECT_AUTHENTICATE_TOKEN")

    @property
    def _server_session_id(self) -> Optional[str]:
        """The server-side session ID that the responses carried, or None before the first
        response, as SparkConnectClient's. Set it to simulate a server-side session change: the
        next response then closes the client, as a changed ID does."""
        return self._session._observed_server_session_id()

    @_server_session_id.setter
    def _server_session_id(self, value: Optional[str]) -> None:
        self._session._set_observed_server_session_id(value)

    @property
    def _closed(self) -> bool:
        """Whether the client is closed, as SparkConnectClient's ``_closed``. Setting it to True
        closes the session, which the session registry then no longer returns, and setting it
        to False opens it again, as PySpark's requests are sent again then."""
        return self._session.is_stopped

    @_closed.setter
    def _closed(self, value: bool) -> None:
        if value:
            self._session._close()
        else:
            self._session._reopen()

    @property
    def _session_id(self) -> str:
        """The session ID that the requests of the client carry, as SparkConnectClient's
        ``_session_id``: the session's, until it is set to another, which the requests carry from
        then on, as PySpark's do; the server answers them of the session of that ID, and a
        response of another server-side session than the one that the client observed closes
        the client. The session's ``session_id`` stays, as PySpark's does, and so does that of
        the requests of the artifacts, as PySpark's ArtifactManager keeps the ID that the client
        was built with."""
        return self._session._client_session_id()

    @_session_id.setter
    def _session_id(self, value: str) -> None:
        self._session._set_client_session_id(value)

    @property
    def _user_id(self) -> Optional[str]:
        """The user of the requests of the client, as SparkConnectClient's ``_user_id``: of the
        connection string, else of ``user_id``, else of ``SPARK_USER`` or ``USER``."""
        return self._session._user_id()

    @property
    def _use_reattachable_execute(self) -> bool:
        """Whether the executions of the client are reattachable, as SparkConnectClient's
        ``_use_reattachable_execute``, which ``enable_reattachable_execute`` and
        ``disable_reattachable_execute`` set, and which a caller may set."""
        return self._session._reattachable_execute()

    @_use_reattachable_execute.setter
    def _use_reattachable_execute(self, value: bool) -> None:
        self._session._set_reattachable_execute(bool(value))

    def _get_operation_statuses(
        self,
        operation_ids: Optional[list[str]] = None,
        operation_extensions: Optional[list[Any]] = None,
        request_extensions: Optional[list[Any]] = None,
    ) -> "pb2.GetStatusResponse":
        """The statuses of the operations of the session, as SparkConnectClient's: of those of
        ``operation_ids``, or of all of them if None or empty, of the per-operation
        ``operation_extensions`` and the request-level ``request_extensions`` (each a
        ``google.protobuf.Any``) that the request carries, as the full ``GetStatusResponse``. The
        Rust transport sets the IDs of the session, the client type and the user of the request,
        as PySpark's does."""
        import pyspark.sql.connect.proto as pb2

        req = pb2.GetStatusRequest()
        req.operation_status.SetInParent()
        if operation_ids:
            req.operation_status.operation_ids.extend(operation_ids)
        if operation_extensions:
            req.operation_status.extensions.extend(operation_extensions)
        if request_extensions:
            req.extensions.extend(request_extensions)
        return pb2.GetStatusResponse.FromString(self._session._get_status(req.SerializeToString()))

    @property
    def is_closed(self) -> bool:
        """Whether the session of the client is stopped, as SparkConnectClient's ``is_closed``,
        whether its channel is closed."""
        return self._session.is_stopped

    def _proto_to_string(self, p: Any, truncate: bool = False) -> str:
        """A one-line string of the protobuf message ``p``, truncated if ``truncate``, as
        SparkConnectClient's ``_proto_to_string``, which plan logging uses. A message of the
        Rust protobuf runtime (see ``pyspark.sql.connect.proto``) writes its text format itself,
        as protobuf's ``text_format`` writes one of protobuf's."""
        try:
            max_level = 8 if truncate else sys.maxsize
            p2 = self._truncate(p, max_level) if truncate else p
            if isinstance(p2, _pyspark.ProtoMessage):
                return _pyspark._proto_message_to_string(p2, as_one_line=True)
            from google.protobuf import text_format

            return text_format.MessageToString(p2, as_one_line=True)
        except RecursionError:
            return "<Truncated message due to recursion error>"
        except Exception:
            return "<Truncated message due to truncation error>"

    def _truncate(self, p: Any, allowed_recursion_depth: int) -> Any:
        """A copy of ``p`` with its nested messages cut at ``allowed_recursion_depth``, and its
        long strings and bytes cut, as SparkConnectClient's ``_truncate``."""

        def truncate_str(s: str) -> str:
            if len(s) > 1024:
                return s[:1024] + "[truncated]"
            return s

        def truncate_bytes(b: bytes) -> bytes:
            if len(b) > 8:
                return b[:8] + b"[truncated]"
            return b

        p2 = copy.deepcopy(p)

        for descriptor, value in p.ListFields():
            if value is not None:
                field_name = descriptor.name

                if descriptor.type == descriptor.TYPE_MESSAGE:
                    if allowed_recursion_depth == 0:
                        p2.ClearField(field_name)
                    elif descriptor.label == descriptor.LABEL_REPEATED:
                        p2.ClearField(field_name)
                        getattr(p2, field_name).extend(
                            [self._truncate(v, allowed_recursion_depth - 1) for v in value]
                        )
                    else:
                        getattr(p2, field_name).CopyFrom(
                            self._truncate(value, allowed_recursion_depth - 1)
                        )

                elif descriptor.type == descriptor.TYPE_STRING:
                    if descriptor.label == descriptor.LABEL_REPEATED:
                        p2.ClearField(field_name)
                        getattr(p2, field_name).extend([truncate_str(v) for v in value])
                    else:
                        setattr(p2, field_name, truncate_str(value))

                elif descriptor.type == descriptor.TYPE_BYTES:
                    if descriptor.label == descriptor.LABEL_REPEATED:
                        p2.ClearField(field_name)
                        getattr(p2, field_name).extend([truncate_bytes(v) for v in value])
                    else:
                        setattr(p2, field_name, truncate_bytes(value))

        return p2

    def get_configs(self, *keys: str) -> tuple[Optional[str], ...]:
        """The values of the configurations ``keys``, in one request, as SparkConnectClient's
        ``get_configs``."""
        return tuple(self._session.conf._get_many(list(keys)))

    def get_config_dict(self, *keys: str) -> Mapping[str, Optional[str]]:
        """The values of the configurations ``keys`` by their keys, in one request, as
        SparkConnectClient's ``get_config_dict``."""
        return dict(zip(keys, self.get_configs(*keys)))

    def get_config_with_defaults(
        self, *pairs: tuple[str, Optional[str]]
    ) -> tuple[Optional[str], ...]:
        """The values of the configurations of ``pairs`` of a key and its default, the default of
        a key that is not set, in one request, as SparkConnectClient's
        ``get_config_with_defaults``, of which pandas-on-Spark reads its ANSI mode."""
        return tuple(self._session.conf._get_with_defaults(list(pairs)))

    def config(self, operation: "pb2.ConfigRequest.Operation") -> ConfigResult:
        """Send ``operation``, a ``ConfigRequest.Operation``, as SparkConnectClient's ``config``:
        the result of the configuration request of the session."""
        import pyspark.sql.connect.proto as pb2

        req = pb2.ConfigRequest()
        req.operation.CopyFrom(operation)
        response = pb2.ConfigResponse.FromString(self._session._config(req.SerializeToString()))
        return ConfigResult.fromProto(response)

    def _analyze(self, method: str, **kwargs: Any) -> AnalyzeResult:
        """The result of the analysis ``method`` of the arguments ``kwargs``, as
        SparkConnectClient's ``_analyze``: an AnalyzePlanRequest of the plans or the relations it
        is given, as they are."""
        import pyspark.sql.connect.proto as pb2
        from pyspark.sql.connect.conversion import storage_level_to_proto

        req = pb2.AnalyzePlanRequest()
        if method == "schema":
            req.schema.plan.CopyFrom(kwargs.get("plan"))
        elif method == "explain":
            req.explain.plan.CopyFrom(kwargs.get("plan"))
            explain_mode = kwargs.get("explain_mode")
            allowed_explain_modes = ["simple", "extended", "codegen", "cost", "formatted"]
            if explain_mode not in allowed_explain_modes:
                raise PySparkValueError(
                    errorClass="VALUE_NOT_ALLOWED",
                    messageParameters={
                        "arg_name": "explain_mode",
                        "allowed_values": str(allowed_explain_modes),
                    },
                )
            modes = pb2.AnalyzePlanRequest.Explain.ExplainMode
            req.explain.explain_mode = {
                "simple": modes.EXPLAIN_MODE_SIMPLE,
                "extended": modes.EXPLAIN_MODE_EXTENDED,
                "cost": modes.EXPLAIN_MODE_COST,
                "codegen": modes.EXPLAIN_MODE_CODEGEN,
                "formatted": modes.EXPLAIN_MODE_FORMATTED,
            }[explain_mode]
        elif method == "tree_string":
            req.tree_string.plan.CopyFrom(kwargs.get("plan"))
            level = kwargs.get("level")
            if level and isinstance(level, int):
                req.tree_string.level = level
        elif method == "is_local":
            req.is_local.plan.CopyFrom(kwargs.get("plan"))
        elif method == "is_streaming":
            req.is_streaming.plan.CopyFrom(kwargs.get("plan"))
        elif method == "input_files":
            req.input_files.plan.CopyFrom(kwargs.get("plan"))
        elif method == "spark_version":
            req.spark_version.SetInParent()
        elif method == "ddl_parse":
            req.ddl_parse.ddl_string = kwargs.get("ddl_string")
        elif method == "same_semantics":
            req.same_semantics.target_plan.CopyFrom(kwargs.get("plan"))
            req.same_semantics.other_plan.CopyFrom(kwargs.get("other"))
        elif method == "semantic_hash":
            req.semantic_hash.plan.CopyFrom(kwargs.get("plan"))
        elif method == "persist":
            req.persist.relation.CopyFrom(kwargs.get("relation"))
            if kwargs.get("storage_level", None) is not None:
                storage_level = kwargs.get("storage_level")
                req.persist.storage_level.CopyFrom(storage_level_to_proto(storage_level))
        elif method == "unpersist":
            req.unpersist.relation.CopyFrom(kwargs.get("relation"))
            if kwargs.get("blocking", None) is not None:
                req.unpersist.blocking = kwargs.get("blocking")
        elif method == "get_storage_level":
            req.get_storage_level.relation.CopyFrom(kwargs.get("relation"))
        elif method == "json_to_ddl":
            req.json_to_ddl.json_string = kwargs.get("json_string")
        else:
            raise PySparkValueError(
                errorClass="UNSUPPORTED_OPERATION",
                messageParameters={
                    "operation": method,
                },
            )
        data = self._session._analyze_plan(req.SerializeToString())
        return AnalyzeResult.fromProto(pb2.AnalyzePlanResponse.FromString(data))

    def schema(self, plan: "pb2.Plan") -> StructType:
        """The schema of ``plan``, a ``spark.connect.Plan``, as SparkConnectClient's ``schema``:
        of its analysis by the server."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Schema for plan: {self._proto_to_string(plan, True)}")
        schema = self._analyze(method="schema", plan=plan).schema
        assert schema is not None
        # Server side should populate the struct field which is the schema.
        assert isinstance(schema, StructType)
        return schema

    def explain_string(self, plan: "pb2.Plan", explain_mode: str = "extended") -> str:
        """The explain string of ``plan`` in ``explain_mode``, as SparkConnectClient's
        ``explain_string``: ``VALUE_NOT_ALLOWED`` for a mode other than ``simple``,
        ``extended``, ``codegen``, ``cost`` and ``formatted``."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Explain (mode={explain_mode}) for plan {self._proto_to_string(plan, True)}"
            )
        result = self._analyze(
            method="explain", plan=plan, explain_mode=explain_mode
        ).explain_string
        assert result is not None
        return result

    def same_semantics(self, plan: "pb2.Plan", other: "pb2.Plan") -> bool:
        """Whether ``plan`` and ``other`` have the same semantics, as SparkConnectClient's
        ``same_semantics``."""
        result = self._analyze(method="same_semantics", plan=plan, other=other).is_same_semantics
        assert result is not None
        return result

    def semantic_hash(self, plan: "pb2.Plan") -> int:
        """The ``hashCode`` of the logical plan of ``plan``, as SparkConnectClient's
        ``semantic_hash``."""
        result = self._analyze(method="semantic_hash", plan=plan).semantic_hash
        assert result is not None
        return result

    def _execute_plan(self, plan: "pb2.Plan", observations: Any, reported: bool) -> Any:
        """Start the execution of ``plan``, a ``spark.connect.Plan`` that the client sends as it
        is, as SparkConnectClient's request of a copy of it: its responses, whose observed metrics
        set the Observations of ``observations``, a dict, by their names, and whose progress
        reaches the progress handlers when ``reported``. Of an ``observations`` that is not a
        dict, such as None or an empty list, none is set: PySpark's client reads it only when
        observed metrics arrive (see ``_results``)."""
        items = list(observations.items()) if isinstance(observations, Mapping) else []
        return self._session._execute_plan(plan.SerializeToString(), items, reported)

    def _results(self, responses: Any, observations: Any) -> Iterator[Any]:
        """The results of ``responses``, as SparkConnectClient's
        ``_execute_and_fetch_as_iterator`` yields them of each response: its plan metrics, its
        observed metrics, the schema of its result, the properties of the result of a command by
        their field names, its extension, and the record batches of its Arrow batch. The Rust
        transport has joined the chunks of a batch, checked their offsets and their rows, updated
        the progress of the execution and set the Observations. The execution ends with them."""
        import pyspark.sql.connect.proto as pb2
        from pyspark.sql.connect import types

        try:
            for data, metrics, observed_metrics in responses:
                yield from metrics
                for observed in observed_metrics:
                    # As PySpark's client, which looks the name of the observed metrics up in
                    # `observations` when they arrive: the TypeError of `in` of an object that is
                    # not a container, such as None.
                    if observed.name != "__python_accumulator__":
                        _ = observed.name in observations
                    yield observed
                b = pb2.ExecutePlanResponse.FromString(data)
                if b.HasField("schema"):
                    dt = types.proto_schema_to_pyspark_data_type(b.schema)
                    assert isinstance(dt, StructType)
                    yield dt
                if b.HasField("sql_command_result"):
                    yield {"sql_command_result": b.sql_command_result.relation}
                if b.HasField("write_stream_operation_start_result"):
                    field = "write_stream_operation_start_result"
                    yield {field: b.write_stream_operation_start_result}
                if b.HasField("streaming_query_command_result"):
                    yield {"streaming_query_command_result": b.streaming_query_command_result}
                if b.HasField("streaming_query_manager_command_result"):
                    cmd_result = b.streaming_query_manager_command_result
                    yield {"streaming_query_manager_command_result": cmd_result}
                if b.HasField("streaming_query_listener_events_result"):
                    event_result = b.streaming_query_listener_events_result
                    yield {"streaming_query_listener_events_result": event_result}
                if b.HasField("pipeline_command_result"):
                    yield {"pipeline_command_result": b.pipeline_command_result}
                if b.HasField("pipeline_event_result"):
                    yield {"pipeline_event_result": b.pipeline_event_result}
                if b.HasField("get_resources_command_result"):
                    from pyspark.resource.information import ResourceInformation

                    resources = {}
                    for key, resource in b.get_resources_command_result.resources.items():
                        name = resource.name
                        addresses = [address for address in resource.addresses]
                        resources[key] = ResourceInformation(name, addresses)
                    yield {"get_resources_command_result": resources}
                if b.HasField("extension"):
                    yield b.extension
                if b.HasField("arrow_batch"):
                    import pyarrow as pa

                    with pa.ipc.open_stream(b.arrow_batch.data) as reader:
                        yield from reader
                if b.HasField("create_resource_profile_command_result"):
                    profile_id = b.create_resource_profile_command_result.profile_id
                    yield {"create_resource_profile_command_result": profile_id}
                if b.HasField("checkpoint_command_result"):
                    from pyspark.sql.connect.conversion import proto_to_remote_cached_dataframe

                    yield {
                        "checkpoint_command_result": proto_to_remote_cached_dataframe(
                            b.checkpoint_command_result.relation
                        )
                    }
                if b.HasField("ml_command_result"):
                    yield {"ml_command_result": b.ml_command_result}
        finally:
            responses.close()

    def _execute_and_fetch(
        self, plan: "pb2.Plan", observations: Any, self_destruct: bool = False
    ) -> tuple[Any, Optional[StructType], list[Any], list[Any], dict[str, Any], str]:
        """The results of the execution of ``plan``, as SparkConnectClient's
        ``_execute_and_fetch``: the table of its record batches, or None without any, its schema,
        its plan metrics, its observed metrics, its properties, and its operation ID."""
        import pyarrow as pa

        observed_metrics: list[PlanObservedMetrics] = []
        metrics: list[PlanMetrics] = []
        batches: list[pa.RecordBatch] = []
        schema: Optional[StructType] = None
        properties: dict[str, Any] = {}

        responses = self._execute_plan(plan, observations, True)
        for response in self._results(responses, observations):
            if isinstance(response, StructType):
                schema = response
            elif isinstance(response, pa.RecordBatch):
                batches.append(response)
            elif isinstance(response, PlanMetrics):
                metrics.append(response)
            elif isinstance(response, PlanObservedMetrics):
                observed_metrics.append(response)
            elif isinstance(response, dict):
                properties.update(**response)
            else:
                raise PySparkValueError(
                    errorClass="UNKNOWN_RESPONSE",
                    messageParameters={
                        "response": response,
                    },
                )

        operation_id = responses.operation_id
        if len(batches) > 0:
            if self_destruct:
                results = []
                for batch in batches:
                    # self_destruct frees memory column-wise, but Arrow record batches are
                    # oriented row-wise, so copies each column into its own allocation
                    batch = pa.RecordBatch.from_arrays(
                        [
                            # This call actually reallocates the array
                            pa.concat_arrays([array])
                            for array in batch
                        ],
                        schema=batch.schema,
                    )
                    results.append(batch)
                table = pa.Table.from_batches(batches=results)
                # Ensure only the table has a reference to the batches, so that
                # self_destruct (if enabled) is effective
                del results
                del batches
            else:
                table = pa.Table.from_batches(batches=batches)
            return table, schema, metrics, observed_metrics, properties, operation_id
        else:
            return None, schema, metrics, observed_metrics, properties, operation_id

    def to_table_as_iterator(
        self, plan: "pb2.Plan", observations: dict[str, "Observation"]
    ) -> Iterator[Union[StructType, "pa.Table"]]:
        """The result of ``plan`` as it arrives, as SparkConnectClient's
        ``to_table_as_iterator``: its schema, once the server sends it, and a PyArrow Table of
        each record batch. The execution sets the Observations of ``observations`` by their
        names, and starts with the first item."""
        import pyarrow as pa

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Executing plan {self._proto_to_string(plan, True)}")
        sent = _plan_of(plan)
        for response in self._results(self._execute_plan(sent, observations, True), observations):
            if isinstance(response, StructType):
                yield response
            elif isinstance(response, pa.RecordBatch):
                yield pa.Table.from_batches([response])

    def to_table(
        self, plan: "pb2.Plan", observations: dict[str, "Observation"]
    ) -> tuple["pa.Table", Optional[StructType], ExecutionInfo]:
        """The result of ``plan`` as a PyArrow Table, as SparkConnectClient's ``to_table``: the
        table of the record batches of the server, the schema that it sent, or None, and the
        ``ExecutionInfo`` of the execution, which sets the Observations of ``observations``."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Executing plan {self._proto_to_string(plan, True)}")
        sent = _plan_of(plan)
        table, schema, metrics, observed_metrics, _, operation_id = self._execute_and_fetch(
            sent, observations
        )
        # Create a query execution object.
        ei = ExecutionInfo(metrics, observed_metrics, operation_id)
        assert table is not None
        return table, schema, ei

    def to_pandas(
        self, plan: "pb2.Plan", observations: dict[str, "Observation"], **kwargs: Any
    ) -> tuple["pd.DataFrame", ExecutionInfo]:
        """The result of ``plan`` as a pandas DataFrame, as SparkConnectClient's ``to_pandas``:
        each column converted by its Spark type, in the session time zone, of the
        ``spark.sql.execution.pandas.structHandlingMode``, or of a ``pandasStructHandlingMode``
        keyword argument, and the ``ExecutionInfo`` of the execution."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Executing plan {self._proto_to_string(plan, True)}")
        sent = _plan_of(plan)

        # Get all related configs in a batch
        (
            timezone,
            structHandlingMode,
            selfDestruct,
        ) = self.get_configs(
            "spark.sql.session.timeZone",
            "spark.sql.execution.pandas.structHandlingMode",
            "spark.sql.execution.arrow.pyspark.selfDestruct.enabled",
        )

        # if pandasStructHandlingMode is explicitly set, override the runtime config
        if "pandasStructHandlingMode" in kwargs:
            structHandlingMode = str(kwargs["pandasStructHandlingMode"])

        table, schema, metrics, observed_metrics, _, operation_id = self._execute_and_fetch(
            sent, observations, selfDestruct == "true"
        )
        assert table is not None
        ei = ExecutionInfo(metrics, observed_metrics, operation_id)
        pdf = _to_pandas(
            table,
            schema,
            metrics,
            observed_metrics,
            timezone=timezone,
            struct_handling_mode=structHandlingMode,
            self_destruct=selfDestruct,
        )
        return pdf, ei

    def execute_command(
        self, command: "pb2.Command", observations: Optional[dict[str, "Observation"]] = None
    ) -> tuple[Optional["pd.DataFrame"], dict[str, Any], ExecutionInfo]:
        """Execute ``command``, a ``spark.connect.Command``, as SparkConnectClient's
        ``execute_command``: the pandas DataFrame of the record batches of the responses, or
        None without any, the properties of the results of the responses, of their field names,
        as SparkConnectClient converts them (the resources of a GetResourcesCommand, the
        profile ID of a resource profile, and the DataFrame of a checkpoint), and the
        ``ExecutionInfo`` of the execution, which sets the Observations of ``observations``."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Execute command for command {self._proto_to_string(command, True)}")
        data, _, metrics, observed_metrics, properties, operation_id = self._execute_and_fetch(
            _plan_of_command(command), observations or {}
        )
        # Create a query execution object.
        ei = ExecutionInfo(metrics, observed_metrics, operation_id)
        if data is not None:
            return (data.to_pandas(), properties, ei)
        else:
            return (None, properties, ei)

    def execute_command_as_iterator(
        self, command: "pb2.Command", observations: Optional[dict[str, "Observation"]] = None
    ) -> Iterator[dict[str, Any]]:
        """Execute ``command``, as ``execute_command``, and yield the properties of the result of
        each response as it arrives, as SparkConnectClient's ``execute_command_as_iterator``:
        ``UNKNOWN_RESPONSE`` for a result that is not one, such as metrics or a record batch. Its
        progress does not reach the progress handlers, as PySpark's does not."""
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"Execute command as iterator for command {self._proto_to_string(command, True)}"
            )
        plan = _plan_of_command(command)
        observations = observations or {}
        for response in self._results(self._execute_plan(plan, observations, False), observations):
            if isinstance(response, dict):
                yield response
            else:
                raise PySparkValueError(
                    errorClass="UNKNOWN_RESPONSE",
                    messageParameters={
                        "response": str(response),
                    },
                )

    def add_threadlocal_user_context_extension(self, extension: Any) -> str:
        """Add ``extension``, a ``google.protobuf.Any``, to the user context of the requests that
        the current thread sends, as SparkConnectClient's: to the list of
        ``thread_local.user_context_extensions``, of its ID."""
        if not hasattr(self.thread_local, "user_context_extensions"):
            self.thread_local.user_context_extensions = list()
        extension_id = "threadlocal_" + str(uuid.uuid4())
        self.thread_local.user_context_extensions.append((extension_id, extension))
        return extension_id

    def add_global_user_context_extension(self, extension: Any) -> str:
        """Add ``extension``, a ``google.protobuf.Any``, to the user context of the requests of
        all the threads, as SparkConnectClient's: its ID, of the extension that its
        ``global_user_context_extensions`` list holds."""
        extension_id = "global_" + str(uuid.uuid4())
        with self.global_user_context_extensions_lock:
            self.global_user_context_extensions.append((extension_id, extension))
        return extension_id

    def remove_user_context_extension(self, extension_id: str) -> None:
        """Remove the user context extension of ``extension_id``, as SparkConnectClient's: of the
        current thread, which rebinds ``thread_local.user_context_extensions`` to a list without
        it, or a global one, which rebinds ``global_user_context_extensions``."""
        if extension_id.find("threadlocal_") == 0:
            if not hasattr(self.thread_local, "user_context_extensions"):
                return
            self.thread_local.user_context_extensions = list(
                filter(lambda ex: ex[0] != extension_id, self.thread_local.user_context_extensions)
            )
        elif extension_id.find("global_") == 0:
            with self.global_user_context_extensions_lock:
                self.global_user_context_extensions = list(
                    filter(lambda ex: ex[0] != extension_id, self.global_user_context_extensions)
                )

    def clear_user_context_extensions(self) -> None:
        """Remove the user context extensions of the current thread and the global ones, as
        SparkConnectClient's: each is a new, empty list, that of the thread if it has one."""
        if hasattr(self.thread_local, "user_context_extensions"):
            self.thread_local.user_context_extensions = list()
        with self.global_user_context_extensions_lock:
            self.global_user_context_extensions = list()

    def add_tag(self, tag: str) -> None:
        """Add ``tag`` to the tags of the current thread, which the executions that it starts
        carry, as SparkConnectClient's: to the set of ``thread_local.tags``, of the errors of
        ``_throw_if_invalid_tag``."""
        self._throw_if_invalid_tag(tag)
        if not hasattr(self.thread_local, "tags"):
            self.thread_local.tags = set()
        self.thread_local.tags.add(tag)

    def remove_tag(self, tag: str) -> None:
        """Remove ``tag`` from the tags of the current thread, as SparkConnectClient's: of the
        errors of ``add_tag``, and nothing for a tag that the thread does not have."""
        self._throw_if_invalid_tag(tag)
        if not hasattr(self.thread_local, "tags"):
            self.thread_local.tags = set()
        # Use discard, not remove: removing an absent tag is a documented no-op
        # (see SparkSession.removeTag), matching the Classic behavior.
        self.thread_local.tags.discard(tag)

    def get_tags(self) -> set[str]:
        """The tags of the current thread, as SparkConnectClient's: the set of
        ``thread_local.tags``, which the later changes of the tags change too."""
        if not hasattr(self.thread_local, "tags"):
            self.thread_local.tags = set()
        return self.thread_local.tags

    def clear_tags(self) -> None:
        """Remove the tags of the current thread, as SparkConnectClient's: ``thread_local.tags``
        is a new, empty set."""
        self.thread_local.tags = set()

    def _throw_if_invalid_tag(self, tag: str) -> None:
        """
        Validate if a tag for ExecutePlanRequest.tags is valid. Throw ``ValueError`` if
        not.
        """
        spark_job_tags_sep = ","
        if tag is None:
            raise PySparkValueError(
                errorClass="CANNOT_BE_NONE", messageParameters={"arg_name": "Spark Connect tag"}
            )
        if spark_job_tags_sep in tag:
            raise PySparkValueError(
                errorClass="VALUE_ALLOWED",
                messageParameters={
                    "arg_name": "Spark Connect tag",
                    "disallowed_value": spark_job_tags_sep,
                },
            )
        if len(tag) == 0:
            raise PySparkValueError(
                errorClass="VALUE_NOT_NON_EMPTY_STR",
                messageParameters={"arg_name": "Spark Connect tag", "arg_value": tag},
            )

    def interrupt_all(self) -> Optional[list[str]]:
        """Interrupt the executions of the session, as SparkConnectClient's: the IDs of the
        operations interrupted."""
        return self._session.interruptAll()

    def interrupt_tag(self, tag: str) -> Optional[list[str]]:
        """Interrupt the executions of the session of the tag ``tag``, as SparkConnectClient's:
        the IDs of the operations interrupted."""
        # As PySpark's request of an interrupt of a tag, which asserts that there is one.
        assert tag is not None
        return self._session.interruptTag(tag)

    def interrupt_operation(self, op_id: str) -> Optional[list[str]]:
        """Interrupt the execution of the operation ``op_id``, as SparkConnectClient's: the IDs
        of the operations interrupted."""
        # As PySpark's request of an interrupt of an operation, which asserts that there is one.
        assert op_id is not None
        return self._session.interruptOperation(op_id)

    def register_udf(
        self,
        function: Any,
        return_type: "Union[DataType, str]",
        name: Optional[str] = None,
        eval_type: int = PythonEvalType.SQL_BATCHED_UDF,
        deterministic: bool = True,
    ) -> str:
        """Register the Python function ``function`` as a temporary function of the session of
        the name ``name``, else of a name that the client generates, as SparkConnectClient's
        ``register_udf``: the name. A DDL ``return_type`` is parsed by the server, and the
        function is pickled with the type, as PySpark's ``PythonUDF`` does. As PySpark's, the
        name and ``deterministic`` are set in the fields of the command first, then the return
        type is parsed and converted, then ``eval_type`` is set, and the function is pickled
        last, so each raises the error of its protobuf field or of its conversion in that
        order."""
        import pyspark.sql.connect.proto as pb2
        from pyspark.serializers import CloudPickleSerializer

        if name is None:
            name = f"fun_{uuid.uuid4().hex}"
        # As PySpark's CommonInlineUserDefinedFunction.to_plan_udf, which sets these first.
        fun = pb2.CommonInlineUserDefinedFunction()
        fun.function_name = name
        fun.deterministic = deterministic
        # And then its PythonUDF.to_plan: the parse of a DDL string, the proto of the return type,
        # the eval type, and the pickle.
        output_type = return_type
        if isinstance(return_type, str):
            output_type = self._session._parse_ddl(return_type)
            assert isinstance(output_type, DataType)
        else:
            _check_data_type(return_type)
        udf = pb2.PythonUDF()
        udf.eval_type = eval_type
        command = CloudPickleSerializer().dumps((function, output_type))
        self._session._registerPythonUdf(
            fun.function_name,
            output_type,
            udf.eval_type,
            command,
            f"{sys.version_info.major}.{sys.version_info.minor}",
            fun.deterministic,
        )
        return name

    def register_udtf(
        self,
        function: Any,
        return_type: "Optional[Union[DataType, str]]",
        name: str,
        eval_type: int = PythonEvalType.SQL_TABLE_UDF,
        deterministic: bool = True,
    ) -> str:
        """Register the user-defined table function class ``function`` as the temporary
        function ``name`` of the session, as SparkConnectClient's ``register_udtf``: the name. A
        DDL ``return_type`` is left to the server to parse. As PySpark's, the ``__name__`` of the
        handler class is read, the name and ``deterministic`` are set in the fields of the
        command, the return type is converted and ``eval_type`` set before the handler is
        pickled, so a handler that does not pickle raises ``UDTF_SERIALIZATION_ERROR`` only of
        valid arguments."""
        import pyspark.sql.connect.proto as pb2
        from pyspark.sql.connect.utils import get_python_ver
        from pyspark.sql.udtf import _pickled_handler

        # As PySpark's PythonUDTF, which reads the name of the handler class when it is made.
        _ = function.__name__
        # As PySpark's CommonInlineUserDefinedTableFunction.udtf_plan, which sets these first, and
        # then converts the return type and sets the eval type (PythonUDTF.to_plan) before it
        # pickles the handler: `_registerPythonUdtf` converts them, and then calls the pickling.
        fun = pb2.CommonInlineUserDefinedTableFunction()
        fun.function_name = name
        fun.deterministic = deterministic
        self._session._registerPythonUdtf(
            fun.function_name,
            return_type,
            eval_type,
            functools.partial(_pickled_handler, function),
            get_python_ver(),
            fun.deterministic,
        )
        return name

    def register_data_source(self, dataSource: type["DataSource"]) -> None:
        """Register the Python data source class ``dataSource`` with the session, of its
        ``name()``, as SparkConnectClient's ``register_data_source``."""
        self._session.dataSource.register(dataSource)

    def register_java(
        self,
        name: str,
        javaClassName: str,
        return_type: "Optional[Union[DataType, str]]" = None,
        aggregate: bool = False,
    ) -> None:
        """Register the Java function of the class ``javaClassName`` as the temporary function
        ``name`` of the session, as SparkConnectClient's ``register_java``: an aggregate one for
        ``aggregate``, of the class's own type, and of a ``return_type``, a DataType or a DDL
        string that the server parses, a function that is not an aggregate one, as PySpark's
        ``JavaUDF`` of an output type is. As PySpark's, the name, the class name, the return
        type and then ``aggregate``, unless there is a return type, are set in the fields of the
        command in that order, so each raises the error of its protobuf field in that order."""
        import pyspark.sql.connect.proto as pb2

        # As PySpark's CommonInlineUserDefinedFunction.to_plan_judf and JavaUDF.to_plan.
        fun = pb2.CommonInlineUserDefinedFunction()
        fun.function_name = name
        java = pb2.JavaUDF()
        java.class_name = javaClassName
        if return_type is not None and not isinstance(return_type, str):
            _check_data_type(return_type)
        java.aggregate = aggregate if return_type is None else False
        self._session._registerJavaFunction(
            fun.function_name, java.class_name, return_type, java.aggregate
        )

    def add_artifacts(self, *paths: str, pyfile: bool, archive: bool, file: bool) -> None:
        """Add the local files, or ``file:`` URIs, ``paths`` as artifacts of the session, as
        SparkConnectClient's ``add_artifacts``: each named by the first of ``pyfile``,
        ``archive`` and ``file`` that it matches, which the client, unlike
        ``SparkSession.addArtifacts``, does not check against each other, else a jar."""
        self._session._add_artifacts_of_flags(*paths, pyfile=pyfile, archive=archive, file=file)

    def copy_from_local_to_fs(self, local_path: str, dest_path: str) -> None:
        """Copy the local file ``local_path`` to the absolute path ``dest_path`` of the file
        system of the server, as SparkConnectClient's ``copy_from_local_to_fs``: of the
        assertions and the errors of its ArtifactManager, without the check of a scheme of
        ``SparkSession.copyFromLocalToFs``."""
        abs_path = Path(local_path).absolute()
        assert abs_path.is_file(), "local path must be a file path."
        assert Path(dest_path).is_absolute(), "destination FS path must be an absolute path."
        # As the name of the artifact of PySpark's ArtifactManager, which raises the TypeError of
        # the concatenation for a destination that is not a str.
        _ = "forward_to_fs" + dest_path
        self._session._copy_from_local_to_fs(str(abs_path), dest_path)

    def cache_artifact(self, blob: bytes) -> str:
        """Cache ``blob``, a bytes-like object, as an artifact of the session, as
        SparkConnectClient's ``cache_artifact``: its SHA-256 hash, of which it is uploaded unless
        the server has it."""
        return self.cache_artifacts([blob])[0]

    def cache_artifacts(self, blobs: list[bytes]) -> list[str]:
        """Cache ``blobs``, bytes-like objects, as artifacts of the session, as
        SparkConnectClient's ``cache_artifacts``: their SHA-256 hashes, in order, of which those
        that the server does not have are uploaded, once each, in one request. As PySpark's
        ``ArtifactManager``, the hashes of ``hashlib`` come first, so a blob that is not
        bytes-like raises its ``TypeError`` before anything is sent."""
        import hashlib

        blobs = list(blobs)
        for blob in blobs:
            hashlib.sha256(blob)
        return self._session._cache_artifacts(
            [
                blob if isinstance(blob, (bytes, bytearray)) else bytes(memoryview(blob))
                for blob in blobs
            ]
        )

    def newSession(self) -> "SparkConnectClient":
        """A client of a new session of the same connection, as SparkConnectClient's
        ``newSession``: of none of the state of this session, of its hooks, its deadlines and
        whether its executions are reattachable, and of the default retry policy."""
        builder = None
        if self._builder is not None:
            builder = copy.deepcopy(self._builder)
            builder._params.pop(ChannelBuilder.PARAM_SESSION_ID, None)
        session = self._session.newSession()
        # A client that no session object owns, as PySpark's, lives until the exit.
        session._keep_until_exit()
        return SparkConnectClient._of_session(session, builder)

    def clone(self, new_session_id: Optional[str] = None) -> "SparkConnectClient":
        """A client of a clone of this session on the server, of the ID ``new_session_id``, a
        UUID, if one is given, as SparkConnectClient's ``clone``: of a copy of its SQL
        configurations, temporary views, registered functions and catalog state, of the hooks,
        the deadlines and the reattachable executions of this client, and of the default retry
        policy."""
        session = self._session.cloneSession(new_session_id)
        # A client that no session object owns, as PySpark's, lives until the exit.
        session._keep_until_exit()
        builder = None
        if self._builder is not None:
            builder = copy.deepcopy(self._builder)
            builder.set(ChannelBuilder.PARAM_SESSION_ID, session.session_id)
        return SparkConnectClient._of_session(session, builder)

    def register_progress_handler(self, handler: Any) -> None:
        """Register ``handler`` to be called with the progress of the executions of the session,
        once, as SparkConnectClient's ``register_progress_handler``."""
        self._session.registerProgressHandler(handler)

    def remove_progress_handler(self, handler: Any) -> None:
        """Remove the progress handler ``handler``, as SparkConnectClient's
        ``remove_progress_handler``: a ``ValueError`` for one that is not registered."""
        self._session.removeProgressHandler(handler)

    def clear_progress_handlers(self) -> None:
        """Remove the progress handlers, as SparkConnectClient's ``clear_progress_handlers``."""
        self._session.clearProgressHandlers()

    def disable_reattachable_execute(self) -> "SparkConnectClient":
        """Make the executions of the session not reattachable from the next one on, as
        SparkConnectClient's ``disable_reattachable_execute``: the client."""
        self._session._set_reattachable_execute(False)
        return self

    def enable_reattachable_execute(self) -> "SparkConnectClient":
        """Make the executions of the session reattachable from the next one on, as
        SparkConnectClient's ``enable_reattachable_execute``: the client."""
        self._session._set_reattachable_execute(True)
        return self

    def set_retry_policies(self, policies: Iterable["RetryPolicy"]) -> None:
        """Set the retry policies of the RPCs of the session, as SparkConnectClient's
        ``set_retry_policies``: a failure is retried by the first of ``policies`` that can retry
        it, with its backoff, until its retries are spent. The Rust transport retries with the
        parameters of the policies as they are now, and classifies the failures as PySpark's
        ``DefaultPolicy.can_retry`` does, which a ``RetryPolicy`` never retries, so a policy of
        another ``can_retry`` or ``to_state`` raises ``NOT_IMPLEMENTED``."""
        self._set_retry_policies(list(policies), None)

    def get_retry_policies(self) -> list["RetryPolicy"]:
        """The retry policies of the RPCs of the session, as SparkConnectClient's
        ``get_retry_policies``: those that ``set_retry_policies`` set, else a ``DefaultPolicy``
        of the parameters that the transport retries with."""
        policies = _RETRY_POLICIES.get(self._session)
        if policies is None:
            from pyspark.sql.connect.client.retries import DefaultPolicy

            parameters = self._session._retry_policy()
            parameters.pop("max_retry_exception_elapsed_time")
            parameters.pop("retries_failures")
            policies = [DefaultPolicy(**parameters)]
            _RETRY_POLICIES[self._session] = policies
        return list(policies)

    def _set_retry_policies(
        self, policies: Optional[list[Any]], max_retry_exception_elapsed_time: Optional[float]
    ) -> None:
        """Make the Rust transport retry with ``policies``, if given, and for up to
        ``max_retry_exception_elapsed_time`` seconds of the retries of a ``RetryException``, if
        given, from the calls that start from now on."""
        parameters = self._session._retry_policy()
        if max_retry_exception_elapsed_time is not None:
            parameters["max_retry_exception_elapsed_time"] = _elapsed_time_bound(
                max_retry_exception_elapsed_time
            )
        if policies is not None:
            parameters.update(_retry_parameters(policies))
        self._session._set_retry_policy(**parameters)
        if policies is not None:
            _RETRY_POLICIES[self._session] = policies

    def release_session(self) -> None:
        """Release the session on the server, as SparkConnectClient's: its state there."""
        self._session._release_session()

    def close(self) -> None:
        """Close the client, as SparkConnectClient's: of a session that ``is_closed``, and is
        not released on the server."""
        self._session._close()

    def _resources(self) -> dict[str, Any]:
        """The resources of the cluster, as SparkConnectClient's: of a GetResourcesCommand, by
        their names."""
        import pyspark.sql.connect.proto as pb2

        command = pb2.Command()
        command.get_resources_command.SetInParent()
        _, properties, _ = self.execute_command(command)
        return properties["get_resources_command_result"]

    def _create_profile(self, profile: Any) -> int:
        """Create ``profile``, a ``spark.connect.ResourceProfile``, on the server, as
        SparkConnectClient's: the profile ID that the server gives it. The ResourceProfile of
        pyspark.sql.connect.resource.profile creates its profile with it."""
        import pyspark.sql.connect.proto as pb2

        command = pb2.Command()
        command.create_resource_profile_command.profile.CopyFrom(profile)
        _, properties, _ = self.execute_command(command)
        return properties["create_resource_profile_command_result"]

    def _delete_ml_cache(self, cache_ids: list[str], evict_only: bool = False) -> list[str]:
        """Delete the cached objects of ``cache_ids`` on the server, as SparkConnectClient's: the
        IDs deleted, or none if it fails."""
        import threading

        import pyspark.sql.connect.proto as pb2

        try:
            if len(cache_ids) > 0:
                if self._ml_cache_rpc_thread == threading.get_ident():
                    return []
                command = pb2.Command()
                command.ml_command.delete.obj_refs.extend(
                    [pb2.ObjectRef(id=cache_id) for cache_id in cache_ids]
                )
                command.ml_command.delete.evict_only = evict_only
                self._ml_cache_rpc_thread = threading.get_ident()
                try:
                    _, properties, _ = self.execute_command(command)
                finally:
                    self._ml_cache_rpc_thread = None
                if "ml_command_result" in properties:
                    ml_command_result = properties["ml_command_result"]
                    return ml_command_result.operator_info.obj_ref.id.split(",")
            return []
        except Exception:
            return []

    def _cleanup_ml_cache(self) -> None:
        """Clean the ML cache of the session on the server, as SparkConnectClient's."""
        import threading

        try:
            import pyspark.sql.connect.proto as pb2

            if self._ml_cache_rpc_thread == threading.get_ident():
                return
            command = pb2.Command()
            command.ml_command.clean_cache.SetInParent()
            self._ml_cache_rpc_thread = threading.get_ident()
            try:
                self.execute_command(command)
            finally:
                self._ml_cache_rpc_thread = None
        except Exception:
            pass

    def _get_ml_cache_info(self) -> list[str]:
        """The cached ML objects of the session on the server, as SparkConnectClient's: their
        JSON descriptions."""
        import pyspark.sql.connect.proto as pb2

        command = pb2.Command()
        command.ml_command.get_cache_info.SetInParent()
        _, properties, _ = self.execute_command(command)
        if "ml_command_result" in properties:
            ml_command_result = properties["ml_command_result"]
            return [item.string for item in ml_command_result.param.array.elements]
        return []

    def _query_model_size(self, model_ref_id: str) -> int:
        """The size of the cached model of ``model_ref_id`` on the server, as SparkConnectClient's:
        in bytes."""
        import pyspark.sql.connect.proto as pb2

        command = pb2.Command()
        command.ml_command.get_model_size.CopyFrom(
            pb2.MlCommand.GetModelSize(model_ref=pb2.ObjectRef(id=model_ref_id))
        )
        _, properties, _ = self.execute_command(command)
        return properties["ml_command_result"].param.long


class _GlobalUserContextExtensions(list):
    """The global user context extensions of a client, as the list of PySpark's
    ``SparkConnectClient.global_user_context_extensions``, of their IDs and their
    ``google.protobuf.Any``: a list whose changes the session of the client takes, so that the
    requests of all its threads carry them, as PySpark's requests read the list, while it is the
    list of the clients of the session. Its copies and its pickles are plain lists."""

    def __init__(self, session: Any, extensions: Iterable[Any] = ()) -> None:
        super().__init__(extensions)
        self._session_ref = weakref.ref(session)

    def _sync(self) -> None:
        session = self._session_ref()
        if session is None:
            return
        state = _CLIENT_STATES.get(session)
        # A list that the attribute of the session's clients is no longer set to is not the one
        # that the requests read, as in PySpark, and neither is a copy's.
        if state is None or state.get("global_user_context_extensions") is not self:
            return
        session._set_global_user_context_extensions(_serialized_extensions(self))

    def __reduce_ex__(self, protocol: Any) -> Any:
        return (list, (list(self),))

    def __copy__(self) -> list[Any]:
        return list(self)

    def __deepcopy__(self, memo: Any) -> list[Any]:
        return copy.deepcopy(list(self), memo)


class _ThreadExtensions(list):
    """The user context extensions of the threads that have it, as the list of PySpark's
    ``thread_local.user_context_extensions``, of their IDs and their ``google.protobuf.Any``: the
    list of a thread, which the requests of the thread carry, as PySpark's read it, and of the
    threads that are given it, as those of ``inheritable_thread_target``. Its changes are those of
    the list of the Rust transport that it holds, which the threads that have it share. Its copies
    and its pickles are plain lists."""

    def __init__(self, session: Any, extensions: Iterable[Any]) -> None:
        super().__init__(extensions)
        self._session_ref = weakref.ref(session)
        self._handle = session._new_user_context_extensions(_serialized_extensions(self))

    @classmethod
    def _of(cls, thread_local: Any, extensions: Any) -> Any:
        # The list of the current thread once `thread_local.user_context_extensions` is set to
        # `extensions`: a list of the session, which the thread shares, or a new one of its
        # extensions, which the thread has alone.
        session = thread_local._session_ref()
        if session is None:
            return extensions
        if not (isinstance(extensions, cls) and extensions._session_ref() is session):
            extensions = cls(session, extensions)
        session._share_thread_user_context_extensions(extensions._handle)
        return extensions

    def _sync(self) -> None:
        self._handle._replace(_serialized_extensions(self))

    def _copy(self) -> "_ThreadExtensions":
        # A list of deep copies of the extensions, which no thread has yet.
        session = self._session_ref()
        assert session is not None
        return type(self)(session, copy.deepcopy(list(self)))

    def __reduce_ex__(self, protocol: Any) -> Any:
        return (list, (list(self),))

    def __copy__(self) -> list[Any]:
        return list(self)

    def __deepcopy__(self, memo: Any) -> list[Any]:
        return copy.deepcopy(list(self), memo)


class _ThreadTags(set):
    """The operation tags of the threads that have it, as the set of PySpark's
    ``thread_local.tags``: the set of a thread, which its executions carry, as PySpark's read it,
    and of the threads that are given it, as those of ``inheritable_thread_target``. Its changes
    are those of the set of the Rust transport that it holds, which the threads that have it
    share: a str that UTF-8 cannot encode fails each request, as in PySpark, and an object that is
    neither a str nor bytes raises the ``TypeError`` of the protobuf field of the tags, which
    PySpark raises when it builds a request, when it is added. Its copies and its pickles are plain
    sets."""

    def __init__(self, session: Any, tags: Iterable[Any]) -> None:
        super().__init__(tags)
        self._session_ref = weakref.ref(session)
        self._handle = session._new_tags(list(self))

    @classmethod
    def _of(cls, thread_local: Any, tags: Any) -> Any:
        # The set of the current thread once `thread_local.tags` is set to `tags`: a set of the
        # session, which the thread shares, or a new one of its tags, which the thread has alone.
        session = thread_local._session_ref()
        if session is None:
            return tags
        if not (isinstance(tags, cls) and tags._session_ref() is session):
            tags = cls(session, tags)
        session._share_thread_tags(tags._handle)
        return tags

    def _sync(self) -> None:
        self._handle._replace(list(self))

    def _copy(self) -> "_ThreadTags":
        # A set of the same tags, which no thread has yet.
        session = self._session_ref()
        assert session is not None
        return type(self)(session, copy.deepcopy(set(self)))

    def __repr__(self) -> str:
        # As the repr of PySpark's plain set.
        return repr(set(self))

    def __reduce_ex__(self, protocol: Any) -> Any:
        return (set, (list(self),))

    def __copy__(self) -> set[Any]:
        return set(self)

    def __deepcopy__(self, memo: Any) -> set[Any]:
        return copy.deepcopy(set(self), memo)


def _serialized_extensions(extensions: Iterable[Any]) -> list[tuple[Any, bytes]]:
    # The IDs and the bytes of the google.protobuf.Any of `extensions`, as the Rust transport takes
    # them for the user context of its requests.
    return [(extension_id, extension.SerializeToString()) for extension_id, extension in extensions]


def _synced(base: type, name: str) -> Any:
    # A method of `base`, a list or a set, that changes it, after which the Rust transport takes
    # the extensions or the tags: a change that it does not take, of an object that it cannot
    # send, is undone, and its error raised.
    method = getattr(base, name)

    def synced(self: Any, *args: Any, **kwargs: Any) -> Any:
        before = base(self)
        result = method(self, *args, **kwargs)
        try:
            self._sync()
        except BaseException:
            base.clear(self)
            if base is list:
                base.extend(self, before)
            else:
                base.update(self, before)
            raise
        return result

    synced.__name__ = synced.__qualname__ = name
    return synced


for _list in (_GlobalUserContextExtensions, _ThreadExtensions):
    for _name in (
        "append",
        "extend",
        "insert",
        "remove",
        "pop",
        "clear",
        "sort",
        "reverse",
        "__setitem__",
        "__delitem__",
        "__iadd__",
        "__imul__",
    ):
        setattr(_list, _name, _synced(list, _name))
for _name in (
    "add",
    "discard",
    "remove",
    "pop",
    "clear",
    "update",
    "difference_update",
    "intersection_update",
    "symmetric_difference_update",
    "__ior__",
    "__iand__",
    "__isub__",
    "__ixor__",
):
    setattr(_ThreadTags, _name, _synced(set, _name))
del _list, _name


def _extensions_of(extensions: Any) -> list[Any]:
    # The user context extensions of the Rust transport, as SparkConnectClient keeps them: the
    # ID and the google.protobuf.Any of each, of the Any of pyspark.sql.connect.proto's runtime.
    from pyspark.sql.connect.proto._rust import Any as ProtoAny

    return [(extension_id, ProtoAny.FromString(data)) for extension_id, data in extensions]


class _ThreadLocalOfClient(threading.local):
    """The state of each thread of the session of a client, as SparkConnectClient's
    ``thread_local``, a ``threading.local``: the attributes that a thread sets are its own, and
    PySpark's ``tags`` and ``user_context_extensions`` exist once a thread uses them. The set of
    the tags and the list of the extensions that a thread sets are those that its requests carry
    (see ``_ThreadTags`` and ``_ThreadExtensions``), and so are those of the threads that are given
    them, as by ``inheritable_thread_target``. Deleting them leaves the thread with none."""

    # The session of the clients, held weakly, which all threads see.
    __slots__ = ("_session_ref",)

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "tags":
            value = _ThreadTags._of(self, value)
        elif name == "user_context_extensions":
            value = _ThreadExtensions._of(self, value)
        super().__setattr__(name, value)

    def __delattr__(self, name: str) -> None:
        super().__delattr__(name)
        session = self._session_ref()
        if session is None:
            return
        if name == "tags":
            session._clear_thread_tags()
        elif name == "user_context_extensions":
            session._clear_thread_user_context_extensions()


def _thread_local_copies(thread_local: Any) -> list[tuple[str, Any]]:
    """The attributes of the current thread of ``thread_local``, a client's, as PySpark's
    ``inheritable_thread_target`` copies them for the threads of a function, of ``copy.deepcopy``
    of each: the copies of the set of the tags and of the list of the extensions are a set and a
    list that the threads that are given them share, as the threads share PySpark's copies."""
    copies = []
    for attr, value in thread_local.__dict__.items():
        if isinstance(value, (_ThreadTags, _ThreadExtensions)):
            value = value._copy()
        else:
            value = copy.deepcopy(value)
        copies.append((attr, value))
    return copies


# The attributes of the clients of each session, which they share, as PySpark's session has one
# client (see `SparkConnectClient.__slots__`): held weakly by their session, which they hold
# weakly too.
_CLIENT_STATES: "weakref.WeakKeyDictionary[Any, dict[str, Any]]" = weakref.WeakKeyDictionary()
_CLIENT_STATES_LOCK = threading.Lock()


def _client_state(session: Any, builder: Optional[ChannelBuilder]) -> dict[str, Any]:
    """The attributes of the clients of ``session``, made with the first of them: its channel
    builder ``builder``, as PySpark's ``_builder``, else a ``DefaultChannelBuilder`` of the
    connection of the session, of the host and the token of the client; its ``thread_local``; its
    ``global_user_context_extensions``, those that the session has, whose changes, and a list that
    the attribute is set to, are the session's (see ``SparkConnectClient.__setattr__``); and their
    lock."""
    with _CLIENT_STATES_LOCK:
        state = _CLIENT_STATES.get(session)
        if state is None:
            connection_string = getattr(session, "_connection_string", None)
            if builder is None and connection_string is not None:
                # As PySpark's client of a session, of the channel builder of its connection.
                builder = DefaultChannelBuilder(connection_string())
            thread_local = _ThreadLocalOfClient()
            thread_local._session_ref = weakref.ref(session)
            extensions = _extensions_of(session._user_context_extensions(True))
            state = {
                "_builder": builder,
                "thread_local": thread_local,
                "global_user_context_extensions": _GlobalUserContextExtensions(
                    session, extensions
                ),
                "global_user_context_extensions_lock": threading.Lock(),
            }
            _CLIENT_STATES[session] = state
            # As PySpark's requests, which serialize the extensions when they are built: the calls
            # of the session serialize them again before they send requests.
            session._set_user_context_refresh(
                functools.partial(_refresh_user_context_extensions, weakref.ref(session))
            )
    return state


def _refresh_user_context_extensions(session_ref: Any) -> None:
    """Serialize the user context extensions of the session of ``session_ref`` again for the
    Rust transport, of its clients' global list and of the list of the current thread, before a
    call of the session sends requests, as PySpark serializes them when it builds each request:
    so a ``google.protobuf.Any`` of the lists that was changed in place is sent as it is now."""
    session = session_ref()
    if session is None:
        return
    state = _CLIENT_STATES.get(session)
    if state is None:
        return
    extensions = state.get("global_user_context_extensions")
    if isinstance(extensions, _GlobalUserContextExtensions) and extensions:
        extensions._sync()
    own = getattr(state.get("thread_local"), "user_context_extensions", None)
    if isinstance(own, _ThreadExtensions) and own:
        own._sync()


# The client of each session, by its ID, while it is in use.
_CLIENTS: "weakref.WeakValueDictionary[str, SparkConnectClient]" = weakref.WeakValueDictionary()

# The retry policies that `set_retry_policies` set on each session, which its clients give back,
# whatever client of the session set them, as PySpark's client keeps the policies objects it is
# given. The Rust transport retries with their parameters.
_RETRY_POLICIES: "weakref.WeakKeyDictionary[Any, list[Any]]" = weakref.WeakKeyDictionary()


def _client_of(session: Any) -> "SparkConnectClient":
    """The client of ``session``, the same object while it is in use, as ``SparkSession.client``
    is one object per session."""
    client = _CLIENTS.get(session.session_id)
    if client is None or client._session is not session:
        client = SparkConnectClient._of_session(session)
        _CLIENTS[session.session_id] = client
    return client


def _check_data_type(data_type: Any) -> None:
    """Raise PySpark's ``UNSUPPORTED_OPERATION`` of ``data_type`` unless it is a ``DataType``, as
    ``pyspark_types_to_proto_types`` raises it for the return type of a UDF that is neither a
    ``DataType`` nor a DDL string."""
    if not isinstance(data_type, DataType):
        raise PySparkValueError(
            errorClass="UNSUPPORTED_OPERATION",
            messageParameters={"operation": f"data type {data_type}"},
        )


def _hook_factory(hook: Any) -> Any:
    """The factory of the hook ``hook`` of SparkConnectClient's ``session_hooks``, which the
    session calls with itself, as ``SparkSession(hook_factories=...)`` calls its factories: the
    hook itself, whose ``on_execute_plan`` the executions call."""
    return lambda _session: hook


def _plan_of(plan: Any) -> "pb2.Plan":
    """A copy of ``plan``, as SparkConnectClient copies a plan into its request: protobuf's
    ``TypeError`` for an object that is not a ``spark.connect.Plan``."""
    import pyspark.sql.connect.proto as pb2

    sent = pb2.Plan()
    sent.CopyFrom(plan)
    return sent


def _plan_of_command(command: Any) -> "pb2.Plan":
    """The plan of ``command``, as SparkConnectClient sets a command in the plan of its request:
    protobuf's ``TypeError`` for an object that is not a ``spark.connect.Command``."""
    import pyspark.sql.connect.proto as pb2

    sent = pb2.Plan()
    sent.command.CopyFrom(command)
    return sent


def _retry_parameters(policies: list[Any]) -> dict[str, Any]:
    """The parameters of ``policies`` that the Rust transport retries with, as PySpark's
    ``Retrying`` retries with them: a failure is retried by the first policy that can retry it, a
    ``DefaultPolicy`` of the failures that the transport retries, as a ``RetryPolicy`` can retry
    none, so those of the first ``DefaultPolicy`` retry, and without one no failure is retried,
    nor are retries spent, of which ``Retrying`` warns. A policy of another ``can_retry``, of the
    failures that it retries, or of another ``to_state``, of the waits of its retries, raises
    ``NOT_IMPLEMENTED``: the transport cannot call them."""
    from pyspark.sql.connect.client.retries import DefaultPolicy, RetryPolicy

    effective = None
    for policy in policies:
        can_retry = getattr(getattr(policy, "can_retry", None), "__func__", None)
        to_state = getattr(getattr(policy, "to_state", None), "__func__", None)
        if to_state is not RetryPolicy.to_state or can_retry not in (
            DefaultPolicy.can_retry,
            RetryPolicy.can_retry,
        ):
            raise PySparkNotImplementedError(
                errorClass="NOT_IMPLEMENTED",
                messageParameters={
                    "feature": f"The retry policy {type(policy).__name__} of its own can_retry "
                    "or to_state"
                },
            )
        if effective is None and can_retry is DefaultPolicy.can_retry:
            effective = policy
    if effective is None:
        return {"max_retries": 0, "retries_failures": False}
    max_retries = effective.max_retries
    if max_retries is not None:
        # As RetryPolicyState, which retries while the attempts are fewer than it.
        max_retries = min(max(math.ceil(max_retries), 0), 2**32 - 1)
    max_backoff = effective.max_backoff
    max_server_retry_delay = effective.max_server_retry_delay
    return {
        "max_retries": max_retries,
        "initial_backoff": _milliseconds("initial_backoff", effective.initial_backoff),
        "max_backoff": None if max_backoff is None else _milliseconds("max_backoff", max_backoff),
        "backoff_multiplier": float(effective.backoff_multiplier),
        "jitter": _milliseconds("jitter", effective.jitter),
        "min_jitter_threshold": _milliseconds(
            "min_jitter_threshold", effective.min_jitter_threshold
        ),
        "recognize_server_retry_delay": bool(effective.recognize_server_retry_delay),
        "max_server_retry_delay": (
            None
            if max_server_retry_delay is None
            else _milliseconds("max_server_retry_delay", max_server_retry_delay)
        ),
        "retries_failures": True,
    }


def _milliseconds(name: str, value: Any) -> int:
    """The milliseconds ``value`` of the parameter ``name`` of a retry policy, a whole number of
    them, which the Rust transport waits: ``NOT_IMPLEMENTED`` for a negative or fractional one."""
    if (
        isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0
        and value == int(value)
    ):
        return int(value)
    raise PySparkNotImplementedError(
        errorClass="NOT_IMPLEMENTED",
        messageParameters={
            "feature": f"The {name} {value!r} of a retry policy, which is not a whole number of "
            "milliseconds,"
        },
    )


def _elapsed_time_bound(seconds: float) -> Optional[float]:
    """The bound of the retries of a ``RetryException``, of ``max_retry_exception_elapsed_time``
    seconds, as PySpark's ``Retrying`` compares the time elapsed with it: None for an infinite
    bound, and 0 for a negative one, which the first of them passes."""
    if math.isinf(seconds) and seconds > 0:
        return None
    return max(float(seconds), 0.0)


# The message of the warning of PySpark's `Retrying` once the retries of a call are spent, and the
# module whose `Retrying._wait` warns it.
_RETRIES_EXCEEDED = "[RETRIES_EXCEEDED] The maximum number of retries has been exceeded."
_RETRIES_MODULE = __name__ + ".retries"
# The registry of the warnings of that module that were shown: its own once it is imported, which
# it imports grpc to be, and this one until then, which the module takes as its own.
_RETRIES_WARNING_REGISTRY: dict[Any, Any] = {}


@functools.cache
def _retries_exceeded_locations() -> Optional[tuple[str, int, int]]:
    """The file of pyspark.sql.connect.client.retries and the lines of the two ``warnings.warn``
    calls of its ``Retrying._wait``: of the retries of a ``RetryException`` past their elapsed
    time, then of a policy whose retries are spent. Read of the source of the module, which is not
    imported, as it imports grpc. None for a module of no source."""
    import importlib.util
    import linecache

    spec = importlib.util.find_spec(_RETRIES_MODULE)
    if spec is None or spec.origin is None:
        return None
    call = f'warnings.warn("{_RETRIES_EXCEEDED}")'
    lines = [
        number
        for number, line in enumerate(linecache.getlines(spec.origin), start=1)
        if line.strip() == call
    ]
    if len(lines) != 2:
        return None
    return spec.origin, lines[0], lines[1]


def _warn_retries_exceeded(retry_exception: bool) -> None:
    """Warn as PySpark's ``Retrying._wait`` warns once the retries of a call are spent, before it
    raises the failure, for the Rust transport, which calls this on the thread of the call: the
    UserWarning ``[RETRIES_EXCEEDED]``, of the location that ``warnings.warn`` gives it there, the
    line of the call in pyspark.sql.connect.client.retries, of that module and of its registry of
    the warnings that it showed. So the filters of the module (``filterwarnings(...,
    module="pyspark.sql.connect.client.retries")``) and the actions of a location (``"default"``,
    ``"module"``) take it as they take PySpark's, and a filter of ``"error"`` raises it in place of
    the failure. ``retry_exception`` is of the retries of a ``RetryException`` past their elapsed
    time, else of a policy. Of a module of no source, the warning is of this function."""
    locations = _retries_exceeded_locations()
    if locations is None:
        warnings.warn(_RETRIES_EXCEEDED)
        return
    filename, retry_exception_line, policy_line = locations
    module = sys.modules.get(_RETRIES_MODULE)
    registry = (
        _RETRIES_WARNING_REGISTRY
        if module is None
        else vars(module).setdefault("__warningregistry__", _RETRIES_WARNING_REGISTRY)
    )
    warnings.warn_explicit(
        _RETRIES_EXCEEDED,
        UserWarning,
        filename,
        retry_exception_line if retry_exception else policy_line,
        _RETRIES_MODULE,
        registry,
    )


def _log_grpc_error(file: str, line: int, message: str) -> None:
    """Log an error of a channel that the client goes on from as grpcio does, for the Rust
    transport, which calls this on the thread of the channel: a proxy variable that is not an HTTP
    proxy, or a file of TLS roots that cannot be read. PySpark's client shows them only as the
    line that gRPC's C core writes to the stderr of the process for an error, of the form
    ``E<month><day> <time> <thread> <file>:<line>] <message>``, where ``file`` and ``line`` are of
    the Rust source of the call here."""
    import datetime
    import threading

    now = datetime.datetime.now()
    entry = f"E{now:%m%d %H:%M:%S.%f} {threading.get_native_id():7d} {file}:{line}] {message}\n"
    try:
        os.write(2, entry.encode())
    except OSError:
        # As gRPC's core, which goes on when stderr is closed.
        pass
