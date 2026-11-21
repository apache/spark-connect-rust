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

# The module of PySpark's ``pyspark.sql.connect.client.core``, which defines the client of a
# Spark Connect session, ``SparkConnectClient``, the class of ``SparkSession.client``, and its
# classes, as PySpark's: its channel builders, its RPC deadlines, and the results of its analyses,
# its configurations and its observed metrics. Its package exports the classes of ``__all__``, as
# PySpark's does. Official PySpark code and tests import them here, and a pickle names them here.
#
# The client runs over the Rust transport of its session, which sends the messages that it is
# given, such as the plans of ``to_table`` and ``schema``, as they are: the client builds the
# requests of those as PySpark Connect's does, and reads their responses as it does. The retry
# policies, the reattachable executions, the result chunking and the RPC deadlines of the client are
# those of the transport, which replaces the gRPC stub of PySpark's client.

__all__ = [
    "ChannelBuilder",
    "DefaultChannelBuilder",
    "RpcDeadlines",
    "SparkConnectClient",
]

import contextvars
import copy
import functools
import inspect
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

import pyspark

# As PySpark's core, which imports the messages of Spark Connect, so that protobuf's default pool
# has their files once this module is imported: a pickle of PySpark's PlanObservedMetrics, which
# imports this module before it makes the Expression.Literal messages of its metrics of their full
# names (protobuf's `_InternalConstructMessage`), loads in a new process, as with PySpark.
import pyspark.sql.connect.proto  # noqa: F401
from pyspark import _pyspark
from pyspark.errors import (
    PySparkNotImplementedError,
    PySparkValueError,
)
from pyspark.errors.exceptions.connect import SparkConnectException
from pyspark.sql.connect.logging import logger

# As PySpark's core, which has the metrics classes of the execution info of its results.
from pyspark.sql.metrics import (  # noqa: F401
    ExecutionInfo,
    MetricValue,
    ObservedMetrics,
    PlanMetrics,
)
from pyspark.sql.types import DataType, StructType
from pyspark.util import PythonEvalType
from pyspark.version import __version__

if TYPE_CHECKING:
    import pandas as pd
    import pyarrow as pa

    import pyspark.sql.connect.proto as pb2
    from pyspark.sql.connect.client.retries import RetryPolicy
    from pyspark.sql.connect.observation import Observation
    from pyspark.sql.connect.session import SparkSession
    from pyspark.sql.connect.shell.progress import ProgressHandler
    from pyspark.sql.datasource import DataSource
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


# As PySpark's core: the directory of the pyspark package, whose files `_is_pyspark_source` tells.
PYSPARK_ROOT = os.path.dirname(pyspark.__file__)


@dataclass(frozen=True)
class RpcDeadlines:
    """Per-RPC timeout configuration for :class:`SparkConnectClient`.

    Each field controls the timeout (in seconds as a float) for one gRPC call type. Set a field to
    ``None`` to disable the per-RPC timeout for that call. Use :meth:`RpcDeadlines.disabled` to
    create an instance with all timeouts disabled.

    Note on ``reattachable_execute_plan`` and ``reattach_execute``: these timeouts apply to each
    individual gRPC stream segment, not to the overall query execution lifetime. When a deadline
    fires, the server-side operation continues running; the client opens a new ReattachExecute
    stream to resume receiving results. Non-reattachable ExecutePlan has no deadline because a
    timeout there would kill the execution with no recovery path.
    """

    # As PySpark Connect's. The Rust transport applies the deadline of each of its RPCs,
    # ``get_status`` that of the GetStatus RPC of ``SparkConnectClient._get_operation_statuses``.

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


def _import_zstandard_if_available() -> Optional[Any]:
    """
    Import zstandard if available, otherwise return None.
    This is used to handle the case when zstandard is not installed.
    """
    try:
        import zstandard

        return zstandard
    except ImportError:
        return None


class ChannelBuilder:
    """
    This is a helper class that is used to create a GRPC channel based on the given
    connection string per the documentation of Spark Connect.

    The standard implementation is in :class:`DefaultChannelBuilder`.
    """
    # The parameters of a Spark Connect connection, as PySpark Connect's ``ChannelBuilder``:
    # what ``SparkSession.builder.channelBuilder`` and ``SparkSession(connection=...)`` take. The
    # Rust transport opens the channel of the session from the ``host``, the ``port`` and the
    # ``_params`` of the builder, as a ``sc://`` connection string, so the endpoint and the
    # parameters apply. A custom gRPC channel (``toChannel``), its interceptors and its channel
    # options do not: the transport has no gRPC channel to give them to.

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
        """
        Parameters
        ----------
        key : str
            Parameter key name.
        Returns
        -------
        The parameter value if present, raises exception otherwise.
        """
        # The value of the parameter ``key``; raises ``KeyError`` if it is not set.
        return self._params[key]

    def getDefault(self, key: str, default: Any) -> Any:
        # The value of the parameter ``key``, or ``default`` if it is not set.
        return self._params.get(key, default)

    def set(self, key: str, value: Any) -> None:
        # Set the parameter ``key`` to ``value``.
        self._params[key] = value

    @property
    def host(self) -> str:
        """
        The hostname where this client intends to connect.
        This is used for end-user display purpose in REPL
        """
        # The hostname where this client intends to connect, which a channel builder of its own
        # gives: PySpark's raises without one (``PySparkNotImplementedError``).
        raise PySparkNotImplementedError

    @property
    def userId(self) -> Optional[str]:
        """
        Returns
        -------
        The user_id (extracted from connection string or configured by other means).
        """
        return self._params.get(ChannelBuilder.PARAM_USER_ID, None)

    @property
    def token(self) -> Optional[str]:
        # The authentication token: of the parameters, else of the environment variable
        # ``SPARK_CONNECT_AUTHENTICATE_TOKEN``.
        return self._params.get(
            ChannelBuilder.PARAM_TOKEN, os.environ.get("SPARK_CONNECT_AUTHENTICATE_TOKEN")
        )

    @property
    def keepalive_enabled(self) -> bool:
        """
        Whether the client sends gRPC/HTTP2 keepalive PINGs to detect a silently-dead
        connection. Enabled by default; can be turned off as an escape hatch, e.g. if it
        interacts badly with a particular network path, or a client environment is prone to
        stalls long enough to trip false-positive disconnects.
        """
        # Whether the client sends gRPC/HTTP2 keepalive PINGs to detect a silently-dead
        # connection. Enabled by default.
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
        """
        Returns
        -------
        The session_id extracted from the parameters of the connection string or `None` if not
        specified.
        """
        # The session_id of the parameters, or None if it is not set.
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
        """
        Returns
        -------
        user_agent : str
            The user_agent parameter specified in the connection string,
            or "_SPARK_CONNECT_PYTHON" when not specified.
            The returned value will be percent encoded.
        """
        # The user agent: the ``user_agent`` parameter, else the environment variable
        # ``SPARK_CONNECT_USER_AGENT``, else ``_SPARK_CONNECT_PYTHON``, with the versions of
        # Spark, the operating system and Python. Its URL-escaped form is at most 2048 characters.
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
        """
        Builds the GRPC specific metadata list to be injected into the request. All
        parameters will be converted to metadata except ones that are explicitly used
        by the channel.
        Returns
        -------
        A list of tuples (key, value)
        """
        # The parameters that are not used by the channel, as gRPC metadata of each request.
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
    """
    This is a helper class that is used to create a GRPC channel based on the given
    connection string per the documentation of Spark Connect.

    .. versionadded:: 3.4.0

    Examples
    --------
    >>> cb =  DefaultChannelBuilder("sc://localhost")
    ... cb.endpoint
    "localhost:15002"

    >>> cb = DefaultChannelBuilder("sc://localhost/;use_ssl=true;token=aaa")
    ... cb.secure
    True

    >>> cb = DefaultChannelBuilder("sc://localhost/;grpc_keepalive_time_ms=30000")
    ... cb.keepalive_time_ms
    30000
    """
    # The channel builder of a ``sc://`` connection string, as PySpark Connect's
    # ``DefaultChannelBuilder``.
    #
    # Examples
    # --------
    # >>> cb = DefaultChannelBuilder("sc://localhost")
    # >>> cb.endpoint
    # 'localhost:15002'
    # >>> cb = DefaultChannelBuilder("sc://localhost/;use_ssl=true;token=aaa")
    # >>> cb.secure
    # True

    @staticmethod
    def default_port() -> int:
        # The port of a Spark Connect server, which the drop-in does not discover from a local
        # JVM as PySpark does in the testing mode.
        return 15002

    def __init__(self, url: str, channelOptions: Optional[list[tuple[str, Any]]] = None) -> None:
        """
        Constructs a new channel builder. This is used to create the proper GRPC channel from
        the connection string.

        Parameters
        ----------
        url : str
            Spark Connect connection string
        channelOptions: list of tuple, optional
            Additional options that can be passed to the GRPC channel construction.
        """
        # Construct a channel builder of the connection string ``url``, such as
        # ``sc://hostname:port/;key=value``.
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
        self._port = (
            self.url.port if self.url.port is not None else DefaultChannelBuilder.default_port()
        )

    @property
    def secure(self) -> bool:
        # Whether the connection is secured: by TLS, or by a token.
        return self.use_ssl or self.token is not None

    @property
    def use_ssl(self) -> bool:
        return self.getDefault(ChannelBuilder.PARAM_USE_SSL, "").lower() == "true"

    @property
    def host(self) -> str:
        """
        The hostname where this client intends to connect.
        """
        # The hostname where this client intends to connect.
        return self._host

    @property
    def port(self) -> int:
        """The port where this client intends to connect."""
        return self._port

    @property
    def endpoint(self) -> str:
        # The ``host:port`` of the connection.
        return f"{self._host}:{self._port}"


class PlanObservedMetrics(ObservedMetrics):
    # Observed metrics of one ``observe(...)`` node, as carried by
    # ``DataFrame.executionInfo.flows``, as PySpark Connect's: its ``metrics`` are the
    # ``Expression.Literal`` messages (of ``pyspark.sql.connect.proto``) that the server sent,
    # which ``pairs`` converts with ``LiteralExpression._to_value``, and so raises PySpark's
    # ``UNSUPPORTED_LITERAL`` of a literal that it does not convert.

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
        """Return a JSON-serializable dictionary representation of this observed metrics.

        Returns
        -------
        dict
            A dictionary with keys 'name', 'keys', and 'pairs'.
        """
        return {
            "name": self._name,
            "keys": self._keys,
            "pairs": self.pairs,
        }


class AnalyzeResult:
    # The result of an analysis of SparkConnectClient's ``_analyze``, as PySpark Connect's: the
    # one field of the result of the analysis, and None for the others.

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
    # The result of SparkConnectClient's ``config``, as PySpark Connect's: the pairs of a key and
    # its value, None for a key without one, and the warnings of the server.

    def __init__(self, pairs: list[tuple[str, Optional[str]]], warnings: list[str]):
        self.pairs = pairs
        self.warnings = warnings

    @classmethod
    def fromProto(cls, pb: "pb2.ConfigResponse") -> "ConfigResult":
        return ConfigResult(
            pairs=[(pair.key, pair.value if pair.HasField("value") else None) for pair in pb.pairs],
            warnings=list(pb.warnings),
        )


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


def _is_pyspark_source(filename: str) -> bool:
    """Check if the given filename is from the pyspark package."""
    return filename.startswith(PYSPARK_ROOT)


class SparkConnectClient:
    """
    Conceptually the remote spark session that communicates with the server
    """
    # The client of a Spark Connect session, as PySpark Connect's ``SparkConnectClient``: the
    # client of ``SparkSession.client``, and one that ``SparkConnectClient(connection, ...)`` makes
    # of a session of its own. It sends the messages that it is given, of
    # ``pyspark.sql.connect.proto``, over the Rust transport of its session, as the modules of
    # PySpark that build their own messages, such as ``pyspark.ml``, send them: commands, plans and
    # their analyses, configurations, and the artifacts, the tags, the interrupts and the progress
    # handlers of the session.

    # The session of the client is kept apart from its attributes, which are those of its session:
    # the clients of a session share them (see `_client_state`), as PySpark's session has one
    # client, of one `thread_local`, one list of global user context extensions, its lock, the
    # session ID and the user of its requests, and the attributes that a caller sets. The client
    # holds its session weakly, as PySpark's session object holds its client and not the other way
    # (see `_session`), and the state of the session (`_context`); a copy of a client sends its
    # requests of its own attributes (`_copy_context`).
    __slots__ = (
        "_session_ref",
        "_session_cache",
        "_url",
        "_dropped",
        "_context",
        "_copy_context",
        "__dict__",
        "__weakref__",
    )

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
        session_hooks: Optional[list["SparkSession.Hook"]] = None,
        allow_arrow_batch_chunking: bool = True,
        preferred_arrow_chunk_size: Optional[int] = None,
        rpc_deadlines: Optional[RpcDeadlines] = None,
        max_retry_exception_elapsed_time: Optional[float] = None,
    ):
        """
        Creates a new SparkSession for the Spark Connect interface.

        Parameters
        ----------
        connection : str or :class:`ChannelBuilder`
            Connection string that is used to extract the connection parameters and configure
            the GRPC connection. Or instance of ChannelBuilder that creates GRPC connection.
            Defaults to `sc://localhost`.
        user_id : str, optional
            Optional unique user ID that is used to differentiate multiple users and
            isolate their Spark Sessions. If the `user_id` is not set, will default to
            the $USER environment. Defining the user ID as part of the connection string
            takes precedence.
        channel_options: list of tuple, optional
            Additional options that can be passed to the GRPC channel construction.
        retry_policy: dict of str and any, optional
            Additional configuration for retrying. There are four configurations as below
                * ``max_retries``
                    Maximum number of tries default 15
                * ``backoff_multiplier``
                    Backoff multiplier for the policy. Default: 4(ms)
                * ``initial_backoff``
                    Backoff to wait before the first retry. Default: 50(ms)
                * ``max_backoff``
                    Maximum backoff controls the maximum amount of time to wait before retrying
                    a failed request. Default: 60000(ms).
        use_reattachable_execute: bool
            Enable reattachable execution.
        session_hooks: list[SparkSession.Hook], optional
            List of session hooks to call.
        allow_arrow_batch_chunking: bool
            Whether to allow the server to split large Arrow batches into smaller chunks.
            Although Arrow results are split into batches with a size limit according to estimation,
            the size of the batches is not guaranteed to be less than the limit, especially when a
            single row is larger than the limit, in which case the server will fail to split it
            further into smaller batches. As a result, the client may encounter a gRPC error stating
            "Received message larger than max" when a batch is too large.
            If true, the server will split large Arrow batches into smaller chunks, and the client
            is expected to handle the chunked Arrow batches.
            If false, the server will not chunk large Arrow batches.
        preferred_arrow_chunk_size: Optional[int]
            Optional preferred Arrow batch size in bytes for the server to use when sending Arrow
            results.
            The server will attempt to use this size if it is set and within the valid range
            ([1KB, max batch size on server]). Otherwise, the server's maximum batch size is used.
        rpc_deadlines : RpcDeadlines, optional
            Per-RPC gRPC call timeouts in seconds (10 min for most RPCs,
            1 hour for analyze/addArtifacts, none for non-reattachable execute).
            Use :meth:`RpcDeadlines.disabled` to turn off all deadlines.
        max_retry_exception_elapsed_time : float, optional
            Maximum cumulative elapsed time in seconds the client will keep retrying a
            RetryException (raised internally when a reattach attempt keeps hitting
            DEADLINE_EXCEEDED, or when the initial ExecutePlan never reached the server) before
            giving up and raising the underlying error. Defaults to
            :data:`~pyspark.sql.connect.client.retries.DEFAULT_MAX_RETRY_EXCEPTION_ELAPSED_TIME`
            (1 hour).
        """
        # A client of a new session of the connection string or the channel builder
        # ``connection``, as PySpark Connect's ``SparkConnectClient``: of the user ``user_id``
        # unless the connection names one, else of ``SPARK_USER`` or ``USER``; of the retries of a
        # ``DefaultPolicy`` of the keyword arguments ``retry_policy``, and of a
        # ``RetryException`` for up to ``max_retry_exception_elapsed_time`` seconds; of reattachable
        # executions unless ``use_reattachable_execute`` is False; whose executions call the
        # ``on_execute_plan`` of ``session_hooks``; of the result chunking of
        # ``allow_arrow_batch_chunking`` and ``preferred_arrow_chunk_size``; and of the deadlines
        # of ``rpc_deadlines``. The session is neither the default session nor the active one.
        #
        # The Rust transport opens the channel of the connection, so a ``channel_options`` of a
        # connection string raises ``NOT_IMPLEMENTED``, as its ``DefaultChannelBuilder`` does; as
        # PySpark's, the client of a channel builder does not read it.
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
        # As PySpark's client, which logs the deadlines that it was given as it is made.
        _log_rpc_deadlines(rpc_deadlines if rpc_deadlines is not None else RpcDeadlines())
        _init_client_state(session, builder, user_id)
        self._init_of_session(session)
        if not use_reattachable_execute:
            self.disable_reattachable_execute()
        if policies is not None or max_retry_exception_elapsed_time is not None:
            self._set_retry_policies(policies, max_retry_exception_elapsed_time)
        # As PySpark's client, which registers its exit hook once it is made: the registration
        # keeps it until the exit, and no session object owns it, whose __del__ would close it.
        self._keep_until_exit(session)

    def _keep_until_exit(self, session: Any) -> None:
        # The session of a client that no session object owns, which lives until the exit, as
        # PySpark's `atexit` registration keeps its client: the client holds its session object.
        session._keep_until_exit()
        self._session_cache = lambda: session

    @classmethod
    def _of_session(cls, session: Any) -> "SparkConnectClient":
        """The client of ``session``, a session of the drop-in, as ``SparkSession.client`` and
        the clients of ``newSession`` and ``clone`` are."""
        client = cls.__new__(cls)
        client._init_of_session(session)
        return client

    def _init_of_session(self, session: Any) -> None:
        self._session_ref = session._downgrade()
        self._session_cache = weakref.ref(session)
        self._url = session._connection_string()
        self._dropped = None
        self._copy_context = None
        # The attributes of the client are those of its session, which its other clients have
        # too: its channel builder, its thread_local, its global user context extensions and their
        # lock, the session ID and the user of its requests, and the attributes that a caller set.
        self._context = _client_context(session)
        object.__setattr__(self, "__dict__", self._context.state)

    @property
    def _session(self) -> Any:
        """The session object of the client, while the session lives: while the program holds it
        or an object that holds it, such as a DataFrame, a reader, a writer, its catalog or its
        streams, as PySpark's objects hold the session object. Once the program dropped them, as
        PySpark's ``SparkSession.__del__`` closes its client then, it is a closed session of the
        same connection, whose requests raise ``NO_ACTIVE_SESSION``, and the ``ValueError`` of a
        closed channel once ``_closed`` is set to False."""
        # The session object of the last access, while it lives, else the object of the session.
        session = self._session_cache()
        if session is not None:
            return session
        session = self._session_ref._session()
        if session is not None:
            self._session_cache = weakref.ref(session)
            return session
        dropped = self._dropped
        if dropped is None:
            dropped = _pyspark.SparkSession._of_client_options(
                self._url, None, [], True, None, None
            )
            dropped._close()
            self._dropped = dropped
        return dropped

    def __copy__(self) -> "SparkConnectClient":
        # As copy.copy of PySpark's client, a plain object: a client of the same session and of a
        # copy of its attributes, whose requests carry its own session ID, user and lists of
        # user context extensions, and whose rebinding of one of them is its own.
        copied = object.__new__(type(self))
        copied._session_ref = self._session_ref
        copied._session_cache = self._session_cache
        copied._url = self._url
        copied._dropped = self._dropped
        copied._context = self._context
        copied._copy_context = self._context
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
        """
        The hostname where this client intends to connect.
        """
        # The hostname where this client intends to connect, as SparkConnectClient's ``host``:
        # of its channel builder.
        if self._builder is not None:
            return self._builder.host
        return self._session._endpoint_host()

    @property
    def token(self) -> Optional[str]:
        """
        The authentication bearer token during connection.
        If authentication is not using a bearer token, None will be returned.
        """
        # The authentication bearer token during connection, as SparkConnectClient's ``token``:
        # that of its channel builder, the ``token`` parameter of the connection string, else the
        # environment variable ``SPARK_CONNECT_AUTHENTICATE_TOKEN``, or None.
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
        closes the session, which the session registry then no longer returns, and its requests
        raise ``NO_ACTIVE_SESSION`` but those of its artifacts; setting it to False opens it
        again, as PySpark's requests are sent again then, or fail with the ``ValueError`` of a
        closed channel once ``close()`` or ``stop()`` closed it."""
        return self._session.is_stopped

    @_closed.setter
    def _closed(self, value: bool) -> None:
        self._session._set_closed(bool(value))

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
        data = self._session._get_status(req.SerializeToString())
        return _parse_response(pb2.GetStatusResponse, data)

    @property
    def is_closed(self) -> bool:
        """
        Returns if the channel was closed previously using close() method
        """
        # Whether the session of the client is stopped, as SparkConnectClient's ``is_closed``,
        # whether its channel is closed.
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
        # The values of the configurations ``keys``, in one request, as SparkConnectClient's
        # ``get_configs``.
        return tuple(self._session.conf._get_many(list(keys)))

    def get_config_dict(self, *keys: str) -> Mapping[str, Optional[str]]:
        # The values of the configurations ``keys`` by their keys, in one request, as
        # SparkConnectClient's ``get_config_dict``.
        return dict(zip(keys, self.get_configs(*keys)))

    def get_config_with_defaults(
        self, *pairs: tuple[str, Optional[str]]
    ) -> tuple[Optional[str], ...]:
        # The values of the configurations of ``pairs`` of a key and its default, the default of
        # a key that is not set, in one request, as SparkConnectClient's
        # ``get_config_with_defaults``, of which pandas-on-Spark reads its ANSI mode.
        return tuple(self._session.conf._get_with_defaults(list(pairs)))

    def config(self, operation: "pb2.ConfigRequest.Operation") -> ConfigResult:
        """
        Call the config RPC of Spark Connect.

        Parameters
        ----------
        operation : str
           Operation kind

        Returns
        -------
        The result of the config call.
        """
        # Send ``operation``, a ``ConfigRequest.Operation``, as SparkConnectClient's ``config``:
        # the result of the configuration request of the session.
        import pyspark.sql.connect.proto as pb2

        req = pb2.ConfigRequest()
        req.operation.CopyFrom(operation)
        data = self._session._config(req.SerializeToString())
        return ConfigResult.fromProto(_parse_response(pb2.ConfigResponse, data))

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
        return AnalyzeResult.fromProto(_parse_response(pb2.AnalyzePlanResponse, data))

    def schema(self, plan: "pb2.Plan") -> StructType:
        """
        Return schema for given plan.
        """
        # The schema of ``plan``, a ``spark.connect.Plan``, as SparkConnectClient's ``schema``:
        # of its analysis by the server, of the plan as it is given, which it logs first.
        if logger.isEnabledFor(logging.DEBUG):
            self._log_plan("Schema for plan: ", plan)
        schema = self._analyze(method="schema", plan=plan).schema
        assert schema is not None
        # Server side should populate the struct field which is the schema.
        assert isinstance(schema, StructType)
        return schema

    def explain_string(self, plan: "pb2.Plan", explain_mode: str = "extended") -> str:
        """
        Return explain string for given plan.
        """
        # The explain string of ``plan`` in ``explain_mode``, as SparkConnectClient's
        # ``explain_string``: ``VALUE_NOT_ALLOWED`` for a mode other than ``simple``,
        # ``extended``, ``codegen``, ``cost`` and ``formatted``, once it logged the plan, of the
        # mode as it is given.
        if logger.isEnabledFor(logging.DEBUG):
            self._log_plan(f"Explain (mode={explain_mode}) for plan ", plan)
        result = self._analyze(
            method="explain", plan=plan, explain_mode=explain_mode
        ).explain_string
        assert result is not None
        return result

    def _log_plan(self, prefix: str, plan: "pb2.Plan") -> None:
        # Log ``prefix`` and the text of ``plan``, a ``spark.connect.Plan`` that the client is
        # given, truncated, as PySpark's client logs it, of the relations of its placeholders, as
        # the transport sends them: of a caller that checked that the logger writes it.
        self._session._log_plan(prefix, _plan_of(plan).SerializeToString())

    def _set_relation_in_plan(self, plan: "pb2.Plan", relation: "pb2.Relation") -> None:
        """Sets the relation in the plan, attempting compression if configured."""
        import pyspark.sql.connect.proto as pb2

        self._try_compress_and_set_plan(
            plan=plan,
            message=relation,
            op_type=pb2.Plan.CompressedOperation.OpType.OP_TYPE_RELATION,
        )

    def _set_command_in_plan(self, plan: "pb2.Plan", command: "pb2.Command") -> None:
        """Sets the command in the plan, attempting compression if configured."""
        import pyspark.sql.connect.proto as pb2

        self._try_compress_and_set_plan(
            plan=plan,
            message=command,
            op_type=pb2.Plan.CompressedOperation.OpType.OP_TYPE_COMMAND,
        )

    def _try_compress_and_set_plan(self, plan: "pb2.Plan", message: Any, op_type: int) -> None:
        """
        Tries to compress a protobuf message and sets it on the plan.
        If compression is not enabled, not effective, or not available,
        it falls back to the original message.
        """
        # As SparkConnectClient's, of the plan compression of the transport: its settings, which
        # the first plan of the session looks up, and the compression of the message, of the
        # relations of its placeholders, with what PySpark's logs of them. A message that it does
        # not compress is set as it is, of its placeholders, which the transport sends.
        import pyspark.sql.connect.proto as pb2

        relation = op_type == pb2.Plan.CompressedOperation.OpType.OP_TYPE_RELATION
        data = message.SerializeToString()
        # Of the settings that the session read, a message of no more bytes than the threshold,
        # or of the compression disabled, is not compressed, and the transport is not called: the
        # relation that a placeholder holds is of fewer bytes than the placeholder, but that of a
        # placeholder of its DataFrame alone, which only the transport knows.
        threshold = self._session._plan_compression_threshold()
        if threshold is not None and (
            threshold < 0 or (len(data) <= threshold and _MARKER_PLACEHOLDER not in data)
        ):
            compressed = None
        else:
            compressed = self._session._compressed_plan(_plan_of_bytes(1 if relation else 2, data))
        if compressed is not None:
            plan.compressed_operation.CopyFrom(pb2.Plan.FromString(compressed).compressed_operation)
        elif relation:
            plan.root.CopyFrom(message)
        else:
            plan.command.CopyFrom(message)

    def same_semantics(self, plan: "pb2.Plan", other: "pb2.Plan") -> bool:
        """
        return if two plans have the same semantics.
        """
        # Whether ``plan`` and ``other`` have the same semantics, as SparkConnectClient's
        # ``same_semantics``.
        result = self._analyze(method="same_semantics", plan=plan, other=other).is_same_semantics
        assert result is not None
        return result

    def semantic_hash(self, plan: "pb2.Plan") -> int:
        """
        returns a `hashCode` of the logical query plan.
        """
        # The ``hashCode`` of the logical plan of ``plan``, as SparkConnectClient's
        # ``semantic_hash``.
        result = self._analyze(method="semantic_hash", plan=plan).semantic_hash
        assert result is not None
        return result

    def _execute_plan(
        self, plan: "pb2.Plan", observations: Any, reported: bool, log: Optional[str] = None
    ) -> Any:
        """Start the execution of ``plan``, a ``spark.connect.Plan`` that the client sends as it
        is, as SparkConnectClient's request of a copy of it: its responses, whose observed metrics
        set the Observations of ``observations``, a dict, by their names, and whose progress
        reaches the progress handlers when ``reported``. Of an ``observations`` that is not a
        dict, such as None or an empty list, none is set: PySpark's client reads it only when
        observed metrics arrive (see ``_results``). The transport logs what the method of PySpark's
        client of the name ``log`` logs of the execution, of its plan as it is sent."""
        items = list(observations.items()) if isinstance(observations, Mapping) else []
        return self._session._execute_plan(plan.SerializeToString(), items, reported, log)

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
                b = _parse_response(pb2.ExecutePlanResponse, data, streaming=True)
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
        self,
        plan: "pb2.Plan",
        observations: Any,
        self_destruct: bool = False,
        log: Optional[str] = None,
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

        responses = self._execute_plan(plan, observations, True, log)
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
        """
        Return given plan as a PyArrow Table iterator.
        """
        # The result of ``plan`` as it arrives, as SparkConnectClient's
        # ``to_table_as_iterator``: its schema, once the server sends it, and a PyArrow Table of
        # each record batch. The execution sets the Observations of ``observations`` by their
        # names, and starts with the first item.
        import pyarrow as pa

        # The transport logs the plan as it sends it, as it is (`Executing plan ...`).
        sent = _plan_of(plan)
        responses = self._execute_plan(sent, observations, True, "to_table_as_iterator")
        for response in self._results(responses, observations):
            if isinstance(response, StructType):
                yield response
            elif isinstance(response, pa.RecordBatch):
                yield pa.Table.from_batches([response])

    def to_table(
        self, plan: "pb2.Plan", observations: dict[str, "Observation"]
    ) -> tuple["pa.Table", Optional[StructType], ExecutionInfo]:
        """
        Return given plan as a PyArrow Table.
        """
        # The result of ``plan`` as a PyArrow Table, as SparkConnectClient's ``to_table``: the
        # table of the record batches of the server, the schema that it sent, or None, and the
        # ``ExecutionInfo`` of the execution, which sets the Observations of ``observations``.
        # The transport logs the plan as it sends it, as it is (`Executing plan ...`).
        sent = _plan_of(plan)
        table, schema, metrics, observed_metrics, _, operation_id = self._execute_and_fetch(
            sent, observations, log="to_table"
        )
        # Create a query execution object.
        ei = ExecutionInfo(metrics, observed_metrics, operation_id)
        assert table is not None
        return table, schema, ei

    def to_pandas(
        self, plan: "pb2.Plan", observations: dict[str, "Observation"], **kwargs: Any
    ) -> tuple["pd.DataFrame", ExecutionInfo]:
        """
        Return given plan as a pandas DataFrame.
        """
        # The result of ``plan`` as a pandas DataFrame, as SparkConnectClient's ``to_pandas``:
        # each column converted by its Spark type, in the session time zone, of the
        # ``spark.sql.execution.pandas.structHandlingMode``, or of a ``pandasStructHandlingMode``
        # keyword argument, and the ``ExecutionInfo`` of the execution.
        # The plan, sent as it is, which is logged before the settings are read.
        sent = _plan_of(plan)
        if logger.isEnabledFor(logging.DEBUG):
            self._log_plan("Executing plan ", sent)

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
            sent, observations, selfDestruct == "true", log="fetch"
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
        """
        Execute given command.
        """
        # Execute ``command``, a ``spark.connect.Command``, as SparkConnectClient's
        # ``execute_command``: the pandas DataFrame of the record batches of the responses, or
        # None without any, the properties of the results of the responses, of their field names,
        # as SparkConnectClient converts them (the resources of a GetResourcesCommand, the
        # profile ID of a resource profile, and the DataFrame of a checkpoint), and the
        # ``ExecutionInfo`` of the execution, which sets the Observations of ``observations``.
        # The transport logs the command as it builds its request (`Execute command for
        # command ...`).
        data, _, metrics, observed_metrics, properties, operation_id = self._execute_and_fetch(
            _plan_of_command(command), observations or {}, log="execute_command"
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
        """
        Execute given command. Similar to execute_command, but the value is returned using yield.
        """
        # Execute ``command``, as ``execute_command``, and yield the properties of the result of
        # each response as it arrives, as SparkConnectClient's ``execute_command_as_iterator``:
        # ``UNKNOWN_RESPONSE`` for a result that is not one, such as metrics or a record batch. Its
        # progress does not reach the progress handlers, as PySpark's does not.
        # The transport logs the command as it builds its request (`Execute command as iterator
        # for command ...`).
        plan = _plan_of_command(command)
        observations = observations or {}
        responses = self._execute_plan(plan, observations, False, "execute_command_as_iterator")
        for response in self._results(responses, observations):
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
        # Add ``extension``, a ``google.protobuf.Any``, to the user context of the requests that
        # the current thread sends, as SparkConnectClient's: to the list of
        # ``thread_local.user_context_extensions``, of its ID.
        if not hasattr(self.thread_local, "user_context_extensions"):
            self.thread_local.user_context_extensions = list()
        extension_id = "threadlocal_" + str(uuid.uuid4())
        self.thread_local.user_context_extensions.append((extension_id, extension))
        return extension_id

    def add_global_user_context_extension(self, extension: Any) -> str:
        # Add ``extension``, a ``google.protobuf.Any``, to the user context of the requests of
        # all the threads, as SparkConnectClient's: its ID, of the extension that its
        # ``global_user_context_extensions`` list holds.
        extension_id = "global_" + str(uuid.uuid4())
        with self.global_user_context_extensions_lock:
            self.global_user_context_extensions.append((extension_id, extension))
        return extension_id

    def remove_user_context_extension(self, extension_id: str) -> None:
        # Remove the user context extension of ``extension_id``, as SparkConnectClient's: of the
        # current thread, which rebinds ``thread_local.user_context_extensions`` to a list without
        # it, or a global one, which rebinds ``global_user_context_extensions``.
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
        # Remove the user context extensions of the current thread and the global ones, as
        # SparkConnectClient's: each is a new, empty list, that of the thread if it has one.
        if hasattr(self.thread_local, "user_context_extensions"):
            self.thread_local.user_context_extensions = list()
        with self.global_user_context_extensions_lock:
            self.global_user_context_extensions = list()

    def add_tag(self, tag: str) -> None:
        # Add ``tag`` to the tags of the current thread, which the executions that it starts
        # carry, as SparkConnectClient's: to the set of ``thread_local.tags``, of the errors of
        # ``_throw_if_invalid_tag``.
        self._throw_if_invalid_tag(tag)
        if not hasattr(self.thread_local, "tags"):
            self.thread_local.tags = set()
        self.thread_local.tags.add(tag)

    def remove_tag(self, tag: str) -> None:
        # Remove ``tag`` from the tags of the current thread, as SparkConnectClient's: of the
        # errors of ``add_tag``, and nothing for a tag that the thread does not have.
        self._throw_if_invalid_tag(tag)
        if not hasattr(self.thread_local, "tags"):
            self.thread_local.tags = set()
        # Use discard, not remove: removing an absent tag is a documented no-op
        # (see SparkSession.removeTag), matching the Classic behavior.
        self.thread_local.tags.discard(tag)

    def get_tags(self) -> set[str]:
        # The tags of the current thread, as SparkConnectClient's: the set of
        # ``thread_local.tags``, which the later changes of the tags change too.
        if not hasattr(self.thread_local, "tags"):
            self.thread_local.tags = set()
        return self.thread_local.tags

    def clear_tags(self) -> None:
        # Remove the tags of the current thread, as SparkConnectClient's: ``thread_local.tags``
        # is a new, empty set.
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
        # Interrupt the executions of the session, as SparkConnectClient's: the IDs of the
        # operations interrupted.
        return self._session.interruptAll()

    def interrupt_tag(self, tag: str) -> Optional[list[str]]:
        # Interrupt the executions of the session of the tag ``tag``, as SparkConnectClient's:
        # the IDs of the operations interrupted.
        # As PySpark's request of an interrupt of a tag, which asserts that there is one.
        assert tag is not None
        return self._session.interruptTag(tag)

    def interrupt_operation(self, op_id: str) -> Optional[list[str]]:
        # Interrupt the execution of the operation ``op_id``, as SparkConnectClient's: the IDs
        # of the operations interrupted.
        # As PySpark's request of an interrupt of an operation, which asserts that there is one.
        assert op_id is not None
        return self._session.interruptOperation(op_id)

    @classmethod
    def _retrieve_stack_frames(cls) -> list[Any]:
        """
        Return a list of CallSites representing the relevant stack frames in the callstack.
        """
        # As SparkConnectClient's: from the outermost frame to the first frame of the pyspark
        # package, each of the function of the frame after it.
        import traceback

        from pyspark.traceback_utils import CallSite

        frames = traceback.extract_stack()

        filtered_stack_frames = []
        for i, frame in enumerate(frames):
            filename, lineno, func, _ = frame
            if _is_pyspark_source(filename):
                # Do not include PySpark internal frames as they are not user application code
                break
            if i + 1 < len(frames):
                _, _, func, _ = frames[i + 1]
            filtered_stack_frames.append(CallSite(function=func, file=filename, linenum=lineno))

        return filtered_stack_frames

    @classmethod
    def _build_call_stack_trace(cls) -> Optional[Any]:
        """
        Build a call stack trace for the current Spark Connect action
        Returns
        -------
        FetchErrorDetailsResponse.Error: An Error object containing list of stack frames
        of the user code packed as Any protobuf
        """
        # As SparkConnectClient's, when SPARK_CONNECT_DEBUG_CLIENT_CALL_STACK is set, which its
        # requests carry. The requests of the Rust transport carry the call stack of their call,
        # which it reads as the call starts (see `_call_stack_of_call`).
        import pyspark.sql.connect.proto as pb2

        if os.getenv("SPARK_CONNECT_DEBUG_CLIENT_CALL_STACK", "false").lower() in ("true", "1"):
            stack_frames = cls._retrieve_stack_frames()
            call_stack = pb2.FetchErrorDetailsResponse.Error()
            for call_site in stack_frames:
                stack_trace_element = pb2.FetchErrorDetailsResponse.StackTraceElement()
                stack_trace_element.declaring_class = ""  # unknown information
                stack_trace_element.method_name = call_site.function
                stack_trace_element.file_name = call_site.file
                stack_trace_element.line_number = call_site.linenum
                call_stack.stack_trace.append(stack_trace_element)
            if len(call_stack.stack_trace) > 0:
                from pyspark.sql.connect.proto._rust import Any as AnyMessage

                call_stack_details = AnyMessage()
                call_stack_details.Pack(call_stack)
                return call_stack_details
        return None

    def _update_request_with_user_context_extensions(self, req: Any) -> None:
        # As SparkConnectClient's: the global user context extensions of the client, and those of
        # the current thread, appended to the user context of ``req``.
        with self.global_user_context_extensions_lock:
            for _, extension in self.global_user_context_extensions:
                req.user_context.extensions.append(extension)
        if not hasattr(self.thread_local, "user_context_extensions"):
            return
        for _, extension in self.thread_local.user_context_extensions:
            req.user_context.extensions.append(extension)

    def _execute_plan_request_with_metadata(
        self, operation_id: Optional[str] = None
    ) -> "pb2.ExecutePlanRequest":
        # As SparkConnectClient's: an ExecutePlanRequest of the session ID, the user agent and the
        # tags of the client, the server-side session ID that it observed, its user, its result
        # chunking options, and ``operation_id``, else a new one (``INVALID_OPERATION_UUID_ID`` of
        # one that is not a UUID), of its user context extensions and then of the call stack of
        # the user's code. The transport builds the requests that it sends of its own.
        import pyspark.sql.connect.proto as pb2

        req = pb2.ExecutePlanRequest(
            session_id=self._session_id,
            client_type=self._builder.userAgent,
            tags=list(self.get_tags()),
        )
        if self._server_session_id is not None:
            req.client_observed_server_side_session_id = self._server_session_id
        if self._user_id:
            req.user_context.user_id = self._user_id
        # Add request option to allow result chunking.
        allow_arrow_batch_chunking, preferred_arrow_chunk_size = self._session._result_chunking()
        chunking = pb2.ResultChunkingOptions(allow_arrow_batch_chunking=allow_arrow_batch_chunking)
        if preferred_arrow_chunk_size is not None:
            chunking.preferred_arrow_chunk_size = preferred_arrow_chunk_size
        req.request_options.append(
            pb2.ExecutePlanRequest.RequestOption(result_chunking_options=chunking)
        )
        if operation_id is None:
            operation_id = str(uuid.uuid4())
        else:
            try:
                uuid.UUID(operation_id, version=4)
            except ValueError as ve:
                # Of the ValueError as its context, as PySpark's raises it.
                raise PySparkValueError(  # noqa: B904
                    errorClass="INVALID_OPERATION_UUID_ID",
                    messageParameters={"arg_name": "operation_id", "origin": str(ve)},
                )
        req.operation_id = operation_id
        self._update_request_with_user_context_extensions(req)

        if call_stack_trace := self.__class__._build_call_stack_trace():
            req.user_context.extensions.append(call_stack_trace)
        return req

    def _analyze_plan_request_with_metadata(self) -> "pb2.AnalyzePlanRequest":
        # As SparkConnectClient's: an AnalyzePlanRequest of the session ID of the client, the
        # server-side session ID that it observed, its user agent and its user, of its user context
        # extensions and then of the call stack of the user's code.
        import pyspark.sql.connect.proto as pb2

        req = pb2.AnalyzePlanRequest()
        req.session_id = self._session_id
        if self._server_session_id is not None:
            req.client_observed_server_side_session_id = self._server_session_id
        req.client_type = self._builder.userAgent
        if self._user_id:
            req.user_context.user_id = self._user_id
        self._update_request_with_user_context_extensions(req)
        if call_stack_trace := self.__class__._build_call_stack_trace():
            req.user_context.extensions.append(call_stack_trace)
        return req

    def _config_request_with_metadata(self) -> "pb2.ConfigRequest":
        # As SparkConnectClient's: a ConfigRequest of the session ID of the client, the
        # server-side session ID that it observed, its user agent and its user, of its user context
        # extensions and then of the call stack of the user's code.
        import pyspark.sql.connect.proto as pb2

        req = pb2.ConfigRequest()
        req.session_id = self._session_id
        if self._server_session_id is not None:
            req.client_observed_server_side_session_id = self._server_session_id
        req.client_type = self._builder.userAgent
        if self._user_id:
            req.user_context.user_id = self._user_id
        self._update_request_with_user_context_extensions(req)
        if call_stack_trace := self.__class__._build_call_stack_trace():
            req.user_context.extensions.append(call_stack_trace)
        return req

    def register_udf(
        self,
        function: Any,
        return_type: "Union[DataType, str]",
        name: Optional[str] = None,
        eval_type: int = PythonEvalType.SQL_BATCHED_UDF,
        deterministic: bool = True,
    ) -> str:
        """
        Create a temporary UDF in the session catalog on the other side. We generate a
        temporary name for it.
        """
        # Register the Python function ``function`` as a temporary function of the session of
        # the name ``name``, else of a name that the client generates, as SparkConnectClient's
        # ``register_udf``: the name. A DDL ``return_type`` is parsed by the server, and the
        # function is pickled with the type, as PySpark's ``PythonUDF`` does. As PySpark's, the
        # name and ``deterministic`` are set in the fields of the command first, then the return
        # type is parsed and converted, then ``eval_type`` is set, and the function is pickled
        # last, so each raises the error of its protobuf field or of its conversion in that
        # order.
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
        """
        Register a user-defined table function (UDTF) in the session catalog
        as a temporary function. The return type, if specified, must be a
        struct type and it's validated when building the proto message
        for the PythonUDTF.
        """
        # Register the user-defined table function class ``function`` as the temporary
        # function ``name`` of the session, as SparkConnectClient's ``register_udtf``: the name. A
        # DDL ``return_type`` is left to the server to parse. As PySpark's, the ``__name__`` of the
        # handler class is read, the name and ``deterministic`` are set in the fields of the
        # command, the return type is converted and ``eval_type`` set before the handler is
        # pickled, so a handler that does not pickle raises ``UDTF_SERIALIZATION_ERROR`` only of
        # valid arguments.
        import pyspark.sql.connect.proto as pb2
        from pyspark.sql.connect.udtf import _pickled_handler
        from pyspark.sql.connect.utils import get_python_ver

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
        """
        Register a data source in the session catalog.
        """
        # Register the Python data source class ``dataSource`` with the session, of its
        # ``name()``, as SparkConnectClient's ``register_data_source``.
        self._session.dataSource.register(dataSource)

    def register_java(
        self,
        name: str,
        javaClassName: str,
        return_type: "Optional[Union[DataType, str]]" = None,
        aggregate: bool = False,
    ) -> None:
        # Register the Java function of the class ``javaClassName`` as the temporary function
        # ``name`` of the session, as SparkConnectClient's ``register_java``: an aggregate one for
        # ``aggregate``, of the class's own type, and of a ``return_type``, a DataType or a DDL
        # string that the server parses, a function that is not an aggregate one, as PySpark's
        # ``JavaUDF`` of an output type is. As PySpark's, the name, the class name, the return
        # type and then ``aggregate``, unless there is a return type, are set in the fields of the
        # command in that order, so each raises the error of its protobuf field in that order.
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
        # Add the local files, or ``file:`` URIs, ``paths`` as artifacts of the session, as
        # SparkConnectClient's ``add_artifacts``: each named by the first of ``pyfile``,
        # ``archive`` and ``file`` that it matches, which the client, unlike
        # ``SparkSession.addArtifacts``, does not check against each other, else a jar.
        self._session._add_artifacts_of_flags(*paths, pyfile=pyfile, archive=archive, file=file)

    def copy_from_local_to_fs(self, local_path: str, dest_path: str) -> None:
        # Copy the local file ``local_path`` to the absolute path ``dest_path`` of the file
        # system of the server, as SparkConnectClient's ``copy_from_local_to_fs``: of the
        # assertions and the errors of its ArtifactManager, without the check of a scheme of
        # ``SparkSession.copyFromLocalToFs``.
        abs_path = Path(local_path).absolute()
        assert abs_path.is_file(), "local path must be a file path."
        assert Path(dest_path).is_absolute(), "destination FS path must be an absolute path."
        # As the name of the artifact of PySpark's ArtifactManager, which raises the TypeError of
        # the concatenation for a destination that is not a str.
        _ = "forward_to_fs" + dest_path
        self._session._copy_from_local_to_fs(str(abs_path), dest_path)

    def cache_artifact(self, blob: bytes) -> str:
        # Cache ``blob``, a bytes-like object, as an artifact of the session, as
        # SparkConnectClient's ``cache_artifact``: its SHA-256 hash, of which it is uploaded unless
        # the server has it.
        return self.cache_artifacts([blob])[0]

    def cache_artifacts(self, blobs: list[bytes]) -> list[str]:
        # Cache ``blobs``, bytes-like objects, as artifacts of the session, as
        # SparkConnectClient's ``cache_artifacts``: their SHA-256 hashes, in order, of which those
        # that the server does not have are uploaded, once each, in one request. As PySpark's
        # ``ArtifactManager``, the hashes of ``hashlib`` come first, so a blob that is not
        # bytes-like raises its ``TypeError`` before anything is sent.
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
        """
        Create a new client against the same endpoint with a fresh, independent server-side
        session that does NOT inherit any state from this client's session. Unlike
        :meth:`clone`, no state (SQL configurations, temporary views, registered functions,
        catalog state) is copied over, and no server round-trip is made: the new client is
        built from a copy of this client's connection configuration with the session ID
        cleared, so a fresh session ID is generated and the server lazily creates an empty
        isolated session for it.

        Returns
        -------
        SparkConnectClient
            A new SparkConnectClient instance bound to a fresh, empty session.
        """
        # A client of a new session of the same connection string, over the connection of the
        # process to its server, as SparkConnectClient's ``newSession``: of none of the state of this
        # session, of its hooks, its deadlines and whether its executions are reattachable, of the
        # user of ``_user_id`` unless the connection has one, and of the default retry policy.
        session = _new_session_of(self._session, client_state=self.__dict__)
        # A client that no session object owns, as PySpark's, lives until the exit.
        client = SparkConnectClient._of_session(session)
        client._keep_until_exit(session)
        return client

    def clone(self, new_session_id: Optional[str] = None) -> "SparkConnectClient":
        """
        Clone this client session on the server side. The server-side session is cloned with
        all its current state (SQL configurations, temporary views, registered functions,
        catalog state) copied over to a new independent session. The returned client with the
        cloned session is isolated from this client's session - any subsequent changes to
        either session's server-side state will not be reflected in the other.

        Parameters
        ----------
        new_session_id : str, optional
            Custom session ID to use for the cloned session (must be a valid UUID).
            If not provided, a new UUID will be generated.

        Returns
        -------
        SparkConnectClient
            A new SparkConnectClient instance with the cloned session.

        Notes
        -----
        This creates a new server-side session with the specified or generated session ID
        while preserving the current session's configuration and state.

        .. note::
            This is a developer API.
        """
        # A client of a clone of this session on the server, of the ID ``new_session_id``, a
        # UUID, if one is given, as SparkConnectClient's ``clone``: of a copy of its SQL
        # configurations, temporary views, registered functions and catalog state, of the hooks,
        # the deadlines and the reattachable executions of this client, of the user of its
        # ``_user_id`` unless the connection has one, and of the default retry policy.
        session = _new_session_of(
            self._session, True, new_session_id, client_state=self.__dict__
        )
        # A client that no session object owns, as PySpark's, lives until the exit.
        client = SparkConnectClient._of_session(session)
        client._keep_until_exit(session)
        return client

    def register_progress_handler(self, handler: "ProgressHandler") -> None:
        """
        Register a progress handler to be called when a progress message is received.

        Parameters
        ----------
        handler : ProgressHandler
          The callable that will be called with the progress information.

        """
        # Register ``handler`` to be called with the progress of the executions of the session,
        # once, as SparkConnectClient's ``register_progress_handler``.
        self._session.registerProgressHandler(handler)

    def remove_progress_handler(self, handler: "ProgressHandler") -> None:
        """
        Remove a progress handler from the list of registered handlers.

        Parameters
        ----------
        handler : ProgressHandler
          The callable to remove from the list of progress handlers.

        """
        # Remove the progress handler ``handler``, as SparkConnectClient's
        # ``remove_progress_handler``: a ``ValueError`` for one that is not registered.
        self._session.removeProgressHandler(handler)

    def clear_progress_handlers(self) -> None:
        # Remove the progress handlers, as SparkConnectClient's ``clear_progress_handlers``.
        self._session.clearProgressHandlers()

    def disable_reattachable_execute(self) -> "SparkConnectClient":
        # Make the executions of the session not reattachable from the next one on, as
        # SparkConnectClient's ``disable_reattachable_execute``: the client.
        self._session._set_reattachable_execute(False)
        return self

    def enable_reattachable_execute(self) -> "SparkConnectClient":
        # Make the executions of the session reattachable from the next one on, as
        # SparkConnectClient's ``enable_reattachable_execute``: the client.
        self._session._set_reattachable_execute(True)
        return self

    def set_retry_policies(self, policies: Iterable["RetryPolicy"]) -> None:
        """
        Sets list of policies to be used for retries.
        I.e. set_retry_policies([DefaultPolicy(), CustomPolicy()]).

        """
        # Set the retry policies of the RPCs of the session, as SparkConnectClient's
        # ``set_retry_policies``: a failure is retried by the first of ``policies`` that can retry
        # it, with its backoff, until its retries are spent. The Rust transport retries with the
        # parameters of PySpark's own ``DefaultPolicy`` and ``RetryPolicy`` as they are now, of
        # whole milliseconds, without Python; any other policy, as a subclass of its own
        # ``can_retry`` or ``to_state``, is asked by PySpark's ``Retrying`` of each call, as
        # PySpark's client asks it, of the ``grpc.RpcError`` of each failed RPC.
        self._set_retry_policies(list(policies), None)

    def get_retry_policies(self) -> list["RetryPolicy"]:
        """
        Return list of currently used policies
        """
        # The retry policies of the RPCs of the session, as SparkConnectClient's
        # ``get_retry_policies``: those that ``set_retry_policies`` set, else a ``DefaultPolicy``
        # of the parameters that the transport retries with.
        policies = _RETRY_POLICIES.get(self._context)
        if policies is None:
            from pyspark.sql.connect.client.retries import DefaultPolicy

            parameters = self._session._retry_policy()
            parameters.pop("max_retry_exception_elapsed_time")
            parameters.pop("retries_failures")
            policies = [DefaultPolicy(**parameters)]
            _RETRY_POLICIES[self._context] = policies
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
        native = None
        if policies is not None:
            native = _retry_parameters(policies)
            if native is not None:
                parameters.update(native)
        self._session._set_retry_policy(**parameters)
        if policies is not None:
            # Policies that the transport does not retry with as they are, PySpark's `Retrying`
            # decides, of a copy of the list, as PySpark's client keeps one.
            self._session._set_retry_decider(None if native is not None else list(policies))
            _RETRY_POLICIES[self._context] = policies

    def release_session(self) -> None:
        # Release the session on the server, as SparkConnectClient's: its state there.
        self._session._release_session()

    def close(self) -> None:
        """
        Close the channel.
        """
        # Close the client, as SparkConnectClient's: of a session that ``is_closed``, and is
        # not released on the server, whose channel is closed, so that its requests raise the
        # ``ValueError`` of a closed channel once ``_closed`` is set to False.
        self._session._close()

    def _resources(self) -> dict[str, Any]:
        """The resources of the cluster, as SparkConnectClient's: of a GetResourcesCommand, by
        their names."""
        import pyspark.sql.connect.proto as pb2

        logger.debug("Fetching the resources")
        command = pb2.Command()
        command.get_resources_command.SetInParent()
        _, properties, _ = self.execute_command(command)
        return properties["get_resources_command_result"]

    def _create_profile(self, profile: Any) -> int:
        """Create ``profile``, a ``spark.connect.ResourceProfile``, on the server, as
        SparkConnectClient's: the profile ID that the server gives it. The ResourceProfile of
        pyspark.sql.connect.resource.profile creates its profile with it."""
        import pyspark.sql.connect.proto as pb2

        logger.debug("Creating the ResourceProfile")
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
                    logger.warning(
                        "Skipping re-entrant ML cache delete of %s object ref(s) while another "
                        "ML-cache RPC is in flight on this thread (avoids a re-entrant gRPC hang).",
                        len(cache_ids),
                    )
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
                logger.warning(
                    "Skipping re-entrant ML cache cleanup while another ML-cache RPC is in flight "
                    "on this thread (avoids a re-entrant gRPC hang)."
                )
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


class _ProtoText:
    # What SparkConnectClient._proto_to_string reads of its client: its _truncate.
    _truncate = SparkConnectClient._truncate


def _proto_text(type_name: str, data: bytes, truncate: bool) -> str:
    """The text of the message of the full name ``type_name`` of the bytes ``data``, a message of
    Spark Connect that the Rust transport builds or reads, of any depth, as
    ``SparkConnectClient._proto_to_string`` writes it in a log of the client: on one line,
    truncated when ``truncate``, as PySpark's client logs its plans, its requests and its
    responses."""
    import pyspark.sql.connect.proto as pb2

    message = _pyspark._proto_parse_deep(getattr(pb2, type_name.rsplit(".", 1)[-1]), data)
    return SparkConnectClient._proto_to_string(_ProtoText(), message, truncate)  # type: ignore


def _call_stack_of_call(
    innermost: Optional[str] = None, cached_property: Optional[str] = None
) -> list[tuple[str, str, int]]:
    """The call stack of the user's code of the call of the client that runs, of the frames of
    the caller of the method of the extension that calls this, as PySpark's
    ``SparkConnectClient._retrieve_stack_frames`` reads it: from the outermost frame to the first
    frame of the pyspark package, the function, the file and the line of each, the function of the
    frame that it calls, of the frame after it. The methods of the extension have no frames: the
    innermost frame of the user's code that calls one has the name ``innermost``, as ``__del__`` of
    a finalizer, else the name of the method or the attribute that the source of its expression
    names (``collect`` of ``df.collect()``), as PySpark's has the name of the function of PySpark
    that it calls, or else its own, as PySpark's last frame. Of the attribute ``cached_property``,
    a ``functools.cached_property`` of PySpark's, it calls the ``__get__`` of ``functools``, whose
    frame calls the function of the attribute, as PySpark's."""
    import traceback

    caller = sys._getframe(1)
    frames = traceback.extract_stack(caller)
    stack = []
    for i, frame in enumerate(frames):
        filename, lineno, func, _ = frame
        if _is_pyspark_source(filename):
            break
        if i + 1 < len(frames):
            _, _, func, _ = frames[i + 1]
        elif cached_property is not None and _cached_property_frame() is not None:
            func = "__get__"
        else:
            func = innermost or _called_name(caller) or func
        stack.append((func, filename, lineno))
    else:
        getter = _cached_property_frame() if cached_property is not None else None
        if stack and getter is not None:
            stack.append((cached_property, *getter))
    return stack


_CACHED_PROPERTY_FRAME: list[Optional[tuple[str, int]]] = []


def _cached_property_frame() -> Optional[tuple[str, int]]:
    """The file and the line of the frame of the ``__get__`` of ``functools.cached_property`` as it
    calls the function of the attribute, of this Python, as PySpark's frames of a
    ``cached_property`` have them; None of a ``__get__`` that has no frame of Python."""
    if not _CACHED_PROPERTY_FRAME:

        class _Probe:
            @functools.cached_property
            def frame(self) -> Optional[tuple[str, int]]:
                getter = sys._getframe(1)
                if getter.f_code.co_name != "__get__":
                    return None
                return getter.f_code.co_filename, getter.f_lineno

        _CACHED_PROPERTY_FRAME.append(_Probe().frame)
    return _CACHED_PROPERTY_FRAME[0]


# The names of `_called_name`, by the identity of the code and the instruction of a frame, with the
# code, of a size of at most `_CALLED_NAMES_SIZE`: a call of a site reads its source once. Of the
# `id` of the code, which hashes at once, where the hash of a code hashes its constants, of a
# module of many lines too; the code that the entry keeps is compared by identity, so that an id
# that another code takes again is not of its entry.
_CALLED_NAMES: dict[tuple[int, int], tuple[Any, Optional[str]]] = {}
_CALLED_NAMES_SIZE = 256


def _called_name(frame: Any) -> Optional[str]:
    """The name of the method or the attribute that ``frame`` calls or reads at its instruction,
    of the source of the expression of the instruction: ``collect`` of ``df.collect()`` and
    ``schema`` of ``df.schema``; None of a source that names none, as an alias, or of no source."""
    try:
        code = frame.f_code
        key = (id(code), frame.f_lasti)
    except AttributeError:
        return None
    entry = _CALLED_NAMES.get(key)
    if entry is not None and entry[0] is code:
        return entry[1]
    called = _called_name_of(frame)
    if len(_CALLED_NAMES) >= _CALLED_NAMES_SIZE:
        _CALLED_NAMES.clear()
    _CALLED_NAMES[key] = (code, called)
    return called


def _called_name_of(frame: Any) -> Optional[str]:
    """`_called_name` of ``frame``, of its source."""
    import ast
    import linecache

    try:
        index = frame.f_lasti // 2
        positions = _positions_of(frame.f_code)
        lineno, end_lineno, col, end_col = positions[index]
    except (AttributeError, IndexError, ValueError):
        return None
    if None in (lineno, end_lineno, col, end_col):
        return None
    source = linecache.getlines(frame.f_code.co_filename, frame.f_globals)
    lines = source[lineno - 1 : end_lineno]
    if len(lines) != end_lineno - lineno + 1:
        return None
    # Of the UTF-8 offsets of the columns of the positions, within one line or over several.
    encoded = [line.encode("utf-8") for line in lines]
    before = encoded[0][:col]
    encoded[-1] = encoded[-1][:end_col]
    encoded[0] = encoded[0][col:]
    try:
        node = ast.parse("(" + b"".join(encoded).decode("utf-8") + ")", mode="eval").body
    except (SyntaxError, UnicodeDecodeError, ValueError):
        return None
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Attribute):
        return node.attr
    if not isinstance(node, ast.Name):
        return None
    # Of an attribute of another line than its object, the positions start at its name, after the
    # dot, of this line or of one before it, past the lines of comments only. The tokens are of the
    # source from a line that starts outside any string (see `_token_line`), so that a string of
    # several lines is read as one, of no indentation, which a start inside brackets would misread.
    first = _token_line(frame.f_code, positions, index, source, lineno)
    text = "".join(line.lstrip(" \t\f") for line in source[first - 1 : lineno - 1])
    last = _last_token(text + before.decode("utf-8", "replace").lstrip(" \t\f"))
    return node.id if last == "." else None


# The positions of the instructions of the codes of at least `_POSITIONS_CACHED_FROM` of them, as
# their `co_positions()` gives them, of at most `_POSITIONS_SIZE` codes: the calls of a module of
# many lines read its positions once. By the identity of the code, with the code, as
# `_CALLED_NAMES`.
_POSITIONS: dict[int, tuple[Any, list[Any]]] = {}
_POSITIONS_SIZE = 4
_POSITIONS_CACHED_FROM = 1024


def _positions_of(code: Any) -> list[Any]:
    """The positions of the instructions of ``code``, by their index, of `_POSITIONS`."""
    entry = _POSITIONS.get(id(code))
    if entry is not None and entry[0] is code:
        return entry[1]
    positions = list(code.co_positions())
    if len(positions) >= _POSITIONS_CACHED_FROM:
        if len(_POSITIONS) >= _POSITIONS_SIZE:
            _POSITIONS.clear()
        _POSITIONS[id(code)] = (code, positions)
    return positions


def _token_line(
    code: Any, positions: list[Any], index: int, source: list[str], lineno: int
) -> int:
    """The line of ``source``, the source of ``code``, from which `_called_name` tokenizes the
    source before a position of the line ``lineno``, of the instruction ``index`` of ``code``,
    whose instructions have the positions ``positions``: the last line before it, of an
    instruction before it, whose first token is where that instruction starts, which no string
    holds, after whitespace only, so that the line starts outside any string; else the first line
    of ``code``, the start of a function or a class, else the first line of the file, as of a
    lambda or a comprehension."""
    indents: dict[int, int] = {}
    for line, _, col, _ in reversed(positions[:index]):
        if line is None or col is None or not 0 < line < lineno or line > len(source):
            continue
        if line not in indents:
            text = source[line - 1].encode("utf-8")
            indents[line] = len(text) - len(text.lstrip(b" \t\f"))
        if col == indents[line]:
            return line
    first = code.co_firstlineno
    if code.co_name.startswith("<") or not 0 < first <= lineno:
        return 1
    return first


def _last_token(source: str) -> Optional[str]:
    """The last token of ``source``, Python source from the start of a statement, that is neither
    a comment nor a line break, or None of a source of none, as one of blank lines or comments."""
    import io
    import tokenize

    last = None
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type not in (
                tokenize.COMMENT,
                tokenize.NL,
                tokenize.NEWLINE,
                tokenize.INDENT,
                tokenize.DEDENT,
                tokenize.ENDMARKER,
            ):
                last = token.string
    except (tokenize.TokenError, SyntaxError):
        # Of a source that a statement goes on after, as one of an open bracket: its tokens so far.
        pass
    return last


# The type URL of the placeholder of the relation of a DataFrame of the transport, of the value of
# its marker alone, 8 bytes, which stands for the relation of the DataFrame, in the encoding of a
# message that holds it: `google.protobuf.Any.type_url` (1) and `value` (2).
_MARKER_PLACEHOLDER = b"type.googleapis.com/spark.connect.rust.DataFramePlaceholder\x12\x08"


def _plan_of_bytes(field: int, data: bytes) -> bytes:
    """The bytes of a ``spark.connect.Plan`` of the message of the bytes ``data`` in its field
    ``field``: its relation (``root``, 1) or its command (``command``, 2)."""
    header = bytearray([field << 3 | 2])
    size = len(data)
    while size > 0x7F:
        header.append(size & 0x7F | 0x80)
        size >>= 7
    header.append(size)
    return bytes(header) + data


def _proto_str(value: Any) -> str:
    # The text of `value` that a string field of protobuf's default backend (upb) takes, as
    # PySpark's requests set it: a str as it is, which UTF-8 must encode, and bytes of UTF-8, of the
    # UnicodeEncodeError and the UnicodeDecodeError of those conversions, and the TypeError of any
    # other object.
    if isinstance(value, str):
        value.encode("utf-8")
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8")
    raise TypeError("bad argument type for built-in operation")


_MISSING = object()


def _client_attribute(state: dict[str, Any], name: str) -> Any:
    # The attribute `name` of the clients of `state`, or the AttributeError of a client that has
    # none, as PySpark's requests read the attributes of its client.
    try:
        return state[name]
    except KeyError:
        raise AttributeError(f"'SparkConnectClient' object has no attribute '{name}'") from None


def _serialized_extensions(extensions: Any) -> list[bytes]:
    # The bytes of the google.protobuf.Any of each of `extensions`, pairs of an ID and an Any, as
    # the requests of the Rust transport carry them. Each is added to the extensions of a
    # UserContext, as PySpark's `_update_request_with_user_context_extensions` adds it to the
    # request that it builds, so an item that is not a pair, or an extension that is not an Any
    # (another message, or not a message), fails as it fails there, of upb's TypeError.
    if type(extensions) is list and not extensions:
        return []
    from pyspark.sql.connect.proto import UserContext

    context = UserContext()
    for _, extension in extensions:
        context.extensions.append(extension)
    return [extension.SerializeToString() for extension in context.extensions]


# The refresher of the session of a copy of a client whose method runs, and the attributes of the
# copy, which the requests of the method carry (see `SparkConnectClient.__copy__`).
_COPY_CALL: "contextvars.ContextVar[Optional[tuple[Any, dict[str, Any]]]]" = (
    contextvars.ContextVar("_COPY_CALL", default=None)
)


class _RequestContext:
    """The attributes of the clients of a session, which the session keeps as long as it lives,
    as PySpark's session keeps its client (see ``_client_state``), and the refresher that the Rust
    transport calls before each call of the session that sends requests, and before the request
    of a finalizer, which gives it what the requests carry of them, as PySpark's requests read
    them when they are built: the session ID of ``_session_id`` and the user of ``_user_id``, the
    user context extensions of ``global_user_context_extensions`` and of the
    ``thread_local.user_context_extensions`` of the calling thread, and the tags of its
    ``thread_local.tags``, which only the executions carry, of the client that the thread calls.
    Of the mode 2, the calls whose requests carry the extensions (executions, analyses,
    configurations and interrupts) and the finalizers, a value that the field of a request does not
    take raises the error of protobuf, as PySpark's request raises it, but for a tag, with which
    only the executions fail; of the mode 1 (``ReleaseSession``, ``GetStatus``,
    ``CloneSession``), only those of the session ID and the user are raised; of the mode 0 (the
    artifacts), none is. A value that a refresh of no error does not read is ``...``."""

    __slots__ = ("state", "__weakref__")

    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state

    def __call__(self, mode: int) -> tuple[Any, ...]:
        # The attributes of the copy of a client whose method calls, else those of the session,
        # which a finalizer (the mode 3, of the mode 2) reads, as PySpark's reads the client of
        # its session; the contextvar of the copy is None while the call runs, so that a call that
        # it makes meanwhile, of a progress handler or a finalizer, is of the session (see
        # `_after_call`).
        state, token = self.state, None
        if mode < 3:
            token = self._hold()
            if token is not None:
                state = token.old_value[1]
        try:
            values = _request_values(state, min(mode, 2))
        except BaseException:
            if token is not None:
                _COPY_CALL.reset(token)
            raise
        return values + (token, state is self.state)

    def _hold(self) -> Any:
        # The token of the contextvar of the copy of a client of this session whose method calls,
        # which is None while the call runs, until `_after_call` sets it again, or None: of a call
        # that reads nothing of the client, as the `next()` of the iterator of an execution.
        call = _COPY_CALL.get()
        if call is not None and call[0] is self:
            return _COPY_CALL.set(None)
        return None

    def _enter_call_of(self, token: Any) -> Any:
        # Make the calls that the calling thread makes from now on calls of the client of the call
        # whose refresh held the contextvar of `token` (see `_hold`), the copy of a client whose
        # method made it, as calls of its methods, until `_leave_call_of` takes the returned
        # token: as PySpark's execution calls `interrupt_operation` of the client of the execution.
        if token is None or token.old_value is contextvars.Token.MISSING:
            return None
        return _COPY_CALL.set(token.old_value)

    def _leave_call_of(self, token: Any) -> None:
        # The calls of the calling thread are of the client that they were of before
        # `_enter_call_of` gave `token`.
        if token is not None:
            _COPY_CALL.reset(token)

    def _values_of_call(self, token: Any = None, mode: int = 0) -> tuple[Any, ...]:
        # The values of the requests of a call of a client of this session that runs on the calling
        # thread, read again as PySpark's requests of the rest of the call read the attributes of
        # its client then: of the copy of a client whose method made the call, of the token of the
        # contextvar of its refresh (see `_hold`), else of the session. Of the mode 0, of none of
        # their errors, once a call or a finalizer that ran during it ended; of the mode 2, of
        # those of the session ID, the user and the extensions, when the requests of the details
        # of its failure or of the interrupt of a signal are built, which read no tags.
        state = self.state
        if token is not None and token.old_value is not contextvars.Token.MISSING:
            state = token.old_value[1]
        return _request_values(state, mode, strict_tags=False)

    def _after_call(self, read_tags: bool, handled_error: bool, token: Any = None) -> None:
        # What PySpark's client leaves in the `thread_local` of the calling thread once a call
        # ended: the empty set of tags that the `get_tags` of the request of an execution gives a
        # thread of none (`read_tags`), and the `inside_error_handling` that its `_handle_error`
        # sets while it handles the failure of a request, and leaves False (`handled_error`); and
        # the contextvar of a copy of a client whose method made the call, set again.
        try:
            state = self.state
            if token is not None and token.old_value is not contextvars.Token.MISSING:
                state = token.old_value[1]
            if read_tags or handled_error:
                local = _client_attribute(state, "thread_local")
                if read_tags and not hasattr(local, "tags"):
                    local.tags = set()
                if handled_error:
                    local.inside_error_handling = False
        finally:
            if token is not None:
                _COPY_CALL.reset(token)


def _request_values(state: dict[str, Any], mode: int, strict_tags: bool = True) -> tuple[Any, ...]:
    # The session ID, the user, the global extensions, the extensions of the thread and the tags
    # of the thread of the requests of the client of `state`, of the errors of the mode of
    # `_RequestContext`, those of the tags of none unless `strict_tags`.
    def read(strict: bool, value: Any) -> Any:
        if strict:
            return value()
        try:
            return value()
        except Exception:
            return ...

    def global_extensions() -> list[bytes]:
        # As PySpark's requests, of the global extensions first.
        with _client_attribute(state, "global_user_context_extensions_lock"):
            return _serialized_extensions(
                _client_attribute(state, "global_user_context_extensions")
            )

    def own_extensions() -> Optional[list[bytes]]:
        local = _client_attribute(state, "thread_local")
        own = getattr(local, "user_context_extensions", _MISSING)
        return None if own is _MISSING else _serialized_extensions(own)

    def tags() -> Any:
        local = _client_attribute(state, "thread_local")
        tags = getattr(local, "tags", _MISSING)
        if tags is _MISSING:
            return None
        try:
            return list(tags)
        except TypeError as error:
            # The error of `list(self.get_tags())` of PySpark's request of an execution, which only
            # the executions raise.
            return error

    def user_id() -> Optional[str]:
        user = _client_attribute(state, "_user_id")
        # As PySpark's requests, which set the user of a `_user_id` that is not empty.
        return _proto_str(user) if user else None

    identity, extensions = mode >= 1, mode >= 2
    return (
        read(identity, lambda: _proto_str(_client_attribute(state, "_session_id"))),
        read(identity, user_id),
        read(extensions, global_extensions),
        read(extensions, own_extensions),
        read(extensions and strict_tags, tags),
    )


def _parse_response(message_class: Any, data: bytes, streaming: bool = False) -> Any:
    """The response of the server of the bytes ``data``, a message of ``message_class``, as
    PySpark's client has the response of gRPC, whose deserializer is upb's ``FromString``: of at
    most 100 levels of messages, or 65535 once upb's ``SetAllowOversizeProtos(True)`` is called,
    as the Rust transport reads the responses, whose ``RecursionError`` of a response whose kept
    messages nest more than 1024 levels it raises too (see docs/compatibility.md). Bytes that do
    not parse raise PySpark's error of a response that gRPC's deserializer refuses: the
    ``SparkConnectGrpcException`` of the status ``INTERNAL`` of grpc's error of the call
    (``_MultiThreadedRendezvous`` of a call of a stream of responses, ``streaming``, else
    ``_InactiveRpcError``), whose details are "Exception deserializing response!"."""
    from pyspark.sql.connect.proto._rust import DecodeError

    try:
        _pyspark._proto_check_response_depth(message_class.DESCRIPTOR.full_name, data)
        return message_class.FromString(data)
    except DecodeError:
        from pyspark.errors.exceptions.connect import SparkConnectGrpcException, StatusCode

        call = "_MultiThreadedRendezvous" if streaming else "_InactiveRpcError"
        raise SparkConnectGrpcException(
            f"<{call} of RPC that terminated with:\n\tstatus = StatusCode.INTERNAL\n"
            '\tdetails = "Exception deserializing response!"\n\tdebug_error_string = "None"\n>',
            grpc_status_code=StatusCode.INTERNAL,
        ) from None


def _in_copy_context(method: Any) -> Any:
    # A method of SparkConnectClient whose requests, of a copy of a client, carry the attributes
    # of the copy, as the requests of a copy of PySpark's client read its own.
    if inspect.isgeneratorfunction(method):

        @functools.wraps(method)
        def generator(self: Any, *args: Any, **kwargs: Any) -> Any:
            items = method(self, *args, **kwargs)
            context = getattr(self, "_copy_context", None)
            if context is None:
                return (yield from items)
            call = (context, self.__dict__)
            try:
                while True:
                    token = _COPY_CALL.set(call)
                    try:
                        item = next(items)
                    except StopIteration as stop:
                        return stop.value
                    finally:
                        _COPY_CALL.reset(token)
                    yield item
            finally:
                token = _COPY_CALL.set(call)
                try:
                    items.close()
                finally:
                    _COPY_CALL.reset(token)

        return _named_as(generator, method)

    @functools.wraps(method)
    def call(self: Any, *args: Any, **kwargs: Any) -> Any:
        # None of a client whose `__init__` has not set it yet.
        context = getattr(self, "_copy_context", None)
        if context is None:
            return method(self, *args, **kwargs)
        token = _COPY_CALL.set((context, self.__dict__))
        try:
            return method(self, *args, **kwargs)
        finally:
            _COPY_CALL.reset(token)

    return _named_as(call, method)


def _named_as(wrapper: Any, method: Any) -> Any:
    # `wrapper` of the code names of `method`, as the frames of its calls, which tracebacks and the
    # call stack of the requests show, are of PySpark's method itself.
    code = method.__code__
    names = {"co_name": code.co_name}
    if hasattr(code, "co_qualname"):
        names["co_qualname"] = code.co_qualname
    wrapper.__code__ = wrapper.__code__.replace(**names)
    return wrapper


for _name, _member in list(vars(SparkConnectClient).items()):
    if not _name.startswith("__") and inspect.isfunction(_member):
        setattr(SparkConnectClient, _name, _in_copy_context(_member))
del _name, _member


_CLIENT_STATES_LOCK = threading.Lock()


def _new_client_state(session: Any, builder: Any, user_id: Any) -> dict[str, Any]:
    """The attributes of a new client of ``session``, as PySpark's ``SparkConnectClient`` makes
    them: its channel builder ``builder``, as ``_builder``, else a ``DefaultChannelBuilder`` of the
    connection of the session; a ``threading.local`` as its ``thread_local``; an empty list of
    global user context extensions, and its lock; the session ID of the session, as
    ``_session_id``; and as ``_user_id``, the ``user_id`` of the channel builder, else
    ``user_id`` as it is given, else the user of the connection of the session, of
    ``SPARK_USER`` or ``USER``."""
    if builder is None:
        connection_string = getattr(session, "_connection_string", None)
        if connection_string is not None:
            # As PySpark's client of a session, of the channel builder of its connection.
            builder = DefaultChannelBuilder(connection_string())
    if builder is not None and builder.userId is not None:
        user = builder.userId
    elif user_id is not None:
        user = user_id
    else:
        user = session._user_id()
    return {
        "thread_local": threading.local(),
        "_builder": builder,
        "_user_id": user,
        "_session_id": session._client_session_id(),
        "global_user_context_extensions": [],
        "global_user_context_extensions_lock": threading.Lock(),
    }


def _client_context(session: Any) -> _RequestContext:
    """The state of the clients of ``session``, which the session keeps as long as it lives, as
    PySpark's session keeps its client: so the clients of a session share their attributes, and so
    do the objects of the session that are made again while it lives, as a session object that a
    DataFrame gives back once the program dropped the one it had. Its first client makes them (see
    ``_new_client_state``), unless the session was made with them (``_init_client_state``)."""
    with _CLIENT_STATES_LOCK:
        context = session._user_context_refresh()
        if context is None:
            context = _RequestContext(_new_client_state(session, None, None))
            session._set_user_context_refresh(context)
    return context


def _client_state(session: Any) -> dict[str, Any]:
    """The attributes of the clients of ``session`` (see ``_client_context``)."""
    return _client_context(session).state


def _init_client_state(session: Any, connection: Any, user_id: Any = None) -> None:
    """Give ``session``, a new session of ``connection``, the attributes of PySpark's client of
    ``connection`` and ``user_id``: of the channel builder of ``connection`` (the object itself if
    it is one), and of the user ``user_id`` as it is given, unless the connection has one, as
    PySpark's ``SparkSession(connection, userId)`` and ``SparkConnectClient(connection, user_id)``
    make their client."""
    builder = (
        connection if isinstance(connection, ChannelBuilder) else DefaultChannelBuilder(connection)
    )
    with _CLIENT_STATES_LOCK:
        session._set_user_context_refresh(
            _RequestContext(_new_client_state(session, builder, user_id))
        )


def _new_session_of(
    session: Any,
    clone: bool = False,
    new_session_id: Any = None,
    client_state: Optional[dict[str, Any]] = None,
) -> Any:
    """The session of ``newSession()``, or of ``cloneSession(new_session_id)``, of ``session``, as
    PySpark's ``SparkSession.newSession`` and ``cloneSession`` make it of the ``newSession`` and
    the ``clone`` of its client, of the attributes ``client_state`` of a client, else of those of
    the clients of the session: a session of the user of its ``_user_id`` unless the connection
    has one, of the TypeError of one that is not a str, and of the channel builder of its
    ``_builder``, deep-copied, without a session ID, or of that of the clone."""
    state = client_state
    if state is None:
        context = session._user_context_refresh()
        state = None if context is None else context.state
    if state is None:
        if clone:
            return session._clone_session(new_session_id)
        # Of the user of the connection, as the client of the session would have it.
        return session._new_session(session._user_id())
    builder = _client_attribute(state, "_builder")
    user = _client_attribute(state, "_user_id")
    new_builder = None if builder is None else copy.deepcopy(builder)
    if new_builder is not None and not clone:
        new_builder._params.pop(ChannelBuilder.PARAM_SESSION_ID, None)
    if new_builder is not None and new_builder.userId is not None:
        new_user = new_builder.userId
    elif user is not None:
        new_user = user
    else:
        new_user = os.getenv("SPARK_USER", os.getenv("USER", None))
    if clone:
        # The request of the clone carries the session ID and the user of the client (see
        # `crate::interrupt::RequestContext::Identity`), of their errors; the clone is of the new
        # user, as the client of PySpark's `clone`, which PySpark checks once the clone is made.
        new = session._clone_session(
            new_session_id, None if new_user is None else _proto_str(new_user), True
        )
        if new_builder is not None:
            new_builder.set(ChannelBuilder.PARAM_SESSION_ID, new.session_id)
    else:
        # As the ArtifactManager of PySpark's new client of the user, which sets it in the string
        # field of its user context, before the session is made: its requests and those of its
        # artifacts carry it.
        new = session._new_session(None if new_user is None else _proto_str(new_user))
    with _CLIENT_STATES_LOCK:
        new._set_user_context_refresh(
            _RequestContext(_new_client_state(new, new_builder, new_user))
        )
    return new


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
    if client is None or (
        client._session_cache() is not session and not client._session_ref._is_of(session)
    ):
        client = SparkConnectClient._of_session(session)
        _CLIENTS[session.session_id] = client
    return client


def _checked_channel_builder(connection: Any) -> ChannelBuilder:
    """The channel builder of ``connection``, as PySpark's ``SparkConnectClient`` reads it when it
    is made: ``connection`` itself if it is a ``ChannelBuilder``, else ``DefaultChannelBuilder``
    of it, which raises the errors of an object that is not a connection string, such as the
    ``TypeError`` of its slicing, of its ``session_id``, of its ``userId``, and of the channel of
    its ``toChannel``, which a ``ChannelBuilder`` that is not a ``DefaultChannelBuilder`` has of
    its own: PySpark's raises without one (``PySparkNotImplementedError``). The Rust transport
    connects to the ``sc://`` URL of the ``endpoint`` of a ``DefaultChannelBuilder``, of its
    ``_host`` and ``_port``, or of the ``host`` and the ``port`` of a builder of its own, and of
    the ``_params`` of the
    builder, and not to the channel of its ``toChannel``, which it calls for its errors only: a
    ``DefaultChannelBuilder`` has none, so that one of a subclass, which may call the one of its
    base, is not called."""
    builder = (
        connection if isinstance(connection, ChannelBuilder) else DefaultChannelBuilder(connection)
    )
    _ = builder.session_id
    _ = builder.userId
    if not isinstance(builder, DefaultChannelBuilder):
        to_channel = getattr(builder, "toChannel", None)
        if to_channel is None:
            # As PySpark's `ChannelBuilder.toChannel`, which a builder of its own overrides.
            raise PySparkNotImplementedError
        to_channel()
    return builder


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


def _retry_parameters(policies: list[Any]) -> Optional[dict[str, Any]]:
    """The parameters of ``policies`` that the Rust transport retries with, as PySpark's
    ``Retrying`` retries with them: a failure is retried by the first policy that can retry it, a
    ``DefaultPolicy`` of the failures that the transport retries, as a ``RetryPolicy`` can retry
    none, so those of the first ``DefaultPolicy`` retry, and without one no failure is retried,
    nor are retries spent, of which ``Retrying`` warns. None for policies that the transport does
    not retry with as they are, which PySpark's ``Retrying`` decides (see ``_CallRetries``): any
    that is not exactly a ``DefaultPolicy`` or a ``RetryPolicy`` of PySpark, of methods or a name
    that may be its own, and a ``DefaultPolicy`` of a backoff, a jitter or a delay that is not a
    whole number of milliseconds that the transport waits."""
    from pyspark.sql.connect.client.retries import DefaultPolicy, RetryPolicy

    if any(type(policy) not in (DefaultPolicy, RetryPolicy) for policy in policies):
        return None
    effective = next((policy for policy in policies if type(policy) is DefaultPolicy), None)
    if effective is None:
        return {"max_retries": 0, "retries_failures": False}
    max_retries = effective.max_retries
    if max_retries is not None:
        if (
            not isinstance(max_retries, (int, float))
            or isinstance(max_retries, bool)
            or not math.isfinite(max_retries)
        ):
            return None
        # As RetryPolicyState, which retries while the attempts are fewer than it.
        max_retries = min(max(math.ceil(max_retries), 0), 2**32 - 1)
    milliseconds = {
        name: getattr(effective, name)
        for name in (
            "initial_backoff",
            "max_backoff",
            "jitter",
            "min_jitter_threshold",
            "max_server_retry_delay",
        )
    }
    optional = ("max_backoff", "max_server_retry_delay")
    multiplier = effective.backoff_multiplier
    if (
        not all(
            (value is None and name in optional) or _whole_milliseconds(value)
            for name, value in milliseconds.items()
        )
        or not isinstance(multiplier, (int, float))
        or isinstance(multiplier, bool)
        or not math.isfinite(multiplier)
    ):
        return None
    return {
        "max_retries": max_retries,
        **{name: None if value is None else int(value) for name, value in milliseconds.items()},
        "backoff_multiplier": float(multiplier),
        "recognize_server_retry_delay": bool(effective.recognize_server_retry_delay),
        "retries_failures": True,
    }


def _whole_milliseconds(value: Any) -> bool:
    """Whether ``value``, a parameter of a retry policy, is a whole number of milliseconds that
    the Rust transport waits."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
        and 0 <= value < 2**63
        and value == int(value)
    )


class _CallRetries:
    """PySpark's ``Retrying`` of the retry policies of a client, of one call, which the Rust
    transport asks what to do of each failed attempt of the call, of policies that it does not
    retry with itself (see ``_retry_parameters``): as PySpark's client makes one ``Retrying`` of
    each call, which calls the ``to_state`` of the policies, and asks it, of the exception of each
    failed attempt, whether to retry it (``accept_exception``, which calls their ``can_retry``),
    and then for the wait before the retry (``_wait``, which calls the ``next_attempt`` of the
    state of the policy that retries it, logs, and warns and raises once its retries are spent).
    Its ``sleep`` keeps the wait, which the transport waits."""

    def __init__(self, policies: list[Any]) -> None:
        from pyspark.sql.connect.client.retries import Retrying

        self._waits: list[float] = []
        self._retrying = Retrying(policies, sleep=self._waits.append)

    def decide(
        self,
        streaming: bool,
        code: int,
        details: str,
        debug_error_string: Optional[str],
        trailing_metadata: list[tuple[str, bytes]],
        text: Optional[str] = None,
    ) -> Optional[float]:
        """The wait in seconds before the retry of the attempt that failed of the gRPC error of
        ``code``, ``details``, ``debug_error_string`` (None of a response that the transport
        failed to decode, as grpcio's) and ``trailing_metadata``, of grpcio's text
        ``text``, of a call of a stream of responses if ``streaming``, or None when the failure is
        the error of the call: no policy retries it, or their retries are spent, of which
        ``Retrying`` warned. The exception that ``Retrying`` raises in place of the failure, as
        that of a filter of "error" of its warning, is raised."""
        exception = _grpc_error(
            streaming, code, details, debug_error_string, trailing_metadata, text
        )
        # As PySpark's `AttemptManager.__exit__`, which asks the policies of the exception of the
        # attempt as it is handled, so that what they raise has it as its `__context__`; the wait
        # is out of the handling, as `Retrying._wait` waits.
        try:
            raise exception
        except BaseException:
            if not self._retrying.accept_exception(exception):
                return None
        try:
            self._retrying._wait()
        except BaseException as raised:
            if raised is exception:
                return None
            raise
        # The transport sleeps a wait that `time.sleep` refuses, negative or NaN, and one of
        # decades, of Python's `time.sleep`, as PySpark's sleep does.
        return self._waits.pop()


class _InactiveRpcError(Exception):
    """The error of a failed call of one response, of grpcio's class name and text, where grpcio
    is not installed, of which ``_grpc_error`` makes grpcio's own."""

    def __init__(self, text: str) -> None:
        super().__init__(text)
        self._text = text

    def __str__(self) -> str:
        return self._text

    __repr__ = __str__


class _MultiThreadedRendezvous(_InactiveRpcError):
    """As ``_InactiveRpcError``, of a call of a stream of responses."""


def _log_rpc_error(
    streaming: bool,
    code: int,
    details: str,
    debug_error_string: Optional[str],
    trailing_metadata: list[tuple[str, bytes]],
    text: str,
) -> None:
    """Log the gRPC error of a failed call of the Rust transport, as PySpark's
    ``_handle_rpc_error`` logs each that it converts, of the exception (``logger.exception("GRPC
    Error received")``): grpcio's ``RpcError`` of the call (see ``_grpc_error``), or, without
    grpcio, an exception of its class name and its text, of no frames of PySpark's client."""
    if not logger.isEnabledFor(logging.ERROR):
        return
    error = _rpc_error(streaming, code, details, debug_error_string, trailing_metadata, text)
    logger.error("GRPC Error received", exc_info=error)


def _rpc_error(
    streaming: bool,
    code: int,
    details: str,
    debug_error_string: Optional[str],
    trailing_metadata: list[tuple[str, bytes]],
    text: str,
) -> BaseException:
    """grpcio's ``RpcError`` of a failed call of the Rust transport (see ``_grpc_error``), or,
    without grpcio, an exception of its class name and its text, of no frames of PySpark's client:
    the exception that PySpark's ``_handle_rpc_error`` handles of the call."""
    try:
        return _grpc_error(streaming, code, details, debug_error_string, trailing_metadata, text)
    except Exception:
        return (_MultiThreadedRendezvous if streaming else _InactiveRpcError)(text)


def _log_rpc_deadlines(deadlines: Optional["RpcDeadlines"] = None, **seconds: Any) -> None:
    """Log the RPC deadlines of a new client, ``deadlines``, else of ``seconds``, as PySpark's
    ``SparkConnectClient`` logs them when it is made."""
    if logger.isEnabledFor(logging.INFO):
        logger.info(
            "Spark Connect RPC deadlines: %s",
            deadlines if deadlines is not None else RpcDeadlines(**seconds),
        )


def _grpc_error(
    streaming: bool,
    code: int,
    details: str,
    debug_error_string: Optional[str],
    trailing_metadata: list[tuple[str, bytes]],
    text: Optional[str] = None,
) -> Any:
    """grpcio's ``RpcError`` of a failed call, as PySpark's client sees it: a
    ``_MultiThreadedRendezvous`` of a call of a stream of responses, ``ExecutePlan`` and
    ``ReattachExecute``, else an ``_InactiveRpcError``, of the code ``code``, the details
    ``details``, the ``debug_error_string`` ``debug_error_string`` and the trailing metadata
    ``trailing_metadata``, whose binary entries are bytes, as ``grpc_status.rpc_status`` reads
    them, each grpcio's ``_Metadatum`` of its key and value where it has one. The
    ``debug_error_string`` of a response that the transport failed to decode is None, as grpcio's
    of a response that it fails to deserialize: ``debug_error_string()`` of such a
    ``_MultiThreadedRendezvous`` waits for one, which nothing sets, as grpcio's does. Those classes
    are grpcio's private ones, made of its private ``_RPCState``: where they cannot be made so, as
    of another version of grpcio, the error is of the public classes of grpc of the same names, of
    grpcio's text of the call, ``text`` (see ``_rpc_error_classes``). Raises ImportError without
    grpcio."""
    import grpc

    status = next(
        (member for member in grpc.StatusCode if member.value[0] == code), grpc.StatusCode.UNKNOWN
    )
    metadata = tuple(
        (key, value if key.endswith("-bin") else value.decode("latin-1"))
        for key, value in trailing_metadata
    )
    try:
        from grpc import _channel

        try:
            from grpc._cython import cygrpc

            metadata = tuple(cygrpc._Metadatum(key, value) for key, value in metadata)
        except Exception:  # noqa: BLE001 - of plain tuples where grpcio has no _Metadatum
            pass
        state = _channel._RPCState((), (), metadata, status, details)
        state.debug_error_string = debug_error_string
        if streaming:
            return _channel._MultiThreadedRendezvous(state, None, None, None)
        return _channel._InactiveRpcError(state)
    except Exception:
        inactive, rendezvous = _rpc_error_classes()
        return (rendezvous if streaming else inactive)(
            status, details, debug_error_string, metadata, text
        )


@functools.cache
def _rpc_error_classes() -> tuple[type, type]:
    """The ``RpcError`` of a failed call of the public classes of grpc, ``grpc.RpcError`` and
    ``grpc.Call``, where grpcio's own cannot be made (see ``_grpc_error``): of the names of
    grpcio's ``_InactiveRpcError`` and ``_MultiThreadedRendezvous``, of the code, the details,
    the ``debug_error_string`` and the trailing metadata of the call, and of grpcio's text of it,
    as ``str()`` and ``repr()`` give it."""
    import grpc

    class _InactiveRpcError(grpc.RpcError, grpc.Call):  # type: ignore[misc]
        def __init__(
            self,
            code: Any,
            details: str,
            debug_error_string: Optional[str],
            trailing_metadata: tuple,
            text: Optional[str],
        ) -> None:
            super().__init__()
            self._code = code
            self._details = details
            self._debug_error_string = debug_error_string
            self._trailing_metadata = trailing_metadata
            self._text = (
                text
                if text is not None
                else (
                    f"<{type(self).__name__} of RPC that terminated with:\n\tstatus = {code}\n"
                    f'\tdetails = "{details}"\n\tdebug_error_string = "{debug_error_string}"\n>'
                )
            )

        def initial_metadata(self) -> tuple:
            return ()

        def trailing_metadata(self) -> tuple:
            return self._trailing_metadata

        def code(self) -> Any:
            return self._code

        def details(self) -> str:
            return self._details

        def debug_error_string(self) -> Optional[str]:
            return self._debug_error_string

        def is_active(self) -> bool:
            return False

        def time_remaining(self) -> Optional[float]:
            return None

        def cancel(self) -> bool:
            return False

        def add_callback(self, callback: Any) -> bool:
            return False

        def __str__(self) -> str:
            return self._text

        __repr__ = __str__

    class _MultiThreadedRendezvous(_InactiveRpcError):
        pass

    return _InactiveRpcError, _MultiThreadedRendezvous


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
_RETRIES_MODULE = "pyspark.sql.connect.client.retries"
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


def _log_grpc_error(file: str, line: int, message: str, severity: str = "E") -> None:
    """Log what a channel logs as grpcio does, for the Rust transport, which calls this on the
    thread of the channel, or on its delivery thread: an error of a channel that the client goes
    on from, such as a proxy variable that is not an HTTP proxy, a file of TLS roots that cannot
    be read or a proxy that refused a tunnel, or the information of a TLS handshake that failed
    (``severity`` ``"I"``). PySpark's client shows them only as the line that gRPC's C core writes
    to the stderr of the process, of the form ``E<month><day> <time> <thread> <file>:<line>]
    <message>``, where ``file`` and ``line`` are of the Rust source of the call here."""
    import datetime
    import threading

    now = datetime.datetime.now()
    entry = (
        f"{severity}{now:%m%d %H:%M:%S.%f} {threading.get_native_id():7d} {file}:{line}] "
        f"{message}\n"
    )
    try:
        os.write(2, entry.encode())
    except OSError:
        # As gRPC's core, which goes on when stderr is closed.
        pass
