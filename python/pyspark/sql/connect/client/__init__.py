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
``SparkSession.client``, and related classes."""

__all__ = ["ChannelBuilder", "DefaultChannelBuilder", "SparkConnectClient"]


import copy
import os
import sys
import urllib.parse
import uuid
import weakref
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from pyspark.errors import PySparkNotImplementedError, PySparkValueError
from pyspark.sql.metrics import ObservedMetrics


class ChannelBuilder:
    """The parameters of a Spark Connect connection, as PySpark Connect's ``ChannelBuilder``:
    what ``SparkSession.builder.channelBuilder`` and ``SparkSession(connection=...)`` take. The
    Rust transport opens the channel of the session from the ``host``, the ``port`` and the
    ``_params`` of the builder, as a ``sc://`` connection string, so the endpoint and the
    parameters apply. A custom gRPC channel (``toChannel``), its interceptors and its channel
    options do not: the transport has no gRPC channel to give them to."""

    PARAM_USE_SSL = "use_ssl"
    PARAM_TOKEN = "token"
    PARAM_USER_ID = "user_id"
    PARAM_SESSION_ID = "session_id"

    def __init__(
        self,
        channelOptions: Optional[List[Tuple[str, Any]]] = None,
        params: Optional[Dict[str, str]] = None,
    ):
        if channelOptions:
            raise PySparkNotImplementedError(
                errorClass="NOT_IMPLEMENTED", messageParameters={"feature": "channelOptions"}
            )
        self._params: Dict[str, str] = params or dict()

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
    def session_id(self) -> Optional[str]:
        """The session_id of the parameters, or None if it is not set."""
        session_id = self._params.get(ChannelBuilder.PARAM_SESSION_ID, None)
        if session_id is not None:
            try:
                uuid.UUID(session_id, version=4)
            except ValueError as ve:
                raise PySparkValueError(
                    errorClass="INVALID_SESSION_UUID_ID",
                    messageParameters={"arg_name": "session_id", "origin": str(ve)},
                )
        return session_id

    def metadata(self) -> Iterable[Tuple[str, str]]:
        """The parameters that are not used by the channel, as gRPC metadata of each request."""
        reserved = [
            ChannelBuilder.PARAM_TOKEN,
            ChannelBuilder.PARAM_USE_SSL,
            ChannelBuilder.PARAM_USER_ID,
            "user_agent",
            ChannelBuilder.PARAM_SESSION_ID,
            "grpc_keepalive_enabled",
            "grpc_keepalive_time_ms",
            "grpc_keepalive_timeout_ms",
            "grpc_keepalive_without_calls",
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

    @staticmethod
    def default_port() -> int:
        # The port of a Spark Connect server, which the drop-in does not discover from a local
        # JVM as PySpark does in the testing mode.
        return 15002

    def __init__(self, url: str, channelOptions: Optional[List[Tuple[str, Any]]] = None) -> None:
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


class SparkConnectClient:
    """The client of ``SparkSession.client``: of PySpark Connect's ``SparkConnectClient``, what
    the modules of PySpark that build their own messages, such as ``pyspark.ml``, use: their
    commands and the ML cache, over the Rust transport of the session, and the session ID. The
    messages are those of the generated ``pyspark.sql.connect.proto``. The Rust transport is the
    rest of the client, of the DataFrame API."""

    def __init__(self, session: Any) -> None:
        import threading

        self._session = session
        # The thread of an ML cache RPC in flight, as SparkConnectClient's re-entrancy guard of a
        # RemoteModelRef finalizer that deletes a model.
        self._ml_cache_rpc_thread: Any = None
        # The state of the current thread of the session, as SparkConnectClient's thread_local.
        self.thread_local = _ThreadLocalOfClient(session)
        self.global_user_context_extensions_lock = threading.Lock()

    @property
    def global_user_context_extensions(self) -> List[Any]:
        """The global user context extensions, as SparkConnectClient's: of their IDs."""
        return _extensions_of(self._session._user_context_extensions(True))

    @property
    def host(self) -> str:
        """The hostname where this client intends to connect, as SparkConnectClient's ``host``."""
        return self._session._endpoint_host()

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
        closes the session, which the session registry then no longer returns; a closed client
        stays closed."""
        return self._session.is_stopped

    @_closed.setter
    def _closed(self, value: bool) -> None:
        if value:
            self._session._close()

    @property
    def is_closed(self) -> bool:
        """Whether the session of the client is stopped, as SparkConnectClient's ``is_closed``,
        whether its channel is closed."""
        return self._session.is_stopped

    def _proto_to_string(self, p: Any, truncate: bool = False) -> str:
        """A one-line string of the protobuf message ``p``, truncated if ``truncate``, as
        SparkConnectClient's ``_proto_to_string``, which plan logging uses."""
        from google.protobuf import text_format

        try:
            max_level = 8 if truncate else sys.maxsize
            p2 = self._truncate(p, max_level) if truncate else p
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

    def get_configs(self, *keys: str) -> Tuple[Optional[str], ...]:
        """The values of the configurations ``keys``, in one request, as SparkConnectClient's
        ``get_configs``."""
        return tuple(self._session.conf._get_many(list(keys)))

    def get_config_dict(self, *keys: str) -> Mapping[str, Optional[str]]:
        """The values of the configurations ``keys`` by their keys, in one request, as
        SparkConnectClient's ``get_config_dict``."""
        return dict(zip(keys, self.get_configs(*keys)))

    def get_config_with_defaults(
        self, *pairs: Tuple[str, Optional[str]]
    ) -> Tuple[Optional[str], ...]:
        """The values of the configurations of ``pairs`` of a key and its default, the default of
        a key that is not set, in one request, as SparkConnectClient's
        ``get_config_with_defaults``, of which pandas-on-Spark reads its ANSI mode."""
        return tuple(self._session.conf._get_with_defaults(list(pairs)))

    def execute_command(self, command: Any, observations: Any = None) -> Any:
        """Execute ``command``, a ``spark.connect.Command``, as SparkConnectClient's
        ``execute_command``: no data, the properties of the results of the responses, of their
        field names, as SparkConnectClient converts them (the resources of a
        GetResourcesCommand, the profile ID of a resource profile, and the DataFrame of a
        checkpoint), and no execution information."""
        import pyspark.sql.connect.proto as pb2

        properties: Dict[str, Any] = {}
        for data in self._session._execute_command(command.SerializeToString()):
            response = pb2.ExecutePlanResponse.FromString(data)
            # The results of commands, as SparkConnectClient._execute_and_fetch_as_iterator
            # yields them.
            if response.HasField("sql_command_result"):
                properties["sql_command_result"] = response.sql_command_result.relation
            for field in (
                "write_stream_operation_start_result",
                "streaming_query_command_result",
                "streaming_query_manager_command_result",
                "streaming_query_listener_events_result",
                "pipeline_command_result",
                "pipeline_event_result",
                "ml_command_result",
            ):
                if response.HasField(field):
                    properties[field] = getattr(response, field)
            if response.HasField("get_resources_command_result"):
                from pyspark.resource.information import ResourceInformation

                properties["get_resources_command_result"] = {
                    key: ResourceInformation(resource.name, list(resource.addresses))
                    for key, resource in response.get_resources_command_result.resources.items()
                }
            if response.HasField("create_resource_profile_command_result"):
                properties["create_resource_profile_command_result"] = (
                    response.create_resource_profile_command_result.profile_id
                )
            if response.HasField("checkpoint_command_result"):
                from pyspark.sql.connect.conversion import proto_to_remote_cached_dataframe

                properties["checkpoint_command_result"] = proto_to_remote_cached_dataframe(
                    response.checkpoint_command_result.relation
                )
        return None, properties, None

    def add_threadlocal_user_context_extension(self, extension: Any) -> str:
        """Add ``extension``, a ``google.protobuf.Any``, to the user context of the requests that
        the current thread sends, as SparkConnectClient's: its ID."""
        return self._session._add_user_context_extension(extension.SerializeToString(), False)

    def add_global_user_context_extension(self, extension: Any) -> str:
        """Add ``extension``, a ``google.protobuf.Any``, to the user context of the requests of
        all the threads, as SparkConnectClient's: its ID."""
        return self._session._add_user_context_extension(extension.SerializeToString(), True)

    def remove_user_context_extension(self, extension_id: str) -> None:
        """Remove the user context extension of ``extension_id``, as SparkConnectClient's."""
        self._session._remove_user_context_extension(extension_id)

    def clear_user_context_extensions(self) -> None:
        """Remove the user context extensions of the current thread and the global ones, as
        SparkConnectClient's."""
        self._session._clear_user_context_extensions()

    def release_session(self) -> None:
        """Release the session on the server, as SparkConnectClient's: its state there."""
        self._session._release_session()

    def close(self) -> None:
        """Close the client, as SparkConnectClient's: of a session that ``is_closed``, and is
        not released on the server."""
        self._session._close()

    def _resources(self) -> Dict[str, Any]:
        """The resources of the cluster, as SparkConnectClient's: of a GetResourcesCommand, by
        their names."""
        import pyspark.sql.connect.proto as pb2

        command = pb2.Command()
        command.get_resources_command.SetInParent()
        _, properties, _ = self.execute_command(command)
        return properties["get_resources_command_result"]

    def _delete_ml_cache(self, cache_ids: List[str], evict_only: bool = False) -> List[str]:
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

    def _get_ml_cache_info(self) -> List[str]:
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


def _extensions_of(extensions: Any) -> List[Any]:
    # The user context extensions of the Rust transport, as SparkConnectClient keeps them: the
    # ID and the google.protobuf.Any of each.
    from google.protobuf import any_pb2

    return [(extension_id, any_pb2.Any.FromString(data)) for extension_id, data in extensions]


class _ThreadLocalOfClient:
    """The state of the current thread of the session of a client, as SparkConnectClient's
    ``thread_local``: its user context extensions and its tags, which the Rust transport keeps
    for each thread."""

    def __init__(self, session: Any) -> None:
        self._session = session

    @property
    def user_context_extensions(self) -> List[Any]:
        return _extensions_of(self._session._user_context_extensions(False))

    @property
    def tags(self) -> Any:
        return set(self._session.getTags())


# The client of each session, by its ID, while it is in use.
_CLIENTS: "weakref.WeakValueDictionary[str, SparkConnectClient]" = weakref.WeakValueDictionary()


def _client_of(session: Any) -> "SparkConnectClient":
    """The client of ``session``, the same object while it is in use, as ``SparkSession.client``
    is one object per session."""
    client = _CLIENTS.get(session.session_id)
    if client is None or client._session is not session:
        client = SparkConnectClient(session)
        _CLIENTS[session.session_id] = client
    return client
