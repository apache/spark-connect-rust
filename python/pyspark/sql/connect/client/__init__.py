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

__all__ = ["SparkConnectClient"]


import weakref
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


class SparkConnectClient:
    """The client of ``SparkSession.client``: of PySpark Connect's ``SparkConnectClient``, what
    the modules of PySpark that build their own messages, such as ``pyspark.ml``, use: their
    commands and the ML cache, over the Rust transport of the session, and the session ID. The
    messages are those of the generated ``pyspark.sql.connect.proto``. The Rust transport is the
    rest of the client, of the DataFrame API."""

    def __init__(self, session: Any) -> None:
        self._session = session
        # The thread of an ML cache RPC in flight, as SparkConnectClient's re-entrancy guard of a
        # RemoteModelRef finalizer that deletes a model.
        self._ml_cache_rpc_thread: Any = None

    @property
    def _server_session_id(self) -> str:
        return self._session.session_id

    def execute_command(self, command: Any, observations: Any = None) -> Any:
        """Execute ``command``, a ``spark.connect.Command``, as SparkConnectClient's
        ``execute_command``: no data, the properties of the results of the responses, of their
        field names, and no execution information."""
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
                "get_resources_command_result",
                "ml_command_result",
                "pipeline_command_result",
            ):
                if response.HasField(field):
                    properties[field] = getattr(response, field)
        return None, properties, None

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
