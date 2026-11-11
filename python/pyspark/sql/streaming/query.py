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

"""
Streaming query management.

This module re-exports the PyO3-implemented streaming query classes that provide
the client-side API for managing and inspecting streaming queries, and extends them
with Python-side listener bus implementation.
"""

import json
import warnings
from threading import Lock, Thread
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union

from pyspark import cloudpickle
from pyspark._pyspark import (
    StreamingQuery as _StreamingQuery,
)
from pyspark._pyspark import (
    StreamingQueryManager as _StreamingQueryManager,
)

# As PySpark's, the exception of a failed query is that of pyspark.errors.
from pyspark.errors import (
    PySparkValueError,
    StreamingQueryException,  # noqa: F401
)
from pyspark.sql.streaming.listener import (
    QueryIdleEvent,
    QueryProgressEvent,
    QueryStartedEvent,
    QueryTerminatedEvent,
    StreamingQueryListener,
)

if TYPE_CHECKING:
    from pyspark.sql.session import SparkSession

# StreamingQueryEventType values (spark.connect.StreamingQueryEventType); the Rust
# listener stream yields these as the event_type int alongside each event's JSON. The started
# event of a query comes with the result of its start, which the Rust core posts as
# spark_connect::streaming::QUERY_STARTED_EVENT, of no StreamingQueryEventType.
_QUERY_STARTED_EVENT = -1
_QUERY_PROGRESS_EVENT = 1
_QUERY_TERMINATED_EVENT = 2
_QUERY_IDLE_EVENT = 3

# Re-export the core classes from the Rust extension
StreamingQuery = _StreamingQuery


def _deserialize_listener_event(event_type, event_json, *, started=False):
    """Decode a protocol event before entering callback exception isolation.

    Started events belong to the synchronous start response, not the asynchronous
    protocol enum. Malformed JSON and missing fields propagate to the caller, as in
    reference PySpark's writer and listener-bus deserializers.
    """
    if started and event_type == _QUERY_STARTED_EVENT:
        return QueryStartedEvent.fromJson(json.loads(event_json))
    if event_type == _QUERY_PROGRESS_EVENT:
        return QueryProgressEvent.fromJson(json.loads(event_json))
    if event_type == _QUERY_TERMINATED_EVENT:
        return QueryTerminatedEvent.fromJson(json.loads(event_json))
    if event_type == _QUERY_IDLE_EVENT:
        return QueryIdleEvent.fromJson(json.loads(event_json))
    raise PySparkValueError(
        errorClass="UNKNOWN_VALUE_FOR",
        messageParameters={"var": f"proto.StreamingQueryEventType: {event_type}"},
    )


def _dispatch_listener_event(listener, event):
    """Warn on callback Exception; BaseException and warning-as-error escape to the bus."""
    try:
        if isinstance(event, QueryStartedEvent):
            listener.onQueryStarted(event)
        elif isinstance(event, QueryProgressEvent):
            listener.onQueryProgress(event)
        elif isinstance(event, QueryIdleEvent):
            listener.onQueryIdle(event)
        elif isinstance(event, QueryTerminatedEvent):
            listener.onQueryTerminated(event)
        else:
            warnings.warn(f"Unknown StreamingQueryListener event: {event}")
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"Listener {str(listener)} threw an exception\n{e}")


# ``spark.streams`` returns the native (Rust) StreamingQueryManager, which owns the
# client-side listener bus: addListener / removeListener / close are implemented in the
# Rust core (so Rust clients get the listener feature too). Re-export it as the public
# class.
StreamingQueryManager = _StreamingQueryManager


__all__ = ["StreamingQuery", "StreamingQueryManager"]
