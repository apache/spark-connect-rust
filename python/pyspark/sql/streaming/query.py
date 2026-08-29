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
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Union
from threading import Thread, Lock
from pyspark import cloudpickle

from pyspark.sql.streaming.listener import (
    StreamingQueryListener,
    QueryStartedEvent,
    QueryProgressEvent,
    QueryIdleEvent,
    QueryTerminatedEvent,
)

from pyspark._pyspark import (
    StreamingQuery as _StreamingQuery,
    StreamingQueryManager as _StreamingQueryManager,
    StreamingQueryStatus as _StreamingQueryStatus,
    StreamingQueryException as _StreamingQueryException,
)

if TYPE_CHECKING:
    from pyspark.sql.session import SparkSession

# StreamingQueryEventType values (spark.connect.StreamingQueryEventType); the Rust
# listener stream yields these as the event_type int alongside each event's JSON.
_QUERY_PROGRESS_EVENT = 1
_QUERY_TERMINATED_EVENT = 2
_QUERY_IDLE_EVENT = 3

# Re-export the core classes from the Rust extension
StreamingQuery = _StreamingQuery
StreamingQueryStatus = _StreamingQueryStatus
StreamingQueryException = _StreamingQueryException


def _dispatch_listener_event(listener, event_type, event_json):
    """Deserialize a listener event and invoke the matching callback on ``listener``.

    Called from the native (Rust) listener bus for each event. Mirrors the reference
    ``StreamingQueryListenerBus`` dispatch: build the typed event and route it to
    onQueryProgress / onQueryIdle / onQueryTerminated. Callback exceptions are surfaced
    as warnings so one listener cannot break the dispatch thread.
    """
    try:
        j = json.loads(event_json)
        if event_type == _QUERY_PROGRESS_EVENT:
            listener.onQueryProgress(QueryProgressEvent.fromJson(j))
        elif event_type == _QUERY_IDLE_EVENT:
            listener.onQueryIdle(QueryIdleEvent.fromJson(j))
        elif event_type == _QUERY_TERMINATED_EVENT:
            listener.onQueryTerminated(QueryTerminatedEvent.fromJson(j))
    except Exception as e:  # noqa: BLE001
        warnings.warn(f"Listener callback raised exception: {e}")


# ``spark.streams`` returns the native (Rust) StreamingQueryManager, which owns the
# client-side listener bus: addListener / removeListener / close are implemented in the
# Rust core (so Rust clients get the listener feature too). Re-export it as the public
# class.
StreamingQueryManager = _StreamingQueryManager


__all__ = [
    "StreamingQuery",
    "StreamingQueryManager",
    "StreamingQueryStatus",
    "StreamingQueryException",
]
