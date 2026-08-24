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
    from pyspark.sql.connect.session import SparkSession

# Re-export the core classes from the Rust extension
StreamingQuery = _StreamingQuery
StreamingQueryStatus = _StreamingQueryStatus
StreamingQueryException = _StreamingQueryException


class StreamingQueryListenerBus:
    """
    A client side listener bus that is responsible for buffering client side listeners,
    receiving listener events and invoking correct listener callbacks.
    """

    def __init__(self, sqm: "StreamingQueryManager") -> None:
        self._sqm = sqm
        self._listener_bus: List[StreamingQueryListener] = []
        self._execution_thread: Optional[Thread] = None
        self._lock = Lock()

    def close(self) -> None:
        """Close all listeners."""
        listeners_copy = list(self._listener_bus)
        for listener in listeners_copy:
            self.remove(listener)

    def append(self, listener: StreamingQueryListener) -> None:
        """
        Append a listener to the local listener bus. When the added listener is
        the first listener, request the server to start streaming listener events.
        """
        with self._lock:
            self._listener_bus.append(listener)

            if len(self._listener_bus) == 1:
                assert self._execution_thread is None
                try:
                    # Start the listener event thread
                    self._execution_thread = Thread(
                        target=self._query_event_handler,
                        daemon=True,
                    )
                    self._execution_thread.start()
                except Exception as e:
                    warnings.warn(
                        f"Failed to add the listener because of exception: {e}\n"
                        f"The listener is not added, please add it again."
                    )
                    self._listener_bus.remove(listener)
                    return

    def remove(self, listener: StreamingQueryListener) -> None:
        """
        Remove the listener from the local listener bus.

        When the listener is the last listener, request the server to stop
        streaming listener events. This function blocks until all events are processed.
        """
        with self._lock:
            if listener not in self._listener_bus:
                return

            self._listener_bus.remove(listener)

            if len(self._listener_bus) == 0 and self._execution_thread is not None:
                # Stop the execution thread
                if self._execution_thread.is_alive():
                    self._execution_thread.join(timeout=5)
                self._execution_thread = None

    def _query_event_handler(self) -> None:
        """
        Handler function passed to the thread. Receives listener events from the server
        and dispatches them to registered listeners.
        """
        try:
            # Stream events from the server
            events = self._sqm.inner.streamListenerEvents()
            for event_type, event_json in events:
                with self._lock:
                    if not self._listener_bus:
                        break
                deserialized_event = self.deserialize(event_type, event_json)
                self.post_to_all(deserialized_event)
        except Exception as e:
            warnings.warn(
                "StreamingQueryListenerBus Handler thread received exception, all client side "
                f"listeners are removed and handler thread is terminated. The error is: {e}"
            )
            with self._lock:
                self._execution_thread = None
                self._listener_bus.clear()
            return

    @staticmethod
    def deserialize(event_type: int, event_json: str) -> Union[
        QueryProgressEvent, QueryIdleEvent, QueryTerminatedEvent
    ]:
        """Deserialize a listener event from JSON."""
        from pyspark.sql.connect import proto

        j = json.loads(event_json)
        if event_type == proto.StreamingQueryEventType.QUERY_PROGRESS_EVENT:
            return QueryProgressEvent.fromJson(j)
        elif event_type == proto.StreamingQueryEventType.QUERY_TERMINATED_EVENT:
            return QueryTerminatedEvent.fromJson(j)
        elif event_type == proto.StreamingQueryEventType.QUERY_IDLE_EVENT:
            return QueryIdleEvent.fromJson(j)
        else:
            raise ValueError(f"Unknown event type: {event_type}")

    def post_to_all(
        self,
        event: Union[QueryProgressEvent, QueryIdleEvent, QueryTerminatedEvent],
    ) -> None:
        """Post an event to all registered listeners."""
        with self._lock:
            listeners = list(self._listener_bus)

        for listener in listeners:
            try:
                if isinstance(event, QueryProgressEvent):
                    listener.onQueryProgress(event)
                elif isinstance(event, QueryIdleEvent):
                    listener.onQueryIdle(event)
                elif isinstance(event, QueryTerminatedEvent):
                    listener.onQueryTerminated(event)
            except Exception as e:
                warnings.warn(f"Listener callback raised exception: {e}")


class StreamingQueryManager(_StreamingQueryManager):
    """Wrapper around the Rust StreamingQueryManager with Python-side listener bus."""

    def __init__(self, inner: _StreamingQueryManager) -> None:
        self.inner = inner
        self._sqlb = StreamingQueryListenerBus(self)

    def close(self) -> None:
        """Close the listener bus."""
        self._sqlb.close()

    def addListener(self, listener: StreamingQueryListener) -> None:
        """
        Add a listener to be notified of streaming query events.

        Parameters
        ----------
        listener : StreamingQueryListener
            The listener to add.
        """
        listener._init_listener_id()
        self._sqlb.append(listener)

    def removeListener(self, listener: StreamingQueryListener) -> None:
        """
        Remove a listener from receiving streaming query events.

        Parameters
        ----------
        listener : StreamingQueryListener
            The listener to remove.
        """
        self._sqlb.remove(listener)


__all__ = [
    "StreamingQuery",
    "StreamingQueryManager",
    "StreamingQueryStatus",
    "StreamingQueryException",
]
