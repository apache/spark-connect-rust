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
Streaming Query Listener - defines listener interface and event classes
for streaming query lifecycle management.

This module vendors classes from pyspark.sql.streaming.listener for use
in the Connect streaming client.
"""

import json
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set, TYPE_CHECKING
from pyspark.sql import Row

if TYPE_CHECKING:
    pass

__all__ = [
    "StreamingQueryListener",
    "QueryStartedEvent",
    "QueryProgressEvent",
    "QueryIdleEvent",
    "QueryTerminatedEvent",
    "StreamingQueryProgress",
    "StateOperatorProgress",
    "SourceProgress",
    "SinkProgress",
]


class StreamingQueryListener(ABC):
    """
    Interface for listening to events related to StreamingQuery.

    Notes
    -----
    The methods are not thread-safe as they may be called from different threads.
    The events received are identical with Scala API.

    Examples
    --------
    >>> class MyListener(StreamingQueryListener):
    ...    def onQueryStarted(self, event: QueryStartedEvent) -> None:
    ...        pass
    ...
    ...    def onQueryProgress(self, event: QueryProgressEvent) -> None:
    ...        pass
    ...
    ...    def onQueryIdle(self, event: QueryIdleEvent) -> None:
    ...        pass
    ...
    ...    def onQueryTerminated(self, event: QueryTerminatedEvent) -> None:
    ...        pass
    """

    def _init_listener_id(self) -> None:
        self._id = str(uuid.uuid4())

    @abstractmethod
    def onQueryStarted(self, event: "QueryStartedEvent") -> None:
        """Called when a query is started."""
        pass

    @abstractmethod
    def onQueryProgress(self, event: "QueryProgressEvent") -> None:
        """Called when there is some status update."""
        pass

    def onQueryIdle(self, event: "QueryIdleEvent") -> None:
        """Called when the query is idle and waiting for new data to process."""
        pass

    @abstractmethod
    def onQueryTerminated(self, event: "QueryTerminatedEvent") -> None:
        """Called when a query is stopped, with or without error."""
        pass


class QueryStartedEvent:
    """Event representing the start of a query."""

    def __init__(
        self,
        id: uuid.UUID,
        runId: uuid.UUID,
        name: Optional[str],
        timestamp: str,
        jobTags: Set[str],
    ) -> None:
        self._id: uuid.UUID = id
        self._runId: uuid.UUID = runId
        self._name: Optional[str] = name
        self._timestamp: str = timestamp
        self._jobTags: Set[str] = jobTags

    @classmethod
    def fromJson(cls, j: Dict[str, Any]) -> "QueryStartedEvent":
        job_tags = j.get("jobTags", [])
        return cls(
            id=uuid.UUID(j["id"]),
            runId=uuid.UUID(j["runId"]),
            name=j.get("name"),
            timestamp=j.get("timestamp", ""),
            jobTags=set(job_tags),
        )

    @property
    def id(self) -> uuid.UUID:
        """A unique query id that persists across restarts."""
        return self._id

    @property
    def runId(self) -> uuid.UUID:
        """A query id that is unique for every start/restart."""
        return self._runId

    @property
    def name(self) -> Optional[str]:
        """User-specified name of the query, `None` if not specified."""
        return self._name

    @property
    def timestamp(self) -> str:
        """The timestamp to start a query."""
        return self._timestamp

    @property
    def jobTags(self) -> Set[str]:
        """The job tags of the query."""
        return self._jobTags


class QueryProgressEvent:
    """Event representing any progress updates in a query."""

    def __init__(self, progress: "StreamingQueryProgress") -> None:
        self._progress: StreamingQueryProgress = progress

    @classmethod
    def fromJson(cls, j: Dict[str, Any]) -> "QueryProgressEvent":
        return cls(progress=StreamingQueryProgress.fromJson(j.get("progress", j)))

    @property
    def progress(self) -> "StreamingQueryProgress":
        """The query progress updates."""
        return self._progress


class QueryIdleEvent:
    """Event representing that query is idle and waiting for new data to process."""

    def __init__(self, id: uuid.UUID, runId: uuid.UUID, timestamp: str) -> None:
        self._id: uuid.UUID = id
        self._runId: uuid.UUID = runId
        self._timestamp: str = timestamp

    @classmethod
    def fromJson(cls, j: Dict[str, Any]) -> "QueryIdleEvent":
        return cls(
            id=uuid.UUID(j["id"]),
            runId=uuid.UUID(j["runId"]),
            timestamp=j.get("timestamp", ""),
        )

    @property
    def id(self) -> uuid.UUID:
        """A unique query id that persists across restarts."""
        return self._id

    @property
    def runId(self) -> uuid.UUID:
        """A query id that is unique for every start/restart."""
        return self._runId

    @property
    def timestamp(self) -> str:
        """The timestamp when the latest no-batch trigger happened."""
        return self._timestamp


class QueryTerminatedEvent:
    """Event representing that termination of a query."""

    def __init__(
        self,
        id: uuid.UUID,
        runId: uuid.UUID,
        exception: Optional[str],
        errorClassOnException: Optional[str],
    ) -> None:
        self._id: uuid.UUID = id
        self._runId: uuid.UUID = runId
        self._exception: Optional[str] = exception
        self._errorClassOnException: Optional[str] = errorClassOnException

    @classmethod
    def fromJson(cls, j: Dict[str, Any]) -> "QueryTerminatedEvent":
        return cls(
            id=uuid.UUID(j["id"]),
            runId=uuid.UUID(j["runId"]),
            exception=j.get("exception"),
            errorClassOnException=j.get("errorClassOnException"),
        )

    @property
    def id(self) -> uuid.UUID:
        """A unique query id that persists across restarts."""
        return self._id

    @property
    def runId(self) -> uuid.UUID:
        """A query id that is unique for every start/restart."""
        return self._runId

    @property
    def exception(self) -> Optional[str]:
        """The exception message if the query terminated with an exception."""
        return self._exception

    @property
    def errorClassOnException(self) -> Optional[str]:
        """The error class from the exception if applicable."""
        return self._errorClassOnException


class StateOperatorProgress(dict):
    """Progress information for state operators."""

    def __init__(
        self,
        operatorName: str,
        numRowsTotal: int,
        numRowsUpdated: int,
        numRowsRemoved: int,
        allUpdatesTimeMs: int,
        allRemovalsTimeMs: int,
        commitTimeMs: int,
        memoryUsedBytes: int,
        numRowsDroppedByWatermark: int,
        numShufflePartitions: int,
        numStateStoreInstances: int,
        customMetrics: Dict[str, int],
        jdict: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            operatorName=operatorName,
            numRowsTotal=numRowsTotal,
            numRowsUpdated=numRowsUpdated,
            numRowsRemoved=numRowsRemoved,
            allUpdatesTimeMs=allUpdatesTimeMs,
            allRemovalsTimeMs=allRemovalsTimeMs,
            commitTimeMs=commitTimeMs,
            memoryUsedBytes=memoryUsedBytes,
            numRowsDroppedByWatermark=numRowsDroppedByWatermark,
            numShufflePartitions=numShufflePartitions,
            numStateStoreInstances=numStateStoreInstances,
            customMetrics=customMetrics,
        )
        self._jdict: Optional[Dict[str, Any]] = jdict

    @classmethod
    def fromJson(cls, j: Dict[str, Any]) -> "StateOperatorProgress":
        return cls(
            jdict=j,
            operatorName=j.get("operatorName", ""),
            numRowsTotal=j.get("numRowsTotal", 0),
            numRowsUpdated=j.get("numRowsUpdated", 0),
            numRowsRemoved=j.get("numRowsRemoved", 0),
            allUpdatesTimeMs=j.get("allUpdatesTimeMs", 0),
            allRemovalsTimeMs=j.get("allRemovalsTimeMs", 0),
            commitTimeMs=j.get("commitTimeMs", 0),
            memoryUsedBytes=j.get("memoryUsedBytes", 0),
            numRowsDroppedByWatermark=j.get("numRowsDroppedByWatermark", 0),
            numShufflePartitions=j.get("numShufflePartitions", 0),
            numStateStoreInstances=j.get("numStateStoreInstances", 0),
            customMetrics=dict(j.get("customMetrics", {})),
        )

    @property
    def operatorName(self) -> str:
        return self["operatorName"]

    @property
    def numRowsTotal(self) -> int:
        return self["numRowsTotal"]

    @property
    def numRowsUpdated(self) -> int:
        return self["numRowsUpdated"]

    @property
    def allUpdatesTimeMs(self) -> int:
        return self["allUpdatesTimeMs"]

    @property
    def numRowsRemoved(self) -> int:
        return self["numRowsRemoved"]

    @property
    def allRemovalsTimeMs(self) -> int:
        return self["allRemovalsTimeMs"]

    @property
    def commitTimeMs(self) -> int:
        return self["commitTimeMs"]

    @property
    def memoryUsedBytes(self) -> int:
        return self["memoryUsedBytes"]

    @property
    def numRowsDroppedByWatermark(self) -> int:
        return self["numRowsDroppedByWatermark"]

    @property
    def numShufflePartitions(self) -> int:
        return self["numShufflePartitions"]

    @property
    def numStateStoreInstances(self) -> int:
        return self["numStateStoreInstances"]

    @property
    def customMetrics(self) -> dict:
        return self["customMetrics"]

    @property
    def json(self) -> str:
        """The compact JSON representation of this progress."""
        if self._jdict:
            return json.dumps(self._jdict)
        else:
            return json.dumps(dict(self))

    @property
    def prettyJson(self) -> str:
        """The pretty (indented) JSON representation of this progress."""
        if self._jdict:
            return json.dumps(self._jdict, indent=4)
        else:
            return json.dumps(dict(self), indent=4)

    def __str__(self) -> str:
        return self.prettyJson

    def __repr__(self) -> str:
        return self.prettyJson


class SourceProgress(dict):
    """Progress information for data sources."""

    def __init__(
        self,
        description: str,
        startOffset: str,
        endOffset: str,
        latestOffset: str,
        numInputRows: int,
        inputRowsPerSecond: float,
        processedRowsPerSecond: float,
        metrics: Dict[str, str],
        jdict: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            description=description,
            startOffset=startOffset,
            endOffset=endOffset,
            latestOffset=latestOffset,
            numInputRows=numInputRows,
            inputRowsPerSecond=inputRowsPerSecond,
            processedRowsPerSecond=processedRowsPerSecond,
            metrics=metrics,
        )
        self._jdict: Optional[Dict[str, Any]] = jdict

    @classmethod
    def fromJson(cls, j: Dict[str, Any]) -> "SourceProgress":
        def _to_json_string(value: Any) -> str:
            if isinstance(value, str):
                return value
            else:
                return json.dumps(value)

        return cls(
            jdict=j,
            description=j.get("description", ""),
            startOffset=_to_json_string(j.get("startOffset", "")),
            endOffset=_to_json_string(j.get("endOffset", "")),
            latestOffset=_to_json_string(j.get("latestOffset", "")),
            numInputRows=j.get("numInputRows", 0),
            inputRowsPerSecond=j.get("inputRowsPerSecond", 0.0),
            processedRowsPerSecond=j.get("processedRowsPerSecond", 0.0),
            metrics=dict(j.get("metrics", {})),
        )

    @property
    def description(self) -> str:
        """Description of the source."""
        return self["description"]

    @property
    def startOffset(self) -> str:
        """The starting offset for data being read."""
        return self["startOffset"]

    @property
    def endOffset(self) -> str:
        """The ending offset for data being read."""
        return self["endOffset"]

    @property
    def latestOffset(self) -> str:
        """The latest offset from this source."""
        return self["latestOffset"]

    @property
    def numInputRows(self) -> int:
        """The number of records read from this source."""
        return self["numInputRows"]

    @property
    def inputRowsPerSecond(self) -> float:
        """The rate at which data is arriving from this source."""
        return self["inputRowsPerSecond"]

    @property
    def processedRowsPerSecond(self) -> float:
        """The rate at which data from this source is being processed by Spark."""
        return self["processedRowsPerSecond"]

    @property
    def metrics(self) -> dict:
        return self["metrics"]

    @property
    def json(self) -> str:
        """The compact JSON representation of this progress."""
        if self._jdict:
            return json.dumps(self._jdict)
        else:
            return json.dumps(dict(self))

    @property
    def prettyJson(self) -> str:
        """The pretty (indented) JSON representation of this progress."""
        if self._jdict:
            return json.dumps(self._jdict, indent=4)
        else:
            return json.dumps(dict(self), indent=4)

    def __str__(self) -> str:
        return self.prettyJson

    def __repr__(self) -> str:
        return self.prettyJson


class SinkProgress(dict):
    """Progress information for sinks."""

    def __init__(
        self,
        description: str,
        numOutputRows: int,
        metrics: Dict[str, str],
        jdict: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            description=description,
            numOutputRows=numOutputRows,
            metrics=metrics,
        )
        self._jdict: Optional[Dict[str, Any]] = jdict

    @classmethod
    def fromJson(cls, j: Dict[str, Any]) -> "SinkProgress":
        return cls(
            jdict=j,
            description=j.get("description", ""),
            numOutputRows=j.get("numOutputRows", 0),
            metrics=dict(j.get("metrics", {})),
        )

    @property
    def description(self) -> str:
        """Description of the sink."""
        return self["description"]

    @property
    def numOutputRows(self) -> int:
        """Number of rows written to the sink."""
        return self["numOutputRows"]

    @property
    def metrics(self) -> Dict[str, str]:
        return self["metrics"]

    @property
    def json(self) -> str:
        """The compact JSON representation of this progress."""
        if self._jdict:
            return json.dumps(self._jdict)
        else:
            return json.dumps(dict(self))

    @property
    def prettyJson(self) -> str:
        """The pretty (indented) JSON representation of this progress."""
        if self._jdict:
            return json.dumps(self._jdict, indent=4)
        else:
            return json.dumps(dict(self), indent=4)

    def __str__(self) -> str:
        return self.prettyJson

    def __repr__(self) -> str:
        return self.prettyJson


class StreamingQueryProgress(dict):
    """Progress information for a streaming query."""

    def __init__(
        self,
        id: uuid.UUID,
        runId: uuid.UUID,
        name: Optional[str],
        timestamp: str,
        batchId: int,
        batchDuration: int,
        durationMs: Dict[str, int],
        eventTime: Dict[str, str],
        stateOperators: List[StateOperatorProgress],
        sources: List[SourceProgress],
        sink: SinkProgress,
        numInputRows: Optional[int],
        inputRowsPerSecond: Optional[float],
        processedRowsPerSecond: Optional[float],
        observedMetrics: Dict[str, Row],
        jdict: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            id=id,
            runId=runId,
            name=name,
            timestamp=timestamp,
            batchId=batchId,
            batchDuration=batchDuration,
            durationMs=durationMs,
            eventTime=eventTime,
            stateOperators=stateOperators,
            sources=sources,
            sink=sink,
            numInputRows=numInputRows,
            inputRowsPerSecond=inputRowsPerSecond,
            processedRowsPerSecond=processedRowsPerSecond,
            observedMetrics=observedMetrics,
        )
        self._jdict: Optional[Dict[str, Any]] = jdict

    @classmethod
    def fromJson(cls, j: Dict[str, Any]) -> "StreamingQueryProgress":
        return cls(
            jdict=j,
            id=uuid.UUID(j.get("id", "")),
            runId=uuid.UUID(j.get("runId", "")),
            name=j.get("name"),
            timestamp=j.get("timestamp", ""),
            batchId=j.get("batchId", 0),
            batchDuration=j.get("batchDuration", 0),
            durationMs=dict(j.get("durationMs", {})),
            eventTime=dict(j.get("eventTime", {})),
            stateOperators=[StateOperatorProgress.fromJson(s) for s in j.get("stateOperators", [])],
            sources=[SourceProgress.fromJson(s) for s in j.get("sources", [])],
            sink=SinkProgress.fromJson(j.get("sink", {})),
            numInputRows=j.get("numInputRows"),
            inputRowsPerSecond=j.get("inputRowsPerSecond"),
            processedRowsPerSecond=j.get("processedRowsPerSecond"),
            observedMetrics={
                k: Row(*row_dict.keys())(*row_dict.values())
                for k, row_dict in j.get("observedMetrics", {}).items()
            },
        )

    @property
    def id(self) -> uuid.UUID:
        """A unique query id that persists across restarts."""
        return super().__getitem__("id")

    @property
    def runId(self) -> uuid.UUID:
        """A query id that is unique for every start/restart."""
        return super().__getitem__("runId")

    @property
    def name(self) -> Optional[str]:
        """User-specified name of the query, `None` if not specified."""
        return self["name"]

    @property
    def timestamp(self) -> str:
        """The timestamp to start a query."""
        return self["timestamp"]

    @property
    def batchId(self) -> int:
        """The current batch id."""
        return self["batchId"]

    @property
    def batchDuration(self) -> int:
        """The duration of the batch in milliseconds."""
        return self["batchDuration"]

    @property
    def durationMs(self) -> Dict[str, int]:
        """The duration information."""
        return self["durationMs"]

    @property
    def eventTime(self) -> Dict[str, str]:
        """The event time information."""
        return self["eventTime"]

    @property
    def stateOperators(self) -> List[StateOperatorProgress]:
        """The state operator progress information."""
        return self["stateOperators"]

    @property
    def sources(self) -> List[SourceProgress]:
        """The source progress information."""
        return self["sources"]

    @property
    def sink(self) -> SinkProgress:
        """The sink progress information."""
        return self["sink"]

    @property
    def numInputRows(self) -> Optional[int]:
        """The number of input rows."""
        return self["numInputRows"]

    @property
    def inputRowsPerSecond(self) -> Optional[float]:
        """The input rows per second."""
        return self["inputRowsPerSecond"]

    @property
    def processedRowsPerSecond(self) -> Optional[float]:
        """The processed rows per second."""
        return self["processedRowsPerSecond"]

    @property
    def observedMetrics(self) -> Dict[str, Row]:
        """The observed metrics."""
        return self["observedMetrics"]

    @property
    def json(self) -> str:
        """The compact JSON representation of this progress."""
        if self._jdict:
            return json.dumps(self._jdict)
        else:
            return json.dumps(dict(self))

    @property
    def prettyJson(self) -> str:
        """The pretty (indented) JSON representation of this progress."""
        if self._jdict:
            return json.dumps(self._jdict, indent=4)
        else:
            return json.dumps(dict(self), indent=4)

    def __str__(self) -> str:
        return self.prettyJson

    def __repr__(self) -> str:
        return self.prettyJson
