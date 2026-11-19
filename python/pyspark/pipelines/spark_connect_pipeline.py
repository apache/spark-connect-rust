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
"""Drive a Spark Declarative Pipeline against a Spark Connect server.

This mirrors upstream ``pyspark.pipelines.spark_connect_pipeline`` but, because this
drop-in uses a native Rust transport instead of the Python gRPC client, the
``PipelineCommand`` construction + execution live in the Rust core
(``pyspark._pyspark.pipeline_*``); these functions are thin wrappers over them.
"""
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator, Mapping, Optional, Sequence

from pyspark.sql import SparkSession
from pyspark.pipelines.logging_utils import log_with_provided_timestamp
from pyspark._pyspark import (
    pipeline_create_dataflow_graph as _create_dataflow_graph,
    pipeline_start_run as _start_run,
)


def create_dataflow_graph(
    spark: SparkSession,
    default_catalog: Optional[str],
    default_database: Optional[str],
    sql_conf: Optional[Mapping[str, str]],
) -> str:
    """Create a dataflow graph in the Spark Connect server.

    :returns: The ID of the created dataflow graph.
    """
    return _create_dataflow_graph(
        spark,
        default_catalog,
        default_database,
        dict(sql_conf) if sql_conf else None,
    )


# The epoch of the microseconds of the timestamps of the events of a run.
_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def handle_pipeline_events(iter: Iterator[Any]) -> None:
    """Print out the pipeline events received from the Spark Connect server.

    The Rust run stream yields ``(message, timestamp_micros)`` per pipeline event, the
    microseconds of the event's Timestamp since the epoch, of its nanoseconds truncated, and
    raises the mapped pyspark exception (e.g. ``AnalysisException``) if the run fails. The time
    of an event is exact, as PySpark's ``event.timestamp.ToDatetime()`` in UTC: a float of its
    seconds would round the microseconds of a time far from 1970, into the next second too.
    """
    for message, timestamp_micros in iter:
        dt = _EPOCH + timedelta(microseconds=timestamp_micros)
        log_with_provided_timestamp(message, dt)


def start_run(
    spark: SparkSession,
    dataflow_graph_id: str,
    full_refresh: Optional[Sequence[str]],
    full_refresh_all: bool,
    refresh: Optional[Sequence[str]],
    dry: bool,
    storage: str,
) -> Iterator[Any]:
    """Start a run of the dataflow graph in the Spark Connect server.

    :param spark: SparkSession.
    :param dataflow_graph_id: The ID of the dataflow graph to start.
    :param full_refresh: List of datasets to reset and recompute.
    :param full_refresh_all: Perform a full graph reset and recompute.
    :param refresh: List of datasets to update.
    :param dry: If true, the run will not actually execute any flows, but only validate the graph.
    :param storage: The storage location to store metadata such as streaming checkpoints.
    """
    return _start_run(
        spark,
        dataflow_graph_id,
        list(full_refresh) if full_refresh else None,
        full_refresh_all,
        list(refresh) if refresh else None,
        dry,
        storage,
    )
