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
"""The ``__reduce__`` of the readers, the writers, the grouped data and the stream readers and
writers of the drop-in, of their state.

Each ``*_reduce`` function gives the reduce of an object of its state: a function, which
cloudpickle pickles by value, and the state. On the drop-in, the function gives back the
drop-in's object, as its ``_of_state`` does. On a Python worker of the server, whose PySpark has
no drop-in (such as the worker of a ``foreachBatch`` function), it gives back PySpark Connect's
object of the same state, built through PySpark's own calls, so that everything it calls there
is PySpark's.

This module registers itself with cloudpickle to pickle by value, so that its functions unpickle
on that worker, whichever shim was imported first.
"""
import sys
from typing import Any, Callable, Optional

from pyspark import cloudpickle as _cloudpickle

_cloudpickle.register_pickle_by_value(sys.modules[__name__])

# The names of the groupings of the drop-in, as PySpark's GroupedData takes them.
_GROUP_TYPES = {
    "groupBy": "groupby",
    "rollup": "rollup",
    "cube": "cube",
    "pivot": "pivot",
    "groupingSets": "grouping_sets",
}


def _state_reduce(
    drop_in: str, worker: Callable[..., Any], args: tuple[Any, ...]
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """The reduce of an object of the drop-in class ``drop_in`` of ``args``, its state: the
    function, which cloudpickle pickles by value, gives back the drop-in's object of the state
    through its ``_of_state``, or, on a worker without ``pyspark._pyspark``, what ``worker``
    builds of the state with PySpark's own objects."""

    def creator(*state: Any) -> Any:
        try:
            from pyspark import _pyspark
        except ImportError:
            return worker(*state)
        return getattr(_pyspark, drop_in)._of_state(*state)

    # Named as PySpark's creator of a session, which a pickle of these objects fails at, so that
    # the error of a plain pickle names the same function.
    creator.__qualname__ = "SparkSession.__reduce__.<locals>.creator"
    return creator, args


def _worker_reader(
    session: Any,
    format: Optional[str],
    schema: str,
    options: dict[str, str],
    unsent: Any,
) -> Any:
    """PySpark's DataFrameReader of ``session`` with the format, the schema and the options, as
    the drop-in's reader of that state. A format that is kept (``unsent``) is given to PySpark
    as it is, which fails it at the load, as the drop-in's reader does."""
    reader = session.read
    source = format if unsent is None else unsent
    if source is not None:
        reader = reader.format(source)
    if schema:
        reader = reader.schema(schema)
    if options:
        reader = reader.options(**options)
    return reader


def reader_reduce(
    session: Any,
    format: Optional[str],
    schema: str,
    options: dict[str, str],
    unsent: Any,
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """The reduce of a DataFrameReader of ``session``, of the state that ``_of_state`` takes."""
    return _state_reduce(
        "DataFrameReader",
        _worker_reader,
        (session, format, schema, options, unsent),
    )


def _worker_writer(
    frame: Any,
    format: Optional[str],
    mode: str,
    options: dict[str, str],
    partition_cols: list[str],
    num_buckets: Optional[int],
    bucket_cols: list[str],
    sort_cols: list[str],
    cluster_cols: list[str],
    refused: Any,
    kept_format: Any,
    kept_partition: Any,
    kept_clustering: Any,
    kept_mode: Any,
) -> Any:
    """PySpark's DataFrameWriter of the DataFrame ``frame``, which replays the calls of the
    drop-in's writer of that state: the format, the save mode, the options, the partitioning,
    the buckets, the sorting and the clustering. A bucket count that the proto refuses is
    replayed as it was given, which PySpark fails at each write. The kept arguments are given to
    PySpark as the user gave them, so that PySpark converts them, or fails, at the write: a kept
    partitioning or clustering column comes after the others, and fails the write in the same
    way."""
    writer = frame.write
    source = format if kept_format is None else kept_format
    if source is not None:
        writer = writer.format(source)
    writer = writer.mode(mode if kept_mode is None else kept_mode).options(**options)
    if partition_cols or kept_partition is not None:
        cols = [*partition_cols, *([] if kept_partition is None else [kept_partition])]
        writer = writer.partitionBy(*cols)
    if num_buckets is not None:
        writer = writer.bucketBy(num_buckets, *bucket_cols)
    elif refused is not None:
        writer = writer.bucketBy(refused, *bucket_cols)
    if sort_cols:
        writer = writer.sortBy(*sort_cols)
    if cluster_cols or kept_clustering is not None:
        cols = [*cluster_cols, *([] if kept_clustering is None else [kept_clustering])]
        writer = writer.clusterBy(*cols)
    return writer


def writer_reduce(
    frame: Any,
    format: Optional[str],
    mode: str,
    options: dict[str, str],
    partition_cols: list[str],
    num_buckets: Optional[int],
    bucket_cols: list[str],
    sort_cols: list[str],
    cluster_cols: list[str],
    refused: Any,
    kept_format: Any,
    kept_partition: Any,
    kept_clustering: Any,
    kept_mode: Any,
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """The reduce of a DataFrameWriter of ``frame``, of the state that ``_of_state`` takes."""
    return _state_reduce(
        "DataFrameWriter",
        _worker_writer,
        (
            frame,
            format,
            mode,
            options,
            partition_cols,
            num_buckets,
            bucket_cols,
            sort_cols,
            cluster_cols,
            refused,
            kept_format,
            kept_partition,
            kept_clustering,
            kept_mode,
        ),
    )


def _worker_writer_v2(
    frame: Any,
    table: str,
    provider: Optional[str],
    options: dict[str, str],
    table_properties: dict[str, str],
    partition_cols: list[Any],
    cluster_cols: list[str],
    unsent_table: Any,
    kept_provider: Any,
    kept_clustering: Any,
) -> Any:
    """PySpark's DataFrameWriterV2 of the DataFrame ``frame`` and the table ``table``, with the
    provider, the options, the table properties, the partitioning and the clustering. The kept
    table, provider and clustering column are given to PySpark as the user gave them, as the
    writer's kept format is."""
    writer = frame.writeTo(table if unsent_table is None else unsent_table)
    provider = provider if kept_provider is None else kept_provider
    if provider is not None:
        writer = writer.using(provider)
    writer = writer.options(**options)
    for name, value in table_properties.items():
        writer = writer.tableProperty(name, value)
    if partition_cols:
        writer = writer.partitionedBy(*partition_cols)
    if cluster_cols or kept_clustering is not None:
        cols = [*cluster_cols, *([] if kept_clustering is None else [kept_clustering])]
        writer = writer.clusterBy(*cols)
    return writer


def writer_v2_reduce(
    frame: Any,
    table: str,
    provider: Optional[str],
    options: dict[str, str],
    table_properties: dict[str, str],
    partition_cols: list[Any],
    cluster_cols: list[str],
    unsent_table: Any,
    kept_provider: Any,
    kept_clustering: Any,
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """The reduce of a DataFrameWriterV2 of ``frame``, of the state that ``_of_state`` takes."""
    return _state_reduce(
        "DataFrameWriterV2",
        _worker_writer_v2,
        (
            frame,
            table,
            provider,
            options,
            table_properties,
            partition_cols,
            cluster_cols,
            unsent_table,
            kept_provider,
            kept_clustering,
        ),
    )


def _worker_grouped(  # pragma: no cover - only the server's worker runs it
    frame: Any,
    group_type: str,
    cols: list[Any],
    pivot_col: Any,
    pivot_values: list[Any],
    grouping_sets: list[list[Any]],
) -> Any:
    """PySpark's GroupedData of ``frame`` through its constructor, which takes the same state.
    PySpark's pivot values are None when none were given, not an empty list."""
    from pyspark.sql.connect.group import GroupedData

    return GroupedData(
        frame,
        _GROUP_TYPES[group_type],
        list(cols),
        pivot_col,
        list(pivot_values) if pivot_values else None,
        grouping_sets if grouping_sets else None,
    )


def grouped_reduce(
    frame: Any,
    group_type: str,
    cols: list[Any],
    pivot_col: Any,
    pivot_values: list[Any],
    grouping_sets: list[list[Any]],
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """The reduce of GroupedData of ``frame``, of the state that ``_of_state`` takes."""
    return _state_reduce(
        "GroupedData",
        _worker_grouped,
        (frame, group_type, cols, pivot_col, pivot_values, grouping_sets),
    )


def _worker_cogrouped(  # pragma: no cover - only the server's worker runs it
    left: Any, right: Any
) -> Any:
    """PySpark's PandasCogroupedOps of the two grouped data ``left`` and ``right``."""
    from pyspark.sql.connect.group import PandasCogroupedOps

    return PandasCogroupedOps(left, right)


def cogrouped_reduce(left: Any, right: Any) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """The reduce of the cogrouped ops of the grouped data ``left`` and ``right``."""
    return _state_reduce("PandasCogroupedOps", _worker_cogrouped, (left, right))


def _worker_stream_reader(
    session: Any,
    format: Optional[str],
    schema: str,
    options: dict[str, str],
    source_name: Optional[str],
    unsent: Any,
) -> Any:
    """PySpark's DataStreamReader of ``session`` with the format, the schema, the options and the
    source name, as the drop-in's reader of that state. A format that is kept (``unsent``) is
    given to PySpark as it is."""
    reader = session.readStream
    source = format if unsent is None else unsent
    if source is not None:
        reader = reader.format(source)
    if schema:
        reader = reader.schema(schema)
    if options:
        reader = reader.options(**options)
    if source_name is not None:
        reader = reader.name(source_name)
    return reader


def stream_reader_reduce(
    session: Any,
    format: Optional[str],
    schema: str,
    options: dict[str, str],
    source_name: Optional[str],
    unsent: Any,
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """The reduce of a DataStreamReader of ``session``, of the state that ``_of_state`` takes."""
    return _state_reduce(
        "DataStreamReader",
        _worker_stream_reader,
        (session, format, schema, options, source_name, unsent),
    )


def _worker_stream_writer(
    frame: Any,
    source: Any,
    output_mode: Optional[str],
    options: dict[str, str],
    partition_cols: list[str],
    cluster_cols: list[str],
    query_name: Optional[str],
    trigger: Optional[tuple[str, Optional[str]]],
    functions: list[tuple[str, Any]],
) -> Any:
    """PySpark's DataStreamWriter of the DataFrame ``frame``, which replays the calls of the
    drop-in's writer of that state, the functions that it applied last."""
    writer = frame.writeStream
    if source is not None:
        writer = writer.format(source)
    if output_mode is not None:
        writer = writer.outputMode(output_mode)
    writer = writer.options(**options)
    if partition_cols:
        writer = writer.partitionBy(*partition_cols)
    if cluster_cols:
        writer = writer.clusterBy(*cluster_cols)
    if query_name is not None:
        writer = writer.queryName(query_name)
    if trigger is not None:
        kind, value = trigger
        writer = writer.trigger(**{kind: True if value is None else value})
    for kind, function in functions:
        if kind == "foreachBatch":
            writer = writer.foreachBatch(function)
        else:
            writer = writer.foreach(function)
    return writer


def stream_writer_reduce(
    frame: Any,
    format: Optional[str],
    output_mode: Optional[str],
    options: dict[str, str],
    partition_cols: list[str],
    cluster_cols: list[str],
    query_name: Optional[str],
    trigger: Optional[tuple[str, Optional[str]]],
    functions: list[tuple[str, Any]],
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """The reduce of a DataStreamWriter of ``frame``, of the state that ``_of_state`` takes."""
    return _state_reduce(
        "DataStreamWriter",
        _worker_stream_writer,
        (
            frame,
            format,
            output_mode,
            options,
            partition_cols,
            cluster_cols,
            query_name,
            trigger,
            functions,
        ),
    )


def builder_reduce(
    options: list[tuple[str, Optional[str]]],
    channel_builder: Any,
    hook_factories: list[Any],
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """The reduce of a session Builder of its options, its channel builder and its hook
    factories, as PySpark's Builder keeps them in its ``__dict__``. The builder that the function
    gives back is the drop-in's, or PySpark Connect's on a Python worker, which is built through
    the same calls: ``config``, ``channelBuilder`` and ``_registerHook``."""

    def creator(
        options: list[tuple[str, Optional[str]]],
        channel_builder: Any,
        hook_factories: list[Any],
    ) -> Any:
        try:
            from pyspark import _pyspark

            builder = _pyspark.Builder()
        except ImportError:
            from pyspark.sql.connect.session import SparkSession

            builder = SparkSession.Builder()
        for key, value in options:
            builder.config(key, value)
        if channel_builder is not None:
            builder.channelBuilder(channel_builder)
        for hook_factory in hook_factories:
            builder._registerHook(hook_factory)
        return builder

    return creator, (options, channel_builder, hook_factories)


def _worker_streaming_query(  # pragma: no cover - only the server's worker runs it
    session: Any, query_id: str, run_id: str, name: Optional[str]
) -> Any:
    """PySpark's StreamingQuery of ``session``, of the ID of the query, of its run and of its
    name, as the drop-in's query of that state."""
    from pyspark.sql.connect.streaming.query import StreamingQuery

    return StreamingQuery(session, query_id, run_id, name)


def streaming_query_reduce(
    session: Any, query_id: str, run_id: str, name: Optional[str]
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """The reduce of a StreamingQuery of ``session``, of the state that ``_of_state`` takes."""
    return _state_reduce(
        "StreamingQuery",
        _worker_streaming_query,
        (session, query_id, run_id, name),
    )
