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
"""The ``__reduce__`` of the stream writers and the session builder of the drop-in, of their
state. The readers and the writers are PySpark's classes, which pickle as PySpark's plain objects.

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


def _state_reduce(
    drop_in: str, worker: Callable[..., Any], args: tuple[Any, ...], cls: Any = None
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """The reduce of an object of the drop-in class ``drop_in`` of ``args``, its state: the
    function, which cloudpickle pickles by value, gives back the drop-in's object of the state
    through its ``_of_state``, or, on a worker without ``pyspark._pyspark``, what ``worker``
    builds of the state with PySpark's own objects. An object of ``cls``, a Python subclass of
    the drop-in class, is given back of ``cls``, which the reduce names, as a pickle of PySpark's
    object of a subclass is of that subclass: on such a worker, ``cls`` is the subclass of
    PySpark's class, whose object PySpark's pickle makes without its ``__init__``, as here."""
    if cls is not None:

        def creator(cls: Any, *state: Any) -> Any:
            try:
                from pyspark import _pyspark  # noqa: F401
            except ImportError:
                obj = worker(*state)
                if isinstance(cls, type) and issubclass(cls, type(obj)):
                    obj.__class__ = cls
                return obj
            return cls._of_state(*state)

        creator.__qualname__ = "SparkSession.__reduce__.<locals>.creator"
        return creator, (cls, *args)

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


def _worker_foreach_commands(writer: Any, commands: list[tuple[str, bytes, str]]) -> Any:
    """PySpark's DataStreamWriter ``writer`` with the pickles of the functions of
    ``foreachBatch`` and ``foreach``, each of its kind and of the version of Python that pickled
    it, in the fields where PySpark's ``foreachBatch`` and ``foreach`` set them, as an unpickled
    writer of PySpark keeps them: PySpark's writer keeps the pickle of a function, and not the
    function."""
    for kind, command, python_ver in commands:
        proto = writer._write_proto
        sink = proto.foreach_batch if kind == "foreachBatch" else proto.foreach_writer
        sink.python_function.command = command
        sink.python_function.python_ver = python_ver
    return writer


def _worker_stream_writer(
    frame: Any,
    source: Any,
    output_mode: Optional[str],
    options: dict[str, str],
    partition_cols: list[str],
    cluster_cols: list[str],
    query_name: Optional[str],
    trigger: Optional[tuple[str, Optional[str]]],
    commands: list[tuple[str, bytes, str]],
) -> Any:
    """PySpark's DataStreamWriter of the DataFrame ``frame``, which replays the calls of the
    drop-in's writer of that state, with the pickles of the functions of ``foreachBatch`` and
    ``foreach`` that it keeps (``_worker_foreach_commands``)."""
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
    if commands:
        writer = _worker_foreach_commands(writer, commands)
    return writer


class _PlanOfDataFrame:
    """The DataFrame ``df``, which a pickle keeps as PySpark's pickle keeps the plan of a
    DataFrame of an object that holds its plan and not the DataFrame, as PySpark's
    DataStreamWriter does: it unpickles as the DataFrame, without the request of the schema of the
    unpickle of a DataFrame (``_reduce_dataframe``)."""

    def __init__(self, df: Any) -> None:
        self.df = df

    def __reduce__(self) -> tuple[Callable[..., Any], tuple[Any, ...]]:
        creator, args = self.df.__reduce__()
        return creator, (*args[:-1], False)


def stream_writer_reduce(
    frame: Any,
    format: Optional[str],
    output_mode: Optional[str],
    options: dict[str, str],
    partition_cols: list[str],
    cluster_cols: list[str],
    query_name: Optional[str],
    trigger: Optional[tuple[str, Optional[str]]],
    commands: list[tuple[str, bytes, str]],
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """The reduce of a DataStreamWriter of ``frame``, of the state that ``_of_state`` takes, which
    holds the plan of ``frame``, as PySpark's writer holds the plan and not the DataFrame, and the
    pickles of its functions, as PySpark's writer holds them and not the functions."""
    return _state_reduce(
        "DataStreamWriter",
        _worker_stream_writer,
        (
            _PlanOfDataFrame(frame),
            format,
            output_mode,
            options,
            partition_cols,
            cluster_cols,
            query_name,
            trigger,
            commands,
        ),
    )
