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
Streaming data readers and writers.

Re-exports the PyO3-implemented streaming classes. `DataStreamWriter.foreachBatch`
and `.foreach` accept a Python callable and cloudpickle it inside the extension (with
``pyspark.serializers.CloudPickleSerializer``), so no Python subclass of the (non-
subclassable) PyO3 class is needed; `.foreach` sends the function of
`_construct_foreach_function`, as PySpark does.
"""

from collections.abc import Iterator
from typing import Any, Callable, Union, cast

from pyspark._pyspark import (
    DataStreamReader as _DataStreamReader,
)
from pyspark._pyspark import (
    DataStreamWriter as _DataStreamWriter,
)
from pyspark._pyspark import (
    Trigger as _Trigger,
)
from pyspark.errors import PySparkAttributeError, PySparkRuntimeError

DataStreamReader = _DataStreamReader
DataStreamWriter = _DataStreamWriter
Trigger = _Trigger


def _construct_foreach_function(
    f: Union[Callable[[Any], None], Any],
) -> Callable[[Any, Iterator], Iterator]:
    """
    The function of the partition and the rows of each partition that ``foreach(f)`` sends,
    which the Python worker of the server runs, as PySpark's
    ``DataStreamWriter._construct_foreach_function``: for a function ``f``, one that calls ``f``
    with each row, and for an object, one that calls its ``open(partition_id, epoch_id)``, its
    ``process(row)`` with each row, and its ``close(error)``.

    Ported from PySpark, but for ``TaskContext``, which the function imports on the worker.
    """
    if callable(f):
        # The provided object is a callable function that is supposed to be called on each row.
        # Construct a function that takes an iterator and calls the provided function on each
        # row.
        def func_without_process(_: Any, iterator: Iterator) -> Iterator:
            for x in iterator:
                f(x)
            return iter([])

        return func_without_process

    else:
        # The provided object is not a callable function. Then it is expected to have a
        # 'process(row)' method, and optional 'open(partition_id, epoch_id)' and
        # 'close(error)' methods.

        if not hasattr(f, "process"):
            raise PySparkAttributeError(
                errorClass="ATTRIBUTE_NOT_CALLABLE",
                messageParameters={"attr_name": "process", "obj_name": "f"},
            )

        if not callable(f.process):
            raise PySparkAttributeError(
                errorClass="ATTRIBUTE_NOT_CALLABLE",
                messageParameters={"attr_name": "process", "obj_name": "f"},
            )

        def doesMethodExist(method_name: str) -> bool:
            exists = hasattr(f, method_name)
            if exists and not callable(getattr(f, method_name)):
                raise PySparkAttributeError(
                    errorClass="ATTRIBUTE_NOT_CALLABLE",
                    messageParameters={"attr_name": method_name, "obj_name": "f"},
                )
            return exists

        open_exists = doesMethodExist("open")
        close_exists = doesMethodExist("close")

        def func_with_open_process_close(partition_id: Any, iterator: Iterator) -> Iterator:
            from pyspark.taskcontext import TaskContext

            epoch_id = cast(TaskContext, TaskContext.get()).getLocalProperty(
                "streaming.sql.batchId"
            )
            if epoch_id:
                int_epoch_id = int(epoch_id)
            else:
                raise PySparkRuntimeError(
                    errorClass="CANNOT_GET_BATCH_ID",
                    messageParameters={"obj_name": "TaskContext"},
                )

            # Check if the data should be processed
            should_process = True
            if open_exists:
                should_process = f.open(partition_id, int_epoch_id)

            error = None

            try:
                if should_process:
                    for x in iterator:
                        f.process(x)
            except Exception as ex:
                error = ex
            finally:
                if close_exists:
                    f.close(error)
                if error:
                    raise error

            return iter([])

        return func_with_open_process_close

__all__ = ["DataStreamReader", "DataStreamWriter", "Trigger"]
