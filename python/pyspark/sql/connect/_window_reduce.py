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
"""The function that a pickled WindowSpec of the client unpickles with.

``pyspark.sql.connect.window`` registers this module with cloudpickle to pickle by value, so a
window that a UDF or a ``foreachBatch`` function captures unpickles on a Python worker of the
server, whose PySpark has no Rust client, while ``pickle`` pickles the function by reference, of
this module, on the client.
"""
from collections.abc import Sequence
from typing import Any, Optional


def window_of(
    partition: Sequence[Any],
    order: Sequence[tuple[Any, str]],
    frame: Optional[tuple[bool, int, int]],
) -> Any:
    """The WindowSpec of the Columns ``partition`` and of the sort orders ``order``, each a
    Column of its child and the name of the method of its direction and nulls, such as
    ``desc_nulls_last``, and of ``frame``, the ROWS flag and the start and end of its frame, if
    any, built through ``Window`` as PySpark's WindowSpec is, on the client and on a Python worker
    of the server."""
    from pyspark.sql.window import Window

    spec = Window.partitionBy(*partition).orderBy(
        *[getattr(child, method)() for child, method in order]
    )
    if frame is not None:
        is_rows, start, end = frame
        spec = spec.rowsBetween(start, end) if is_rows else spec.rangeBetween(start, end)
    return spec
