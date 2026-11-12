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
"""Rust-backed Column under the connect import path (pandas-on-Spark and connect tests import ConnectColumn from here)."""
from typing import Any, Callable, Optional

from pyspark import cloudpickle as _cloudpickle
from pyspark._pyspark import Column
from pyspark.sql.connect import _column_reduce

__all__ = ["Column"]

_cloudpickle.register_pickle_by_value(_column_reduce)


def _reduce_column(
    expression: bytes,
    rendered: str,
    branches: Optional[list[tuple[Any, Any]]] = None,
    else_value: Any = None,
    sort: Optional[tuple[Any, str]] = None,
    levels: int = 1,
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """The ``__reduce__`` of a Column of the bytes of its ``spark.connect.Expression`` and of its
    rendering, as PySpark Connect's Column pickles its expression: the function of
    ``pyspark.sql.connect._column_reduce``, which gives back the Column, which sends the
    expression as it is. A CASE WHEN gives its branches and else value too, as Columns, so the
    Column gives back a CASE WHEN. A sort order gives its child Column and its sort method, so
    the Column gives back a sort order. cloudpickle pickles that function by value, also to a
    Python worker of the server, where it gives back PySpark Connect's Column, and ``pickle`` by
    reference."""
    return _column_reduce.column_of, (expression, rendered, branches, else_value, sort, levels)
