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
"""Rust-backed DataFrame under the connect import path (testing/connectutils imports from here)."""
from collections.abc import Iterator
from typing import Any, Callable, Optional

from pyspark import cloudpickle as _cloudpickle

# The native Connect classes, as PySpark Connect's: virtual subclasses of the classic classes of
# pyspark.sql.dataframe (pyspark.sql._classic_bases).
from pyspark._pyspark import DataFrame, DataFrameNaFunctions, DataFrameStatFunctions
from pyspark.sql.connect import _column_reduce

__all__ = ["DataFrame", "DataFrameNaFunctions", "DataFrameStatFunctions"]

_cloudpickle.register_pickle_by_value(_column_reduce)


def _local_rows(rows: Any) -> Iterator[Any]:
    """The generator that ``DataFrame.toLocalIterator`` returns, as PySpark's: the first row
    starts the execution of the rows, and a close, a throw, an error or the end of the generator
    ends it."""
    try:
        yield from rows
    finally:
        rows.close()


_local_rows.__name__ = "toLocalIterator"
_local_rows.__qualname__ = "DataFrame.toLocalIterator"
# And the name of its code, as PySpark's generator, which tracebacks and `inspect` show.
_local_rows.__code__ = _local_rows.__code__.replace(
    co_name="toLocalIterator",
    **(
        {"co_qualname": "DataFrame.toLocalIterator"}
        if hasattr(_local_rows.__code__, "co_qualname")
        else {}
    ),
)


def _reduce_dataframe(
    session: Any,
    relation: bytes,
    levels: int = 1,
    support_repr_html: bool = False,
    cached_schema: Any = None,
    build_error: Any = None,
    holes: Optional[list[tuple[bytes, str, Any]]] = None,
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """The ``__reduce__`` of a DataFrame of ``session`` and the bytes of its ``relation`` of
    ``levels`` levels, as PySpark Connect's, of its plan, its session and its state: the function
    ``_column_reduce.dataframe_of``, which cloudpickle pickles by value and ``pickle`` by
    reference, of the objects of the functions of its UDFs (``holes``), which unpickle first, as
    PySpark's plan, of the session, of the relation, whose holes the requests fill with the
    commands of those functions, as PySpark's unpickled plan pickles them, and of
    ``support_repr_html`` and ``cached_schema``, the ``_support_repr_html`` and the
    ``_cached_schema`` of the state, which the DataFrame that it gives back keeps. A plan that did
    not build is pickled of ``build_error``, the description of its error, which the unpickle
    raises. So a plain pickle fails at the first function that it does not pickle, such as a
    lambda, and else at the session, of a local function, as PySpark's pickle fails at its plan
    and then at its session. The last of the arguments is whether the unpickle requests the
    schema, which ``_state_reduce._PlanOfDataFrame`` replaces; the one before it, of holes, the
    function that fills them on a Python worker of the server (``_column_reduce.fill_holes``), so
    that only the pickle of a DataFrame of holes carries its code."""
    return _column_reduce.dataframe_of, (
        list(holes or []),
        session,
        relation,
        levels,
        support_repr_html,
        cached_schema,
        build_error,
        _column_reduce.fill_holes if holes else None,
        True,
    )
