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
from typing import Any, Callable

from pyspark import cloudpickle as _cloudpickle
from pyspark._pyspark import DataFrame
from pyspark.sql.connect import _column_reduce
from pyspark.sql.connect._column_reduce import parse_deep
from pyspark.sql.connect._column_reduce import raise_error as _raise_error
from pyspark.sql.dataframe import DataFrameNaFunctions, DataFrameStatFunctions

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
    request_schema: bool = True,
) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """The ``__reduce__`` of a DataFrame of ``session`` and the bytes of its ``relation`` of
    ``levels`` levels, as PySpark Connect's, of its session, its plan and its state: a function,
    which cloudpickle pickles by value, of the session, which unpickles first, of the relation,
    and of ``support_repr_html`` and ``cached_schema``, the ``_support_repr_html`` and the
    ``_cached_schema`` of the state, which the DataFrame that it gives back keeps.

    As PySpark's unpickle, which builds the plan of the new DataFrame and requests its schema, as
    pickle looks its ``__setstate__`` up through its ``__getattr__``, which reads its columns,
    the function requests the schema of the relation, whose error it raises, and then sets the
    ``_cached_schema`` of the state. A plan that did not build is pickled of ``build_error``, the
    description of its error, which it raises instead (``_column_reduce.raise_error``): but for
    an ``AttributeError``, of which PySpark's ``_schema`` falls back to the Column of the
    ``__getattr__`` of the DataFrame, whose ``fields``, a Column too, its columns iterate, which
    raises ``NOT_ITERABLE``. Without ``request_schema``, of an object that PySpark pickles of the
    plan of the DataFrame and not of the DataFrame, such as a DataStreamWriter, the schema is not
    requested.

    On a Python worker of the server, whose PySpark has no Rust client, such as the worker of a
    ``foreachBatch`` function, which has the active session of the query, the function gives
    back PySpark Connect's DataFrame of the relation, as PySpark's DataFrame of its plan unpickles
    there. So everything that it calls there is of PySpark, which the worker has."""

    def creator(
        session: Any,
        relation: bytes,
        levels: int,
        support_repr_html: bool,
        cached_schema: Any = None,
        build_error: Any = None,
        request_schema: bool = True,
    ) -> Any:
        if build_error is not None:
            if request_schema and build_error[0] in ("AttributeError", "PySparkAttributeError"):
                from pyspark.errors import PySparkTypeError

                raise PySparkTypeError(
                    errorClass="NOT_ITERABLE", messageParameters={"objectName": "Column"}
                )
            _raise_error(build_error)
        try:
            from pyspark import _pyspark
        except ImportError:
            from pyspark.sql.connect import proto
            from pyspark.sql.connect.dataframe import DataFrame as ConnectDataFrame
            from pyspark.sql.connect.plan import LogicalPlan

            # Without annotations: the class body looks its names up by name, which cloudpickle
            # does not pickle with the function.
            class Relation(LogicalPlan):
                """The plan of a relation as it was sent, of its plan ID, which the columns
                that refer to it by that ID resolve against."""

                def __init__(self, relation):  # type: ignore[no-untyped-def]
                    super().__init__(None)
                    self._relation = relation
                    if relation.common.HasField("plan_id"):
                        self._root_plan_id = relation.common.plan_id

                def plan(self, session):  # type: ignore[no-untyped-def]
                    sent = proto.Relation()
                    sent.CopyFrom(self._relation)
                    return sent

            df = ConnectDataFrame(Relation(parse_deep(proto.Relation, relation)), session)
            # Its columns, a request of its schema, and then the state, as PySpark's unpickle.
            if request_schema:
                _ = df.columns
            df._support_repr_html = support_repr_html
            df._cached_schema = cached_schema
            return df

        return _pyspark.DataFrame._of_relation(
            session, relation, levels, support_repr_html, cached_schema, request_schema
        )

    # Named as PySpark's creator of a session, which a pickle of a DataFrame fails at, so that the
    # error of a plain pickle names the same function.
    creator.__qualname__ = "SparkSession.__reduce__.<locals>.creator"
    args = (session, relation, levels, support_repr_html, cached_schema, build_error)
    return creator, (*args, request_schema)
