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

from pyspark._pyspark import DataFrame
from pyspark.sql.dataframe import DataFrameNaFunctions, DataFrameStatFunctions

__all__ = ["DataFrame", "DataFrameNaFunctions", "DataFrameStatFunctions"]


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


def _reduce_dataframe(session: Any, relation: bytes) -> tuple[Callable[..., Any], tuple[Any, ...]]:
    """The ``__reduce__`` of a DataFrame of ``session`` and the bytes of its ``relation``, as
    PySpark Connect's, of its session and its plan: a function, which cloudpickle pickles by
    value, of the session, which unpickles first, and of the relation.

    On a Python worker of the server, whose PySpark has no Rust client, such as the worker of a
    ``foreachBatch`` function, which has the active session of the query, the function gives
    back PySpark Connect's DataFrame of the relation, as PySpark's DataFrame of its plan unpickles
    there. So everything that it calls there is of PySpark, which the worker has."""

    def creator(session: Any, relation: bytes) -> Any:
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

            return ConnectDataFrame(Relation(proto.Relation.FromString(relation)), session)

        return _pyspark.DataFrame._of_relation(session, relation)

    return creator, (session, relation)
