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
"""The function that a pickled Column of the client unpickles with.

``pyspark.sql.connect.column`` registers this module with cloudpickle to pickle by value, so a
Column that a UDF or a ``foreachBatch`` function captures unpickles on a Python worker of the
server, whose PySpark has no Rust client, while ``pickle`` pickles the function by reference, of
this module, on the client.
"""
from typing import Any, Optional


def column_of(
    expression: bytes,
    rendered: str,
    branches: Optional[list[tuple[Any, Any]]] = None,
    else_value: Any = None,
    sort: Optional[tuple[Any, str]] = None,
) -> Any:
    """The Column of ``expression``, the bytes of a ``spark.connect.Expression``, and of its
    rendering ``rendered``, which sends the expression as it is: of the Rust client, or, on a
    Python worker of the server, PySpark Connect's Column of the expression, as PySpark Connect's
    Column of its expression unpickles there, so everything that it calls there is of PySpark.
    For a CASE WHEN, ``branches`` are the (condition, value) Columns of its branches and
    ``else_value`` the Column of its else value, so that it is a CASE WHEN again. For a sort
    order, ``sort`` is its child Column and the name of its sort method, such as
    ``desc_nulls_last``, so that it is a sort order again."""
    if sort is not None:
        # The child is a Column of the client, or PySpark Connect's on a worker, whose sort
        # method gives back the sort order, as PySpark's SortOrder of its child.
        child, method = sort
        return getattr(child, method)()
    try:
        from pyspark import _pyspark
    except ImportError:
        from pyspark.sql.connect import proto
        from pyspark.sql.connect.column import Column as ConnectColumn
        from pyspark.sql.connect.expressions import CaseWhen, Expression

        if branches is not None:
            # As PySpark's CaseWhen of its branches and else value.
            return ConnectColumn(
                CaseWhen(
                    [(condition._expr, value._expr) for condition, value in branches],
                    None if else_value is None else else_value._expr,
                )
            )

        # Without annotations: the class body looks its names up by name, which cloudpickle
        # does not pickle with the function.
        class Sent(Expression):
            """An expression as it was sent, of its rendering."""

            def __init__(self, expression, rendered):  # type: ignore[no-untyped-def]
                super().__init__()
                self._expression = expression
                self._rendered = rendered

            def to_plan(self, session):  # type: ignore[no-untyped-def]
                sent = proto.Expression()
                sent.CopyFrom(self._expression)
                return sent

            def __repr__(self):  # type: ignore[no-untyped-def]
                return self._rendered

        return ConnectColumn(Sent(proto.Expression.FromString(expression), rendered))

    return _pyspark.Column._of_expression(expression, rendered, branches, else_value)
