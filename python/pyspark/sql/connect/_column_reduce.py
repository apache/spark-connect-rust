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
"""The functions that a pickled Column and table argument of the client unpickle with, and the
parser of the deep messages that a pickled Column, DataFrame or table argument carries.

``pyspark.sql.connect.column`` registers this module with cloudpickle to pickle by value, so a
Column that a UDF or a ``foreachBatch`` function captures unpickles on a Python worker of the
server, whose PySpark has no Rust client, while ``pickle`` pickles the function by reference, of
this module, on the client.
"""
from collections import deque
from typing import Any, Optional


def parse_deep(message_class: Any, data: bytes) -> Any:
    """The message of ``message_class`` that the bytes ``data`` encode, as its ``FromString``
    gives it, but of any depth: protobuf's parser refuses a message that nests more than 100
    messages, and a plan that PySpark's serializer writes may nest deeper. Such a message is
    filled field by field, from a work list in place of recursion: each message field is parsed
    on its own, and one that the parser refuses is filled in turn, so that the parser never
    nests past its limit. Where ``FromString`` succeeds, this gives the same message."""
    if _is_of_the_rust_runtime(message_class):
        # The Rust protobuf runtime of the drop-in parses a message of any depth of its own parse,
        # and needs no protobuf.
        from pyspark import _pyspark

        return _pyspark._proto_parse_deep(message_class, data)
    from google.protobuf.descriptor import FieldDescriptor
    from google.protobuf.message import DecodeError

    try:
        return message_class.FromString(data)
    except DecodeError:
        pass
    message = message_class()
    pending = deque([(message, data)])
    try:
        while pending:
            target, payload = pending.popleft()
            fields = target.DESCRIPTOR.fields_by_number
            # The message fields whose earlier occurrence the parser refused: their later
            # occurrences are filled after that one, in order, rather than parsed at once.
            deferred = set()
            pos = 0
            while pos < len(payload):
                start = pos
                tag, pos = _varint(payload, pos)
                number, wire_type = tag >> 3, tag & 7
                if wire_type == 0:
                    _, pos = _varint(payload, pos)
                    end = pos
                elif wire_type == 1:
                    end = pos = pos + 8
                elif wire_type == 5:
                    end = pos = pos + 4
                elif wire_type == 2:
                    length, pos = _varint(payload, pos)
                    end = pos + length
                    if end > len(payload):
                        raise DecodeError("a field longer than its message")
                    field = fields.get(number)
                    if (
                        field is not None
                        and field.type == FieldDescriptor.TYPE_MESSAGE
                        and not field.message_type.GetOptions().map_entry
                    ):
                        # The element or sub-message of the field, created before the parse
                        # of its bytes, which fills it in either way.
                        if _is_repeated(field):
                            child = getattr(target, field.name).add()
                        else:
                            child = getattr(target, field.name)
                        child.SetInParent()
                        if number not in deferred:
                            try:
                                fresh = type(child).FromString(payload[pos:end])
                            except DecodeError:
                                deferred.add(number)
                            else:
                                child.MergeFrom(fresh)
                                pos = end
                                continue
                        pending.append((child, payload[pos:end]))
                        pos = end
                        continue
                elif wire_type in (3, 4):
                    raise DecodeError("a group, which Spark Connect messages do not use")
                else:
                    raise DecodeError("an unknown wire type")
                target.MergeFromString(payload[start:end])
                pos = end
    except IndexError as error:
        raise DecodeError("a truncated varint") from error
    return message


def _varint(data: bytes, pos: int) -> tuple[int, int]:
    """The varint of ``data`` at ``pos``, and the position after it, as protobuf's wire format
    encodes it. Truncated data raises IndexError."""
    result = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


def _is_of_the_rust_runtime(message_class: Any) -> bool:
    """Whether ``message_class`` is a message of the drop-in's Rust protobuf runtime, which a
    Python worker of the server (of PySpark's own package, which has no ``pyspark._pyspark``)
    never has."""
    try:
        from pyspark import _pyspark
    except ImportError:
        return False
    return isinstance(message_class, type) and issubclass(message_class, _pyspark.ProtoMessage)


def _is_repeated(field: Any) -> bool:
    """Whether ``field`` is a repeated field, of protobuf's descriptor of either version."""
    from google.protobuf.descriptor import FieldDescriptor

    if hasattr(field, "is_repeated"):
        return field.is_repeated
    return field.label == FieldDescriptor.LABEL_REPEATED


def column_of(
    expression: bytes,
    rendered: str,
    branches: Optional[list[tuple[Any, Any]]] = None,
    else_value: Any = None,
    sort: Optional[tuple[Any, str]] = None,
    levels: int = 1,
    build_error: Optional[tuple[str, str, Any]] = None,
    repr_error: Optional[tuple[str, str, Any]] = None,
    subqueries: Optional[list[tuple[bytes, int]]] = None,
) -> Any:
    """The Column of ``expression``, the bytes of a ``spark.connect.Expression`` of ``levels``
    levels, and of its rendering ``rendered``, which sends the expression as it is: of the Rust
    client, or, on a Python worker of the server, PySpark Connect's Column of the expression, as
    PySpark Connect's Column of its expression unpickles there, so everything that it calls there
    is of PySpark. For a CASE WHEN, ``branches`` are the (condition, value) Columns of its
    branches and ``else_value`` the Column of its else value, so that it is a CASE WHEN again. For
    a sort order, ``sort`` is its child Column and the name of its sort method, such as
    ``desc_nulls_last``, so that it is a sort order again. ``subqueries`` are the bytes and the
    levels of the relation of each plan that the subqueries of the expression refer to, of its
    plan ID, which a plan of the Column sends in its ``WithRelations``, as PySpark's plan sends
    the plans of the subqueries of its expressions."""
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

        from pyspark.sql.connect.expressions import SubqueryExpression

        # Without annotations: the class body looks its names up by name, which cloudpickle
        # does not pickle with the function.
        class Sent(Expression):
            """An expression as it was sent, of its rendering, whose children are a subquery of
            the plan of each relation that its subqueries refer to, which a plan of it collects
            as its references (``LogicalPlan._collect_references``), and so sends."""

            def __init__(self, expression, rendered, subqueries):  # type: ignore[no-untyped-def]
                super().__init__()
                self._expression = expression
                self._rendered = rendered
                # The type of such a subquery is not sent: the expression is sent as it is.
                self._subqueries = [
                    SubqueryExpression(sent_plan(parse_deep(proto.Relation, relation)), "scalar")
                    for relation, _ in subqueries or []
                ]

            @property
            def children(self):  # type: ignore[no-untyped-def]
                return self._subqueries

            def to_plan(self, session):  # type: ignore[no-untyped-def]
                if build_error is not None:
                    raise_error(build_error)
                sent = proto.Expression()
                sent.CopyFrom(self._expression)
                return sent

            def __repr__(self):  # type: ignore[no-untyped-def]
                if repr_error is not None:
                    raise_error(repr_error)
                return self._rendered

        return ConnectColumn(Sent(parse_deep(proto.Expression, expression), rendered, subqueries))

    return _pyspark.Column._of_expression(
        expression, rendered, branches, else_value, levels, build_error, repr_error, subqueries
    )


def table_arg_of(
    session_id: str,
    relation: bytes,
    levels: int,
    partition: list[Any],
    order: list[Any],
    single_partition: bool,
    build_error: Optional[tuple[str, str, Any]] = None,
) -> Any:
    """The table argument of ``relation``, the bytes of the relation of ``levels`` levels of the
    plan of a DataFrame of the session of ``session_id``, partitioned by the Columns of
    ``partition``, or of a single partition, and ordered by the Columns of ``order``, as a
    pickled table argument unpickles: with no request, as PySpark's unpickle of the objects of
    its plan sends none. Of ``build_error``, the description of the error of a plan that did not
    build, the requests of a plan of it raise that error, as PySpark's raise it when they build
    the plan.

    On the client, it is the ``asTable()`` of the DataFrame of the relation, of its plan ID, of
    the active session of that ID (``SparkSession._getActiveSessionIfMatches``), and of the calls
    that made it: ``partitionBy`` or ``withSinglePartition``, then ``orderBy``. On a Python worker
    of the server, whose PySpark has no Rust client and no session, it is PySpark Connect's table
    argument of the relation, of the same calls, as PySpark's table argument, which holds no
    session, unpickles there. A function of the module, so that a plain pickle of a table argument
    pickles it by its name, as a plain pickle of PySpark's table argument succeeds."""
    try:
        from pyspark import _pyspark
    except ImportError:
        from pyspark.sql.connect import proto
        from pyspark.sql.connect.expressions import SubqueryExpression
        from pyspark.sql.connect.table_arg import TableArg

        plan = sent_plan(parse_deep(proto.Relation, relation), build_error)
        table = TableArg(SubqueryExpression(plan, subquery_type="table_arg"))
    else:
        from pyspark.sql.connect.session import SparkSession

        session = SparkSession._getActiveSessionIfMatches(session_id)
        table = _pyspark.TableArg._of_relation(session, relation, levels, build_error)
    if partition:
        table = table.partitionBy(*partition)
    if single_partition:
        table = table.withSinglePartition()
    if order:
        table = table.orderBy(*order)
    return table


def sent_plan(relation: Any, build_error: Optional[tuple[str, str, Any]] = None) -> Any:
    """PySpark Connect's plan of ``relation``, a ``spark.connect.Relation`` as it was sent, of its
    plan ID, which a subquery refers to, on a Python worker of the server: its ``plan`` sends the
    relation as it is, or raises the error of ``build_error``, the description of the error of a
    plan that did not build."""
    from pyspark.sql.connect import proto
    from pyspark.sql.connect.plan import LogicalPlan

    # Without annotations: the class body looks its names up by name, which cloudpickle does not
    # pickle with the function.
    class Relation(LogicalPlan):
        """The plan of a relation as it was sent, of its plan ID."""

        def __init__(self, relation):  # type: ignore[no-untyped-def]
            super().__init__(None)
            self._relation = relation
            if relation.common.HasField("plan_id"):
                self._root_plan_id = relation.common.plan_id

        def plan(self, session):  # type: ignore[no-untyped-def]
            if build_error is not None:
                raise_error(build_error)
            sent = proto.Relation()
            sent.CopyFrom(self._relation)
            return sent

    return Relation(relation)


def raise_error(error: tuple[str, str, Any]) -> None:
    """Raise the error of ``error``, the ``(kind, text, payload)`` of a pickled Column or
    DataFrame of an expression or a plan that a request did not build, or of an expression whose
    ``__repr__`` failed, as PySpark raised it: a builtin error of its message, the error of UTF-8
    of the str or bytes of the payload, or the error of one of PySpark's error classes, such as
    ``PySparkAttributeError``, of its error class and parameters."""
    kind, text, payload = error
    if kind == "UnicodeEncodeError":
        payload.encode("utf-8")
    if kind == "UnicodeDecodeError":
        payload.decode("utf-8")
    if kind.startswith("PySpark"):
        import pyspark.errors

        raise getattr(pyspark.errors, kind)(errorClass=text, messageParameters=payload)
    builtins = {
        "TypeError": TypeError,
        "ValueError": ValueError,
        "OverflowError": OverflowError,
        "AttributeError": AttributeError,
    }
    raise builtins[kind](text)
