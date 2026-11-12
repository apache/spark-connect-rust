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
"""The function that a pickled Column of the client unpickles with, and the parser of the deep
messages that a pickled Column or DataFrame carries.

``pyspark.sql.connect.column`` registers this module with cloudpickle to pickle by value, so a
Column that a UDF or a ``foreachBatch`` function captures unpickles on a Python worker of the
server, whose PySpark has no Rust client, while ``pickle`` pickles the function by reference, of
this module, on the client.
"""
from collections import deque
from typing import Any, Optional

from google.protobuf.descriptor import FieldDescriptor
from google.protobuf.message import DecodeError


def parse_deep(message_class: Any, data: bytes) -> Any:
    """The message of ``message_class`` that the bytes ``data`` encode, as its ``FromString``
    gives it, but of any depth: protobuf's parser refuses a message that nests more than 100
    messages, and a plan that PySpark's serializer writes may nest deeper. Such a message is
    filled field by field, from a work list in place of recursion: each message field is parsed
    on its own, and one that the parser refuses is filled in turn, so that the parser never
    nests past its limit. Where ``FromString`` succeeds, this gives the same message."""
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


def _is_repeated(field: Any) -> bool:
    """Whether ``field`` is a repeated field, of protobuf's descriptor of either version."""
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
) -> Any:
    """The Column of ``expression``, the bytes of a ``spark.connect.Expression`` of ``levels``
    levels, and of its rendering ``rendered``, which sends the expression as it is: of the Rust
    client, or, on a Python worker of the server, PySpark Connect's Column of the expression, as
    PySpark Connect's Column of its expression unpickles there, so everything that it calls there
    is of PySpark. For a CASE WHEN, ``branches`` are the (condition, value) Columns of its
    branches and ``else_value`` the Column of its else value, so that it is a CASE WHEN again. For
    a sort order, ``sort`` is its child Column and the name of its sort method, such as
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

        return ConnectColumn(Sent(parse_deep(proto.Expression, expression), rendered))

    return _pyspark.Column._of_expression(expression, rendered, branches, else_value, levels)
