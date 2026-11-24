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
"""The functions that a pickled Column, DataFrame and table argument of the client unpickle with,
the parser of the deep messages that they carry, and the fill of the holes of those messages with
the commands of the functions of their UDFs, which a pickle carries in place of the commands.

``pyspark.sql.connect.column`` registers this module with cloudpickle to pickle by value, so a
Column or a DataFrame that a UDF or a ``foreachBatch`` function captures unpickles on a Python
worker of the server, whose PySpark has no Rust client, while ``pickle`` pickles the function by
reference, of this module, on the client.

A pickle by value carries the code of each function that it holds, and the docstring of each, so
the functions here are described by comments. The fill of the holes (``fill_holes``) is given to
the functions that unpickle an object of holes as an argument, so that only the pickle of such an
object carries its code.
"""
from collections import deque
from typing import Any, Callable, Optional

# The start of each marker of a hole, which stands in a pickled message for the command of a UDF,
# and the width of the markers that follow it: the token of the maker of the markers of the
# pickle, 32 hexadecimal digits, a dash and the number of the marker, 20 decimal digits, as
# ``spark_connect::udf::SentHoleMarkers`` makes them, so that no marker is the prefix of another.
_MARKER_PREFIX = b"spark-connect-rust-udf-"
_MARKER_WIDTH = len(_MARKER_PREFIX) + 32 + 1 + 20


# The message of ``message_class`` that the bytes ``data`` encode, as its ``FromString`` gives
# it, but of any depth: protobuf's parser refuses a message that nests more than 100 messages,
# and a plan that PySpark's serializer writes may nest deeper. Such a message is filled field by
# field, from a work list in place of recursion: each message field is parsed on its own, and
# one that the parser refuses is filled in turn, so that the parser never nests past its limit.
# Where ``FromString`` succeeds, this gives the same message.
def parse_deep(message_class: Any, data: bytes) -> Any:
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


# The varint of ``data`` at ``pos``, and the position after it, as protobuf's wire format encodes
# it. Truncated data raises IndexError.
def _varint(data: bytes, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        byte = data[pos]
        pos += 1
        result |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return result, pos
        shift += 7


# Whether ``message_class`` is a message of the drop-in's Rust protobuf runtime, which a Python
# worker of the server (of PySpark's own package, which has no ``pyspark._pyspark``) never has.
def _is_of_the_rust_runtime(message_class: Any) -> bool:
    try:
        from pyspark import _pyspark
    except ImportError:
        return False
    return isinstance(message_class, type) and issubclass(message_class, _pyspark.ProtoMessage)


# Whether ``field`` is a repeated field, of protobuf's descriptor of either version.
def _is_repeated(field: Any) -> bool:
    from google.protobuf.descriptor import FieldDescriptor

    if hasattr(field, "is_repeated"):
        return field.is_repeated
    return field.label == FieldDescriptor.LABEL_REPEATED


# The Column of ``expression``, the bytes of a ``spark.connect.Expression`` of ``levels`` levels,
# and of its rendering ``rendered``, which sends the expression as it is: of the Rust client, or,
# on a Python worker of the server, PySpark Connect's Column of the expression, as PySpark
# Connect's Column of its expression unpickles there, so everything that it calls there is of
# PySpark. For a CASE WHEN, ``branches`` are the (condition, value) Columns of its branches and
# ``else_value`` the Column of its else value, so that it is a CASE WHEN again. For a sort order,
# ``sort`` is its child Column and the name of its sort method, such as ``desc_nulls_last``, so
# that it is a sort order again. ``subqueries`` are the bytes, the levels and the holes of the
# relation of each plan that the subqueries of the expression refer to, of its plan ID, which a
# plan of the Column sends in its ``WithRelations``, as PySpark's plan sends the plans of the
# subqueries of its expressions. ``holes`` are those of the expression, of the objects of the
# functions of its UDFs, whose commands each plan of the Column pickles, as PySpark's Column
# keeps the functions, which ``fill`` fills on a worker (``fill_holes``), given when there are
# holes.
def column_of(
    expression: bytes,
    rendered: str,
    branches: Optional[list[tuple[Any, Any]]] = None,
    else_value: Any = None,
    sort: Optional[tuple[Any, str]] = None,
    levels: int = 1,
    build_error: Optional[tuple[str, str, Any]] = None,
    repr_error: Optional[tuple[str, str, Any]] = None,
    subqueries: Optional[list[tuple[Any, ...]]] = None,
    holes: Optional[list[tuple[bytes, str, Any]]] = None,
    fill: Optional[Callable[..., bytes]] = None,
) -> Any:
    if sort is not None:
        # The child is a Column of the client, or PySpark Connect's on a worker, whose sort
        # method gives back the sort order, as PySpark's SortOrder of its child, of the origin
        # that it was pickled with.
        child, method = sort
        return with_origin_of(getattr(child, method)(), expression)
    try:
        from pyspark import _pyspark
    except ImportError:
        from pyspark.sql.connect import proto
        from pyspark.sql.connect.column import Column as ConnectColumn
        from pyspark.sql.connect.expressions import CaseWhen, Expression

        if branches is not None:
            # As PySpark's CaseWhen of its branches and else value.
            return with_origin_of(
                ConnectColumn(
                    CaseWhen(
                        [(condition._expr, value._expr) for condition, value in branches],
                        None if else_value is None else else_value._expr,
                    )
                ),
                expression,
            )

        from pyspark.sql.connect.expressions import SubqueryExpression

        # Without annotations: the class body looks its names up by name, which cloudpickle
        # does not pickle with the function.
        class Sent(Expression):
            # An expression as it was sent, of its rendering, whose children are a subquery of
            # the plan of each relation that its subqueries refer to, which a plan of it collects
            # as its references (``LogicalPlan._collect_references``), and so sends.

            def __init__(self, expression, rendered, subqueries):  # type: ignore[no-untyped-def]
                super().__init__()
                self._expression = expression
                self._rendered = rendered
                # The type of such a subquery is not sent: the expression is sent as it is.
                self._subqueries = [
                    SubqueryExpression(
                        sent_plan(
                            subquery[0],
                            None,
                            subquery[2] if len(subquery) > 2 else None,
                            fill,
                        ),
                        "scalar",
                    )
                    for subquery in subqueries or []
                ]

            @property
            def children(self):  # type: ignore[no-untyped-def]
                return self._subqueries

            def to_plan(self, session):  # type: ignore[no-untyped-def]
                if build_error is not None and build_error[0] == "plan":
                    plan_of_error(build_error).plan(session)
                if build_error is not None:
                    raise_error(build_error)
                if holes:
                    return parse_deep(proto.Expression, fill(expression, holes))
                sent = proto.Expression()
                sent.CopyFrom(self._expression)
                return sent

            def __repr__(self):  # type: ignore[no-untyped-def]
                if repr_error is not None:
                    raise_error(repr_error)
                return self._rendered

        return ConnectColumn(Sent(parse_deep(proto.Expression, expression), rendered, subqueries))

    return _pyspark.Column._of_expression(
        expression,
        rendered,
        branches,
        else_value,
        levels,
        build_error,
        repr_error,
        subqueries,
        holes,
    )


# The object of ``cls``, a subclass of the class of the object that ``make(*args)`` gives back, a
# Column or a table argument, as a pickle of an object of a subclass unpickles, as PySpark's
# pickle of a plain object unpickles it (``copyreg.__newobj__`` of the class, then its
# ``__dict__``): an object of ``cls`` of no state, of the entries of the ``__dict__`` of that
# object, and of its session of a table argument of the client, which the client keeps natively.
# The pickle sets the other entries of the ``__dict__`` of the object then, its state.
def instance_of(cls: Any, make: Callable[..., Any], args: tuple[Any, ...]) -> Any:
    made = make(*args)
    obj = cls.__new__(cls)
    try:
        from pyspark import _pyspark
    except ImportError:
        pass
    else:
        if isinstance(made, _pyspark.TableArg):
            _pyspark.TableArg._take_session_of(obj, made)
    obj.__dict__.update(made.__dict__)
    return obj


# ``column`` of the origin of ``expression``, the bytes of the expression of a pickled Column, as
# PySpark's unpickled expression keeps its ``origin``: of a Column that the unpickle makes again of
# its parts, a sort order or a CASE WHEN, which takes the origin of the call that unpickles it
# otherwise, while the capture of origins is on (``pyspark.errors.utils.is_debugging_enabled``).
def with_origin_of(column: Any, expression: bytes) -> Any:
    try:
        from pyspark import _pyspark
    except ImportError:
        from pyspark.sql.connect import proto

        sent = parse_deep(proto.Expression, expression)
        column._expr.origin = sent.common.origin if sent.common.HasField("origin") else None
        return column
    return _pyspark.Column._with_origin_of(column, expression)


# The table argument of ``relation``, the bytes of the relation of ``levels`` levels of the plan
# of a DataFrame of the session of ``session_id``, partitioned by the Columns of ``partition``,
# or of a single partition, and ordered by the Columns of ``order``, as a pickled table argument
# unpickles: with no request, as PySpark's unpickle of the objects of its plan sends none. Of
# ``build_error``, the description of the error of a plan that did not build, the requests of a
# plan of it raise that error, as PySpark's raise it when they build the plan. ``holes`` are
# those of the relation, of the objects of the functions of its UDFs, whose commands each request
# pickles, which ``fill`` fills on a worker (``fill_holes``).
#
# On the client, it is the ``asTable()`` of the DataFrame of the relation, of its plan ID, of the
# active session of that ID (``SparkSession._getActiveSessionIfMatches``), and of the calls that
# made it: ``partitionBy`` or ``withSinglePartition``, then ``orderBy``. On a Python worker of the
# server, whose PySpark has no Rust client and no session, it is PySpark Connect's table argument
# of the relation, of the same calls, as PySpark's table argument, which holds no session,
# unpickles there. A function of the module, so that a plain pickle of a table argument pickles
# it by its name, as a plain pickle of PySpark's table argument succeeds.
def table_arg_of(
    session_id: str,
    relation: bytes,
    levels: int,
    partition: list[Any],
    order: list[Any],
    single_partition: bool,
    build_error: Optional[tuple[str, str, Any]] = None,
    holes: Optional[list[tuple[bytes, str, Any]]] = None,
    fill: Optional[Callable[..., bytes]] = None,
) -> Any:
    try:
        from pyspark import _pyspark
    except ImportError:
        from pyspark.sql.connect.expressions import SubqueryExpression
        from pyspark.sql.connect.table_arg import TableArg

        plan = sent_plan(relation, build_error, holes, fill)
        table = TableArg(SubqueryExpression(plan, subquery_type="table_arg"))
    else:
        from pyspark.sql.connect.session import SparkSession

        session = SparkSession._getActiveSessionIfMatches(session_id)
        table = _pyspark.TableArg._of_relation(session, relation, levels, build_error, holes)
    if partition:
        table = table.partitionBy(*partition)
    if single_partition:
        table = table.withSinglePartition()
    if order:
        table = table.orderBy(*order)
    return table


# PySpark Connect's plan of ``relation``, the bytes of a ``spark.connect.Relation`` as it was
# sent, of its plan ID, which a subquery and the columns of a DataFrame refer to, on a Python
# worker of the server: its ``plan`` sends the relation as it is, of its ``holes`` filled by
# ``fill`` with the commands of the functions of its UDFs, which it pickles then, as PySpark's
# plan pickles them when it is built (``fill_holes``), or raises the error of ``build_error``,
# the description of the error of a plan that did not build.
def sent_plan(
    relation: bytes,
    build_error: Optional[tuple[str, str, Any]] = None,
    holes: Optional[list[tuple[bytes, str, Any]]] = None,
    fill: Optional[Callable[..., bytes]] = None,
    observations: Optional[dict[str, Any]] = None,
) -> Any:
    from pyspark.sql.connect import proto
    from pyspark.sql.connect.plan import LogicalPlan

    if build_error is not None and build_error[0] == "plan":
        return plan_of_error(build_error)

    # Without annotations: the class body looks its names up by name, which cloudpickle does not
    # pickle with the function.
    class Relation(LogicalPlan):
        # The plan of a relation as it was sent, of its plan ID.

        def __init__(self, relation):  # type: ignore[no-untyped-def]
            super().__init__(None)
            self._relation = relation
            if relation.common.HasField("plan_id"):
                self._root_plan_id = relation.common.plan_id

        def plan(self, session):  # type: ignore[no-untyped-def]
            if build_error is not None:
                raise_error(build_error)
            if holes:
                return parse_deep(proto.Relation, fill(data, holes))
            sent = proto.Relation()
            sent.CopyFrom(self._relation)
            return sent

        @property
        def observations(self):  # type: ignore[no-untyped-def]
            # The Observations of the DataFrame of the relation by their names, which the pickle
            # carried, as PySpark's plan objects carry those of their observe.
            return dict(observations or {})

        def __reduce__(self):  # type: ignore[no-untyped-def]
            # As PySpark's plan object, which a pickle carries again: of the relation as it was
            # sent, of this function rather than of this class, and of the functions of this
            # module by their names, which a plain pickle names also after a cloudpickle gave them.
            args = (data, build_error, holes, _by_name(fill), observations)
            return _by_name(sent_plan), args

    data = relation
    return Relation(parse_deep(proto.Relation, relation))


# The DataFrame of ``relation``, the bytes of the relation of ``levels`` levels of a DataFrame of
# ``session`` that was pickled, of its ``holes``, the objects of the functions of its UDFs, which
# each request of the DataFrame pickles, as PySpark's unpickled plan pickles them (``fill`` fills
# them on a worker, ``fill_holes``), and of the ``_support_repr_html`` and the ``_cached_schema``
# of the state of the pickle. As PySpark's unpickle, which builds the plan of the new DataFrame
# and requests its schema, as pickle looks its ``__setstate__`` up through its ``__getattr__``,
# which reads its columns, it requests the schema of the relation, whose error it raises, and
# then sets the ``_cached_schema`` of the state. A plan that did not build is of ``build_error``,
# the description of its error, which it raises instead (``raise_error``): but for an
# ``AttributeError``, of which PySpark's ``_schema`` falls back to the Column of the
# ``__getattr__`` of the DataFrame, whose ``fields``, a Column too, its columns iterate, which
# raises ``NOT_ITERABLE``. Of a plan of the Python client whose build raised, the description
# carries the plan (``plan_of_error``), which the unpickle builds again, as PySpark's does, and
# whose error the read of the columns raises, a new one, of ``NOT_ITERABLE`` for an
# ``AttributeError`` so too.
#
# On a Python worker of the server, whose PySpark has no Rust client, such as the worker of a
# ``foreachBatch`` function, which has the active session of the query, it gives back PySpark
# Connect's DataFrame of the relation, as PySpark's DataFrame of its plan unpickles there. So
# everything that it calls there is of PySpark, which the worker has. A function of the module,
# which cloudpickle pickles by value, and ``pickle`` by reference.
def dataframe_of(
    holes: list[tuple[bytes, str, Any]],
    session: Any,
    relation: bytes,
    levels: int = 1,
    support_repr_html: bool = False,
    cached_schema: Any = None,
    build_error: Optional[tuple[str, str, Any]] = None,
    fill: Optional[Callable[..., bytes]] = None,
) -> Any:
    of_plan = build_error is not None and build_error[0] == "plan"
    if build_error is not None and not of_plan:
        kind = build_error[0]
        if kind in ("AttributeError", "PySparkAttributeError"):
            from pyspark.errors import PySparkTypeError

            raise PySparkTypeError(
                errorClass="NOT_ITERABLE", messageParameters={"objectName": "Column"}
            )
        raise_error(build_error)
    try:
        from pyspark import _pyspark
    except ImportError:
        from pyspark.sql.connect.dataframe import DataFrame as ConnectDataFrame

        df = ConnectDataFrame(sent_plan(relation, build_error, holes, fill), session)
        # Its columns, a request of its schema, and then the state, as PySpark's unpickle.
        _ = df.columns
        df._support_repr_html = support_repr_html
        df._cached_schema = cached_schema
        return df

    if not of_plan:
        return _pyspark.DataFrame._of_relation(
            session, relation, levels, support_repr_html, cached_schema, holes
        )
    # The DataFrame of the plan built again, as PySpark's unpickle builds its plan, and its
    # columns, which raise the error of its build again, if it raises.
    _, derived, plan = build_error
    df = _pyspark.DataFrame._of_python_plan(plan, session, bool(derived))
    _ = df.columns
    return df


# The object of ``function(*args)``, of the pickle of a DataFrame, a Column or a table argument
# that reaches the objects of ``reached`` first: those that PySpark's pickle reaches first of the
# plan of a model that it keeps (the functions of the UDFs before it, the summary dataset of a
# training summary, whose session a plain pickle refuses, and the lock of the ``RemoteModelRef``
# of the model), so that a pickle fails at the first of them that it cannot pickle, as PySpark's
# does, and never gets here.
def unpickled_after(reached: list[Any], function: Callable[..., Any], args: tuple[Any, ...]) -> Any:
    return function(*args)


# The plan of ``error``, ``("plan", "derived" or "", plan)``, of a plan of the Python client whose
# build raised when an object of it was pickled, which the pickle carries as PySpark's carries its
# plan objects: the plan itself, which PySpark's DataFrame or table argument builds at each request
# and raises its error then, a new one; of an object over the DataFrame of the plan, such as a
# Column of its subquery, whose own plan the pickle could not carry, a plan that builds it, of its
# error, and raises the ``RuntimeError`` of the drop-in's client if it builds.
def plan_of_error(error: tuple[str, str, Any]) -> Any:
    _, derived, plan = error
    if not derived:
        return plan
    from pyspark.sql.connect.plan import LogicalPlan

    # Without annotations: the class body looks its names up by name, which cloudpickle does not
    # pickle with the function.
    class Derived(LogicalPlan):
        def __init__(self, of):  # type: ignore[no-untyped-def]
            super().__init__(None)
            self._of_error = of

        def plan(self, session):  # type: ignore[no-untyped-def]
            self._of_error.plan(session)
            raise RuntimeError(
                "the plan of the Python client that the pickled object was made over failed to "
                "build when it was pickled and builds when it is unpickled, so the pickle has no "
                "plan of the object itself"
            )

        def __reduce__(self):  # type: ignore[no-untyped-def]
            # Of this function rather than of this class, by its name, as the reduce of the plan
            # of ``sent_plan``.
            return _by_name(plan_of_error), (("plan", "derived", self._of_error),)

    return Derived(plan)


# ``function``, if it is a function of this module, as the module has it where it is importable,
# as on the client, which a plain pickle names: a function that cloudpickle gave by value is a copy
# that a plain pickle cannot name. On a Python worker of the server, whose PySpark has no module of
# the drop-in, and of another object, ``function`` itself.
def _by_name(function: Any) -> Any:
    module_name = "pyspark.sql.connect._column_reduce"
    if getattr(function, "__module__", None) != module_name:
        return function
    import importlib

    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return function
    return getattr(module, function.__qualname__, function)


# Raise the error of ``error``, the ``(kind, text, payload)`` of a pickled Column or DataFrame of
# an expression or a plan that a request did not build, or of an expression whose ``__repr__``
# failed, as PySpark raised it: a builtin error of its message, the error of UTF-8 of the str or
# bytes of the payload, the error of one of PySpark's error classes, such as
# ``PySparkAttributeError``, of its error class and parameters.
def raise_error(error: tuple[str, str, Any]) -> None:
    kind, text, payload = error
    if kind == "UnicodeEncodeError":
        # As the build of PySpark's plan, which sets the str in a field of its message: the error
        # of the protobuf of this process, the ValueError of that of a worker of the server.
        from pyspark.sql.connect import proto

        proto.Relation().read.named_table.unparsed_identifier = payload
        # Of a protobuf that takes the str, the error of UTF-8.
        payload.encode("utf-8")  # pragma: no cover
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


# ``data``, the bytes of a message of a pickled DataFrame, Column or table argument, with its
# ``holes`` filled, on a Python worker of the server: a hole is the marker that stands for the
# command of a UDF in the message, as the whole value of a field of the message that holds it, of
# the kind of that message and of the objects that PySpark's ``PythonUDF``, ``PythonUDTF`` or
# plan keeps, of which it builds that message again, as PySpark's ``to_plan`` builds it: the
# output type converted and the function pickled with ``CloudPickleSerializer`` of a UDF, the
# return type and the handler class of a UDTF, of ``UDTF_SERIALIZATION_ERROR`` for a handler that
# does not pickle, and the type of a ``DataType``. The holes are pickled in turn, in the order of
# the build of the plan, which ``holes`` keep, as PySpark's plan pickles its functions when it is
# built, however the message orders them.
#
# As ``spark_connect::udf::SentHoles::fill``, in a time of the length of the bytes: the markers
# are found in one pass, a field is read only when its value holds one, from a work list in place
# of recursion, and the filled bytes are written once, of the lengths of the fields that change,
# computed first. The rest of the bytes stay as they are.
def fill_holes(data: bytes, holes: list[tuple[bytes, str, Any]]) -> bytes:
    import bisect

    hole_of = {marker: index for index, (marker, _, _) in enumerate(holes)}
    offsets: list[int] = []
    at = data.find(_MARKER_PREFIX)
    while at >= 0:
        hole = hole_of.get(data[at : at + _MARKER_WIDTH])
        if hole is None:
            at = data.find(_MARKER_PREFIX, at + 1)
        else:
            offsets.append(at)
            at = data.find(_MARKER_PREFIX, at + _MARKER_WIDTH)

    def holds(start: int, end: int) -> bool:
        first = bisect.bisect_left(offsets, start)
        return first < len(offsets) and offsets[first] + _MARKER_WIDTH <= end

    # The messages that hold a marker, the first the whole of ``data``: where each starts and
    # ends, where the key of its field ends in the message above it, which one that is, and the
    # hole whose message it is; each message comes after the one above it.
    starts, ends, key_ends, above = [0], [len(data)], [0], [-1]
    below: list[list[int]] = [[]]
    hole_at: list[Optional[int]] = [None]
    places: list[list[int]] = [[] for _ in holes]
    pending = [0]
    while pending:
        message = pending.pop()
        fields = _fields_holding(data, starts[message], ends[message], holds)
        # A field of the width of a marker that holds one is that marker.
        held = next(
            (hole_of[data[start:end]] for _, start, end in fields if end - start == _MARKER_WIDTH),
            None,
        )
        if held is not None:
            # This message holds the marker of the hole as the value of a field: the message
            # of the hole, or the unparsed type below the ``DataType`` that is, which is the
            # value of a field too, so never the whole of ``data``.
            of = above[message] if holes[held][1] == "datatype" else message
            if of > 0:
                hole_at[of] = held
                places[held].append(of)
            continue
        for key_end, start, end in fields:
            below[message].append(len(starts))
            pending.append(len(starts))
            starts.append(start)
            ends.append(end)
            key_ends.append(key_end)
            above.append(message)
            below.append([])
            hole_at.append(None)
    if any(not found for found in places):
        raise ValueError("a message as it was sent has no place of the command of a UDF")
    # Each hole pickled in turn, of the message that held its marker.
    filled = [
        _hole_message(kind, payload, data[starts[places[index][0]] : ends[places[index][0]]])
        for index, (_, kind, payload) in enumerate(holes)
    ]
    # The length of each message as it is filled, the last first, as each comes after the one
    # above it.
    lengths = [0] * len(starts)
    for message in range(len(starts) - 1, -1, -1):
        if hole_at[message] is not None:
            lengths[message] = len(filled[hole_at[message]])
            continue
        length = ends[message] - starts[message]
        for field in below[message]:
            length += len(_varint_bytes(lengths[field])) + lengths[field]
            length -= ends[field] - key_ends[field]
        lengths[message] = length
    out = bytearray()
    writing = [[0, 0, starts[0]]]
    while writing:
        message, next_field, at = writing[-1]
        if hole_at[message] is not None:
            out += filled[hole_at[message]]
            writing.pop()
        elif next_field < len(below[message]):
            field = below[message][next_field]
            out += data[at : key_ends[field]]
            out += _varint_bytes(lengths[field])
            writing[-1][1:] = [next_field + 1, ends[field]]
            writing.append([field, 0, starts[field]])
        else:
            out += data[at : ends[message]]
            writing.pop()
    return bytes(out)


# The length-delimited fields of the message of ``data[start:end]`` whose values hold a marker
# whole (``holds``), each where its key ends, and where its value starts and ends; none when the
# bytes are not a message, as those of a string, which holds no hole then.
def _fields_holding(
    data: bytes, start: int, end: int, holds: Callable[[int, int], bool]
) -> list[tuple[int, int, int]]:
    fields = []
    pos = start
    try:
        while pos < end:
            tag, pos = _varint(data, pos)
            wire_type = tag & 7
            if tag >> 3 == 0:
                return []
            if wire_type == 0:
                _, pos = _varint(data, pos)
            elif wire_type == 1:
                pos += 8
            elif wire_type == 5:
                pos += 4
            elif wire_type == 2:
                key_end = pos
                length, value_start = _varint(data, pos)
                pos = value_start + length
                if pos <= end and holds(value_start, pos):
                    fields.append((key_end, value_start, pos))
            else:
                return []
            if pos > end:
                return []
    except IndexError:
        return []
    return fields


# The varint of ``value``.
def _varint_bytes(value: int) -> bytes:
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


# The message of a hole of ``kind`` of ``payload``, as PySpark's ``to_plan`` builds it, of the
# fields of ``message``, the message that held the marker, but for its type and command.
def _hole_message(kind: str, payload: Any, message: bytes) -> bytes:
    import pickle

    from pyspark.serializers import CloudPickleSerializer
    from pyspark.sql.connect import proto
    from pyspark.sql.connect.types import UnparsedDataType, pyspark_types_to_proto_types

    def converted(data_type: Any) -> Any:
        if isinstance(data_type, str):
            data_type = UnparsedDataType(data_type)
        return pyspark_types_to_proto_types(data_type)

    if kind == "udf":
        func, return_type = payload
        udf = proto.PythonUDF.FromString(message)
        udf.output_type.CopyFrom(converted(return_type))
        udf.command = CloudPickleSerializer().dumps((func, return_type))
        return udf.SerializeToString()
    if kind in ("udtf", "udtf-call"):
        udtf = proto.PythonUDTF.FromString(message)
        if kind == "udtf-call":
            built = proto.PythonUDTF.FromString(payload())
            if built.HasField("return_type"):
                udtf.return_type.CopyFrom(built.return_type)
            udtf.command = built.command
            return udtf.SerializeToString()
        func, name, return_type = payload
        if return_type is not None:
            udtf.return_type.CopyFrom(converted(return_type))
        try:
            udtf.command = CloudPickleSerializer().dumps(func)
        except pickle.PicklingError:
            from pyspark.errors import PySparkPicklingError

            # Of the PicklingError as its context, as PySpark's PythonUDTF.to_plan raises it.
            raise PySparkPicklingError(  # noqa: B904
                errorClass="UDTF_SERIALIZATION_ERROR",
                messageParameters={
                    "name": name,
                    "message": "Please check the stack trace and "
                    "make sure the function is serializable.",
                },
            )
        return udtf.SerializeToString()
    return converted(payload).SerializeToString()
