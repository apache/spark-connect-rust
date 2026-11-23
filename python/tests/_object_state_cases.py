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
"""The objects of a session and of a DataFrame of `test_dropin_object_state_offline.py`, and the
specifications of what their state holds, which `describe` makes of PySpark 4.3.0's objects for
the goldens of that module and `check` compares the drop-in's with.

A specification of a value is a tuple:
- `("is", path)`: the object at `path` of the objects of the test (`Objects.resolve`);
- `("eq", value)`: a value equal to `value`, of its type (None, a bool, an int, a str, an empty
  dict or list, or a dict of str);
- `("eq_of", path)`: a value equal to the object at `path`;
- `("type", name)`: a value of the class of the qualified `name`;
- `("instance", name)`: a value that is an instance of the class of the qualified `name`;
- `("plan",)`: a plan of `pyspark.sql.connect.plan`, a `LogicalPlan`;
- `("expr", repr)`: an expression, of that `repr` (or of any, for None);
- `("seq", kind, items)`: a list or a tuple of the values of `items`;
- `("vars", name, entries)`: a value of the class of `name` whose `vars()` holds `entries`;
- `("message", name, text)`: a protobuf message of the class of `name`, of the text format
  `text` (or of any, for None).
"""

import copy
import pickle

from pyspark import cloudpickle


def qualified(cls):
    return f"{cls.__module__}.{cls.__qualname__}"


class Objects:
    """The objects of the tests, of the session `spark` of the mock server, as PySpark's API makes
    them, and those that the states of the others hold, which `resolve` finds by their paths."""

    def __init__(self, spark, service):
        self.spark = spark
        self.service = service
        self.df = spark.range(3)
        self.sdf = spark.readStream.format("rate").load()

    def resolve(self, path):
        obj = self
        for name in path.split("."):
            obj = getattr(obj, name)
        return obj

    def peek(self, path):
        """The object at `path`, as `resolve` gives it, without making the objects that the
        session keeps once they are read (its `streams` and its `catalog`, which PySpark's
        session keeps in its `__dict__` then): None of one that it does not keep yet."""
        state = vars(self.spark)
        if path == "spark.streams":
            return state.get("_sqm")
        if path == "spark.streams._sqlb":
            manager = state.get("_sqm")
            return None if manager is None else vars(manager).get("_sqlb")
        if path == "spark.catalog":
            return state.get("catalog")
        return self.resolve(path)


# The paths of the objects that a state may hold, in the order that `describe` tries them: the
# first whose object is the value names it.
PATHS = (
    "spark",
    "spark.client",
    "spark.streams",
    "spark.streams._sqlb",
    "spark.catalog",
    "df",
    "df._plan",
    "df._cached_schema",
    "sdf._plan",
    "mw",
    "gd",
    "owner",
)


def _expression_class():
    from pyspark.sql.connect.expressions import Expression

    return Expression


def _plan_class():
    from pyspark.sql.connect.plan import LogicalPlan

    return LogicalPlan


def _is_message(value):
    # A message of the protos of Spark Connect, of either runtime; not a Column, whose
    # `__getattr__` gives a Column of any name.
    return type(value).__module__.startswith("pyspark.sql.connect.proto")


# The classes whose objects' states the specifications hold too, of their `vars()`.
NESTED = (
    "pyspark.sql.connect.plan.WriteOperation",
    "pyspark.sql.connect.plan.WriteOperationV2",
    "pyspark.sql.connect.plan.WriteStreamOperation",
    "pyspark.sql.connect.streaming.query.StreamingQueryListenerBus",
    "pyspark.sql.connect.window.WindowFrame",
)


def describe(objects, value, owner=None, depth=0, key=None, exclude=()):
    """The specification of `value`, the entry `key` of the state of `owner`, of PySpark's
    objects, for a golden: not of the paths of `exclude`, the entries of the object itself."""
    if key in ("_root_plan_id", "_plan_id_with_rel") and isinstance(value, int):
        # The plan IDs of a process, which depend on the plans that it made before.
        return ("type", qualified(type(value)))
    for path in PATHS:
        if path in exclude:
            continue
        try:
            if path == "owner":
                if owner is not None and value is owner:
                    return ("is", "owner")
                continue
            known = objects.peek(path)
            if known is not None and value is known:
                return ("is", path)
        except AttributeError:
            continue
    if isinstance(value, (bool, int, float, str, type(None))) and not isinstance(value, type):
        if path_of_equal(objects, value) is not None:
            return ("eq_of", path_of_equal(objects, value))
        if isinstance(value, int) and not isinstance(value, bool) and value > 1000:
            return ("type", qualified(type(value)))
        return ("eq", value)
    if isinstance(value, dict) and all(
        isinstance(k, str) and isinstance(v, (str, type(None))) for k, v in value.items()
    ):
        return ("eq", dict(value))
    if isinstance(value, (list, tuple)):
        kind = "tuple" if isinstance(value, tuple) else "list"
        return ("seq", kind, [describe(objects, item, owner, depth + 1) for item in value])
    if isinstance(value, _plan_class()) and qualified(type(value)) not in NESTED:
        return ("plan",)
    if qualified(type(value)) == "pyspark.sql.connect.column.Column":
        return ("column", repr(value))
    if isinstance(value, _expression_class()):
        text = repr(value)
        return ("expr", None if " at 0x" in text else text)
    if _is_message(value):
        text = str(value)
        return ("message", qualified(type(value)), None if "call_site" in text else text)
    if qualified(type(value)) in NESTED and depth < 2:
        return (
            "vars",
            qualified(type(value)),
            [
                (k, describe(objects, v, value, depth + 1, key=k))
                for k, v in list(vars(value).items())
            ],
        )
    if qualified(type(value)) in ("threading.Thread",):
        return ("instance", "threading.Thread")
    return ("type", qualified(type(value)))


def path_of_equal(objects, value):
    # A str that is a session ID, equal to that of the session, which the goldens name by its path.
    if isinstance(value, str) and value == objects.spark.session_id:
        return "spark.session_id"
    return None


def check(objects, value, spec, owner=None, where="value"):
    """Assert that `value` is what `spec` says, of the objects of the test."""
    kind = spec[0]
    if kind == "is":
        expected = owner if spec[1] == "owner" else objects.resolve(spec[1])
        assert value is expected, f"{where}: {value!r} is not {spec[1]}"
    elif kind == "eq":
        expected = spec[1]
        assert type(value) is type(expected) and value == expected, f"{where}: {value!r}"
    elif kind == "eq_of":
        assert value == objects.resolve(spec[1]), f"{where}: {value!r} != {spec[1]}"
    elif kind == "type":
        assert qualified(type(value)) == spec[1], f"{where}: {qualified(type(value))} != {spec[1]}"
    elif kind == "instance":
        import threading

        assert isinstance(value, threading.Thread), f"{where}: {value!r} is not a {spec[1]}"
    elif kind == "column":
        assert qualified(type(value)) == "pyspark.sql.connect.column.Column", f"{where}: {value!r}"
        assert repr(value) == spec[1], f"{where}: {value!r} != {spec[1]}"
    elif kind == "plan":
        assert isinstance(value, _plan_class()), f"{where}: {value!r} is not a LogicalPlan"
    elif kind == "expr":
        assert spec[1] is None or repr(value) == spec[1], f"{where}: {value!r} != {spec[1]}"
        assert hasattr(value, "to_plan"), f"{where}: {value!r} is not an expression"
    elif kind == "seq":
        expected_type = tuple if spec[1] == "tuple" else list
        assert type(value) is expected_type, f"{where}: {type(value).__name__} != {spec[1]}"
        assert len(value) == len(spec[2]), f"{where}: {value!r} has not {len(spec[2])} items"
        for i, (item, item_spec) in enumerate(zip(value, spec[2])):
            check(objects, item, item_spec, owner, f"{where}[{i}]")
    elif kind == "message":
        assert qualified(type(value)) == spec[1], f"{where}: {qualified(type(value))} != {spec[1]}"
        assert spec[2] is None or str(value) == spec[2], f"{where}: {str(value)!r} != {spec[2]!r}"
    elif kind == "vars":
        assert qualified(type(value)) == spec[1], f"{where}: {qualified(type(value))} != {spec[1]}"
        check_vars(objects, value, spec[2], where)
    else:
        raise AssertionError(f"unknown specification {spec!r}")


def check_vars(objects, obj, entries, where="vars"):
    """Assert that `vars(obj)` holds `entries`, (key, specification) in the order of the keys."""
    state = vars(obj)
    assert list(state) == [key for key, _ in entries], f"{where}: {list(state)}"
    for key, spec in entries:
        check(objects, state[key], spec, obj, f"{where}[{key!r}]")


_ATOMS = (bool, int, float, str, bytes, type(None))


def _is_handle(value):
    # A plan or an expression, or a Column, or a list or a tuple of them: their copies are other
    # objects, which only PySpark's expression classes compare equal.
    if isinstance(value, (list, tuple)):
        return any(_is_handle(item) for item in value)
    return (
        isinstance(value, (_plan_class(), _expression_class()))
        or qualified(type(value)) == "pyspark.sql.connect.column.Column"
    )


def _same(a, b):
    """Whether `b`, an entry of a copy, is `a` ("is"), an object equal to it ("eq"), or another
    ("new"): an immutable atom is "eq" when it is equal, whether or not it is the same object."""
    if a is b and not isinstance(a, _ATOMS):
        return "is"
    if _is_handle(a):
        return "new"
    try:
        if type(a) is type(b) and bool(a == b):
            return "eq"
    except Exception:
        pass
    return "new"


def outcome(obj, how):
    """What a copy or a pickle of `obj` gives: the error that it raises, or the class of the copy,
    whether it is `obj`, and for each key of `vars(obj)` whether the copy's is the same object,
    an equal one or another (`is`, `eq`, `new`)."""
    makers = {
        "copy": copy.copy,
        "deepcopy": copy.deepcopy,
        "pickle": lambda o: pickle.loads(pickle.dumps(o)),
        "cloudpickle": lambda o: pickle.loads(cloudpickle.dumps(o)),
    }
    try:
        made = makers[how](obj)
    except BaseException as e:
        message = str(e).splitlines()[0] if str(e) else ""
        # The address of a lambda of the test, which pickle names, is not of the golden.
        if " at 0x" in message:
            message = message.split(" at 0x")[0]
        return ("err", f"{type(e).__name__}: {message}")
    if made is obj:
        return ("same",)
    original, state = vars(obj), vars(made)
    return (
        "ok",
        qualified(type(made)) == qualified(type(obj)),
        [(key, _same(original[key], state[key]) if key in original else "added") for key in state],
    )


def _connect_session():
    from pyspark.sql.connect.session import SparkSession

    return SparkSession


def _class(path):
    import importlib

    module, _, name = path.rpartition(".")
    try:
        return getattr(importlib.import_module(module), name)
    except (ImportError, AttributeError):
        # A nested class, such as `MergeIntoWriter.WhenMatched`.
        module, _, outer = module.rpartition(".")
        return getattr(getattr(importlib.import_module(module), outer), name)


def _merge(o):
    from pyspark.sql import functions as F

    o.mw = o.df.mergeInto("dictsub_t", F.col("id") == 1)
    return o.mw


def _grouped(o):
    o.gd = o.df.groupBy("id")
    return o.gd


def _schema_and_streaming(o):
    _ = o.df.schema
    _ = o.df.isStreaming
    return o.df


def _filter_of_a_cached_schema(o):
    from pyspark.sql import functions as F

    _ = o.df.schema
    return o.df.filter(F.col("id") > 0)


def _streams_and_catalog(o):
    _ = o.spark.streams
    _ = o.spark.catalog
    return o.spark


def _struct_schema(o):
    from pyspark.sql.types import LongType, StructField, StructType

    return o.spark.read.schema(StructType([StructField("a", LongType())]))


def _window(o):
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    return Window.partitionBy("id").orderBy(F.col("id").desc()).rowsBetween(-1, 1)


def _col(name):
    from pyspark.sql import functions as F

    return F.col(name)


# The objects of the tests, by a name of how PySpark's API makes them: of a new session of the
# mock server, of `Objects`. Those that the states of the others hold are kept on it (`mw`, `gd`).
OBJECTS = {
    "SparkSession": lambda o: o.spark,
    "SparkSession of streams and catalog": _streams_and_catalog,
    "SparkSession.Builder": lambda o: _connect_session().builder,
    "SparkSession.Builder of its setters": lambda o: _connect_session()
    .builder.remote("sc://localhost:1")
    .config("spark.x", 1)
    .config(map={"a": True})
    .appName("n"),
    "RuntimeConf": lambda o: o.spark.conf,
    "Catalog": lambda o: o.spark.catalog,
    "StreamingQueryManager": lambda o: o.spark.streams,
    "StreamingQueryListenerBus": lambda o: o.spark.streams._sqlb,
    "StreamingQuery": lambda o: _class("pyspark.sql.connect.streaming.query.StreamingQuery")(
        o.spark, "dictsub-id", "dictsub-run", "dictsub"
    ),
    "DataFrameReader": lambda o: o.spark.read,
    "DataFrameReader of its setters": lambda o: o.spark.read.format("csv")
    .option("header", True)
    .option("x", None)
    .schema("a int")
    .options(y=1),
    "DataFrameReader of a struct schema": _struct_schema,
    "DataStreamReader": lambda o: o.spark.readStream,
    "DataStreamReader of its setters": lambda o: o.spark.readStream.format("rate")
    .option("rowsPerSecond", 1)
    .name("src"),
    "DataFrameWriter": lambda o: o.df.write,
    "DataFrameWriter of its setters": lambda o: o.df.write.mode("overwrite")
    .format("parquet")
    .option("a", 1)
    .partitionBy("id")
    .bucketBy(2, "id")
    .sortBy("id"),
    "DataFrameWriterV2": lambda o: o.df.writeTo("dictsub_t"),
    "DataFrameWriterV2 of its setters": lambda o: o.df.writeTo("dictsub_t")
    .using("parquet")
    .option("a", 1)
    .tableProperty("p", "v")
    .partitionedBy(_col("id")),
    "DataStreamWriter": lambda o: o.sdf.writeStream,
    "DataStreamWriter of its setters": lambda o: o.sdf.writeStream.format("memory")
    .queryName("dictsub_q")
    .outputMode("append")
    .option("a", 1)
    .trigger(processingTime="1 second"),
    "MergeIntoWriter": _merge,
    "MergeIntoWriter of its clauses": lambda o: _merge(o)
    .whenMatched()
    .updateAll()
    .whenNotMatched(_col("id") > 1)
    .insertAll()
    .withSchemaEvolution(),
    "WhenMatched": lambda o: _merge(o).whenMatched(),
    "WhenNotMatched of a condition": lambda o: _merge(o).whenNotMatched(_col("id") > 1),
    "WhenNotMatchedBySource": lambda o: _merge(o).whenNotMatchedBySource(),
    "UDFRegistration": lambda o: o.spark.udf,
    "UDTFRegistration": lambda o: o.spark.udtf,
    "DataSourceRegistration": lambda o: o.spark.dataSource,
    "TableValuedFunction": lambda o: o.spark.tvf,
    "GroupedData of groupBy": lambda o: o.df.groupBy("id"),
    "GroupedData of cube": lambda o: o.df.cube("id"),
    "GroupedData of rollup": lambda o: o.df.rollup("id"),
    "GroupedData of pivot": lambda o: o.df.groupBy("id").pivot("id", [1, 2]),
    "GroupedData of groupingSets": lambda o: o.df.groupingSets([["id"], []], "id"),
    "PandasCogroupedOps": lambda o: _grouped(o).cogroup(o.df.groupBy("id")),
    "DataFrameNaFunctions": lambda o: o.df.na,
    "DataFrameStatFunctions": lambda o: o.df.stat,
    "Observation": lambda o: _class("pyspark.sql.Observation")("dictsub_o"),
    "DataFrame": lambda o: o.df,
    "DataFrame of schema and isStreaming": _schema_and_streaming,
    "DataFrame of select": lambda o: o.df.select("id"),
    "DataFrame of filter of a cached schema": _filter_of_a_cached_schema,
    "Column": lambda o: _col("id"),
    "Column of a DataFrame": lambda o: o.df.id,
    "WindowSpec": _window,
    "TableArg": lambda o: o.df.asTable(),
    "TableArg of partitionBy": lambda o: o.df.asTable().partitionBy("id"),
    "SparkSession.Hook": lambda o: _connect_session().Hook(),
    "df.plot": lambda o: o.df.plot,
}

# A data source of no reader, for the registrations of the tests.
_DATA_SOURCE = int


def _new_session(o):
    return (o.service.remote,)


# The classes of the objects of a session of PySpark Connect, by their qualified names: the
# arguments of PySpark's constructor of each, of the objects of the test; the call that an
# object of a subclass whose `__init__` never ran makes first; and an `int` among the arguments
# where PySpark keeps any object, with the call that uses it first.
CLASSES = {
    "pyspark.sql.connect.session.SparkSession": dict(
        args=_new_session,
        uninit=lambda s: s.version,
    ),
    "pyspark.sql.connect.session.SparkSession.Builder": dict(
        args=lambda o: (),
        uninit=lambda b: b.config("a", "b"),
    ),
    "pyspark.sql.connect.conf.RuntimeConf": dict(
        args=lambda o: (o.spark.client,),
        uninit=lambda c: c.get("spark.sql.shuffle.partitions"),
        foreign=(lambda o: (1,), lambda c: c.get("spark.sql.shuffle.partitions")),
    ),
    "pyspark.sql.connect.catalog.Catalog": dict(
        args=lambda o: (o.spark,),
        uninit=lambda c: c.currentDatabase(),
        foreign=(lambda o: (1,), lambda c: c.currentDatabase()),
    ),
    "pyspark.sql.connect.streaming.query.StreamingQueryManager": dict(
        args=lambda o: (o.spark,),
        uninit=lambda m: m.active,
        foreign=(lambda o: (1,), lambda m: m.active),
    ),
    "pyspark.sql.connect.streaming.query.StreamingQueryListenerBus": dict(
        args=lambda o: (o.spark.streams,),
        uninit=lambda b: b.remove(None),
    ),
    "pyspark.sql.connect.streaming.query.StreamingQuery": dict(
        args=lambda o: (o.spark, "dictsub-id", "dictsub-run", "dictsub"),
        uninit=lambda q: q.isActive,
        foreign=(lambda o: (1, "dictsub-id", "dictsub-run"), lambda q: q.isActive),
    ),
    "pyspark.sql.connect.readwriter.DataFrameReader": dict(
        args=lambda o: (o.spark,),
        uninit=lambda r: r.format("json").table("x"),
        foreign=(lambda o: (1,), lambda r: r.format("csv").table("dictsub_v").collect()),
    ),
    "pyspark.sql.connect.streaming.readwriter.DataStreamReader": dict(
        args=lambda o: (o.spark,),
        uninit=lambda r: r.format("rate").load(),
        foreign=(lambda o: (1,), lambda r: type(r.format("rate").load()).__name__),
    ),
    "pyspark.sql.connect.readwriter.DataFrameWriter": dict(
        args=lambda o: (o.df._plan, o.spark),
        uninit=lambda w: w.mode("overwrite"),
        foreign=(lambda o: (o.df._plan, 1), lambda w: w.format("noop").save()),
    ),
    "pyspark.sql.connect.readwriter.DataFrameWriterV2": dict(
        args=lambda o: (o.df._plan, o.spark, "dictsub_t"),
        uninit=lambda w: w.option("a", "b"),
        foreign=(lambda o: (o.df._plan, 1, "dictsub_t"), lambda w: w.create()),
    ),
    "pyspark.sql.connect.streaming.readwriter.DataStreamWriter": dict(
        args=lambda o: (o.sdf._plan, o.spark),
        uninit=lambda w: w.format("memory"),
        foreign=(lambda o: (o.sdf._plan, 1), lambda w: w.format("memory").start()),
    ),
    "pyspark.sql.connect.merge.MergeIntoWriter": dict(
        args=lambda o: (o.df._plan, o.spark, "dictsub_t", _col("id") == 1),
        uninit=lambda m: type(m.withSchemaEvolution()).__name__,
        foreign=(lambda o: (o.df._plan, 1, "dictsub_t", _col("id") == 1), lambda m: m.merge()),
    ),
    "pyspark.sql.connect.merge.MergeIntoWriter.WhenMatched": dict(
        args=lambda o: (_merge(o), None),
        uninit=lambda c: c.delete(),
        foreign=(lambda o: (1, None), lambda c: c.delete()),
    ),
    "pyspark.sql.connect.merge.MergeIntoWriter.WhenNotMatched": dict(
        args=lambda o: (_merge(o), None),
        uninit=lambda c: c.insertAll(),
        foreign=(lambda o: (1, None), lambda c: c.insertAll()),
    ),
    "pyspark.sql.connect.merge.MergeIntoWriter.WhenNotMatchedBySource": dict(
        args=lambda o: (_merge(o), None),
        uninit=lambda c: c.delete(),
        foreign=(lambda o: (1, None), lambda c: c.delete()),
    ),
    "pyspark.sql.connect.udf.UDFRegistration": dict(
        args=lambda o: (o.spark,),
        uninit=lambda r: r.register("dictsub_f", lambda x: x),
        foreign=(lambda o: (1,), lambda r: r.register("dictsub_f", lambda x: x)),
    ),
    "pyspark.sql.connect.udtf.UDTFRegistration": dict(
        args=lambda o: (o.spark,),
        uninit=lambda r: r.register("dictsub_t", None),
    ),
    "pyspark.sql.connect.datasource.DataSourceRegistration": dict(
        args=lambda o: (o.spark,),
        uninit=lambda r: r.register(_DATA_SOURCE),
        foreign=(lambda o: (1,), lambda r: r.register(_DATA_SOURCE)),
    ),
    "pyspark.sql.connect.tvf.TableValuedFunction": dict(
        args=lambda o: (o.spark,),
        uninit=lambda t: t.range(1),
        foreign=(lambda o: (1,), lambda t: t.range(1)),
    ),
    "pyspark.sql.connect.group.GroupedData": dict(
        args=lambda o: (o.df, "groupby", [_col("id")]),
        uninit=lambda g: g.count(),
        foreign=(lambda o: (1, "groupby", []), lambda g: g.count()),
    ),
    "pyspark.sql.connect.group.PandasCogroupedOps": dict(
        args=lambda o: (_grouped(o), o.df.groupBy("id")),
        uninit=lambda c: c.applyInPandas(lambda a, b: a, "id long"),
        foreign=(lambda o: (1, 2), lambda c: c.applyInPandas(lambda a, b: a, "id long")),
    ),
    "pyspark.sql.connect.dataframe.DataFrameNaFunctions": dict(
        args=lambda o: (o.df,),
        uninit=lambda n: n.drop(),
        foreign=(lambda o: (1,), lambda n: n.drop()),
    ),
    "pyspark.sql.connect.dataframe.DataFrameStatFunctions": dict(
        args=lambda o: (o.df,),
        uninit=lambda s: s.corr("id", "id"),
        foreign=(lambda o: (1,), lambda s: s.corr("id", "id")),
    ),
    "pyspark.sql.connect.observation.Observation": dict(
        args=lambda o: ("dictsub_o",),
        uninit=lambda b: b.get,
    ),
    "pyspark.sql.connect.dataframe.DataFrame": dict(
        args=lambda o: (o.df._plan, o.spark),
        uninit=lambda d: d.columns,
        foreign=(lambda o: (o.df._plan, 1), lambda d: d.collect()),
    ),
    "pyspark.sql.connect.column.Column": dict(
        args=lambda o: (_col("id")._expr,),
        uninit=lambda c: c + 1,
        foreign=(lambda o: (1,), lambda c: c + 1),
    ),
    "pyspark.sql.connect.window.WindowSpec": dict(
        args=lambda o: (_window(o)._partitionSpec, _window(o)._orderSpec, _window(o)._frame),
        uninit=lambda w: w.rowsBetween(0, 1),
        foreign=(lambda o: (1, [], None), lambda w: w.rowsBetween(0, 1)),
    ),
    "pyspark.sql.connect.table_arg.TableArg": dict(
        args=lambda o: (o.df.asTable()._subquery_expr,),
        uninit=lambda t: t.partitionBy("id"),
        foreign=(lambda o: (1,), lambda t: t.partitionBy("id")),
    ),
}


def run(call):
    """What `call()` gives: ("ok", the name of the class of its result) or ("err", the class and
    the first line of the message of what it raises)."""
    try:
        result = call()
    except BaseException as e:
        message = str(e).splitlines()[0] if str(e) else ""
        if " at 0x" in message:
            message = message.split(" at 0x")[0]
        if isinstance(e, RecursionError):
            # Of the depth of the stack where the limit is reached, which names the call that it
            # makes there ("... while calling a Python object") or none.
            message = "maximum recursion depth exceeded"
        return ("err", f"{type(e).__name__}: {message}")
    return ("ok", result if isinstance(result, str) else type(result).__name__)


def no_init(cls):
    """A subclass of `cls` whose `__init__` does not run that of `cls`."""
    return type("NoInit", (cls,), {"__init__": lambda self, *args, **kwargs: None})


def signature_of(cls):
    import inspect

    try:
        parameters = inspect.signature(cls).parameters.values()
    except (TypeError, ValueError) as e:
        return ("err", f"{type(e).__name__}")
    return [
        (p.name, p.kind.name, None if p.default is inspect.Parameter.empty else repr(p.default))
        for p in parameters
    ]


def _own_paths(objects, obj):
    # The paths of the entries of `obj` itself, which a golden names by what they hold.
    return ("df._cached_schema",) if obj is objects.df else ()


def capture(objects_of):
    """The goldens of the objects and the classes, of `objects_of()`, a new `Objects` of a session
    of the mock server, as this client gives them: of PySpark 4.3.0 for the goldens."""
    golden = {"objects": {}, "classes": {}}
    for name, make in OBJECTS.items():
        o = objects_of()
        obj = make(o)
        entry = {
            "class": qualified(type(obj)),
            "vars": [
                (k, describe(o, v, key=k, exclude=_own_paths(o, obj)))
                for k, v in list(vars(obj).items())
            ],
        }
        for how in ("copy", "deepcopy", "pickle", "cloudpickle"):
            entry[how] = outcome(OBJECTS[name](objects_of()), how)
        golden["objects"][name] = entry
    for path, case in CLASSES.items():
        o = objects_of()
        cls = _class(path)
        entry = {"signature": signature_of(cls)}
        made = run(lambda: cls(*case["args"](o)))
        entry["construct"] = made
        if made[0] == "ok":
            entry["construct_vars"] = [k for k in vars(cls(*case["args"](o)))]
        sub = type("Sub", (cls,), {})
        entry["subclass_construct"] = run(lambda: type(sub(*case["args"](o))).__qualname__)
        args = case["args"](o) if path.endswith(".WindowSpec") else ()
        entry["uninit"] = run(lambda: case["uninit"](no_init(cls)(*args)))
        if "foreign" in case:
            foreign_args, use = case["foreign"]
            made = run(lambda: cls(*foreign_args(o)))
            entry["foreign_construct"] = made
            if made[0] == "ok":
                entry["foreign_use"] = run(lambda: use(cls(*foreign_args(o))))
        golden["classes"][path] = entry
    return golden
